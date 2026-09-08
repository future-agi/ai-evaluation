"""Native Retell API-chat execution for hosted RL Environment runs."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Mapping

from fi.simulate.agent.wrapper import (
    AgentInput,
    AgentResponse,
    AgentWrapper,
    SimulationEvent,
)

from .call_runner import ArtifactUploader, CallRunnerContext, RETELL_API_KEY_ALIAS
from .chat_call_runner import (
    _HostedChatTarget,
    _conversation_scenario,
    _drive_conversation,
    _duration_ms,
    _scenario_document,
    _tool_world,
)
from .contract import AgentContract
from .hosted_scheduler import CallAborted, CallOutcome, Scenario, World
from .outbound import ArtifactKind, format_rfc3339_millis
from .process_runtime import EnvironmentRuntime


class RetellChatError(RuntimeError):
    """A bounded, credential-safe Retell Chat API error."""


class RetellChatEnded(Exception):
    """Retell reports that the remote chat reached a normal terminal state."""


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {"_raw": value}
        return dict(parsed) if isinstance(parsed, Mapping) else {"_raw": value}
    return {}


def _completion_messages(
    messages: Any,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(messages, list):
        raise RetellChatError("retell_chat_completion_messages_invalid")
    answer = ""
    calls: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or "")
        if role == "agent" and str(item.get("content") or "").strip():
            answer = str(item["content"]).strip()
        elif role == "tool_call_invocation":
            calls.append(
                {
                    "id": str(item.get("tool_call_id") or item.get("message_id") or ""),
                    "name": str(item.get("name") or ""),
                    "arguments": _arguments(item.get("arguments")),
                }
            )
        elif role == "tool_call_result":
            responses.append(
                {
                    "tool_call_id": str(item.get("tool_call_id") or ""),
                    "content": item.get("content"),
                    "success": not bool(item.get("error")),
                    "error": item.get("error"),
                }
            )
    return answer, calls, responses


class RetellChatWrapper(AgentWrapper):
    def __init__(
        self,
        *,
        api_key: str,
        agent_id: str,
        agent_version: int | None = None,
        dynamic_variables: Mapping[str, Any] | None = None,
        api_base_url: str = "https://api.retellai.com",
    ) -> None:
        if not api_key:
            raise RetellChatError("retell_chat_api_key_missing")
        if not agent_id:
            raise RetellChatError("retell_chat_agent_id_missing")
        self._api_key = api_key
        self._agent_id = agent_id
        self._agent_version = agent_version
        self._dynamic_variables = {
            str(key): str(value) for key, value in dict(dynamic_variables or {}).items()
        }
        self._base = api_base_url.rstrip("/")
        self._chat_id: str | None = None

    def _request(
        self, method: str, path: str, payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._base + path,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "FutureAGI-ALK/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = " ".join(exc.read(512).decode("utf-8", errors="replace").split())
            if exc.code == 400 and "chat already ended" in detail.lower():
                raise RetellChatEnded from exc
            raise RetellChatError(
                f"retell_chat_api_error: {method} returned HTTP {exc.code}"
                + (f": {detail}" if detail else "")
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RetellChatError(
                f"retell_chat_api_unavailable: {type(exc).__name__}"
            ) from exc
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except ValueError as exc:
            raise RetellChatError("retell_chat_api_response_invalid_json") from exc
        if not isinstance(value, dict):
            raise RetellChatError("retell_chat_api_response_must_be_object")
        return value

    def _is_ended(self) -> bool:
        if self._chat_id is None:
            return False
        result = self._request("GET", f"/get-chat/{self._chat_id}")
        return str(result.get("chat_status") or "").strip().lower() == "ended"

    def _ended_response(
        self,
        *,
        answer: str = "",
        calls: list[dict[str, Any]] | None = None,
        responses: list[dict[str, Any]] | None = None,
    ) -> AgentResponse:
        return AgentResponse(
            content=answer,
            tool_calls=calls or [],
            tool_responses=responses or [],
            events=[
                SimulationEvent(
                    type="external_agent",
                    name="retell_chat_ended",
                    payload={"provider": "retell", "chat_id": self._chat_id},
                )
            ],
            metadata={
                "external_agent": {"success": True},
                "retell_chat_id": self._chat_id,
                "conversation_ended": True,
            },
        )

    def _complete(self, content: str) -> AgentResponse:
        if self._chat_id is None:
            create: dict[str, Any] = {
                "agent_id": self._agent_id,
                "metadata": {"source": "futureagi-rl-environment"},
            }
            if self._agent_version is not None:
                create["agent_version"] = self._agent_version
            if self._dynamic_variables:
                create["retell_llm_dynamic_variables"] = self._dynamic_variables
            result = self._request("POST", "/create-chat", create)
            self._chat_id = str(result.get("chat_id") or "").strip()
            if not self._chat_id:
                raise RetellChatError("retell_chat_create_missing_chat_id")
        try:
            result = self._request(
                "POST",
                "/create-chat-completion",
                {"chat_id": self._chat_id, "content": content},
            )
        except RetellChatEnded:
            return self._ended_response()
        answer, calls, responses = _completion_messages(result.get("messages"))
        explicitly_ended = (
            bool(result.get("call_ended"))
            or str(result.get("chat_status") or "").strip().lower() == "ended"
        )
        if not answer:
            if explicitly_ended or self._is_ended():
                return self._ended_response(calls=calls, responses=responses)
            raise RetellChatError("retell_chat_completion_missing_agent_message")
        if explicitly_ended:
            return self._ended_response(answer=answer, calls=calls, responses=responses)
        event = SimulationEvent(
            type="external_agent",
            name="retell_chat_completion",
            payload={
                "provider": "retell",
                "chat_id": self._chat_id,
                "tool_call_count": len(calls),
            },
        )
        return AgentResponse(
            content=answer,
            tool_calls=calls,
            tool_responses=responses,
            events=[event],
            metadata={
                "external_agent": {"success": True},
                "retell_chat_id": self._chat_id,
            },
        )

    async def call(self, input: AgentInput) -> AgentResponse:
        content = str((input.new_message or {}).get("content") or "").strip()
        if not content:
            raise RetellChatError("retell_chat_user_message_missing")
        return await asyncio.to_thread(self._complete, content)

    async def aclose(self) -> None:
        if self._chat_id is None:
            return
        chat_id, self._chat_id = self._chat_id, None
        try:
            await asyncio.to_thread(self._request, "PATCH", f"/end-chat/{chat_id}")
        except (RetellChatError, RetellChatEnded):
            # Closing is idempotent. A chat may have ended as part of its final
            # completion, in which case Retell rejects the redundant PATCH with
            # "Chat already ended". Cleanup must never replace a valid call
            # outcome with that harmless terminal response.
            return


class RetellChatCallRunner:
    """Drive multi-turn scenarios directly through Retell's native Chat API."""

    def __init__(self, adapter: ArtifactUploader, context: CallRunnerContext) -> None:
        self._adapter = adapter
        self._context = context
        path = context.bundle_dir / "contract.json"
        self._contract = (
            AgentContract.model_validate_json(path.read_text())
            if path.is_file()
            else None
        )

    async def run(
        self,
        scenario: Scenario,
        runtime: EnvironmentRuntime,
        *,
        world: World | None = None,
    ) -> CallOutcome:
        del world
        if self._contract is None:
            raise CallAborted(
                "chat_contract_unavailable: bundle/contract.json is absent"
            )
        agent_id = str(
            runtime.metadata.get("provider_target_id")
            or self._context.job.agent.config.get("agent_id")
            or ""
        ).strip()
        api_key = self._context.target_provider_secret_values.get(
            RETELL_API_KEY_ALIAS, ""
        )
        config = self._context.job.agent.config
        raw_dynamic = config.get("dynamic_variables")
        dynamic_variables = raw_dynamic if isinstance(raw_dynamic, Mapping) else {}
        version_value = config.get("agent_version")
        agent_version = int(version_value) if version_value not in (None, "") else None
        wrapper = RetellChatWrapper(
            api_key=api_key,
            agent_id=agent_id,
            agent_version=agent_version,
            dynamic_variables=dynamic_variables,
            api_base_url=str(
                config.get("provider_api_base_url") or "https://api.retellai.com"
            ),
        )
        document = _scenario_document(self._context.bundle_dir, scenario.scenario_key)
        conversation_scenario = _conversation_scenario(document)
        target_world = _tool_world(
            self._context.bundle_dir,
            self._contract,
            runtime,
            self._context.source_directory,
        )
        target = _HostedChatTarget(
            wrapper=wrapper,
            contract=self._contract,
            world=target_world,
            scenario_key=scenario.scenario_key,
            scenario_id=scenario.scenario_id,
        )
        started = datetime.now(timezone.utc)
        try:
            transcript = await _drive_conversation(
                target, conversation_scenario, self._contract, self._context.bundle_dir
            )
        except CallAborted:
            raise
        except Exception as exc:
            raise CallAborted(
                f"retell_chat_target_failed: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            await wrapper.aclose()
        ended = datetime.now(timezone.utc)
        transcript_id = await self._adapter.upload_artifact(
            (transcript.spoken() + "\n").encode(),
            kind=ArtifactKind.TRANSCRIPT,
            scenario_key=scenario.scenario_key,
        )
        calls = tuple(transcript.calls)
        if calls:
            trace = "\n".join(
                json.dumps(
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "result": call.result,
                        "ok": call.ok,
                        "error": call.error,
                        "refused": call.refused,
                        "at": call.at,
                    },
                    sort_keys=True,
                    default=str,
                )
                for call in calls
            ).encode()
            await self._adapter.upload_artifact(
                trace, kind=ArtifactKind.TOOL_TRACE, scenario_key=scenario.scenario_key
            )
        return CallOutcome(
            calls=calls,
            turns=len(transcript.exchanges),
            started_at=format_rfc3339_millis(started),
            ended_at=format_rfc3339_millis(ended),
            duration_ms=_duration_ms(started, ended),
            transcript_artifact=transcript_id,
        )


__all__ = [
    "RetellChatCallRunner",
    "RetellChatEnded",
    "RetellChatError",
    "RetellChatWrapper",
]
