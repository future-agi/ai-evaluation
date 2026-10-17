"""Hosted HTTP chat calls against an already-provisioned Bundle V2 process world."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urljoin

from fi.simulate.agent.wrapper import AgentInput
from fi.simulate.agent.wrappers.http import HTTPAgentWrapper

from .call_runner import (
    ArtifactUploader,
    CallRunnerContext,
    _clear_file_tool_calls,
    _collect_file_tool_calls,
    _with_scenario_edits,
)
from .contract import AgentContract
from .hosted_scheduler import CallAborted, CallOutcome, Scenario, World
from .outbound import ArtifactKind, format_rfc3339_millis
from .process_runtime import EnvironmentRuntime
from .run.conversation import TargetConversationEnded, Transcript, converse
from .scenario import Scenario as ConversationScenario
from .usage import (
    UsageDenied,
    UsageUnavailable,
    failure_domain_for_code,
    simulator_funding,
)
from .world.runtime import Call, GeneratedWorld
from .world.stores.postgres import AttachedPostgresStore

logger = logging.getLogger(__name__)


async def _record_text_usage(
    reporter: Any,
    *,
    scenario_key: str,
    started: datetime,
    funding: str,
    transcript: Transcript | None,
    failure: BaseException | None,
) -> None:
    if reporter is None:
        return
    partial = transcript or getattr(failure, "partial_transcript", None)
    simulator_tokens = (
        partial.simulator_input_tokens + partial.simulator_output_tokens
        if isinstance(partial, Transcript)
        else 0
    )
    completed = transcript is not None
    try:
        await asyncio.to_thread(
            reporter.record,
            action="text_call",
            scenario_key=scenario_key,
            amount=simulator_tokens,
            funding=funding,
            occurred_at=started,
            outcome="completed" if completed else "failed",
            failure_domain=(
                None
                if completed
                else failure_domain_for_code(getattr(failure, "code", "call_failed"))
            ),
        )
    except Exception:  # noqa: BLE001 - call failure must not hide a metering warning
        logger.exception(
            "Could not record hosted text usage for scenario %s",
            scenario_key,
        )
    if simulator_tokens == 0:
        logger.warning(
            "Hosted text call produced no simulator tokens for scenario %s",
            scenario_key,
        )


DEFAULT_CHAT_TARGET_TIMEOUT_SECONDS = 120.0


def _chat_target_timeout_seconds() -> float:
    """Return the per-turn target deadline without accepting unusable values."""
    raw = os.getenv("ALK_CHAT_TARGET_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_CHAT_TARGET_TIMEOUT_SECONDS
    try:
        configured = float(raw)
    except ValueError:
        return DEFAULT_CHAT_TARGET_TIMEOUT_SECONDS
    return (
        configured
        if 1.0 <= configured <= 600.0
        else DEFAULT_CHAT_TARGET_TIMEOUT_SECONDS
    )


def _duration_ms(started: datetime, ended: datetime) -> int:
    return max(0, round((ended - started).total_seconds() * 1000))


def _scenario_document(bundle_dir: Path, key: str) -> dict[str, Any]:
    for path in sorted((bundle_dir / "scenarios").glob("*/scenario.json")):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(body, dict) and body.get("scenario_key") == key:
            return body
    raise CallAborted(f"chat_scenario_document_unavailable: scenario_key={key!r}")


def _qmark_to_postgres(statement: str) -> str:
    """Translate generated SQLite-style positional placeholders, never SQL structure."""
    return re.sub(r"\?", "%s", statement)


class _HostedToolStore:
    """Generated handler ``Db`` adapter over the leased world's attached Postgres database."""

    key = "hosted-postgres"

    def __init__(self, dsn: str) -> None:
        self._store = AttachedPostgresStore(dsn)

    def start(self) -> None:
        self._store.start()

    def stop(self) -> None:
        return

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        return self._store.query(_qmark_to_postgres(sql), params)

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        return self._store.execute(_qmark_to_postgres(sql), params)

    def state(
        self, only: Sequence[str] | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        return self._store.state(only=only)

    def collections(self) -> list[str]:
        return list(self._store.state())

    def records(self, collection: str) -> list[dict[str, Any]]:
        return self._store.table(collection)

    def add(self, collection: str, record: Mapping[str, Any]) -> dict[str, Any]:
        return self._store.add(collection, dict(record))


def _tool_world(
    bundle_dir: Path,
    contract: AgentContract,
    runtime: EnvironmentRuntime,
    source_directory: Path | None = None,
) -> GeneratedWorld:
    endpoint = runtime.endpoints.get("world_db")
    if endpoint is None or endpoint.protocol != "postgres":
        raise CallAborted(
            "chat_world_unavailable: world_db postgres endpoint is absent"
        )
    world = GeneratedWorld(store=_HostedToolStore(endpoint.address))
    world.tools = [tool.model_dump(mode="json") for tool in contract.tools]
    world.handlers = {}
    for tool in contract.tools:
        path = bundle_dir / "handlers" / f"{tool.name}.py"
        if path.is_file():
            world.handlers[tool.name] = path.read_text(encoding="utf-8")
    world.refusal_signature = contract.refusal_signature
    if source_directory is not None:
        world.reach(str(source_directory))
    return world


def _tools(contract: AgentContract) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for spec in contract.tools:
        properties = {
            argument: {"type": _json_type(spec.arg_types.get(argument, "string"))}
            for argument in spec.args
        }
        values.append(
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(spec.args),
                },
            }
        )
    return values


def _json_type(declared: str) -> str:
    normalized = str(declared or "").lower()
    if any(mark in normalized for mark in ("int", "float", "number")):
        return "number"
    if "bool" in normalized:
        return "boolean"
    if any(mark in normalized for mark in ("list", "array", "sequence")):
        return "array"
    if any(mark in normalized for mark in ("dict", "map", "object")):
        return "object"
    return "string"


def _tool_call(call: dict[str, Any], index: int) -> tuple[str, dict[str, Any], str]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = str(call.get("name") or function.get("name") or "")
    raw = call.get("arguments", function.get("arguments", {}))
    if isinstance(raw, str):
        try:
            arguments = json.loads(raw)
        except ValueError:
            arguments = {"_raw": raw}
    else:
        arguments = dict(raw or {}) if isinstance(raw, dict) else {}
    return name, arguments, str(call.get("id") or f"call_{index}")


def _tool_response_result(response: Mapping[str, Any]) -> Any:
    """Return the callback's real tool result without inventing a second execution."""
    value = response.get("result", response.get("content"))
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _record_completed_tool_call(
    world: GeneratedWorld,
    *,
    name: str,
    arguments: Mapping[str, Any],
    response: Mapping[str, Any],
) -> None:
    """Record a tool the submitted callback already executed.

    Callback-backed agents can return the request and its completed response together. Replaying
    that request through the generated world both risks repeating a side effect and incorrectly
    turns a real success into ``no such tool`` when no mock handler was authored. The callback's
    response is the authoritative execution evidence at this seam.
    """
    error_value = response.get("error")
    error = str(error_value) if error_value not in (None, "") else ""
    refused = bool(response.get("refused", False))
    declared_success = response.get("success", response.get("ok"))
    ok = (
        bool(declared_success)
        if declared_success is not None
        else not error and not refused
    )
    world.calls.append(
        Call(
            name=name,
            arguments=dict(arguments),
            result=_tool_response_result(response),
            ok=ok,
            refused=refused,
            error=error,
            at=time.time(),
        )
    )


class _HostedChatTarget:
    """The already-running Bundle V2 chat process, exposed as the normal conversation target.

    ``converse`` owns the simulated customer's turns.  This target owns only the submitted
    agent's side of the exchange and response-carried tool evidence.  Keeping that split identical
    to the local repository target prevents the hosted lane from silently becoming a one-message
    smoke test again.
    """

    key = "hosted_repository"

    def __init__(
        self,
        *,
        wrapper: Any,
        contract: AgentContract,
        world: GeneratedWorld,
        scenario_key: str,
        scenario_id: str,
        seed: str = "",
    ) -> None:
        self._wrapper = wrapper
        # The call's own seed: one draw decides whether a participant bound by probability enters.
        self.seed = seed
        self._contract = contract
        self.world = world
        self._scenario_key = scenario_key
        self._scenario_id = scenario_id
        self._messages: list[dict[str, Any]] = []
        self._turn = 0

    async def open(self) -> None:
        return

    def interject(self, name: str, text: str) -> None:
        """A second participant's message, heard by the agent with the person's next turn."""
        label = re.sub(r"[^A-Za-z0-9_-]", "_", name)[:64] or "participant"
        self._messages.append({"role": "user", "name": label, "content": text})

    async def say(self, utterance: str) -> str:
        self._messages.append({"role": "user", "content": utterance})
        for continuation in range(8):
            response = await self._wrapper.call(
                AgentInput(
                    thread_id=self._scenario_key,
                    execution_id=self._scenario_id,
                    turn_index=self._turn,
                    scenario_name=self._scenario_key,
                    modality="text",
                    messages=list(self._messages),
                    new_message=dict(self._messages[-1]),
                    tools=_tools(self._contract)
                    if self._contract.runtime
                    and self._contract.runtime.interface
                    and self._contract.runtime.interface.include_tools
                    else [],
                )
            )
            trace = dict((response.metadata or {}).get("external_agent") or {})
            if trace and not trace.get("success", False):
                raise RuntimeError(
                    str(trace.get("error") or "submitted endpoint request failed")
                )
            conversation_ended = bool(
                (response.metadata or {}).get("conversation_ended")
            )
            returned = list(response.tool_calls or [])
            if not returned:
                answer = response.content.strip()
                if conversation_ended:
                    raise TargetConversationEnded(answer)
                self._messages.append({"role": "assistant", "content": answer})
                self._turn += 1
                return answer

            self._messages.append(
                {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": returned,
                }
            )
            response_by_id = {
                str(item.get("tool_call_id") or item.get("id") or ""): item
                for item in response.tool_responses or []
                if isinstance(item, Mapping)
                and (item.get("tool_call_id") or item.get("id"))
            }
            returned_ids: set[str] = set()
            for index, call in enumerate(returned, start=1):
                name, arguments, call_id = _tool_call(call, index)
                returned_ids.add(call_id)
                provided = response_by_id.get(call_id)
                if provided is not None:
                    _record_completed_tool_call(
                        self.world,
                        name=name,
                        arguments=arguments,
                        response=provided,
                    )
                    result_content = provided.get("content", provided.get("result"))
                    if not isinstance(result_content, str):
                        result_content = json.dumps(result_content, default=str)
                else:
                    result = self.world.handle_tool_call(
                        {"id": call_id, "name": name, "arguments": arguments}
                    )
                    result_content = (
                        result.content if result is not None else f"no such tool {name}"
                    )
                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": result_content,
                    }
                )

            # A callback-backed repository has already run its own real tools.  It returns their
            # responses beside the final text; replaying the calls above records deterministic
            # evidence in the generated world, but asking the callback a second time would run the
            # tools twice.  HTTP agents that only return requests continue normally with the
            # generated-world responses appended above.
            provided_ids = set(response_by_id)
            if returned_ids and returned_ids.issubset(provided_ids):
                answer = response.content.strip()
                if conversation_ended:
                    raise TargetConversationEnded(answer)
                self._messages.append({"role": "assistant", "content": answer})
                self._turn += 1
                return answer
        raise RuntimeError(
            "submitted chat agent exceeded 8 tool continuations in one turn"
        )

    async def close(self) -> None:
        return

    @property
    def spent_usd(self) -> float:
        # Provider-side target cost is not observable at this transport seam.
        return 0.0


def _conversation_scenario(document: dict[str, Any]) -> ConversationScenario:
    try:
        normalized = dict(document)
        normalized.setdefault(
            "name", str(normalized.get("scenario_key") or "hosted-chat-scenario")
        )
        return ConversationScenario.model_validate(normalized)
    except Exception as exc:  # noqa: BLE001 - normalize malformed bundle content at the call seam
        raise CallAborted(f"chat_scenario_invalid: {exc}") from exc


async def _drive_conversation(
    target: _HostedChatTarget,
    scenario: ConversationScenario,
    contract: AgentContract,
    bundle_dir: Path,
) -> Transcript:
    return await converse(
        target,
        scenario,
        contract,
        world_root=bundle_dir,
        seed=target.seed,
    )


class HostedChatCallRunner:
    """Drive a repository chat ingress inside its leased world."""

    def __init__(self, adapter: ArtifactUploader, context: CallRunnerContext) -> None:
        self._adapter = adapter
        self._context = context
        contract_path = context.bundle_dir / "contract.json"
        if not contract_path.is_file():
            self._contract: AgentContract | None = None
        else:
            self._contract = AgentContract.model_validate_json(
                contract_path.read_text(encoding="utf-8")
            )

    async def run(
        self,
        scenario: Scenario,
        runtime: EnvironmentRuntime,
        *,
        world: World | None = None,
    ) -> CallOutcome:
        del (
            world
        )  # Handler execution uses the same leased world's endpoint from runtime.
        if self._contract is None:
            raise CallAborted(
                "chat_contract_unavailable: bundle/contract.json is absent"
            )
        interface = self._contract.runtime.interface if self._contract.runtime else None
        if interface is None or interface.kind not in {"http", "callable"}:
            raise CallAborted(
                "chat_interface_unsupported: an HTTP or callable runtime interface is required"
            )
        endpoint = runtime.endpoints.get("target_http")
        if endpoint is None:
            raise CallAborted(
                "chat_capability_unavailable: target_http endpoint is absent"
            )

        source_key = getattr(scenario, "source_scenario_key", None) or scenario.scenario_key
        document = _with_scenario_edits(
            _scenario_document(self._context.bundle_dir, source_key),
            getattr(self._context.job, "metadata", None),
            source_key,
        )
        conversation_scenario = _conversation_scenario(document)
        if not conversation_scenario.instruction.strip():
            raise CallAborted("chat_scenario_invalid: instruction is empty")
        target_world = _tool_world(
            self._context.bundle_dir,
            self._contract,
            runtime,
            self._context.source_directory,
        )
        adapter_path = "/invoke" if interface.kind == "callable" else interface.path
        adapter_protocol = (
            "fi.alk" if interface.kind == "callable" else interface.protocol
        )
        wrapper = HTTPAgentWrapper(
            endpoint=urljoin(
                endpoint.address.rstrip("/") + "/", adapter_path.lstrip("/")
            ),
            protocol=adapter_protocol,
            include_tools=interface.include_tools,
            request_template=interface.request_template,
            response_path=interface.response_path,
            setup_requests=[
                item.model_dump(mode="python") for item in interface.setup_requests
            ],
            timeout=_chat_target_timeout_seconds(),
            metadata={
                "target": "hosted_repository_runtime",
                "scenario": scenario.scenario_key,
            },
        )
        funding = simulator_funding()
        if self._context.usage_reporter is not None and funding == "platform":
            try:
                await asyncio.to_thread(self._context.usage_reporter.check, "text_call")
            except UsageDenied as exc:
                raise CallAborted(
                    f"text_usage_check_denied: {exc}", code="usage_exhausted"
                ) from exc
            except UsageUnavailable as exc:
                raise CallAborted(
                    f"text_usage_check_failed: {exc}", code="usage_check_failed"
                ) from exc
        started = datetime.now(timezone.utc)
        transcript: Transcript | None = None
        failure: CallAborted | None = None
        _clear_file_tool_calls(runtime)
        # Drawn once per call, so a probability binding varies across runs and agrees within one.
        seed = secrets.token_hex(8)
        try:
            transcript = await _drive_conversation(
                _HostedChatTarget(
                    wrapper=wrapper,
                    contract=self._contract,
                    world=target_world,
                    scenario_key=scenario.scenario_key,
                    scenario_id=scenario.scenario_id,
                    seed=seed,
                ),
                conversation_scenario,
                self._contract,
                self._context.bundle_dir,
            )
        except CallAborted as exc:
            failure = exc
            raise
        except Exception as exc:  # noqa: BLE001 - convert target transport failures to call faults
            wrapped = CallAborted(
                f"chat_target_failed: {type(exc).__name__}: {exc}",
                code="target_agent_failed",
            )
            partial = getattr(exc, "partial_transcript", None)
            if partial is not None:
                setattr(wrapped, "partial_transcript", partial)
            failure = wrapped
            raise wrapped from exc
        finally:
            await _record_text_usage(
                self._context.usage_reporter,
                scenario_key=scenario.scenario_key,
                started=started,
                funding=funding,
                transcript=transcript,
                failure=failure,
            )

        # The submitted agent may execute tools entirely inside its Python process instead
        # of returning tool requests over HTTP. These observed calls belong to the same attempt;
        # do not replay the tool through the generated world or fabricate evidence from text.
        traced_calls = _collect_file_tool_calls(runtime)
        if traced_calls and not transcript.calls:
            transcript.calls.extend(traced_calls)

        ended = datetime.now(timezone.utc)
        transcript_id = await self._adapter.upload_artifact(
            transcript.artifact(),
            kind=ArtifactKind.TRANSCRIPT,
            scenario_key=scenario.scenario_key,
        )
        calls = tuple(transcript.calls)
        if calls:
            tool_trace = "\n".join(
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
            ).encode("utf-8")
            await self._adapter.upload_artifact(
                tool_trace,
                kind=ArtifactKind.TOOL_TRACE,
                scenario_key=scenario.scenario_key,
            )
        return CallOutcome(
            calls=calls,
            turns=sum(1 for turn in transcript.exchanges if turn.speaker != "sub_actor"),
            started_at=format_rfc3339_millis(started),
            ended_at=format_rfc3339_millis(ended),
            duration_ms=_duration_ms(started, ended),
            transcript_artifact=transcript_id,
            messages=tuple(transcript.canonical_messages()),
        )
