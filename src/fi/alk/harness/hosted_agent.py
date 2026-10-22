"""Our own agent (connector `hosted_agent`): instructions and a model, reaching a world over MCP.

It runs in the trusted chat worker, never in the sandbox's bundle processes, because it calls the
model through Agent Command Center on the platform's key. Each turn is an OpenAI-compatible tool
loop: the model answers or asks for tools, the tools run through the world's `/mcp` endpoint one
at a time (Gemini ignores `parallel_tool_calls`), and tool-call ids go back unchanged (Gemini 3
keeps its thought signature in them). The agent reports nothing about its calls: the world's
`_alk_tool_trace`, written by `WorldTools`, is the evidence.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .bundle_v2 import _canonical_json
from .call_runner import (
    ArtifactUploader,
    CallRunnerContext,
    _clear_tool_trace_calls,
    _find_postgres_endpoint,
    _read_tool_trace,
    _upload_tool_trace,
)
from .chat_call_runner import (
    _conversation_scenario,
    _duration_ms,
    _record_text_usage,
    _scenario_document,
    _with_scenario_edits,
)
from .contract import AgentContract
from .hosted_scheduler import CallAborted, CallOutcome, Scenario, World
from .outbound import ArtifactKind, format_rfc3339_millis
from .process_runtime import EnvironmentRuntime
from .releases import HOSTED_AGENT as CONNECTOR
from .releases import WORLD_MCP_CAPABILITY
from .run.conversation import Transcript, converse
from .usage import simulator_funding
from .world_mcp import MCP_PATH

# Gateway names: the `vertex_ai/` prefix routes to Vertex; a bare name would go to the Gemini API.
Model = Literal["vertex_ai/gemini-3.7-flash", "vertex_ai/gemini-3.8-flash"]
MAX_TOOL_ROUNDS = 12


class AgentSpec(BaseModel):
    """What one of our agents is. Its digest is the agent's identity in a binding."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    model: Model = "vertex_ai/gemini-3.7-flash"

    def digest(self) -> str:
        return (
            "sha256:" + hashlib.sha256(_canonical_json(self.model_dump())).hexdigest()
        )


def _text(result: Any) -> str:
    return "\n".join(
        part.text for part in result.content if getattr(part, "type", "") == "text"
    )


class McpAgent:
    """The conversation target `converse` drives: one agent, one world, one conversation."""

    key = CONNECTOR
    spent_usd = 0.0

    def __init__(self, spec: AgentSpec, mcp_url: str, client: Any) -> None:
        self._spec = spec
        self._url = mcp_url
        self._client = client
        self._stack = AsyncExitStack()
        self._session: Any = None
        self._tools: list[dict[str, Any]] = []
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": spec.instructions}
        ]
        self.tokens = 0

    async def open(self) -> None:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        try:
            read, write, _ = await self._stack.enter_async_context(
                streamablehttp_client(self._url)
            )
            self._session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )
            await self._session.initialize()
            listed = (await self._session.list_tools()).tools
        except Exception as exc:
            raise CallAborted(
                f"world_mcp_unavailable: {type(exc).__name__}", code="world_unavailable"
            ) from exc
        self._tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or tool.name,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in listed
        ]

    async def say(self, utterance: str) -> str:
        self._messages.append({"role": "user", "content": utterance})
        for _round in range(MAX_TOOL_ROUNDS + 1):
            response = await self._client.chat.completions.create(
                model=self._spec.model,
                messages=self._messages,
                tools=self._tools,
            )
            if response.usage is not None:
                self.tokens += (response.usage.prompt_tokens or 0) + (
                    response.usage.completion_tokens or 0
                )
            message = response.choices[0].message
            if not message.tool_calls:
                answer = (message.content or "").strip()
                self._messages.append({"role": "assistant", "content": answer})
                return answer
            self._messages.append(message.model_dump(exclude_none=True))
            for call in message.tool_calls:
                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": await self._call(
                            call.function.name, call.function.arguments
                        ),
                    }
                )
        raise RuntimeError(
            f"the agent asked for tools more than {MAX_TOOL_ROUNDS} times in one turn"
        )

    async def _call(self, name: str, raw: str) -> str:
        try:
            arguments = json.loads(raw or "{}")
        except ValueError:
            return "error: the arguments are not valid JSON"
        result = await self._session.call_tool(name, arguments)
        return ("error: " if result.isError else "") + _text(result)

    async def close(self) -> None:
        await self._stack.aclose()


def _client() -> Any:
    from openai import AsyncOpenAI

    from fi.agentcc import gateway

    configured = gateway()
    if configured is None:
        raise CallAborted("hosted_agent_model_unavailable: AGENTCC_API_KEY is not set")
    return AsyncOpenAI(base_url=configured.openai_base_url, api_key=configured.api_key)


class HostedAgentCallRunner:
    """Run one scenario's conversation with our agent in its leased world."""

    def __init__(self, adapter: ArtifactUploader, context: CallRunnerContext) -> None:
        self._adapter = adapter
        self._context = context
        self._spec = AgentSpec.model_validate(context.job.agent.config.get("spec"))
        self._contract = AgentContract.model_validate_json(
            (context.bundle_dir / "contract.json").read_text(encoding="utf-8")
        )

    async def run(
        self,
        scenario: Scenario,
        runtime: EnvironmentRuntime,
        *,
        world: World | None = None,
    ) -> CallOutcome:
        del world  # The agent reaches the leased world through its MCP endpoint.
        endpoint = runtime.endpoints.get(WORLD_MCP_CAPABILITY)
        world_db = _find_postgres_endpoint(runtime)
        if endpoint is None or world_db is None:
            raise CallAborted(
                "hosted_agent_world_unavailable: world_mcp or world_db endpoint is absent",
                code="world_unavailable",
            )
        source_key = (
            getattr(scenario, "source_scenario_key", None) or scenario.scenario_key
        )
        conversation_scenario = _conversation_scenario(
            _with_scenario_edits(
                _scenario_document(self._context.bundle_dir, source_key),
                getattr(self._context.job, "metadata", None),
                source_key,
            )
        )
        agent = McpAgent(self._spec, endpoint.address.rstrip("/") + MCP_PATH, _client())
        funding = simulator_funding()
        _clear_tool_trace_calls(world_db.address)
        started = datetime.now(timezone.utc)
        transcript: Transcript | None = None
        failure: CallAborted | None = None
        try:
            transcript = await converse(
                agent,
                conversation_scenario,
                self._contract,
                world_root=self._context.bundle_dir,
            )
        except CallAborted as exc:
            failure = exc
            raise
        except Exception as exc:  # the agent's or the model's failure is the call's
            wrapped = CallAborted(
                f"hosted_agent_failed: {type(exc).__name__}: {exc}",
                code="target_agent_failed",
            )
            partial = getattr(exc, "partial_transcript", None)
            if partial is not None:
                wrapped.partial_transcript = partial  # type: ignore[attr-defined]
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
                agent_tokens=agent.tokens,
            )
        ended = datetime.now(timezone.utc)
        trace = _read_tool_trace(runtime)
        calls = trace or ()
        transcript.calls = list(calls)
        transcript_id = await self._adapter.upload_artifact(
            transcript.artifact(),
            kind=ArtifactKind.TRANSCRIPT,
            scenario_key=scenario.scenario_key,
        )
        await _upload_tool_trace(
            self._adapter, calls, scenario_key=scenario.scenario_key, binding=None
        )
        return CallOutcome(
            calls=calls,
            turns=len(transcript.exchanges),
            started_at=format_rfc3339_millis(started),
            ended_at=format_rfc3339_millis(ended),
            duration_ms=_duration_ms(started, ended),
            transcript_artifact=transcript_id,
            messages=tuple(transcript.canonical_messages()),
            evidence_collected=trace is not None,
        )
