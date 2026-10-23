"""The Claude Code backend: the loop this harness grew up on.

Adapts the neutral ``SessionSpec`` to ``ClaudeAgentOptions`` exactly the way the stages built
them before the seam existed: same gate hooks, same permission callback, same provider env,
same disallowed list. With ``ALK_HARNESS`` unset this backend runs, so nothing here may drift
from what the stages did on their own.

Claude Code supplies Read/Glob/Grep and AskUserQuestion itself, so builtins are granted by
name rather than implemented here.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from typing import Any, AsyncIterator

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SdkMcpTool,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
)

from fi import agentcc

from .base import (
    DELEGATE_TOOL,
    MOST_WORKERS_AT_ONCE,
    Call,
    ModelReply,
    Say,
    SessionOpened,
    SessionSpec,
    StageDone,
    TERMINAL_SAVE_TOOLS,
    ToolReturned,
    ToolServer,
    WorkerSpec,
    tool_input_schema,
)

# The loop is Claude Code; the model it bills is not.
DEFAULT_MODEL = "gemini-3.7-flash"

# The SDK reports its worker tool under either name depending on version, so both are granted.
DELEGATION_TOOLS = ("Agent", "Task")

def _can_reach_its_workers(spec: SessionSpec, allowed: list[str]) -> None:
    """Refuse a session whose workers it has no way to call."""
    if not spec.workers:
        return
    reachable = set(allowed)
    if reachable & set(DELEGATION_TOOLS):
        return
    if any(name in DELEGATION_TOOLS for name in reachable):
        return
    raise ValueError(
        f"this stage has workers ({', '.join(sorted(spec.workers))}) and no way to call them: "
        "no delegation tool is reachable, so it can only read. Reachable: "
        f"{sorted(reachable)}"
    )


def _gateway_compatible_schema(value: Any) -> Any:
    def normalize(item: Any) -> Any:
        if isinstance(item, list):
            return [normalize(entry) for entry in item]
        if not isinstance(item, dict):
            return item
        result = {key: normalize(entry) for key, entry in item.items()}
        kind = result.get("type")
        if isinstance(kind, list):
            concrete = [entry for entry in kind if entry != "null"]
            if len(concrete) == 1:
                result["type"] = concrete[0]
        enum = result.get("enum")
        if isinstance(enum, list) and None in enum:
            result["enum"] = [entry for entry in enum if entry is not None]
        return result

    return normalize(tool_input_schema(value))


def _sdk_server(server: ToolServer, *, gateway_compatible: bool = False) -> Any:
    """A ToolServer as the in-process MCP server the SDK routes calls to."""
    return create_sdk_mcp_server(
        name=server.name,
        version=server.version,
        tools=[
            SdkMcpTool(
                name=spec.name,
                description=spec.description,
                input_schema=(
                    _gateway_compatible_schema(spec.input_schema)
                    if gateway_compatible
                    else spec.input_schema
                ),
                handler=spec.handler,
            )
            for spec in server.tools
        ],
    )


def _definition(worker: WorkerSpec, parent: SessionSpec) -> AgentDefinition:
    """A ``WorkerSpec`` as this SDK's own sub-agent definition."""
    tools = [name for name in worker.granted(parent) if name != DELEGATE_TOOL]
    # Only when the worker itself asks for it, never inherited from the stage.
    if DELEGATE_TOOL in (worker.builtins or ()):
        tools.extend(DELEGATION_TOOLS)
    return AgentDefinition(
        description=worker.description,
        prompt=worker.instructions,
        tools=tools,
        mcpServers=list(worker.servers or parent.servers),
        model=worker.model or "inherit",
        maxTurns=worker.max_turns,
        background=False,
    )


def _said(text: str, *, is_error: bool = False) -> dict[str, Any]:
    """A tool result in the shape every harness tool already returns."""
    reply: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if is_error:
        reply["is_error"] = True
    return reply


def _child_of(parent: SessionSpec, worker: WorkerSpec) -> SessionSpec:
    """The session one worker runs in, falling back to the parent's settings."""
    return SessionSpec(
        system_prompt=worker.instructions,
        servers=dict(worker.servers or parent.servers),
        builtins=tuple(
            name for name in (worker.builtins or parent.builtins) if name != DELEGATE_TOOL
        ),
        cwd=parent.cwd,
        max_turns=worker.max_turns,
        model=worker.model or parent.model,
        ask=parent.ask,
        gated=parent.gated,
        thinking=parent.thinking,
        permission_override=parent.permission_override,
        idle_timeout_seconds=parent.idle_timeout_seconds,
    )


def _flattened(content: Any) -> str:
    """A tool result's content as one string, however the SDK packaged it."""
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return content if isinstance(content, str) else str(content)


class ClaudeSession:
    """One Claude Code session, translated to the neutral reply vocabulary."""

    def __init__(
        self,
        options: ClaudeAgentOptions,
        *,
        reported_model: str | None = None,
        streaming: bool = False,
    ) -> None:
        self._options = options
        self._reported_model = reported_model
        self._streaming = streaming
        self._client: ClaudeSDKClient | None = None
        self._mirror_errors: list[str] = []
        self._tool_names: dict[str, str] = {}
        self._terminal_save_succeeded = False

    async def start(self) -> None:
        self._client = ClaudeSDKClient(options=self._options)
        await self._client.connect()

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.disconnect()
            self._client = None

    async def send(self, message: str) -> None:
        if self._client is None:
            raise RuntimeError("session is not open")
        await self._client.query(message)

    async def interrupt(self) -> bool:
        if self._client is None:
            return False
        await self._client.interrupt()
        return True

    async def resume(self, invocation_id: str) -> None:
        del invocation_id
        raise RuntimeError(
            "Claude Agent SDK resumes sessions, not interrupted invocations"
        )

    async def replies(self) -> AsyncIterator[Any]:
        if self._client is None:
            raise RuntimeError("session is not open")
        terminal = False
        async for received in self._client.receive_response():
            if terminal and not isinstance(received, ResultMessage):
                continue
            for reply in self._translate(received):
                yield reply
            if self._terminal_save_succeeded:
                self._terminal_save_succeeded = False
                terminal = True
                await self._client.interrupt()

    def _translate(self, received: Any) -> list[Any]:
        if isinstance(received, StreamEvent):
            event = received.event
            delta = event.get("delta") if isinstance(event, dict) else None
            if (
                isinstance(event, dict)
                and event.get("type") == "content_block_delta"
                and isinstance(delta, dict)
                and delta.get("type") == "text_delta"
                and delta.get("text")
            ):
                return [
                    ModelReply(
                        parts=[
                            Say(
                                text=str(delta["text"]),
                                partial=True,
                                event_id=received.uuid,
                            )
                        ]
                    )
                ]
            return []
        if type(received).__name__ == "MirrorErrorMessage":
            self._mirror_errors.append(
                str(getattr(received, "error", "session transcript mirror failed"))
            )
            return []
        if isinstance(received, SystemMessage):
            data = received.data if isinstance(received.data, dict) else {}
            return [SessionOpened(session_id=data.get("session_id"))]
        if isinstance(received, AssistantMessage):
            parts: list[Any] = []
            for block in received.content:
                if isinstance(block, TextBlock) and not self._streaming:
                    parts.append(Say(text=block.text))
                elif isinstance(block, ToolUseBlock):
                    self._tool_names[block.id] = block.name
                    parts.append(
                        Call(id=block.id, name=block.name, arguments=block.input)
                    )
            return [
                ModelReply(
                    parts=parts,
                    model=self._reported_model
                    or getattr(received, "model", "")
                    or "",
                )
            ]
        if isinstance(received, ResultMessage):
            # subtype alone is not the outcome. A call that failed upstream still arrives with
            # subtype "success", so the error facts ride along and Stage decides what failed.
            counted = _tokens(getattr(received, "model_usage", None))
            errors = [
                *list(getattr(received, "errors", None) or []),
                *self._mirror_errors,
            ]
            return [
                StageDone(
                    outcome=received.subtype,
                    turns=received.num_turns,
                    cost_usd=_cost(
                        getattr(received, "model_usage", None),
                        counted,
                        received.total_cost_usd,
                    ),
                    **counted,
                    session_id=received.session_id,
                    models=(
                        {self._reported_model}
                        if self._reported_model
                        else set(getattr(received, "model_usage", None) or {})
                    ),
                    is_error=bool(
                        getattr(received, "is_error", False) or self._mirror_errors
                    ),
                    api_error_status=getattr(received, "api_error_status", None),
                    errors=errors,
                )
            ]
        blocks = getattr(received, "content", None)
        if isinstance(blocks, list):
            returned = []
            for block in blocks:
                if isinstance(block, ToolResultBlock):
                    name = self._tool_names.pop(block.tool_use_id, "")
                    self._terminal_save_succeeded = bool(
                        name in TERMINAL_SAVE_TOOLS
                        and not getattr(block, "is_error", False)
                    )
                    returned.append(
                        ToolReturned(
                            id=block.tool_use_id,
                            text=_flattened(block.content),
                            is_error=bool(getattr(block, "is_error", False)),
                        )
                    )
            return returned
        return []


def _cost(model_usage: Any, counted: dict[str, int], reported: float | None) -> float | None:
    """What the run cost, priced here where possible, else the CLI's own figure."""
    from .vertex_gemini import priced

    named = [str(name) for name in (model_usage or {})]
    ours = [
        priced(name, counted["tokens_in"], counted["tokens_out"], counted["tokens_cached"])
        for name in named
    ]
    known = [one for one in ours if one is not None]
    if not known:
        return reported
    # One stage runs on one model: summing would price the same tokens once per reported id.
    return max(known)


def _tokens(model_usage: Any) -> dict[str, int]:
    """Input and output tokens across every model a stage used, for the ledger to audit against.

    Read defensively: this is the SDK's shape, not ours, and a stage must not fail over accounting.
    """
    read = 0
    written = 0
    cached = 0
    for usage in (model_usage or {}).values():
        if isinstance(usage, dict):
            read += int(usage.get("inputTokens") or usage.get("input_tokens") or 0)
            written += int(usage.get("outputTokens") or usage.get("output_tokens") or 0)
            cached += int(
                usage.get("cacheReadInputTokens")
                or usage.get("cache_read_input_tokens")
                or 0
            )
        else:
            read += int(getattr(usage, "input_tokens", 0) or 0)
            written += int(getattr(usage, "output_tokens", 0) or 0)
            cached += int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    # The SDK reports cache reads beside fresh input; the ledger reads them as part of tokens_in.
    return {"tokens_in": read + cached, "tokens_out": written, "tokens_cached": cached}


class ClaudeBackend:
    name = "claude"
    default_model = DEFAULT_MODEL

    def can_drive(self, model: str) -> bool:
        # Agent CC's native Anthropic endpoint accepts the Claude Agent SDK wire format and
        # translates it for any provider configured behind the virtual key. Without a gateway,
        # this backend only advertises models supported by Claude Code directly.
        named = (model or "").lower()
        if "claude" in named:
            return True
        gateway_ready = agentcc.gateway() is not None or bool(
            os.environ.get("ALK_CLAUDE_GATEWAY_URL", "").strip()
            and os.environ.get("ALK_CLAUDE_GATEWAY_API_KEY", "").strip()
        )
        return gateway_ready and bool(named)

    def create(self, spec: SessionSpec) -> ClaudeSession:
        from ..config import gateway_wire_model

        # An alias on the wire is not what was billed, so report the model the run chose.
        reported = spec.model if gateway_wire_model(spec.model) != spec.model else None
        context = spec.conversation
        return ClaudeSession(
            self._options(spec),
            reported_model=reported,
            streaming=bool(context and context.streaming),
        )

    def _options(self, spec: SessionSpec) -> ClaudeAgentOptions:
        """The SDK options for one session."""
        from ..config import (
            UNWANTED,
            gate_hooks,
            permission_gate,
            behind_gateway,
            gateway_wire_model,
            provider_env,
            thinking_config,
        )

        wire_model = gateway_wire_model(spec.model)
        gateway_compatible = behind_gateway(spec.model)
        # A worker's calls are made inside this session, so the gate covers its tools too.
        allowed = [name for name in spec.granted_anywhere() if name != DELEGATE_TOOL]
        # Union per server name; each agent is still restricted by its own `tools` allowlist.
        servers: dict[str, ToolServer] = {}
        for source in (spec.servers, *(worker.servers for worker in spec.workers.values())):
            for server_name, server in (source or {}).items():
                existing = servers.get(server_name)
                if existing is None:
                    servers[server_name] = server
                    continue
                seen = {tool.name for tool in existing.tools}
                servers[server_name] = dataclasses.replace(
                    existing,
                    tools=[*existing.tools, *(t for t in server.tools if t.name not in seen)],
                )
        environment = dict(provider_env(spec.model))
        # A whole skill with its references is one tool result; the default cap refuses it.
        environment.setdefault("MAX_MCP_OUTPUT_TOKENS", "60000")
        context = spec.conversation
        if context is not None:
            environment["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
            environment["CLAUDE_CODE_PROJECT_DIR_NAME"] = context.session_id
            if context.config_dir:
                environment["CLAUDE_CONFIG_DIR"] = context.config_dir
        if spec.workers:
            environment["CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS"] = str(MOST_WORKERS_AT_ONCE)
            # A backgrounded worker lets the stage end before the worker reports.
            environment["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] = "1"
            allowed.extend(DELEGATION_TOOLS)
        _can_reach_its_workers(spec, allowed)
        options = ClaudeAgentOptions(
            system_prompt=_prompt_for(spec.system_prompt),
            allowed_tools=allowed,
            mcp_servers={
                server_name: _sdk_server(
                    server, gateway_compatible=gateway_compatible
                )
                for server_name, server in servers.items()
            },
            agents=(
                {
                    name: _definition(worker, spec)
                    for name, worker in spec.workers.items()
                }
                or None
            ),
            strict_mcp_config=True,
            setting_sources=[],
            max_turns=spec.max_turns,
            model=wire_model,
            env=environment,
            include_partial_messages=bool(context and context.streaming),
            resume=context.resume_session_id if context is not None else None,
            session_store=context.transcript_store if context is not None else None,
            session_store_flush="eager",
        )
        if spec.cwd is not None:
            options.cwd = spec.cwd
        if spec.gated:
            # Not acceptEdits: that auto-approves Edit and Write before the permission callback
            # is consulted, so a stage could rewrite an artifact by hand and skip the tool whose
            # whole job is to validate that change.
            options.permission_mode = "default"
            options.disallowed_tools = [name for name in UNWANTED if name not in spec.builtins]
            options.hooks = gate_hooks(allowed)
            options.can_use_tool = spec.permission_override or permission_gate(
                spec.ask, allowed
            )
        if spec.thinking:
            options.thinking = thinking_config()
        return options


# A string system prompt goes onto the CLI's argv, so anything larger goes through a file.
_PROMPT_ON_ARGV = 16_000


def _prompt_for(prompt: str | None) -> Any:
    """The prompt as the SDK should receive it: inline while small, a file once it is not."""
    if not prompt or len(prompt) <= _PROMPT_ON_ARGV:
        return prompt
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".md", prefix="alk-system-prompt-", delete=False, encoding="utf-8"
    )
    with handle as written:
        written.write(prompt)
    return {"type": "file", "path": handle.name}
