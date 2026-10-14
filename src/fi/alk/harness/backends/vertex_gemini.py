"""A harness backend that runs stages on Gemini through Vertex AI, on Google's ADK.

The Agent Development Kit is Google's counterpart to the Claude Agent SDK: it owns the agentic
loop, executes tools, and holds session history, the way this harness expects a backend to. We
adapt at the same seam as the Claude backend and nothing more: a ``ToolSpec`` becomes an ADK
tool through ADK's own extension point (``BaseTool`` with an explicit declaration), and ADK's
event stream is translated into the neutral reply vocabulary. The loop itself is not ours.

The Gemini 3.x models this exists for are served from the ``global`` endpoint only, which is
why ``ALK_VERTEX_LOCATION`` defaults to ``global`` rather than to a region. Regional Vertex
deployments of older models can point it elsewhere. With Agent Command Center configured
(``AGENTCC_API_KEY``) the same requests go to the gateway instead, which reaches Vertex itself.

Read, Glob and Grep come from files.py when a stage grants them. AskUserQuestion is offered only
when an attended session supplies an operator callback; unattended authoring keeps the tool absent.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import date
from typing import Any

from fi import agentcc

from .base import (
    ASK_TOOL,
    FILE_TOOLS,
    Call,
    ModelReply,
    Say,
    SessionOpened,
    SessionSpec,
    StageDone,
    ToolReturned,
    ToolSpec,
    qualified,
)
from .files import file_tools

DEFAULT_MODEL = "gemini-3.7-flash"

_TERMINAL_SAVE_TOOLS = frozenset(
    {
        "mcp__world__save_world",
        "mcp__provision__save_environment",
        "mcp__scenarios__save_scenarios",
        # Source-data review is another persisted authoring boundary.  Its handler only
        # succeeds after every scenario was reviewed and at least one executable invariant
        # was declared.  Letting ADK take another turn after that success can burn the entire
        # call budget and turn a completed review into a spurious validation failure.
        "mcp__source_data__finish_review",
        # A validated repair patch is itself the complete output of the repair stage.  Stop
        # immediately after accepting it instead of spending the remaining agent budget on
        # another model turn that can only restate or replace an already-valid patch.
        "mcp__repair__submit_world_ir_patch",
    }
)

# Vertex list pricing per 1M tokens: (input, output, the day this pair was last checked against
# the platform's litellm model table). An unknown or stale model reports no cost rather than a
# wrong one, and shows up in `unpriced_turns`.
# What Vertex charges for a cache read, as a share of the input rate.
CACHE_READ_SHARE = 0.10

PRICES_PER_MILLION = {
    "gemini-3.8-flash": (0.75, 3.75, "2026-12-31"),
    "gemini-3.7-flash": (0.75, 3.75, "2026-12-31"),
    "gemini-3.6-flash": (0.75, 3.75, "2026-12-31"),
    "gemini-3.5-transcribe-preview": (2.5, 12, "2026-12-31"),
    "gemini-3.5-transcribe-live-preview": (3.5, 21, "2026-12-31"),
    "gemini-3.5-flash-lite": (0.3, 2.5, "2026-12-31"),
    "gemini-3.5-flash": (1.5, 9, "2026-12-31"),
    "gemini-3.1-pro-preview-customtools": (2, 12, "2026-12-31"),
    "gemini-3.1-pro-preview": (2, 12, "2026-12-31"),
    "gemini-3.1-flash-lite-preview": (0.25, 1.5, "2026-12-31"),
    "gemini-3.1-flash-lite-image": (0.25, 1.5, "2026-12-31"),
    "gemini-3.1-flash-lite": (0.25, 1.5, "2026-12-31"),
    "gemini-3.1-flash-image-preview": (0.5, 3, "2026-12-31"),
    "gemini-3.1-flash-image": (0.5, 3, "2026-12-31"),
    "gemini-3-pro-preview": (2, 12, "2026-12-31"),
    "gemini-3-pro-image-preview": (2, 12, "2026-12-31"),
    "gemini-3-pro-image": (2, 12, "2026-12-31"),
    "gemini-3-flash-preview": (0.5, 3, "2026-12-31"),
}

_PYTHON_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

# The keys Vertex's Schema type accepts. JSON Schema carries more; anything else is dropped
# rather than passed through, because an unknown key fails declaration validation and takes
# the whole stage down before its first turn.
_SCHEMA_KEYS = (
    "description",
    "enum",
    "format",
    "items",
    "maximum",
    "maxItems",
    "minimum",
    "minItems",
    "nullable",
    "pattern",
    "properties",
    "required",
    "type",
    "anyOf",
)


def _gemini_schema(schema: Any) -> dict[str, Any]:
    """A JSON Schema fragment as Vertex's Schema dialect.

    The real difference is nullability: JSON Schema says ``"type": ["string", "null"]``,
    Vertex says ``"type": "string", "nullable": true``. Everything Vertex does not know is
    dropped, recursively, so a tool schema written for the loosest backend still declares.
    """
    if not isinstance(schema, dict):
        return {"type": "string"}
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _SCHEMA_KEYS:
            continue
        if key == "type" and isinstance(value, list):
            bare = [entry for entry in value if entry != "null"]
            cleaned["type"] = bare[0] if bare else "string"
            if "null" in value:
                cleaned["nullable"] = True
        elif key == "properties" and isinstance(value, dict):
            cleaned["properties"] = {
                name: _gemini_schema(inner) for name, inner in value.items()
            }
        elif key == "items":
            cleaned["items"] = _gemini_schema(value)
        elif key == "anyOf" and isinstance(value, list):
            cleaned["anyOf"] = [_gemini_schema(inner) for inner in value]
        elif key == "enum" and isinstance(value, list):
            # Vertex enums are strings; None inside one is JSON Schema's way of saying
            # nullable, and everything else is stringified the way the model will echo it.
            cleaned["enum"] = [str(entry) for entry in value if entry is not None]
            if None in value:
                cleaned["nullable"] = True
        else:
            cleaned[key] = value
    # Vertex refuses an array that does not say what it holds. JSON Schema treats items as
    # optional, and most such arrays here carry row-shaped dicts, so an open object is the
    # faithful default.
    if cleaned.get("type") == "array" and "items" not in cleaned:
        cleaned["items"] = {"type": "object"}
    return cleaned


def _json_schema(schema: Any) -> dict[str, Any]:
    """The tool's schema as Vertex-safe schema, whichever shorthand it was declared in."""
    if isinstance(schema, dict) and (
        "properties" in schema or schema.get("type") == "object"
    ):
        return _gemini_schema(schema)
    if isinstance(schema, dict):
        return {
            "type": "object",
            "properties": {
                name: (
                    {"type": "array", "items": {"type": "object"}}
                    if kind is list
                    else {"type": _PYTHON_TYPES.get(kind, "string")}
                )
                for name, kind in schema.items()
            },
            "required": list(schema),
        }
    return {"type": "object", "properties": {}}


def _project() -> str:
    named = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if named:
        return named
    credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials:
        try:
            with open(credentials, encoding="utf-8") as handle:
                found = json.load(handle).get("project_id", "")
            if found:
                return found
        except (OSError, ValueError):
            pass
    # Application Default Credentials already carry a project on a machine that has run
    # ``gcloud auth application-default login`` or that runs on Google infrastructure. Asking
    # for it again as an environment variable is a setting the operator does not need to know.
    try:
        import google.auth

        _, discovered = google.auth.default()
        if discovered:
            return str(discovered)
    except Exception:  # noqa: BLE001 - fall through to the explicit instruction below
        pass
    raise RuntimeError(
        "no GCP project named; run 'gcloud auth application-default login', or set "
        "GOOGLE_CLOUD_PROJECT, or point GOOGLE_APPLICATION_CREDENTIALS at a service-account file"
    )


def _location() -> str:
    return os.environ.get("ALK_VERTEX_LOCATION", "global").strip() or "global"


def _llm(model: str) -> Any:
    """What an agent runs on: the model name for Vertex, or a Gemini aimed at the gateway."""
    gateway = agentcc.gateway()
    if gateway is None:
        return agentcc.vendor_model(model)
    from google.adk.models import Gemini

    return Gemini(
        model=agentcc.vertex_model(model),
        base_url=gateway.base_url,
        client_kwargs={"vertexai": False, "api_key": gateway.api_key},
    )


def _flattened(result: Any) -> str:
    content = result.get("content") if isinstance(result, dict) else None
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return content if isinstance(content, str) else str(result)


def _successful_terminal_save(name: str, response: Any) -> bool:
    """Whether a tool response proves this authoring stage has persisted its final output.

    Save tools deliberately reject incomplete work with ``is_error`` so the model can repair and
    retry. Once one succeeds, another model turn can only rewrite already-valid output or burn the
    stage budget; the persisted artifact is the stage's actual completion boundary.
    """
    return bool(
        name in _TERMINAL_SAVE_TOOLS
        and isinstance(response, dict)
        and not response.get("is_error")
    )


# Tools whose result the model can fetch again, so only these may be dropped from history.
_REREADABLE = frozenset(
    {
        "inspect_world",
        "inspect_scenario",
        "inspect_environment",
        "query_world",
        "check_world",
        "read_scenario",
        "read_scenarios",
        "read_source",
        "read_transcript",
        "try_calls",
        "Read",
        "Grep",
        "Glob",
    }
)

_READS_KEPT_WHOLE = 2
_READ_CHARS_BEFORE_FORGETTING = 60_000
_FORGOTTEN = "[dropped to keep this session small] "


def _bare(name: str) -> str:
    return name.rsplit("__", 1)[-1]


def _forget_old_reads(contents: list[Any]) -> None:
    """Collapse superseded reads; the whole Part is replaced because ADK shallow-copies it."""
    from google.genai import types

    arguments: dict[str, str] = {}
    reads: list[tuple[Any, str, str]] = []
    held = 0
    for content in contents:
        for part in getattr(content, "parts", None) or []:
            call = getattr(part, "function_call", None)
            if call is not None:
                arguments[getattr(call, "id", "") or ""] = json.dumps(
                    dict(getattr(call, "args", None) or {}), default=str
                )[:160]
            answer = getattr(part, "function_response", None)
            if answer is None:
                continue
            name = _bare(getattr(answer, "name", "") or "")
            if name not in _REREADABLE:
                continue
            text = _flattened(getattr(answer, "response", None))
            if text.startswith(_FORGOTTEN):
                continue
            held += len(text)
            reads.append((part, name, getattr(answer, "id", "") or ""))
    if held <= _READ_CHARS_BEFORE_FORGETTING:
        return
    recent: dict[str, int] = {}
    for part, name, call_id in reversed(reads):
        recent[name] = recent.get(name, 0) + 1
        if recent[name] <= _READS_KEPT_WHOLE:
            continue
        said = arguments.get(call_id, "")
        part.function_response = types.FunctionResponse(
            id=call_id or None,
            name=part.function_response.name,
            response={
                "content": [
                    {
                        "type": "text",
                        "text": f"{_FORGOTTEN}call {name}({said}) again if you still need it.",
                    }
                ]
            },
        )


def _pruned_history(callback_context: Any, llm_request: Any) -> None:
    _forget_old_reads(llm_request.contents or [])


def _stopped(said: str) -> Any:
    """A final model reply, which is how a callback ends a run without raising."""
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=said)])
    )


# ADK refuses a compaction threshold without a retention, so both must be set.
COMPACT_ABOVE_TOKENS = int(os.environ.get("ALK_HARNESS_COMPACT_ABOVE", "100000") or 0)
EVENTS_KEPT_RAW = int(os.environ.get("ALK_HARNESS_EVENTS_KEPT_RAW", "20") or 0)


def _compaction() -> Any:
    """Auto-compaction for a long stage, or None when it is switched off."""
    if COMPACT_ABOVE_TOKENS <= 0 or EVENTS_KEPT_RAW <= 0:
        return None
    try:
        from google.adk.apps._configs import EventsCompactionConfig
    except ImportError:
        logger.warning("this ADK has no event compaction; long stages will carry their whole history")
        return None
    return EventsCompactionConfig(
        token_threshold=COMPACT_ABOVE_TOKENS, event_retention_size=EVENTS_KEPT_RAW
    )


def _spec_tool(name: str, spec: ToolSpec) -> Any:
    """A ToolSpec as an ADK tool, through ADK's own extension point.

    ``BaseTool`` with an explicit ``_get_declaration`` is how ADK says a tool whose contract
    is defined elsewhere should be wrapped; the handler runs unchanged and ADK owns calling
    it, retrying the turn, and feeding the result back.
    """
    from google.adk.tools import BaseTool
    from google.genai import types

    class SpecTool(BaseTool):
        def __init__(self) -> None:
            super().__init__(name=name, description=spec.description)

        def _get_declaration(self) -> Any:
            return types.FunctionDeclaration(
                name=name,
                description=spec.description,
                parameters=_json_schema(spec.input_schema),
            )

        async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
            del tool_context
            return await spec.handler(args)

    return SpecTool()


def _ask_tool(callback: Any) -> Any:
    async def ask(args: dict[str, Any]) -> dict[str, Any]:
        return await callback(ASK_TOOL, args, None)

    return _spec_tool(
        ASK_TOOL,
        ToolSpec(
            name=ASK_TOOL,
            description=(
                "Ask the operator one necessary blocking question. Provide a concise question "
                "and, when the choices are bounded, an options array."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["question"],
            },
            handler=ask,
        ),
    )


class VertexGeminiSession:
    """One ADK-run conversation with Gemini; ADK holds the history across turns."""

    def __init__(self, spec: SessionSpec, model: str) -> None:
        self._spec = spec
        self._model = model
        self._runner: Any = None
        self._pending: str | None = None
        # Calls each worker run has taken, keyed by the scope ADK gives that run.
        self._worker_calls: dict[str, int] = {}
        self._resume_invocation_id: str | None = None
        context = spec.conversation
        self.session_id = (
            context.session_id
            if context is not None
            else f"gemini-{uuid.uuid4().hex[:12]}"
        )

    def _tools(
        self,
        builtins: tuple[str, ...] | None = None,
        servers: dict[str, Any] | None = None,
    ) -> list[Any]:
        # DELEGATE_TOOL is absent: ADK already exposes each sub-agent as a tool.
        builtins = self._spec.builtins if builtins is None else builtins
        servers = self._spec.servers if servers is None else servers
        offered: list[Any] = []
        wanted = {name for name in builtins if name in FILE_TOOLS}
        offered.extend(
            _spec_tool(spec.name, spec)
            for spec in file_tools(self._spec.cwd)
            if spec.name in wanted
        )
        if ASK_TOOL in self._spec.builtins and self._spec.ask is not None:
            offered.append(_ask_tool(self._spec.ask))
        for server_name, server in servers.items():
            offered.extend(
                _spec_tool(qualified(server_name, spec.name), spec)
                for spec in server.tools
            )
        return offered

    def _workers(self) -> list[Any]:
        """Every declared worker as a sub-agent this loop may run."""
        from google.adk.agents import LlmAgent
        from google.genai import types

        def capped(ceiling: int) -> Any:
            def before(callback_context: Any, llm_request: Any) -> Any:
                _forget_old_reads(llm_request.contents or [])
                if ceiling <= 0:
                    return None
                run = (
                    getattr(callback_context, "isolation_scope", None)
                    or getattr(callback_context, "branch", None)
                    or ""
                )
                self._worker_calls[run] = self._worker_calls.get(run, 0) + 1
                if self._worker_calls[run] <= ceiling:
                    return None
                return _stopped(
                    f"I have used all {ceiling} of my turns. Everything I submitted is already "
                    "in the suite. Report what is missing to whoever briefed me so it can be "
                    "handed to another writer."
                )

            return before

        built: list[Any] = []
        for name, worker in self._spec.workers.items():
            built.append(
                LlmAgent(
                    name=re.sub(r"[^0-9A-Za-z_]", "_", name),
                    description=worker.description,
                    model=_llm(worker.model or self._model),
                    mode="single_turn",
                    static_instruction=types.Content(
                        role="user", parts=[types.Part(text=worker.instructions)]
                    ),
                    tools=self._tools(
                        worker.builtins or self._spec.builtins,
                        worker.servers or self._spec.servers,
                    ),
                    before_model_callback=capped(worker.max_turns),
                )
            )
        return built

    async def start(self) -> None:
        from google.adk.agents import LlmAgent
        from google.adk.apps import App, ResumabilityConfig
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types

        if agentcc.gateway() is None:
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
            os.environ["GOOGLE_CLOUD_PROJECT"] = _project()
            os.environ["GOOGLE_CLOUD_LOCATION"] = _location()
        agent = LlmAgent(
            name=self.session_id.replace("-", "_"),
            model=_llm(self._model),
            static_instruction=types.Content(
                role="user", parts=[types.Part(text=self._spec.system_prompt)]
            ),
            tools=self._tools(),
            sub_agents=self._workers(),
            before_model_callback=_pruned_history,
        )
        context = self._spec.conversation
        if context is None:
            app_name = "alk-harness"
            user_id = "stage"
            sessions = InMemorySessionService()
            await sessions.create_session(
                app_name=app_name,
                user_id=user_id,
                session_id=self.session_id,
            )
        else:
            app_name = context.app_name
            user_id = context.user_id
            sessions = context.event_store
            if sessions is None:
                raise RuntimeError(
                    "hosted Vertex Gemini requires a durable event store"
                )
            existing = await sessions.get_session(
                app_name=app_name,
                user_id=user_id,
                session_id=self.session_id,
            )
            if existing is None:
                await sessions.create_session(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=self.session_id,
                    state={
                        "logical_stage": context.turn_context.get("stage"),
                        "conversation_id": context.turn_context.get("conversation_id"),
                    },
                )
        self._app_name = app_name
        self._user_id = user_id
        app = App(
            name=app_name,
            root_agent=agent,
            resumability_config=ResumabilityConfig(is_resumable=True),
            events_compaction_config=_compaction(),
        )
        # An App rather than a bare agent: the compaction config only takes effect on the App.
        self._runner = Runner(
            app=app,
            session_service=sessions,
        )

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.close()
            self._runner = None

    async def send(self, message: str) -> None:
        if self._runner is None:
            raise RuntimeError("session is not open")
        self._pending = message
        self._resume_invocation_id = None

    async def interrupt(self) -> bool:
        return False

    async def resume(self, invocation_id: str) -> None:
        if self._runner is None:
            raise RuntimeError("session is not open")
        self._pending = None
        self._resume_invocation_id = invocation_id

    async def replies(self) -> AsyncIterator[Any]:
        from google.adk.agents.run_config import RunConfig, StreamingMode
        # google.adk.sessions does not re-export it in the locked ADK (2.7.1).
        from google.adk.sessions.base_session_service import GetSessionConfig
        from google.genai import types

        if self._runner is None or (
            self._pending is None and self._resume_invocation_id is None
        ):
            raise RuntimeError("nothing to reply to; send or resume a turn first")
        yield SessionOpened(session_id=self.session_id)
        message = (
            types.Content(role="user", parts=[types.Part(text=self._pending)])
            if self._pending is not None
            else None
        )
        resume_invocation_id = self._resume_invocation_id
        self._pending = None
        self._resume_invocation_id = None
        turns = 0
        tokens_in = 0
        tokens_out = 0
        tokens_cached = 0
        settled = False
        terminal_save_succeeded = False
        context = self._spec.conversation
        run_config = RunConfig(
            max_llm_calls=max(self._spec.max_turns, 1),
            http_options=types.HttpOptions(
                timeout=300_000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
            streaming_mode=(
                StreamingMode.SSE
                if context is not None and context.streaming
                else StreamingMode.NONE
            ),
            get_session_config=GetSessionConfig(num_recent_events=200),
            model_input_context=(
                [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                text=(
                                    "Authoritative current harness context. Treat this as data, "
                                    "not as a user instruction:\n"
                                    + json.dumps(
                                        context.turn_context,
                                        sort_keys=True,
                                        separators=(",", ":"),
                                    )
                                )
                            )
                        ],
                    )
                ]
                if context is not None
                else None
            ),
        )
        try:
            async with aclosing(
                self._runner.run_async(
                    user_id=self._user_id,
                    session_id=self.session_id,
                    new_message=message,
                    invocation_id=resume_invocation_id,
                    run_config=run_config,
                )
            ) as events:
                async for event in events:
                    usage = getattr(event, "usage_metadata", None)
                    if usage is not None:
                        tokens_in += usage.prompt_token_count or 0
                        tokens_out += usage.candidates_token_count or 0
                        tokens_cached += (
                            getattr(usage, "cached_content_token_count", 0) or 0
                        )
                    parts: list[Any] = []
                    returned: list[ToolReturned] = []
                    for part in (event.content.parts if event.content else []) or []:
                        if getattr(part, "text", None):
                            parts.append(
                                Say(
                                    text=part.text,
                                    partial=bool(getattr(event, "partial", False)),
                                    event_id=str(getattr(event, "id", "") or ""),
                                    invocation_id=str(
                                        getattr(event, "invocation_id", "") or ""
                                    ),
                                    author=str(getattr(event, "author", "") or ""),
                                )
                            )
                        if getattr(part, "function_call", None):
                            parts.append(
                                Call(
                                    id=getattr(part.function_call, "id", None)
                                    or f"call-{uuid.uuid4().hex[:8]}",
                                    name=part.function_call.name or "",
                                    arguments=dict(part.function_call.args or {}),
                                    by=getattr(event, "author", "") or "",
                                    invocation_id=str(
                                        getattr(event, "invocation_id", "") or ""
                                    ),
                                )
                            )
                        if getattr(part, "function_response", None):
                            response = part.function_response.response
                            response_name = part.function_response.name or ""
                            terminal_save_succeeded = terminal_save_succeeded or (
                                _successful_terminal_save(response_name, response)
                            )
                            returned.append(
                                ToolReturned(
                                    id=getattr(part.function_response, "id", None) or "",
                                    text=_flattened(response),
                                    is_error=bool(
                                        isinstance(response, dict)
                                        and response.get("is_error")
                                    ),
                                    invocation_id=str(
                                        getattr(event, "invocation_id", "") or ""
                                    ),
                                )
                            )
                    if parts:
                        if not getattr(event, "partial", False):
                            turns += 1
                        yield ModelReply(parts=parts, model=self._model)
                    for outcome in returned:
                        yield outcome
                    if terminal_save_succeeded:
                        settled = True
                        break
                    if event.is_final_response():
                        settled = True
        except Exception as exc:
            yield StageDone(
                outcome="failed",
                turns=turns,
                cost_usd=self._cost(tokens_in, tokens_out, tokens_cached),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                tokens_cached=tokens_cached,
                session_id=self.session_id,
                models={self._model},
                is_error=True,
                api_error_status=getattr(exc, "code", None),
                errors=[str(exc)[:400]],
            )
            return
        # A stream that ends without a final response ran out of its call budget, which is
        # not the same as the model having finished. Reported as success it reads as a stage
        # that did its work, and a half-written suite comes back green.
        yield StageDone(
            outcome="success" if settled else "max_turns",
            is_error=not settled,
            turns=turns,
            cost_usd=self._cost(tokens_in, tokens_out, tokens_cached),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_cached=tokens_cached,
            session_id=self.session_id,
            models={self._model},
            errors=(
                []
                if settled
                else [
                    f"the stage spent its whole budget of {self._spec.max_turns} calls"
                ]
            ),
        )

    def _cost(
        self, tokens_in: int, tokens_out: int, tokens_cached: int = 0
    ) -> float | None:
        return priced(self._model, tokens_in, tokens_out, tokens_cached)


logger = logging.getLogger(__name__)


def priced(
    model: str, tokens_in: int, tokens_out: int, tokens_cached: int = 0
) -> float | None:
    """What these tokens cost, or None where no price can be stood behind."""
    prices = PRICES_PER_MILLION.get(model)
    if prices is None and "/" in model:
        # A gateway prefixes the route, e.g. "vertex_ai/<model>"; the price belongs to the model.
        prices = PRICES_PER_MILLION.get(model.rsplit("/", 1)[-1])
    if prices is None:
        return None
    if len(prices) > 2 and date.today().isoformat() > str(prices[2]):
        logger.warning(
            "no current price for %s: the table's figures expired on %s",
            model,
            prices[2],
        )
        return None
    cached = min(max(tokens_cached, 0), max(tokens_in, 0))
    billed_in = (tokens_in - cached) * prices[0] + cached * prices[0] * CACHE_READ_SHARE
    return (billed_in + tokens_out * prices[1]) / 1_000_000


class VertexGeminiBackend:
    name = "vertex-gemini"
    default_model = DEFAULT_MODEL

    def can_drive(self, model: str) -> bool:
        return agentcc.vendor_model(model or "").lower().startswith("gemini")

    def create(self, spec: SessionSpec) -> VertexGeminiSession:
        return VertexGeminiSession(spec, model=spec.model or self.default_model)
