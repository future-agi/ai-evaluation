"""A harness backend that runs stages on Gemini through Vertex AI, on Google's ADK.

The Agent Development Kit is Google's counterpart to the Claude Agent SDK: it owns the agentic
loop, executes tools, and holds session history, the way this harness expects a backend to. We
adapt at the same seam as the Claude backend and nothing more: a ``ToolSpec`` becomes an ADK
tool through ADK's own extension point (``BaseTool`` with an explicit declaration), and ADK's
event stream is translated into the neutral reply vocabulary. The loop itself is not ours.

The Gemini 3.x models this exists for are served from the ``global`` endpoint only, which is
why ``ALK_VERTEX_LOCATION`` defaults to ``global`` rather than to a region. Regional Vertex
deployments of older models can point it elsewhere.

Read, Glob and Grep come from files.py when a stage grants them. AskUserQuestion is not
implemented here yet: unattended runs never use it, and an attended run on this backend simply
proceeds without the option, which is said out loud in the session rather than hidden.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, AsyncIterator

from .base import (
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

# Vertex list pricing per 1M tokens (input, output), as of 2026-08; verify before relying on
# cost figures. An unknown model reports no cost rather than a wrong one.
PRICES_PER_MILLION = {
    "gemini-3.7-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
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


def _flattened(result: Any) -> str:
    content = result.get("content") if isinstance(result, dict) else None
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return content if isinstance(content, str) else str(result)


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
            return await spec.handler(args)

    return SpecTool()


class VertexGeminiSession:
    """One ADK-run conversation with Gemini; ADK holds the history across turns."""

    def __init__(self, spec: SessionSpec, model: str) -> None:
        self._spec = spec
        self._model = model
        self._runner: Any = None
        self._pending: str | None = None
        self.session_id = f"gemini-{uuid.uuid4().hex[:12]}"

    def _tools(self) -> list[Any]:
        # ASK_TOOL is deliberately absent: unattended runs never call it, and declaring a tool
        # this backend cannot answer would cost the model a turn finding that out.
        offered: list[Any] = []
        wanted = {name for name in self._spec.builtins if name in FILE_TOOLS}
        offered.extend(
            _spec_tool(spec.name, spec)
            for spec in file_tools(self._spec.cwd)
            if spec.name in wanted
        )
        for server_name, server in self._spec.servers.items():
            offered.extend(
                _spec_tool(qualified(server_name, spec.name), spec)
                for spec in server.tools
            )
        return offered

    async def start(self) -> None:
        from google.adk.agents import LlmAgent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types

        # ADK builds its Vertex client from the environment, the same way the Claude backend
        # passes provider env through its options.
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
        os.environ["GOOGLE_CLOUD_PROJECT"] = _project()
        os.environ["GOOGLE_CLOUD_LOCATION"] = _location()
        # static_instruction, not instruction: the skills are full of literal JSON braces,
        # and ADK templates {placeholders} in `instruction` from session state. Static
        # content is sent verbatim and is what ADK context-caches.
        agent = LlmAgent(
            name=self.session_id.replace("-", "_"),
            model=self._model,
            static_instruction=types.Content(
                role="user", parts=[types.Part(text=self._spec.system_prompt)]
            ),
            tools=self._tools(),
        )
        sessions = InMemorySessionService()
        await sessions.create_session(
            app_name="alk-harness", user_id="stage", session_id=self.session_id
        )
        self._runner = Runner(
            agent=agent, app_name="alk-harness", session_service=sessions
        )

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.close()
            self._runner = None

    async def send(self, message: str) -> None:
        if self._runner is None:
            raise RuntimeError("session is not open")
        self._pending = message

    async def replies(self) -> AsyncIterator[Any]:
        from google.adk.agents.run_config import RunConfig
        from google.genai import types

        if self._runner is None or self._pending is None:
            raise RuntimeError("nothing to reply to; send a message first")
        yield SessionOpened(session_id=self.session_id)
        message = types.Content(
            role="user", parts=[types.Part(text=self._pending)]
        )
        self._pending = None
        turns = 0
        tokens_in = 0
        tokens_out = 0
        settled = False
        try:
            async for event in self._runner.run_async(
                user_id="stage",
                session_id=self.session_id,
                new_message=message,
                run_config=RunConfig(max_llm_calls=max(self._spec.max_turns, 1)),
            ):
                usage = getattr(event, "usage_metadata", None)
                if usage is not None:
                    tokens_in += usage.prompt_token_count or 0
                    tokens_out += usage.candidates_token_count or 0
                parts: list[Any] = []
                returned: list[ToolReturned] = []
                for part in (event.content.parts if event.content else []) or []:
                    if getattr(part, "text", None):
                        parts.append(Say(text=part.text))
                    if getattr(part, "function_call", None):
                        parts.append(
                            Call(
                                id=getattr(part.function_call, "id", None)
                                or f"call-{uuid.uuid4().hex[:8]}",
                                name=part.function_call.name or "",
                                arguments=dict(part.function_call.args or {}),
                            )
                        )
                    if getattr(part, "function_response", None):
                        response = part.function_response.response
                        returned.append(
                            ToolReturned(
                                id=getattr(part.function_response, "id", None) or "",
                                text=_flattened(response),
                                is_error=bool(
                                    isinstance(response, dict)
                                    and response.get("is_error")
                                ),
                            )
                        )
                if parts:
                    turns += 1
                    yield ModelReply(parts=parts, model=self._model)
                for outcome in returned:
                    yield outcome
                if event.is_final_response():
                    settled = True
        except Exception as exc:
            yield StageDone(
                outcome="failed",
                turns=turns,
                cost_usd=self._cost(tokens_in, tokens_out),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
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
            cost_usd=self._cost(tokens_in, tokens_out),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            session_id=self.session_id,
            models={self._model},
            errors=(
                []
                if settled
                else [f"the stage spent its whole budget of {self._spec.max_turns} calls"]
            ),
        )

    def _cost(self, tokens_in: int, tokens_out: int) -> float | None:
        prices = PRICES_PER_MILLION.get(self._model)
        if prices is None:
            return None
        return (tokens_in * prices[0] + tokens_out * prices[1]) / 1_000_000


class VertexGeminiBackend:
    name = "vertex-gemini"
    default_model = DEFAULT_MODEL

    def can_drive(self, model: str) -> bool:
        return (model or "").lower().startswith("gemini")

    def create(self, spec: SessionSpec) -> VertexGeminiSession:
        return VertexGeminiSession(spec, model=spec.model or self.default_model)
