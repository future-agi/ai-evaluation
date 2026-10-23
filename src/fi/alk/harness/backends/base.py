"""What a harness backend is, said without naming any vendor.

A stage of this harness is a conversation: a system prompt, a set of tools the model may call,
a turn budget, and a loop that feeds tool results back until the model stops. Today that loop is
Claude Code's; tomorrow it may be Gemini's, Bedrock's, or a harness of our own. The stages do
not care, so nothing they say may mention a vendor.

This module is that neutrality, in four pieces:

- ``ToolSpec`` / ``ToolServer``: a tool as the harness defines one, with the async handler that
  executes it. Backends adapt these to whatever their loop natively speaks.
- ``SessionSpec``: everything a stage asks of a session. This is the real contract the ten
  construction sites were already expressing through a vendor options class.
- The reply vocabulary (``SessionOpened``, ``ModelReply``, ``ToolReturned``, ``StageDone``):
  what a running session emits, which ``Stage`` renders into events. A backend translates its
  provider's stream into these and nothing else leaks through.
- ``HarnessBackend`` / ``HarnessSession``: the two protocols a new backend implements. A backend
  with its own loop supplies its own session type; the harness never sees inside it.

Nothing here imports a provider SDK, so a deployment that uses one backend does not need the
other's dependencies installed.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

_PYTHON_SCHEMA_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def tool_input_schema(schema: Any) -> dict[str, Any]:
    """Return one provider-safe JSON Schema with local references inlined."""
    if isinstance(schema, dict) and any(
        key in schema
        for key in ("type", "properties", "$ref", "$defs", "anyOf", "oneOf")
    ):
        root = schema
    elif isinstance(schema, dict):
        root = {
            "type": "object",
            "properties": {
                name: (
                    {"type": "array", "items": {"type": "object"}}
                    if kind is list
                    else {"type": _PYTHON_SCHEMA_TYPES.get(kind, "string")}
                )
                for name, kind in schema.items()
            },
            "required": list(schema),
        }
    else:
        root = {"type": "object", "properties": {}}

    def target(ref: str) -> Any:
        if not ref.startswith("#/"):
            raise ValueError("tool_schema_external_reference_unsupported")
        value: Any = root
        for raw in ref[2:].split("/"):
            key = raw.replace("~1", "/").replace("~0", "~")
            if not isinstance(value, dict) or key not in value:
                raise ValueError("tool_schema_reference_missing")
            value = value[key]
        return value

    def normalize(value: Any, resolving: tuple[str, ...] = ()) -> Any:
        if isinstance(value, list):
            return [normalize(item, resolving) for item in value]
        if not isinstance(value, dict):
            return value
        if "$ref" in value:
            ref = value["$ref"]
            if not isinstance(ref, str):
                raise ValueError("tool_schema_reference_invalid")
            if ref in resolving:
                raise ValueError("tool_schema_recursive_reference_unsupported")
            resolved = normalize(target(ref), (*resolving, ref))
            if not isinstance(resolved, dict):
                raise ValueError("tool_schema_reference_invalid")
            siblings = {
                key: normalize(item, resolving)
                for key, item in value.items()
                if key != "$ref"
            }
            return {**resolved, **siblings}
        return {
            key: normalize(item, resolving)
            for key, item in value.items()
            if key not in {"$defs", "definitions"}
        }

    return normalize(root)


def qualified(server: str, tool_name: str) -> str:
    """The fully qualified name a session grants and a model calls.

    The ``mcp__{server}__{tool}`` convention comes from the first backend, but the skills and
    gates all speak it, so every backend keeps it. Renaming per backend would mean rewriting
    every prompt that names a tool.
    """
    return f"mcp__{server}__{tool_name}"


@dataclass
class ToolSpec:
    """One tool: its contract for the model, and the code that executes it.

    ``input_schema`` is either a JSON Schema dict or the shorthand ``{"arg": str}`` mapping the
    tool decorator accepts. ``handler`` receives the arguments dict and returns
    ``{"content": [{"type": "text", "text": ...}], "is_error"?: bool}``, the shape every
    existing tool already returns.
    """

    name: str
    description: str
    input_schema: Any
    handler: ToolHandler


@dataclass
class ToolServer:
    """A named group of tools granted to a session together."""

    name: str
    version: str = "0.1.0"
    tools: list[ToolSpec] = field(default_factory=list)


def tool(
    name: str, description: str, input_schema: Any
) -> Callable[[ToolHandler], ToolSpec]:
    """Declare a tool. Same signature the stages have always used, no vendor behind it."""

    def decorator(handler: ToolHandler) -> ToolSpec:
        return ToolSpec(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
        )

    return decorator


def tool_server(
    name: str, version: str = "0.1.0", tools: list[ToolSpec] | None = None
) -> ToolServer:
    """Group tools under a server name, as the stages have always done."""
    return ToolServer(name=name, version=version, tools=list(tools or []))


# Host tools a backend may be asked to supply itself. Claude Code ships these; a backend without
# a host CLI implements them from files.py. Anything else asked for as a builtin is refused at
# session build time rather than silently dropped.
FILE_TOOLS = ("Read", "Glob", "Grep")
ASK_TOOL = "AskUserQuestion"
# Backend-neutral name for running part of a stage in a sub-session.
DELEGATE_TOOL = "Delegate"
# A shell and file edits, for the one session that operates a whole environment the way a
# developer would. Never granted to an authoring stage, whose writes go through checked tools.
OPERATOR_TOOLS = ("Bash", "Write", "Edit")
KNOWN_BUILTINS = (*FILE_TOOLS, ASK_TOOL, DELEGATE_TOOL, *OPERATOR_TOOLS)

# Ceiling on workers in flight at once.
MOST_WORKERS_AT_ONCE = int(os.environ.get("ALK_HARNESS_WORKERS_AT_ONCE", "48") or 48)

TERMINAL_SAVE_TOOLS = frozenset(
    {
        "mcp__world__save_world",
        "mcp__provision__save_environment",
        "mcp__scenarios__save_scenarios",
        "mcp__source_data__finish_review",
        "mcp__repair__submit_world_ir_patch",
        "mcp__runtime_repair__submit_runtime_plan_patch",
    }
)


@dataclass
class WorkerSpec:
    """A worker the model may run to do part of its stage, in its own session."""

    description: str
    instructions: str
    servers: dict[str, ToolServer] = field(default_factory=dict)
    builtins: tuple[str, ...] = ()
    max_turns: int = 40
    # Empty inherits the parent's model.
    model: str = ""

    def granted(self, parent: "SessionSpec") -> list[str]:
        """Every tool name this worker may call, falling back to the parent's."""
        servers = self.servers or parent.servers
        names = [*(self.builtins or parent.builtins)]
        for server_name, server in servers.items():
            names.extend(qualified(server_name, spec.name) for spec in server.tools)
        return names


@dataclass
class ConversationSession:
    """Durable provider identity and fresh context for a hosted stage session."""

    app_name: str
    user_id: str
    session_id: str
    event_store: Any = None
    transcript_store: Any = None
    resume_session_id: str | None = None
    config_dir: str | None = None
    turn_context: dict[str, Any] = field(default_factory=dict)
    streaming: bool = True


@dataclass
class SessionSpec:
    """Everything a stage asks of a session, with no vendor vocabulary in it.

    ``builtins`` are host tools by bare name (``Read``, ``Glob``, ``Grep``,
    ``AskUserQuestion``); ``servers`` are the harness's own tools. ``ask`` is the operator
    callback consulted when the model asks a question; None means the run is unattended.
    ``gated`` selects the deny-by-default permission regime every tool-bearing stage runs
    under; the one stage that runs bare (the simulated customer, which has no tools) turns it
    off to keep its behaviour byte-identical.
    ``thinking`` opts into the harness's thinking policy (config.thinking_config); stages that
    never set one keep their backend's default.
    """

    system_prompt: str
    servers: dict[str, ToolServer] = field(default_factory=dict)
    builtins: tuple[str, ...] = ()
    cwd: str | None = None
    max_turns: int = 40
    model: str = ""
    ask: Any = None
    gated: bool = True
    thinking: bool = False
    # A ready-made permission callable that replaces the backend's own gate wholesale. One
    # stage (understand, interactive) passes its gate in fully built; backends without a
    # permission callback concept ignore it, which is safe because their gating is structural.
    permission_override: Any = None
    # How long this stage may go without emitting anything before it is treated as hung. Zero
    # takes the harness default. A stage whose work happens inside one long tool call needs its
    # own bound: it is working the whole time and has nothing to say while it does, so the
    # default reads honest work as a hang and kills it.
    idle_timeout_seconds: float = 0.0
    # Workers this session may run, by name.
    workers: dict[str, WorkerSpec] = field(default_factory=dict)
    conversation: ConversationSession | None = None

    def granted(self) -> list[str]:
        """Every tool name this session may call, qualified the way the model calls it."""
        names = [*self.builtins]
        for server_name, server in self.servers.items():
            names.extend(qualified(server_name, spec.name) for spec in server.tools)
        return names

    def granted_anywhere(self) -> list[str]:
        """Every tool name this session or any of its workers may call."""
        names = list(self.granted())
        for worker in self.workers.values():
            names.extend(worker.granted(self))
        return list(dict.fromkeys(names))

    def grant(self, server_name: str, server: ToolServer) -> None:
        """Add a tool server before the session opens."""
        self.servers[server_name] = server


# -- what a running session emits --------------------------------------------------------------


@dataclass
class SessionOpened:
    """The session exists and has an identity, if the backend assigns one."""

    session_id: str | None = None


@dataclass
class Say:
    """The model said something."""

    text: str
    partial: bool = False
    event_id: str = ""
    invocation_id: str = ""
    author: str = ""


@dataclass
class Call:
    """The model called a tool."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    # Which agent made the call; empty when the backend does not distinguish one.
    by: str = ""
    invocation_id: str = ""


@dataclass
class ModelReply:
    """One assistant message: text and tool calls, in the order they were produced."""

    parts: list[Any] = field(default_factory=list)
    model: str = ""


@dataclass
class ToolReturned:
    """What a tool call produced, flattened to text."""

    id: str
    text: str
    is_error: bool = False
    invocation_id: str = ""


@dataclass
class StageDone:
    """The exchange is over. The raw facts; Stage turns them into words."""

    outcome: str = "success"
    turns: int = 0
    cost_usd: float | None = None
    # The units behind the price, so a bill can be checked rather than trusted.
    tokens_in: int = 0
    tokens_out: int = 0
    # Input the provider served from its own cache. Included in tokens_in, and billed well below
    # fresh input, so a ledger that does not carry it overstates a rerun. Carried rather than
    # discounted here: inventing a cache rate would be a guess presented as a price.
    tokens_cached: int = 0
    session_id: str | None = None
    models: set[str] = field(default_factory=set)
    is_error: bool = False
    api_error_status: Any = None
    errors: list[Any] = field(default_factory=list)


# -- what a backend implements -----------------------------------------------------------------


@runtime_checkable
class HarnessSession(Protocol):
    """One open session. Backends with their own loop implement this around it."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def send(self, message: str) -> None: ...

    async def interrupt(self) -> bool: ...
    async def resume(self, invocation_id: str) -> None: ...

    def replies(self) -> AsyncIterator[Any]:
        """Everything the session emits for the message just sent, ending with StageDone."""
        ...


@runtime_checkable
class HarnessBackend(Protocol):
    """A way of running stages. Selected by name through the registry."""

    name: str
    default_model: str

    def create(self, spec: SessionSpec) -> HarnessSession: ...

    def can_drive(self, model: str) -> bool:
        """Whether this backend can actually run the named model.

        A backend handed a model it cannot reach must refuse loudly here. Left unchecked it
        produces a session that answers nothing, which downstream reads as an agent that ignored
        the person rather than as a configuration mistake.
        """
        ...
