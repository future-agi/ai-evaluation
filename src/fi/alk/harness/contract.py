"""The agent contract: what the agent verifiably is, read from its own source.

Everything downstream is confined to this. A world may only implement tools listed here, a
scenario may only reference values grounded in here, and a checkpoint may only assert against
what is here. It is the anti-hallucination device for every later stage.

The harness produces it by reading the agent's code and calling ``submit_contract``. Validation
runs inside that tool, so problems are returned into the conversation and the model tries again
rather than a bad contract reaching disk.
"""

from __future__ import annotations

import json
import shlex
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# How a person reaches an agent. This decides how it is later run — voice goes out as a live
# call, everything else runs locally — so it is defined once and referenced, never retyped.
MODALITIES = ("voice", "chat", "browser")
# Voice only, and only two: either the agent placed the call or it answered one.
CALL_DIRECTIONS = ("inbound", "outbound")

_STRING_FIELDS = (
    "agent",
    "one_liner",
    "modality",
    "call_direction",
    "system_prompt_excerpt",
    "notes",
)
_LIST_FIELDS = (
    "hard_constraints",
    "real_use_cases",
    "amendments",
    "chosen_evals",
)
_DICT_FIELDS = ("data_schema", "base_environment")

# What each field gets called when it is not called what we call it. Every one of these was
# written by a model that had read the schema and still reached for the more obvious word.
_ALIASES = {
    "real_use_cases": ("use_cases", "usecases", "scenarios", "capabilities"),
    "hard_constraints": ("constraints", "rules", "policies", "policy", "guardrails"),
    "system_prompt_excerpt": ("system_prompt", "prompt", "instructions"),
    "base_environment": ("data", "seed_data", "starting_data", "records"),
    "data_schema": ("schema", "record_schema", "data_shape"),
    "agent": ("name", "agent_name"),
    "one_liner": ("summary", "description"),
    "notes": ("observations", "remarks"),
}


class ToolSpec(BaseModel):
    """One tool the agent really has.

    ``args`` is the load-bearing field: the world's handlers, the probes and every scenario are
    built from these exact names. It is also the one most often written under another name —
    ``parameters``, ``arguments``, ``params`` — or left out while ``arg_types`` names every
    argument anyway. All of those are the same information, so they are accepted and normalised
    rather than rejected, because a contract bounced for a synonym costs a full turn and teaches
    nothing about the agent.
    """

    @model_validator(mode="before")
    @classmethod
    def _normalize_args(cls, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        if not payload.get("args"):
            for alias in ("parameters", "arguments", "params", "arg_names"):
                value = payload.get(alias)
                if isinstance(value, list) and value:
                    payload["args"] = value
                    break
                # Some writers give {name: type} where a list was asked for. The keys are the
                # argument names, which is exactly what was wanted.
                if isinstance(value, dict) and value:
                    payload["args"] = list(value)
                    payload.setdefault(
                        "arg_types", {k: str(v) for k, v in value.items()}
                    )
                    break
        if not payload.get("args"):
            # Nothing named the arguments directly, but a per-argument map still names them.
            for source in ("arg_types", "arg_values"):
                mapping = payload.get(source)
                if isinstance(mapping, dict) and mapping:
                    payload["args"] = list(mapping)
                    break
        if isinstance(payload.get("args"), str):
            payload["args"] = [payload["args"]]
        if isinstance(payload.get("args"), list):
            payload["args"] = [str(one) for one in payload["args"]]
        return payload

    name: str
    args: list[str] = Field(default_factory=list)
    arg_types: dict[str, str] = Field(default_factory=dict)
    arg_values: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    # Tools that must have run before this one stops refusing, and only for state the agent builds
    # during the conversation. Empty means callable first thing. Marking a tool gated when it is not
    # costs every future test of it a preamble it never needed.
    requires: list[str] = Field(default_factory=list)


class ToolEntry(BaseModel):
    """How to reach the agent's own implementation of one tool.

    Recorded rather than assumed, because there is no shape every agent shares. A benchmark
    writes static methods on a class; a framework agent writes closures inside ``__init__`` that
    cannot be imported at all. What the environment does about a tool is decided from ``mode``,
    so a tool nobody can reach is visible here rather than quietly reimplemented.
    """

    tool: str
    # import: a module-level callable. construct: a method needing an instance built first.
    # service: reachable over HTTP. unreachable: no runnable seam was found and building stops.
    # The harness never generates agent behavior.
    mode: str = "unreachable"
    module: str = ""
    callable: str = ""
    # An expression that builds the object a `construct` tool hangs off.
    factory: str = ""
    # What the agent's own state is passed as, where a tool takes it as an argument.
    first_arg: str = ""
    # For a service-backed tool, the submitted service and HTTP path its implementation calls.
    # The path is recorded separately from the semantic tool name because production APIs often
    # use a different route name (for example check_status -> /get_status).
    service: str = ""
    endpoint: str = ""
    method: str = "POST"
    notes: str = ""


class DataStore(BaseModel):
    """What the agent's tools read and write, and how to be there instead of it.

    Nothing recorded here is a change to the agent. It is what the agent **already expects**,
    written down so the environment can be built to match: the same host, the same port, the same
    database, the same user. Where it reads a value from configuration we set that configuration;
    where it hardcodes one we shape our own store to it, which is why a hardcoded value is worth
    recording rather than treated as a dead end.

    That inversion is the point. The alternative, editing the agent until it points at us, means
    testing something other than what ships.
    """

    # Read off the agent, never chosen for it. Postgres and ClickHouse disagree about dialect,
    # types and what a transaction even means, so an agent tested against the wrong one is graded
    # on queries it never runs. Free text because the next agent will be on an engine nobody has
    # written down yet.
    kind: str = ""
    version: str = ""

    # The easiest seam, and the one most agents have: one variable or config key holding the whole
    # connection string. Set it at launch and nothing else matters.
    configured_by: str = ""
    config_key: str = ""

    # What the agent expects to find, whether it reads these from config or has them written into
    # its source. A hardcoded host is not an obstacle: a network alias makes that name resolve to
    # our container, and the agent connects to us believing nothing changed.
    host: str = ""
    port: int | None = None
    database: str = ""
    user: str = ""
    # Deliberately never the password itself. A contract is written to disk and read by people, so
    # a secret in it outlives the run that needed it. What is recorded is where the value comes
    # from; if it is genuinely needed it is read at build time and not persisted.
    password_from: str = ""

    # An agent that holds its data in memory is reached by calling the function that loads it, not
    # by connecting to anything. Recorded so the environment can call the agent's own loader
    # rather than reading its files and rebuilding the structure itself, which would be a second
    # implementation of the one thing this path exists to stop reimplementing.
    schema_from: str = ""
    loaded_by: str = ""
    loader_module: str = ""

    def has_seam(self) -> bool:
        """Whether there is any way to point this agent at our store.

        An agent with no seam at all is a finding, not a thing to work around: it cannot be tested
        without one, and saying so is more useful than editing it until it can.
        """
        return bool(
            self.configured_by
            or self.config_key
            or self.host
            or self.port
            or self.database
            or self.loader_module
            or self.loaded_by
        )


class Reached(BaseModel):
    """How an existing agent reaches a dependency, without storing its secret values."""

    dsn_env: str = ""
    config_key: str = ""
    host: str = ""
    port: int | None = None
    database: str = ""
    user: str = ""
    password_from: str = ""
    loader_module: str = ""
    loader_function: str = ""

    def has_seam(self) -> bool:
        return bool(
            self.dsn_env
            or self.config_key
            or self.host
            or self.port
            or self.database
            or self.loader_module
        )


class RuntimeInterface(BaseModel):
    """The submitted runtime's existing conversational ingress.

    This is connection metadata, not generated agent behavior. The harness may publish the
    declared container port and translate its request/response envelope, but it never adds an
    endpoint the repository does not already implement.
    """

    kind: str = ""
    protocol: str = "fi.alk"
    port: int | None = Field(default=None, ge=1, le=65535)
    path: str = ""
    health_path: str = ""
    include_tools: bool = True

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, value: str) -> str:
        normalized = str(value or "").strip().lower().replace("-", "_")
        aliases = {"openai": "http", "openai_compatible": "http"}
        return aliases.get(normalized, normalized)

    @field_validator("protocol")
    @classmethod
    def _known_protocol(cls, value: str) -> str:
        normalized = str(value or "fi.alk").strip().lower().replace("-", "_")
        aliases = {
            "openai": "openai_chat",
            "openai_compatible": "openai_chat",
            "chat_completions": "openai_chat",
            "http": "fi.alk",
        }
        return aliases.get(normalized, normalized)

    @field_validator("path", "health_path")
    @classmethod
    def _absolute_http_path(cls, value: str) -> str:
        path = str(value or "").strip()
        if path and not path.startswith("/"):
            path = "/" + path
        return path

    @model_validator(mode="after")
    def _complete(self) -> "RuntimeInterface":
        if self.kind in {"http", "websocket"}:
            if self.port is None:
                raise ValueError(f"runtime_{self.kind}_interface_requires_port")
            if not self.path:
                raise ValueError(f"runtime_{self.kind}_interface_requires_path")
        if self.kind == "http":
            if self.protocol not in {"fi.alk", "openai_chat"}:
                raise ValueError(
                    "runtime_http_protocol_unsupported: expected fi.alk or openai_chat"
                )
        if self.kind == "websocket" and self.protocol != "fi.alk":
            raise ValueError("runtime_websocket_protocol_unsupported: expected fi.alk")
        return self


class Runtime(BaseModel):
    """What it takes to run the agent's code."""

    # Empty means detect from the submitted dependency manifest. Defaulting this to Python makes
    # an otherwise unambiguous Node repository fail the generated-runtime admission path.
    language: str = ""
    version: str = ""
    install: str = ""
    # Optional dependency groups declared by the repository itself (for example ``voice`` in
    # pyproject.toml). Generated packaging validates these names against the manifest.
    extras: list[str] = Field(default_factory=list)
    workdir: str = ""
    # Select one submitted Compose file when a repository contains multiple runnable stacks.
    compose_file: str = ""
    dockerfile: str = ""
    # For repositories without container metadata, this is an argv vector for the submitted
    # process. It is optional when one conventional entrypoint can be proven from source.
    command: list[str] = Field(default_factory=list)
    # Repository-relative generated-build exclusions selected during understanding. This is for
    # large checked-in outputs or documentation, never for dependency manifests or source code.
    context_excludes: list[str] = Field(default_factory=list)
    # Preserve a repository-declared target architecture (for example linux/amd64 on an ARM
    # runner). This is execution metadata, not a change to the submitted application.
    platform: str = ""

    # How a turn-based simulator reaches the submitted process after it starts. Voice has a
    # standard rendezvous (LiveKit dispatch); chat repositories do not. Recording this seam is
    # what lets the harness start the real runtime instead of reconstructing the agent from its
    # prompt. Empty is valid for voice/browser agents and for contracts that are not backed by a
    # repository.
    interface: RuntimeInterface | None = None

    @field_validator("command", mode="before")
    @classmethod
    def _normalize_command(cls, value: Any) -> Any:
        if isinstance(value, str):
            return shlex.split(value)
        return value

    @field_validator("extras", mode="before")
    @classmethod
    def _normalize_extras(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


class Dependency(BaseModel):
    """Something the agent reaches for that has to exist before it can work.

    This is what tells the environment stage there is a service to stand up, rather than leaving
    it to notice halfway through that a tool has nothing to answer it. The world is a sandbox:
    whatever is named here gets built inside it, so the agent's call goes to something real that
    happens to be ours.
    """

    name: str
    # datastore, service, file, queue — whatever kind of thing this is. Left open rather than
    # enumerated, because the next agent will need a kind nobody has thought of yet.
    kind: str = ""
    what: str = ""
    # The tools that cannot work without it. An unreferenced dependency is usually a mistake.
    used_by: list[str] = Field(default_factory=list)
    engine: str = ""
    version: str = ""
    reached: Reached = Field(default_factory=Reached)

    def provisionable(self) -> bool:
        return bool(self.engine) and self.reached.has_seam()


def _reached(one: Dependency) -> str:
    if not one.engine and not one.reached.has_seam():
        return ""
    said: list[str] = []
    if one.engine:
        said.append(f"stand up {one.engine}{' ' + one.version if one.version else ''}")
    where = one.reached
    if where.loader_module or where.loader_function:
        said.append(
            "call the agent's own "
            f"{where.loader_module or 'MODULE NOT RECORDED'}."
            f"{where.loader_function or 'load_data'} for it; nothing is connected to and no "
            "server is involved"
        )
    elif where.dsn_env:
        said.append(f"point it there with ${where.dsn_env}")
    elif where.config_key:
        said.append(f"point it there with {where.config_key} in its config")
    expected = [
        f"{label} {value}"
        for label, value in (
            ("host", where.host),
            ("port", where.port),
            ("database", where.database),
            ("user", where.user),
        )
        if value
    ]
    if expected:
        said.append("build it to match " + ", ".join(expected))
    if one.engine and not where.has_seam():
        said.append("NO CONFIGURATION SEAM RECORDED")
    return "; ".join(said)


class AgentContract(BaseModel):
    """What the agent verifiably is. Nothing downstream may contradict this."""

    @model_validator(mode="before")
    @classmethod
    def _normalize_shapes(cls, payload: Any) -> Any:
        """Model JSON varies in benign ways: a list where prose was asked, a bare string where a
        list was, a field under the obvious name rather than ours. Normalize instead of
        rejecting, because none of that is a grounding error and rejecting it burns turns on
        something that does not matter."""
        if not isinstance(payload, dict):
            return payload
        # The name we chose is not always the obvious one. `real_use_cases` in particular gets
        # written as `use_cases`, and the answer it then gets — "no-use-cases" — reads as
        # missing rather than misnamed, so the same submission comes back again and again with
        # the shape changed and the name untouched.
        for ours, others in _ALIASES.items():
            if payload.get(ours):
                continue
            for other in others:
                if payload.get(other):
                    payload[ours] = payload[other]
                    break
        for key in _STRING_FIELDS:
            value = payload.get(key)
            if isinstance(value, list):
                payload[key] = "\n".join(str(item) for item in value)
            elif value is not None and not isinstance(value, str):
                payload[key] = str(value)
        for key in _LIST_FIELDS:
            value = payload.get(key)
            if isinstance(value, str):
                payload[key] = [value]
            elif isinstance(value, list):
                payload[key] = [
                    str(item) if not isinstance(item, str) else item for item in value
                ]
        for key in _DICT_FIELDS:
            value = payload.get(key)
            if value is not None and not isinstance(value, dict):
                payload[key] = {"value": value}
        return payload

    # Defaulted rather than mandatory so a submission that forgets it reaches validate_contract,
    # which says what to do about it, instead of dying in the schema layer with a type error.
    agent: str = ""
    one_liner: str = ""
    modality: str = "chat"
    # Voice only: whether this agent places calls or answers them. Chat is always started by the
    # person, so it stays inbound. It changes how the simulated person is briefed, not who speaks
    # first: an outbound agent still greets, it just has to say who it is and why it called.
    call_direction: str = "inbound"
    conversational: bool = True
    system_prompt_excerpt: str = ""
    hard_constraints: list[str] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)
    data_schema: dict[str, Any] = Field(default_factory=dict)
    base_environment: dict[str, Any] = Field(default_factory=dict)
    # What the environment stage has to build before any tool can be answered.
    dependencies: list[Dependency] = Field(default_factory=list)
    # Transport/model connections are configuration, not business services to reconstruct.
    runtime_dependencies: list[Dependency] = Field(default_factory=list)
    # Whether the agent ships code for its tools: present, absent, or partial. Missing code is a
    # build blocker: the harness never supplies replacement agent behavior.
    implementation: str = ""
    tool_entrypoints: list[ToolEntry] = Field(default_factory=list)
    # How this agent's tools say no in a value they return, rather than by raising. Without it a
    # refusal cannot be told from a success once the agent's own code is answering the call.
    refusal_signature: str = ""
    data_store: DataStore | None = None
    runtime: Runtime | None = None
    real_use_cases: list[str] = Field(default_factory=list)
    # Free-form. The fields above are the fixed core because code consumes them; this is where
    # the reader records whatever else about *this* agent is worth carrying forward — quirks,
    # traps, names that look real but are not — in whatever form fits. It is shown verbatim to
    # every later stage.
    notes: str = ""
    open_questions: list[str] = Field(default_factory=list)
    # Names only; the platform owns the catalogue. Empty means judge by the scenarios' checks alone.
    chosen_evals: list[str] = Field(default_factory=list)
    # Anything in here was not read from the agent's source. The contract is meant to be what
    # the agent verifiably is, so when the harness widens it the difference is recorded rather
    # than blended in, and whoever reads it later can tell the two apart.
    amendments: list[str] = Field(default_factory=list)

    def tool_names(self) -> set[str]:
        return {tool.name for tool in self.tools}

    def brief(self, *, full_schema: bool = True, with_data: bool = False) -> str:
        """The grounding block handed to the model on every downstream call.

        ``with_data`` includes the agent's real starting records rather than only their shape.
        A stage that writes scenarios needs to know a menu exists; a stage that builds the world
        has to reproduce it row for row, and a shape without records is not enough to do that.
        """
        lines: list[str] = []
        for tool in self.tools:
            signature = ", ".join(
                f"{arg}: {tool.arg_types[arg]}" if arg in tool.arg_types else arg
                for arg in tool.args
            )
            values = (
                f"  [values: {json.dumps(tool.arg_values)[:300]}]"
                if tool.arg_values
                else ""
            )
            # Preconditions belong on the tool line or they are not read. A writer that cannot see
            # what a tool refuses until another has run replays the agent's whole flow to reach it.
            needs = f"  [after: {', '.join(tool.requires)}]" if tool.requires else ""
            lines.append(
                f"  - {tool.name}({signature}){values}{needs} : {tool.description[:140]}"
            )
        parts = [
            f"AGENT: {self.agent} - {self.one_liner}",
            f"MODALITY: {self.modality}",
            "REAL TOOLS (use ONLY these, with these exact arg names and types):\n"
            + ("\n".join(lines) or "  (none)"),
        ]
        # Voice only, and stated plainly: a scenario written as though the person dialled in tests
        # nothing when the agent is the one placing the call.
        if self.modality == "voice" and self.call_direction:
            parts.insert(
                2,
                f"CALL DIRECTION: {self.call_direction} - "
                + (
                    "this agent places the call, so the person did not dial and has no request "
                    "to open with"
                    if self.call_direction == "outbound"
                    else "people dial in to this agent"
                ),
            )
        if self.hard_constraints:
            parts.append(
                "HARD CONSTRAINTS the agent MUST follow (nothing may contradict these):\n  - "
                + "\n  - ".join(self.hard_constraints[:14])
            )
        if self.data_schema and full_schema:
            parts.append(
                "DATA SHAPE (the fields each record has):\n"
                + json.dumps(self.data_schema)[: 24000 if with_data else 2400]
            )
        if self.base_environment and with_data:
            parts.append(
                "THE AGENT'S REAL STARTING DATA. Reproduce this exactly, including anything\n"
                "that looks like a mistake: a misspelled id, an item marked unavailable, an odd\n"
                "price. The world is a replica of what the agent has, not a corrected version,\n"
                "and a test written against a corrected world will not catch the real bug.\n"
                + json.dumps(self.base_environment, ensure_ascii=False)
            )
        if self.dependencies:
            parts.append(
                "WHAT THIS AGENT DEPENDS ON (the environment has to provide each of these):\n  - "
                + "\n  - ".join(
                    f"{one.name} ({one.kind or 'unspecified'}): {one.what}"
                    + (f" — used by {', '.join(one.used_by)}" if one.used_by else "")
                    + (f" — {_reached(one)}" if _reached(one) else "")
                    for one in self.dependencies
                )
                + "\nThe agent's code is never edited; the environment must match its existing seam."
            )
        if self.real_use_cases:
            # Every one of them. A scenario writer covers what it is shown, so a truncated
            # list silently caps coverage at the cut rather than at the agent's surface.
            parts.append(
                "REAL USE CASES (what this agent is actually for):\n  - "
                + "\n  - ".join(self.real_use_cases)
            )
        if self.tool_entrypoints:
            parts.append(
                "THE AGENT'S OWN TOOL CODE. Run these rather than writing replacements:\n  - "
                + "\n  - ".join(
                    f"{one.tool}: {one.mode}"
                    + (f" {one.module}.{one.callable}" if one.module else "")
                    + (f", state passed as {one.first_arg}" if one.first_arg else "")
                    + (f", build with {one.factory}" if one.factory else "")
                    for one in self.tool_entrypoints
                )
            )
        if self.runtime_dependencies:
            parts.append(
                "RUNTIME CONNECTIONS (not business-world state; do not rebuild):\n"
                + "\n".join(
                    f"  {one.name}: {one.what} {_reached(one)}"
                    for one in self.runtime_dependencies
                )
            )
        if self.refusal_signature:
            parts.append(
                "HOW THIS AGENT REFUSES, in a value rather than by raising:\n  "
                f"{self.refusal_signature}"
            )
        if self.data_store:
            store = self.data_store
            parts.append(
                "ITS DATA STORE:\n"
                f"  kind: {store.kind or 'unspecified'}\n"
                f"  connection comes from: {store.configured_by or 'unknown'}\n"
                f"  schema from: {store.schema_from or 'unknown'}\n"
                f"  its own loader: {store.loaded_by or 'none'}"
            )
        if self.runtime:
            run = self.runtime
            parts.append(
                "RUNNING ITS CODE:\n"
                f"  {run.language or 'language unspecified'} {run.version}, "
                f"install with {run.install or 'unknown'}"
                + (f", imports resolve from {run.workdir}" if run.workdir else "")
                + (
                    f", its own Compose file at {run.compose_file}"
                    if run.compose_file
                    else ""
                )
                + (
                    f", its own Dockerfile at {run.dockerfile}"
                    if run.dockerfile
                    else ""
                )
                + (
                    f", reached over {run.interface.kind} {run.interface.protocol} on "
                    f"port {run.interface.port}{run.interface.path}"
                    if run.interface
                    else ""
                )
            )
        if self.notes:
            parts.append(f"NOTES from reading the agent:\n{self.notes[:1500]}")
        return "\n\n".join(parts)

    def entry_for(self, tool: str) -> ToolEntry | None:
        for one in self.tool_entrypoints:
            # Coerced rather than assumed. Assigning this field directly bypasses validation, so
            # an entry can arrive as a plain mapping, and reading it as an object would raise
            # somewhere far from the assignment.
            found = one if isinstance(one, ToolEntry) else ToolEntry(**dict(one))
            if found.tool == tool:
                return found
        return None

    def adoptable(self, tool: str) -> bool:
        """Whether this tool has code of its own that should be run instead of replaced."""
        found = self.entry_for(tool)
        return bool(found and found.mode in ("import", "construct", "service"))


def is_data_free_conversation(contract: AgentContract) -> bool:
    """Whether the contract claims conversation without custom tools or business state.

    Callers must also inspect the actual world; this claim alone is not a runtime exemption.
    """
    store = (
        contract.data_store.model_dump(exclude_defaults=True)
        if contract.data_store
        else {}
    )
    if store.get("kind") in {"", "none", "in_process"}:
        store.pop("kind", None)
    return bool(
        contract.conversational
        and not (
            contract.tools
            or contract.tool_entrypoints
            or contract.dependencies
            or contract.data_schema
            or contract.base_environment
            or store
        )
    )


def validate_contract(contract: AgentContract) -> list[str]:
    """Structural problems that make a contract unusable downstream.

    Deliberately narrow. This cannot tell whether the model read the agent correctly, only
    whether the result is shaped well enough to build a world from. Semantic grounding is the
    operator's job, which is why the harness surfaces the contract for review.
    """
    problems: list[str] = []
    if not contract.agent.strip():
        problems.append("empty:agent")
    for dependency in contract.dependencies:
        if dependency.reached.dsn_env in {"LIVEKIT_URL", "LIVEKIT_INFERENCE_URL"}:
            problems.append(
                f"dependency[{dependency.name}]:runtime-connection-in-world — "
                "LiveKit RTC/Inference is a runtime connection, not business-world data. "
                "Move it to runtime_dependencies. Do not generate tables or tool sequences for it."
            )
    for dependency in contract.runtime_dependencies:
        if (
            dependency.kind.lower() in {"datastore", "database", "file", "queue"}
            or dependency.reached.database
        ):
            problems.append(
                f"dependency[{dependency.name}]:business-data-in-runtime — "
                "Business data belongs in dependencies, not runtime_dependencies."
            )
    for index, tool in enumerate(contract.tools):
        if not tool.name.strip():
            problems.append(f"tool[{index}]:no-name")
            continue
        unknown = sorted(set(tool.arg_types) - set(tool.args))
        if unknown:
            problems.append(
                f"tool[{tool.name}]:types-for-unknown-args:{','.join(unknown)}"
            )
    # Conversational agents may expose no tools. Zero-argument tools are also legitimate;
    # cardinality alone is not evidence that authoring omitted something.
    if not contract.real_use_cases:
        problems.append("no-use-cases")
    # Iterate the tools, not tool_names(): that returns a set, so duplicates collapse before
    # they can be counted and the check silently never fires.
    names = [tool.name for tool in contract.tools if tool.name.strip()]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        problems.append(f"duplicate-tool-names:{','.join(duplicates)}")
    return problems
