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
import re
import shlex
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from fi.simulate.agent.wrappers.http_events import (
    CompletedToolEvents,
    CompletedToolRows,
)

from .observations import HttpObservation

# How a person reaches an agent. This decides how it is later run — voice goes out as a live
# call, everything else runs locally — so it is defined once and referenced, never retyped.
MODALITIES = ("voice", "chat", "browser")


def known_modalities() -> tuple[str, ...]:
    """The modalities a contract may declare: these, plus whatever a kind file adds."""
    from .config import declared_modalities

    known = list(MODALITIES)
    known.extend(one for one in declared_modalities() if one not in known)
    return tuple(known)


_RUNTIME_TEMPLATE_INPUTS = frozenset(
    {
        "thread_id",
        "execution_id",
        "turn_index",
        "scenario_name",
        "persona",
        "situation",
        "expected_outcome",
        "messages",
        "new_message",
        "new_message_content",
        "tools",
        "metadata",
    }
)


def _runtime_template_names(value: Any) -> set[str]:
    if isinstance(value, dict):
        return (
            set().union(*(_runtime_template_names(item) for item in value.values()))
            if value
            else set()
        )
    if isinstance(value, list):
        return (
            set().union(*(_runtime_template_names(item) for item in value))
            if value
            else set()
        )
    if isinstance(value, str):
        return set(re.findall(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}", value))
    return set()


# Voice only, and only two: either the agent placed the call or it answered one.
CALL_DIRECTIONS = ("inbound", "outbound")

_STRING_FIELDS = (
    "agent",
    "one_liner",
    "modality",
    "call_direction",
    "system_prompt_excerpt",
    "caller_identity",
    "notes",
)
_LIST_FIELDS = (
    "agent_languages",
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
    # What a scenario writer cannot see from the signature: refusals by value or by something outside
    # the agent's control, what a call changes or clears, and arguments taken from an earlier output.
    refuses: str = ""
    effects: str = ""
    inputs_from: dict[str, str] = Field(default_factory=dict)


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


class PromptSource(BaseModel):
    """Lines of the agent's source that hold part of its prompt."""

    path: str
    start_line: int
    end_line: int


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


class RuntimeHttpFailureRule(BaseModel):
    status_path: str = Field(min_length=1, max_length=512)
    failure_values: list[str | int | float | bool | None] | None = Field(
        default=None, min_length=1, max_length=32
    )
    success_values: list[str | int | float | bool | None] | None = Field(
        default=None, min_length=1, max_length=32
    )
    message_path: str = Field(default="", max_length=512)

    @field_validator("status_path", "message_path")
    @classmethod
    def _path(cls, value: str) -> str:
        if value and not re.fullmatch(r"(?:\$\.)?[^.\s]+(?:\.[^.\s]+)*", value):
            raise ValueError("runtime_http_failure_path_invalid")
        return value

    @field_validator("failure_values", "success_values", mode="before")
    @classmethod
    def _scalars(cls, values):
        import math

        if values is None:
            return values
        if not isinstance(values, list) or any(
            type(value) not in (str, int, float, bool, type(None))
            or (isinstance(value, float) and not math.isfinite(value))
            for value in values
        ):
            raise ValueError("runtime_http_failure_values_must_be_finite_scalars")
        return values

    @model_validator(mode="after")
    def _one_predicate(self):
        if (self.failure_values is None) == (self.success_values is None):
            raise ValueError("runtime_http_failure_requires_one_predicate")
        return self


class RuntimeHttpSetupRequest(BaseModel):
    """One source-evidenced request required before conversational turns begin."""

    method: str = "POST"
    path: str
    body_template: Any = Field(default_factory=dict)
    encoding: Literal["json", "form"] = "json"
    headers_template: dict[str, str] = Field(default_factory=dict)
    capture_headers: dict[str, str] = Field(default_factory=dict)
    accepted_statuses: list[int] = Field(default_factory=lambda: [200, 201, 204, 409])
    failure_rule: RuntimeHttpFailureRule | None = None
    # Captured response values become placeholders for later setup requests and turns. This
    # models generic create-then-use APIs without teaching the harness about a framework.
    capture: dict[str, str] = Field(default_factory=dict)

    @field_validator("method")
    @classmethod
    def _safe_method(cls, value: str) -> str:
        method = str(value or "POST").strip().upper()
        if method not in {"GET", "POST", "PUT", "PATCH"}:
            raise ValueError("runtime_http_setup_method_unsupported")
        return method

    @field_validator("path")
    @classmethod
    def _absolute_path(cls, value: str) -> str:
        path = str(value or "").strip()
        if not path:
            raise ValueError("runtime_http_setup_path_required")
        return path if path.startswith("/") else "/" + path

    @field_validator("accepted_statuses")
    @classmethod
    def _bounded_statuses(cls, values: list[int]) -> list[int]:
        normalized = list(dict.fromkeys(int(value) for value in values))
        if (
            not normalized
            or len(normalized) > 16
            or any(value < 100 or value > 599 for value in normalized)
        ):
            raise ValueError("runtime_http_setup_statuses_invalid")
        return normalized

    @field_validator("headers_template")
    @classmethod
    def _safe_headers(cls, values):
        from fi.simulate.agent.wrappers.http import validate_http_headers

        return validate_http_headers(values)

    @field_validator("capture_headers")
    @classmethod
    def _safe_header_captures(cls, values):
        values = cls._safe_captures(values)
        if any(
            not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,128}", value)
            or value.lower() == "set-cookie"
            for value in values.values()
        ):
            raise ValueError("runtime_http_capture_header_invalid")
        return values

    @field_validator("capture")
    @classmethod
    def _safe_captures(cls, values: dict[str, str]) -> dict[str, str]:
        if len(values) > 16:
            raise ValueError("runtime_http_setup_captures_too_many")
        normalized: dict[str, str] = {}
        for raw_name, raw_path in values.items():
            name = str(raw_name or "").strip()
            path = str(raw_path or "").strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError("runtime_http_setup_capture_name_invalid")
            if not path:
                raise ValueError("runtime_http_setup_capture_path_required")
            normalized[name] = path
        return normalized


class LocalHttpAuth(BaseModel):
    """Names for a source-owned server token generated privately per world."""

    model_config = {"extra": "forbid"}
    server_environment: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
    template_variable: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")

    def reject_collisions(self, names) -> None:
        if {self.server_environment, self.template_variable} & set(names):
            raise ValueError("runtime_local_auth_name_collision")


class RuntimeInterface(BaseModel):
    """The submitted runtime's existing conversational ingress.

    This is connection metadata, not generated agent behavior. The harness may publish the
    declared container port and translate its request/response envelope, but it never adds an
    endpoint the repository does not already implement.
    """

    kind: str = ""
    protocol: str = "fi.alk"
    model: str | None = Field(
        default=None,
        description="Source-owned model or profile identifier accepted by the OpenAI-compatible "
        "endpoint. Preserve its exact value; it need not be the backing provider model.",
    )
    port: int | None = Field(default=None, ge=1, le=65535)
    path: str = ""
    health_path: str = ""
    include_tools: bool = True
    # ``json_template`` is the generic boundary for a source-owned HTTP API that is neither the
    # ALK nor OpenAI envelope. Templates are data, not executable code. A value consisting solely
    # of ``{{name}}`` preserves the canonical value's JSON type; embedded placeholders render as
    # strings. Dotted/list response paths select the assistant content from arbitrary JSON/SSE.
    request_template: Any = None
    response_path: str = ""
    headers_template: dict[str, str] = Field(default_factory=dict)
    completed_tool_events: CompletedToolEvents | None = None
    completed_tools: CompletedToolRows | None = None
    local_auth: LocalHttpAuth | None = Field(
        default=None,
        description="Native HTTP only: source-declared local server token environment name and "
        "private header template variable. ALK generates a per-world token. Never use for "
        "external account/API credentials; never provide token values.",
    )
    setup_secrets: dict[str, str] = Field(default_factory=dict)
    failure_rule: RuntimeHttpFailureRule | None = None
    setup_requests: list[RuntimeHttpSetupRequest] = Field(
        default_factory=list, max_length=8
    )

    @field_validator("headers_template")
    @classmethod
    def _safe_headers(cls, values):
        from fi.simulate.agent.wrappers.http import validate_http_headers

        return validate_http_headers(values)

    @field_validator("setup_secrets")
    @classmethod
    def _setup_secret_aliases(cls, values):
        if len(values) > 16 or any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)
            for pair in values.items()
            for value in pair
        ):
            raise ValueError("runtime_http_setup_secret_invalid")
        if set(values) & _RUNTIME_TEMPLATE_INPUTS:
            raise ValueError("runtime_http_setup_secret_shadows_input")
        return values

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
        if self.local_auth:
            if len(self.setup_secrets) >= 16:
                raise ValueError("runtime_local_auth_setup_secret_limit")
            if self.kind != "http":
                raise ValueError("runtime_local_auth_requires_http")
            self.local_auth.reject_collisions(
                _RUNTIME_TEMPLATE_INPUTS
                | set(self.setup_secrets)
                | set(self.setup_secrets.values())
            )
            non_header_templates = [self.path, self.request_template]
            for setup in self.setup_requests:
                non_header_templates.extend([setup.path, setup.body_template])
            if self.local_auth.template_variable in _runtime_template_names(
                non_header_templates
            ):
                raise ValueError("runtime_local_auth_requires_header_only")
            if self.local_auth.template_variable not in _runtime_template_names(
                self.headers_template
            ):
                raise ValueError("runtime_local_auth_header_binding_required")
        if self.kind in {"http", "websocket"}:
            if self.port is None:
                raise ValueError(f"runtime_{self.kind}_interface_requires_port")
            if not self.path:
                raise ValueError(f"runtime_{self.kind}_interface_requires_path")
        if self.completed_tools and (
            self.completed_tool_events
            or self.kind != "http"
            or self.protocol != "json_template"
        ):
            raise ValueError(
                "completed tool rows require an exclusive JSON template interface"
            )
        if (
            self.completed_tools
            and not self.completed_tools.error_path
            and not (self.failure_rule and self.failure_rule.success_values is not None)
        ):
            raise ValueError(
                "completed tool rows without an error flag require a success predicate"
            )
        if (
            self.completed_tool_events
            and self.completed_tool_events.source == "response"
            and self.protocol != "json_template"
        ):
            raise ValueError("response tool events require a JSON template interface")
        if self.setup_secrets and self.kind != "http":
            raise ValueError("runtime_http_setup_secrets_require_http")
        if self.completed_tool_events is not None and self.kind != "http":
            raise ValueError("completed tool events require an HTTP interface")
        if self.kind == "http":
            if self.protocol not in {"fi.alk", "openai_chat", "json_template"}:
                raise ValueError(
                    "runtime_http_protocol_unsupported: expected fi.alk, openai_chat, or "
                    "json_template"
                )
            if self.protocol == "json_template" and self.request_template is None:
                raise ValueError("runtime_json_template_requires_request_template")
            available = set(_RUNTIME_TEMPLATE_INPUTS)
            if self.local_auth:
                available.add(self.local_auth.template_variable)
            for setup in self.setup_requests:
                unknown = _runtime_template_names(setup.path) - available
                unknown |= (
                    _runtime_template_names(setup.headers_template)
                    | _runtime_template_names(setup.body_template)
                ) - (available | set(self.setup_secrets))
                if unknown:
                    raise ValueError(
                        "runtime_http_setup_unknown_placeholders: "
                        + ", ".join(sorted(unknown))
                    )
                captures = set(setup.capture) | set(setup.capture_headers)
                duplicate = (captures & (available | set(self.setup_secrets))) | (
                    set(setup.capture) & set(setup.capture_headers)
                )
                if duplicate:
                    raise ValueError(
                        "runtime_http_setup_capture_shadows_input: "
                        + ", ".join(sorted(duplicate))
                    )
                available.update(captures)
            if self.protocol == "json_template":
                unknown = _runtime_template_names(self.request_template) - available
                if unknown:
                    raise ValueError(
                        "runtime_http_request_unknown_placeholders: "
                        + ", ".join(sorted(unknown))
                    )
            unknown = _runtime_template_names(self.path) - available
            unknown |= _runtime_template_names(self.headers_template) - (
                available | set(self.setup_secrets)
            )
            if unknown:
                raise ValueError(
                    "runtime_http_request_unknown_placeholders: "
                    + ", ".join(sorted(unknown))
                )
        if self.kind == "websocket" and self.protocol != "fi.alk":
            raise ValueError("runtime_websocket_protocol_unsupported: expected fi.alk")
        return self


class RuntimeSecretFile(BaseModel):
    model_config = {"extra": "forbid"}

    alias: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    destination: str
    environment_variable: str | None = Field(
        default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$"
    )

    @model_validator(mode="after")
    def _contained_destination(self):
        from pathlib import PurePosixPath

        path = PurePosixPath(self.destination)
        forbidden = (
            "/proc",
            "/sys",
            "/dev",
            "/etc",
            "/bin",
            "/sbin",
            "/usr",
            "/lib",
            "/lib64",
            "/var/run",
            "/run/docker",
            "/run/containerd",
        )
        if (
            not path.is_absolute()
            or str(path) != self.destination
            or len(path.parts) < 3
            or ".." in path.parts
            or "$" in self.destination
            or any(
                path == PurePosixPath(root) or path.is_relative_to(root)
                for root in forbidden
            )
            or self.environment_variable
            in {"PATH", "HOME", "LD_PRELOAD", "LD_LIBRARY_PATH"}
            or (self.environment_variable or "").startswith(
                ("DOCKER_", "COMPOSE_", "BUILDKIT_", "BUILDX_")
            )
        ):
            raise ValueError("runtime_secret_file_destination_invalid")
        return self


class RuntimeConfigFile(BaseModel):
    model_config = {"extra": "forbid"}

    format: Literal["dotenv"] = "dotenv"
    values: dict[str, str] = Field(default_factory=dict)
    secret_aliases: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _bounded_values(self):
        if (set(self.values) & set(self.secret_aliases)
            or len(self.values) + len(self.secret_aliases) > 256
            or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
                   for key in [*self.values, *self.secret_aliases, *self.secret_aliases.values()])
            or sum(len(value.encode()) for value in self.values.values()) > 65536
            or any("\x00" in value or "\n" in value or "\r" in value for value in self.values.values())):
            raise ValueError("runtime_config_file_invalid")
        return self


def runtime_public_environment(runtime: dict) -> dict[str, str]:
    result = {}
    confidential = set()
    for raw in runtime.get("config_files", {}).values():
        resource = RuntimeConfigFile.model_validate(raw)
        confidential.update(resource.secret_aliases.values())
        for key, value in resource.values.items():
            if key in result and result[key] != value:
                raise ValueError("compose_config_public_value_conflict")
            result[key] = value
    if confidential & result.keys():
        raise ValueError("compose_config_public_secret_alias_conflict")
    return result


def validate_runtime_resource_paths(config_files, directories):
    from pathlib import PurePosixPath

    if (not isinstance(config_files, dict) or not isinstance(directories, list)
        or any(not isinstance(value, str) for value in [*config_files, *directories])
        or len(config_files) > 32 or len(directories) > 64 or len(set(directories)) != len(directories)):
        raise ValueError("runtime_resources_limit")
    paths = [*config_files, *directories]
    for value in paths:
        path = PurePosixPath(value)
        if (not value or path.is_absolute() or str(path) != value or value == "."
            or ".." in path.parts or any(char in value for char in "$\\\x00")
            or any(part in {".git", ".aws", ".ssh"} for part in path.parts)):
            raise ValueError("runtime_resource_path_invalid")
    for value in config_files:
        path = PurePosixPath(value)
        if not (path.name == ".env" or path.suffix in {".env", ".dotenv"}):
            raise ValueError("runtime_config_file_extension_invalid")
    for index, value in enumerate(paths):
        if any(PurePosixPath(value).is_relative_to(old) or PurePosixPath(old).is_relative_to(value)
               for old in paths[:index]):
            raise ValueError("runtime_resources_overlap")


class Runtime(BaseModel):
    """What it takes to run the agent's code."""

    # Empty means detect from the submitted dependency manifest. Defaulting this to Python makes
    # an otherwise unambiguous Node repository fail the generated-runtime admission path.
    language: str = Field(
        default="",
        description="Source language. Native process supports Python or Bun with a "
        "source-declared exact packageManager version and one root bun.lock or bun.lockb. "
        "Bun requires a matching image toolchain and HTTP command.",
    )
    version: str = ""
    install: str = Field(
        default="",
        description="Source install command recorded as evidence only; never executed. Required source-declared optional groups must also appear in extras.",
    )
    # Optional dependency groups declared by the repository itself (for example ``voice`` in
    # pyproject.toml). Generated packaging validates these names against the manifest.
    extras: list[str] = Field(
        default_factory=list,
        description="Executable selection of source-declared optional dependency groups required by this runtime. Include groups required by install evidence; never invent undeclared groups.",
    )
    requirements_files: list[str] = Field(
        default_factory=list,
        description="Additional source-declared Python requirements files relative to workdir, installed alongside project dependencies and extras. Select files required by install evidence; never invent dependencies.",
    )
    observations: dict[str, HttpObservation] = Field(
        default_factory=dict,
        max_length=32,
        description="Source-reviewed fixed HTTP read operations with bounded scalar parameters. Server read-only semantics require source evidence.",
    )
    workspaces: dict[str, str] = Field(
        default_factory=dict,
        description="Named task workspace roots relative to the agent process working directory. Trace the actual file-tool directory in source/configuration; uploaded source is not automatically the task workspace. These bindings enable private scenario file setup and readback.",
    )
    workdir: str = ""
    # Select one submitted Compose file when a repository contains multiple runnable stacks.
    execution: Literal["auto", "native", "compose"] = "auto"
    compose_file: str = ""
    compose_overrides: list[str] = Field(default_factory=list, max_length=8)
    service: str = ""
    additional_services: list[str] = Field(default_factory=list, max_length=32)
    required_environment: dict[str, list[str]] = Field(default_factory=dict)
    secret_files: dict[str, list[RuntimeSecretFile]] = Field(default_factory=dict)
    config_files: dict[str, RuntimeConfigFile] = Field(default_factory=dict)
    directories: list[str] = Field(default_factory=list)
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

    @model_validator(mode="after")
    def _validate_selection(self) -> "Runtime":
        from .world.files import validate_bindings

        if any(
            not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", name)
            for name in self.observations
        ):
            raise ValueError("runtime_observation_name_invalid")
        validate_bindings(self.workspaces)
        validate_runtime_resource_paths(self.config_files, self.directories)
        if self.interface and self.interface.local_auth:
            if self.execution == "compose" or self.compose_file:
                raise ValueError("runtime_local_auth_requires_native_process")
            for resource in self.config_files.values():
                self.interface.local_auth.reject_collisions(
                    set(resource.values)
                    | set(resource.secret_aliases)
                    | set(resource.secret_aliases.values())
                )
        if self.compose_overrides:
            if self.execution == "native" or not (self.compose_file and self.service):
                raise ValueError("runtime_compose_overrides_require_compose_selection")
            if (
                len(set([self.compose_file, *self.compose_overrides]))
                != len(self.compose_overrides) + 1
            ):
                raise ValueError("runtime_compose_overrides_duplicate")
            if any(
                not path
                or len(path) > 1024
                or "\\" in path
                or "$" in path
                or any(ord(c) < 32 for c in path)
                for path in self.compose_overrides
            ):
                raise ValueError("runtime_compose_overrides_path_invalid")
        if self.additional_services:
            if self.execution == "native" or not (self.compose_file and self.service):
                raise ValueError(
                    "runtime_additional_services_require_compose_selection"
                )
            if (
                len(set(self.additional_services)) != len(self.additional_services)
                or self.service in self.additional_services
                or any(
                    len(name) > 128
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
                    for name in self.additional_services
                )
            ):
                raise ValueError("runtime_additional_services_invalid")
        if self.execution == "native" and (self.compose_file or self.service):
            raise ValueError("runtime_selection_conflict: native cannot select Compose")
        if self.execution == "native" and not self.command:
            raise ValueError(
                "runtime_native_command_missing: explicit native selection needs command"
            )
        if self.execution == "compose" and not (self.compose_file and self.service):
            raise ValueError(
                "runtime_selection_incomplete: Compose needs compose_file and service"
            )
        if self.service and not self.compose_file:
            raise ValueError("runtime_selection_incomplete: service needs compose_file")
        for path in (self.compose_file, self.dockerfile, *self.compose_overrides):
            if path and (path.startswith("/") or ".." in path.split("/")):
                raise ValueError(
                    "runtime_selection_path_invalid: use a repository-relative path"
                )
        for process, names in self.required_environment.items():
            if (
                not process
                or len(names) != len(set(names))
                or any(
                    not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in names
                )
            ):
                raise ValueError("runtime_required_environment_invalid")
        if any(any(char in path for char in "*?[]") for path in self.context_excludes):
            raise ValueError("runtime_context_excludes_glob_unsupported")
        return self

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


def _sampled(environment: Any, keep: int) -> tuple[Any, str]:
    """A few rows per table, and a line saying where the rest is."""
    if not isinstance(environment, dict):
        return environment, ""
    trimmed: dict[str, Any] = {}
    held = 0
    for name, rows in environment.items():
        if isinstance(rows, list):
            trimmed[name] = rows[:keep]
            held += max(0, len(rows) - keep)
        else:
            trimmed[name] = rows
    if not held:
        return environment, ""
    return trimmed, (
        f"Showing the first {keep} rows of each table; {held} more are in the world. "
        "inspect_world returns any table in full, so read it there rather than assuming.\n"
    )


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
    # Voice only: the languages the agent's own instructions say it speaks. Empty when they do not say.
    agent_languages: list[str] = Field(default_factory=list)
    conversational: bool = True
    system_prompt_excerpt: str = ""
    # When given, the excerpt is replaced by these lines copied from the source as written.
    system_prompt_from: list[PromptSource] = Field(default_factory=list)
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
    # How the agent knows who it is talking to, and what it loads from that before any tool runs.
    caller_identity: str = ""
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
    instrumentation: str = "none"

    def tool_names(self) -> set[str]:
        return {tool.name for tool in self.tools}

    def brief(
        self, *, full_schema: bool = True, with_data: bool = False, sample_rows: int = 0
    ) -> str:
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
                f"  - {tool.name}({signature}){values}{needs} : {tool.description}"
                + "".join(
                    f"\n      {label}: {text}"
                    for label, text in (
                        ("refuses", tool.refuses),
                        ("effects", tool.effects),
                        ("arguments from", ", ".join(f"{arg} <- {source}" for arg, source in tool.inputs_from.items())),
                    )
                    if text
                )
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
        if self.caller_identity.strip():
            parts.append(
                "WHO THE AGENT IS TALKING TO (how it identifies them, and what it loads from that "
                "before any tool runs):\n" + self.caller_identity.strip()
            )
        if self.system_prompt_excerpt.strip():
            parts.append(
                "THE AGENT'S OWN PROMPT (what it does in each situation, including where a rule "
                "overrides its usual flow):\n" + self.system_prompt_excerpt.strip()
            )
        if self.hard_constraints:
            parts.append(
                "HARD CONSTRAINTS the agent MUST follow (nothing may contradict these):\n  - "
                + "\n  - ".join(self.hard_constraints)
            )
        if self.data_schema and full_schema:
            parts.append(
                "DATA SHAPE (the fields each record has):\n"
                + json.dumps(self.data_schema)[: 24000 if with_data else 2400]
            )
        if self.base_environment and with_data:
            data, note = self.base_environment, ""
            if sample_rows:
                data, note = _sampled(self.base_environment, sample_rows)
            parts.append(
                "THE AGENT'S REAL STARTING DATA. Reproduce this exactly, including anything\n"
                "that looks like a mistake: a misspelled id, an item marked unavailable, an odd\n"
                "price. The world is a replica of what the agent has, not a corrected version,\n"
                "and a test written against a corrected world will not catch the real bug.\n"
                + note
                + json.dumps(data, ensure_ascii=False)
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


def validate_tool_workflow(contract: AgentContract) -> None:
    declared = {tool.name for tool in contract.tools if tool.name.strip()}
    for entry in contract.tool_entrypoints:
        if entry.tool not in declared:
            continue
        if entry.mode == "import" and entry.module.strip() and entry.callable.strip():
            return
        if (
            entry.mode == "construct"
            and entry.module.strip()
            and entry.callable.strip()
            and entry.factory.strip()
        ):
            return
        if entry.mode == "service" and entry.service.strip() and entry.endpoint.strip():
            return
    raise ValueError(
        "required_tool_workflow_unavailable: no declared tool has an executable entrypoint"
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
    # An `import`/`construct` entry without both halves is unusable: bundling compiles a binding
    # from exactly these two fields. Caught here so the model is told while it can still fix the
    # entry, rather than the run dying much later in `_compile_source_tool_handlers` after the
    # runtime-validation attempts have been spent.
    for entry in contract.tool_entrypoints:
        if entry.mode not in {"import", "construct"}:
            continue
        if entry.module.strip() and entry.callable.strip():
            continue
        problems.append(
            f"tool_entrypoint[{entry.tool}]:{entry.mode}-needs-module-and-callable — "
            "give the importable module path and the callable name, or record a mode that "
            "matches what the repository actually exposes."
        )
    # Iterate the tools, not tool_names(): that returns a set, so duplicates collapse before
    # they can be counted and the check silently never fires.
    names = [tool.name for tool in contract.tools if tool.name.strip()]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        problems.append(f"duplicate-tool-names:{','.join(duplicates)}")
    return problems
