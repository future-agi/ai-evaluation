"""Environment releases, suite releases, agent bindings and run manifests.

The normative text is `api_contracts/harness/environment-releases.md` in the platform repository.
This module is its only sealer and verifier: producers call it and nobody reimplements a digest,
for the same reason `bundle_v2.seal_bundle_v2` is the only bundle sealer.

An environment release is a world (the existing `world.snapshot` layout, so `restore` and `prove`
work on it unchanged), its canonical tool definitions and its policy. A suite release is a set of
scenario folders proved against exactly one environment digest. Sealing a suite runs every
scenario's proof itself; a suite whose scenarios were not executed against its environment
cannot be sealed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

from .bindings import AgentTool, ToolBinding, binding_problems
from .bundle import _safe_relative
from .bundle_v2 import BundleFileV2, _canonical_json

ENVIRONMENT_RELEASE_SCHEMA = "futureagi.environment-release.v1"
SUITE_RELEASE_SCHEMA = "futureagi.suite-release.v1"
AGENT_BINDING_SCHEMA = "futureagi.agent-binding.v1"
RUN_MANIFEST_SCHEMA = "futureagi.run-manifest.v1"

ENVIRONMENT_MANIFEST = "release.json"
SUITE_MANIFEST = "suite.json"
TOOLS = "tools.json"
POLICY = "policy.json"
SERVERS = "servers.json"
WORLD = "world"
PROOFS = "proofs.json"
# The platform creates a suite's scenario records from this, exactly as the guest's provision
# request would describe them (`scenario_source._presented`), without re-deriving the mapping.
PROVISION = "provision.json"
SCENARIOS = "scenarios"
SUB_GOALS = "sub_goals.json"

# Configurable parameters live in the world store, so a handler reads them through the same `db`
# it reads everything else from. Defaults are sealed with the environment; a suite's policy
# profile overwrites them before any scenario's setup runs, in proofs and in runs alike.
POLICY_TABLE = "environment_policy"
WORLD_IMPLEMENTATION = "environment"

_DIGEST_PATTERN = r"sha256:[0-9a-f]{64}"
_SLUG_PATTERN = r"[a-z][a-z0-9_]*"
_TOOL_NAME_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*"
_UNSEALED = "sha256:" + "0" * 64
_STRAY_NAMES = {".DS_Store", "__pycache__"}
_STRAY_SUFFIXES = (".pyc", ".pyo")


class ReleaseError(RuntimeError):
    """A release, binding or manifest is malformed, tampered with, or breaks a contract rule."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DigestRef(_Strict):
    digest: str = Field(pattern=_DIGEST_PATTERN)


class ReleaseRef(DigestRef):
    name: str = Field(pattern=_SLUG_PATTERN)


# --- environment contents ------------------------------------------------------------------------


class ToolDefinition(_Strict):
    name: str = Field(pattern=_TOOL_NAME_PATTERN)
    description: str = Field(min_length=1)
    parameters: dict[str, JsonValue]
    mutates_state: bool
    returns: dict[str, JsonValue] | None = None

    @field_validator("parameters")
    @classmethod
    def _object_schema(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if value.get("type") != "object":
            raise ValueError("parameters must be a JSON Schema object")
        return value


class DatabaseReader(_Strict):
    """Read access to these world tables through the reader role, nothing else."""

    tables: list[str] = Field(min_length=1)


ATTACHED = "attached"


class ServerAttach(_Strict):
    database_reader: DatabaseReader | None = None
    # A folder of the release under `attached/`, the only one the server may read.
    directory: str | None = None


class AttachedServer(_Strict):
    """One MCP server from `mcp_servers.REGISTRY`, started fresh in every world. Its allowed
    `tools` reach the agent as `<id>__<tool>`."""

    id: str = Field(pattern=_SLUG_PATTERN)
    server: str
    tools: list[str] = Field(min_length=1)
    attach: ServerAttach = Field(default_factory=ServerAttach)

    def tool_names(self) -> list[str]:
        return [f"{self.id}__{tool}" for tool in self.tools]


def load_servers(root: Path) -> list[AttachedServer]:
    """The release's `servers.json`; an environment without one attaches nothing."""
    path = Path(root) / SERVERS
    if not path.is_file():
        return []
    raw = _read_json(path, "environment_servers_invalid")
    if not isinstance(raw, list):
        raise ReleaseError("environment_servers_invalid", f"{SERVERS} must be a list")
    return [
        _validated(AttachedServer, item, "environment_servers_invalid") for item in raw
    ]


def attached_tool_names(root: Path) -> set[str]:
    return {name for server in load_servers(root) for name in server.tool_names()}


def _check_servers(root: Path, tools: Sequence[ToolDefinition]) -> None:
    from .mcp_servers import DATABASE_READER, DIRECTORY, REGISTRY

    servers = load_servers(root)
    problems: list[str] = []
    if repeated := sorted(
        name
        for name, count in Counter(server.id for server in servers).items()
        if count > 1
    ):
        problems.append(f"duplicate server ids: {repeated}")
    native = {tool.name for tool in tools}
    world_tables = set(_parent_tables(Path(root) / WORLD / "world.sqlite"))
    for server in servers:
        registered = REGISTRY.get(server.server)
        if registered is None:
            problems.append(
                f"{server.id}: unknown server {server.server!r}; "
                f"known: {', '.join(sorted(REGISTRY))}"
            )
            continue
        if clashes := sorted(set(server.tool_names()) & native):
            problems.append(f"{server.id}: tool names clash with {clashes}")
        # The agent sees `<id>__<tool>` as a tool name; one it cannot be offered as fails only
        # when a world starts, far from the release that listed it.
        if invalid := sorted(
            name for name in server.tool_names() if not re.fullmatch(_TOOL_NAME_PATTERN, name)
        ):
            problems.append(f"{server.id}: not valid tool names: {invalid}")
        if repeated := sorted(name for name, count in Counter(server.tools).items() if count > 1):
            problems.append(f"{server.id}: tools listed twice: {repeated}")
        reader = server.attach.database_reader
        if reader is not None:
            if DATABASE_READER not in registered.attach:
                problems.append(f"{server.id}: {server.server} takes no database reader")
            for table in reader.tables:
                if table not in world_tables:
                    problems.append(f"{server.id}: no world table {table!r}")
                elif table.startswith("_alk_") or table == POLICY_TABLE:
                    problems.append(f"{server.id}: {table} is internal")
        directory = server.attach.directory
        if directory is not None:
            if DIRECTORY not in registered.attach:
                problems.append(f"{server.id}: {server.server} takes no directory")
            path = Path(directory)
            if (
                path.is_absolute()
                or ".." in path.parts
                or path.parts[:1] != (ATTACHED,)
                or not (Path(root) / path).is_dir()
            ):
                problems.append(
                    f"{server.id}: directory must be a folder of the release under {ATTACHED}/"
                )
        if DIRECTORY in registered.attach and directory is None:
            problems.append(f"{server.id}: {server.server} needs a directory")
    if problems:
        raise ReleaseError("environment_servers_invalid", "; ".join(problems))


def reader_grants_sql(environment_root: Path) -> str:
    """The seed lines that let the reader role SELECT the tables attached servers read. They run
    in the template database, so every world cloned from it keeps them."""
    from .mcp_servers import READER_ROLE

    tables = sorted(
        {
            table
            for server in load_servers(environment_root)
            if server.attach.database_reader is not None
            for table in server.attach.database_reader.tables
        }
    )
    if not tables:
        return ""
    return (
        f"DO $$ BEGIN CREATE ROLE {READER_ROLE} NOLOGIN; "
        "EXCEPTION WHEN duplicate_object OR unique_violation THEN NULL; END $$;\n"
        + "".join(f'GRANT SELECT ON "{table}" TO {READER_ROLE};\n' for table in tables)
    )


class RuleClass(str, Enum):
    SYSTEM_INVARIANT = "system_invariant"
    CONFIGURABLE_ENVIRONMENT_POLICY = "configurable_environment_policy"
    AGENT_POLICY = "agent_policy"


_PARAMETER_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
}


def _matches(kind: str, value: Any) -> bool:
    if kind != "boolean" and isinstance(value, bool):
        return False
    return isinstance(value, _PARAMETER_TYPES[kind])


class PolicyParameter(_Strict):
    type: Literal["string", "integer", "number", "boolean"]
    default: JsonValue
    allowed: list[JsonValue] | None = None

    @model_validator(mode="after")
    def _default_is_valid(self) -> "PolicyParameter":
        self.check(self.default, label="default")
        return self

    def check(self, value: Any, *, label: str) -> None:
        if not _matches(self.type, value):
            raise ValueError(f"{label} {value!r} is not a {self.type}")
        if self.allowed is not None and value not in self.allowed:
            raise ValueError(f"{label} {value!r} is not one of {self.allowed}")


class PolicyRule(_Strict):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(pattern=_SLUG_PATTERN)
    text: str = Field(min_length=1)
    rule_class: RuleClass = Field(alias="class")
    parameters: dict[str, PolicyParameter] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _parameters_are_configuration(self) -> "PolicyRule":
        if (
            self.parameters
            and self.rule_class is not RuleClass.CONFIGURABLE_ENVIRONMENT_POLICY
        ):
            raise ValueError(
                f"rule {self.id}: only configurable_environment_policy rules take parameters"
            )
        return self


class Policy(_Strict):
    rules: list[PolicyRule]

    @model_validator(mode="after")
    def _unique(self) -> "Policy":
        rules = Counter(rule.id for rule in self.rules)
        parameters = Counter(name for rule in self.rules for name in rule.parameters)
        repeated = sorted(
            [name for name, count in rules.items() if count > 1]
            + [name for name, count in parameters.items() if count > 1]
        )
        if repeated:
            raise ValueError(f"duplicate rule or parameter names: {repeated}")
        return self

    def parameters(self) -> dict[str, PolicyParameter]:
        return {
            name: parameter
            for rule in self.rules
            for name, parameter in rule.parameters.items()
        }

    def rule_ids(self, rule_class: RuleClass) -> set[str]:
        return {rule.id for rule in self.rules if rule.rule_class is rule_class}


class HandlerMode(str, Enum):
    ENVIRONMENT = "environment"
    CUSTOMER_SOURCE = "customer_source"
    CUSTOMER_SPEC = "customer_spec"


class HandlerEntry(_Strict):
    mode: HandlerMode


class ReleaseProvenance(_Strict):
    generator: str = "fi.alk.harness"
    generator_version: str = "1"
    derived_from: str | None = None


class EnvironmentReleaseV1(_Strict):
    schema_version: Literal["futureagi.environment-release.v1"]
    digest: str = Field(pattern=_DIGEST_PATTERN)
    name: str = Field(pattern=_SLUG_PATTERN)
    extends: ReleaseRef | None = None
    handlers: dict[str, HandlerEntry]
    source: DigestRef | None = None
    files: list[BundleFileV2]
    provenance: ReleaseProvenance = Field(default_factory=ReleaseProvenance)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _source_iff_customer_source(self) -> "EnvironmentReleaseV1":
        bound = any(
            entry.mode is HandlerMode.CUSTOMER_SOURCE
            for entry in self.handlers.values()
        )
        if bound != (self.source is not None):
            raise ValueError(
                "source is required exactly when a handler is customer_source"
            )
        return self


# --- suites ----------------------------------------------------------------------------------------


class ScenarioOrigin(str, Enum):
    BASE = "base"
    SPECIALIZED = "specialized"
    CUSTOM = "custom"


class SuiteScenario(_Strict):
    key: str = Field(min_length=1)
    path: str
    origin: ScenarioOrigin
    derived_from: str | None = None

    @model_validator(mode="after")
    def _shape(self) -> "SuiteScenario":
        _safe_relative(self.path)
        parts = Path(self.path).parts
        if len(parts) != 2 or parts[0] != SCENARIOS:
            raise ValueError(f"scenario path must be {SCENARIOS}/<name>: {self.path}")
        if self.origin is ScenarioOrigin.CUSTOM and self.derived_from is not None:
            raise ValueError(f"{self.key}: a custom scenario derives from nothing")
        if self.origin is ScenarioOrigin.SPECIALIZED and self.derived_from is None:
            raise ValueError(f"{self.key}: a specialized scenario names its parent")
        return self


class PolicyProfile(_Strict):
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    # Each scored agent_policy rule and the sub-goals that grade it. A rule the policy declares
    # but this map leaves out is not scored by the suite, and the report says so.
    scored_agent_rules: dict[str, list[str]] = Field(default_factory=dict)


class SuiteReleaseV1(_Strict):
    schema_version: Literal["futureagi.suite-release.v1"]
    digest: str = Field(pattern=_DIGEST_PATTERN)
    name: str = Field(pattern=_SLUG_PATTERN)
    environment: DigestRef
    parent: DigestRef | None = None
    policy_profile: PolicyProfile = Field(default_factory=PolicyProfile)
    scenarios: list[SuiteScenario] = Field(min_length=1)
    files: list[BundleFileV2]
    provenance: ReleaseProvenance = Field(default_factory=ReleaseProvenance)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique(self) -> "SuiteReleaseV1":
        for field in ("key", "path"):
            counts = Counter(getattr(entry, field) for entry in self.scenarios)
            repeated = sorted(value for value, count in counts.items() if count > 1)
            if repeated:
                raise ValueError(f"duplicate scenario {field}s: {repeated}")
        if self.parent is None and any(
            entry.derived_from is not None for entry in self.scenarios
        ):
            raise ValueError("a suite without a parent derives no scenario")
        return self


# --- bindings and run manifests --------------------------------------------------------------------


Connector = Literal["livekit", "vapi", "retell", "retell_chat", "phone", "hosted_agent"]


class AgentRef(_Strict):
    """Who a binding is for: the platform's agent version and the connection it supplied, plus
    the provider agent as adaptation inspected it. Any of them changing is another binding."""

    agent_definition_id: str = Field(min_length=1)
    agent_version_id: str = Field(min_length=1)
    configuration_digest: str = Field(pattern=_DIGEST_PATTERN)
    connector: Connector
    # The connection's non-secret config (provider target id, dynamic variables).
    agent_config_digest: str = Field(pattern=_DIGEST_PATTERN)
    provider_target_id: str = Field(min_length=1)
    # `provider_profile_digest` of the inspected, redacted provider agent.
    provider_digest: str = Field(pattern=_DIGEST_PATTERN)


class BindingMode(str, Enum):
    CANONICAL = "canonical"
    TOOL_ROUTED = "tool_routed"
    CONVERSATION_ONLY = "conversation_only"


class AgentBindingV1(_Strict):
    schema_version: Literal["futureagi.agent-binding.v1"]
    digest: str = Field(pattern=_DIGEST_PATTERN)
    agent: AgentRef
    environment: DigestRef
    mode: BindingMode
    agent_tools: list[AgentTool] = Field(default_factory=list)
    tools: list[ToolBinding] = Field(default_factory=list)
    fixtures: dict[str, JsonValue] = Field(default_factory=dict)
    aliases: dict[str, dict[str, str]] = Field(default_factory=dict)
    rendering: dict[str, str] = Field(default_factory=dict)
    prompt_changes: list[JsonValue] = Field(default_factory=list)

    @model_validator(mode="after")
    def _mode_rules(self) -> "AgentBindingV1":
        if self.prompt_changes:
            raise ValueError("prompt_changes must be empty in v1")
        if self.mode is not BindingMode.TOOL_ROUTED and (
            self.agent_tools or self.tools or self.fixtures or self.aliases
        ):
            raise ValueError(
                f"a {self.mode.value} binding has no agent_tools, tools, fixtures or aliases"
            )
        return self


class SimulatorLlm(_Strict):
    provider: str
    model: str
    location: str | None = None
    thinking: str | None = None
    # Tried in turn when `model` fails; frozen so a later deployment cannot change a run.
    fallback_models: list[str] = Field(default_factory=list)


class SimulatorVoice(_Strict):
    stt: str | None = None
    tts: str | None = None
    voice: str | None = None


class SimulatorPersona(_Strict):
    prompt_digest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)
    temperature: float | None = None
    interrupt_sensitivity: float | None = None
    conversation_speed: float | None = None
    finished_speaking_sensitivity: float | None = None
    max_call_duration_in_minutes: int | None = None
    initial_message_delay: int | None = None


class SimulatorSettings(_Strict):
    llm: SimulatorLlm
    voice: SimulatorVoice | None = None
    persona: SimulatorPersona = Field(default_factory=SimulatorPersona)


class ScorerEval(_Strict):
    name: str
    version: str


class ScorerSettings(_Strict):
    alk_version: str
    evals: list[ScorerEval] = Field(default_factory=list)


class RuntimeImage(_Strict):
    provider: str
    image_name: str
    image_digest: str


class Selection(_Strict):
    scenario_keys: list[str] = Field(min_length=1)
    trials: int = Field(ge=1, le=20)

    @field_validator("scenario_keys")
    @classmethod
    def _unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("scenario_keys must be unique")
        return value


class RunManifestV1(_Strict):
    schema_version: Literal["futureagi.run-manifest.v1"]
    digest: str = Field(pattern=_DIGEST_PATTERN)
    environment: list[ReleaseRef] = Field(min_length=1, max_length=2)
    suite: DigestRef
    binding: DigestRef
    agent: AgentRef
    selection: Selection
    simulator: SimulatorSettings
    scorer: ScorerSettings
    runtime: RuntimeImage


# --- digests and files ---------------------------------------------------------------------------


def object_digest(model: BaseModel) -> str:
    """The contract's section 3 digest, identical in construction to bundle v2's section 2d."""
    core = model.model_dump(mode="json", by_alias=True)
    core.pop("digest", None)
    core.pop("files", None)
    digest = hashlib.sha256(_canonical_json(core))
    for record in getattr(model, "files", None) or []:
        encoded = _canonical_json(record.model_dump(mode="json"))
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return "sha256:" + digest.hexdigest()


def _sealed(model: BaseModel) -> BaseModel:
    return model.model_copy(update={"digest": object_digest(model)})


def _is_stray(relative: Path) -> bool:
    return any(
        part in _STRAY_NAMES for part in relative.parts
    ) or relative.name.endswith(_STRAY_SUFFIXES)


def list_files(root: Path, manifest_name: str) -> list[BundleFileV2]:
    """Every file under `root` except its manifest, hashed, in path order.

    Strays and symlinks are refused rather than skipped: a file that is present but unlisted would
    fail verification later, far from whoever put it there.
    """
    records: list[BundleFileV2] = []
    for path in sorted(Path(root).rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ReleaseError("release_symlink", str(relative))
        if _is_stray(relative):
            raise ReleaseError("release_stray_file", str(relative))
        if path.is_dir() or relative.as_posix() == manifest_name:
            continue
        content = path.read_bytes()
        records.append(
            BundleFileV2(
                path=relative.as_posix(),
                sha256=hashlib.sha256(content).hexdigest(),
                size=len(content),
            )
        )
    return records


def _verify_files(
    root: Path, manifest_name: str, listed: Sequence[BundleFileV2]
) -> None:
    present = {record.path: record for record in list_files(root, manifest_name)}
    declared = {record.path: record for record in listed}
    if missing := sorted(set(declared) - set(present)):
        raise ReleaseError("release_file_missing", ", ".join(missing))
    if unlisted := sorted(set(present) - set(declared)):
        raise ReleaseError("release_file_unlisted", ", ".join(unlisted))
    if changed := sorted(path for path in declared if declared[path] != present[path]):
        raise ReleaseError("release_file_changed", ", ".join(changed))


def _read_json(path: Path, code: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleaseError(code, f"{path.name} is missing") from exc
    except (OSError, ValueError) as exc:
        raise ReleaseError(code, f"{path.name}: {exc}") from exc


def _validated(model: type[BaseModel], value: Any, code: str) -> Any:
    try:
        return model.model_validate(value)
    except ValidationError as exc:
        raise ReleaseError(code, str(exc)) from exc


def _write_manifest(root: Path, name: str, model: BaseModel) -> None:
    (root / name).write_text(
        json.dumps(model.model_dump(mode="json", by_alias=True), indent=2) + "\n",
        encoding="utf-8",
    )


def _check_digest(model: BaseModel, code: str) -> None:
    expected = object_digest(model)
    if model.digest != expected:  # type: ignore[attr-defined]
        raise ReleaseError(code, f"declared {model.digest}, computed {expected}")  # type: ignore[attr-defined]


# --- environment releases ------------------------------------------------------------------------


def load_tools(root: Path) -> list[ToolDefinition]:
    raw = _read_json(Path(root) / TOOLS, "environment_tools_invalid")
    if not isinstance(raw, list):
        raise ReleaseError("environment_tools_invalid", f"{TOOLS} must be a list")
    tools = [
        _validated(ToolDefinition, item, "environment_tools_invalid") for item in raw
    ]
    repeated = sorted(
        name
        for name, count in Counter(tool.name for tool in tools).items()
        if count > 1
    )
    if repeated:
        raise ReleaseError("environment_tools_invalid", f"duplicate tools: {repeated}")
    return tools


def load_policy(root: Path) -> Policy:
    return _validated(
        Policy,
        _read_json(Path(root) / POLICY, "environment_policy_invalid"),
        "environment_policy_invalid",
    )


def _forbidden_imports(source: str, name: str) -> list[str]:
    """Imports an environment-mode handler may not make.

    Its whole point is to execute against the world alone, so it may use the standard library and
    the world runtime and nothing else. Dynamic import is refused outright, because it would make
    the static check meaningless.
    """
    try:
        tree = ast.parse(source, filename=f"handlers/{name}.py")
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    found: list[str] = []
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                found.append("relative import")
                continue
            modules = [node.module or ""]
        elif isinstance(node, ast.Name) and node.id == "__import__":
            found.append("__import__")
        for module in modules:
            top = module.split(".")[0]
            if module == "fi.alk.harness.world" or module.startswith(
                "fi.alk.harness.world."
            ):
                continue
            if top == "importlib" or top not in sys.stdlib_module_names:
                found.append(module)
    return found


def _world_manifest(root: Path) -> dict[str, Any]:
    manifest = _read_json(
        Path(root) / WORLD / "manifest.json", "environment_world_invalid"
    )
    if not isinstance(manifest, dict):
        raise ReleaseError(
            "environment_world_invalid", "world manifest is not an object"
        )
    return manifest


def _policy_rows(database: Path) -> dict[str, Any] | None:
    if not database.is_file():
        return None
    connection = sqlite3.connect(database)
    try:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (POLICY_TABLE,)
        ).fetchone()
        if not exists:
            return None
        return {
            name: json.loads(value)
            for name, value in connection.execute(
                f"SELECT name, value FROM {POLICY_TABLE}"
            )
        }
    finally:
        connection.close()


def _check_environment_contents(
    root: Path, handlers: Mapping[str, HandlerEntry]
) -> tuple[list[ToolDefinition], Policy]:
    tools = load_tools(root)
    policy = load_policy(root)
    names = {tool.name for tool in tools}
    if set(handlers) != names:
        raise ReleaseError(
            "environment_handlers_mismatch",
            f"handlers {sorted(handlers)} vs tools {sorted(names)}",
        )
    world = _world_manifest(root)
    if world.get("tool_implementation") != WORLD_IMPLEMENTATION:
        raise ReleaseError(
            "environment_world_invalid",
            f"world tool_implementation must be {WORLD_IMPLEMENTATION!r}",
        )
    if set(world.get("tools") or []) != names:
        raise ReleaseError(
            "environment_handlers_mismatch",
            f"world handlers {sorted(world.get('tools') or [])} vs tools {sorted(names)}",
        )
    if world.get("runtime_tools"):
        raise ReleaseError(
            "environment_world_invalid",
            f"runtime-owned tools cannot be proved: {world['runtime_tools']}",
        )
    # Replacing a prebuilt tool with the customer's own code (contract §2.2) is not built: nothing
    # composes or runs such a handler differently, so a release claiming one would run the copy.
    if unsupported := sorted(
        name for name, entry in handlers.items() if entry.mode is not HandlerMode.ENVIRONMENT
    ):
        raise ReleaseError(
            "environment_handler_unsupported",
            f"only environment handlers are supported: {unsupported}",
        )
    if world.get("source_root"):
        raise ReleaseError(
            "environment_world_invalid", "an environment world does not import source"
        )
    if (Path(root) / WORLD / "state.json").exists():
        raise ReleaseError(
            "environment_world_invalid",
            "environment handlers keep their state in the store, not in state.json",
        )
    handler_files = {
        path.relative_to(Path(root) / WORLD / "handlers").as_posix()
        for path in (Path(root) / WORLD / "handlers").rglob("*")
        if path.is_file() and not _is_stray(path.relative_to(Path(root)))
    }
    if strays := sorted(handler_files - {f"{name}.py" for name in names}):
        raise ReleaseError(
            "environment_handlers_mismatch",
            f"handler files for no tool: {strays}",
        )
    problems: list[str] = []
    for name, entry in handlers.items():
        path = Path(root) / WORLD / "handlers" / f"{name}.py"
        if not path.is_file():
            problems.append(f"{name}: handler file missing")
            continue
        if entry.mode is HandlerMode.ENVIRONMENT:
            problems.extend(
                f"{name}: imports {module}"
                for module in _forbidden_imports(path.read_text(encoding="utf-8"), name)
            )
    if problems:
        raise ReleaseError("environment_handler_invalid", "; ".join(problems))
    defaults = {
        name: parameter.default for name, parameter in policy.parameters().items()
    }
    seeded = _policy_rows(Path(root) / WORLD / "world.sqlite")
    if (seeded or {}) != defaults:
        raise ReleaseError(
            "environment_policy_unseeded",
            f"{POLICY_TABLE} holds {seeded} but the policy defaults are {defaults}",
        )
    _check_servers(root, tools)
    return tools, policy


def _parent_tables(database: Path) -> dict[str, list[str]]:
    connection = sqlite3.connect(database)
    try:
        entries: dict[str, list[str]] = {}
        for table, sql in connection.execute(
            "SELECT tbl_name, sql FROM sqlite_master "
            "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ):
            entries.setdefault(table, []).append(sql)
        return entries
    finally:
        connection.close()


def _table_rows(database: Path, table: str) -> list[tuple[Any, ...]]:
    connection = sqlite3.connect(database)
    try:
        return sorted(
            connection.execute(f'SELECT * FROM "{table}"').fetchall(), key=repr
        )
    finally:
        connection.close()


def _attached_files(root: Path) -> dict[str, bytes]:
    folder = Path(root) / ATTACHED
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file() and not _is_stray(path.relative_to(Path(root)))
    }


def _check_extension(
    root: Path, parent_root: Path, parent: EnvironmentReleaseV1
) -> None:
    """An extension may add to its parent and may change nothing the parent already has."""
    if parent.extends is not None:
        raise ReleaseError(
            "environment_extension_depth", "an extension cannot be extended"
        )
    problems: list[str] = []
    child_tools = {tool.name: tool for tool in load_tools(root)}
    for tool in load_tools(parent_root):
        if child_tools.get(tool.name) != tool:
            problems.append(f"tool {tool.name} removed or redefined")
            continue
        handler = Path(WORLD) / "handlers" / f"{tool.name}.py"
        if (root / handler).read_bytes() != (parent_root / handler).read_bytes():
            problems.append(f"handler {tool.name} changed")
    child_rules = {rule.id: rule for rule in load_policy(root).rules}
    for rule in load_policy(parent_root).rules:
        if child_rules.get(rule.id) != rule:
            problems.append(f"rule {rule.id} removed or changed")
    if load_servers(root) != load_servers(parent_root):
        problems.append(f"{SERVERS} changed; attached servers belong to the base release")
    if _attached_files(root) != _attached_files(parent_root):
        problems.append(f"{ATTACHED}/ changed; attached folders belong to the base release")
    parent_db = parent_root / WORLD / "world.sqlite"
    child_db = root / WORLD / "world.sqlite"
    if parent_db.is_file():
        if not child_db.is_file():
            problems.append("world store removed")
        else:
            child_schema = _parent_tables(child_db)
            for table, sql in _parent_tables(parent_db).items():
                if child_schema.get(table) != sql:
                    problems.append(f"table {table} altered")
                elif table != POLICY_TABLE and _table_rows(
                    child_db, table
                ) != _table_rows(parent_db, table):
                    problems.append(f"table {table} rows changed")
    if problems:
        raise ReleaseError("environment_extension_invalid", "; ".join(problems))


def _lineage(document: Mapping[str, Any]) -> dict[str, Any]:
    """What a sealed manifest derives from: the release it extends or the suite it was derived
    from, and each scenario's origin and source."""
    return {
        "extends": document.get("extends"),
        "parent": document.get("parent"),
        "scenarios": sorted(
            (str(entry.get("key")), str(entry.get("origin")), str(entry.get("derived_from")))
            for entry in document.get("scenarios") or []
        ),
    }


def _unchanged_lineage(manifest: Path, now: Mapping[str, Any]) -> None:
    """Resealing a sealed root must keep what it derives from: sealing again with the defaults
    would quietly turn an extension into a root, or a derived suite into a base one."""
    if not manifest.is_file():
        return
    try:
        before = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    old, new = _lineage(before), _lineage(now)
    if changed := [name for name in old if old[name] != new[name]]:
        raise ReleaseError(
            "release_lineage_changed",
            f"{manifest.name} was sealed with a different {', '.join(changed)}; seal into a "
            "fresh directory, or pass the same parent and scenario origins",
        )


def seal_environment_release(
    root: str | Path,
    *,
    name: str,
    parent_root: str | Path | None = None,
    derived_from: str | None = None,
    metadata: Mapping[str, JsonValue] | None = None,
) -> EnvironmentReleaseV1:
    """Check an environment directory against the contract and write its `release.json`."""
    root = Path(root)
    handlers = {
        tool.name: HandlerEntry(mode=HandlerMode.ENVIRONMENT) for tool in load_tools(root)
    }
    _check_environment_contents(root, handlers)
    extends = None
    if parent_root is not None:
        parent = load_environment_release(parent_root)
        _check_extension(root, Path(parent_root), parent)
        extends = ReleaseRef(digest=parent.digest, name=parent.name)
    _unchanged_lineage(
        root / ENVIRONMENT_MANIFEST, {"extends": extends.model_dump() if extends else None}
    )
    release = _validated(
        EnvironmentReleaseV1,
        {
            "schema_version": ENVIRONMENT_RELEASE_SCHEMA,
            "digest": _UNSEALED,
            "name": name,
            "extends": extends.model_dump() if extends else None,
            "handlers": {
                tool: entry.model_dump(mode="json") for tool, entry in handlers.items()
            },
            "source": None,
            "files": [
                record.model_dump() for record in list_files(root, ENVIRONMENT_MANIFEST)
            ],
            "provenance": {"derived_from": derived_from},
            "metadata": dict(metadata or {}),
        },
        "environment_release_invalid",
    )
    release = _sealed(release)
    _write_manifest(root, ENVIRONMENT_MANIFEST, release)
    return release


def load_environment_release(
    root: str | Path, *, parent_root: str | Path | None = None
) -> EnvironmentReleaseV1:
    """Parse and verify one environment release: schema, every file, the digest, the contents.

    An extension's parent rules need the parent; without `parent_root` an extension is verified on
    its own and its `extends` is only checked for shape.
    """
    root = Path(root)
    raw = _read_json(root / ENVIRONMENT_MANIFEST, "environment_release_invalid")
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != ENVIRONMENT_RELEASE_SCHEMA
    ):
        raise ReleaseError(
            "environment_release_invalid",
            f"schema_version must be {ENVIRONMENT_RELEASE_SCHEMA}",
        )
    release = _validated(EnvironmentReleaseV1, raw, "environment_release_invalid")
    _verify_files(root, ENVIRONMENT_MANIFEST, release.files)
    _check_digest(release, "environment_digest_mismatch")
    _check_environment_contents(root, release.handlers)
    if parent_root is not None:
        if release.extends is None:
            raise ReleaseError(
                "environment_extension_invalid", "release extends nothing"
            )
        parent = load_environment_release(parent_root)
        if parent.digest != release.extends.digest:
            raise ReleaseError(
                "environment_extension_invalid",
                f"extends {release.extends.digest}, parent is {parent.digest}",
            )
        _check_extension(root, Path(parent_root), parent)
    return release


def apply_policy_profile(
    world_root: str | Path, policy: Policy, profile: PolicyProfile
) -> None:
    """Write the effective parameters into a world copy before any scenario's setup runs."""
    parameters = policy.parameters()
    database = Path(world_root) / "world.sqlite"
    if not parameters:
        return
    connection = sqlite3.connect(database)
    try:
        with connection:
            for name, parameter in parameters.items():
                value = profile.parameters.get(name, parameter.default)
                connection.execute(
                    f"UPDATE {POLICY_TABLE} SET value = ? WHERE name = ?",
                    (json.dumps(value), name),
                )
    finally:
        connection.close()


# --- suite releases ------------------------------------------------------------------------------


def _check_profile(profile: PolicyProfile, policy: Policy) -> None:
    parameters = policy.parameters()
    problems: list[str] = []
    for name, value in profile.parameters.items():
        parameter = parameters.get(name)
        if parameter is None:
            problems.append(f"{name} is not a configurable parameter")
            continue
        try:
            parameter.check(value, label=name)
        except ValueError as exc:
            problems.append(str(exc))
    agent_rules = policy.rule_ids(RuleClass.AGENT_POLICY)
    for rule, sub_goals in profile.scored_agent_rules.items():
        if rule not in agent_rules:
            problems.append(f"{rule} is not an agent_policy rule")
        if not sub_goals:
            problems.append(f"{rule} is scored but names no sub-goal that grades it")
    if problems:
        raise ReleaseError("suite_policy_profile_invalid", "; ".join(problems))


def _check_rule_coverage(
    root: Path, profile: PolicyProfile, loaded: Sequence[tuple[SuiteScenario, Any]]
) -> None:
    """Every scored rule is graded by a catalogue sub-goal that a scenario in this suite names."""
    from .catalogue import load_catalogue

    catalogue = load_catalogue(root)
    named = {name for _entry, scenario in loaded for name in scenario.sub_goals}
    problems: list[str] = []
    for rule, sub_goals in profile.scored_agent_rules.items():
        unknown = [name for name in sub_goals if catalogue.named(name) is None]
        if unknown:
            problems.append(
                f"{rule}: {', '.join(unknown)} not in the sub-goal catalogue"
            )
        elif not set(sub_goals) & named:
            problems.append(f"{rule}: no scenario names {', '.join(sub_goals)}")
    if problems:
        raise ReleaseError("suite_policy_profile_invalid", "; ".join(problems))


def _check_simulator_prompt(root: Path) -> None:
    """The simulated person is the suite's prompt with each scenario's persona and instruction
    filled in; a prompt without those slots runs every scenario as the same aimless caller."""
    from .simulator import load_simulator_prompt, validate_simulator_prompt, variables_in

    prompt = load_simulator_prompt(root)
    if not prompt:
        return
    problems = validate_simulator_prompt(prompt, require_persona=True)
    if "instruction" not in variables_in(prompt):
        problems.append("no {{ instruction }} slot for the scenario's task")
    if problems:
        raise ReleaseError("suite_simulator_prompt_invalid", "; ".join(problems))


def graded_rules(
    profile: PolicyProfile, scenario_sub_goals: Sequence[Sequence[str]]
) -> PolicyProfile:
    """The profile restricted to rules some remaining scenario still grades."""
    named = {name for sub_goals in scenario_sub_goals for name in sub_goals}
    return profile.model_copy(
        update={
            "scored_agent_rules": {
                rule: sub_goals
                for rule, sub_goals in profile.scored_agent_rules.items()
                if set(sub_goals) & named
            }
        }
    )


def _scenario_files(root: Path, path: str) -> dict[str, str]:
    folder = root / path
    return {
        file.relative_to(folder).as_posix(): hashlib.sha256(
            file.read_bytes()
        ).hexdigest()
        for file in sorted(folder.rglob("*"))
        if file.is_file()
    }


def _check_suite_contents(
    root: Path,
    scenarios: Sequence[SuiteScenario],
    *,
    parent: tuple[Path, SuiteReleaseV1] | None,
) -> list[Any]:
    from .folder import read_folder

    listed = {Path(entry.path).name for entry in scenarios}
    present = (
        {folder.name for folder in (root / SCENARIOS).iterdir() if folder.is_dir()}
        if (root / SCENARIOS).is_dir()
        else set()
    )
    if unlisted := sorted(present - listed):
        raise ReleaseError("suite_scenario_unlisted", ", ".join(unlisted))
    if not (root / SUB_GOALS).is_file():
        raise ReleaseError("suite_release_invalid", f"{SUB_GOALS} is missing")
    loaded = []
    problems: list[str] = []
    parent_entries = (
        {entry.key: entry for entry in parent[1].scenarios} if parent else {}
    )
    for entry in scenarios:
        folder = Path(entry.path).name
        scenario = read_folder(root, folder)
        if scenario is None:
            problems.append(f"{entry.key}: {entry.path}/scenario.json missing")
            continue
        if scenario.scenario_key != entry.key:
            problems.append(
                f"{entry.key}: scenario.json says scenario_key {scenario.scenario_key!r}"
            )
        if entry.derived_from is not None and parent is not None:
            source = parent_entries.get(entry.derived_from)
            if source is None:
                problems.append(
                    f"{entry.key}: parent has no scenario {entry.derived_from}"
                )
            else:
                same = _scenario_files(root, entry.path) == _scenario_files(
                    parent[0],
                    source.path,  # type: ignore[index]
                )
                if (entry.origin is ScenarioOrigin.BASE) != same:
                    problems.append(
                        f"{entry.key}: origin {entry.origin.value} but files are "
                        f"{'identical to' if same else 'changed from'} {entry.derived_from}"
                    )
        loaded.append((entry, scenario))
    if problems:
        raise ReleaseError("suite_scenario_invalid", "; ".join(problems))
    return loaded


def _prove_all(
    root: Path,
    environment_root: Path,
    policy: Policy,
    profile: PolicyProfile,
    loaded: Sequence[tuple[SuiteScenario, Any]],
) -> dict[str, Any]:
    from .catalogue import load_catalogue
    from .prove import prove

    catalogue = load_catalogue(root)
    attached = frozenset(attached_tool_names(environment_root))
    results: dict[str, Any] = {}
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="suite-proof-") as scratch:
        world_root = Path(scratch) / WORLD
        shutil.copytree(environment_root / WORLD, world_root)
        apply_policy_profile(world_root, policy, profile)
        for entry, scenario in loaded:
            proof = prove(scenario, catalogue, world_root, postgres_only=attached)
            results[entry.key] = {
                "holds": proof.holds,
                "gates": proof.gates(),
                "weak": proof.weak,
                "assumed": proof.assumed,
                "refused": proof.refused,
                "broken": proof.broken,
                "judged_only": proof.judged_only,
                # Only when true, so releases without attached servers keep their digests.
                **({"postgres_only": True} if proof.postgres_only else {}),
            }
            if not proof.holds or proof.assumed:
                reason = (
                    f"steps assumed, not executed: {proof.assumed}"
                    if proof.holds
                    else proof.why()
                )
                failures.append(f"{entry.key}: {reason}")
    if failures:
        raise ReleaseError("suite_scenario_unproved", "\n".join(failures))
    return results


RUNTIME_ENGINE = "postgres"


@dataclass
class Attempt:
    """One scenario played once on a fresh PostgreSQL world and graded."""

    ready: Any
    faults: list[str]
    checks: list[tuple[str, Any]]
    # Each played call's `BoundResult`, in order.
    outcomes: list[Any]
    # The calls the checks were graded on.
    played: list[Any]

    @property
    def failed(self) -> list[str]:
        return [outcome.said for _, outcome in self.checks if not outcome.held]

    @property
    def checks_hold(self) -> bool:
        """Every deterministic check holds; what an adversarial variant must never reach."""
        return self.ready.ok and bool(self.checks) and not self.failed

    @property
    def holds(self) -> bool:
        return self.checks_hold and not self.faults


class PostgresProof:
    """Scenarios graded on PostgreSQL through the tool server's own execution path.

    That is the engine and the code a hosted run uses, so a handler written for SQLite alone is
    refused here rather than faulting in a run. One template database is seeded exactly as a run's
    world is; every attempt gets a fresh clone of it with the scenario's setup applied, plays its
    calls through `WorldTools` (with a binding, or canonically when it is None) and is graded
    through the hosted world handle. Suite sealing and binding proofs share it.
    """

    def __init__(
        self,
        suite_root: Path,
        environment_root: Path,
        profile: PolicyProfile,
        admin_dsn: str,
    ) -> None:
        import secrets

        from .catalogue import load_catalogue
        from .world_tools import load_handlers

        self._environment_root = environment_root
        self._profile = profile
        self._admin_dsn = admin_dsn
        self._catalogue = load_catalogue(suite_root)
        self._handlers = load_handlers(
            environment_root / TOOLS, environment_root / WORLD / "handlers"
        )
        self.environment_tools = load_tools(environment_root)
        self._servers = load_servers(environment_root)
        self._base = f"alk_proof_{secrets.token_hex(6)}"
        self._target = f"{self._base}_w"
        self.server_version = ""

    def _dsn(self, database: str) -> str:
        from psycopg import conninfo

        return conninfo.make_conninfo(self._admin_dsn, dbname=database)

    def _admin(self, *statements: str) -> None:
        import psycopg

        with psycopg.connect(self._admin_dsn, autocommit=True) as admin:
            for statement in statements:
                admin.execute(statement)

    def __enter__(self) -> PostgresProof:
        import psycopg

        with psycopg.connect(self._admin_dsn, autocommit=True) as admin:
            self.server_version = str(
                admin.execute("SHOW server_version").fetchone()[0]
            )
        self._admin(f'CREATE DATABASE "{self._base}"')
        try:
            with psycopg.connect(self._dsn(self._base), autocommit=True) as connection:
                connection.execute(
                    world_seed_sql(self._environment_root, self._profile)
                )
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_: object) -> None:
        self._admin(
            f'DROP DATABASE IF EXISTS "{self._target}" WITH (FORCE)',
            f'DROP DATABASE IF EXISTS "{self._base}" WITH (FORCE)',
            f'DROP ROLE IF EXISTS "{self._target}_reader"',
        )

    def checks(self, scenario: Any) -> list[tuple[str, str]]:
        return [
            (name, goal.check)
            for name in scenario.sub_goals
            if (goal := self._catalogue.named(name)) is not None
            and goal.deterministic()
        ]

    def attempt(
        self,
        scenario: Any,
        calls: Sequence[tuple[str, Any]],
        binding: AgentBindingV1 | None = None,
        *,
        crashed: Sequence[Any] | None = None,
    ) -> Attempt:
        """`calls` are `(tool, arguments)` in `binding`'s vocabulary, played in order. With
        `crashed`, nothing is played: the checks see an untouched world and those calls, each
        failed with no result, the way an execution failure leaves them."""
        import random
        from dataclasses import replace

        from .checks import run_check
        from .folder import apply_setup, check_ready
        from .world.handle import HostedWorld
        from .world.runtime import Call
        from .world.stores.postgres import AttachedPostgresStore
        from .world_tools import WorldTools, attach_servers

        self._admin(
            f'DROP DATABASE IF EXISTS "{self._target}" WITH (FORCE)',
            f'CREATE DATABASE "{self._target}" TEMPLATE "{self._base}"',
        )
        dsn = self._dsn(self._target)
        store = AttachedPostgresStore(dsn)
        tools = WorldTools(
            self._handlers,
            self.environment_tools,
            dsn,
            binding=binding,
            attached=attach_servers(self._servers, dsn, cwd=self._environment_root),
        )
        counts = {table: len(rows) for table, rows in store.state().items()}
        world = HostedWorld(store, 0, random.Random(0), counts)
        ready = apply_setup(scenario, world)
        if ready.ok:
            ready = check_ready(scenario, world)
        played: list[Any] = []
        faults: list[str] = []
        outcomes: list[Any] = []
        try:
            for tool, arguments in calls if ready.ok and crashed is None else ():
                outcome = tools.call(tool, arguments)
                outcomes.append(outcome)
                if outcome.status == "fault":
                    faults.append(f"{tool}: {outcome.error}")
                ok, refused = outcome.status == "ok", outcome.status == "refused"
                # The calls a run's checks read are the trace rows: environment operations, all
                # failed when the agent call was not ok, or one row named by the agent tool.
                played.extend(
                    Call(
                        name=operation.tool,
                        arguments=dict(operation.arguments),
                        result=operation.result if ok else None,
                        ok=ok,
                        refused=refused,
                        error=outcome.error,
                    )
                    for operation in outcome.operations
                )
                if not outcome.operations:
                    played.append(
                        Call(
                            name=tool,
                            arguments=arguments if isinstance(arguments, dict) else {},
                            ok=False,
                            refused=refused,
                            error=outcome.error,
                        )
                    )
        finally:
            tools.close()
        if ready.ok and crashed is not None:
            played = [
                replace(call, ok=False, result=None, error="execution failed", refused=False)
                for call in crashed
            ]
        return Attempt(
            ready,
            faults,
            [
                (name, run_check(source, world, played, name=name))
                for name, source in self.checks(scenario)
            ]
            if ready.ok
            else [],
            outcomes,
            played,
        )


def _prove_on_postgres(
    root: Path,
    environment_root: Path,
    profile: PolicyProfile,
    loaded: Sequence[tuple[SuiteScenario, Any]],
    admin_dsn: str,
) -> dict[str, Any]:
    """Every scenario again, on PostgreSQL (`PostgresProof`), canonically: it must hold with its
    reference solution, and fail when nothing is done and when every call it made crashed."""
    results: dict[str, Any] = {}
    failures: list[str] = []
    with PostgresProof(root, environment_root, profile, admin_dsn) as proof:
        for entry, scenario in loaded:
            solved = proof.attempt(
                scenario,
                [(step.tool, dict(step.arguments)) for step in scenario.solution],
            )
            untouched = proof.attempt(scenario, [])
            weak = [name for name, outcome in untouched.checks if outcome.held]
            # As on SQLite: a check that credits an attempted call without its effect.
            crashed = proof.attempt(scenario, [], crashed=solved.played) if solved.played else None
            credited = [
                name
                for name, outcome in (crashed.checks if crashed else [])
                if outcome.held and name not in weak
            ]
            holds = solved.holds and not weak and not credited
            results[entry.key] = {
                "holds": holds,
                "ready": solved.ready.ok,
                "faults": solved.faults,
                "failed": solved.failed,
                "weak": weak,
            }
            if not holds:
                reason = (
                    solved.ready.said
                    if not solved.ready.ok
                    else "; ".join(
                        [
                            *solved.faults,
                            *solved.failed,
                            *(f"{name} holds untouched" for name in weak),
                            *(f"{name} holds when every call crashed" for name in credited),
                        ]
                    )
                    or "no deterministic check"
                )
                failures.append(f"{entry.key}: {reason}")
    if failures:
        raise ReleaseError(
            "suite_scenario_unproved", "on postgres:\n" + "\n".join(failures)
        )
    return {
        "engine": RUNTIME_ENGINE,
        "server_version": proof.server_version,
        "scenarios": results,
    }


def seal_suite_release(
    root: str | Path,
    *,
    name: str,
    environment_root: str | Path,
    scenarios: Sequence[SuiteScenario | Mapping[str, Any]] | None = None,
    origin: ScenarioOrigin | str = ScenarioOrigin.BASE,
    parent_root: str | Path | None = None,
    policy_profile: PolicyProfile | Mapping[str, Any] | None = None,
    derived_from: str | None = None,
    metadata: Mapping[str, JsonValue] | None = None,
    postgres_dsn: str | None = None,
) -> SuiteReleaseV1:
    """Prove every scenario against the environment, then write `proofs.json` and `suite.json`.

    `scenarios` defaults to every folder under `scenarios/`, keyed by its own `scenario_key`, all
    with `origin`. A derived suite passes its entries explicitly. With `postgres_dsn` (a role that
    may create databases) every scenario is also proved on PostgreSQL, which hosted runs require.
    """
    from .folder import read_folder

    root = Path(root)
    environment_root = Path(environment_root)
    environment = load_environment_release(environment_root)
    policy = load_policy(environment_root)
    profile = _validated(
        PolicyProfile,
        policy_profile.model_dump()
        if isinstance(policy_profile, PolicyProfile)
        else dict(policy_profile or {}),
        "suite_policy_profile_invalid",
    )
    _check_profile(profile, policy)
    if scenarios is None:
        if not (root / SCENARIOS).is_dir():
            raise ReleaseError("suite_release_invalid", f"{SCENARIOS}/ is missing")
        entries: list[SuiteScenario] = []
        for folder in sorted(
            path for path in (root / SCENARIOS).iterdir() if path.is_dir()
        ):
            scenario = read_folder(root, folder.name)
            if scenario is None:
                raise ReleaseError(
                    "suite_scenario_invalid", f"{folder.name}: scenario.json missing"
                )
            entries.append(
                SuiteScenario(
                    key=scenario.scenario_key,
                    path=f"{SCENARIOS}/{folder.name}",
                    origin=ScenarioOrigin(origin),
                )
            )
    else:
        entries = [
            entry
            if isinstance(entry, SuiteScenario)
            else _validated(SuiteScenario, dict(entry), "suite_scenario_invalid")
            for entry in scenarios
        ]
    parent = None
    if parent_root is not None:
        parent = (Path(parent_root), load_suite_release(parent_root))
    _unchanged_lineage(
        root / SUITE_MANIFEST,
        {
            "parent": {"digest": parent[1].digest} if parent else None,
            "scenarios": [entry.model_dump(mode="json") for entry in entries],
        },
    )
    loaded = _check_suite_contents(root, entries, parent=parent)
    _check_rule_coverage(root, profile, loaded)
    _check_simulator_prompt(root)
    (root / PROOFS).unlink(missing_ok=True)
    proofs = _prove_all(root, environment_root, policy, profile, loaded)
    if not postgres_dsn and any(entry.get("postgres_only") for entry in proofs.values()):
        raise ReleaseError(
            "suite_runtime_unproved",
            "scenarios that call attached MCP servers are proved only on PostgreSQL; "
            "seal with a postgres DSN",
        )
    runtime = (
        _prove_on_postgres(root, environment_root, profile, loaded, postgres_dsn)
        if postgres_dsn
        else None
    )
    (root / PROOFS).write_text(
        json.dumps(
            {
                "environment": environment.digest,
                "policy_profile": profile.model_dump(mode="json"),
                "scenarios": proofs,
                "runtime": runtime,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    from .scenario_source import _presented

    (root / PROVISION).write_text(
        json.dumps(
            {
                "personas": [
                    {
                        **_presented(
                            json.loads(
                                (root / entry.path / "scenario.json").read_text()
                            ),
                            scenario_key=entry.key,
                        ),
                        "scenario_key": entry.key,
                    }
                    for entry in entries
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    suite = _validated(
        SuiteReleaseV1,
        {
            "schema_version": SUITE_RELEASE_SCHEMA,
            "digest": _UNSEALED,
            "name": name,
            "environment": {"digest": environment.digest},
            "parent": {"digest": parent[1].digest} if parent else None,
            "policy_profile": profile.model_dump(mode="json"),
            "scenarios": [entry.model_dump(mode="json") for entry in entries],
            "files": [
                record.model_dump() for record in list_files(root, SUITE_MANIFEST)
            ],
            "provenance": {"derived_from": derived_from},
            "metadata": dict(metadata or {}),
        },
        "suite_release_invalid",
    )
    suite = _sealed(suite)
    _write_manifest(root, SUITE_MANIFEST, suite)
    return suite


def load_suite_release(root: str | Path) -> SuiteReleaseV1:
    """Parse and verify one suite release: schema, every file, the digest, the scenario index.

    Proofs are not re-run here; the digest covers `proofs.json`, which only the sealer writes.
    """
    root = Path(root)
    raw = _read_json(root / SUITE_MANIFEST, "suite_release_invalid")
    if not isinstance(raw, dict) or raw.get("schema_version") != SUITE_RELEASE_SCHEMA:
        raise ReleaseError(
            "suite_release_invalid", f"schema_version must be {SUITE_RELEASE_SCHEMA}"
        )
    suite = _validated(SuiteReleaseV1, raw, "suite_release_invalid")
    _verify_files(root, SUITE_MANIFEST, suite.files)
    _check_digest(suite, "suite_digest_mismatch")
    _check_suite_contents(root, suite.scenarios, parent=None)
    proofs = _read_json(root / PROOFS, "suite_release_invalid")
    if proofs.get("environment") != suite.environment.digest or set(
        proofs.get("scenarios") or {}
    ) != {entry.key for entry in suite.scenarios}:
        raise ReleaseError(
            "suite_release_invalid",
            f"{PROOFS} does not cover this suite and environment",
        )
    return suite


# --- bindings and run manifests --------------------------------------------------------------------


def seal_agent_binding(
    fields: Mapping[str, Any],
    *,
    environment_root: str | Path,
    agent_tools: Sequence[str] = (),
) -> AgentBindingV1:
    """Seal a binding after checking it against the release it targets.

    `agent_tools` are the tool names the agent version declares. A canonical agent calls the
    environment's tools by their own names, so every one of them must exist there. A
    `tool_routed` binding is checked statically (§2.4.2); every reason is listed.
    """
    environment = load_environment_release(environment_root)
    binding = _validated(
        AgentBindingV1,
        {
            **fields,
            "schema_version": AGENT_BINDING_SCHEMA,
            "digest": _UNSEALED,
            "environment": {"digest": environment.digest},
        },
        "binding_invalid",
    )
    if binding.mode is BindingMode.CANONICAL:
        known = {tool.name for tool in load_tools(Path(environment_root))}
        if unknown := sorted(set(agent_tools) - known):
            raise ReleaseError("binding_tool_unknown", ", ".join(unknown))
    if binding.mode is BindingMode.TOOL_ROUTED:
        check_tool_routed(binding, load_tools(Path(environment_root)))
    return _sealed(binding)  # type: ignore[return-value]


def check_tool_routed(
    binding: AgentBindingV1, environment_tools: Sequence[ToolDefinition]
) -> None:
    """Refuse a `tool_routed` binding that breaks §2.4.2, listing every reason, one per line."""
    if problems := binding_problems(
        agent_tools=binding.agent_tools,
        tools=binding.tools,
        fixtures=binding.fixtures,
        aliases=binding.aliases,
        environment_tools=environment_tools,
    ):
        raise ReleaseError("binding_invalid", "\n".join(problems))


def load_agent_binding(path: str | Path) -> AgentBindingV1:
    raw = _read_json(Path(path), "binding_invalid")
    if not isinstance(raw, dict) or raw.get("schema_version") != AGENT_BINDING_SCHEMA:
        raise ReleaseError(
            "binding_invalid", f"schema_version must be {AGENT_BINDING_SCHEMA}"
        )
    binding = _validated(AgentBindingV1, raw, "binding_invalid")
    _check_digest(binding, "binding_digest_mismatch")
    return binding


def seal_run_manifest(fields: Mapping[str, Any]) -> RunManifestV1:
    manifest = _validated(
        RunManifestV1,
        {**fields, "schema_version": RUN_MANIFEST_SCHEMA, "digest": _UNSEALED},
        "run_manifest_invalid",
    )
    return _sealed(manifest)  # type: ignore[return-value]


def load_run_manifest(path: str | Path) -> RunManifestV1:
    raw = _read_json(Path(path), "run_manifest_invalid")
    if not isinstance(raw, dict) or raw.get("schema_version") != RUN_MANIFEST_SCHEMA:
        raise ReleaseError(
            "run_manifest_invalid", f"schema_version must be {RUN_MANIFEST_SCHEMA}"
        )
    manifest = _validated(RunManifestV1, raw, "run_manifest_invalid")
    _check_digest(manifest, "run_manifest_digest_mismatch")
    return manifest


def check_run_inputs(
    manifest: RunManifestV1,
    *,
    environments: Sequence[EnvironmentReleaseV1],
    suite: SuiteReleaseV1,
    binding: AgentBindingV1,
) -> None:
    """Refuse a run whose delivered releases are not the ones its manifest froze."""
    problems: list[str] = []
    delivered = [ReleaseRef(digest=one.digest, name=one.name) for one in environments]
    if delivered != manifest.environment:
        problems.append("environment chain differs from the manifest")
    if len(environments) == 2 and (
        environments[1].extends is None
        or environments[1].extends.digest != environments[0].digest
    ):
        problems.append("the extension does not extend the base release")
    if len(environments) == 1 and environments[0].extends is not None:
        problems.append("an extension arrived without its base release")
    target = environments[-1].digest if environments else None
    if suite.digest != manifest.suite.digest:
        problems.append("suite differs from the manifest")
    if suite.environment.digest != target:
        problems.append("suite was proved against another environment")
    if binding.digest != manifest.binding.digest:
        problems.append("binding differs from the manifest")
    if binding.environment.digest != target:
        problems.append("binding targets another environment")
    if binding.agent != manifest.agent:
        problems.append("binding agent differs from the manifest")
    keys = {entry.key for entry in suite.scenarios}
    if unknown := sorted(set(manifest.selection.scenario_keys) - keys):
        problems.append(f"selected scenarios not in the suite: {unknown}")
    if problems:
        raise ReleaseError("run_inputs_mismatch", "; ".join(problems))


_COMPARABLE = ("environment", "suite", "simulator", "scorer", "runtime")


def compare(first: RunManifestV1, second: RunManifestV1) -> list[str]:
    """Fields that stop two runs being a direct comparison; empty when they are one."""
    differing = [
        field
        for field in _COMPARABLE
        if getattr(first, field) != getattr(second, field)
    ]
    if first.selection.scenario_keys != second.selection.scenario_keys:
        differing.append("selection.scenario_keys")
    return differing


# --- run assembly ----------------------------------------------------------------------------------

RELEASES_ENVIRONMENT = "environment"
RELEASES_SUITE = "suite"
TOOL_SERVER = "tool-server"
TOOL_SERVER_DIRECTORY = "generated/tool-server"
TOOLS_CAPABILITY = "tools"
PROVIDER_TOOL_PATH = "/provider/tools"
PROVIDER_EVENT_PATH = "/provider/events"
# Our own agent (`hosted_agent.py`) reaches the world through its MCP endpoint (`world_mcp.py`).
HOSTED_AGENT = "hosted_agent"
WORLD_MCP = "world-mcp"
WORLD_MCP_CAPABILITY = "world_mcp"
_COMPOSABLE_CONNECTORS = {"retell_chat", "retell", "vapi", HOSTED_AGENT}
_COMPOSABLE_MODES = {BindingMode.CANONICAL, BindingMode.TOOL_ROUTED}
_ARG_TYPES = {"string": "str", "integer": "int", "number": "float", "boolean": "bool"}


def environment_directory(releases_root: str | Path, digest: str) -> Path:
    """Where a delivered environment release lives: `environment/<64 hex>`."""
    return Path(releases_root) / RELEASES_ENVIRONMENT / digest.removeprefix("sha256:")


def world_seed_sql(environment_root: str | Path, profile: PolicyProfile) -> str:
    """The PostgreSQL seed of a release's world with a suite's policy profile applied.

    Built by the same converter agent-first bundles use, after the shared prefix that creates the
    readiness sentinel and `_alk_tool_trace`, and the request-binding tables (§6.2), then the
    reader role's grants for attached servers.
    """
    from .bundle_author_v2 import WORLD_SEED_PREFIX, _sql_literal, _sqlite_sql
    from .request_binding import REQUEST_BINDING_SQL

    environment_root = Path(environment_root)
    return (
        WORLD_SEED_PREFIX
        + REQUEST_BINDING_SQL
        + _sqlite_sql(environment_root / WORLD / "world.sqlite")
        + "".join(
            f"UPDATE {POLICY_TABLE} SET value = {_sql_literal(json.dumps(value))} "
            f"WHERE name = {_sql_literal(name)};\n"
            for name, value in (
                (name, profile.parameters.get(name, parameter.default))
                for name, parameter in load_policy(environment_root)
                .parameters()
                .items()
            )
        )
        + reader_grants_sql(environment_root)
    )


def _require_runtime_proof(suite_root: Path, suite: SuiteReleaseV1) -> None:
    runtime = (
        _read_json(suite_root / PROOFS, "suite_release_invalid").get("runtime") or {}
    )
    scenarios = runtime.get("scenarios") or {}
    if runtime.get("engine") != RUNTIME_ENGINE or not all(
        (scenarios.get(entry.key) or {}).get("holds") for entry in suite.scenarios
    ):
        raise ReleaseError(
            "suite_runtime_unproved",
            f"{suite.name} was not proved on {RUNTIME_ENGINE}; seal it with a postgres DSN",
        )


def _contract(name: str, modality: str, tools: Sequence[Any]) -> dict[str, Any]:
    """The `contract.json` the runners read: the agent's view of the tools it calls (release
    `ToolDefinition`s, or a `tool_routed` binding's `agent_tools`)."""
    from .contract import AgentContract, ToolSpec

    specs = []
    for tool in tools:
        properties = tool.parameters.get("properties") or {}
        specs.append(
            ToolSpec(
                name=tool.name,
                description=tool.description,
                args=list(properties),
                arg_types={
                    argument: _ARG_TYPES.get(str(schema.get("type")), "str")
                    for argument, schema in properties.items()
                    if isinstance(schema, dict)
                },
            )
        )
    return AgentContract(agent=name, modality=modality, tools=specs).model_dump(
        mode="json"
    )


def compose_bundle(
    releases_root: str | Path,
    *,
    binding_path: str | Path,
    manifest_path: str | Path,
    job_path: str | Path,
    output: str | Path,
    interpreter: str = sys.executable,
) -> Any:
    """Verify a run's delivered releases and write the sealed v2 bundle the guest runs.

    The world becomes the managed PostgreSQL `world_db` (seeded from the release's world with the
    suite's policy profile applied), the release's handlers run in a per-world tool server that
    records `_alk_tool_trace`, and the provider target is imported against that server.
    """
    from .bundle import CapabilityProtocol
    from .bundle_author_v2 import WORLD_SEED_SENTINEL_QUERY, _managed_world_db
    from .bundle_v2 import (
        BUNDLE_V2_MANIFEST,
        BUNDLE_V2_SCHEMA_VERSION,
        BaselineStrategy,
        BundleProvenanceV2,
        BundleRuntimeV2,
        CapabilityV2,
        EnvironmentBundleV2,
        EvidenceSeam,
        ManagedEngine,
        ProcessUser,
        ReadinessProbeV2,
        RuntimeKindV2,
        Seed,
        Sentinel,
        SourceProcess,
        StartedCheck,
        StoreBaseline,
        StoreEntry,
        compute_inputs_digest,
        seal_bundle_v2,
    )
    from .job import HarnessJob
    from .provider_import import ProviderImportSpec
    from .provider_lifecycle import ProviderType
    from .release_tool_server import BINDING_FILE

    releases_root, output = Path(releases_root), Path(output)
    manifest = load_run_manifest(manifest_path)
    binding = load_agent_binding(binding_path)
    directories = [
        environment_directory(releases_root, ref.digest) for ref in manifest.environment
    ]
    environments = [load_environment_release(directories[0])]
    if len(directories) == 2:
        environments.append(
            load_environment_release(directories[1], parent_root=directories[0])
        )
    suite_root = releases_root / RELEASES_SUITE
    suite = load_suite_release(suite_root)
    check_run_inputs(manifest, environments=environments, suite=suite, binding=binding)
    _require_runtime_proof(suite_root, suite)
    try:
        job = HarnessJob.model_validate_json(Path(job_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReleaseError("compose_job_invalid", str(exc)) from exc
    connector = job.agent.connector.strip().lower()
    if connector != binding.agent.connector:
        raise ReleaseError(
            "run_inputs_mismatch",
            f"job connector {connector} differs from the binding's {binding.agent.connector}",
        )
    hosted = connector == HOSTED_AGENT
    if (
        binding.mode not in _COMPOSABLE_MODES
        or connector not in _COMPOSABLE_CONNECTORS
        or (hosted and binding.mode is not BindingMode.CANONICAL)
    ):
        raise ReleaseError(
            "compose_unsupported",
            f"{binding.mode.value} bindings on {connector} are not composed; canonical on "
            f"{', '.join(sorted(_COMPOSABLE_CONNECTORS))} and tool_routed on provider agents are",
        )
    target_id = ""
    if hosted:
        from .hosted_agent import AgentSpec

        try:
            spec = AgentSpec.from_config(job.agent.config)
        except ValueError as exc:
            raise ReleaseError("compose_job_invalid", str(exc)) from exc
        if spec.digest() != binding.agent.provider_digest:
            raise ReleaseError(
                "run_inputs_mismatch", "the run's agent spec is not the one bound"
            )
    else:
        target_key = "assistant_id" if connector == "vapi" else "agent_id"
        target_id = str(job.agent.config.get(target_key) or "").strip()
        if not target_id:
            raise ReleaseError(
                "compose_job_invalid", f"agent.config.{target_key} is required"
            )
        if target_id != binding.agent.provider_target_id:
            raise ReleaseError(
                "run_inputs_mismatch",
                f"the run calls provider agent {target_id}, the binding was adapted for "
                f"{binding.agent.provider_target_id}",
            )
    if output.exists():
        raise ReleaseError("compose_output_exists", str(output))

    world_root = directories[-1]
    tools = load_tools(world_root)
    agent_tools = (
        list(binding.agent_tools) if binding.mode is BindingMode.TOOL_ROUTED else tools
    )
    agent_tool_names = sorted(tool.name for tool in agent_tools)
    modality = "chat" if connector in ("retell_chat", HOSTED_AGENT) else "voice"
    # Provider agents reach the world's tools through the webhook tool server; our agent reaches
    # the same tools over MCP. Both run from the same directory through WorldTools.
    if hosted:
        server_name, capability, health = WORLD_MCP, WORLD_MCP_CAPABILITY, "/healthz"
        run_command = [interpreter, "-m", "fi.alk.harness.world_mcp"]
        server_environment: dict[str, str] = {}
    else:
        server_name, capability, health = TOOL_SERVER, TOOLS_CAPABILITY, "/health"
        run_command = [interpreter, "server.py"]
        server_environment = {
            "TOOL_PATH": PROVIDER_TOOL_PATH,
            "EVENT_PATH": PROVIDER_EVENT_PATH,
            "REQUEST_BINDING": "1",
        }
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        shutil.copytree(suite_root / SCENARIOS, staging / SCENARIOS)
        for name in (SUB_GOALS, "simulator_prompt.md"):
            if (suite_root / name).is_file():
                shutil.copyfile(suite_root / name, staging / name)
        shutil.copytree(world_root / WORLD / "handlers", staging / "handlers")
        server = staging / TOOL_SERVER_DIRECTORY
        shutil.copytree(world_root / WORLD / "handlers", server / "handlers")
        shutil.copyfile(world_root / TOOLS, server / TOOLS)
        if (world_root / SERVERS).is_file():
            shutil.copyfile(world_root / SERVERS, server / SERVERS)
        if (world_root / ATTACHED).is_dir():
            shutil.copytree(world_root / ATTACHED, server / ATTACHED)
        shutil.copyfile(binding_path, server / BINDING_FILE)
        shutil.copyfile(
            Path(__file__).with_name("release_tool_server.py"), server / "server.py"
        )
        (staging / "contract.json").write_text(
            json.dumps(
                _contract(suite.name, modality, agent_tools), indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        seed = staging / "seed"
        seed.mkdir()
        (seed / "world.sql").write_text(
            world_seed_sql(world_root, suite.policy_profile), encoding="utf-8"
        )
        migrations = ["seed/world.sql"]
        bundle = EnvironmentBundleV2(
            schema_version=BUNDLE_V2_SCHEMA_VERSION,
            digest=_UNSEALED,
            name=suite.name,
            runtime=BundleRuntimeV2(
                kind=RuntimeKindV2.PROCESS, evidence_seam=EvidenceSeam.TOOL_TRACE
            ),
            processes=[
                _managed_world_db(),
                SourceProcess(
                    name=server_name,
                    working_directory=TOOL_SERVER_DIRECTORY,
                    source_origin="bundle",
                    run_command=run_command,
                    environment={
                        "PORT": f"{{{{PORT_{server_name}}}}}",
                        "DATABASE_URL": "{{WORLD_DATABASE_URL}}",
                        **server_environment,
                    },
                    started_check=StartedCheck(port=True, timeout_seconds=180),
                    user=ProcessUser.SVC_TOOLS,
                    depends_on=["world-db"],
                ),
            ],
            seed=Seed(
                stores=[
                    StoreEntry(
                        capability="world_db",
                        migrations=migrations,
                        baseline=StoreBaseline(
                            strategy=BaselineStrategy.TEMPLATE_DATABASE,
                            inputs_digest=compute_inputs_digest(
                                staging,
                                migrations,
                                [],
                                engine=ManagedEngine.POSTGRES,
                                version="16",
                            ),
                        ),
                        sentinel=Sentinel(
                            query=WORLD_SEED_SENTINEL_QUERY, expected="ready"
                        ),
                    )
                ]
            ),
            capabilities={
                "world_db": CapabilityV2(
                    protocol=CapabilityProtocol.POSTGRES,
                    service="world-db",
                    container_port=5432,
                    configuration_name="WORLD_DATABASE_URL",
                ),
                capability: CapabilityV2(
                    protocol=CapabilityProtocol.HTTP,
                    service=server_name,
                    container_port=8080,
                    configuration_name=f"{capability.upper()}_URL",
                ),
            },
            readiness=[
                ReadinessProbeV2(capability="world_db", timeout_seconds=180),
                ReadinessProbeV2(capability=capability, path=health, timeout_seconds=180),
            ],
            files=list_files(staging, BUNDLE_V2_MANIFEST),
            provenance=BundleProvenanceV2(
                source_kind="release",
                source_digest=manifest.digest.removeprefix("sha256:"),
                generator="fi.alk.harness.releases",
                generator_version="1",
                adopted_files=["scenarios/", SUB_GOALS, "handlers/", "seed/world.sql"],
                generated_files=[
                    BUNDLE_V2_MANIFEST,
                    "contract.json",
                    TOOL_SERVER_DIRECTORY,
                ],
            ),
            metadata={
                **(
                    {}
                    if hosted
                    else {
                        "provider_import": ProviderImportSpec(
                            type=ProviderType.VAPI
                            if connector == "vapi"
                            else ProviderType.RETELL,
                            source_target_id=target_id,
                            public_capability=TOOLS_CAPABILITY,
                            environment_tools=agent_tool_names,
                            event_path=PROVIDER_EVENT_PATH,
                            tool_path=PROVIDER_TOOL_PATH,
                            target_modality=modality,
                            admission="per_tool",
                            request_binding=True,
                            profile_digest=binding.agent.provider_digest,
                        ).model_dump(mode="json")
                    }
                ),
                "release": {
                    "run_manifest": manifest.digest,
                    "environment": [ref.digest for ref in manifest.environment],
                    "suite": suite.digest,
                    "binding": binding.digest,
                    "binding_mode": binding.mode.value,
                },
            },
        )
        bundle = bundle.model_copy(update={"digest": seal_bundle_v2(bundle)})
        (staging / BUNDLE_V2_MANIFEST).write_text(
            json.dumps(bundle.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return bundle


# --- command line ----------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fi.alk.harness.releases")
    commands = parser.add_subparsers(dest="command", required=True)

    environment = commands.add_parser("seal-environment")
    environment.add_argument("root")
    environment.add_argument("--name", required=True)
    environment.add_argument("--extends", dest="parent_root")
    environment.add_argument("--derived-from")

    suite = commands.add_parser("seal-suite")
    suite.add_argument("root")
    suite.add_argument("--name", required=True)
    suite.add_argument("--environment", required=True)
    suite.add_argument("--origin", default=ScenarioOrigin.BASE.value)
    suite.add_argument("--profile", help="policy profile JSON file")
    suite.add_argument("--derived-from")
    suite.add_argument(
        "--postgres-dsn",
        default=os.environ.get("ALK_RELEASE_POSTGRES_DSN"),
        help="role that may create databases; proves every scenario on PostgreSQL",
    )

    verify = commands.add_parser("verify")
    verify.add_argument("root")
    verify.add_argument("--extends", dest="parent_root")

    comparison = commands.add_parser("compare")
    comparison.add_argument("first")
    comparison.add_argument("second")

    composition = commands.add_parser("compose")
    composition.add_argument("--releases", required=True)
    composition.add_argument("--binding", required=True)
    composition.add_argument("--manifest", required=True)
    composition.add_argument("--job", required=True)
    composition.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "seal-environment":
            sealed: BaseModel = seal_environment_release(
                args.root,
                name=args.name,
                parent_root=args.parent_root,
                derived_from=args.derived_from,
            )
            print(json.dumps({"digest": sealed.digest}))  # type: ignore[attr-defined]
        elif args.command == "seal-suite":
            profile = (
                _read_json(Path(args.profile), "suite_policy_profile_invalid")
                if args.profile
                else None
            )
            sealed = seal_suite_release(
                args.root,
                name=args.name,
                environment_root=args.environment,
                origin=args.origin,
                policy_profile=profile,
                derived_from=args.derived_from,
                postgres_dsn=args.postgres_dsn,
            )
            print(json.dumps({"digest": sealed.digest}))  # type: ignore[attr-defined]
        elif args.command == "verify":
            root = Path(args.root)
            if (root / ENVIRONMENT_MANIFEST).is_file():
                verified: BaseModel = load_environment_release(
                    root, parent_root=args.parent_root
                )
            elif (root / SUITE_MANIFEST).is_file():
                verified = load_suite_release(root)
            else:
                raise ReleaseError(
                    "release_unknown", f"no {ENVIRONMENT_MANIFEST} or {SUITE_MANIFEST}"
                )
            print(json.dumps({"verified": verified.digest}))  # type: ignore[attr-defined]
        elif args.command == "compose":
            bundle = compose_bundle(
                args.releases,
                binding_path=args.binding,
                manifest_path=args.manifest,
                job_path=args.job,
                output=args.output,
            )
            print(json.dumps({"bundle": bundle.digest}))
        else:
            differing = compare(
                load_run_manifest(args.first), load_run_manifest(args.second)
            )
            print(json.dumps({"comparable": not differing, "differing": differing}))
    except ReleaseError as exc:
        print(json.dumps({"error": exc.code, "detail": exc.detail}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
