"""`futureagi.environment-bundle.v2` — the hosted provisioner's manifest shape (`hosted-execution-
seams.md` v1.9).

v1 (`bundle.py`) describes a `command`-per-service compose world and embeds the repository
source. v2 describes `/work/source` as already present and a job that starts plain processes on
localhost: `processes` (managed engines and copied-and-built source trees), `seed` (how each
store's baseline is built and proven), and the same `capabilities`/`readiness`/`files`/
`provenance` shape widened for both. v1 stays untouched — this module is additive, not a
replacement, and the two schema versions are never interchangeable: a hosted provisioner that
receives a `…bundle.v1` manifest rejects it rather than guessing.

What lives here is model-layer only: the shapes, the closed vocabularies, and the rules that need
nothing but the manifest's own fields to decide. Rules that need the bundle's actual files (secret
scanning, digest/file verification) or the job it will run under (`compose_not_hosted`,
`engine_unsupported`, `no_sql_store`, the `depends_on` graph, placeholder-vocabulary checking,
reserved-name scanning of migration content) are the §2e preflight checklist's job, not this
module's — see `hosted-execution-seams.md` §2e. Also deferred to that preflight: translating
pydantic's `extra="forbid"` rejection of an unknown process-entry key into §2b's `unknown_field`
code — that translation belongs where error surfacing is owned, not here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Sequence, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

from .bundle import CapabilityProtocol, _reject_secret_values, _safe_relative

BUNDLE_V2_SCHEMA_VERSION = "futureagi.environment-bundle.v2"
BUNDLE_V2_MANIFEST = "manifest.json"


class BundleV2Error(RuntimeError):
    """A v2 bundle manifest is wrong-versioned, malformed, or fails a model-layer rule."""


# --- §2a runtime -------------------------------------------------------------------------------


class RuntimeKindV2(str, Enum):
    PROCESS = "process"
    EXTERNAL = "external"
    COMPOSE = "compose"


class EvidenceSeam(str, Enum):
    HTTP_TOOL = "http_tool"
    TOOL_TRACE = "tool_trace"


class BundleRuntimeV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: RuntimeKindV2
    control_service: str | None = None
    evidence_seam: EvidenceSeam | None = None
    # Carried over from v1 for `kind: compose` only (local SDK runs); a hosted `process` bundle
    # has no document to point at, since `/work/source` is already on disk.
    document: str | None = None

    @model_validator(mode="after")
    def _kind_specific_rules(self) -> "BundleRuntimeV2":
        if self.kind is RuntimeKindV2.PROCESS and self.evidence_seam is None:
            raise ValueError("evidence_seam_required: kind=process")
        if self.kind is RuntimeKindV2.COMPOSE and not self.document:
            raise ValueError("compose_runtime_requires_document")
        if self.kind is not RuntimeKindV2.COMPOSE and self.document is not None:
            raise ValueError("document_only_for_compose")
        if self.document:
            _safe_relative(self.document)
        return self


# --- §2b processes -------------------------------------------------------------------------


class ProcessKind(str, Enum):
    MANAGED = "managed"
    SOURCE = "source"


class ManagedEngine(str, Enum):
    POSTGRES = "postgres"
    REDIS = "redis"
    RABBITMQ = "rabbitmq"


class ProcessUser(str, Enum):
    """The snapshot's fixed, bundle-assignable users (§0). `svc-control` runs ALK itself and is
    never a process's own user."""

    SVC_AGENT = "svc-agent"
    SVC_TOOLS = "svc-tools"
    SVC_DATA = "svc-data"


class SecretPurpose(str, Enum):
    TARGET_PROVIDER = "target_provider"
    SIMULATOR_PROVIDER = "simulator_provider"
    SOURCE_CHECKOUT = "source_checkout"


# §0 (v1.8): a process `name` is path-joined into `/work/build/<name>/` and
# `/work/worlds/w<N>/<name>/` verbatim (§2b) — the pattern below is the closed shape that makes
# `/`, `..`, and an absolute form unspellable at the model layer, matching every §2b example
# (including `tools-api`).
_PROCESS_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _validate_process_name(name: str) -> str:
    if not _PROCESS_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f"process_name_invalid: {name!r} must match ^[a-z0-9][a-z0-9_-]*$"
        )
    return name


class StartedCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # §2b (v1.8): "the value selects the port-probe variant, it is not a literal port number" —
    # the probed port is always the dependency's own allocated port (`port_plan.port_for`,
    # `process_runtime.py`), honoring `fixed_port` when the process declares one. A prior version
    # of this field carried a literal int; `bool` makes the "not a literal" rule unspellable
    # wrong rather than merely documented.
    port: bool | None = None
    log_marker: str | None = None
    timeout_seconds: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def _exactly_one_probe(self) -> "StartedCheck":
        has_port = bool(self.port)
        has_marker = self.log_marker is not None
        if has_port == has_marker:
            raise ValueError("started_check_requires_exactly_one_of_port_or_log_marker")
        return self


class ManagedProcess(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: Literal[ProcessKind.MANAGED] = ProcessKind.MANAGED
    engine: ManagedEngine
    version: str = Field(min_length=1)
    user: ProcessUser
    depends_on: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_shape(cls, value: str) -> str:
        return _validate_process_name(value)


class SourceProcess(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: Literal[ProcessKind.SOURCE] = ProcessKind.SOURCE
    working_directory: str
    source_origin: Literal["repository", "bundle"] = "repository"
    build_commands: list[list[str]] = Field(default_factory=list)
    run_command: list[str] = Field(min_length=1)
    environment: dict[str, str] = Field(default_factory=dict)
    build_environment: dict[str, str] | None = None
    fixed_port: int | None = Field(default=None, ge=1, le=65535)
    started_check: StartedCheck | None = None
    secret_purposes: list[SecretPurpose] = Field(default_factory=list)
    user: ProcessUser
    depends_on: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_shape(cls, value: str) -> str:
        return _validate_process_name(value)

    @model_validator(mode="after")
    def _shape(self) -> "SourceProcess":
        _safe_relative(self.working_directory)
        for step in self.build_commands:
            if not step:
                raise ValueError("build_command_step_empty")
        return self


ProcessEntry = Annotated[
    Union[ManagedProcess, SourceProcess], Field(discriminator="kind")
]


# --- §2c seed ------------------------------------------------------------------------------


class BaselineStrategy(str, Enum):
    TEMPLATE_DATABASE = "template_database"
    DATADIR_COPY = "datadir_copy"
    EMPTY = "empty"


# The §2b catalog table. The engine a store answers to is the *capability's* protocol (resolved in
# the root validator, where `capabilities` is in scope) — a store entry carries no `engine` field
# of its own, and the sentinel's shape is a proof of that engine, not a second source for it.
_ENGINE_STRATEGIES: dict[ManagedEngine, frozenset[BaselineStrategy]] = {
    ManagedEngine.POSTGRES: frozenset(
        {BaselineStrategy.TEMPLATE_DATABASE, BaselineStrategy.DATADIR_COPY}
    ),
    ManagedEngine.REDIS: frozenset(
        {BaselineStrategy.DATADIR_COPY, BaselineStrategy.EMPTY}
    ),
    ManagedEngine.RABBITMQ: frozenset({BaselineStrategy.DATADIR_COPY}),
}


class StoreBaseline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: BaselineStrategy
    inputs_digest: str

    @model_validator(mode="after")
    def _digest_shape(self) -> "StoreBaseline":
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.inputs_digest):
            raise ValueError("inputs_digest_invalid")
        return self


class Sentinel(BaseModel):
    """A store's per-protocol read-only proof, per §2c: postgres `{query, expected}`, redis
    `{key, expected}`, rabbitmq `{queue, expected_depth}` — exactly one shape, never a mix."""

    model_config = ConfigDict(extra="forbid")

    query: str | None = None
    expected: str | None = None
    key: str | None = None
    queue: str | None = None
    expected_depth: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _one_protocol_shape(self) -> "Sentinel":
        if self.implied_engine is None:
            raise ValueError(
                "sentinel_shape_invalid: expected exactly one of "
                "postgres{query,expected}, redis{key,expected}, rabbitmq{queue,expected_depth}"
            )
        return self

    @property
    def implied_engine(self) -> ManagedEngine | None:
        postgres = self.query is not None and self.expected is not None
        redis = self.key is not None and self.expected is not None
        rabbitmq = self.queue is not None and self.expected_depth is not None
        shapes = [
            (
                postgres,
                ManagedEngine.POSTGRES,
                {self.key, self.queue, self.expected_depth},
            ),
            (redis, ManagedEngine.REDIS, {self.query, self.queue, self.expected_depth}),
            (rabbitmq, ManagedEngine.RABBITMQ, {self.query, self.key, self.expected}),
        ]
        matched = [
            engine for present, engine, others in shapes if present and others == {None}
        ]
        return matched[0] if len(matched) == 1 else None


class StoreEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability: str = Field(min_length=1)
    migrations: list[str] = Field(default_factory=list)
    seed_files: list[str] = Field(default_factory=list)
    baseline: StoreBaseline
    sentinel: Sentinel

    @model_validator(mode="after")
    def _paths(self) -> "StoreEntry":
        for relative_path in (*self.migrations, *self.seed_files):
            _safe_relative(relative_path)
        return self


class Seed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stores: list[StoreEntry] = Field(default_factory=list)


# --- §2d capabilities, readiness, files, provenance -----------------------------------------


# §2b's closed placeholder vocabulary, mirrored here (not imported from `process_preflight.py`,
# which imports this module) so a `configuration_name` can never shadow a builtin token — the
# reverse dependency direction is preflight -> model, not model -> preflight.
_RESERVED_CONFIGURATION_NAMES = {"JOB_ID", "WORLD_INDEX", "WORLD_DIR", "DB_NAME"}
_RESERVED_CONFIGURATION_PREFIX = re.compile(r"^(PORT|HOST)_")


class CapabilityV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol: CapabilityProtocol
    service: str = Field(min_length=1)
    container_port: int | None = Field(default=None, ge=1, le=65535)
    configuration_name: str | None = None

    @model_validator(mode="after")
    def _configuration_name_not_reserved(self) -> "CapabilityV2":
        # A `configuration_name` colliding with a fixed placeholder or a `{{PORT_/HOST_}}` prefix
        # would render the builtin token instead of this capability's address, with no error and
        # no way for the producer to spell the intended value (F8, p4-round1-review).
        name = self.configuration_name
        if name and (
            name in _RESERVED_CONFIGURATION_NAMES
            or _RESERVED_CONFIGURATION_PREFIX.match(name)
        ):
            raise ValueError(f"configuration_name_reserved: {name}")
        return self


class ReadinessProbeV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability: str
    path: str | None = None
    timeout_seconds: float = Field(default=120.0, gt=0, le=1800)
    interval_seconds: float = Field(default=1.0, gt=0, le=60)


class BundleFileV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    size: int = Field(ge=0)

    @model_validator(mode="after")
    def _valid_path(self) -> "BundleFileV2":
        _safe_relative(self.path)
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError(f"file_sha256_invalid: {self.path}")
        return self


class BundleProvenanceV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_kind: str
    repository: str | None = None
    commit: str | None = None
    source_digest: str
    generator: str = "fi.alk.harness"
    generator_version: str = "1"
    adopted_files: list[str] = Field(default_factory=list)
    generated_files: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _valid_source_digest(self) -> "BundleProvenanceV2":
        # Bare 64-hex, matching what `source_fingerprint` (v1's producer) actually emits — no
        # `sha256:` prefix, unlike `digest`/`inputs_digest`.
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_digest):
            raise ValueError(f"source_digest_invalid: {self.source_digest}")
        return self


# --- manifest root ---------------------------------------------------------------------------

_CAPABILITY_SLUG = re.compile(r"[a-z][a-z0-9_]*")

# §2c: a store's engine is the engine behind its capability's protocol, not a field the store
# carries itself. Only postgres/redis/amqp capabilities can host a store at all (§2c); any other
# protocol on a store's capability is a producer error §2e is left to catch.
_STORE_ENGINE_BY_PROTOCOL: dict[CapabilityProtocol, ManagedEngine] = {
    CapabilityProtocol.POSTGRES: ManagedEngine.POSTGRES,
    CapabilityProtocol.REDIS: ManagedEngine.REDIS,
    CapabilityProtocol.AMQP: ManagedEngine.RABBITMQ,
}


class EnvironmentBundleV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    digest: str
    name: str
    runtime: BundleRuntimeV2
    processes: list[ProcessEntry] = Field(default_factory=list)
    seed: Seed | None = None
    capabilities: dict[str, CapabilityV2] = Field(default_factory=dict)
    readiness: list[ReadinessProbeV2] = Field(default_factory=list)
    files: list[BundleFileV2] = Field(default_factory=list)
    provenance: BundleProvenanceV2
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_manifest(self) -> "EnvironmentBundleV2":
        if self.schema_version != BUNDLE_V2_SCHEMA_VERSION:
            raise ValueError(f"bundle_schema_unsupported: {self.schema_version}")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.digest):
            raise ValueError("bundle_digest_invalid")

        if self.runtime.kind is RuntimeKindV2.PROCESS and not self.processes:
            raise ValueError("processes_required: kind=process")
        if self.runtime.kind is RuntimeKindV2.EXTERNAL and (
            self.processes or self.seed is not None
        ):
            raise ValueError("processes_and_seed_forbidden: kind=external")

        for slug in self.capabilities:
            if not _CAPABILITY_SLUG.fullmatch(slug):
                raise ValueError(f"capability_slug_invalid: {slug}")

        process_names = Counter(process.name for process in self.processes)
        duplicated_names = sorted(
            name for name, count in process_names.items() if count > 1
        )
        if duplicated_names:
            raise ValueError("process_name_duplicate: " + ", ".join(duplicated_names))
        known_names = set(process_names)
        processes_by_name = {process.name: process for process in self.processes}

        # B3 (p3-round2-review): only `kind: process` has a `processes` array to resolve against —
        # `external` omits `processes` entirely (§2a) and `compose` addresses services through its
        # own `document`, not this array. Gating here, rather than by emptying `known_names`,
        # keeps the duplicate-name check above meaningful for every runtime kind.
        if self.runtime.kind is RuntimeKindV2.PROCESS:
            service_unresolved = {
                slug: capability.service
                for slug, capability in self.capabilities.items()
                if capability.service not in known_names
            }
            if service_unresolved:
                detail = ", ".join(
                    f"{slug}: {service}"
                    for slug, service in sorted(service_unresolved.items())
                )
                raise ValueError(f"service_unresolved: {detail}")

            control_service = self.runtime.control_service
            if control_service is not None and control_service not in known_names:
                raise ValueError(f"control_service_unresolved: {control_service}")
            if control_service is not None and isinstance(
                processes_by_name[control_service], ManagedProcess
            ):
                # §2a: control_service is the agent-side service the world handle and evidence
                # seam attach to — a datastore in that role is incoherent, and would otherwise
                # silently resolve and take svc-agent below (N9, p4-round2-review).
                raise ValueError(
                    f"control_service_unresolved: {control_service} is a managed engine, not a "
                    "source process"
                )

            # §2b/§0 (v1.6): the snapshot's SERVICE users are assigned by role, not authored —
            # the control service gets svc-agent, every other source process svc-tools, every
            # managed engine svc-data. Decidable from the manifest's own fields alone once
            # `control_service` is resolved, which is why it lands here rather than in preflight
            # (F5, p4-round1-review).
            for process in self.processes:
                if isinstance(process, ManagedProcess):
                    expected_user = ProcessUser.SVC_DATA
                elif process.name == control_service:
                    expected_user = ProcessUser.SVC_AGENT
                else:
                    expected_user = ProcessUser.SVC_TOOLS
                if process.user is not expected_user:
                    raise ValueError(
                        f"user_assignment_invalid: {process.name} must be "
                        f"{expected_user.value}, got {process.user.value}"
                    )

        names_to_slugs: dict[str, list[str]] = {}
        for slug, capability in self.capabilities.items():
            if capability.configuration_name:
                names_to_slugs.setdefault(capability.configuration_name, []).append(
                    slug
                )
        duplicated = {
            name: slugs for name, slugs in names_to_slugs.items() if len(slugs) > 1
        }
        if duplicated:
            detail = ", ".join(
                f"{name} ({', '.join(sorted(slugs))})"
                for name, slugs in sorted(duplicated.items())
            )
            raise ValueError(f"configuration_name_duplicate: {detail}")

        unresolved = {
            probe.capability
            for probe in self.readiness
            if probe.capability not in self.capabilities
        }
        if self.seed is not None:
            for store in self.seed.stores:
                if store.capability not in self.capabilities:
                    unresolved.add(store.capability)
        if unresolved:
            raise ValueError("capability_unresolved: " + ", ".join(sorted(unresolved)))

        # B1 (p3-round2-review): a capability's *declared* protocol can disagree with the process
        # actually backing it. F19 (p4-round1-review) widened this from "only capabilities with a
        # seed store" to every capability whose protocol names a managed engine — a redis
        # capability with no store entry at all (used only for a `{{...}}` address, never seeded)
        # was previously never checked, and could point `service` at a postgres process silently.
        for slug, capability in self.capabilities.items():
            engine = _STORE_ENGINE_BY_PROTOCOL.get(capability.protocol)
            if engine is None:
                continue
            backing = processes_by_name.get(capability.service)
            if isinstance(backing, ManagedProcess) and backing.engine is not engine:
                raise ValueError(
                    f"capability_engine_mismatch: {slug}: protocol {capability.protocol.value} "
                    f"resolves to {engine.value}, but {capability.service} is a "
                    f"{backing.engine.value} process"
                )

        if self.seed is not None:
            # Every store's capability resolved above, so its protocol is known — that protocol,
            # not the sentinel's own shape, is the authoritative engine (§2c classifies stores by
            # capability protocol; the sentinel only proves that engine, it doesn't select it).
            for store in self.seed.stores:
                capability = self.capabilities[store.capability]
                engine = _STORE_ENGINE_BY_PROTOCOL.get(capability.protocol)
                if engine is None:
                    # B2 (p3-round2-review): a store on a capability outside the three protocols
                    # this module knows how to seed (http, mongodb, ...) has no engine to check
                    # its sentinel/strategy against — a producer error, not a silent pass-through.
                    raise ValueError(
                        f"store_protocol_unsupported: {store.capability}: protocol "
                        f"{capability.protocol.value} cannot host a seed store"
                    )
                backing = processes_by_name.get(capability.service)
                if not isinstance(backing, ManagedProcess):
                    # F19 (p4-round1-review): a store on a capability backed by a `SourceProcess`
                    # has no managed engine to migrate or seed at all — the all-capabilities
                    # engine pass above only fires for a *wrong* managed engine, not a missing one.
                    raise ValueError(
                        f"store_service_not_managed: {store.capability}: service "
                        f"{capability.service!r} is not a managed engine"
                    )
                if store.sentinel.implied_engine is not engine:
                    raise ValueError(
                        f"sentinel_shape_mismatch: {store.capability}: sentinel implies "
                        f"{store.sentinel.implied_engine.value}, capability protocol resolves to "
                        f"{engine.value}"
                    )
                if store.baseline.strategy not in _ENGINE_STRATEGIES[engine]:
                    raise ValueError(
                        f"seed_strategy_unsupported: {store.capability}: {engine.value} does not "
                        f"support {store.baseline.strategy.value}"
                    )

        if self.seed is not None:
            # A store names its capability directly (no placeholder to resolve), so this half of
            # the configuration_name rule is decidable here; the process-`environment` half needs
            # placeholder scanning against the closed `{{...}}` vocabulary, which is §2e's job.
            missing_name = [
                store.capability
                for store in self.seed.stores
                if not self.capabilities[store.capability].configuration_name
            ]
            if missing_name:
                raise ValueError(
                    "configuration_name_required: "
                    + ", ".join(sorted(set(missing_name)))
                )

        # v1's whole-manifest resolved-secret guard (`bundle.py`), reapplied here rather than
        # dropped: v2 newly carries free-form `environment`/`build_environment` dicts, exactly
        # where a resolved credential lands if an authoring stage ever inlines one instead of
        # routing it through `secret_purposes`. `secret_purposes` itself is excluded from the
        # dump — its key matches the secret-field pattern (`secret_...`) but it holds purpose
        # identifiers, never values, the same reasoning v1 exempts `secret_refs` under.
        _reject_secret_values(
            self.model_dump(
                exclude={"digest": True, "processes": {"__all__": {"secret_purposes"}}}
            )
        )
        return self


# --- §2c inputs_digest ------------------------------------------------------------------------


def compute_inputs_digest(
    root: str | Path,
    migrations: Sequence[str],
    seed_files: Sequence[str],
    *,
    engine: ManagedEngine,
    version: str,
) -> str:
    """The byte-exact `seed.stores[].baseline.inputs_digest` construction from §2c.

    sha256 over, for each file in ``migrations`` then ``seed_files`` **in listed order** (never
    sorted — order is part of the identity, since migrations must apply in sequence),
    ``<relative_path>\\n<content_length>\\n<content_bytes>``, followed by ``<engine>:<version>\\n``.
    Runs at authoring time, against ``migrations``/``seed_files`` as bundle-relative paths under
    ``root`` (the bundle staging root) — the same paths the sealed manifest records, so the digest
    is reproducible from the bundle's own field values. ``engine``/``version`` must be the store's
    engine's own declared `ManagedProcess.engine`/`.version`, verbatim.
    """
    root = Path(root)
    digest = hashlib.sha256()
    for relative_path in (*migrations, *seed_files):
        _safe_relative(relative_path)
        content = (root / relative_path).read_bytes()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\n")
        digest.update(str(len(content)).encode("utf-8"))
        digest.update(b"\n")
        digest.update(content)
    digest.update(f"{engine.value}:{version}\n".encode("utf-8"))
    return "sha256:" + digest.hexdigest()


# --- §2d bundle digest ------------------------------------------------------------------------


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def seal_bundle_v2(manifest: EnvironmentBundleV2) -> str:
    """The byte-exact `digest` construction from §2d (v1.7) — the single normative
    implementation; producers call this, never reimplement it.

    sha256 over the canonical dump of the manifest with ``digest`` and ``files`` removed, then for
    each ``files[]`` record, IN LISTED ORDER, the canonical dump of ``{path, sha256, size}``
    prefixed by its byte length as 8 bytes big-endian. "Canonical" = ``json.dumps(...,
    sort_keys=True, separators=(",", ":"), ensure_ascii=False)``, both times. Operates on
    `BundleFileV2` and `EnvironmentBundleV2.model_dump(mode="json")` directly — v1's `BundleFile`
    never enters this construction, so a field added to v1's model cannot silently rekey a v2
    bundle's digest (F4, p4-round1-review). The hash covers the NORMALIZED dump, so adding an
    optional field to `EnvironmentBundleV2` re-keys every previously sealed bundle — sealer and
    verifier must ship together, which is exactly why there is only one implementation.
    """
    core = manifest.model_dump(mode="json")
    core.pop("digest", None)
    core.pop("files", None)
    digest = hashlib.sha256(_canonical_json(core))
    for record in manifest.files:
        encoded = _canonical_json(record.model_dump(mode="json"))
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return "sha256:" + digest.hexdigest()


def load_bundle_v2(path: str | Path) -> EnvironmentBundleV2:
    """Parse and validate one `futureagi.environment-bundle.v2` manifest.

    ``path`` is either the manifest file itself or the bundle directory containing it. Schema
    version is checked before full model validation runs, so a `…bundle.v1` manifest — or
    anything else — is named explicitly rather than failing on an unrelated field.
    """
    path = Path(path).expanduser()
    target = path if path.is_file() else path / BUNDLE_V2_MANIFEST
    if not target.is_file():
        raise BundleV2Error(f"bundle_manifest_missing: {target}")
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleV2Error(f"bundle_manifest_invalid: {exc}") from exc
    if not isinstance(raw, dict):
        raise BundleV2Error("bundle_manifest_invalid: not a JSON object")
    schema_version = raw.get("schema_version")
    if schema_version != BUNDLE_V2_SCHEMA_VERSION:
        raise BundleV2Error(f"bundle_schema_unsupported: {schema_version!r}")
    try:
        return EnvironmentBundleV2.model_validate(raw)
    except ValidationError as exc:
        raise BundleV2Error(f"bundle_manifest_invalid: {exc}") from exc


__all__ = [
    "BUNDLE_V2_MANIFEST",
    "BUNDLE_V2_SCHEMA_VERSION",
    "BaselineStrategy",
    "BundleFileV2",
    "BundleProvenanceV2",
    "BundleRuntimeV2",
    "BundleV2Error",
    "CapabilityV2",
    "EnvironmentBundleV2",
    "EvidenceSeam",
    "ManagedEngine",
    "ManagedProcess",
    "ProcessKind",
    "ProcessUser",
    "ReadinessProbeV2",
    "RuntimeKindV2",
    "Seed",
    "SecretPurpose",
    "Sentinel",
    "SourceProcess",
    "StartedCheck",
    "StoreBaseline",
    "StoreEntry",
    "compute_inputs_digest",
    "load_bundle_v2",
    "seal_bundle_v2",
]
