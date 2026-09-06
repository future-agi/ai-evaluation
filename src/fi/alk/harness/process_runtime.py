"""The execution half of the in-sandbox provisioner — `hosted-execution-seams.md` v1.12, §2b/§3/§4/§5.

Builds one world's running processes from an already-`preflight_bundle`-cleared
`EnvironmentBundleV2`: port allocation, `{{...}}` placeholder rendering, copy-based build trees,
process spawn (managed engines and `source` processes), `depends_on` wait, and `healthy()`
readiness probing. This is `provision()` UP TO process spawn only — seed/baseline application,
the world-reset mechanism, and the conformance gate (§4) are a later phase, deliberately left out
so they compose on top of `spawn_world`/`build_process_tree` rather than reworking them. The
`RuntimeProvider` Protocol itself (`runtime.py`) is untouched here; a later phase wires a
Protocol-conforming, stateful adapter around the pure functions below.

Nothing here assumes Docker, a container runtime, or a network provider — §0: "No Docker inside
the sandbox." Every engine/process concept is a plain `subprocess`, matched against the module's
own `ProcessRunner`/`CapabilityProber` seams so tests substitute fakes; the one exception
(`default_capability_prober`'s postgres branch) import-guards `psycopg` and falls back to a bare
TCP probe when it is absent, the same fallback a bundle's own store would get before a role or
database exists to authenticate against.

§0/§2b `user`: every build tree and every spawned process is chowned/spawned under its declared
`user`, resolved through `pwd.getpwnam` (`default_user_resolver`). A dev box has none of the
snapshot's `svc-*` accounts, so resolution failing is the expected local-lane shape, not an error
by itself — `require_declared_user=False` (the default) falls back to running unprivileged and
logs it; a hosted caller that wants a missing user to be a typed failure instead sets it `True`.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import pwd
import re
import secrets
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence
from urllib.parse import quote as urlquote, urlsplit

from pydantic import BaseModel, Field, JsonValue

from .bundle import CapabilityProtocol
from .bundle_v2 import (
    BaselineStrategy,
    CapabilityV2,
    EnvironmentBundleV2,
    ManagedEngine,
    ManagedProcess,
    ProcessUser,
    ReadinessProbeV2,
    SecretPurpose,
    SourceProcess,
    StoreEntry,
)
from .job import FailureDomain
from .provider_import import (
    ProviderImportError,
    ProviderImportSpec,
    clone_provider_target,
    destroy_imported_target,
)
from .provider_lifecycle import (
    ProviderContext,
    ProviderLifecycleError,
    ProviderLifecycleSpec,
    ProviderProvisionReceipt,
    build_lifecycle_invocation,
    run_lifecycle_invocation,
    validate_provision_receipt,
)

logger = logging.getLogger(__name__)

# --- errors --------------------------------------------------------------------------------


class ProcessRuntimeError(RuntimeError):
    """A runtime-execution failure below the §2e preflight gate.

    The bundle has already passed `preflight_bundle` by the time any of this module runs, so
    these are process/filesystem/timing failures. v1.8 added §2f: a CLOSED table for the subset
    of these that cross the outbound seam (`source_tree_unavailable`, `build_failed`,
    `runtime_unsupported`, `spawn_failed`, `depends_on_timeout`, `unsupported_capability_protocol`),
    each mapped to a `FailureDomain` per §4.6 — every code this module raises THAT crosses the
    seam is in that table. A handful of `code` values prefixed `internal_`
    (`internal_unknown_placeholder`, `internal_missing_credentials`) are deliberately NOT in it:
    each marks a precondition `preflight_bundle` should already have made impossible, so it is a
    bug to fix here, not a failure the outbound seam ever needs a name for. `stage` names which
    phase failed (`build`, `spawn`, `depends_on`, `render`); `process` names the process involved,
    when there is one.

    v1.15 §2f: the PRODUCER resolves `domain` — a row like `spawn_failed` splits managed-vs-source
    on a fact (the failing process's declared `kind`) only known at the raise site, so every raise
    site for a §2f code sets `domain` here rather than leaving the consumer to re-derive it from
    `code` alone (which cannot recover the managed/source split). `domain` is `None` only where
    the raise site genuinely cannot tell which process kind failed (a mixed-kind catch-all); such
    sites are rare and `SECTION_2F_DOMAIN` below exists as their fallback.
    """

    def __init__(
        self,
        stage: str,
        code: str,
        message: str,
        *,
        process: str | None = None,
        domain: FailureDomain | None = None,
    ) -> None:
        self.stage = stage
        self.code = code
        self.process = process
        self.domain = domain
        located = f" ({process})" if process else ""
        super().__init__(f"{stage}/{code}{located}: {message}")


# §2f's closed build/run failure-code table (hosted-execution-seams.md §4.6) -> FailureDomain.
# THE single home for this map (v1.15's §2f-map cleanup) — `hosted_scheduler.py`/
# `hosted_entrypoint.py` import it rather than keeping their own copies. It is a FALLBACK ONLY:
# every raise site above sets `ProcessRuntimeError.domain` directly, so a consumer reads that
# first and falls back to this map only for an error that reaches it with no carried domain.
# `spawn_failed`'s entry is the pre-v1.15 conservative guess (matches its source-process half);
# a `spawn_failed` raised with a resolved `domain` never consults this entry at all.
SECTION_2F_DOMAIN: dict[str, FailureDomain] = {
    "source_tree_unavailable": FailureDomain.ENVIRONMENT,
    "build_failed": FailureDomain.AGENT,
    "runtime_unsupported": FailureDomain.ENVIRONMENT,
    "spawn_failed": FailureDomain.INFRASTRUCTURE,
    "depends_on_timeout": FailureDomain.INFRASTRUCTURE,
    "unsupported_capability_protocol": FailureDomain.ENVIRONMENT,
    "seed_failed": FailureDomain.ENVIRONMENT,
    "store_statement_failed": FailureDomain.INFRASTRUCTURE,
}


# --- §3 EnvironmentRuntime -------------------------------------------------------------------


class RuntimeState(str, Enum):
    PREPARING = "preparing"
    READY = "ready"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"


class RuntimeEndpoint(BaseModel):
    capability: str
    protocol: str
    address: str
    configuration_name: str | None = None


class EnvironmentRuntime(BaseModel):
    """§3's per-world provisioner output. No `provider` field (§4 delta: "no remote provider
    exists at this seam") — the class attribute `RuntimeProvider.name` (`runtime.py`) is retained
    there for logging only, not carried onto each world's own record."""

    runtime_id: str
    world_index: int = Field(ge=0)
    bundle_digest: str
    state: RuntimeState
    endpoints: dict[str, RuntimeEndpoint] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


def new_runtime_id(bundle_digest: str, world_index: int) -> str:
    """Opaque per §3 ("nothing parses it") — still traceable in logs without decoding."""
    return f"{bundle_digest}:w{world_index}:{secrets.token_hex(4)}"


# --- §2b port allocation ---------------------------------------------------------------------

_PER_WORLD_BASE = 15000
_PER_WORLD_STRIDE = 100
_JOB_SHARED_BASE = 14000


def _ordinal_map(manifest: EnvironmentBundleV2) -> dict[str, int]:
    """§2b: "ordinal = the process's 0-based index in the `processes` array as authored" — one
    map, shared by both port ranges."""
    return {process.name: index for index, process in enumerate(manifest.processes)}


def _job_shared_process_names(manifest: EnvironmentBundleV2) -> frozenset[str]:
    """§2b instancing rule: a managed engine runs once per job iff some store backed by one of
    its capabilities uses `template_database`; `datadir_copy`, `empty`, or no `seed.stores` entry
    at all is per-world — per-world is the only safe default for an engine one world's reset
    could otherwise corrupt for the others. Only postgres's catalog entry permits
    `template_database` at all (`bundle_v2._ENGINE_STRATEGIES`), so this can never mark a redis
    or rabbitmq process job-shared.
    """
    service_by_capability = {
        slug: cap.service for slug, cap in manifest.capabilities.items()
    }
    strategies_by_service: dict[str, set[BaselineStrategy]] = {}
    if manifest.seed is not None:
        for store in manifest.seed.stores:
            service = service_by_capability.get(store.capability)
            if service is not None:
                strategies_by_service.setdefault(service, set()).add(
                    store.baseline.strategy
                )
    return frozenset(
        process.name
        for process in manifest.processes
        if isinstance(process, ManagedProcess)
        and BaselineStrategy.TEMPLATE_DATABASE
        in strategies_by_service.get(process.name, set())
    )


@dataclass(frozen=True)
class PortPlan:
    """A job's whole port assignment, computed once from the manifest and the requested instance
    count. `fixed_port` (§2b) forces `effective_instances` to 1 — `port_for` honors the fixed
    value exactly for that process, in every world index, which is consistent because there can
    only ever be world 0 once `effective_instances` is 1. Emitting the `parallelism_degraded`
    event for `degraded_reason` is the scheduler's job (§2b), not this module's — this only
    records the fact.
    """

    ordinals: dict[str, int]
    job_shared: frozenset[str]
    fixed_ports: dict[str, int]
    effective_instances: int
    degraded_reason: str | None

    def is_job_shared(self, process_name: str) -> bool:
        return process_name in self.job_shared

    def port_for(self, process_name: str, world_index: int) -> int:
        if process_name in self.fixed_ports:
            return self.fixed_ports[process_name]
        ordinal = self.ordinals[process_name]
        if process_name in self.job_shared:
            return _JOB_SHARED_BASE + ordinal
        return _PER_WORLD_BASE + _PER_WORLD_STRIDE * world_index + ordinal


def plan_ports(manifest: EnvironmentBundleV2, *, instances: int) -> PortPlan:
    ordinals = _ordinal_map(manifest)
    job_shared = _job_shared_process_names(manifest)
    fixed_ports = {
        process.name: process.fixed_port
        for process in manifest.processes
        if isinstance(process, SourceProcess) and process.fixed_port is not None
    }
    return PortPlan(
        ordinals=ordinals,
        job_shared=job_shared,
        fixed_ports=fixed_ports,
        effective_instances=1 if fixed_ports else instances,
        degraded_reason="fixed_port" if fixed_ports and instances > 1 else None,
    )


# --- engine credentials, generated once per job -----------------------------------------------


@dataclass(frozen=True)
class EngineCredentials:
    username: str
    password: str


def generate_engine_credentials(
    manifest: EnvironmentBundleV2, *, token: Callable[[], str] | None = None
) -> dict[str, EngineCredentials]:
    """§2b catalog: postgres/rabbitmq use role/user `harness`, "password generated per job";
    redis carries no auth in V1 and gets no entry here. Keyed by process name (not engine) since
    a job can carry more than one postgres process. `token` is injectable for deterministic
    tests; production callers rely on the `secrets` module default.
    """
    make_token = token or (lambda: secrets.token_urlsafe(24))
    return {
        process.name: EngineCredentials(username="harness", password=make_token())
        for process in manifest.processes
        if isinstance(process, ManagedProcess)
        and process.engine in (ManagedEngine.POSTGRES, ManagedEngine.RABBITMQ)
    }


def render_capability_address(
    capability: CapabilityV2,
    *,
    port: int,
    world_index: int,
    credentials: EngineCredentials | None,
) -> str:
    """§2b: `{{DATABASE_URL}}` etc. render "with the catalog role, the generated password, the
    allocated port, and `{{DB_NAME}}`." `{{DB_NAME}}` is always `w<N>` (§2b), including under
    `template_database`, where the logical database differs per world on one shared server."""
    host = "localhost"
    protocol = capability.protocol
    if protocol is CapabilityProtocol.POSTGRES:
        if credentials is None:
            # S11/P4's N8(2) precedent: a bare `assert` for a real precondition is stripped under
            # `python -O` — this is a should-never-happen internal invariant (`generate_engine_
            # credentials` always produces one for every postgres/rabbitmq `ManagedProcess`), not
            # a bundle defect, so it is `internal_`-prefixed, not a §2f code.
            raise ProcessRuntimeError(
                "render",
                "internal_missing_credentials",
                "postgres capability requires generated credentials but none were supplied",
            )
        auth = f"{credentials.username}:{credentials.password}"
        return f"postgresql://{auth}@{host}:{port}/w{world_index}"
    if protocol is CapabilityProtocol.AMQP:
        if credentials is None:
            raise ProcessRuntimeError(
                "render",
                "internal_missing_credentials",
                "amqp capability requires generated credentials but none were supplied",
            )
        return f"amqp://{credentials.username}:{credentials.password}@{host}:{port}/"
    if protocol is CapabilityProtocol.REDIS:
        return f"redis://{host}:{port}"
    if protocol is CapabilityProtocol.HTTP:
        return f"http://{host}:{port}"
    # F10, p5-round1-review: no other capability protocol (grpc, mongodb, s3, ...) has a defined
    # address shape at this seam — §3 names exactly two worked examples. A bare
    # `<scheme>://host:port` used to be handed to a customer process as its own
    # `{{CONFIGURATION_NAME}}` value, which is not a working address for any of those protocols;
    # failing loudly beats handing out an address that cannot work.
    raise ProcessRuntimeError(
        "render",
        "unsupported_capability_protocol",
        f"{protocol.value} has no defined address shape at this seam",
        domain=FailureDomain.ENVIRONMENT,
    )


def build_endpoints(
    manifest: EnvironmentBundleV2,
    *,
    world_index: int,
    port_plan: PortPlan,
    credentials: dict[str, EngineCredentials],
) -> dict[str, RuntimeEndpoint]:
    """§3's `endpoints` map, built fresh for one world — every declared capability gets an
    entry regardless of whether any process placeholder ever references its `configuration_name`
    (a null `configuration_name` is legal; the scheduler/simulator still read endpoints by
    capability slug, per §3's own reader list)."""
    endpoints: dict[str, RuntimeEndpoint] = {}
    for slug, capability in manifest.capabilities.items():
        port = port_plan.port_for(capability.service, world_index)
        address = render_capability_address(
            capability,
            port=port,
            world_index=world_index,
            credentials=credentials.get(capability.service),
        )
        endpoints[slug] = RuntimeEndpoint(
            capability=slug,
            protocol=capability.protocol.value,
            address=address,
            configuration_name=capability.configuration_name,
        )
    return endpoints


def configuration_addresses_from_endpoints(
    endpoints: dict[str, RuntimeEndpoint],
) -> dict[str, str]:
    return {
        endpoint.configuration_name: endpoint.address
        for endpoint in endpoints.values()
        if endpoint.configuration_name
    }


# --- §2b placeholder renderer ------------------------------------------------------------------

_PLACEHOLDER = re.compile(r"\{\{([^{}]+)\}\}")
_NAMED_PLACEHOLDER = re.compile(r"^(PORT|HOST)_(.+)$")


def render_template(
    value: str,
    *,
    process_name: str,
    job_id: str = "local",
    world_index: int,
    world_dir: Path,
    port_plan: PortPlan,
    configuration_addresses: dict[str, str],
) -> str:
    """§2b's closed placeholder vocabulary. `preflight_bundle` has already validated every token
    in `value` against this exact vocabulary before this ever runs, so a token this function
    cannot resolve is an internal bug, not a bundle defect — raised as `ProcessRuntimeError`,
    never one of `PreflightError`'s §2e codes (those are preflight's alone; see its own
    docstring: "crosses the outbound seam")."""

    def resolve(match: re.Match[str]) -> str:
        token = match.group(1)
        if token == "JOB_ID":
            return job_id
        if token == "WORLD_INDEX":
            return str(world_index)
        if token == "WORLD_DIR":
            return str(world_dir)
        if token == "DB_NAME":
            return f"w{world_index}"
        named = _NAMED_PLACEHOLDER.match(token)
        if named:
            kind, name = named.groups()
            if kind == "HOST":
                return "localhost"
            try:
                return str(port_plan.port_for(name, world_index))
            except KeyError:
                raise ProcessRuntimeError(
                    "render",
                    "internal_unknown_placeholder",
                    f"{{{{{token}}}}} names {name!r}, which is not in this bundle's processes",
                    process=process_name,
                ) from None
        if token in configuration_addresses:
            return configuration_addresses[token]
        raise ProcessRuntimeError(
            "render",
            "internal_unknown_placeholder",
            f"{{{{{token}}}}} has no resolution; preflight should have rejected this bundle",
            process=process_name,
        )

    return _PLACEHOLDER.sub(resolve, value)


def render_environment(
    process: SourceProcess,
    *,
    job_id: str = "local",
    world_index: int,
    world_dir: Path,
    port_plan: PortPlan,
    configuration_addresses: dict[str, str],
) -> dict[str, str]:
    """`build_environment` is deliberately excluded — §2b: it "takes NO placeholders at all," so
    it is merged raw by the env builders below, never passed through this renderer."""
    return {
        key: render_template(
            value,
            process_name=process.name,
            job_id=job_id,
            world_index=world_index,
            world_dir=world_dir,
            port_plan=port_plan,
            configuration_addresses=configuration_addresses,
        )
        for key, value in process.environment.items()
    }


def select_process_secrets(
    process: SourceProcess,
    *,
    secret_values: dict[str, str],
    secret_purposes: dict[str, str],
) -> dict[str, str]:
    """§2b: "the provisioner injects every alias whose ref's `purpose` is listed [in
    `secret_purposes`], under the **alias** as the env-var name." `secret_purposes` maps each
    job-level alias to its `SecretRef.purpose` value — the same shape `process_preflight.
    preflight_bundle`'s own `secret_refs` argument uses, so a caller that already ran preflight
    has this for free.

    `SecretPurpose.SOURCE_CHECKOUT` and `SecretPurpose.SIMULATOR_PROVIDER` are excluded
    unconditionally. Checkout credentials are gateway-only. Simulator credentials belong to the
    harness control/caller and must never enter a customer-authored process environment, even if
    an untrusted or generated bundle attempts to claim that purpose.
    """
    claimed = {
        purpose.value
        for purpose in process.secret_purposes
        if purpose
        not in {
            SecretPurpose.SOURCE_CHECKOUT,
            SecretPurpose.SIMULATOR_PROVIDER,
        }
    }
    return {
        alias: value
        for alias, value in secret_values.items()
        if secret_purposes.get(alias) in claimed
    }


# §2b enumerates exactly what a process receives: rendered `environment`, `build_environment`,
# purpose-matched secrets, and the PATH prepend below — the ambient `svc-control` environment is
# not on that list, and §1 marks the source untrusted (F12, p5-round1-review). A short, fixed
# allowlist of the interpreter/locale plumbing a process cannot run without, rather than
# `os.environ` wholesale — nothing here exports a secret today, but the entrypoint that will
# (a bearer token, a `FUTUREAGI_*` marker) is exactly the next thing wired on top of this module.
_INHERITED_ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "TZ", "TMPDIR")
_GOOGLE_ADC_JSON_ALIAS = "GOOGLE_APPLICATION_CREDENTIALS_JSON"
_GOOGLE_ADC_PATH_ALIAS = "GOOGLE_APPLICATION_CREDENTIALS"


def _allowlisted_ambient_env(source: dict[str, str]) -> dict[str, str]:
    env = {
        key: value for key, value in source.items() if key in _INHERITED_ENV_ALLOWLIST
    }
    env.update({key: value for key, value in source.items() if key.startswith("LC_")})
    return env


def _materialize_process_secret_files(
    process: SourceProcess,
    *,
    injected: dict[str, str],
    world_dir: Path,
    resolved_user: "pwd.struct_passwd | None",
    chown: Callable[[Path, int, int], None],
) -> dict[str, str]:
    """Turn file-shaped target credentials into process-local files.

    Bundle V2 executes plain processes, so a Compose mount such as
    ``/etc/vertex/creds.json`` cannot survive translation as a usable path.  The platform sends
    the uploaded service-account document under ``GOOGLE_APPLICATION_CREDENTIALS_JSON``.  Write
    it into this process's private per-world directory, replace the stale container path with the
    real path, and do not expose the raw JSON in the child environment.
    """
    raw = injected.get(_GOOGLE_ADC_JSON_ALIAS)
    if not raw:
        return injected
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProcessRuntimeError(
            "spawn",
            "spawn_failed",
            "GOOGLE_APPLICATION_CREDENTIALS_JSON is not valid JSON",
            process=process.name,
            domain=FailureDomain.AGENT,
        ) from exc
    if not isinstance(parsed, dict):
        raise ProcessRuntimeError(
            "spawn",
            "spawn_failed",
            "GOOGLE_APPLICATION_CREDENTIALS_JSON must contain an object",
            process=process.name,
            domain=FailureDomain.AGENT,
        )

    secret_dir = world_dir / ".secrets"
    # A world is spawned again after every reset.  Its scratch tree deliberately survives until
    # the reset code has restored the stores, so the private credential directory from the
    # previous spawn is expected to exist here.  Reuse that directory, but never follow a path a
    # customer process replaced with a symlink (the old process has been terminated before this
    # point).  The credential itself is recreated with O_EXCL below so a stale file or symlink
    # cannot be followed.
    if secret_dir.is_symlink():
        raise ProcessRuntimeError(
            "spawn",
            "spawn_failed",
            f"{secret_dir} must not be a symlink",
            process=process.name,
            domain=FailureDomain.INFRASTRUCTURE,
        )
    secret_dir.mkdir(mode=0o700, exist_ok=True)
    if not secret_dir.is_dir():
        raise ProcessRuntimeError(
            "spawn",
            "spawn_failed",
            f"{secret_dir} is not a directory",
            process=process.name,
            domain=FailureDomain.INFRASTRUCTURE,
        )
    secret_dir.chmod(0o700)
    credential_path = secret_dir / "google-application-credentials.json"
    credential_path.unlink(missing_ok=True)
    fd = os.open(
        credential_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(fd, raw.encode("utf-8"))
    finally:
        os.close(fd)
    if resolved_user is not None:
        chown(secret_dir, resolved_user.pw_uid, resolved_user.pw_gid)
        chown(credential_path, resolved_user.pw_uid, resolved_user.pw_gid)

    result = dict(injected)
    result.pop(_GOOGLE_ADC_JSON_ALIAS, None)
    result[_GOOGLE_ADC_PATH_ALIAS] = str(credential_path)
    return result


def _base_process_env(
    build_dir: Path,
    extra: dict[str, str] | None = None,
    *,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """§2b: `build_environment` "merged" env, plus the provisioner's own unconditional PATH
    prepend — applied last so it always wins even if `extra` (or the ambient environment) sets
    its own `PATH`, for both build and run per the contract's own words. `base` defaults to an
    allowlisted slice of the ambient environment (F12), not `os.environ` wholesale."""
    env = dict(base if base is not None else _allowlisted_ambient_env(os.environ))
    if extra:
        env.update(extra)
    prepend = [
        str(build_dir / ".venv" / "bin"),
        str(build_dir / "node_modules" / ".bin"),
    ]
    # F14, p5-round1-review: `filter(None, ...)` — an unset/empty `PATH` would otherwise leave a
    # trailing `:`, and POSIX `execvp` reads an empty PATH element as "current directory." cwd for
    # both build and run is the customer's own build tree, so that would let a repo shipping an
    # executable literally named `ls`/`git`/`sh` shadow the real one for any bare-name argv[0].
    env["PATH"] = os.pathsep.join(filter(None, [*prepend, env.get("PATH", "")]))
    return env


# --- §2b copy-based build trees -----------------------------------------------------------------

# §0 (v1.7): "a repo needing an interpreter the snapshot lacks fails at BUILD time... reported
# `runtime_unsupported`, naming what the snapshot ships." Detection surface: an exec of a build
# step's argv[0] that looks like an interpreter name (`python`, `python3`, `python3.11`, `node`,
# `node20`, ...) raising `FileNotFoundError` — anything else that fails to exec, or exits
# nonzero, is `build_failed`. This is a judgement call the contract states as a rule
# ("the build step's failure is reported `runtime_unsupported`") without naming a detection
# mechanism; §0 also promises "python 3.11 and 3.12, node 20 and 22" specifically, so the pattern
# is scoped to those two families rather than every possible interpreter name.
_INTERPRETER_PATTERN = re.compile(r"^(python\d*(\.\d+)?|node\d*)(\.exe)?$")


def _looks_like_missing_interpreter(argv0: str) -> bool:
    return bool(_INTERPRETER_PATTERN.match(Path(argv0).name))


# --- process identity: user resolution and path containment (F1/F3, p5-round1-review) ---------


def default_user_resolver(username: str) -> "pwd.struct_passwd | None":
    """§0: the base snapshot guarantees `svc-agent`/`svc-tools`/`svc-data` exist — on the hosted
    path this always resolves. A dev box has none of them; returning `None` rather than raising
    is what lets `_resolve_process_user`'s local-lane fallback work without a second seam."""
    try:
        return pwd.getpwnam(username)
    except KeyError:
        return None


def _resolve_process_user(
    user: ProcessUser,
    *,
    resolver: Callable[[str], "pwd.struct_passwd | None"],
    require: bool,
    process_name: str,
    stage: str,
    domain: FailureDomain,
) -> "pwd.struct_passwd | None":
    """F1, p5-round1-review: every build tree and every spawned process must run under its
    declared `user`, not the harness's own `svc-control` — otherwise an untrusted `agent` process
    runs with read access to the whole-attempt capabilities bearer (§0 step 4) and, since every
    process then shares one uid, mutual `/proc/<pid>/environ` visibility into every other
    process's injected secrets regardless of `secret_purposes`.

    `require=False` (the default, the local test lane's shape — no `svc-*` accounts on a dev box)
    logs and returns `None`, so the caller runs unprivileged rather than failing every local run.
    `require=True` is for a caller that knows it is on the hosted path, where the snapshot's own
    guarantee means resolution failing is itself a typed failure. `domain` is the caller's own
    §2f `spawn_failed` split (managed vs source) — this function has no process-kind knowledge of
    its own, so it never guesses it.
    """
    resolved = resolver(user.value)
    if resolved is None:
        if require:
            raise ProcessRuntimeError(
                stage,
                "spawn_failed",
                f"{user.value!r} has no passwd entry; the hosted snapshot must guarantee it",
                process=process_name,
                domain=domain,
            )
        logger.warning(
            "process %s declares user=%s but it is not resolvable on this host; running "
            "unprivileged (local test lane fallback, not the hosted path)",
            process_name,
            user.value,
        )
    return resolved


def _default_chown(path: Path, uid: int, gid: int) -> None:
    # `follow_symlinks=False`: F2 copies a within-tree symlink as a link, never dereferenced —
    # chowning through it would touch whatever it points at instead of the link itself.
    os.chown(path, uid, gid, follow_symlinks=False)


def _chown_tree(
    root: Path, *, uid: int, gid: int, chown: Callable[[Path, int, int], None]
) -> None:
    chown(root, uid, gid)
    for dirpath, dirnames, filenames in os.walk(root):
        for name in (*dirnames, *filenames):
            chown(Path(dirpath) / name, uid, gid)


def _ensure_within(path: Path, root: Path, *, process_name: str, stage: str) -> Path:
    """Defense in depth for F3, p5-round1-review. The model layer now constrains a process `name`
    to `^[a-z0-9][a-z0-9_-]*$` (`bundle_v2.py`), which already makes a `/`, `..`, or absolute name
    unspellable — a process named `/etc` used to path-join into `build_dir = Path("/etc")`
    directly, which the very next line then `rmtree`'d as `svc-control`. This is the backstop for
    anything that reaches a name-derived directory without going through that validator, checked
    right before it is handed to `rmtree`/`mkdir`. Returns the ORIGINAL `path`, not
    `path.resolve()`'d — this is a containment check, not a normalization; returning the resolved
    form would silently rewrite every caller's path (e.g. through a `/tmp` -> `/private/tmp`
    symlink on macOS) for no reason connected to the check itself.
    """
    resolved = path.resolve()
    resolved_root = root.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ProcessRuntimeError(
            stage,
            "process_name_invalid",
            f"{process_name!r} resolves to {resolved}, which escapes {resolved_root}",
            process=process_name,
        )
    return path


def _reject_escaping_symlinks(
    tree_root: Path, allowed_root: Path, *, process_name: str
) -> None:
    """F2, p5-round1-review: §2e item 2's symlink rejection is scoped to the bundle's OWN files —
    nothing scans `/work/source` for a symlink pointing outside it, since the bundle "does NOT
    embed the repository source" (§2 preamble). Left unchecked, a repo can ship
    `config/creds.json -> /run/futureagi/capabilities.json` and have the provisioner (running as
    svc-control until F1's privilege drop even applies) materialize that 0600 file's bytes into
    the customer's own build tree. Walked against the SOURCE tree before anything is copied, so a
    rejection never partially materializes a tree first. `.resolve()` (not raw `os.readlink`)
    deliberately, so a relative target or a multi-hop symlink chain is followed all the way to
    where it actually lands, not just its first hop.
    """
    for entry in tree_root.rglob("*"):
        if not entry.is_symlink():
            continue
        target = entry.resolve()
        if not target.is_relative_to(allowed_root):
            raise ProcessRuntimeError(
                "build",
                "source_tree_unavailable",
                f"{entry.relative_to(tree_root)} is a symlink to {target}, which escapes "
                "/work/source",
                process=process_name,
                domain=FailureDomain.ENVIRONMENT,
            )


def _copytree_preserving_symlinks(src: Path, dst: Path) -> None:
    # F2: `symlinks=True` — copy a link as a link rather than dereferencing it. `shutil.
    # copytree`'s default (`symlinks=False`) follows every symlink and copies the TARGET's
    # contents as a regular file, which is what let a symlink out of the checkout read as
    # svc-control in the first place. A link that escaped `/work/source` was already rejected by
    # `_reject_escaping_symlinks` before this ever runs; one that stays inside the tree is copied
    # as-is and simply works (or dangles harmlessly) under the chowned, unprivileged build tree.
    shutil.copytree(src, dst, symlinks=True)


_DEFAULT_BUILD_STEP_TIMEOUT_SECONDS = 600.0


def build_process_tree(
    process: SourceProcess,
    *,
    source_root: Path,
    build_root: Path,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    copy: Callable[[Path, Path], None] | None = None,
    user_resolver: Callable[[str], "pwd.struct_passwd | None"] = default_user_resolver,
    require_declared_user: bool = False,
    chown: Callable[[Path, int, int], None] = _default_chown,
    build_step_timeout_seconds: float = _DEFAULT_BUILD_STEP_TIMEOUT_SECONDS,
) -> Path:
    """§2b: copy `source_root/<working_directory>` to `build_root/<name>/`, chowned to the
    process's `user` (F1), then argv-exec each `build_commands` step there under that same user —
    no shell, so `&&`/`$VAR`/globs/pipes do not work, which is why the model layer
    (`bundle_v2.SourceProcess._shape`) already rejects an empty step. Runs once; the caller is
    responsible for calling this exactly once per job, per process — this function itself has no
    per-job memory. Raises on the first failing step (`ProcessRuntimeError`, stage="build"); never
    partially succeeds silently past a failure.
    """
    build_dir = build_root / process.name
    _ensure_within(build_dir, build_root, process_name=process.name, stage="build")

    source_dir = source_root / process.working_directory
    resolved_source_root = source_root.resolve()
    resolved_source_dir = source_dir.resolve()
    if not resolved_source_dir.is_relative_to(resolved_source_root):
        # F2: a symlinked PATH COMPONENT (e.g. `/work/source/services` itself a symlink to `/`)
        # used to be silently followed by the old bare `.resolve()` — `working_directory` passes
        # the model layer's `_safe_relative` (no `..`, not absolute) without ever naming the
        # escape.
        raise ProcessRuntimeError(
            "build",
            "source_tree_unavailable",
            f"{process.working_directory} resolves outside /work/source",
            process=process.name,
            domain=FailureDomain.ENVIRONMENT,
        )
    if not resolved_source_dir.is_dir():
        # F5, p5-round1-review: named explicitly, before any copy attempt, rather than letting
        # `copytree`'s own `FileNotFoundError`/`NotADirectoryError` climb out untyped.
        raise ProcessRuntimeError(
            "build",
            "source_tree_unavailable",
            f"{process.working_directory} is absent or not a directory in the checkout",
            process=process.name,
            domain=FailureDomain.ENVIRONMENT,
        )
    _reject_escaping_symlinks(
        resolved_source_dir, resolved_source_root, process_name=process.name
    )

    if build_dir.exists():
        shutil.rmtree(build_dir)
    try:
        (copy or _copytree_preserving_symlinks)(resolved_source_dir, build_dir)
    except (OSError, shutil.Error) as exc:
        # F5: `PermissionError` on an unreadable subtree, `shutil.Error`'s aggregated failure on a
        # dangling symlink — every copy-phase failure is the same deterministic, non-retryable
        # fault as a missing directory, so it gets the same code.
        raise ProcessRuntimeError(
            "build",
            "source_tree_unavailable",
            f"copying {process.working_directory}: {exc}",
            process=process.name,
            domain=FailureDomain.ENVIRONMENT,
        ) from exc

    # The source checkout is delivered read-only (the gateway chmods /work/source `a-w` for
    # integrity) and copytree preserves that mode, but the build tree is a private, writable
    # workspace — a build step such as `python -m venv .venv` must create files here. Restore
    # owner-write on the whole copy before any build_commands run.
    for _path in (build_dir, *build_dir.rglob("*")):
        try:
            _path.chmod(_path.stat().st_mode | 0o200)
        except OSError:
            pass

    resolved_user = _resolve_process_user(
        process.user,
        resolver=user_resolver,
        require=require_declared_user,
        process_name=process.name,
        stage="build",
        domain=FailureDomain.AGENT,
    )
    if resolved_user is not None:
        _chown_tree(
            build_dir, uid=resolved_user.pw_uid, gid=resolved_user.pw_gid, chown=chown
        )
    spawn_uid = resolved_user.pw_uid if resolved_user is not None else None
    spawn_gid = resolved_user.pw_gid if resolved_user is not None else None

    env = _base_process_env(build_dir, process.build_environment)
    for step in process.build_commands:
        try:
            result = run(
                step,
                cwd=build_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=build_step_timeout_seconds,
                user=spawn_uid,
                group=spawn_gid,
            )
        except subprocess.TimeoutExpired as exc:
            # F15, p5-round1-review: an install step wedged on a private registry with no DNS
            # answer used to block the provisioner forever — the only backstop was the gateway's
            # whole-job TTL, which arrives as SIGTERM to the entrypoint while this call is still
            # inside an uninterruptible `subprocess.run`.
            raise ProcessRuntimeError(
                "build",
                "build_failed",
                f"{step!r} exceeded the {build_step_timeout_seconds}s build-step timeout",
                process=process.name,
                domain=FailureDomain.AGENT,
            ) from exc
        except FileNotFoundError as exc:
            if not build_dir.is_dir():
                # JC1 ruling (p5-round1-review): a vanished build tree plus a `python*`/`node*`
                # step raises the exact same `FileNotFoundError` as a missing interpreter — this
                # disambiguates a filesystem fault from an interpreter-availability one before the
                # argv[0] heuristic below ever gets a say.
                raise ProcessRuntimeError(
                    "build",
                    "source_tree_unavailable",
                    f"{build_dir} vanished before {step!r} could run",
                    process=process.name,
                    domain=FailureDomain.ENVIRONMENT,
                ) from exc
            if _looks_like_missing_interpreter(step[0]):
                raise ProcessRuntimeError(
                    "build",
                    "runtime_unsupported",
                    f"{step[0]!r} is not on the snapshot's PATH; the snapshot ships python "
                    "3.11/3.12/3.13 and node 20/22 only",
                    process=process.name,
                    domain=FailureDomain.ENVIRONMENT,
                ) from exc
            raise ProcessRuntimeError(
                "build",
                "build_failed",
                f"{step!r}: {exc}",
                process=process.name,
                domain=FailureDomain.AGENT,
            ) from exc
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()[:2000]
            raise ProcessRuntimeError(
                "build",
                "build_failed",
                f"{step!r} exited {result.returncode}"
                + (f": {stderr}" if stderr else ""),
                process=process.name,
                domain=FailureDomain.AGENT,
            )
    return build_dir


# --- process spawn -------------------------------------------------------------------------------


class SpawnedProcess(Protocol):
    """What both a real `subprocess.Popen` wrapper and a test fake must provide — enough for
    `depends_on`'s `log_marker` variant and for cleanup, nothing engine-specific.

    M7, p6-review-r1: `terminate()` alone only SENDS a signal — it does not wait for the process
    to actually exit, which several callers used to assume by running a `copytree`/`rmtree`/port-
    reuse on the very next line. `wait()`/`kill()` let `_terminate_and_wait` (below) make that
    wait real, with a hard escalation for a process that ignores the polite signal.
    """

    def is_running(self) -> bool: ...

    def captured_output(self) -> str: ...

    def terminate(self) -> None: ...

    def interrupt(self) -> None:
        """N12, p6-review-r2: SIGINT — postgres's own FAST shutdown (`pg_ctl -m fast`): rolls
        back in-flight transactions and disconnects clients immediately, unlike `terminate()`'s
        SIGTERM, which postgres treats as a SMART shutdown that waits for every client to
        disconnect on its own. `_terminate_and_wait` prefers this for a postgres handle so tearing
        one down while its own dependents still hold connections does not have to eat the full
        kill escalation every time.
        """
        ...

    def wait(self, timeout: float) -> bool:
        """Blocks up to `timeout` seconds for the process to exit. Returns whether it did."""
        ...

    def kill(self) -> None: ...


class ProcessRunner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
        user: int | None = None,
        group: int | None = None,
    ) -> SpawnedProcess: ...


class CapabilityProber(Protocol):
    def __call__(
        self,
        *,
        protocol: CapabilityProtocol,
        host: str,
        port: int,
        path: str | None,
        user: str | None = None,
        password: str | None = None,
        dbname: str | None = None,
    ) -> bool: ...


@dataclass
class PopenProcess:
    """`SpawnedProcess` backed by a real subprocess. stdout+stderr are redirected to a log file
    under the process's own scratch directory rather than captured through a pipe-reading thread
    — simpler, and `started_check`'s `log_marker` only ever needs to re-read the file's current
    contents, not race a live stream."""

    popen: subprocess.Popen
    log_path: Path

    def is_running(self) -> bool:
        return self.popen.poll() is None

    def captured_output(self) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def terminate(self) -> None:
        self._signal_process_group(signal.SIGTERM, self.popen.terminate)

    def interrupt(self) -> None:
        self._signal_process_group(
            signal.SIGINT, lambda: self.popen.send_signal(signal.SIGINT)
        )

    def wait(self, timeout: float) -> bool:
        try:
            self.popen.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False

    def kill(self) -> None:
        self._signal_process_group(signal.SIGKILL, self.popen.kill)

    def _signal_process_group(self, signum: int, fallback: Callable[[], None]) -> None:
        """Stop the complete runtime process, including launcher descendants.

        Source commands commonly use launchers such as ``uv run``, ``npm``, or shell
        scripts. Signalling only the launcher leaves its actual server orphaned and a
        subsequent world reset cannot bind the same port. ``default_process_runner``
        gives every process its own session, so its pid is also the process-group id.
        Keep the direct-Popen fallback for synthetic handles and platforms without
        ``killpg``.
        """
        try:
            os.killpg(self.popen.pid, signum)
        except (AttributeError, OSError):
            fallback()


_TERMINATE_WAIT_SECONDS = 5.0
# Q12, p6-review-r2/-r3: rabbitmq's own broker shutdown routinely exceeds the 5s default — used
# only for the pre-`datadir_copy`-snapshot terminate, where a SIGKILL mid-mnesia-write would seal
# a corrupt baseline every world then restores from. R5, p6-review-r4: not a 30s ceiling — a
# wedged broker costs this value TWICE (`_terminate_and_wait` re-applies it to the post-kill()
# reap wait too), so the worst case at that call site is 60s.
_RABBITMQ_TERMINATE_WAIT_SECONDS = 30.0


def _terminate_and_wait(
    handle: SpawnedProcess,
    *,
    timeout: float = _TERMINATE_WAIT_SECONDS,
    prefer_interrupt: bool = False,
) -> None:
    """M7, p6-review-r1: `terminate()` alone only sends SIGTERM — postgres treats it as a SMART
    shutdown that may not have even started, let alone finished, so the very next line at several
    call sites (a `copytree` baseline snapshot, an `rmtree` before a port is reused) used to run
    against a data directory a still-live server could still be writing to, or tried to rebind a
    port the dying process had not yet released. Escalates to `kill()` only once `timeout` has
    passed with no exit — a real SIGKILL, not a repeated SIGTERM, since a process that ignored the
    first one is not going to notice a second. Never raises: a handle that is already gone, or
    whose OS-level terminate/kill/wait itself errors, must not block whatever cleanup is calling
    this (`close()`'s own idempotency, m10, is what actually made this matter).

    N12, p6-review-r2: `prefer_interrupt` swaps the FIRST signal from SIGTERM to SIGINT (postgres's
    fast shutdown) — the caller decides this per-engine, since this function has no engine
    knowledge of its own.
    """
    try:
        if prefer_interrupt:
            handle.interrupt()
        else:
            handle.terminate()
    except OSError:
        pass
    if handle.wait(timeout):
        return
    try:
        handle.kill()
    except OSError:
        pass
    # N13, p6-review-r2: the return here used to be discarded — a child still not reaped after
    # SIGKILL (a wedged kernel wait, not something any further signal can fix) left no trace
    # anywhere. Nothing past SIGKILL to escalate TO, but silently swallowing that fact is strictly
    # worse than a log line a caller could act on (flag the sandbox instead of assuming cleanup
    # succeeded).
    if not handle.wait(timeout):
        logger.warning("_terminate_and_wait: process did not exit even after kill()")


def _prefers_interrupt(manifest: EnvironmentBundleV2, process_name: str) -> bool:
    """N12, p6-review-r2: postgres is the one engine whose fast-shutdown signal genuinely differs
    from a plain SIGTERM (see `SpawnedProcess.interrupt`) — every other process (`source` or
    redis/rabbitmq) just gets the ordinary `terminate()` path."""
    processes_by_name = {process.name: process for process in manifest.processes}
    process = processes_by_name.get(process_name)
    return (
        isinstance(process, ManagedProcess) and process.engine is ManagedEngine.POSTGRES
    )


def _default_log_chown(path: Path, uid: int, gid: int) -> None:
    os.chown(path, uid, gid)


def default_process_runner(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    user: int | None = None,
    group: int | None = None,
    chown: Callable[[Path, int, int], None] = _default_log_chown,
) -> PopenProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("wb")
    if user is not None:
        # F1, p5-round1-review: the log file is created by the harness (svc-control) before the
        # child's privilege drop — the child can still WRITE through the inherited fd regardless
        # of on-disk mode (permission checks happen at open(), not at write()), but leaving it
        # svc-control-owned means nothing running as the child's own user could ever open it
        # fresh afterward. Chowned to match, not just handed over unchecked. `chown` is injectable
        # like every other chown call in this module (`os.chown` to a foreign uid needs root/
        # CAP_CHOWN — the same privilege `user=`/`group=` below already requires, so a real
        # hosted deployment has both or neither; a dev-box test fakes it structurally).
        chown(log_path, user, group if group is not None else -1)
    popen = subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        user=user,
        group=group,
        # Each declared runtime process owns a process group. This is necessary for
        # deterministic reset/cleanup when the command is a launcher which forks the
        # real long-lived server (for example ``uv run`` or ``npm start``).
        start_new_session=True,
    )
    return PopenProcess(popen=popen, log_path=log_path)


@dataclass
class SpawnedWorldProcess:
    process_name: str
    handle: SpawnedProcess
    port: int
    world_index: int | None  # None for a job-shared managed engine
    # B2, p6-review-r1: the resolved spawn identity, carried on the handle so a caller that needs
    # to run something ELSE (a seed file) as this same process's user does not have to re-resolve
    # it a second time. `None` in the local-lane fallback, same as `spawn_uid`/`spawn_gid` above.
    uid: int | None = None
    gid: int | None = None
    # The voice dispatch identity (LIVEKIT_AGENT_NAME) exists only in the process's rendered
    # per-world environment, which is discarded after spawn — carried here so the provider can
    # surface it on the world's runtime metadata without re-rendering templates.
    dispatch_agent_name: str | None = None


_AGENT_NAME_SOURCE_DEFAULT = re.compile(
    r"agent_name\s*=\s*os\.(?:environ\.get|getenv)\(\s*"
    r"[\"']LIVEKIT_AGENT_NAME[\"']\s*,\s*[\"'](?P<default>[^\"']+)[\"']"
)


def _agent_name_source_default(source_root: Path) -> str:
    """Recover the agent's source-declared LIVEKIT_AGENT_NAME default when the bundle process
    env does not render one (agent reads os.environ.get("LIVEKIT_AGENT_NAME", "<default>")).
    Mirrors provision._AGENT_NAME_SETTING so a freshly authored bundle carries the dispatch
    identity the same way a cached bundle does."""
    try:
        if not source_root.is_dir():
            return ""
        for path in source_root.rglob("*.py"):
            try:
                match = _AGENT_NAME_SOURCE_DEFAULT.search(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue
            if match:
                return match.group("default").strip()
    except OSError:
        pass
    return ""


def _dispatch_metadata(
    handles: "dict[str, SpawnedWorldProcess]",
) -> dict[str, "JsonValue"]:
    """A world with exactly one voice agent gets its dispatch identity on the runtime metadata;
    anything else leaves the key absent so the call runner's own typed pre-dial failure fires
    instead of a call being dialed at an arbitrarily chosen agent. Ambiguity is loud here
    because the pre-dial message cannot say WHY the key is missing."""
    names = sorted(
        {
            name
            for handle in handles.values()
            if (name := (handle.dispatch_agent_name or "").strip())
        }
    )
    if len(names) == 1:
        return {"livekit_agent_name": names[0]}
    if len(names) > 1:
        logger.warning(
            "world declares %d distinct LIVEKIT_AGENT_NAME values (%s); "
            "leaving dispatch identity unset",
            len(names),
            ", ".join(names),
        )
    return {}


# --- managed-engine launch commands ---------------------------------------------------------
#
# §2b fixes the catalog (engine/version/role/db-name/strategies) and the port formula; it does
# not fix an exact launch invocation for any of the three engines — that is provisioner-internal,
# same status as `LocalComposeRuntimeProvider`'s own Compose mechanics. The commands below are a
# defensible V1 default, kept swappable through `ProcessRunner`/`sync_run` injection; they are
# never executed in this module's own test lane (§0: assumed on PATH, never required in tests).


def postgres_bootstrap_argv(
    *, data_dir: Path, credentials: EngineCredentials, pwfile: Path
) -> list[str]:
    return [
        "initdb",
        "-D",
        str(data_dir),
        "-U",
        credentials.username,
        "--pwfile",
        str(pwfile),
        "-A",
        "scram-sha-256",
    ]


def postgres_daemon_argv(*, data_dir: Path, port: int) -> list[str]:
    # n1, p6-review-r1: every connection this module makes is TCP `-h localhost` (seed, sentinel,
    # canary, probes) — nothing ever dials the unix socket, so `-k ""` closes that listening
    # surface entirely instead of leaving it open at `data_dir` unused.
    return [
        "postgres",
        "-D",
        str(data_dir),
        "-p",
        str(port),
        "-k",
        "",
        "-h",
        "localhost",
    ]


def redis_daemon_argv(*, data_dir: Path, port: int) -> list[str]:
    return [
        "redis-server",
        "--port",
        str(port),
        "--dir",
        str(data_dir),
        "--daemonize",
        "no",
        "--save",
        "",
    ]


def rabbitmq_daemon_argv() -> list[str]:
    return ["rabbitmq-server"]


def rabbitmq_enabled_plugins_text() -> str:
    """M8, p6-review-r1: the Erlang term-list format `rabbitmq-server` reads at boot to decide
    which plugins are on. `rabbitmq_daemon_env` sets node/auth/data-dir env vars only — nothing
    turns the management app (the HTTP API `rabbitmqadmin` seeding and the sentinel/canary queue-
    depth reads both depend on) on; without this file it is off and every one of those calls
    connection-refuses."""
    return "[rabbitmq_management].\n"


def rabbitmq_conf_text(*, management_port: int, credentials: EngineCredentials) -> str:
    """N5, p6-review-r2 (MAJOR): a BARE `rabbitmq-server` (§0: no Docker in the sandbox) reads
    `default_user`/`default_pass` from THIS file — `RABBITMQ_DEFAULT_USER`/`RABBITMQ_DEFAULT_PASS`
    (`rabbitmq_daemon_env`, below) are a convention the official Docker image's ENTRYPOINT
    translates onto these same keys; nothing performs that translation for a directly-`exec`'d
    `rabbitmq-server`, so without this the node initializes with the built-in `guest` account
    instead of the catalog's declared `harness` role (§2b) — every rabbitmq call this module makes
    (seed, sentinel, canary) authenticates as `harness` and would 401 against a node that never
    heard of it. Q4, p6-review-r3: no `loopback_users` line here — `loopback_users = none` does
    the OPPOSITE of what a prior version of this docstring claimed (it widens `guest`'s reach to
    the whole network, RabbitMQ's own default `[guest]` restricts it to loopback already); with
    `default_user = harness` a fresh node never creates a `guest` account at all, so the setting
    buys nothing either way and is left out rather than left wrong.

    Also pins the management HTTP listener to this module's own `+10000`-offset port
    (`_rabbitmq_management_port`) — the plugin's own built-in default is 15672, which sits inside
    §2b's per-world process port band `[15000,15799]` and would alias an allocated process port;
    binding `127.0.0.1` matches every other engine, which is localhost-only in V1.
    """
    return (
        f"default_user = {credentials.username}\n"
        f"default_pass = {credentials.password}\n"
        f"management.tcp.port = {management_port}\n"
        "management.tcp.ip = 127.0.0.1\n"
    )


def rabbitmq_daemon_env(
    *, data_dir: Path, port: int, credentials: EngineCredentials
) -> dict[str, str]:
    """`RABBITMQ_DEFAULT_USER`/`RABBITMQ_DEFAULT_PASS` are kept here as harmless redundancy (they
    are exactly what the Docker image's entrypoint would consume, if the snapshot ever turned out
    to wrap one after all — N5's own open question for the snapshot owner) — `rabbitmq_conf_text`
    above is the credential source a BARE server actually reads.

    Q2, p6-review-r3 (MAJOR); R2, p6-review-r4: both `RABBITMQ_MNESIA_DIR` (a fixed, node-name-free
    path) and `RABBITMQ_MNESIA_BASE` (`data_dir` itself) are set. `MNESIA_DIR` wins for the data
    path, so a `datadir_copy` baseline snapshotted at world 0's port no longer sits under a path
    only world 0's port-derived node name would look in. `MNESIA_BASE` still has to stay set:
    rabbitmq derives OTHER paths from the base, not the dir — `RABBITMQ_PLUGINS_EXPAND_DIR` among
    them — so dropping it moves those outside `data_dir`, to the installation default the spawn
    user cannot write, and every rabbitmq spawn fails to boot (bootstrap included).

    KNOWN DEFECT (R1, p6-review-r4 — disclosed, not redesigned this phase): the mnesia DATA PATH
    above is node-name-free, but the mnesia SCHEMA on disk is not — `schema.DAT` and rabbitmq's own
    cluster-status file both record the node name (`RABBITMQ_NODENAME` below) the tables were
    created under, and per-world node names must stay distinct (every world's broker registers with
    the same epmd in the same sandbox). A `datadir_copy` baseline therefore boots cleanly in world 0
    only; every other world points its node at a directory holding world 0's node-bound schema,
    which the broker will not adopt — expect a boot/readiness failure surfacing as
    `depends_on_timeout` from `_wait_for_store_ready`, not a silent empty node. Recorded
    follow-up: replace the datadir copy with a definitions-export/import restore through the
    existing `default_rabbitmq_definitions_importer` seam, which is per-world safe by
    construction."""
    return {
        "RABBITMQ_NODE_PORT": str(port),
        "RABBITMQ_MNESIA_DIR": str(data_dir / "mnesia"),
        "RABBITMQ_MNESIA_BASE": str(data_dir),
        "RABBITMQ_LOG_BASE": str(data_dir),
        "RABBITMQ_DEFAULT_USER": credentials.username,
        "RABBITMQ_DEFAULT_PASS": credentials.password,
        "RABBITMQ_NODENAME": f"harness-{port}@localhost",
    }


def spawn_managed_process(
    process: ManagedProcess,
    *,
    port: int,
    data_dir: Path,
    credentials: EngineCredentials | None,
    runner: ProcessRunner = default_process_runner,
    sync_run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    user_resolver: Callable[[str], "pwd.struct_passwd | None"] = default_user_resolver,
    require_declared_user: bool = False,
    chown: Callable[[Path, int, int], None] = _default_chown,
) -> SpawnedWorldProcess:
    """Spawns one managed-engine daemon. Postgres alone needs a one-time synchronous bootstrap
    (`initdb`) before the daemon can start at all — run through `sync_run`, not `runner`, since
    it must complete before the daemon exec happens, not run alongside it. Redis/RabbitMQ
    initialize their own data directory on first boot and need no separate step.

    `data_dir` is chowned to the engine's declared `user` (`svc-data`, per §2b) before anything
    runs in it (F1, p5-round1-review) — otherwise a daemon spawned under that user could not even
    write its own data directory, which the harness (running as `svc-control`) created.
    """
    resolved_user = _resolve_process_user(
        process.user,
        resolver=user_resolver,
        require=require_declared_user,
        process_name=process.name,
        stage="spawn",
        domain=FailureDomain.INFRASTRUCTURE,
    )
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        # postgres demands mode 0700 on the directory itself regardless of engine; harmless for
        # redis/rabbitmq, so applied uniformly rather than special-cased per engine. N14,
        # p6-review-r2: BEFORE the chown below, not after — `svc-control` still owns `data_dir` at
        # this point and the chmod is free; once ownership has moved to the engine's own uid, the
        # SAME call would raise `PermissionError` for a `svc-control` that can chown but is not
        # itself root.
        data_dir.chmod(0o700)
        if resolved_user is not None:
            # B6, p6-review-r1: a single top-level `chown` was fine for a FRESH, empty data_dir
            # (the daemon's own first-boot `initdb` then creates everything else as `spawn_uid`
            # itself) but wrong the moment this same call runs against a data_dir a COPY just
            # populated (`freeze_baseline`'s snapshot restore, `_seal_world_store`'s `datadir_copy`
            # reseal) — the copy runs as the provisioner, so every file underneath stayed
            # provisioner-owned and postgres refuses to start against a data directory it does not
            # fully own.
            _chown_tree(
                data_dir,
                uid=resolved_user.pw_uid,
                gid=resolved_user.pw_gid,
                chown=chown,
            )
    except (OSError, shutil.Error) as exc:
        # N9, p6-review-r2: §4.6 — "engine/process/filesystem failures during provisioning ->
        # infrastructure." mkdir/chmod/chown on this engine's own data directory used to raise
        # bare, giving `provision()`'s caller nothing to map; every failure this module's own
        # data-dir setup can hit is the same infrastructure class B5 already typed for the store
        # seams themselves.
        raise ProcessRuntimeError(
            "spawn",
            "spawn_failed",
            f"{process.name}: preparing {data_dir}: {exc}",
            process=process.name,
            domain=FailureDomain.INFRASTRUCTURE,
        ) from exc
    spawn_uid = resolved_user.pw_uid if resolved_user is not None else None
    spawn_gid = resolved_user.pw_gid if resolved_user is not None else None

    env = _allowlisted_ambient_env(os.environ)
    if process.engine is ManagedEngine.POSTGRES:
        if credentials is None:
            raise ProcessRuntimeError(
                "spawn",
                "spawn_failed",
                "postgres requires generated credentials",
                process=process.name,
                domain=FailureDomain.INFRASTRUCTURE,
            )
        if not (data_dir / "PG_VERSION").exists():
            pwfile = data_dir.parent / f".{process.name}.pwfile"
            # F7, p5-round1-review: created at 0600 ATOMICALLY. `write_text` then `chmod` creates
            # the file at `0666 & ~umask` (typically 0644) and only narrows it afterward — a
            # classic create-then-chmod TOCTOU that leaves the generated superuser password
            # world-readable for the window in between. `O_EXCL` additionally refuses to follow a
            # pre-planted symlink/file already sitting at this path.
            fd = os.open(pwfile, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(credentials.password + "\n")
                if resolved_user is not None:
                    # `initdb` itself runs as `resolved_user` below (via `sync_run`'s `user=`) —
                    # it must be able to READ its own 0600 file, so ownership follows the same
                    # user, never the mode widening to let anyone else read it too.
                    chown(pwfile, resolved_user.pw_uid, resolved_user.pw_gid)
                bootstrap_argv = postgres_bootstrap_argv(
                    data_dir=data_dir, credentials=credentials, pwfile=pwfile
                )
                result = sync_run(
                    bootstrap_argv,
                    capture_output=True,
                    text=True,
                    user=spawn_uid,
                    group=spawn_gid,
                )
                if result.returncode != 0:
                    stderr = (result.stderr or "").strip()[:2000]
                    raise ProcessRuntimeError(
                        "spawn",
                        "spawn_failed",
                        f"initdb exited {result.returncode}: {stderr}",
                        process=process.name,
                        domain=FailureDomain.INFRASTRUCTURE,
                    )
            finally:
                pwfile.unlink(missing_ok=True)
        argv = postgres_daemon_argv(data_dir=data_dir, port=port)
    elif process.engine is ManagedEngine.REDIS:
        argv = redis_daemon_argv(data_dir=data_dir, port=port)
    elif process.engine is ManagedEngine.RABBITMQ:
        if credentials is None:
            raise ProcessRuntimeError(
                "spawn",
                "spawn_failed",
                "rabbitmq requires generated credentials",
                process=process.name,
                domain=FailureDomain.INFRASTRUCTURE,
            )
        # M8, p6-review-r1: written fresh on every spawn (job bootstrap AND every world's own
        # instance) — cheap, and means a `datadir_copy` restore can never carry a stale plugin/
        # port config forward from whatever was on disk when the baseline was snapshotted.
        plugins_path = data_dir / "enabled_plugins"
        conf_path = data_dir / "rabbitmq.conf"
        try:
            plugins_path.write_text(rabbitmq_enabled_plugins_text(), encoding="utf-8")
            # N5, p6-review-r2: 0600 on CREATE — this file carries the generated password in
            # cleartext (`rabbitmq_conf_text`). Plain `O_TRUNC`, not `O_EXCL`: unlike the postgres
            # pwfile, `data_dir` legitimately already HAS this file on a `datadir_copy` restore
            # (it was copied in from the baseline snapshot along with everything else), and the
            # mode set at CREATE time already carried through that copy (`shutil.copytree`'s
            # default `copy2` preserves permission bits) — `O_EXCL` would just raise
            # `FileExistsError` on every world after the first.
            fd = os.open(conf_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(
                    rabbitmq_conf_text(
                        management_port=_rabbitmq_management_port(port),
                        credentials=credentials,
                    )
                )
            if resolved_user is not None:
                chown(plugins_path, resolved_user.pw_uid, resolved_user.pw_gid)
                chown(conf_path, resolved_user.pw_uid, resolved_user.pw_gid)
        except OSError as exc:
            raise ProcessRuntimeError(
                "spawn",
                "spawn_failed",
                f"{process.name}: writing rabbitmq config: {exc}",
                process=process.name,
                domain=FailureDomain.INFRASTRUCTURE,
            ) from exc
        env.update(
            rabbitmq_daemon_env(data_dir=data_dir, port=port, credentials=credentials)
        )
        env["RABBITMQ_ENABLED_PLUGINS_FILE"] = str(plugins_path)
        # N6, p6-review-r2: the FULL path, extension included. Modern RabbitMQ (the catalog's
        # 3.13) documents `RABBITMQ_CONFIG_FILE` carrying `.conf` itself and unambiguously accepts
        # it that way — the pre-3.7 classic-config "the server appends .conf for you" behavior
        # this used to depend on is not how the catalog's version works, and the failure was
        # silent: the file on disk is `rabbitmq.conf`, the env var said `rabbitmq`, and if the
        # append never happens the server falls back to defaults — including the bare management
        # port 15672, which sits inside §2b's own per-world port band.
        env["RABBITMQ_CONFIG_FILE"] = str(conf_path)
        argv = rabbitmq_daemon_argv()
    else:  # pragma: no cover - ManagedEngine is closed; unreachable past the model layer.
        raise ProcessRuntimeError(
            "spawn",
            "spawn_failed",
            f"unknown engine {process.engine!r}",
            process=process.name,
            domain=FailureDomain.INFRASTRUCTURE,
        )
    try:
        handle = runner(
            argv,
            cwd=data_dir,
            env=env,
            log_path=data_dir / "process.log",
            user=spawn_uid,
            group=spawn_gid,
        )
    except FileNotFoundError as exc:
        raise ProcessRuntimeError(
            "spawn",
            "spawn_failed",
            str(exc),
            process=process.name,
            domain=FailureDomain.INFRASTRUCTURE,
        ) from exc
    return SpawnedWorldProcess(
        process_name=process.name,
        handle=handle,
        port=port,
        world_index=None,
        uid=spawn_uid,
        gid=spawn_gid,
    )


def spawn_source_process(
    process: SourceProcess,
    *,
    build_dir: Path,
    world_dir: Path,
    world_index: int,
    job_id: str = "local",
    port_plan: PortPlan,
    configuration_addresses: dict[str, str],
    secret_values: dict[str, str],
    secret_purposes: dict[str, str],
    runner: ProcessRunner = default_process_runner,
    user_resolver: Callable[[str], "pwd.struct_passwd | None"] = default_user_resolver,
    require_declared_user: bool = False,
    chown: Callable[[Path, int, int], None] = _default_chown,
) -> SpawnedWorldProcess:
    """§2b: `run_command` "exec'd once per world with cwd `/work/build/<name>/`" — the build tree
    is read-only by convention at run time; `world_dir` (`{{WORLD_DIR}}`) is this process's own
    per-world writable scratch, created here since nothing upstream of spawn needs it to exist.
    Chowned to the process's declared `user` before spawn (F1, p5-round1-review) — otherwise a
    process running as anyone but the harness's own uid could not write into its own `{{WORLD_
    DIR}}` at all, since the harness (as `svc-control`) is the one that just created it.
    """
    resolved_user = _resolve_process_user(
        process.user,
        resolver=user_resolver,
        require=require_declared_user,
        process_name=process.name,
        stage="spawn",
        domain=FailureDomain.AGENT,
    )
    trace_bootstrap: Path | None = None
    try:
        world_dir.mkdir(parents=True, exist_ok=True)
        if "HARNESS_TOOL_TRACE" in process.environment:
            trace_bootstrap = world_dir / "sitecustomize.py"
            # Reset reuses this agent-owned directory. The hook is immutable for the job,
            # so preserve an existing copy instead of overwriting a 0444 file after ownership
            # has already moved from svc-control to svc-agent.
            if not trace_bootstrap.exists():
                trace_bootstrap.write_bytes(
                    Path(__file__)
                    .with_name("livekit_tool_trace_bootstrap.py")
                    .read_bytes()
                )
                trace_bootstrap.chmod(0o444)
        if resolved_user is not None:
            chown(world_dir, resolved_user.pw_uid, resolved_user.pw_gid)
    except OSError as exc:
        # N9, p6-review-r2: same class as `spawn_managed_process`'s own data-dir setup boundary —
        # a permission error creating/chowning this process's `{{WORLD_DIR}}` used to raise bare.
        raise ProcessRuntimeError(
            "spawn",
            "spawn_failed",
            f"{process.name}: preparing {world_dir}: {exc}",
            process=process.name,
            domain=FailureDomain.AGENT,
        ) from exc
    rendered = render_environment(
        process,
        job_id=job_id,
        world_index=world_index,
        world_dir=world_dir,
        port_plan=port_plan,
        configuration_addresses=configuration_addresses,
    )
    injected = select_process_secrets(
        process, secret_values=secret_values, secret_purposes=secret_purposes
    )
    injected = _materialize_process_secret_files(
        process,
        injected=injected,
        world_dir=world_dir,
        resolved_user=resolved_user,
        chown=chown,
    )
    # Capability addresses are provisioner-owned runtime wiring, not customer secrets. A pasted
    # development .env commonly contains values such as ``TOOLS_API_URL=http://harness:8787``;
    # target-provider secret injection used to overwrite the rendered process-runtime endpoint
    # with that stale Compose hostname. In the Dockerless hosted lane this made a healthy local
    # tools process unreachable and crashed the voice worker on its first tool lookup. Preserve
    # the ordinary secret precedence, then make only the capability variables actually declared
    # by this process authoritative again (the same invariant as the Compose provisioner).
    authoritative_endpoints = {
        name: address
        for name, address in configuration_addresses.items()
        if name in process.environment
    }
    env = _base_process_env(
        build_dir,
        {
            **(process.build_environment or {}),
            **rendered,
            **injected,
            **authoritative_endpoints,
        },
    )
    command = list(process.run_command)
    if trace_bootstrap is not None:
        # Python imports ``sitecustomize`` at interpreter startup. Put the ALK-owned hook first
        # on PYTHONPATH so it is installed in the parent worker and every LiveKit job child.
        # Preserve any repository-supplied path after it.
        existing_pythonpath = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(world_dir), existing_pythonpath) if part
        )
    try:
        handle = runner(
            command,
            cwd=build_dir,
            env=env,
            log_path=world_dir / "process.log",
            user=resolved_user.pw_uid if resolved_user is not None else None,
            group=resolved_user.pw_gid if resolved_user is not None else None,
        )
    except FileNotFoundError as exc:
        raise ProcessRuntimeError(
            "spawn",
            "spawn_failed",
            str(exc),
            process=process.name,
            domain=FailureDomain.AGENT,
        ) from exc
    port = port_plan.port_for(process.name, world_index)
    return SpawnedWorldProcess(
        process_name=process.name,
        handle=handle,
        port=port,
        world_index=world_index,
        uid=resolved_user.pw_uid if resolved_user is not None else None,
        gid=resolved_user.pw_gid if resolved_user is not None else None,
        dispatch_agent_name=rendered.get("LIVEKIT_AGENT_NAME") or None,
    )


# --- filesystem layout (§0's guaranteed paths, plus one this module must invent) --------------


def build_tree_dir(work_directory: Path, process_name: str) -> Path:
    return _ensure_within(
        work_directory / "build" / process_name,
        work_directory,
        process_name=process_name,
        stage="build",
    )


def world_scratch_dir(
    work_directory: Path, world_index: int, process_name: str
) -> Path:
    return _ensure_within(
        work_directory / "worlds" / f"w{world_index}" / process_name,
        work_directory,
        process_name=process_name,
        stage="spawn",
    )


def managed_engine_data_dir(
    work_directory: Path, process_name: str, *, world_index: int | None
) -> Path:
    """§0 fixes `/work/build/<proc>/` and `/work/worlds/w<N>/<proc>/` but names no path for a
    managed engine's own data directory. Per-world engines reuse the per-world scratch shape
    (`world_index` given) so `reset`'s `datadir_copy` case (§4.2: "restore its data directory")
    has an unambiguous per-world location. Job-shared engines (`world_index=None`) get a
    job-level directory outside any single world's tree, since nothing about them is per-world —
    `/work/managed/<proc>/`, which §0 v1.8 added to the layout block (JC4, p5-round1-review).
    """
    if world_index is None:
        path = work_directory / "managed" / process_name
    else:
        return world_scratch_dir(work_directory, world_index, process_name)
    return _ensure_within(
        path, work_directory, process_name=process_name, stage="spawn"
    )


# --- §2b depends_on wait -------------------------------------------------------------------------


def _readiness_probes_for_process(
    manifest: EnvironmentBundleV2, process_name: str
) -> list[ReadinessProbeV2]:
    """F8, p5-round1-review: returns EVERY declared probe backing this process, not just the
    first-listed one — matching `healthy()`'s own definition of ready (§4 point 3: "declared
    `readiness` probes," plural, unqualified). A process backing two capabilities used to be
    treated as ready by `depends_on` the moment the first-listed probe passed, then immediately
    reported unhealthy the instant `healthy()` ran, because the two functions disagreed about
    what "ready" meant."""
    capability_slugs = {
        slug
        for slug, capability in manifest.capabilities.items()
        if capability.service == process_name
    }
    return [
        probe for probe in manifest.readiness if probe.capability in capability_slugs
    ]


def _tcp_probe(host: str, port: int, *, timeout: float = 0.75) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _probe_http(
    host: str,
    port: int,
    path: str | None,
    *,
    user: str | None = None,
    password: str | None = None,
    timeout: float = 1.0,
) -> bool:
    """N7, p6-review-r2: `user`/`password`, when both given, ride along as a Basic-auth header —
    added for the rabbitmq management listener probe (`_wait_for_store_ready`), which otherwise
    401s on every attempt regardless of whether the listener is actually up."""
    url = f"http://{host}:{port}/{(path or '').lstrip('/')}"
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(url)
        if user is not None and password is not None:
            credential_bytes = f"{user}:{password}".encode("utf-8")
            request.add_header(
                "Authorization", "Basic " + base64.b64encode(credential_bytes).decode()
            )
        with opener.open(request, timeout=timeout) as response:
            response.read(256)
            return 200 <= response.status < 400
    except Exception:
        # Q6, p6-review-r3: `http.client.HTTPException` (a malformed status line, a truncated
        # body) is not an `OSError` — a probe's contract is bool-not-raise, same broad form
        # `_probe_postgres` already uses below.
        return False


def _probe_postgres(
    host: str,
    port: int,
    *,
    user: str | None = None,
    password: str | None = None,
    dbname: str | None = None,
    timeout: float = 1.0,
) -> bool:
    """F9, p5-round1-review: previously connected as `user="postgres", dbname="postgres"` —
    neither exists, since the catalog's `initdb -U harness` creates only the `harness` role — and
    treated almost any resulting `OperationalError` as "answered, therefore ready," including
    `FATAL: the database system is starting up`, postgres's OWN response while still in recovery
    (it binds its listen socket early). With real generated credentials threaded through, this
    runs an actual `SELECT 1`, which a starting-up server cannot pass. Without credentials (no
    generated role for this call site, or `psycopg` absent) it falls back to a bare TCP probe —
    strictly weaker, but no longer claims false readiness from a substring match either.
    """
    try:
        import psycopg  # type: ignore[import-not-found]
    except ImportError:
        return _tcp_probe(host, port, timeout=timeout)
    if user is None or dbname is None:
        return _tcp_probe(host, port, timeout=timeout)
    try:
        connection = psycopg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname,
            connect_timeout=timeout,
        )
        try:
            connection.execute("SELECT 1")
        finally:
            connection.close()
        return True
    except psycopg.OperationalError:
        return False
    except Exception:
        return False


def default_capability_prober(
    *,
    protocol: CapabilityProtocol,
    host: str,
    port: int,
    path: str | None,
    user: str | None = None,
    password: str | None = None,
    dbname: str | None = None,
) -> bool:
    if protocol is CapabilityProtocol.POSTGRES:
        return _probe_postgres(host, port, user=user, password=password, dbname=dbname)
    if protocol is CapabilityProtocol.HTTP:
        return _probe_http(host, port, path, user=user, password=password)
    return _tcp_probe(host, port)


def _poll_until(
    condition: Callable[[], bool],
    *,
    timeout: float,
    interval: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    timeout_error: Callable[[], Exception],
) -> None:
    deadline = clock() + timeout
    while True:
        if condition():
            return
        if clock() >= deadline:
            raise timeout_error()
        sleep(interval)


def _postgres_probe_credentials(
    capability: CapabilityV2,
    *,
    world_index: int,
    credentials: dict[str, EngineCredentials] | None,
) -> tuple[str | None, str | None, str | None]:
    if capability.protocol is not CapabilityProtocol.POSTGRES:
        return None, None, None
    creds = (credentials or {}).get(capability.service)
    if creds is None:
        return None, None, None
    return creds.username, creds.password, f"w{world_index}"


def wait_for_dependency(
    manifest: EnvironmentBundleV2,
    dependency_name: str,
    *,
    world_index: int,
    port_plan: PortPlan,
    spawned: SpawnedWorldProcess,
    credentials: dict[str, EngineCredentials] | None = None,
    prober: CapabilityProber = default_capability_prober,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """§2b: "the dependent starts only after the dependency's capability `readiness` probe passes
    (or its `started_check`, or immediately after spawn if it has neither)." Priority order per
    the contract's own parenthetical: a declared capability readiness probe first; `started_check`
    only when the dependency backs no such capability (and only `SourceProcess` ever carries one —
    `ManagedProcess` has no `started_check` field); otherwise return immediately, since spawn
    already happened before this is ever called.

    A dependency backing more than one capability must pass EVERY declared probe (F8), matching
    `healthy()`'s own definition — the combined timeout is the longest of them (no probe is cut
    off early), the combined poll interval is the tightest of them (no probe is polled less often
    than it declared). `credentials` (F9) lets a postgres probe run a real `SELECT 1` instead of
    a bare TCP connect; omitted, it degrades to the same TCP-only check as before.
    """
    probes = _readiness_probes_for_process(manifest, dependency_name)
    if probes:
        port = port_plan.port_for(dependency_name, world_index)

        def probe_ready() -> bool:
            for probe in probes:
                capability = manifest.capabilities[probe.capability]
                user, password, dbname = _postgres_probe_credentials(
                    capability, world_index=world_index, credentials=credentials
                )
                if not prober(
                    protocol=capability.protocol,
                    host="localhost",
                    port=port,
                    path=probe.path,
                    user=user,
                    password=password,
                    dbname=dbname,
                ):
                    return False
            return True

        combined_timeout = max(probe.timeout_seconds for probe in probes)
        _poll_until(
            probe_ready,
            timeout=combined_timeout,
            interval=min(probe.interval_seconds for probe in probes),
            clock=clock,
            sleep=sleep,
            timeout_error=lambda: ProcessRuntimeError(
                "depends_on",
                "depends_on_timeout",
                f"{dependency_name}: readiness probe did not pass within {combined_timeout}s",
                process=dependency_name,
                domain=FailureDomain.INFRASTRUCTURE,
            ),
        )
        return

    processes_by_name = {process.name: process for process in manifest.processes}
    dependency = processes_by_name[dependency_name]
    started_check = (
        dependency.started_check if isinstance(dependency, SourceProcess) else None
    )
    if started_check is None:
        return  # neither a readiness probe nor a started_check — ready immediately after spawn.

    if started_check.port:
        # §2b (v1.8): the field only SELECTS the port-probe variant — the dialed port is always
        # the dependency's own allocated one (`port_plan.port_for`, honoring `fixed_port`), never
        # a literal read from the manifest (F4, p5-round1-review). A literal cannot be correct for
        # more than one world, since the formula port differs by `world_index`.
        port = port_plan.port_for(dependency_name, world_index)

        def condition() -> bool:
            return _tcp_probe("localhost", port)
    else:
        marker = started_check.log_marker

        def condition() -> bool:
            return marker in spawned.handle.captured_output()

    def condition_or_exited() -> bool:
        if condition():
            return True
        if not spawned.handle.is_running():
            # A dependency that has already died can never pass its check; waiting out the
            # full timeout only hides the crash. Surface its own output instead: this is the
            # customer's process (`SourceProcess`), so the fault is theirs to read, not infra.
            raise ProcessRuntimeError(
                "depends_on",
                "spawn_failed",
                f"{dependency_name}: exited before started_check passed"
                f"{_output_tail(spawned.handle)}",
                process=dependency_name,
                domain=FailureDomain.AGENT,
            )
        return False

    _poll_until(
        condition_or_exited,
        timeout=started_check.timeout_seconds,
        interval=0.25,
        clock=clock,
        sleep=sleep,
        timeout_error=lambda: ProcessRuntimeError(
            "depends_on",
            "depends_on_timeout",
            f"{dependency_name}: started_check did not pass within "
            f"{started_check.timeout_seconds}s"
            f"{_output_tail(spawned.handle)}",
            process=dependency_name,
            domain=FailureDomain.INFRASTRUCTURE,
        ),
    )


_OUTPUT_TAIL_LINES = 40
_OUTPUT_TAIL_CHARS = 1500


def _output_tail(handle: SpawnedProcess) -> str:
    """The dependency's last output lines, formatted as a message suffix (empty when silent)."""
    lines = handle.captured_output().splitlines()[-_OUTPUT_TAIL_LINES:]
    tail = "\n".join(line.rstrip() for line in lines if line.strip())
    if not tail:
        return "; process wrote no output"
    if len(tail) > _OUTPUT_TAIL_CHARS:
        tail = "…" + tail[-_OUTPUT_TAIL_CHARS:]
    return "; last output:\n" + tail


def _topological_order(manifest: EnvironmentBundleV2) -> list[str]:
    """Dependencies before dependents. `preflight_bundle` (`_verify_depends_on`) already rejects
    cycles and unknown names before this ever runs, so a DAG is assumed here, not re-verified."""
    graph = {process.name: list(process.depends_on) for process in manifest.processes}
    order: list[str] = []
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        for dependency_name in graph[name]:
            visit(dependency_name)
        order.append(name)

    for name in graph:
        visit(name)
    return order


# --- §2c/§5 store command seams -----------------------------------------------------------------
#
# One injectable seam per engine, mirroring `ProcessRunner`/`CapabilityProber`: real production
# code talks to the actual store, a test hands in a spy. `SqlRunner` alone carries the "SQL spy"
# tests assert TEMPLATE/DROP/CREATE/sentinel statements against — postgres is the only engine
# `no_sql_store` (§2e item 6) guarantees, so it is the only one whose statement-level shape this
# module commits to; redis/rabbitmq get the same command-injection treatment but a plainer one
# (`argv`, not a statement AST), matching how much less this contract pins about them (§2c: seeded
# "only if the repo ships seed state for them").


class SqlRunner(Protocol):
    def __call__(
        self,
        *,
        host: str,
        port: int,
        user: str,
        password: str,
        dbname: str,
        statement: str,
        read_only: bool = False,
    ) -> list[tuple[Any, ...]]: ...


def default_sql_runner(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    dbname: str,
    statement: str,
    read_only: bool = False,
) -> list[tuple[Any, ...]]:
    """psycopg-backed, import-guarded like `_probe_postgres` — never exercised in this module's
    own test lane (no real postgres here), always injected by a test's SQL spy instead.

    N8, p6-review-r2 (MAJOR): `read_only`, when set, puts the session into a read-only default
    transaction mode before running `statement` — the ONLY thing that made `sentinel.query` (and
    the baseline row-count reads it shares this code path with) different from an arbitrary write
    was convention: it runs as `harness`, the role `initdb -U harness` makes the postgres
    SUPERUSER, over a plain autocommit session, executing customer-authored content from an
    untrusted repo verbatim. `world-handle-interface.md` v3.4's own `query()` runs on a `SET
    TRANSACTION READ ONLY` connection and calls that "the guard" — this is the provisioner's side
    of the same guard, for every call site that is supposed to be a read (`_call_sql`'s own
    `read_only=True` callers).
    """
    import psycopg  # type: ignore[import-not-found]

    with psycopg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname,
        autocommit=True,
    ) as connection:
        if read_only:
            connection.execute("SET default_transaction_read_only = on")
        cursor = connection.execute(statement)
        try:
            return cursor.fetchall()
        except psycopg.ProgrammingError:
            return []  # a DDL/utility statement (CREATE DATABASE, ALTER DATABASE, ...) has no rows.


class RedisCommandRunner(Protocol):
    def __call__(self, *, host: str, port: int, command: Sequence[str]) -> Any: ...


def default_redis_command_runner(
    *, host: str, port: int, command: Sequence[str]
) -> Any:
    import redis  # type: ignore[import-not-found]

    client = redis.Redis(host=host, port=port)
    try:
        return client.execute_command(*command)
    finally:
        client.close()


class RabbitmqQueueInspector(Protocol):
    def __call__(
        self, *, host: str, port: int, credentials: EngineCredentials, queue: str
    ) -> int: ...


class RabbitmqQueueDeclarer(Protocol):
    def __call__(
        self,
        *,
        host: str,
        port: int,
        credentials: EngineCredentials,
        queue: str,
        message: str,
    ) -> None: ...


class RabbitmqQueueDeleter(Protocol):
    """Q10, p6-review-r3: no remaining caller — N15 (p6-review-r2) moved canary cleanup to the
    reset-from-baseline path this seam predates. Retained deliberately (constructor param,
    `SpawnContext` field) as the explicit-delete seam a future canary strategy may still want,
    rather than ripped out and re-added from scratch."""

    def __call__(
        self,
        *,
        host: str,
        port: int,
        credentials: EngineCredentials,
        queue: str,
    ) -> None: ...


# --- B5, p6-review-r1: typed failures at the store-command seams -----------------------------
#
# Every call below runs AFTER the store's own readiness has already been established (B4's wait,
# or an already-running job-shared engine) — a driver exception here is the store rejecting or
# erroring on the provisioner's OWN statement, never a connection race, so it is §2f's
# `store_statement_failed` (infrastructure, retryable — v1.10), never `depends_on_timeout` (that
# code belongs to the readiness WAIT itself, which `default_capability_prober` already degrades
# to a bool instead of raising) and never `seed_failed` (reserved for the customer's own
# migration/seed CONTENT, applied through `sync_run`'s exit code, not through these seams at
# all). One wrapper per seam, matching the existing one-Protocol-per-engine-concern split.


def _call_sql(
    sql_runner: SqlRunner,
    *,
    stage: str,
    process_name: str,
    host: str,
    port: int,
    user: str,
    password: str,
    dbname: str,
    statement: str,
    read_only: bool = False,
) -> list[tuple[Any, ...]]:
    try:
        return sql_runner(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname,
            statement=statement,
            read_only=read_only,
        )
    except Exception as exc:
        raise ProcessRuntimeError(
            stage,
            "store_statement_failed",
            f"{process_name}: store rejected a provisioner-issued statement: {exc}",
            process=process_name,
            domain=FailureDomain.INFRASTRUCTURE,
        ) from exc


def _call_redis(
    redis_runner: RedisCommandRunner,
    *,
    stage: str,
    process_name: str,
    host: str,
    port: int,
    command: Sequence[str],
) -> Any:
    try:
        return redis_runner(host=host, port=port, command=command)
    except Exception as exc:
        raise ProcessRuntimeError(
            stage,
            "store_statement_failed",
            f"{process_name}: store rejected a provisioner-issued command: {exc}",
            process=process_name,
            domain=FailureDomain.INFRASTRUCTURE,
        ) from exc


def _call_rabbitmq(
    rabbitmq_inspector: RabbitmqQueueInspector,
    *,
    stage: str,
    process_name: str,
    host: str,
    port: int,
    credentials: EngineCredentials,
    queue: str,
) -> int:
    try:
        return rabbitmq_inspector(
            host=host, port=port, credentials=credentials, queue=queue
        )
    except Exception as exc:
        raise ProcessRuntimeError(
            stage,
            "store_statement_failed",
            f"{process_name}: store rejected a provisioner-issued queue inspection: {exc}",
            process=process_name,
            domain=FailureDomain.INFRASTRUCTURE,
        ) from exc


_RABBITMQ_DEPTH_READ_ATTEMPTS = 3
_RABBITMQ_DEPTH_READ_INTERVAL_SECONDS = 0.05


def _call_rabbitmq_with_retry(
    rabbitmq_inspector: RabbitmqQueueInspector,
    *,
    stage: str,
    process_name: str,
    host: str,
    port: int,
    credentials: EngineCredentials,
    queue: str,
    accept: Callable[[int], bool],
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """N15, p6-review-r2 (MINOR): the management API's `messages` field is fed by the node's own
    stats collector (`collect_statistics_interval`, default 5000ms) — a depth read immediately
    after a publish/reset can report a stale value in EITHER direction: 0 when a message really
    is there (the direction that falsely PASSES an isolation check), or a leftover count when it
    really is gone (the direction that falsely FAILS an `expected_depth` sentinel). `accept` is
    the caller's own success predicate; retried a bounded few times before settling for whatever
    the last read reported, so a genuinely wrong result is never masked by retrying forever — only
    the collector's known lag window is absorbed.
    """
    depth = 0
    for attempt in range(_RABBITMQ_DEPTH_READ_ATTEMPTS):
        depth = _call_rabbitmq(
            rabbitmq_inspector,
            stage=stage,
            process_name=process_name,
            host=host,
            port=port,
            credentials=credentials,
            queue=queue,
        )
        if accept(depth):
            return depth
        if attempt < _RABBITMQ_DEPTH_READ_ATTEMPTS - 1:
            sleep(_RABBITMQ_DEPTH_READ_INTERVAL_SECONDS)
    return depth


def _call_rabbitmq_action(
    fn: Callable[..., None],
    *,
    stage: str,
    process_name: str,
    action: str,
    **kwargs: Any,
) -> None:
    """Same B5 typing as `_call_rabbitmq`, for the write-side canary declare/publish call
    `_run_canary_probe` makes through `context.rabbitmq_declare` — the m8 canary's OWN statement
    against a store that has already passed readiness, exactly the class of call B5 covers. (Q10,
    p6-review-r3: `context.rabbitmq_delete` is no longer one of these — N15 moved canary cleanup
    to the reset-from-baseline path; the seam itself is kept, see `RabbitmqQueueDeleter`.)"""
    try:
        fn(**kwargs)
    except Exception as exc:
        raise ProcessRuntimeError(
            stage,
            "store_statement_failed",
            f"{process_name}: store rejected the provisioner's canary {action}: {exc}",
            process=process_name,
            domain=FailureDomain.INFRASTRUCTURE,
        ) from exc


def _rabbitmq_management_port(amqp_port: int) -> int:
    """§2b's catalog/port formula fixes only the AMQP listener port; the management HTTP API
    (needed to seed via `rabbitmqadmin` and to read queue depth for a sentinel) has no formula of
    its own. A fixed `+10000` offset is a defensible, deterministic V1 default — `14000..14099` /
    `15000..15799` (§2b) shifted by it lands at `24000..24099` / `25000..25799`, outside every
    band this contract reserves, so it can never alias a different process's allocated port. Same
    provisioner-internal status as `postgres_daemon_argv` et al.: swappable, never fixed by the
    contract, never exercised against a real broker in this module's own test lane.
    """
    return amqp_port + 10000


def _rabbitmq_auth_header(credentials: EngineCredentials) -> str:
    credential_bytes = f"{credentials.username}:{credentials.password}".encode("utf-8")
    return "Basic " + base64.b64encode(credential_bytes).decode()


def default_rabbitmq_queue_inspector(
    *, host: str, port: int, credentials: EngineCredentials, queue: str
) -> int:
    management_port = _rabbitmq_management_port(port)
    url = f"http://{host}:{management_port}/api/queues/%2F/{urlquote(queue, safe='')}"
    request = urllib.request.Request(url)
    request.add_header("Authorization", _rabbitmq_auth_header(credentials))
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # m8, p6-review-r1: a nonexistent queue is what "empty"/"absent" LOOKS like at this
            # endpoint — the management API's own way of saying so is a 404, not an error the
            # conformance gate's "never raises" promise should have to survive by accident.
            return 0
        raise
    return int(payload.get("messages", 0))


def default_rabbitmq_queue_declare_and_publish(
    *,
    host: str,
    port: int,
    credentials: EngineCredentials,
    queue: str,
    message: str,
) -> None:
    """m8, p6-review-r1: the conformance canary previously only ever INSPECTED a rabbitmq queue —
    since nothing ever created one, world 1's "is it visible" check compared against a queue that
    never existed anywhere, a vacuous pass. Declares the reserved canary queue in world 0 for
    real via the management API (never AMQP directly — same HTTP-only seam as every other
    rabbitmq call this module makes) and publishes one message into it, so world 1's read is
    checking something that could actually leak.
    """
    management_port = _rabbitmq_management_port(port)
    auth = _rabbitmq_auth_header(credentials)

    declare_url = (
        f"http://{host}:{management_port}/api/queues/%2F/{urlquote(queue, safe='')}"
    )
    declare_request = urllib.request.Request(
        declare_url,
        data=json.dumps({"durable": False, "auto_delete": True}).encode("utf-8"),
        method="PUT",
    )
    declare_request.add_header("Content-Type", "application/json")
    declare_request.add_header("Authorization", auth)
    with urllib.request.urlopen(declare_request, timeout=5.0):
        pass

    # The default exchange's routing key IS the queue name — the standard way the management
    # API's own `publish` endpoint targets a specific queue without declaring a binding first.
    publish_url = (
        f"http://{host}:{management_port}/api/exchanges/%2F/amq.default/publish"
    )
    publish_request = urllib.request.Request(
        publish_url,
        data=json.dumps(
            {
                "properties": {},
                "routing_key": queue,
                "payload": message,
                "payload_encoding": "string",
            }
        ).encode("utf-8"),
        method="POST",
    )
    publish_request.add_header("Content-Type", "application/json")
    publish_request.add_header("Authorization", auth)
    with urllib.request.urlopen(publish_request, timeout=5.0):
        pass


def default_rabbitmq_queue_delete(
    *,
    host: str,
    port: int,
    credentials: EngineCredentials,
    queue: str,
) -> None:
    """Cleanup half of the canary (m8) — best-effort: a 404 means it is already gone (e.g. the
    world-0 reset that runs right after already wiped it, since `datadir_copy` is rabbitmq's only
    legal strategy), which is the desired end state, not a failure."""
    management_port = _rabbitmq_management_port(port)
    url = f"http://{host}:{management_port}/api/queues/%2F/{urlquote(queue, safe='')}"
    request = urllib.request.Request(url, method="DELETE")
    request.add_header("Authorization", _rabbitmq_auth_header(credentials))
    try:
        with urllib.request.urlopen(request, timeout=5.0):
            pass
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise


class RabbitmqDefinitionsImporter(Protocol):
    def __call__(
        self,
        *,
        host: str,
        port: int,
        credentials: EngineCredentials,
        file: Path,
    ) -> None: ...


def default_rabbitmq_definitions_importer(
    *,
    host: str,
    port: int,
    credentials: EngineCredentials,
    file: Path,
) -> None:
    """N17, p6-review-r2 (MINOR): `rabbitmqadmin import <file>` used to require the
    `rabbitmqadmin` binary specifically — served BY the management plugin at runtime (fetched
    from the running node's own web UI), never installed by the `rabbitmq-server` package itself,
    and not in §0's guaranteed-binary list (`python`, `node`, `git`, `ffmpeg`, plus the §2b engine
    catalog). `rabbitmqadmin import` is documented as a thin wrapper over exactly this endpoint
    (`POST /api/definitions`) — the same management HTTP API this module already talks to for
    every other rabbitmq call (`default_rabbitmq_queue_inspector` et al.), so seeding no longer
    depends on a binary the snapshot might not ship at all.
    """
    management_port = _rabbitmq_management_port(port)
    url = f"http://{host}:{management_port}/api/definitions"
    request = urllib.request.Request(url, data=file.read_bytes(), method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Authorization", _rabbitmq_auth_header(credentials))
    with urllib.request.urlopen(request, timeout=10.0):
        pass


# --- §2c seed application -------------------------------------------------------------------


def postgres_seed_argv(*, port: int, dbname: str, user: str, file: Path) -> list[str]:
    return [
        "psql",
        "-h",
        "localhost",
        "-p",
        str(port),
        "-U",
        user,
        "-d",
        dbname,
        "-v",
        "ON_ERROR_STOP=1",
        "-f",
        str(file),
    ]


def postgres_seed_env(credentials: EngineCredentials) -> dict[str, str]:
    return {"PGPASSWORD": credentials.password}


def redis_seed_argv(*, port: int) -> list[str]:
    """No `-f`/script flag exists on `redis-cli`; a seed file is plain commands, one per line,
    fed over stdin — the documented way `redis-cli` accepts a batch of ordinary commands (as
    opposed to `--pipe`, which expects pre-encoded RESP, not this module's business to generate).
    """
    return ["redis-cli", "-h", "localhost", "-p", str(port)]


def apply_seed_file(
    engine: ManagedEngine,
    file: Path,
    *,
    port: int,
    dbname: str,
    credentials: EngineCredentials | None,
    process_name: str,
    sync_run: Callable[..., subprocess.CompletedProcess],
    user: int | None = None,
    group: int | None = None,
    rabbitmq_import: RabbitmqDefinitionsImporter = default_rabbitmq_definitions_importer,
) -> None:
    """Applies one migration/seed file, per §2c: "applied in listed order." `postgres` shells out
    to a psql-style command (`-f`, so a large schema file streams rather than loading into this
    process); `redis`/`rabbitmq` use their own bulk-load mechanisms — three small per-engine
    branches, mirroring `postgres_daemon_argv` et al.'s split, so a fourth catalog engine needs
    only one new branch here, never a rewrite of the caller. `sync_run` is `SpawnContext.sync_run`
    (or a test's fake) — the same "run one synchronous step, check its exit code" seam
    `build_process_tree`'s build steps and `spawn_managed_process`'s postgres bootstrap already
    use. Raises on the first failing file (`ProcessRuntimeError`, stage="seed"), never partially
    proceeds silently past one — same rule `build_process_tree` holds for `build_commands`.

    B2, p6-review-r1: `user`/`group` are the store's OWN declared spawn identity (`svc-data`),
    never left to default to the provisioner's own uid — `psql -f` honors backslash meta-commands
    (`\\!`, `\\copy ... program`), so a migration/seed file is a customer-authored-content
    execution path exactly like `build_commands`, which already drops privilege the same way.
    """
    if engine is ManagedEngine.POSTGRES:
        if credentials is None:
            raise ProcessRuntimeError(
                "seed",
                "internal_missing_credentials",
                "postgres requires generated credentials to seed",
                process=process_name,
            )
        argv = postgres_seed_argv(
            port=port, dbname=dbname, user=credentials.username, file=file
        )
        # F12's own rule (§2b's closed env list), reapplied here: `env=` fully REPLACES a child's
        # environment rather than extending the caller's — passing just `{"PGPASSWORD": ...}`
        # would drop `PATH` entirely, and `psql` living anywhere outside `subprocess`'s POSIX
        # fallback path (`/bin:/usr/bin`) — a homebrew or venv install, commonly — would silently
        # fail to exec.
        env = {**_allowlisted_ambient_env(os.environ), **postgres_seed_env(credentials)}
        result = sync_run(
            argv, capture_output=True, text=True, env=env, user=user, group=group
        )
    elif engine is ManagedEngine.REDIS:
        argv = redis_seed_argv(port=port)
        result = sync_run(
            argv,
            capture_output=True,
            text=True,
            input=file.read_text(encoding="utf-8"),
            user=user,
            group=group,
        )
    elif engine is ManagedEngine.RABBITMQ:
        if credentials is None:
            raise ProcessRuntimeError(
                "seed",
                "internal_missing_credentials",
                "rabbitmq requires generated credentials to seed",
                process=process_name,
            )
        # N17, p6-review-r2: seeded over the management HTTP API directly, never through a
        # subprocess/`sync_run` — no `rabbitmqadmin` binary dependency (`user`/`group` are moot
        # here for the same reason: an HTTP call has no OS-level identity to drop).
        try:
            rabbitmq_import(
                host="localhost", port=port, credentials=credentials, file=file
            )
        except Exception as exc:
            raise ProcessRuntimeError(
                "seed",
                "seed_failed",
                f"{file}: {exc}",
                process=process_name,
                domain=FailureDomain.ENVIRONMENT,
            ) from exc
        return
    else:  # pragma: no cover - ManagedEngine is closed; unreachable past the model layer.
        raise ProcessRuntimeError(
            "seed",
            "seed_failed",
            f"unknown engine {engine!r}",
            process=process_name,
            domain=FailureDomain.ENVIRONMENT,
        )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()[:2000]
        raise ProcessRuntimeError(
            "seed",
            "seed_failed",
            f"{file}: exited {result.returncode}" + (f": {stderr}" if stderr else ""),
            process=process_name,
            domain=FailureDomain.ENVIRONMENT,
        )


def apply_store_seed(
    store: StoreEntry,
    *,
    engine: ManagedEngine,
    bundle_dir: Path,
    port: int,
    dbname: str,
    credentials: EngineCredentials | None,
    process_name: str,
    sync_run: Callable[..., subprocess.CompletedProcess],
    user: int | None = None,
    group: int | None = None,
    rabbitmq_import: RabbitmqDefinitionsImporter = default_rabbitmq_definitions_importer,
) -> None:
    """§2c: "migrations then seed_files... applied in listed order" — migrations always precede
    seed_files, regardless of how many files either list holds, and each list keeps its own
    authored order (never sorted)."""
    for relative_path in (*store.migrations, *store.seed_files):
        apply_seed_file(
            engine,
            bundle_dir / relative_path,
            port=port,
            dbname=dbname,
            credentials=credentials,
            process_name=process_name,
            sync_run=sync_run,
            user=user,
            group=group,
            rabbitmq_import=rabbitmq_import,
        )


# --- §2c sentinel checking ---------------------------------------------------------------------


def check_sentinel(
    store: StoreEntry,
    *,
    engine: ManagedEngine,
    host: str,
    port: int,
    dbname: str | None,
    credentials: EngineCredentials | None,
    sql_runner: SqlRunner,
    redis_runner: RedisCommandRunner,
    rabbitmq_inspector: RabbitmqQueueInspector,
    process_name: str = "",
    stage: str = "sentinel",
) -> bool:
    """§2c/§4.2: "a read-only check plus its exact expected value (string compare)." Dispatches on
    the sentinel's own implied shape (`Sentinel.implied_engine`), which the model layer already
    guarantees matches the store's engine (`sentinel_shape_mismatch`) — this only has to act on
    it, not re-verify it. `process_name`/`stage` are for B5's typed-failure wrapper only (default
    to the empty string / "sentinel" so every pre-existing direct caller keeps working unchanged).
    """
    sentinel = store.sentinel
    if engine is ManagedEngine.POSTGRES:
        if credentials is None or dbname is None:
            return False
        rows = _call_sql(
            sql_runner,
            stage=stage,
            process_name=process_name,
            host=host,
            port=port,
            user=credentials.username,
            password=credentials.password,
            dbname=dbname,
            statement=sentinel.query,  # type: ignore[arg-type]
            read_only=True,  # N8, p6-review-r2: customer-authored content, §2c calls this a read.
        )
        actual = str(rows[0][0]) if rows and rows[0] else None
        return actual == sentinel.expected
    if engine is ManagedEngine.REDIS:
        value = _call_redis(
            redis_runner,
            stage=stage,
            process_name=process_name,
            host=host,
            port=port,
            command=["GET", sentinel.key],  # type: ignore[list-item]
        )
        actual = (
            value.decode()
            if isinstance(value, bytes)
            else (None if value is None else str(value))
        )
        return actual == sentinel.expected
    if engine is ManagedEngine.RABBITMQ:
        if credentials is None:
            return False
        depth = _call_rabbitmq_with_retry(
            rabbitmq_inspector,
            stage=stage,
            process_name=process_name,
            host=host,
            port=port,
            credentials=credentials,
            queue=sentinel.queue,  # type: ignore[arg-type]
            accept=lambda value: value == sentinel.expected_depth,  # N15, p6-review-r2.
        )
        return depth == sentinel.expected_depth
    return False  # pragma: no cover - ManagedEngine is closed; unreachable past the model layer.


# --- per-world spawn orchestration --------------------------------------------------------------


@dataclass(frozen=True)
class SpawnContext:
    """The caller-supplied dependencies `spawn_world` needs, grouped so the per-call signature
    stays to (manifest, world_index, shared_handles) — Phase 6's `provision()` builds one of
    these per job and calls `spawn_world` once per world index inside its own reconciliation
    loop; that loop, idempotency across retries, and `EnvironmentRuntime` state assembly across
    all `instances` worlds are its job, not this function's.
    """

    work_directory: Path
    port_plan: PortPlan
    credentials: dict[str, EngineCredentials]
    secret_values: dict[str, str]
    secret_purposes: dict[str, str]
    # A dedicated Daytona sandbox can still share one LiveKit project with other jobs. Carry the
    # stable job identity into placeholder rendering so each worker gets a unique dispatch name.
    job_id: str = "local"
    runner: ProcessRunner = default_process_runner
    # Shared by `build_process_trees`' `build_commands` steps and `spawn_managed_process`'s
    # postgres `initdb` bootstrap — both are "run one synchronous step, check its exit code."
    sync_run: Callable[..., subprocess.CompletedProcess] = subprocess.run
    prober: CapabilityProber = default_capability_prober
    copy: Callable[[Path, Path], None] | None = None
    # F1, p5-round1-review: threaded to every build/spawn call this context drives.
    # `require_declared_user=False` is the local-lane default (no `svc-*` accounts on a dev box);
    # a hosted caller sets it `True` so a snapshot that somehow lacks a declared user fails typed
    # instead of silently running unprivileged.
    user_resolver: Callable[[str], "pwd.struct_passwd | None"] = default_user_resolver
    require_declared_user: bool = False
    chown: Callable[[Path, int, int], None] = _default_chown
    build_step_timeout_seconds: float = _DEFAULT_BUILD_STEP_TIMEOUT_SECONDS
    # §2c/§5 store command seams (below) — defaulted so every existing `SpawnContext(...)` call
    # in P5's own tests keeps working unchanged; Phase 6 callers override them with spies.
    sql_runner: SqlRunner = default_sql_runner
    redis_runner: RedisCommandRunner = default_redis_command_runner
    rabbitmq_inspector: RabbitmqQueueInspector = default_rabbitmq_queue_inspector
    # m8, p6-review-r1: the conformance canary's rabbitmq branch (below `_run_canary_probe`)
    # needs to actually create the reserved queue, not just read one that was never declared.
    rabbitmq_declare: RabbitmqQueueDeclarer = default_rabbitmq_queue_declare_and_publish
    rabbitmq_delete: RabbitmqQueueDeleter = default_rabbitmq_queue_delete
    # N17, p6-review-r2: seeding no longer shells out to `rabbitmqadmin` — the same management
    # HTTP API seam every other rabbitmq call in this context already uses.
    rabbitmq_import: RabbitmqDefinitionsImporter = default_rabbitmq_definitions_importer
    bundle_dir: Path | None = None


@dataclass
class WorldSpawnResult:
    handles: dict[str, SpawnedWorldProcess]
    endpoints: dict[str, RuntimeEndpoint]


def spawn_world(
    manifest: EnvironmentBundleV2,
    *,
    world_index: int,
    context: SpawnContext,
    shared_handles: dict[str, SpawnedWorldProcess] | None = None,
) -> WorldSpawnResult:
    """Spawns every process for ONE world in dependency order, waiting for each dependency and
    every spawned source process's own readiness/started check (§2b). The latter matters for a
    terminal control process: no dependent exists to wait on it, but the world must not become
    READY before that process actually registers. `shared_handles` carries already-running
    job-shared managed engines (from an earlier world's call) so they are reused, never respawned,
    for `world_index > 0` — the caller is expected to thread the same dict through every
    `spawn_world` call for one job. Source process build trees must already exist (via
    `build_process_tree`, called once per process before any world is spawned) — this function only
    creates per-world scratch directories, never a build tree.
    """
    endpoints = build_endpoints(
        manifest,
        world_index=world_index,
        port_plan=context.port_plan,
        credentials=context.credentials,
    )
    configuration_addresses = configuration_addresses_from_endpoints(endpoints)
    processes_by_name = {process.name: process for process in manifest.processes}
    handles: dict[str, SpawnedWorldProcess] = dict(shared_handles or {})

    try:
        for name in _topological_order(manifest):
            if name in handles:
                continue  # a job-shared managed engine already running from an earlier world.
            process = processes_by_name[name]
            for dependency_name in process.depends_on:
                wait_for_dependency(
                    manifest,
                    dependency_name,
                    world_index=world_index,
                    port_plan=context.port_plan,
                    spawned=handles[dependency_name],
                    credentials=context.credentials,
                    prober=context.prober,
                )
            if isinstance(process, ManagedProcess):
                data_dir = managed_engine_data_dir(
                    context.work_directory,
                    name,
                    world_index=None
                    if context.port_plan.is_job_shared(name)
                    else world_index,
                )
                handle = spawn_managed_process(
                    process,
                    port=context.port_plan.port_for(name, world_index),
                    data_dir=data_dir,
                    credentials=context.credentials.get(name),
                    runner=context.runner,
                    sync_run=context.sync_run,
                    user_resolver=context.user_resolver,
                    require_declared_user=context.require_declared_user,
                    chown=context.chown,
                )
            else:
                handle = spawn_source_process(
                    process,
                    build_dir=build_tree_dir(context.work_directory, name),
                    world_dir=world_scratch_dir(
                        context.work_directory, world_index, name
                    ),
                    world_index=world_index,
                    job_id=context.job_id,
                    port_plan=context.port_plan,
                    configuration_addresses=configuration_addresses,
                    secret_values=context.secret_values,
                    secret_purposes=context.secret_purposes,
                    runner=context.runner,
                    user_resolver=context.user_resolver,
                    require_declared_user=context.require_declared_user,
                    chown=context.chown,
                )
            handles[name] = handle
            if isinstance(process, SourceProcess) and process.started_check is not None:
                wait_for_dependency(
                    manifest,
                    name,
                    world_index=world_index,
                    port_plan=context.port_plan,
                    spawned=handle,
                    credentials=context.credentials,
                    prober=context.prober,
                )
    except BaseException as exc:
        # N4, p6-review-r2 (MAJOR): a `depends_on_timeout`/`spawn_failed` partway through this
        # world's own process list used to drop every ALREADY-spawned process of the SAME world on
        # the floor — nothing held it, so a caller's later `close()`/retry could `rmtree` a data
        # directory, or attempt a fresh spawn, out from under a still-live sibling process.
        # `handles` (this world's own accumulated dict, seeded from `shared_handles`) is attached
        # to the exception so the caller can publish/terminate it instead.
        exc.partial_handles = handles  # type: ignore[attr-defined]
        raise
    return WorldSpawnResult(handles=handles, endpoints=endpoints)


def build_process_trees(
    manifest: EnvironmentBundleV2, *, source_root: Path, context: SpawnContext
) -> dict[str, Path]:
    """Builds every `source` process's tree once — the caller invokes this exactly once per job,
    before the first `spawn_world` call for any world (§2b: `build_commands` run "once per
    job")."""
    return {
        process.name: build_process_tree(
            process,
            source_root=(
                context.bundle_dir if process.source_origin == "bundle" else source_root
            ),
            build_root=context.work_directory / "build",
            run=context.sync_run,
            copy=context.copy,
            user_resolver=context.user_resolver,
            require_declared_user=context.require_declared_user,
            chown=context.chown,
            build_step_timeout_seconds=context.build_step_timeout_seconds,
        )
        for process in manifest.processes
        if isinstance(process, SourceProcess)
    }


# --- §4.3 healthy() --------------------------------------------------------------------------


def _split_host_port(address: str) -> tuple[str, int]:
    parsed = urlsplit(address)
    return parsed.hostname or "localhost", parsed.port or 0


def _postgres_credentials_from_address(
    address: str,
) -> tuple[str | None, str | None, str | None]:
    """F9: the rendered `endpoint.address` for a postgres capability already carries the
    generated role/password/database (`postgresql://harness:<pw>@localhost:<port>/w<N>`) — parsed
    back out rather than threading `EngineCredentials` separately into `probe_runtime_health`,
    which only ever sees `EnvironmentRuntime`, not the job's credential map."""
    parsed = urlsplit(address)
    return parsed.username, parsed.password, (parsed.path.lstrip("/") or None)


def probe_runtime_health(
    manifest: EnvironmentBundleV2,
    runtime: EnvironmentRuntime,
    *,
    prober: CapabilityProber = default_capability_prober,
) -> bool:
    """§4 point 3: "`healthy` = declared `readiness` probes, not 'process is running.'" Every
    declared `readiness` entry for a capability this runtime actually exposes must pass; a
    capability with no readiness entry declared carries no health obligation (§2b: "Health/
    readiness is otherwise declared ONLY in the capability-level `readiness` section")."""
    for probe in manifest.readiness:
        endpoint = runtime.endpoints.get(probe.capability)
        capability = manifest.capabilities.get(probe.capability)
        if endpoint is None or capability is None:
            return False
        host, port = _split_host_port(endpoint.address)
        user = password = dbname = None
        if capability.protocol is CapabilityProtocol.POSTGRES:
            user, password, dbname = _postgres_credentials_from_address(
                endpoint.address
            )
        if not prober(
            protocol=capability.protocol,
            host=host,
            port=port,
            path=probe.path,
            user=user,
            password=password,
            dbname=dbname,
        ):
            return False
    return True


async def healthy(
    manifest: EnvironmentBundleV2,
    runtime: EnvironmentRuntime,
    *,
    prober: CapabilityProber = default_capability_prober,
) -> bool:
    """Async wrapper matching §4's `RuntimeProvider.healthy` shape (`runtime.py`'s own
    `LocalComposeRuntimeProvider.healthy` uses the same `asyncio.to_thread` idiom).

    §3's `state` transitions are fixed: `preparing->ready`, `ready->unhealthy`, and
    `unhealthy->ready` ONLY via re-provision reconcile (§4 point 1: "a sick world mid-job is
    recovered by calling `provision` again"). `healthy()` is not a reconcile (F6, p5-round1-
    review) — it may only ever DEMOTE `runtime.state`, never promote it. `ready` stays `ready`
    while still healthy; `unhealthy`/`stopped` are cleared only by `provision()`/`close()`, never
    by a passing probe alone (a world whose sentinel was never re-proved after coming back up must
    not silently re-enter the pool).
    """
    import asyncio

    is_healthy = await asyncio.to_thread(
        probe_runtime_health, manifest, runtime, prober=prober
    )
    if not is_healthy:
        runtime.state = RuntimeState.UNHEALTHY
    elif runtime.state is RuntimeState.PREPARING:
        runtime.state = RuntimeState.READY
    return is_healthy


# --- internal invariant codes -------------------------------------------------------------------
#
# Neither is in §2f's closed table — both mark a precondition upstream layers already guarantee
# (a postgres/rabbitmq `ManagedProcess` always gets generated credentials; a store's backing
# service is always a `ManagedProcess`, per `bundle_v2`'s own `store_service_not_managed`), same
# status as `render_capability_address`'s `internal_missing_credentials` above — a bug to fix
# here, never a bundle defect the outbound seam needs a name for.
_INTERNAL_MISSING_CREDENTIALS = "internal_missing_credentials"
_INTERNAL_INVARIANT_VIOLATED = "internal_invariant_violated"


def _require_credentials(
    credentials: EngineCredentials | None, *, stage: str, process_name: str
) -> EngineCredentials:
    if credentials is None:
        raise ProcessRuntimeError(
            stage,
            _INTERNAL_MISSING_CREDENTIALS,
            f"{process_name}: postgres/rabbitmq requires generated credentials but none were "
            "supplied",
            process=process_name,
        )
    return credentials


# --- §2c baseline: store lookup + naming -----------------------------------------------------


def _store_by_process_name(manifest: EnvironmentBundleV2) -> dict[str, StoreEntry]:
    """Every declared store, keyed by the `ManagedProcess.name` backing its capability — the
    shape every baseline/clone/reset function below actually wants, one hop past what §2c's own
    `seed.stores[].capability` names."""
    result: dict[str, StoreEntry] = {}
    if manifest.seed is not None:
        for store in manifest.seed.stores:
            capability = manifest.capabilities[store.capability]
            result[capability.service] = store
    return result


def _template_database_name(process_name: str) -> str:
    """§2c's own baseline databases are never `w<N>` (§2b reserves that shape for per-world logical
    DBs) — `-` is replaced since a process `name` may carry it (`^[a-z0-9][a-z0-9_-]*$`, §2b) but a
    bare postgres identifier cannot without quoting."""
    return f"alk_baseline_{process_name.replace('-', '_')}"


def _datadir_copy_baseline_path(work_directory: Path, process_name: str) -> Path:
    """A sibling of the live `/work/managed/<proc>/` dir (§0), never nested inside it — `freeze_
    baseline` copies FROM the live dir INTO this one; nesting would make that copy recurse into
    its own destination."""
    return _ensure_within(
        work_directory / "managed" / f"{process_name}.baseline",
        work_directory,
        process_name=process_name,
        stage="baseline",
    )


def _measure_postgres_row_counts(
    *,
    host: str,
    port: int,
    credentials: EngineCredentials,
    dbname: str,
    sql_runner: SqlRunner,
    process_name: str,
) -> dict[str, int]:
    """Per-table row counts as the baseline stands right after seeding — `HostedWorld`'s own
    `baseline_row_counts` input (`world/handle.py`'s `STATE_ROW_CAP` gate), measured once here
    rather than re-measured at scenario time, so the cap it enforces cannot depend on what a
    scenario already wrote to the table."""
    tables = _call_sql(
        sql_runner,
        stage="baseline",
        process_name=process_name,
        host=host,
        port=port,
        user=credentials.username,
        password=credentials.password,
        dbname=dbname,
        statement="SELECT tablename FROM pg_tables WHERE schemaname = 'public'",
        read_only=True,  # N8, p6-review-r2.
    )
    counts: dict[str, int] = {}
    for row in tables:
        table = row[0]
        # M1, p6-review-r1: `table` is the raw `pg_tables.tablename` — from the untrusted repo's
        # own migrations. An unescaped double quote inside it used to close the identifier early
        # and let anything after it execute as a second statement (psycopg3's simple-query
        # protocol runs every statement in an unparameterized `execute()` call).
        safe_table = table.replace('"', '""')
        result = _call_sql(
            sql_runner,
            stage="baseline",
            process_name=process_name,
            host=host,
            port=port,
            user=credentials.username,
            password=credentials.password,
            dbname=dbname,
            statement=f'SELECT COUNT(*) FROM "{safe_table}"',
            read_only=True,  # N8, p6-review-r2.
        )
        counts[table] = int(result[0][0]) if result and result[0] else 0
    return counts


# --- B4, p6-review-r1: readiness wait between a managed-engine spawn and the first statement --

_DEFAULT_STORE_READY_TIMEOUT_SECONDS = 30.0
_DEFAULT_STORE_READY_INTERVAL_SECONDS = 0.25

_PROTOCOL_BY_ENGINE = {
    ManagedEngine.POSTGRES: CapabilityProtocol.POSTGRES,
    ManagedEngine.REDIS: CapabilityProtocol.REDIS,
    ManagedEngine.RABBITMQ: CapabilityProtocol.AMQP,
}


def _wait_for_store_ready(
    manifest: EnvironmentBundleV2,
    process: ManagedProcess,
    *,
    port: int,
    context: SpawnContext,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """B4, p6-review-r1: `wait_for_dependency` already makes every `depends_on` EDGE wait for its
    dependency's readiness before touching it; a freeze/seal call site is not itself anyone's
    declared dependent, so the very same class of race (the engine has just been `exec`'d and has
    not finished booting) went unguarded at every place THIS module spawns a store immediately
    before issuing its own first statement against it. Reuses the manifest's declared `readiness`
    timeout for this process's capability when the bundle names one, else a bounded default —
    `default_capability_prober` already degrades to a bare TCP probe without credentials, so this
    never requires a real login to work. Raises `depends_on_timeout` (§2f) on exhaustion, the same
    code `wait_for_dependency` uses for the identical class of failure.
    """
    probes = _readiness_probes_for_process(manifest, process.name)
    timeout = max(
        (probe.timeout_seconds for probe in probes),
        default=_DEFAULT_STORE_READY_TIMEOUT_SECONDS,
    )
    interval = min(
        (probe.interval_seconds for probe in probes),
        default=_DEFAULT_STORE_READY_INTERVAL_SECONDS,
    )
    credentials = context.credentials.get(process.name)
    user = credentials.username if credentials else None
    password = credentials.password if credentials else None
    # Always the default `postgres` database — guaranteed to exist the instant `initdb` finishes,
    # regardless of which baseline/world database this particular spawn is ultimately building
    # toward. This call only has to prove "the engine accepts connections."
    dbname = "postgres" if process.engine is ManagedEngine.POSTGRES else None
    protocol = _PROTOCOL_BY_ENGINE[process.engine]

    def ready() -> bool:
        if not context.prober(
            protocol=protocol,
            host="localhost",
            port=port,
            path=None,
            user=user,
            password=password,
            dbname=dbname,
        ):
            return False
        if process.engine is ManagedEngine.RABBITMQ:
            # N7, p6-review-r2: the AMQP listener probed above comes up during CORE boot; the
            # management plugin's HTTP listener comes up in the PLUGIN boot step that follows —
            # the same B4 race, unfixed for the one engine whose seed/sentinel/canary statements
            # all travel over the listener that was never probed. Both must answer.
            if not context.prober(
                protocol=CapabilityProtocol.HTTP,
                host="localhost",
                port=_rabbitmq_management_port(port),
                path="/api/overview",
                user=user,
                password=password,
            ):
                return False
        return True

    _poll_until(
        ready,
        timeout=timeout,
        interval=interval,
        clock=clock,
        sleep=sleep,
        timeout_error=lambda: ProcessRuntimeError(
            "baseline",
            "depends_on_timeout",
            f"{process.name}: did not become ready within {timeout}s",
            process=process.name,
            domain=FailureDomain.INFRASTRUCTURE,
        ),
    )


# --- N2, p6-review-r2 (BLOCKER): poll the promote-path health check, don't sample it once -------


def _poll_runtime_health(
    manifest: EnvironmentBundleV2,
    runtime: EnvironmentRuntime,
    *,
    prober: CapabilityProber,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Shared by BOTH promote sites — `provision()`'s PREPARING->READY promotion and `reset()`'s
    post-sentinel probe — which used to call `probe_runtime_health` exactly once, immediately,
    with no wait. `spawn_world` only waits on `depends_on` EDGES (round-1's own M4 note): nobody
    depends on the DAG's TERMINAL process, so nothing ever waited for it. A bundle whose
    readiness-bearing process is terminal (§2a's own documented `vapi`/`retell`-with-a-backend
    shape) was marked `UNHEALTHY` on every provision and every reset regardless of how quickly it
    actually came up, driving §5.4's `world_pool_exhausted` against a healthy environment.

    Same `max(declared timeout, default)` / `min(declared interval, default)` shape `_wait_for_
    store_ready` already established for the identical class of race, applied across every probe
    the runtime's OWN endpoints declare (`manifest.readiness`, the same set `probe_runtime_health`
    itself iterates). Exhaustion returns `False` — never raises: this feeds a state DECISION
    (`READY` vs `UNHEALTHY`), not a provisioning failure the caller must escalate.
    """
    timeout = max(
        (probe.timeout_seconds for probe in manifest.readiness),
        default=_DEFAULT_STORE_READY_TIMEOUT_SECONDS,
    )
    interval = min(
        (probe.interval_seconds for probe in manifest.readiness),
        default=_DEFAULT_STORE_READY_INTERVAL_SECONDS,
    )
    deadline = clock() + timeout
    while True:
        if probe_runtime_health(manifest, runtime, prober=prober):
            return True
        if clock() >= deadline:
            return False
        sleep(interval)


# --- §5.3 build output (`build.json`) ---------------------------------------------------------


@dataclass(frozen=True)
class StoreBaselineRecord:
    """One store's sealed identity, as §5.3 wants it recorded on the build output.
    `baseline_reference`: `template_database` -> the sealed template database's own name;
    `datadir_copy` -> the baseline snapshot directory world-clone/reset copy from;
    `empty` -> `""` (§5.3: "no-op capture" — there is nothing to seal). `row_counts` is populated
    for postgres stores only — `HostedWorld` (the only consumer) is postgres-only."""

    capability: str
    process_name: str
    engine: ManagedEngine
    strategy: BaselineStrategy
    inputs_digest: str
    baseline_reference: str
    row_counts: dict[str, int]

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "capability": self.capability,
            "process_name": self.process_name,
            "engine": self.engine.value,
            "strategy": self.strategy.value,
            "inputs_digest": self.inputs_digest,
            "baseline_reference": self.baseline_reference,
            "row_counts": dict(self.row_counts),
        }


@dataclass
class BuildOutput:
    """§5.3's `build.json` artifact: "record `inputs_digest` + achieved-baseline reference on the
    build output." `conformance`/`conformance_reason` start unset; the conformance gate fills them
    in and the caller re-writes the file — §4's own "record pass/fail on the build output."
    `requested_parallelism`/`effective_parallelism`/`degrade_reason` (m1, p6-review-r1) are what
    P7's own `parallelism_degraded` event payload needs (`{requested, effective, reason}`) for
    EITHER degrade cause — `PortPlan.degraded_reason` (`fixed_port`) used to be computed and never
    read by this provider; only the conformance gate's own reason ever reached `build.json`.
    """

    bundle_digest: str
    stores: list[StoreBaselineRecord]
    conformance: bool | None = None
    conformance_reason: str | None = None
    requested_parallelism: int | None = None
    effective_parallelism: int | None = None
    degrade_reason: str | None = None

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "bundle_digest": self.bundle_digest,
            "stores": [record.to_json() for record in self.stores],
            "conformance": self.conformance,
            "conformance_reason": self.conformance_reason,
            "requested_parallelism": self.requested_parallelism,
            "effective_parallelism": self.effective_parallelism,
            "degrade_reason": self.degrade_reason,
        }


def write_build_output(work_directory: Path, build_output: BuildOutput) -> Path:
    artifacts_dir = work_directory / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    target = artifacts_dir / "build.json"
    target.write_text(
        json.dumps(build_output.to_json(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


# --- §5 step 3: baseline freeze -----------------------------------------------------------------


@dataclass
class FreezeResult:
    """`freeze_baseline`'s full output: the `BuildOutput` §5.3 wants written to `build.json`, plus
    the live handles of every `template_database` engine it just sealed — that engine stays
    running for the rest of the job (§2b: job-shared), so the caller (`ProcessRuntimeProvider`)
    must carry its handle forward into every world rather than re-spawning it."""

    build_output: BuildOutput
    job_shared_handles: dict[str, SpawnedWorldProcess]


def freeze_baseline(
    manifest: EnvironmentBundleV2,
    *,
    bundle_digest: str,
    context: SpawnContext,
) -> FreezeResult:
    """§5 step 3 / §2c: seed each managed store once, then seal it per baseline strategy — the
    FIRST seal; every later world clone/reset reuses what this produces (`_seal_world_store`).
    Runs once per job, before any world exists — the caller is responsible for calling this
    exactly once, same convention as `build_process_trees`. `context.bundle_dir` must be set (the
    bundle's own directory, for locating `migrations`/`seed_files` on disk) — the one place in
    this module a caller must supply it, since seeding is the one place they matter.
    """
    if context.bundle_dir is None:
        raise ProcessRuntimeError(
            "baseline",
            _INTERNAL_INVARIANT_VIOLATED,
            "SpawnContext.bundle_dir is required to seed",
        )
    store_by_process = _store_by_process_name(manifest)
    records: list[StoreBaselineRecord] = []
    job_shared_handles: dict[str, SpawnedWorldProcess] = {}
    try:
        for process in manifest.processes:
            if not isinstance(process, ManagedProcess):
                continue
            store = store_by_process.get(process.name)
            if store is None:
                continue  # §2b default: no store entry -> per-world, nothing to freeze here.
            record, handle = _freeze_one_store(
                manifest, process, store, context=context
            )
            records.append(record)
            if handle is not None:
                job_shared_handles[process.name] = handle
    except BaseException:
        # N4, p6-review-r2 (MAJOR): a raise partway through leaves every EARLIER store's own
        # job-shared engine live and unreferenced anywhere — `self._job_shared_handles` is never
        # assigned on a raise (only after this function RETURNS), so nothing can terminate it and
        # nothing frees the port it holds for a same-bundle retry. Terminated here rather than
        # left running: the whole attempt has already failed (`ProcessRuntimeProvider` resets the
        # job identity on this same raise — N3), so nothing downstream can resume these handles.
        for handle in job_shared_handles.values():
            _terminate_and_wait(handle.handle)
        raise
    return FreezeResult(
        build_output=BuildOutput(bundle_digest=bundle_digest, stores=records),
        job_shared_handles=job_shared_handles,
    )


def _freeze_one_store(
    manifest: EnvironmentBundleV2,
    process: ManagedProcess,
    store: StoreEntry,
    *,
    context: SpawnContext,
) -> tuple[StoreBaselineRecord, SpawnedWorldProcess | None]:
    strategy = store.baseline.strategy
    if strategy is BaselineStrategy.EMPTY:
        # §5.3: "no-op capture" — nothing seeded, nothing sealed here; `_seal_world_store`'s
        # fallback branch (re)establishes this store's declared state on every clone/reset instead.
        return (
            StoreBaselineRecord(
                capability=store.capability,
                process_name=process.name,
                engine=process.engine,
                strategy=strategy,
                inputs_digest=store.baseline.inputs_digest,
                baseline_reference="",
                row_counts={},
            ),
            None,
        )

    port = context.port_plan.port_for(
        process.name, 0
    )  # job-shared or not, world 0's slot is
    # this bootstrap instance's own — a job-shared engine's port never varies by world_index
    # (`PortPlan.port_for`); a `datadir_copy` engine has no real world yet to collide with.
    credentials = context.credentials.get(process.name)
    data_dir = managed_engine_data_dir(
        context.work_directory, process.name, world_index=None
    )
    handle = spawn_managed_process(
        process,
        port=port,
        data_dir=data_dir,
        credentials=credentials,
        runner=context.runner,
        sync_run=context.sync_run,
        user_resolver=context.user_resolver,
        require_declared_user=context.require_declared_user,
        chown=context.chown,
    )
    try:
        # B4, p6-review-r1: the engine has just been `exec`'d — the very next thing this function
        # does is talk to it.
        _wait_for_store_ready(manifest, process, port=port, context=context)

        baseline_dbname = (
            _template_database_name(process.name)
            if process.engine is ManagedEngine.POSTGRES
            else ""
        )
        if process.engine is ManagedEngine.POSTGRES:
            credentials = _require_credentials(
                credentials, stage="baseline", process_name=process.name
            )
            _call_sql(
                context.sql_runner,
                stage="baseline",
                process_name=process.name,
                host="localhost",
                port=port,
                user=credentials.username,
                password=credentials.password,
                dbname="postgres",
                statement=f'CREATE DATABASE "{baseline_dbname}"',
            )
        apply_store_seed(
            store,
            engine=process.engine,
            bundle_dir=context.bundle_dir,
            port=port,
            dbname=baseline_dbname,
            credentials=credentials,
            process_name=process.name,
            sync_run=context.sync_run,
            user=handle.uid,
            group=handle.gid,
            rabbitmq_import=context.rabbitmq_import,
        )
        # m3, p6-review-r1: §2c defines the sentinel as a check "against the freshly seeded
        # baseline" — checked here, before sealing, so a seed that silently produced the wrong
        # state is caught at freeze (surfaced as `seed_failed` — the seed's own content did not
        # produce what its own sentinel expects, a deterministic authoring fault) rather than
        # first noticed at the scheduler's first `reset`, long after `build.json` already
        # recorded success.
        sentinel_dbname = (
            baseline_dbname if process.engine is ManagedEngine.POSTGRES else None
        )
        if not check_sentinel(
            store,
            engine=process.engine,
            host="localhost",
            port=port,
            dbname=sentinel_dbname,
            credentials=credentials,
            sql_runner=context.sql_runner,
            redis_runner=context.redis_runner,
            rabbitmq_inspector=context.rabbitmq_inspector,
            process_name=process.name,
            stage="baseline",
        ):
            raise ProcessRuntimeError(
                "baseline",
                "seed_failed",
                f"{process.name}: sentinel check failed against the freshly seeded baseline",
                process=process.name,
                domain=FailureDomain.ENVIRONMENT,
            )

        row_counts: dict[str, int] = {}
        baseline_reference = ""
        result_handle: SpawnedWorldProcess | None = None
        if strategy is BaselineStrategy.TEMPLATE_DATABASE:
            if process.engine is ManagedEngine.POSTGRES:
                row_counts = _measure_postgres_row_counts(
                    host="localhost",
                    port=port,
                    credentials=credentials,
                    dbname=baseline_dbname,
                    sql_runner=context.sql_runner,
                    process_name=process.name,
                )
            # Sealed: marked a TEMPLATE and closed to new connections, so nothing can write into
            # it after this — "the sealed post-migrate+seed datastore state" (glossary, "Baseline").
            _call_sql(
                context.sql_runner,
                stage="baseline",
                process_name=process.name,
                host="localhost",
                port=port,
                user=credentials.username,
                password=credentials.password,
                dbname="postgres",
                statement=(
                    f'ALTER DATABASE "{baseline_dbname}" WITH IS_TEMPLATE true '
                    "ALLOW_CONNECTIONS false"
                ),
            )
            baseline_reference = baseline_dbname
            result_handle = (
                handle  # stays running — job-shared for the rest of the job.
            )
        elif strategy is BaselineStrategy.DATADIR_COPY:
            if process.engine is ManagedEngine.POSTGRES:
                row_counts = _measure_postgres_row_counts(
                    host="localhost",
                    port=port,
                    credentials=credentials,
                    dbname=baseline_dbname,
                    sql_runner=context.sql_runner,
                    process_name=process.name,
                )
            if process.engine is ManagedEngine.REDIS:
                # Q1, p6-review-r3 (MAJOR): `redis_daemon_argv` disables save points (`--save
                # ""`), so a plain SIGTERM shutdown persists nothing — without an explicit
                # synchronous `SAVE` first, the copied data dir below is an empty baseline.
                _call_redis(
                    context.redis_runner,
                    stage="baseline",
                    process_name=process.name,
                    host="localhost",
                    port=port,
                    command=["SAVE"],
                )
            # M7, p6-review-r1: waits for the engine to actually exit before the snapshot copy
            # below — a bare `terminate()` only sends SIGTERM, and postgres's smart shutdown may
            # not have even started, let alone finished, by the next line. Q12, p6-review-r3: a
            # longer wait for rabbitmq — its broker shutdown routinely exceeds the 5s default and
            # a SIGKILL mid-mnesia-write would seal a corrupt snapshot.
            _terminate_and_wait(
                handle.handle,
                timeout=(
                    _RABBITMQ_TERMINATE_WAIT_SECONDS
                    if process.engine is ManagedEngine.RABBITMQ
                    else _TERMINATE_WAIT_SECONDS
                ),
            )
            baseline_dir = _datadir_copy_baseline_path(
                context.work_directory, process.name
            )
            try:
                if baseline_dir.exists():
                    shutil.rmtree(baseline_dir)
                (context.copy or _copytree_preserving_symlinks)(data_dir, baseline_dir)
            except (OSError, shutil.Error) as exc:
                # N9, p6-review-r2 (MAJOR): §4.6 — this IS the "baseline/seal copies" half of the
                # mapping; sealing the very first snapshot every world clones/resets from is a
                # filesystem operation, not a store-command seam B5 already typed.
                raise ProcessRuntimeError(
                    "baseline",
                    "store_statement_failed",
                    f"{process.name}: sealing the datadir_copy baseline: {exc}",
                    process=process.name,
                    domain=FailureDomain.INFRASTRUCTURE,
                ) from exc
            baseline_reference = str(baseline_dir)
            # `result_handle` stays `None` — every world starts its OWN engine instance from this
            # snapshot (`_seal_world_store`'s `DATADIR_COPY` branch), never this bootstrap one.

        record = StoreBaselineRecord(
            capability=store.capability,
            process_name=process.name,
            engine=process.engine,
            strategy=strategy,
            inputs_digest=store.baseline.inputs_digest,
            baseline_reference=baseline_reference,
            row_counts=row_counts,
        )
        return record, result_handle
    except BaseException:
        # N4, p6-review-r2 (MAJOR): any raise between spawn and the successful return (a readiness
        # timeout, a bad seed file, a failing freeze-time sentinel) used to leave THIS store's own
        # just-spawned engine live and referenced NOWHERE — `freeze_baseline`'s own loop never
        # even sees `handle` when this function raises instead of returning. Best-effort:
        # `DATADIR_COPY`'s own success path has usually already terminated it by this point, and
        # `_terminate_and_wait` tolerates an already-dead handle.
        _terminate_and_wait(handle.handle)
        raise


# --- §4.2/§4 world clone + reset -----------------------------------------------------------------


def _seal_world_store(
    manifest: EnvironmentBundleV2,
    process: ManagedProcess,
    store: StoreEntry | None,
    record: StoreBaselineRecord | None,
    *,
    world_index: int,
    context: SpawnContext,
) -> SpawnedWorldProcess | None:
    """Drives ONE world's managed store to the sealed baseline, per §4.2 — shared by the FIRST
    clone (`ProcessRuntimeProvider._ensure_world`) and every later `reset_world`; the caller is
    expected to have already terminated any existing per-world handle for `process.name` before
    calling this again. Returns the new per-world `SpawnedWorldProcess` for a per-world engine
    (`datadir_copy`/`empty`/no-entry), or `None` for a job-shared `template_database` engine —
    nothing new to spawn there, only its logical `wN` database changes.
    """
    port = context.port_plan.port_for(process.name, world_index)
    data_dir = managed_engine_data_dir(
        context.work_directory, process.name, world_index=world_index
    )
    credentials = context.credentials.get(process.name)
    strategy = store.baseline.strategy if store is not None else None

    if strategy is BaselineStrategy.TEMPLATE_DATABASE:
        if record is None:
            raise ProcessRuntimeError(
                "reset",
                _INTERNAL_INVARIANT_VIOLATED,
                f"{process.name}: template_database has no frozen baseline record",
                process=process.name,
            )
        credentials = _require_credentials(
            credentials, stage="reset", process_name=process.name
        )
        # No spawn on this branch (the job-shared engine is already running, and already went
        # through B4's wait once at freeze/first-clone time) — nothing new to wait on here.
        _reset_template_database(
            host="localhost",
            port=port,
            credentials=credentials,
            world_db=f"w{world_index}",
            template_db=record.baseline_reference,
            sql_runner=context.sql_runner,
            process_name=process.name,
        )
        return None

    if strategy is BaselineStrategy.DATADIR_COPY:
        # R1, p6-review-r4 (known defect, disclosed rather than redesigned this phase): for
        # rabbitmq specifically, this branch restores the mnesia DIRECTORY but not a schema
        # rabbitmq will actually adopt — mnesia data is node-name-bound and per-world node names
        # must stay distinct for epmd (see `rabbitmq_daemon_env`'s docstring), so a rabbitmq
        # `datadir_copy` baseline boots correctly in world 0 only. Recorded follow-up: a
        # definitions-export restore via `default_rabbitmq_definitions_importer`.
        if record is None:
            raise ProcessRuntimeError(
                "reset",
                _INTERNAL_INVARIANT_VIOLATED,
                f"{process.name}: datadir_copy has no frozen baseline record",
                process=process.name,
            )
        try:
            if data_dir.exists():
                shutil.rmtree(data_dir)
            # No pre-`mkdir` here (unlike the fallback branch below) — `copytree`-shaped `copy`
            # creates its own destination and raises `FileExistsError` if it is already there.
            (context.copy or _copytree_preserving_symlinks)(
                Path(record.baseline_reference), data_dir
            )
        except (OSError, shutil.Error) as exc:
            # N9, p6-review-r2 (MAJOR): restoring a world's engine from its sealed baseline
            # snapshot is itself an infrastructure operation (§4.6) — the "baseline/seal copies"
            # half of the mapping, same class `_freeze_one_store`'s own snapshot copy got.
            raise ProcessRuntimeError(
                "reset",
                "store_statement_failed",
                f"{process.name}: restoring the datadir_copy baseline: {exc}",
                process=process.name,
                domain=FailureDomain.INFRASTRUCTURE,
            ) from exc
        new_handle = spawn_managed_process(
            process,
            port=port,
            data_dir=data_dir,
            credentials=credentials,
            runner=context.runner,
            sync_run=context.sync_run,
            user_resolver=context.user_resolver,
            require_declared_user=context.require_declared_user,
            chown=context.chown,
        )
        # B4, p6-review-r1: fresh spawn from a just-restored data directory — the next line talks
        # to it immediately.
        _wait_for_store_ready(manifest, process, port=port, context=context)
        if process.engine is ManagedEngine.POSTGRES:
            credentials = _require_credentials(
                credentials, stage="reset", process_name=process.name
            )
            world_db = f"w{world_index}"
            baseline_dbname = _template_database_name(process.name)
            # §2b: "under `datadir_copy` the provisioner configures each per-world engine's
            # database to `w<N>` as well — one rule, both strategies." The copied dir's own
            # database still carries `_freeze_one_store`'s bootstrap name; renamed here, once per
            # world, never touching the pristine snapshot every OTHER world/reset still copies from.
            _call_sql(
                context.sql_runner,
                stage="reset",
                process_name=process.name,
                host="localhost",
                port=port,
                user=credentials.username,
                password=credentials.password,
                dbname="postgres",
                statement=f'ALTER DATABASE "{baseline_dbname}" RENAME TO "{world_db}"',
            )
        return new_handle

    # `empty` strategy, or no `seed.stores` entry at all: no baseline snapshot exists to copy back
    # in (`freeze_baseline` never ran for it — §5.3's "no-op capture" / §2b's per-world default),
    # so this always starts the engine over a WIPED, empty data directory — unconditionally, since
    # a bare restart alone cannot be trusted to flush every engine (rabbitmq persists to disk
    # regardless of redis's own `--save ""`). An explicit `empty` store then re-applies its own
    # seed_files on top — every clone/reset is where its declared initial state actually gets
    # (re)established, since freeze deliberately skipped it.
    try:
        if data_dir.exists():
            shutil.rmtree(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # N9, p6-review-r2 (MAJOR): a bare wipe-and-recreate of this process/data-dir setup, same
        # class as `spawn_managed_process`'s own boundary.
        raise ProcessRuntimeError(
            "reset",
            "spawn_failed",
            f"{process.name}: preparing {data_dir}: {exc}",
            process=process.name,
            domain=FailureDomain.INFRASTRUCTURE,
        ) from exc
    handle = spawn_managed_process(
        process,
        port=port,
        data_dir=data_dir,
        credentials=credentials,
        runner=context.runner,
        sync_run=context.sync_run,
        user_resolver=context.user_resolver,
        require_declared_user=context.require_declared_user,
        chown=context.chown,
    )
    # B4, p6-review-r1: fresh spawn over a wiped data directory — the CREATE DATABASE/apply_
    # store_seed below talk to it immediately.
    _wait_for_store_ready(manifest, process, port=port, context=context)
    if store is not None and strategy is BaselineStrategy.EMPTY:
        dbname = f"w{world_index}" if process.engine is ManagedEngine.POSTGRES else ""
        if process.engine is ManagedEngine.POSTGRES:
            credentials = _require_credentials(
                credentials, stage="reset", process_name=process.name
            )
            _call_sql(
                context.sql_runner,
                stage="reset",
                process_name=process.name,
                host="localhost",
                port=port,
                user=credentials.username,
                password=credentials.password,
                dbname="postgres",
                statement=f'CREATE DATABASE "{dbname}"',
            )
        if context.bundle_dir is None:
            # Same invariant `freeze_baseline` guards explicitly — a silent `Path()` (cwd) fallback
            # here would resolve `seed_files` against the wrong directory instead of failing typed.
            raise ProcessRuntimeError(
                "reset",
                _INTERNAL_INVARIANT_VIOLATED,
                "SpawnContext.bundle_dir is required to re-seed an empty-strategy store",
                process=process.name,
            )
        apply_store_seed(
            store,
            engine=process.engine,
            bundle_dir=context.bundle_dir,
            port=port,
            dbname=dbname,
            credentials=credentials,
            process_name=process.name,
            sync_run=context.sync_run,
            user=handle.uid,
            group=handle.gid,
            rabbitmq_import=context.rabbitmq_import,
        )
    return handle


def _reset_template_database(
    *,
    host: str,
    port: int,
    credentials: EngineCredentials,
    world_db: str,
    template_db: str,
    sql_runner: SqlRunner,
    process_name: str,
) -> None:
    """§4.2 `template_database`: "drop + recreate the world's logical DB from the template." The
    admin connection targets `postgres` (never `world_db` itself or the template — postgres
    forbids `DROP`/`CREATE DATABASE` from inside the database being touched). Terminates other
    backends on `world_db` first — a connection lingering from that world's own just-terminated
    `source` processes is exactly what would otherwise block the `DROP`.
    """
    admin = dict(
        host=host,
        port=port,
        user=credentials.username,
        password=credentials.password,
        dbname="postgres",
    )
    _call_sql(
        sql_runner,
        stage="reset",
        process_name=process_name,
        **admin,
        statement=(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            f"WHERE datname = '{world_db}' AND pid <> pg_backend_pid()"
        ),
    )
    _call_sql(
        sql_runner,
        stage="reset",
        process_name=process_name,
        **admin,
        statement=f'DROP DATABASE IF EXISTS "{world_db}"',
    )
    _call_sql(
        sql_runner,
        stage="reset",
        process_name=process_name,
        **admin,
        statement=f'CREATE DATABASE "{world_db}" TEMPLATE "{template_db}"',
    )


def _clone_or_reset_world(
    manifest: EnvironmentBundleV2,
    world_index: int,
    *,
    context: SpawnContext,
    baseline: BuildOutput,
    job_shared_handles: dict[str, SpawnedWorldProcess],
    existing_handles: dict[str, SpawnedWorldProcess],
) -> WorldSpawnResult:
    """Shared by `ProcessRuntimeProvider._ensure_world` (first creation / sick-world replace) and
    `reset_world` (mid-job restore) — both are "terminate this world's own per-world handles,
    reseal each managed store from the baseline, respawn `source` processes"; they differ only in
    what the caller does afterward (a fresh/replaced world is left `preparing`, judged later by
    `healthy()`'s declared readiness probes; a reset world is sentinel-checked immediately, per
    §4.2, and marked `unhealthy` on failure).
    """
    # N12, p6-review-r2 (MINOR): reversed — `existing_handles`' insertion order is `spawn_world`'s
    # own `_topological_order` (dependencies first); terminating it forward sends SIGTERM to this
    # world's own postgres while its `tools-api`/`agent` may still hold connections, guaranteeing
    # the full escalation wait every reset. A dict preserves insertion order, so `reversed()` here
    # IS reverse-topological order without recomputing it.
    for name, handle in reversed(list(existing_handles.items())):
        if name not in job_shared_handles:
            # M7, p6-review-r1: waits for real exit before `_seal_world_store` below `rmtree`s
            # this same process's data directory and rebinds its port — a bare `terminate()`
            # racing that `rmtree` is exactly `EADDRINUSE` / "remove a live server's data dir".
            _terminate_and_wait(
                handle.handle,
                prefer_interrupt=_prefers_interrupt(manifest, name),
            )

    store_by_process = _store_by_process_name(manifest)
    baseline_by_process = {record.process_name: record for record in baseline.stores}
    new_handles: dict[str, SpawnedWorldProcess] = dict(job_shared_handles)
    try:
        for process in manifest.processes:
            if not isinstance(process, ManagedProcess):
                continue
            # `_seal_world_store` runs even for an already-running job-shared engine
            # (`process.name` already in `new_handles`, seeded from `job_shared_handles` above)
            # — its own logical `wN` database still needs resetting every time, even though the
            # shared ENGINE PROCESS itself does not; `template_database`'s branch returns `None`,
            # so `new_handles` is left holding the pre-seeded shared handle untouched, exactly as
            # intended.
            store = store_by_process.get(process.name)
            record = baseline_by_process.get(process.name)
            sealed = _seal_world_store(
                manifest,
                process,
                store,
                record,
                world_index=world_index,
                context=context,
            )
            if sealed is not None:
                new_handles[process.name] = sealed
    except BaseException as exc:
        # N4, p6-review-r2 (MAJOR): a raise partway through this loop (a bad seed_files entry on
        # a LATER process, `_wait_for_store_ready` timing out) used to drop every EARLIER
        # process's freshly-(re)sealed handle on the floor — nothing holds it, so `close()`'s own
        # `rmtree` of this world's data directory next runs against a still-live engine. Attached
        # to the exception (never terminated here) so the caller — which alone knows whether this
        # is a first clone with nothing to lose, or a reset it may still want to retry — can merge
        # it into its own live-handle bookkeeping before deciding what to do next.
        exc.partial_handles = new_handles  # type: ignore[attr-defined]
        raise

    # `spawn_world` sets its own (more complete — it starts from `new_handles` and accumulates
    # further) `partial_handles` on a raise, so no extra wrapping is needed here for that case.
    return spawn_world(
        manifest, world_index=world_index, context=context, shared_handles=new_handles
    )


def _check_all_sentinels(
    manifest: EnvironmentBundleV2,
    world_index: int,
    *,
    context: SpawnContext,
) -> bool:
    """§4.2: "after every reset the store's sentinel must pass" — every declared store, not just
    the first; a single failing sentinel fails the whole check (short-circuits nothing, so every
    store is always attempted — useful diagnostics beat an early return here)."""
    store_by_process = _store_by_process_name(manifest)
    ok = True
    for process in manifest.processes:
        if not isinstance(process, ManagedProcess):
            continue
        store = store_by_process.get(process.name)
        if store is None:
            continue
        port = context.port_plan.port_for(process.name, world_index)
        dbname = f"w{world_index}" if process.engine is ManagedEngine.POSTGRES else None
        passed = check_sentinel(
            store,
            engine=process.engine,
            host="localhost",
            port=port,
            dbname=dbname,
            credentials=context.credentials.get(process.name),
            sql_runner=context.sql_runner,
            redis_runner=context.redis_runner,
            rabbitmq_inspector=context.rabbitmq_inspector,
            process_name=process.name,
            stage="reset",
        )
        ok = ok and passed
    return ok


def reset_world(
    manifest: EnvironmentBundleV2,
    world_index: int,
    *,
    context: SpawnContext,
    baseline: BuildOutput,
    job_shared_handles: dict[str, SpawnedWorldProcess],
    existing_handles: dict[str, SpawnedWorldProcess],
) -> tuple[dict[str, SpawnedWorldProcess], bool]:
    """§4.2's per-world reset, exactly — returns the world's refreshed handle map and whether
    every declared store's sentinel passed afterward. NEVER raises for a sentinel failure: §4.2's
    own words, "a sentinel failure marks the world unhealthy," is the caller's job to act on, not
    this function's to escalate. A genuine provisioning failure below this (a `ProcessRuntimeError`
    from spawn/build) still raises — that is not what "sentinel failure" covers.
    """
    result = _clone_or_reset_world(
        manifest,
        world_index,
        context=context,
        baseline=baseline,
        job_shared_handles=job_shared_handles,
        existing_handles=existing_handles,
    )
    ok = _check_all_sentinels(manifest, world_index, context=context)
    return result.handles, ok


# --- §4 conformance gate -------------------------------------------------------------------------

_CONFORMANCE_CANARY_NAME = (
    "_alk_conformance"  # §2c reserved name; mirrors process_preflight.py's
)
# own `_RESERVED_NAME` and `world/handle.py`'s `CONFORMANCE_TABLE` — redeclared locally rather
# than imported, matching the existing split of that same constant across those two modules.
_CONFORMANCE_MARKER = "alk-conformance-canary"

_CANARY_PROTOCOL_PREFERENCE = (
    CapabilityProtocol.POSTGRES,
    CapabilityProtocol.REDIS,
    CapabilityProtocol.AMQP,
)


def _first_canary_store(manifest: EnvironmentBundleV2) -> StoreEntry | None:
    """§4: "the first store by protocol preference postgres > redis > rabbitmq." `no_sql_store`
    (§2e item 6) guarantees a `kind: process` bundle always has a postgres store, so `None` is
    reachable only for the vacuous zero-store case §4 itself names (never actually a `kind:
    process` bundle in practice)."""
    if manifest.seed is None:
        return None
    by_protocol: dict[CapabilityProtocol, StoreEntry] = {}
    for store in manifest.seed.stores:
        capability = manifest.capabilities[store.capability]
        by_protocol.setdefault(capability.protocol, store)
    for protocol in _CANARY_PROTOCOL_PREFERENCE:
        if protocol in by_protocol:
            return by_protocol[protocol]
    return None


def _engine_for_store(
    manifest: EnvironmentBundleV2, store: StoreEntry
) -> ManagedEngine:
    capability = manifest.capabilities[store.capability]
    processes_by_name = {process.name: process for process in manifest.processes}
    process = processes_by_name[capability.service]
    if not isinstance(process, ManagedProcess):
        raise ProcessRuntimeError(
            "conformance",
            _INTERNAL_INVARIANT_VIOLATED,
            f"{store.capability}: backing service is not a managed engine; bundle_v2's "
            "store_service_not_managed should have rejected this bundle",
        )
    return process.engine


def _run_canary_probe(
    manifest: EnvironmentBundleV2,
    store: StoreEntry,
    engine: ManagedEngine,
    *,
    context: SpawnContext,
) -> bool:
    """Creates the reserved object in world 0, then asserts it is NOT visible in world 1 — proving
    the two worlds are really isolated from each other, not just reachable on different ports."""
    capability = manifest.capabilities[store.capability]
    process_name = capability.service
    credentials = context.credentials.get(process_name)
    port0 = context.port_plan.port_for(process_name, 0)
    port1 = context.port_plan.port_for(process_name, 1)

    if engine is ManagedEngine.POSTGRES:
        credentials = _require_credentials(
            credentials, stage="conformance", process_name=process_name
        )
        _call_sql(
            context.sql_runner,
            stage="conformance",
            process_name=process_name,
            host="localhost",
            port=port0,
            user=credentials.username,
            password=credentials.password,
            dbname="w0",
            statement=(
                f'CREATE TABLE "{_CONFORMANCE_CANARY_NAME}" (marker text); '
                f"INSERT INTO \"{_CONFORMANCE_CANARY_NAME}\" VALUES ('{_CONFORMANCE_MARKER}')"
            ),
        )
        rows = _call_sql(
            context.sql_runner,
            stage="conformance",
            process_name=process_name,
            host="localhost",
            port=port1,
            user=credentials.username,
            password=credentials.password,
            dbname="w1",
            statement=f"SELECT to_regclass('{_CONFORMANCE_CANARY_NAME}') IS NOT NULL",
            read_only=True,  # N8, p6-review-r2.
        )
        return not (rows and rows[0] and rows[0][0])
    if engine is ManagedEngine.REDIS:
        _call_redis(
            context.redis_runner,
            stage="conformance",
            process_name=process_name,
            host="localhost",
            port=port0,
            command=["SET", _CONFORMANCE_CANARY_NAME, _CONFORMANCE_MARKER],
        )
        value = _call_redis(
            context.redis_runner,
            stage="conformance",
            process_name=process_name,
            host="localhost",
            port=port1,
            command=["GET", _CONFORMANCE_CANARY_NAME],
        )
        return value is None
    if engine is ManagedEngine.RABBITMQ:
        # rabbitmq is never job-shared (§2b: not in `_ENGINE_STRATEGIES`'s `template_database`
        # set) — world 0 and world 1 are already separate broker processes on separate ports, so
        # there is no shared state left to leak between them. Declaring the queue in world 0 and
        # confirming world 1's own broker never saw it is what is left to prove — m8, p6-review-
        # r1: previously only ever INSPECTED (a read), so world 1's check compared against a
        # queue that had never been created anywhere, a vacuous pass regardless of isolation.
        credentials = _require_credentials(
            credentials, stage="conformance", process_name=process_name
        )
        _call_rabbitmq_action(
            context.rabbitmq_declare,
            stage="conformance",
            process_name=process_name,
            action="declare",
            host="localhost",
            port=port0,
            credentials=credentials,
            queue=_CONFORMANCE_CANARY_NAME,
            message=_CONFORMANCE_MARKER,
        )
        depth = _call_rabbitmq_with_retry(
            context.rabbitmq_inspector,
            stage="conformance",
            process_name=process_name,
            host="localhost",
            port=port1,
            credentials=credentials,
            queue=_CONFORMANCE_CANARY_NAME,
            accept=lambda value: value in (0, None),
        )
        # N15, p6-review-r2 (MINOR): no explicit delete here anymore. `run_conformance_gate`'s own
        # `reset_world` calls for BOTH worlds run right after this returns, and world 0's rabbitmq
        # reset (`datadir_copy`, its only legal strategy — §2b) wipes and restarts it from the
        # pristine baseline snapshot regardless — the same mechanism postgres/redis already rely
        # on for their own canary cleanup. Deleting HERE used to make `_verify_canary_absent`'s
        # LATER read of this same queue vacuous: it was already gone from an earlier, unrelated
        # step, proving nothing about whether the reset itself actually worked. A failed probe
        # (isolation broken) still returns here without a reset ever running — the reserved name
        # keeps that residual harmless (never collides with customer content).
        return depth in (0, None)
    return False  # pragma: no cover - ManagedEngine is closed; unreachable past the model layer.


def _verify_canary_absent(
    manifest: EnvironmentBundleV2,
    store: StoreEntry,
    engine: ManagedEngine,
    *,
    context: SpawnContext,
) -> bool:
    """§4: "assert it is gone" — checked against world 0's OWN namespace after both worlds reset,
    defense in depth against a reset that silently failed to actually seal a fresh baseline."""
    capability = manifest.capabilities[store.capability]
    process_name = capability.service
    credentials = context.credentials.get(process_name)
    port0 = context.port_plan.port_for(process_name, 0)

    if engine is ManagedEngine.POSTGRES:
        credentials = _require_credentials(
            credentials, stage="conformance", process_name=process_name
        )
        rows = _call_sql(
            context.sql_runner,
            stage="conformance",
            process_name=process_name,
            host="localhost",
            port=port0,
            user=credentials.username,
            password=credentials.password,
            dbname="w0",
            statement=f"SELECT to_regclass('{_CONFORMANCE_CANARY_NAME}') IS NOT NULL",
            read_only=True,  # N8, p6-review-r2.
        )
        return not (rows and rows[0] and rows[0][0])
    if engine is ManagedEngine.REDIS:
        value = _call_redis(
            context.redis_runner,
            stage="conformance",
            process_name=process_name,
            host="localhost",
            port=port0,
            command=["GET", _CONFORMANCE_CANARY_NAME],
        )
        return value is None
    if engine is ManagedEngine.RABBITMQ:
        credentials = _require_credentials(
            credentials, stage="conformance", process_name=process_name
        )
        depth = _call_rabbitmq_with_retry(
            context.rabbitmq_inspector,
            stage="conformance",
            process_name=process_name,
            host="localhost",
            port=port0,
            credentials=credentials,
            queue=_CONFORMANCE_CANARY_NAME,
            accept=lambda value: value in (0, None),  # N15, p6-review-r2.
        )
        return depth in (0, None)
    return False  # pragma: no cover - ManagedEngine is closed; unreachable past the model layer.


def run_conformance_gate(
    manifest: EnvironmentBundleV2,
    *,
    context: SpawnContext,
    baseline: BuildOutput,
    job_shared_handles: dict[str, SpawnedWorldProcess],
    world_handles: dict[int, dict[str, SpawnedWorldProcess]],
) -> tuple[bool, str | None]:
    """§4's 2-world canary, run once per attempt after baseline freeze: create the reserved
    `_alk_conformance` object in world 0, assert it is invisible in world 1, `reset` both, assert
    it is gone + every sentinel passes both worlds. TRULY never raises (M3, p6-review-r1) — every
    gate-procedure check returns `(False, "conformance_gate_failed")`, and a `ProcessRuntimeError`
    RAISED below this (B5's typed store-seam failures, `reset_world`'s own spawn/reseal path, a
    missing-credentials invariant) is now caught and degrades the SAME way rather than failing the
    whole job: §4's own words are "Fail -> effective parallelism 1 ... Loud, never silent," which
    governs the GATE'S RESULT, not just a False return from its own checks — a world that has not
    finished booting when the canary dials it (B4 narrows but does not eliminate this race) is an
    isolation-boundary problem exactly like a failed canary, not a reason to fail the job. Requires
    worlds 0 and 1 to already exist in `world_handles` — the caller (`ProcessRuntimeProvider`)
    provisions a throwaway 2-world pair before calling this, then reconciles down to 1 on a
    `False` return, or up to the job's real `W` on `True`.
    """
    store = _first_canary_store(manifest)
    if store is None:
        return (
            True,
            None,
        )  # §2e's own no_sql_store guarantee means this never actually happens.
    engine = _engine_for_store(manifest, store)

    world_index: int | None = None
    try:
        if not _run_canary_probe(manifest, store, engine, context=context):
            return False, "conformance_gate_failed"

        for world_index in (0, 1):
            handles, sentinel_ok = reset_world(
                manifest,
                world_index,
                context=context,
                baseline=baseline,
                job_shared_handles=job_shared_handles,
                existing_handles=world_handles.get(world_index, {}),
            )
            world_handles[world_index] = handles
            if not sentinel_ok:
                return False, "conformance_gate_failed"

        if not _verify_canary_absent(manifest, store, engine, context=context):
            return False, "conformance_gate_failed"
        return True, None
    except (ProcessRuntimeError, OSError, shutil.Error) as exc:
        # N9, p6-review-r2: widened past `ProcessRuntimeError` alone — `reset_world` below this
        # runs `rmtree`/`copytree`/`chown`/`chmod` (`_seal_world_store`'s own work), and a bare
        # filesystem fault there used to escape this "TRULY never raises" (M3) gate untyped.
        # N4, p6-review-r2: a raise from `reset_world` (via `_clone_or_reset_world`) mid-loop
        # carries whatever THIS world's own partial reseal produced (`exc.partial_handles`) —
        # merged here so a live engine the gate's own reset just spawned is not left unreferenced
        # in `world_handles` the moment the gate degrades instead of propagating the failure.
        partial = getattr(exc, "partial_handles", None)
        if partial is not None and world_index is not None:
            world_handles[world_index] = partial
        # The cause cannot travel in the returned `reason` string — that value is §5's own closed
        # `parallelism_degraded.reason` vocabulary (`conformance_gate_failed` | `fixed_port`) — so
        # "loud" means logged here, not encoded in the return.
        if isinstance(exc, ProcessRuntimeError):
            logger.error(
                "conformance gate raised %s/%s: %s degrading to effective parallelism 1 instead "
                "of failing the job",
                exc.stage,
                exc.code,
                exc,
            )
        else:
            logger.error(
                "conformance gate raised %s: %s degrading to effective parallelism 1 instead of "
                "failing the job",
                type(exc).__name__,
                exc,
            )
        return False, "conformance_gate_failed"


# --- §0.3/§4 rule 1 secrets lifetime (B3, p6-review-r1) ---------------------------------------


def _read_job_id(work_directory: Path) -> str:
    """Return the stable job identifier used by explicitly declared ``{{JOB_ID}}`` templates.

    Hosted execution guarantees ``/work/job.json``. Local tests and direct SDK process-runtime
    use may omit it, so they receive a deterministic local-lane value rather than failing for a
    field needed only to isolate shared external dispatch namespaces.
    """
    path = work_directory / "job.json"
    if not path.is_file():
        return "local"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "local"
    if not isinstance(raw, dict):
        return "local"
    value = str(raw.get("job_id") or "").strip()
    return value or "local"


def _read_job_secret_purposes(work_directory: Path) -> dict[str, str]:
    """§0.2/§4.1: `/work/job.json` is the provisioner's own configuration source — `agent.
    secret_refs` (§1: `{alias: {manager, key, version, purpose}}`) is where each alias's REAL
    purpose comes from — never relabelled `target_provider` unconditionally, which would silently
    defeat `select_process_secrets`'s own `SOURCE_CHECKOUT` exclusion the moment a
    `source_checkout` alias ever reaches `secrets.json`. The two raises below use `spawn_failed`
    for a malformed `job.json`, which is a vocabulary stretch — §2f's domain rule for it is
    "infrastructure if a managed engine, `agent` if source," neither of which is what a
    config-read fault actually is; picked as the closest §4.6 code available, domain
    `infrastructure` (a corrupted upload, not a bundle-authoring fault the customer caused).
    """
    job_json_path = work_directory / "job.json"
    if not job_json_path.exists():
        # A real gap on the hosted path (§0.2 guarantees the file), not a local-lane default to
        # paper over silently — the caller still proceeds (a constructor-supplied `secret_purpose_
        # map`, or the local/test lane, never needs this file at all), but every alias in
        # `secrets.json` then has no purpose to match and is correctly dropped below, which would
        # otherwise look like a silent injection failure three layers down with nothing in the
        # log explaining why.
        logger.warning(
            "secrets: %s is absent; every alias in secrets.json will be dropped (no purpose to "
            "match) unless a secret_purpose_map was supplied",
            job_json_path,
        )
        return {}
    try:
        raw = json.loads(job_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProcessRuntimeError(
            "secrets",
            "spawn_failed",
            f"{job_json_path}: unreadable or not valid JSON: {exc}",
            domain=FailureDomain.INFRASTRUCTURE,
        ) from exc
    if not isinstance(raw, dict):
        raise ProcessRuntimeError(
            "secrets",
            "spawn_failed",
            f"{job_json_path}: expected a JSON object, got {type(raw).__name__}",
            domain=FailureDomain.INFRASTRUCTURE,
        )
    agent = raw.get("agent")
    refs = agent.get("secret_refs") if isinstance(agent, dict) else None
    if not isinstance(refs, dict):
        return {}
    return {
        str(alias): ref["purpose"]
        for alias, ref in refs.items()
        if isinstance(ref, dict) and isinstance(ref.get("purpose"), str)
    }


def _load_and_delete_secrets(
    secrets_path: Path,
    *,
    work_directory: Path,
    secret_purpose_map: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """§0 step 3 / §4 rule 1's lifetime rule: "the provisioner loads this file into memory at
    startup and deletes it immediately after loading, BEFORE ANY CUSTOMER PROCESS STARTS. The
    in-memory map lives for the whole job — `reset` restarts and `provision` reconciliations
    re-inject from memory."

    Each alias's purpose comes from `job.json`'s own `agent.secret_refs`
    (`_read_job_secret_purposes`) — NOT invented as `target_provider` for every alias, which would
    retire `select_process_secrets`'s own `SOURCE_CHECKOUT` exclusion the moment a
    `source_checkout` alias ever reached this file. `secret_purpose_map`, when given, overrides
    the job.json read entirely (the local/test lane's own shape — no `/work/job.json` on a dev
    box). An alias in `secrets.json` with no matching ref anywhere is dropped, not injected under
    a guessed purpose, and logged — `select_process_secrets` naturally never matches an alias
    absent from the purposes map, so nothing further is needed to enforce the drop.
    """
    if not secrets_path.exists():
        values: dict[str, str] = {}
    else:
        try:
            raw = json.loads(secrets_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProcessRuntimeError(
                "secrets",
                "spawn_failed",
                f"{secrets_path}: unreadable or not valid JSON: {exc}",
                domain=FailureDomain.INFRASTRUCTURE,
            ) from exc
        secrets_path.unlink(missing_ok=True)
        if not isinstance(raw, dict):
            raise ProcessRuntimeError(
                "secrets",
                "spawn_failed",
                f"{secrets_path}: expected a JSON object of alias -> value, got "
                f"{type(raw).__name__}",
                domain=FailureDomain.INFRASTRUCTURE,
            )
        values = {str(alias): str(value) for alias, value in raw.items()}

    all_purposes = (
        dict(secret_purpose_map)
        if secret_purpose_map is not None
        else _read_job_secret_purposes(work_directory)
    )
    dropped = sorted(alias for alias in values if alias not in all_purposes)
    if dropped:
        logger.warning(
            "secrets: %d alias(es) with no matching agent.secret_refs entry will never be "
            "injected into any process: %s",
            len(dropped),
            dropped,
        )
    purposes = {alias: all_purposes[alias] for alias in values if alias in all_purposes}
    return values, purposes


# --- §4 provision / close: the stateful RuntimeProvider adapter -----------------------------------


class ProcessRuntimeProvider:
    """The §4 `RuntimeProvider`, structurally: a stateful adapter around this module's pure
    functions (`spawn_world`, `freeze_baseline`, `reset_world`, `run_conformance_gate`, ...),
    holding what a §3-shaped `EnvironmentRuntime` cannot carry itself — live process handles, the
    job's generated credentials, and the sealed baseline. `runtime.py`'s own `RuntimeProvider`
    Protocol is untouched (P5's docstring: "a later phase wires a Protocol-conforming, stateful
    adapter around the pure functions below") — wiring the existing local-SDK callers onto this
    shape is the refactor-sized "Implementation delta" §4 itself calls out, a separate change from
    this one; `provision`/`reset`/`close` below match §4's method shapes (not `runtime.py`'s older
    v1 ones — no `provider` field, `provision` takes `instances` and returns a list, `close` takes
    no `runtime` argument), since those are what this phase actually implements.

    Idempotency (§4 rule 1, "idempotent for the job identity") holds for repeated calls against the
    SAME instance — the normal in-job shape: the entrypoint constructs one provider per job and
    calls `provision` once, then `reset`/`provision` again for sick-world recovery, all against
    that one long-lived object for the guest process's whole life. Resuming after the GUEST
    PROCESS ITSELF restarts (a fresh `hosted_entrypoint` invocation after a crash) would need
    filesystem-observable state beyond what this class attempts — `write_build_output`'s own
    `inputs_digest` reuse (§2c: "the provisioner records it... as the baseline identity for
    attempt-retry reuse") is the one piece of that this module produces; resurrecting LIVE process
    handles across a process boundary is the entrypoint's own concern, out of this phase's file
    scope.

    n2, p6-review-r2: this class takes no lock over its own mutable state (`_world_handles`,
    `_runtimes`, `_context`, `_secrets_loaded`, ...) behind the `asyncio.to_thread` calls below —
    that is deliberate, not an oversight. v1.12 §4.5b makes the port NON-REENTRANT and puts
    serialization on the SCHEDULER, never the provider: every `provision`/`reset`/`healthy`/
    `close` call for one job must be serialized by the caller — `healthy` demotes state, so it
    writes and is in the set too (Q9, p6-review-r3). Do not add a lock here on the assumption one
    is missing; add it at the call site instead.
    """

    name = "hosted-process"

    def __init__(
        self,
        *,
        runner: ProcessRunner = default_process_runner,
        sync_run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        prober: CapabilityProber = default_capability_prober,
        sql_runner: SqlRunner = default_sql_runner,
        redis_runner: RedisCommandRunner = default_redis_command_runner,
        rabbitmq_inspector: RabbitmqQueueInspector = default_rabbitmq_queue_inspector,
        rabbitmq_declare: RabbitmqQueueDeclarer = default_rabbitmq_queue_declare_and_publish,
        rabbitmq_delete: RabbitmqQueueDeleter = default_rabbitmq_queue_delete,
        rabbitmq_import: RabbitmqDefinitionsImporter = default_rabbitmq_definitions_importer,
        copy: Callable[[Path, Path], None] | None = None,
        user_resolver: Callable[
            [str], "pwd.struct_passwd | None"
        ] = default_user_resolver,
        chown: Callable[[Path, int, int], None] = _default_chown,
        build_step_timeout_seconds: float = _DEFAULT_BUILD_STEP_TIMEOUT_SECONDS,
        token: Callable[[], str] | None = None,
        secrets_path: Path = Path("/run/futureagi/secrets.json"),
        close_wait_timeout_seconds: float = _TERMINATE_WAIT_SECONDS,
        secret_purpose_map: dict[str, str] | None = None,
        require_declared_user: bool = True,
        public_url_resolver: Callable[[int, int], str] | None = None,
        provider_attempt_id: str | None = None,
        provider_expires_at: datetime | None = None,
    ) -> None:
        self._runner = runner
        self._sync_run = sync_run
        self._prober = prober
        self._sql_runner = sql_runner
        self._redis_runner = redis_runner
        self._rabbitmq_inspector = rabbitmq_inspector
        self._rabbitmq_declare = rabbitmq_declare
        self._rabbitmq_delete = rabbitmq_delete
        self._rabbitmq_import = rabbitmq_import
        self._copy = copy
        self._user_resolver = user_resolver
        self._chown = chown
        self._build_step_timeout_seconds = build_step_timeout_seconds
        self._token = token
        self._secrets_path = secrets_path
        # N12, p6-review-r2: `close()`'s own bound on the per-handle wait/kill escalation — §0.7's
        # 120s flush window is shared by every handle `_teardown_processes_and_directories` tears
        # down, so a caller close to that deadline can pass something tighter than the module
        # default without touching every OTHER termination call site's own timeout.
        self._close_wait_timeout_seconds = close_wait_timeout_seconds
        # N10, p6-review-r2: overrides the `/work/job.json` read entirely when supplied — the
        # local/test lane's own shape (no job.json on a dev box); a hosted caller leaves this
        # `None` and `_load_and_delete_secrets` reads `agent.secret_refs` itself.
        self._secret_purpose_map = secret_purpose_map
        # Construction-time default for `provision(require_declared_user=...)` when the caller
        # passes nothing (WorldPool does). Daytona forces the sandbox to a fixed non-root user
        # (svc-control) and rejects os_user overrides, so the hosted guest constructs this False +
        # a null user_resolver to run every process uniformly, since setuid/chown to svc-* is
        # impossible there.
        self._require_declared_user = require_declared_user
        self._public_url_resolver = public_url_resolver
        self._provider_attempt_id = provider_attempt_id
        self._provider_expires_at = provider_expires_at

        self._manifest: EnvironmentBundleV2 | None = None
        self._bundle_digest: str | None = None
        self._context: SpawnContext | None = None
        self._build_output: BuildOutput | None = None
        self._job_shared_handles: dict[str, SpawnedWorldProcess] = {}
        self._world_handles: dict[int, dict[str, SpawnedWorldProcess]] = {}
        self._runtimes: dict[int, EnvironmentRuntime] = {}
        self._conformance_checked = False
        # B3, p6-review-r1: loaded once (`_load_and_delete_secrets`), on the FIRST provision call
        # for a job identity — `_secrets_loaded` latches so neither a later reconcile call nor a
        # bundle-digest rebuild (M6) ever tries to re-read a file `close()`/the load itself has
        # already deleted; §0.3's "re-inject from memory" is exactly these two dicts surviving on
        # `self` across every `provision`/`reset` call for the object's whole life.
        self._secret_values: dict[str, str] = {}
        self._secret_purposes: dict[str, str] = {}
        self._secrets_loaded = False
        self._provider_receipts: dict[int, ProviderProvisionReceipt] = {}
        self._provider_contexts: dict[int, ProviderContext] = {}

    async def provision(
        self,
        bundle: EnvironmentBundleV2,
        *,
        source: Path,
        bundle_dir: Path,
        work_directory: Path,
        contract: Any
        | None = None,  # accepted for §4 shape-compatibility; not consumed here —
        # evidence-seam wiring is out of this phase's scope, same as `runtime.py`'s own providers.
        instances: int = 1,
        require_declared_user: bool | None = None,
    ) -> list[EnvironmentRuntime]:
        import asyncio

        # None => use the construction-time default (WorldPool passes nothing; the hosted guest
        # constructs the provider with require_declared_user=False for Daytona's fixed-user sandbox).
        if require_declared_user is None:
            require_declared_user = self._require_declared_user
        return await asyncio.to_thread(
            self._provision_sync,
            bundle,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=work_directory,
            instances=instances,
            require_declared_user=require_declared_user,
        )

    def _provision_sync(
        self,
        bundle: EnvironmentBundleV2,
        *,
        source: Path,
        bundle_dir: Path,
        work_directory: Path,
        instances: int,
        require_declared_user: bool,
    ) -> list[EnvironmentRuntime]:
        port_plan = plan_ports(bundle, instances=instances)
        effective = port_plan.effective_instances
        bundle_digest = bundle.digest

        if self._manifest is None or self._bundle_digest != bundle_digest:
            if self._manifest is not None:
                # M6, p6-review-r1: a bundle-digest change used to reassign this instance's own
                # identity fields straight over the PREVIOUS job's still-running processes and
                # still-allocated ports — §4.1's "never duplicates" broken in the one case this
                # branch exists for. Full teardown first (same mechanism `close()` uses, minus
                # the secrets unlink — this is a re-sealed bundle, not a new job identity, so the
                # in-memory secret map must survive it).
                self._teardown_processes_and_directories(work_directory)
            if not self._secrets_loaded:
                # B3 / §0.3, p6-review-r1: loaded and the file deleted BEFORE any customer
                # process starts — done here, before `build_process_trees`/`freeze_baseline`
                # below ever spawn anything. N10, p6-review-r2: purposes come from `job.json`'s
                # own `agent.secret_refs` (or the constructor override), never invented.
                self._secret_values, self._secret_purposes = _load_and_delete_secrets(
                    self._secrets_path,
                    work_directory=work_directory,
                    secret_purpose_map=self._secret_purpose_map,
                )
                self._secrets_loaded = True

            # First call for this job identity, or the bundle changed underneath it (§2c: a
            # digest mismatch forces a rebuild) — build once, seed+freeze once, gate once.
            credentials = generate_engine_credentials(bundle, token=self._token)
            context = SpawnContext(
                work_directory=work_directory,
                port_plan=port_plan,
                credentials=credentials,
                secret_values=self._secret_values,
                secret_purposes=self._secret_purposes,
                job_id=_read_job_id(work_directory),
                runner=self._runner,
                sync_run=self._sync_run,
                prober=self._prober,
                copy=self._copy,
                user_resolver=self._user_resolver,
                require_declared_user=require_declared_user,
                chown=self._chown,
                build_step_timeout_seconds=self._build_step_timeout_seconds,
                sql_runner=self._sql_runner,
                redis_runner=self._redis_runner,
                rabbitmq_inspector=self._rabbitmq_inspector,
                rabbitmq_declare=self._rabbitmq_declare,
                rabbitmq_delete=self._rabbitmq_delete,
                rabbitmq_import=self._rabbitmq_import,
                bundle_dir=bundle_dir,
            )
            # N3, p6-review-r2 (MAJOR): the job identity (`_manifest`/`_bundle_digest`) is
            # committed only AFTER `build_process_trees`/`freeze_baseline` both succeed —
            # committing it first used to leave a failed first `provision()` claiming an identity
            # with no build output, so an in-process retry (the same bundle, e.g. `hosted_
            # scheduler.py`'s background `_reconcile`) took the RECONCILE branch below instead of
            # retrying the build, and hit the "context/build_output unset" invariant instead of
            # the real, often-retryable cause.
            try:
                build_process_trees(bundle, source_root=source, context=context)
                freeze_result = freeze_baseline(
                    bundle, bundle_digest=bundle_digest, context=context
                )
            except (OSError, shutil.Error) as exc:
                # §4.6 — filesystem failures during provisioning are `infrastructure`. This
                # phase's own copy-heavy work (build trees, baseline snapshot/seal) can raise a raw
                # filesystem error covering EITHER a source build tree or a managed baseline seal —
                # `store_statement_failed` is a vocabulary stretch for a build-tree copy fault (the
                # closest §2f code in a closed table with no generic `provisioner_io_failed`), but
                # the domain is `infrastructure` either way, so it is set directly here rather than
                # left to the fallback map.
                self._manifest = None
                self._bundle_digest = None
                raise ProcessRuntimeError(
                    "baseline",
                    "store_statement_failed",
                    str(exc),
                    domain=FailureDomain.INFRASTRUCTURE,
                ) from exc
            except BaseException:
                self._manifest = None
                self._bundle_digest = None
                raise
            self._manifest = bundle
            self._bundle_digest = bundle_digest
            self._context = context
            self._build_output = freeze_result.build_output
            self._job_shared_handles = freeze_result.job_shared_handles
            write_build_output(work_directory, self._build_output)
            self._world_handles = {}
            self._runtimes = {}
            self._conformance_checked = False
        else:
            # Same job identity, a later reconcile call — `instances`/`require_declared_user` may
            # legitimately differ (a sick-world recovery re-call), and so may `work_directory`/
            # `bundle_dir` (m7, p6-review-r1: the two used to silently diverge from whatever this
            # call actually passed, since only `port_plan`/`require_declared_user` were carried
            # forward here). The sealed baseline never re-runs regardless (§4 rule 1).
            self._context = replace(
                self._context,
                port_plan=port_plan,
                require_declared_user=require_declared_user,
                work_directory=work_directory,
                bundle_dir=bundle_dir,
            )

        context = self._context
        build_output = self._build_output
        if context is None or build_output is None:
            # m6, p6-review-r1: both branches above set these unconditionally — a bare `assert`
            # here is stripped under `python -O`, which would turn a real "this is a bug in THIS
            # function" precondition into an opaque `AttributeError` a few lines down instead of
            # a typed failure.
            raise ProcessRuntimeError(
                "provision",
                _INTERNAL_INVARIANT_VIOLATED,
                "context/build_output unset after the first-call/reconcile branch",
            )

        requested = instances
        # At requested=1 nothing can degrade (effective=1 too, so no valid
        # `parallelism_degraded` payload exists; outbound-channels.md's `1 <= effective <
        # requested` bound is empty at requested=1). `port_plan.degraded_reason` is already
        # None at instances=1, but that alone is not sufficient — see the
        # `build_output.degrade_reason` write below, which is what actually enforces it
        # against the conformance-gate paths that reassign `degrade_reason` further down.
        degrade_reason = port_plan.degraded_reason if effective < requested else None

        if effective > 1 and not self._conformance_checked:
            self._ensure_world(0)
            self._ensure_world(1)
            passed, reason = run_conformance_gate(
                bundle,
                context=context,
                baseline=build_output,
                job_shared_handles=self._job_shared_handles,
                world_handles=self._world_handles,
            )
            build_output.conformance = passed
            build_output.conformance_reason = reason
            self._conformance_checked = True
            if not passed:
                effective = 1
                degrade_reason = reason
        elif self._conformance_checked and build_output.conformance is False:
            # A degrade decided by an EARLIER call must keep holding on every later reconcile call
            # too — `effective` above is freshly recomputed from `port_plan` each time and knows
            # nothing about a gate result from a call that already happened.
            effective = 1
            degrade_reason = "conformance_gate_failed"

        build_output.requested_parallelism = requested
        build_output.effective_parallelism = effective
        # The conformance-gate branches above reassign `degrade_reason` unconditionally (they
        # only know a gate result, not this call's `requested`) — re-checking the invariant here,
        # at the single write site, covers those paths too instead of only the fixed_port input.
        build_output.degrade_reason = degrade_reason if effective < requested else None
        write_build_output(work_directory, build_output)

        # Reconcile down first — a prior call may have over-provisioned (the canary's own 2-world
        # pair, before a gate failure dropped `effective` to 1).
        for stale_index in [index for index in self._runtimes if index >= effective]:
            self._teardown_world(stale_index)
        for world_index in range(effective):
            self._ensure_world(world_index)
        # t3 / §4.1, p6-review-r1: "reconciles to exactly `instances` READY worlds" —
        # `_ensure_world` only ever leaves a (re)built world `PREPARING` (§3's transition table
        # promotes it later); promoted here via the same declared-readiness probe `healthy()`
        # uses, so `provision()` itself returns worlds already at their reachable terminal state
        # instead of leaving every caller to independently discover it must call `healthy()`
        # first. A world left `READY`/`UNHEALTHY` by an earlier call is never re-probed here
        # (mirrors `healthy()`'s own F6 "never promote except from PREPARING" rule).
        for world_index in range(effective):
            runtime = self._runtimes[world_index]
            if runtime.state is RuntimeState.PREPARING:
                # N2, p6-review-r2 (BLOCKER): polls (`_poll_runtime_health`), never samples once —
                # nothing waits for the DAG's terminal process (`spawn_world` only waits on
                # `depends_on` edges), so a single-shot probe here marked a bundle whose readiness-
                # bearing process is terminal `UNHEALTHY` on every provision, regardless of how
                # quickly it actually came up.
                healthy_now = _poll_runtime_health(
                    self._manifest, runtime, prober=context.prober
                )
                runtime.state = (
                    RuntimeState.READY if healthy_now else RuntimeState.UNHEALTHY
                )

        for world_index in range(effective):
            runtime = self._runtimes[world_index]
            if runtime.state is RuntimeState.READY:
                self._ensure_provider_target(world_index, runtime)

        return [self._runtimes[index] for index in range(effective)]

    def _provider_spec(self) -> ProviderLifecycleSpec | None:
        if self._manifest is None:
            return None
        raw = self._manifest.metadata.get("provider_lifecycle")
        if not isinstance(raw, dict):
            return None
        try:
            return ProviderLifecycleSpec.model_validate(raw)
        except ValueError as exc:
            raise ProcessRuntimeError(
                "provider_lifecycle",
                "build_failed",
                f"provider lifecycle metadata is invalid: {exc}",
                domain=FailureDomain.ENVIRONMENT,
            ) from exc

    def _provider_import_spec(self) -> ProviderImportSpec | None:
        if self._manifest is None:
            return None
        raw = self._manifest.metadata.get("provider_import")
        if not isinstance(raw, dict):
            return None
        try:
            return ProviderImportSpec.model_validate(raw)
        except ValueError as exc:
            raise ProcessRuntimeError(
                "provider_import",
                "build_failed",
                f"provider import metadata is invalid: {exc}",
                domain=FailureDomain.ENVIRONMENT,
            ) from exc

    def _ensure_provider_target(
        self, world_index: int, runtime: EnvironmentRuntime
    ) -> None:
        spec = self._provider_spec()
        import_spec = self._provider_import_spec()
        if (
            spec is None and import_spec is None
        ) or world_index in self._provider_receipts:
            return
        if self._context is None or self._manifest is None:
            raise ProcessRuntimeError(
                "provider_lifecycle",
                _INTERNAL_INVARIANT_VIOLATED,
                "provider lifecycle ran before process runtime context existed",
            )
        if import_spec is not None:
            self._ensure_imported_provider_target(world_index, runtime, import_spec)
            return
        assert spec is not None  # narrowed by the early return above
        endpoint = runtime.endpoints.get(spec.public_capability)
        if endpoint is None or endpoint.protocol != CapabilityProtocol.HTTP.value:
            raise ProcessRuntimeError(
                "provider_lifecycle",
                "build_failed",
                f"public capability {spec.public_capability!r} must resolve to HTTP",
                domain=FailureDomain.ENVIRONMENT,
            )
        port = urlsplit(endpoint.address).port
        if not port or self._public_url_resolver is None:
            raise ProcessRuntimeError(
                "provider_lifecycle",
                "spawn_failed",
                "a hosted public URL resolver is required for provider webhooks and tools",
                domain=FailureDomain.INFRASTRUCTURE,
            )
        public_base_url = self._public_url_resolver(port, 7200).rstrip("/")
        process_name = spec.process or self._manifest.runtime.control_service
        process = next(
            (entry for entry in self._manifest.processes if entry.name == process_name),
            None,
        )
        if not isinstance(process, SourceProcess):
            raise ProcessRuntimeError(
                "provider_lifecycle",
                "build_failed",
                f"provider lifecycle process {process_name!r} is not a source process",
                domain=FailureDomain.ENVIRONMENT,
            )
        lifecycle_directory = (
            self._context.work_directory / "provider" / f"w{world_index}"
        )
        lifecycle_directory.mkdir(parents=True, exist_ok=True)
        context = ProviderContext(
            attempt_id=self._provider_attempt_id or self._context.job_id,
            world_id=str(world_index),
            provider=spec.type,
            public_base_url=public_base_url,
            event_url=public_base_url + spec.event_path,
            tool_base_url=public_base_url + spec.tool_path,
            provider_resource_prefix=f"alk-{self._context.job_id[:12]}-w{world_index}",
            idempotency_key=f"{self._context.job_id}:{self._bundle_digest}:w{world_index}",
            expires_at=self._provider_expires_at
            or datetime.fromtimestamp(time.time() + 7200, tz=timezone.utc),
        )
        context_path = lifecycle_directory / "context.json"
        context_path.write_text(context.model_dump_json(indent=2), encoding="utf-8")
        context_path.chmod(0o600)
        secret_values = {
            name: self._secret_values.get(name, "") for name in spec.required_secrets
        }
        invocation = build_lifecycle_invocation(
            command=spec.provision,
            # Lifecycle code and its per-process dependency installation live in the immutable
            # build tree. The world scratch directory contains only writable runtime state, so
            # using it as cwd makes a repository command such as `python provider_target.py`
            # fail before it can contact the provider.
            source_directory=build_tree_dir(
                self._context.work_directory, process_name
            ),
            lifecycle_directory=lifecycle_directory,
            context_path=context_path,
            context=context,
            required_secret_values=secret_values,
            output_relative_path=spec.output,
        )
        try:
            returncode = run_lifecycle_invocation(
                invocation,
                log_path=lifecycle_directory / "provision.log",
                run=self._sync_run,
            )
            if returncode != 0:
                raise ProviderLifecycleError(
                    f"provider_provision_failed: command exited {returncode}"
                )
            receipt = validate_provision_receipt(
                invocation.output_path,
                context=context,
                secret_values=secret_values,
            )
        except ProviderLifecycleError as exc:
            raise ProcessRuntimeError(
                "provider_lifecycle",
                "spawn_failed",
                str(exc),
                process=process_name,
                domain=FailureDomain.ENVIRONMENT,
            ) from exc
        self._provider_receipts[world_index] = receipt
        self._provider_contexts[world_index] = context
        runtime.metadata["provider_target_id"] = receipt.target.id
        runtime.metadata["provider_target_kind"] = receipt.target.kind

    def _ensure_imported_provider_target(
        self,
        world_index: int,
        runtime: EnvironmentRuntime,
        spec: ProviderImportSpec,
    ) -> None:
        """Clone an existing provider target only after its isolated backend is ready."""
        assert self._context is not None
        endpoint = runtime.endpoints.get(spec.public_capability)
        if endpoint is None or endpoint.protocol != CapabilityProtocol.HTTP.value:
            raise ProcessRuntimeError(
                "provider_import",
                "build_failed",
                f"public capability {spec.public_capability!r} must resolve to HTTP",
                domain=FailureDomain.ENVIRONMENT,
            )
        port = urlsplit(endpoint.address).port
        if not port or self._public_url_resolver is None:
            raise ProcessRuntimeError(
                "provider_import",
                "spawn_failed",
                "a hosted public URL resolver is required for imported provider tools",
                domain=FailureDomain.INFRASTRUCTURE,
            )
        public_base_url = self._public_url_resolver(port, 7200).rstrip("/")
        context = ProviderContext(
            attempt_id=self._provider_attempt_id or self._context.job_id,
            world_id=str(world_index),
            provider=spec.type,
            public_base_url=public_base_url,
            event_url=public_base_url + spec.event_path,
            tool_base_url=public_base_url + spec.tool_path,
            provider_resource_prefix=f"alk-{self._context.job_id[:12]}-w{world_index}",
            idempotency_key=f"{self._context.job_id}:{self._bundle_digest}:w{world_index}",
            expires_at=self._provider_expires_at
            or datetime.fromtimestamp(time.time() + 7200, tz=timezone.utc),
        )
        secret_name = "VAPI_API_KEY" if spec.type.value == "vapi" else "RETELL_API_KEY"
        try:
            receipt = clone_provider_target(
                spec,
                context=context,
                api_key=self._secret_values.get(secret_name, ""),
            )
        except ProviderImportError as exc:
            raise ProcessRuntimeError(
                "provider_import",
                "spawn_failed",
                str(exc),
                domain=FailureDomain.ENVIRONMENT,
            ) from exc
        self._provider_receipts[world_index] = receipt
        self._provider_contexts[world_index] = context
        runtime.metadata["provider_target_id"] = receipt.target.id
        runtime.metadata["provider_target_kind"] = receipt.target.kind
        runtime.metadata["provider_import_source_target_id"] = spec.source_target_id

    def _destroy_provider_target(self, world_index: int) -> None:
        receipt = self._provider_receipts.pop(world_index, None)
        context = self._provider_contexts.pop(world_index, None)
        spec = self._provider_spec()
        import_spec = self._provider_import_spec()
        if receipt is None or context is None or self._context is None:
            return
        if import_spec is not None:
            secret_name = (
                "VAPI_API_KEY" if import_spec.type.value == "vapi" else "RETELL_API_KEY"
            )
            try:
                destroy_imported_target(
                    import_spec,
                    receipt=receipt,
                    api_key=self._secret_values.get(secret_name, ""),
                )
            except ProviderImportError as exc:
                logger.error(
                    "provider import cleanup failed for world %s: %s", world_index, exc
                )
            return
        if spec is None:
            return
        process_name = spec.process or self._manifest.runtime.control_service
        lifecycle_directory = (
            self._context.work_directory / "provider" / f"w{world_index}"
        )
        invocation = build_lifecycle_invocation(
            command=spec.destroy,
            source_directory=build_tree_dir(
                self._context.work_directory, process_name
            ),
            lifecycle_directory=lifecycle_directory,
            context_path=lifecycle_directory / "context.json",
            context=context,
            required_secret_values={
                name: self._secret_values.get(name, "")
                for name in spec.required_secrets
            },
            output_relative_path=spec.output,
            receipt_path=lifecycle_directory / spec.output,
        )
        try:
            returncode = run_lifecycle_invocation(
                invocation,
                log_path=lifecycle_directory / "destroy.log",
                run=self._sync_run,
            )
            if returncode != 0:
                logger.error(
                    "provider cleanup failed for world %s: exit %s",
                    world_index,
                    returncode,
                )
        except ProviderLifecycleError as exc:
            logger.error("provider cleanup failed for world %s: %s", world_index, exc)

    def _ensure_world(self, world_index: int) -> None:
        """§4 rule 1: "completes or replaces partial/unhealthy worlds and never duplicates." A
        world already `ready`/`preparing` is left alone (never duplicated); anything else (absent,
        `unhealthy`, `stopped`) is (re)built from the sealed baseline via `_clone_or_reset_world`
        — the same primitive `reset_world` uses, so a sick-world replace and a first-time clone are
        one code path, not two.
        """
        if (
            self._manifest is None
            or self._context is None
            or self._build_output is None
        ):
            # m6, p6-review-r1: real precondition (only ever called from within `_provision_sync`,
            # after all three are set) — typed, not a bare `assert` an `-O` run would strip.
            raise ProcessRuntimeError(
                "provision",
                _INTERNAL_INVARIANT_VIOLATED,
                "_ensure_world called before manifest/context/build_output were established",
            )
        existing = self._runtimes.get(world_index)
        if existing is not None and existing.state in (
            RuntimeState.READY,
            RuntimeState.PREPARING,
        ):
            return

        # N11, p6-review-r2. `current_world_index`: Q7, p6-review-r3 — a respawn reseals every
        # tracked world's database, so every world OTHER than this one must be demoted for the
        # scheduler to notice and repair it.
        self._ensure_job_shared_handles_alive(current_world_index=world_index)
        try:
            result = _clone_or_reset_world(
                self._manifest,
                world_index,
                context=self._context,
                baseline=self._build_output,
                job_shared_handles=self._job_shared_handles,
                existing_handles=self._world_handles.get(world_index, {}),
            )
        except (OSError, shutil.Error) as exc:
            # A raw filesystem fault this deep in `_clone_or_reset_world` (mkdir/chown/chmod for a
            # (re)spawned process) does not by itself say which process's kind failed — `domain`
            # is left unset so the §4 seam's fallback map applies (SECTION_2F_DOMAIN), since every
            # raise site that DOES know its process kind already carries its own domain directly.
            partial = getattr(exc, "partial_handles", None)
            if partial is not None:
                self._world_handles[world_index] = (
                    partial  # never orphan a live engine.
                )
            raise ProcessRuntimeError("spawn", "spawn_failed", str(exc)) from exc
        except BaseException as exc:
            # A raise partway through `_clone_or_reset_world`/`spawn_
            # world` used to drop every ALREADY-(re)sealed/spawned handle of THIS world on the
            # floor — nothing held it, so `close()`'s own `rmtree` of this world's data directory
            # next ran against a still-live server. `exc.partial_handles` (set by `freeze_
            # baseline`/`_clone_or_reset_world`/`spawn_world`'s own try/finally) is whatever this
            # (re)build managed before failing; published here so the next reconcile/close can
            # terminate it instead.
            partial = getattr(exc, "partial_handles", None)
            if partial is not None:
                self._world_handles[world_index] = partial
            raise

        self._world_handles[world_index] = result.handles
        # N1, p6-review-r2 (BLOCKER): MUTATES the existing `EnvironmentRuntime` in place rather
        # than constructing a replacement — v1.12 §4.5b's live-object model ("providers hand out
        # live `EnvironmentRuntime` objects") reads as ONE object per world for the provider's
        # whole life. Minting a new object every rebuild meant `reset()`'s own state write (m5)
        # landed on an object no caller who captured an EARLIER reference would ever see again.
        # Static control metadata exposes manifest-declared paths while dispatch metadata reads
        # the effective values from the launched process.  Keep both; effective runtime values
        # win when a declaration was rendered or overridden during provisioning.
        metadata = {
            **self._control_metadata(world_index),
            **_dispatch_metadata(result.handles),
        }
        if existing is not None:
            existing.runtime_id = new_runtime_id(self._bundle_digest, world_index)
            existing.endpoints = result.endpoints
            existing.state = RuntimeState.PREPARING
            existing.metadata = metadata
            self._runtimes[world_index] = existing
        else:
            self._runtimes[world_index] = EnvironmentRuntime(
                runtime_id=new_runtime_id(self._bundle_digest, world_index),
                world_index=world_index,
                bundle_digest=self._bundle_digest,
                state=RuntimeState.PREPARING,
                endpoints=result.endpoints,
                metadata=metadata,
            )

    def _control_metadata(self, world_index: int) -> dict[str, str]:
        """Per-world facts the CallRunner needs, surfaced on `EnvironmentRuntime.metadata`:
        `livekit_agent_name` (the agent's LiveKit dispatch name) and `tool_trace_path` (where the
        agent writes its function-call trace). Both are rendered from the control service's declared
        environment (`LIVEKIT_AGENT_NAME` / `HARNESS_TOOL_TRACE`); a bundle that does not set them
        simply omits the key, and the CallRunner then fails typed-and-loud rather than guessing."""
        control = self._manifest.runtime.control_service
        process = next((p for p in self._manifest.processes if p.name == control), None)
        env = dict(getattr(process, "environment", {}) or {})
        world_dir = world_scratch_dir(
            self._context.work_directory, world_index, control
        )

        def _fill(value: str) -> str:
            return (
                value.replace("{{JOB_ID}}", self._context.job_id)
                .replace("{{WORLD_INDEX}}", str(world_index))
                .replace("{{WORLD_DIR}}", str(world_dir))
            )

        metadata: dict[str, str] = {}
        agent_name = env.get("LIVEKIT_AGENT_NAME")
        if not agent_name:
            # A freshly authored bundle may not render LIVEKIT_AGENT_NAME onto the control
            # process when the agent reads its own default via
            # os.environ.get("LIVEKIT_AGENT_NAME", "<default>"). Recover that source default so a
            # fresh bundle carries the dispatch identity like a cached one and the CallRunner can
            # dispatch instead of failing voice_dispatch_identity_unavailable.
            agent_name = _agent_name_source_default(Path("/work/source"))
        if agent_name:
            metadata["livekit_agent_name"] = _fill(agent_name)
        trace = env.get("HARNESS_TOOL_TRACE")
        if trace:
            metadata["tool_trace_path"] = _fill(trace)
        return metadata

    def _drop_world_shared_databases(self, world_index: int) -> None:
        """m4, p6-review-r1: reconciling W down (e.g. after a conformance-gate degrade from 3 to
        1) used to leave `w1`/`w2` behind forever on a job-shared `template_database` engine —
        nothing ever DROPs a world's logical DB on teardown, only on REUSE (`_seal_world_store`'s
        own `IF EXISTS`). A space leak, not a correctness one; best-effort (logged, not raised) so
        a teardown-time drop failure never blocks the reconcile it is cleaning up after.
        """
        if self._manifest is None or self._context is None:
            return
        store_by_process = _store_by_process_name(self._manifest)
        for process in self._manifest.processes:
            if (
                not isinstance(process, ManagedProcess)
                or process.engine is not ManagedEngine.POSTGRES
            ):
                continue
            store = store_by_process.get(process.name)
            if (
                store is None
                or store.baseline.strategy is not BaselineStrategy.TEMPLATE_DATABASE
            ):
                continue
            if process.name not in self._job_shared_handles:
                continue
            credentials = self._context.credentials.get(process.name)
            if credentials is None:
                continue
            port = self._context.port_plan.port_for(process.name, world_index)
            world_db = f"w{world_index}"
            try:
                # N16, p6-review-r2 (MINOR): mirrors `_reset_template_database`'s own sibling
                # call — a lingering backend on `world_db` (e.g. a scenario connection that never
                # closed) otherwise blocks this DROP exactly the way it would block a reuse-time
                # one, silently re-opening the space leak m4 closed.
                _call_sql(
                    self._context.sql_runner,
                    stage="teardown",
                    process_name=process.name,
                    host="localhost",
                    port=port,
                    user=credentials.username,
                    password=credentials.password,
                    dbname="postgres",
                    statement=(
                        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                        f"WHERE datname = '{world_db}' AND pid <> pg_backend_pid()"
                    ),
                )
                _call_sql(
                    self._context.sql_runner,
                    stage="teardown",
                    process_name=process.name,
                    host="localhost",
                    port=port,
                    user=credentials.username,
                    password=credentials.password,
                    dbname="postgres",
                    statement=f'DROP DATABASE IF EXISTS "{world_db}"',
                )
            except ProcessRuntimeError as exc:
                logger.warning(
                    "teardown: failed to drop %s on %s: %s", world_db, process.name, exc
                )

    def _teardown_world(self, world_index: int) -> None:
        self._destroy_provider_target(world_index)
        handles = self._world_handles.pop(world_index, {})
        # N12, p6-review-r2 (MINOR): reversed — `handles`' insertion order is `spawn_world`'s own
        # `_topological_order` (dependencies first), so terminating it forward sends SIGTERM to a
        # per-world engine while its own dependents (`tools-api`/`agent`) may still hold open
        # connections, guaranteeing the full escalation wait every time. A dict's insertion order
        # is preserved, so `reversed()` here IS reverse-topological order, no recomputation needed.
        for name, handle in reversed(list(handles.items())):
            if name not in self._job_shared_handles:
                prefer_interrupt = self._manifest is not None and _prefers_interrupt(
                    self._manifest, name
                )
                _terminate_and_wait(handle.handle, prefer_interrupt=prefer_interrupt)
        self._drop_world_shared_databases(world_index)  # m4
        runtime = self._runtimes.pop(world_index, None)
        if runtime is not None:
            runtime.state = RuntimeState.STOPPED
        if self._context is not None:
            # m4, p6-review-r1: the per-world scratch tree (`/work/worlds/w<N>/`) covers every
            # process under this world, managed and source alike — left behind otherwise, the
            # other space-leak half of the same finding.
            world_dir = self._context.work_directory / "worlds" / f"w{world_index}"
            if world_dir.exists():
                shutil.rmtree(world_dir, ignore_errors=True)

    async def reset(self, runtime: EnvironmentRuntime, *, work_directory: Path) -> None:
        import asyncio

        await asyncio.to_thread(self._reset_sync, runtime)

    def _reset_sync(self, runtime: EnvironmentRuntime) -> None:
        if (
            self._manifest is None
            or self._context is None
            or self._build_output is None
        ):
            raise ProcessRuntimeError(
                "reset",
                _INTERNAL_INVARIANT_VIOLATED,
                "reset() called before provision()",
            )
        world_index = runtime.world_index
        self._ensure_job_shared_handles_alive(
            current_world_index=world_index
        )  # N11/Q7.
        try:
            handles, sentinel_ok = reset_world(
                self._manifest,
                world_index,
                context=self._context,
                baseline=self._build_output,
                job_shared_handles=self._job_shared_handles,
                existing_handles=self._world_handles.get(world_index, {}),
            )
        except (OSError, shutil.Error) as exc:
            # reset's own filesystem work is fundamentally "reseal this world's stores from
            # baseline" — the infrastructure-domain half of the store-statement mapping.
            partial = getattr(exc, "partial_handles", None)
            if partial is not None:
                self._world_handles[world_index] = partial
            raise ProcessRuntimeError(
                "reset",
                "store_statement_failed",
                str(exc),
                domain=FailureDomain.INFRASTRUCTURE,
            ) from exc
        except BaseException as exc:
            partial = getattr(exc, "partial_handles", None)  # N4, p6-review-r2.
            if partial is not None:
                self._world_handles[world_index] = partial
            raise
        self._world_handles[world_index] = handles
        # m5, p6-review-r1: mutates THIS PROVIDER's own current record for `world_index`, never
        # blindly overwrites it with whatever `EnvironmentRuntime` the caller happened to pass in
        # — a caller holding a STALE object (e.g. one `_ensure_world` already replaced with a
        # fresh `runtime_id`, before this `reset()` call was even made) used to be able to stomp
        # the live record with stale endpoints/id. N1, p6-review-r2: after `_ensure_world` mutates
        # in place instead of replacing, the provider's record and every caller's own reference
        # for this world index are the SAME object for the provider's whole life — the fallback
        # below (constructing from the caller's own `runtime`) is kept only for the unreachable
        # case where this provider somehow has no record for the index at all.
        current = self._runtimes.get(world_index)
        if current is None:
            current = runtime
            self._runtimes[world_index] = current
        if not sentinel_ok:
            current.state = RuntimeState.UNHEALTHY
            return
        # M4, p6-review-r1: §4 point 3 — "`healthy` = declared `readiness` probes, not 'process is
        # running.'" A passing sentinel alone used to be enough to mark `READY`; nothing waited for
        # the LAST process in the world's `depends_on` DAG (nobody depends on the control service,
        # so `spawn_world` never probes it) — the scheduler could dispatch a scenario against an
        # agent still mid-boot. N2, p6-review-r2: polls (`_poll_runtime_health`), the same helper
        # `provision()`'s own promotion uses, instead of sampling `probe_runtime_health` once.
        healthy_now = _poll_runtime_health(
            self._manifest, current, prober=self._context.prober
        )
        current.state = RuntimeState.READY if healthy_now else RuntimeState.UNHEALTHY

    async def close(self, *, work_directory: Path) -> None:
        import asyncio

        await asyncio.to_thread(self._close_sync, work_directory)

    def _teardown_processes_and_directories(self, work_directory: Path) -> None:
        """Shared by `close()` and M6's bundle-digest-change rebuild — terminates every live
        handle and removes every job/world-scoped directory WITHOUT touching secrets or this
        instance's own identity fields; `close()` clears those itself right after calling this,
        while a digest-change rebuild is about to overwrite them with the new bundle's own values
        a few lines later in `_provision_sync`.

        m10, p6-review-r1: every step is individually guarded — a failure removing ONE directory
        (a permission error, a file still open) used to abort the whole method before the LATER
        steps (clearing this instance's own dicts) ever ran, so the second `close()` §4.4 requires
        to be a no-op was no longer one. Logged, never raised, so cleanup always reaches the end.
        """
        for world_index in list(self._provider_receipts):
            self._destroy_provider_target(world_index)

        # N12, p6-review-r2 (MINOR): reversed per world (dependents before the engine they depend
        # on — see `_teardown_world`'s own note) and bounded by `self._close_wait_timeout_seconds`
        # so `close()` can keep itself inside §0.7's 120s flush window regardless of how many
        # per-world engines it has to wait out.
        for handles in self._world_handles.values():
            for name, handle in reversed(list(handles.items())):
                if name not in self._job_shared_handles:
                    prefer_interrupt = (
                        self._manifest is not None
                        and _prefers_interrupt(self._manifest, name)
                    )
                    _terminate_and_wait(  # M7: real wait, not just a sent signal.
                        handle.handle,
                        timeout=self._close_wait_timeout_seconds,
                        prefer_interrupt=prefer_interrupt,
                    )
        for name, handle in reversed(list(self._job_shared_handles.items())):
            prefer_interrupt = self._manifest is not None and _prefers_interrupt(
                self._manifest, name
            )
            _terminate_and_wait(
                handle.handle,
                timeout=self._close_wait_timeout_seconds,
                prefer_interrupt=prefer_interrupt,
            )
        for runtime in self._runtimes.values():
            runtime.state = RuntimeState.STOPPED

        for directory_name in ("build", "worlds", "managed", "provider"):
            directory = work_directory / directory_name
            try:
                if directory.exists():
                    shutil.rmtree(directory)
            except OSError:
                logger.warning("teardown: failed to remove %s; continuing", directory)

        self._job_shared_handles = {}
        self._world_handles = {}
        self._runtimes = {}
        self._conformance_checked = False
        self._provider_receipts = {}
        self._provider_contexts = {}

    def _close_sync(self, work_directory: Path) -> None:
        """§4 rule 4: idempotent hard-clean of everything — processes, data directories, build
        trees, `secrets.json` if still present (the load in `_provision_sync` already deleted it
        on the normal path; this is the "if still present" backstop §4.4 itself names, e.g. a
        `close()` called before `provision()` ever ran). Idempotent by construction: every dict is
        cleared at the end regardless of what failed along the way (m10).
        """
        self._teardown_processes_and_directories(work_directory)
        try:
            self._secrets_path.unlink(
                missing_ok=True
            )  # m9: no separate check-then-act .exists().
        except OSError:
            logger.warning(
                "close(): failed to unlink %s; continuing", self._secrets_path
            )

        self._manifest = None
        self._bundle_digest = None
        self._context = None
        self._build_output = None
        # Full reset, not just the spawn-state dicts `_teardown_processes_and_directories`
        # already cleared — a provider reused for a genuinely NEW job after `close()` must load
        # ITS OWN secrets file fresh, not silently run with the previous job's in-memory map (or
        # none at all, since `_secrets_loaded` would otherwise still read `True`).
        self._secret_values = {}
        self._secret_purposes = {}
        self._secrets_loaded = False

    def _ensure_job_shared_handles_alive(
        self, *, current_world_index: int | None = None
    ) -> None:
        """N11, p6-review-r2 (MAJOR): a job-shared `template_database` engine (§2b) backs EVERY
        world at once, and nothing anywhere ever checked whether it was still running before
        reusing it (`spawn_world`'s own `if name in handles: continue`) — an OOM-killed shared
        postgres used to be carried forward as if healthy for the rest of the job, so every
        subsequent `reset()`/reconcile raised `store_statement_failed` out of `_reset_template_
        database`'s own connection attempt against a dead process, `hosted_scheduler.py` swallowed
        it, and no world ever recovered even though the engine's data directory was sitting intact
        on disk the whole time. Called before `_clone_or_reset_world`/`reset_world` ever touch
        `self._job_shared_handles`. `current_world_index` (Q7, p6-review-r3) is the world the
        caller is already about to reconcile — passed through so a respawn can demote every OTHER
        tracked world instead of leaving it `READY` against a database it just reset underneath.
        """
        for name, handle in list(self._job_shared_handles.items()):
            if not handle.handle.is_running():
                self._job_shared_handles[name] = self._respawn_dead_job_shared_engine(
                    name,
                    current_world_index=current_world_index,
                )

    def _respawn_dead_job_shared_engine(
        self,
        process_name: str,
        *,
        current_world_index: int | None = None,
    ) -> SpawnedWorldProcess:
        """Respawns from the SAME surviving data directory (never wiped — this is not a baseline
        restore; the engine's own on-disk state is exactly what every world's logical database
        still depends on), waits for readiness, then re-seals every world THIS PROVIDER currently
        tracks from the template — the shared engine restarting is otherwise indistinguishable
        from a fresh boot to postgres, but each world's own `w<N>` database is a SEPARATE logical
        database on that one instance and must be re-verified against the template again, same as
        `_reset_template_database` already does for one world at a time.
        """
        if (
            self._manifest is None
            or self._context is None
            or self._build_output is None
        ):
            raise ProcessRuntimeError(
                "reset",
                _INTERNAL_INVARIANT_VIOLATED,
                "_respawn_dead_job_shared_engine called before manifest/context/build_output "
                "were established",
            )
        processes_by_name = {
            process.name: process for process in self._manifest.processes
        }
        process = processes_by_name.get(process_name)
        if not isinstance(process, ManagedProcess):
            raise ProcessRuntimeError(
                "reset",
                _INTERNAL_INVARIANT_VIOLATED,
                f"{process_name}: job-shared handle names a process that is not a managed engine",
                process=process_name,
            )
        context = self._context
        build_output = self._build_output
        port = context.port_plan.port_for(process_name, 0)
        data_dir = managed_engine_data_dir(
            context.work_directory, process_name, world_index=None
        )
        credentials = context.credentials.get(process_name)
        try:
            new_handle = spawn_managed_process(
                process,
                port=port,
                data_dir=data_dir,
                credentials=credentials,
                runner=context.runner,
                sync_run=context.sync_run,
                user_resolver=context.user_resolver,
                require_declared_user=context.require_declared_user,
                chown=context.chown,
            )
            _wait_for_store_ready(self._manifest, process, port=port, context=context)
            if process.engine is ManagedEngine.POSTGRES:
                record = next(
                    (r for r in build_output.stores if r.process_name == process_name),
                    None,
                )
                if record is not None:
                    live_credentials = _require_credentials(
                        credentials, stage="reset", process_name=process_name
                    )
                    for world_index in self._runtimes:
                        _reset_template_database(
                            host="localhost",
                            port=port,
                            credentials=live_credentials,
                            world_db=f"w{world_index}",
                            template_db=record.baseline_reference,
                            sql_runner=context.sql_runner,
                            process_name=process_name,
                        )
                    # Q7, p6-review-r3: the loop above just reset EVERY tracked world's database —
                    # demote every world other than the one the caller is already reconciling so
                    # the scheduler's next `reset()` repairs it, instead of handing out a `READY`
                    # world whose pooled connections point at a database dropped underneath it.
                    for idx, rt in self._runtimes.items():
                        if idx != current_world_index:
                            rt.state = RuntimeState.UNHEALTHY
        except (ProcessRuntimeError, OSError, shutil.Error) as exc:
            raise ProcessRuntimeError(
                "reset",
                "store_statement_failed",
                f"{process_name}: job-shared engine died and could not be respawned: {exc}",
                process=process_name,
                domain=FailureDomain.INFRASTRUCTURE,
            ) from exc
        return new_handle

    async def healthy(
        self, runtime: EnvironmentRuntime, *, work_directory: Path
    ) -> bool:
        """v1.12 §4.3's `RuntimeProvider.healthy` port, closing the gap p6-review-r2's task 2c
        flagged: this class had `provision`/`reset`/`close` but no `healthy`, so it could not
        structurally satisfy `runtime.py`'s `RuntimeProvider` Protocol. Demote-only, unlike the
        module-level `healthy()` above — §3's transition table makes `unhealthy->ready` reachable
        ONLY through a re-provision reconcile, and `provision()` already owns the `preparing->
        ready` promotion (`_poll_runtime_health`, before ever returning a world), so this port
        method must never promote from ANY state, including `preparing`. `work_directory` is
        accepted for §4 shape-compatibility and unused — the provider already holds its own
        paths, same as `provision`'s own `contract` parameter.

        After N1 (p6-review-r2), `self._runtimes[world_index]` IS the same object every caller
        holds for that world, so a single assignment on it is enough; `.get(..., runtime)` still
        falls back to the caller's own object if this provider somehow has no record for the
        index (unreachable in the normal `provision` -> `healthy` flow).
        """
        import asyncio

        if self._manifest is None:
            raise ProcessRuntimeError(
                "healthy",
                _INTERNAL_INVARIANT_VIOLATED,
                "healthy() called before provision()",
            )
        target = self._runtimes.get(runtime.world_index, runtime)
        is_healthy = await asyncio.to_thread(
            probe_runtime_health,
            self._manifest,
            target,
            prober=self._prober,
        )
        if not is_healthy:
            target.state = RuntimeState.UNHEALTHY
        return is_healthy


__all__ = [
    "BuildOutput",
    "CapabilityProber",
    "EngineCredentials",
    "EnvironmentRuntime",
    "FreezeResult",
    "PopenProcess",
    "PortPlan",
    "ProcessRunner",
    "ProcessRuntimeError",
    "ProcessRuntimeProvider",
    "RabbitmqDefinitionsImporter",
    "RabbitmqQueueDeclarer",
    "RabbitmqQueueDeleter",
    "RabbitmqQueueInspector",
    "RedisCommandRunner",
    "RuntimeEndpoint",
    "RuntimeState",
    "SpawnContext",
    "SpawnedProcess",
    "SpawnedWorldProcess",
    "SqlRunner",
    "StoreBaselineRecord",
    "WorldSpawnResult",
    "apply_seed_file",
    "apply_store_seed",
    "build_endpoints",
    "build_process_tree",
    "build_process_trees",
    "build_tree_dir",
    "check_sentinel",
    "configuration_addresses_from_endpoints",
    "default_capability_prober",
    "default_process_runner",
    "default_rabbitmq_definitions_importer",
    "default_rabbitmq_queue_declare_and_publish",
    "default_rabbitmq_queue_delete",
    "default_rabbitmq_queue_inspector",
    "default_redis_command_runner",
    "default_sql_runner",
    "default_user_resolver",
    "freeze_baseline",
    "generate_engine_credentials",
    "healthy",
    "managed_engine_data_dir",
    "new_runtime_id",
    "plan_ports",
    "postgres_bootstrap_argv",
    "postgres_daemon_argv",
    "postgres_seed_argv",
    "postgres_seed_env",
    "probe_runtime_health",
    "rabbitmq_conf_text",
    "rabbitmq_daemon_argv",
    "rabbitmq_daemon_env",
    "rabbitmq_enabled_plugins_text",
    "redis_daemon_argv",
    "redis_seed_argv",
    "render_capability_address",
    "render_environment",
    "render_template",
    "reset_world",
    "run_conformance_gate",
    "select_process_secrets",
    "spawn_managed_process",
    "spawn_source_process",
    "spawn_world",
    "wait_for_dependency",
    "world_scratch_dir",
    "write_build_output",
]
