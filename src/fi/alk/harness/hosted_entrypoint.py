"""The hosted guest's `main()` — `hosted-execution-seams.md` v1.14 §0/§4/§5, `outbound-channels.md`
v1.3, `world-handle-interface.md` v3.4. Everything between "sandbox starts" and "exit code": read
`/work/job.json`, load the platform capability file, run §2e preflight, pre-allocate scenarios
against `endpoints.scenarios`, provision the world pool, drive the scenario loop, adapt its events/
receipts/artifacts onto the real outbound clients, and honor the exit-code contract (§0.6).

Ownership boundary (read this before touching orchestration order): the stages BEFORE bundle
authoring — `understanding_agent`, `generating_environment`, `building_environment`'s bundle-write
half — belong to Rishav's stages (contract §6) and are not implemented anywhere in this repo yet.
This module does not attempt them. `BundleSource`/`ScenarioSource` below are the seams a later
change wires the real stages through; until then their defaults raise a typed, clearly-named error
rather than silently producing a fake bundle or a fake scenario set.

`process_runtime.py`, `hosted_scheduler.py`, and `outbound.py` were being fixed by parallel workers
while this module was written. It codes against the four frozen contracts and the cross-review
obligation lists, not against those files' exact HEAD.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import signal
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from . import outbound as ob
from .bundle_v2 import BundleV2Error, EnvironmentBundleV2, load_bundle_v2
from .call_runner import CallRunnerContext, CallRunnerImpl
from .hosted_scheduler import (
    CallOutcome,
    CallRunner,
    HostedScheduler,
    ResultReceipt,
    RunResult,
    Scenario,
    World,
    WorldFactory,
    WorldPool,
    WorldProvisioner,
)
from .job import (
    ArtifactLevel,
    ExecutionMode,
    FailureDomain,
    HarnessArtifactPolicy,
    HarnessJob,
    HarnessStage,
)
from .process_preflight import PreflightError, preflight_bundle
from .process_runtime import (
    SECTION_2F_DOMAIN,
    EnvironmentRuntime,
    ProcessRuntimeError,
    ProcessRuntimeProvider,
    RuntimeEndpoint,
)
from .scenario_source import (
    BundleScenarioSource,
    ScenarioDocumentInvalid,
    bundle_has_scenarios,
)
from .world.handle import HostedWorld
from .world.stores.postgres import AttachedPostgresStore

logger = logging.getLogger(__name__)


class _JobIdFilter(logging.Filter):
    """Stamp every log record with the job this runner is serving.

    One runner process serves exactly one job, so the id is process-wide rather than per-task
    state. Concurrent runs are separate processes, but their stdout is collected into one place,
    and a line with no job id cannot be attributed to a run at all -- which is the difference
    between reading a log and guessing at it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.job_id = "-"

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "job_id"):
            record.job_id = self.job_id
        return True


_JOB_ID_FILTER = _JobIdFilter()


def configure_runner_logging(job_id: str | None) -> None:
    """Put the job id on every line this process emits, including libraries' lines.

    Installed on the root logger rather than ours, because the lines that are hardest to attribute
    are the ones from livekit, httpx and the model clients.
    """
    _JOB_ID_FILTER.job_id = str(job_id or "-")
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(_JOB_ID_FILTER)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.addFilter(_JOB_ID_FILTER)
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    for handler in logging.getLogger().handlers:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s job=%(job_id)s %(name)s: %(message)s"
            )
        )

# --- §0.6 exit-code contract --------------------------------------------------------------------
#
# 0 = any terminal stage reached (completed/failed/canceled), outbox flushed. 3 = fenced/superseded
# (HostedFencedError anywhere -> stop emitting, no terminal event, exit 3). 4 = the terminal was
# decided but the final drain could not deliver it (the events channel failed, or the platform
# permanently rejected the terminal item itself) -- the gateway treats it exactly like a crash
# (infrastructure retry, fresh channels), but the distinct code tells operators the job DID reach a
# terminal state, unlike a genuine crash. Any other non-zero = the guest crashed before a terminal
# state -- the gateway records `infrastructure`. Capabilities-file failures are explicitly carved
# out of the "any other non-zero" bucket only by CODE (they must never be 3, per
# outbound-channels.md v1.3's rejection table); they still use a non-zero exit here since there is
# no channel to report a terminal FAILED event through.
EXIT_OK = 0
EXIT_FENCED = 3
EXIT_TERMINAL_UNDELIVERED = (
    4  # terminal reached but not provably flushed on the final drain.
)
EXIT_BOOT_FAILURE = (
    1  # capabilities.json could not be loaded -- no channel, no event (v1.3 table).
)
EXIT_CRASHED = 2  # an uncaught failure before any terminal stage was reached.

# Cancellation signal (spine §0 step 7 / outbound-channels.md "Cancellation signal"). The task
# brief that spawned this module named `/work/cancel.json`; the two frozen contracts that actually
# define this file (seams §0 step 7, outbound-channels "Cancellation signal") both name
# `/run/futureagi/cancel.json`. Contracts are authoritative over a task brief.
CANCEL_SIGNAL_PATH = "/run/futureagi/cancel.json"

# STUCK DECISION (fail-safe/reversible; contract gap): the invocation contract
# (spine §0 step 5) pins the entrypoint's argv to exactly `job --source ... --output ...`; `--output`
# is `/work/artifacts` (spine layout block), so `work_directory` (what `preflight_bundle`/
# `provision`/`write_build_output` all want -- the `/work` root) is derived as `output.parent`
# rather than taken as a separate flag, since the frozen invocation line has no room for one.
# `bundle_dir` has no convention anywhere in the frozen documents at all (bundle authoring is not
# built yet); `DEFAULT_BUNDLE_DIR_NAME` is this module's own placeholder location, overridable via
# `BundleSource` injection so a later change can point it at wherever the real authoring stage ends
# up writing without touching this file's orchestration.
DEFAULT_BUNDLE_DIR_NAME = "bundle"
EVENTS_SPOOL_DIR_NAME = (
    "outbound-spool"  # must not live under work_directory/"artifacts".
)

SECRETS_PATH = Path("/run/futureagi/secrets.json")
SIMULATOR_SECRETS_PATH = Path("/run/futureagi/simulator-secrets.json")

# Platform-owned simulator configuration is delivered on a separate control-plane channel.  It
# must never be confused with the customer's ``target_provider`` refs, which are selectively
# injected into the untrusted agent processes by ProcessRuntimeProvider.  These names are the
# complete set the in-process text/voice simulators may consume.
_SIMULATOR_SECRET_ALIASES = frozenset(
    {
        "ALK_BACKGROUND_NOISE",
        "ALK_BACKGROUND_NOISE_CATALOG",
        "ALK_HARNESS",
        "ALK_VOICEMAIL_SCENARIOS",
        "ALK_HARNESS_MODEL",
        "ALK_HARNESS_THINKING",
        "ALK_VERTEX_LOCATION",
        "CARTESIA_API_KEY",
        "DEEPGRAM_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "HARNESS_BACKGROUND_NOISE_VOLUME",
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "OPENAI_API_KEY",
        "SIMULATOR_LLM_MODEL",
        "SIMULATOR_LLM_PROVIDER",
        "SIMULATOR_STT_MODEL",
        "SIMULATOR_STT_PROVIDER",
        "SIMULATOR_TTS_MODEL",
        "SIMULATOR_TTS_PROVIDER",
    }
)


# =================================================================================================
# Boot -- job.json + capabilities.json (§0.2/§0.4; outbound-channels.md Authentication).
# =================================================================================================


def load_job(job_path: Path) -> HarnessJob:
    """§0.2: `/work/job.json` is the provisioner's job-identity and configuration source."""
    job = HarnessJob.model_validate_json(job_path.read_text(encoding="utf-8"))
    if job.execution is not ExecutionMode.HOSTED:
        raise ValueError("hosted_entrypoint_requires_hosted_job")
    return job


def resolve_parallelism(job: HarnessJob) -> int:
    """`job.runtime.parallelism` = W (glossary). Returns the RAW requested value, never
    clamped -- §2e.7 reserves `parallelism_out_of_range` for a W outside 1..8, and
    `preflight_bundle` (called BEFORE any provisioning) is the enforcement point for the UPPER
    bound. The lower bound never reaches preflight at all: `RuntimeRequirements.parallelism`'s own
    `ge=1` rejects a non-positive W earlier, at `load_job`, as a deliberate defense-in-depth floor
    (harmless today since the gateway caps W at admission before a job is ever built). Clamping
    here would silently launder an in-range-but-wrong W and make `parallelism_out_of_range`
    permanently unreachable for the upper bound."""
    return job.runtime.parallelism


def job_secret_purposes(job: HarnessJob) -> dict[str, str]:
    """§1: `agent.secret_refs` alias -> `SecretRef.purpose`, the shape `preflight_bundle` wants."""
    return {alias: ref.purpose for alias, ref in job.agent.secret_refs.items()}


def peek_secret_values(secrets_path: Path) -> tuple[str, ...]:
    """A non-destructive read of `/run/futureagi/secrets.json`'s VALUES ONLY, for outbound
    redaction (`extra_secret_values` — outbound.py's `redact_outbound_text`). §0.3's lifetime rule
    ("the provisioner loads this file into memory at startup and deletes it") is honored by
    `ProcessRuntimeProvider` itself; this is an additional, side-effect-free read (no unlink) done
    once at boot so free-text event/log/failure fields can be scrubbed of every resolved secret
    value, not just URL userinfo. Never fatal: a missing/malformed file just means no extra values
    to scrub, matching `redact_outbound_text`'s own `extra_secret_values=()` default."""
    try:
        raw = json.loads(secrets_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    if not isinstance(raw, dict):
        return ()
    return tuple(str(value) for value in raw.values() if value)


def peek_secret_values_for_purpose(
    secrets_path: Path,
    secret_purposes: dict[str, str],
    purpose: str,
) -> dict[str, str]:
    """The same non-destructive, no-unlink read as `peek_secret_values` (same file, same timing
    constraint -- called BEFORE `pool.start()`, which is what actually deletes the file), but
    ALIAS-preserving and filtered to one explicit purpose -- `peek_secret_values` throws the
    alias away, which is fine for outbound redaction (it only needs the raw values) but useless for
    the real `CallRunner`, which needs to pick e.g. `LIVEKIT_API_KEY` out of the map by name. Never
    fatal: a missing/malformed file just means no target-provider secrets are available yet,
    matching `CallRunnerImpl`'s own pre-dial validation (it reports the gap as a typed
    `CallAborted`, never crashes on an empty map)."""
    try:
        raw = json.loads(secrets_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        str(alias): str(value)
        for alias, value in raw.items()
        if secret_purposes.get(str(alias)) == purpose
    }


def peek_target_provider_secret_values(
    secrets_path: Path, secret_purposes: dict[str, str]
) -> dict[str, str]:
    return peek_secret_values_for_purpose(
        secrets_path, secret_purposes, "target_provider"
    )


def peek_simulator_provider_secret_values(
    secrets_path: Path, secret_purposes: dict[str, str]
) -> dict[str, str]:
    return peek_secret_values_for_purpose(
        secrets_path, secret_purposes, "simulator_provider"
    )


def load_simulator_secret_values(path: Path) -> dict[str, str]:
    """Load and immediately remove the platform-owned simulator secret channel.

    The fixed allowlist is deliberate: a platform deployment cannot accidentally use this file
    to inject arbitrary ambient variables into the control process.  Agent subprocesses still do
    not inherit these values because ``process_runtime`` starts them from its closed environment
    allowlist plus purpose-matched target secrets.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("simulator-secrets.json unlink failed: %s", exc)
    if not isinstance(raw, dict):
        return {}
    return {
        str(alias): str(value)
        for alias, value in raw.items()
        if str(alias) in _SIMULATOR_SECRET_ALIASES and value not in (None, "")
    }


# =================================================================================================
# Bundle source -- §2 bundle authoring is not this module's (or built anywhere yet); injectable.
# =================================================================================================


class BundleUnavailableError(RuntimeError):
    """Raised by a `BundleSource` when no bundle could be produced/located. Mapped the same way as
    a `PreflightError` (FAILED, `FailureDomain.ENVIRONMENT`, stage `validating_environment`) —
    from the entrypoint's point of view "no bundle" and "bad bundle" are the same class of
    environment-authoring fault, and §2e's own failure table has no separate code for it."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class BundleSource(Protocol):
    def load(
        self, job: HarnessJob, *, source: Path, work_directory: Path
    ) -> tuple[EnvironmentBundleV2, Path]: ...


# §2e's closed failure-code table (hosted-execution-seams.md) -- `BundleV2Error` has no typed
# `.code` (a bare `RuntimeError`), so `DefaultBundleSource.load` below string-splits
# its message on ":". `bundle_manifest_missing` (one of the four messages `load_bundle_v2` can
# raise) is not in this table -- a real contract gap -- so both that code AND anything else the
# split produces outside this frozen set fall back to `bundle_manifest_invalid` rather than
# shipping an unlisted code across the outbound seam.
_SECTION_2E_CODES = frozenset(
    {
        "compose_not_hosted",
        "engine_unsupported",
        "no_sql_store",
        "seed_missing",
        "seed_strategy_unsupported",
        "sentinel_shape_mismatch",
        "store_protocol_unsupported",
        "capability_engine_mismatch",
        "store_service_not_managed",
        "reserved_name",
        "unknown_placeholder",
        "unknown_field",
        "secret_in_bundle",
        "secret_unclaimed",
        "secret_missing",
        "secret_purpose_forbidden",
        "build_requires_root",
        "user_assignment_invalid",
        "configuration_name_duplicate",
        "configuration_name_required",
        "configuration_name_reserved",
        "sentinel_shape_invalid",
        "capability_unresolved",
        "service_unresolved",
        "control_service_unresolved",
        "process_name_duplicate",
        "inputs_digest_mismatch",
        "bundle_schema_unsupported",
        "bundle_manifest_invalid",
        "bundle_manifest_drifted",
        "bundle_digest_mismatch",
        "bundle_digest_invalid",
        "inputs_digest_invalid",
        "file_sha256_invalid",
        "source_digest_invalid",
        "bundle_file_missing",
        "bundle_file_changed",
        "bundle_file_unlisted",
        "bundle_symlink_forbidden",
        "bundle_path_unsafe",
        "depends_on_unresolved",
        "depends_on_cycle",
        "seed_file_missing",
        "seed_file_unlisted",
        "process_count_exceeded",
        "parallelism_out_of_range",
        "evidence_seam_required",
        "processes_required",
        "processes_and_seed_forbidden",
        "document_only_for_compose",
        "compose_runtime_requires_document",
        "build_command_step_empty",
        "started_check_requires_exactly_one_of_port_or_log_marker",
        "resolved_secret_forbidden",
        "capability_slug_invalid",
        "process_name_invalid",
        "fixed_port_reserved",
        "certification_missing",
        "certification_invalid",
        "certification_mismatch",
        "certification_incomplete",
    }
)


def _bundle_unavailable_code(raw_message: str) -> str:
    code = raw_message.split(":", 1)[0].strip()
    return code if code in _SECTION_2E_CODES else "bundle_manifest_invalid"


class DefaultBundleSource:
    """Looks for an already-authored bundle at `work_directory / bundle_dir_name`. This is a
    placeholder location this module invented (see the module-level STUCK DECISION note) — a real
    bundle-authoring stage should either write there or be wired in via its own `BundleSource`."""

    def __init__(self, bundle_dir_name: str = DEFAULT_BUNDLE_DIR_NAME) -> None:
        self._bundle_dir_name = bundle_dir_name

    def load(
        self, job: HarnessJob, *, source: Path, work_directory: Path
    ) -> tuple[EnvironmentBundleV2, Path]:
        del job, source  # unused by the default (a real stage would author from these)
        bundle_dir = work_directory / self._bundle_dir_name
        try:
            manifest = load_bundle_v2(bundle_dir)
        except BundleV2Error as exc:
            raise BundleUnavailableError(
                _bundle_unavailable_code(exc.args[0]), str(exc)
            ) from exc
        return manifest, bundle_dir


# =================================================================================================
# Scenario source -- generation is a separate contract (in review, not available here); the
# pre-allocation CALL is this module's (ScenariosClient below). Injectable for the same reason as
# BundleSource: the glue between "generated scenarios" and "pre-allocated against the platform" can
# only be finished once that contract's payload shape lands.
# =================================================================================================


class ScenarioSourceNotWired(RuntimeError):
    """The default `ScenarioSource` — no Scenario Generation Contract implementation exists in this
    repo yet. Raised rather than fabricating scenarios, and mapped to FAILED / `platform_sync` /
    `validating_scenarios`, matching spine §5 step 3.5's own failure mapping for a pre-allocation
    that never completes."""


class ScenarioSource(Protocol):
    async def build(
        self,
        job: HarnessJob,
        bundle: EnvironmentBundleV2,
        scenarios_client: "ScenariosClient",
        *,
        pool: WorldPool,
        world_factory: WorldFactory,
        bundle_dir: Path,
    ) -> Sequence[Scenario]: ...


class NotWiredScenarioSource:
    async def build(
        self,
        job: HarnessJob,
        bundle: EnvironmentBundleV2,
        scenarios_client: "ScenariosClient",
        *,
        pool: WorldPool,
        world_factory: WorldFactory,
        bundle_dir: Path,
    ) -> Sequence[Scenario]:
        del job, bundle, scenarios_client, pool, world_factory, bundle_dir
        raise ScenarioSourceNotWired(
            "no ScenarioSource wired -- scenario generation is not implemented in this repo yet "
            "(Scenario Generation Contract, in review)"
        )


# =================================================================================================
# WorldFactory -- real HostedWorld instances, fed by build.json's row counts (never
# a partial map).
# =================================================================================================


class WorldFactoryError(RuntimeError):
    """The provisioner handed back a runtime this factory cannot build a `World` for — a bug
    upstream (no postgres endpoint despite §2e's `no_sql_store` guarantee, or `build.json` missing
    the row counts for that store), never a scenario-code fault."""


def _process_runtime_error_domain(exc: ProcessRuntimeError) -> FailureDomain:
    """v1.15 §2f: the producer (`process_runtime.py`) resolves and carries `domain` at the raise
    site -- read it directly rather than re-deriving `spawn_failed`'s managed/source split from
    the manifest (the old approach could not tell which process kind failed without one). The
    imported `SECTION_2F_DOMAIN` map is a fallback ONLY, for an error that reaches here with no
    carried domain -- logged when it fires, matching the scheduler's own rule.
    """
    if exc.domain is not None:
        return exc.domain
    if exc.code in SECTION_2F_DOMAIN:
        logger.warning(
            "process_runtime error %r crossed the §4 seam with no carried domain; using the §2f "
            "fallback map (%s)",
            exc.code,
            SECTION_2F_DOMAIN[exc.code].value,
        )
        return SECTION_2F_DOMAIN[exc.code]
    return FailureDomain.INFRASTRUCTURE  # internal_* etc. -- the honest default


_SECTION_2F_CODES: frozenset[str] = frozenset(SECTION_2F_DOMAIN)


def _section_2f_code(code: str) -> str:
    # §2f is closed (contract §4.6) -- `process_runtime.py`'s own `internal_*` codes, and this
    # module's untyped-exception fallback, must never cross the outbound seam unlabeled, matching
    # the discipline `_bundle_unavailable_code` already applies to §2e. The real code is
    # still visible on the wire -- it stays in `message` (`ProcessRuntimeError.__str__` embeds it,
    # and the untyped-exception call site prefixes it explicitly).
    return code if code in _SECTION_2F_CODES else "spawn_failed"


def _find_postgres_endpoint(runtime: EnvironmentRuntime) -> RuntimeEndpoint:
    for endpoint in runtime.endpoints.values():
        if endpoint.protocol == "postgres":
            return endpoint
    raise WorldFactoryError(
        f"world {runtime.world_index}: no postgres-protocol endpoint in {sorted(runtime.endpoints)} "
        "-- §2e's no_sql_store rule should make this unreachable"
    )


def load_build_output(work_directory: Path) -> dict[str, Any]:
    """`write_build_output` (process_runtime.py) writes `<work_directory>/artifacts/build.json`.
    Read fresh each call — cheap, and the row counts are immutable after baseline freeze, so
    re-reading is simpler than a cache invalidation story for the same modest cost."""
    path = work_directory / "artifacts" / "build.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WorldFactoryError(f"build.json unreadable at {path}: {exc}") from exc


def row_counts_for_capability(
    build_output: dict[str, Any], capability: str
) -> dict[str, int]:
    for store in build_output.get("stores", []):
        if store.get("capability") == capability:
            counts = store.get("row_counts") or {}
            return {str(name): int(count) for name, count in counts.items()}
    raise WorldFactoryError(
        f"build.json has no store entry for capability {capability!r} — the provisioner "
        "guarantees a complete row-count map per store, so this bundle's build output is malformed"
    )


class ProcessWorldFactory:
    """Builds a real `HostedWorld` over the runtime's postgres endpoint. `AttachedPostgresStore`
    (not the bare `PostgresStore`) is the correct base here — it takes a raw DSN and never manages
    a container's own lifecycle, matching a hosted world where `ProcessRuntimeProvider` already
    owns the postgres process."""

    def __init__(self, work_directory: Path) -> None:
        self._work_directory = work_directory

    async def create(self, runtime: EnvironmentRuntime, *, rng: random.Random) -> World:
        endpoint = _find_postgres_endpoint(runtime)
        build_output = await asyncio.to_thread(load_build_output, self._work_directory)
        row_counts = row_counts_for_capability(build_output, endpoint.capability)
        store = AttachedPostgresStore(endpoint.address)
        return await asyncio.to_thread(
            HostedWorld, store, runtime.world_index, rng, row_counts
        )


# =================================================================================================
# CallRunner -- the real voice track. Explicit LiveKit jobs and auto-discovered voice contracts
# use it.  Explicit Vapi/Retell jobs remain outside the repository-hosted runner.
# =================================================================================================


class CallRunnerNotWired(RuntimeError):
    """Raised by `NotWiredCallRunner`. `hosted_scheduler._execute` treats any exception out of
    `CallRunner.run` (other than `WorldUnavailable`/`CallAborted`) as `call_failed`
    (`FailureDomain.INFRASTRUCTURE`, retried once) — so a job run with nothing wired here degrades
    every scenario to one retry-then-errored receipt rather than crashing the process."""


class NotWiredCallRunner:
    async def run(
        self,
        scenario: Scenario,
        runtime: EnvironmentRuntime,
        *,
        world: World | None = None,
    ) -> CallOutcome:
        del scenario, runtime, world
        raise CallRunnerNotWired(
            "no CallRunner wired -- the live voice-simulation call runner is a separate track"
        )


_VOICE_CONNECTORS = {"livekit", "vapi", "retell"}


def _bundle_contract_value(bundle_dir: Path, key: str) -> str | None:
    path = bundle_dir / "contract.json"
    if not path.is_file():
        return None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(body, dict):
        return None
    value = str(body.get(key) or "").strip().lower()
    return value or None


def _bundle_contract_modality(bundle_dir: Path) -> str | None:
    return _bundle_contract_value(bundle_dir, "modality")


def _default_build_call_runner(
    adapter: "OutboundAdapter", context: CallRunnerContext
) -> CallRunner:
    """The real factory: `NotWiredCallRunner` stays exactly as documented for every connector
    outside the LiveKit-dispatched voice path; a `"livekit"` job gets a real `CallRunnerImpl`,
    whose OWN pre-dial validation (`call_runner._check_config`) is what surfaces an
    incomplete-but-present config as a typed `call_failed`/infrastructure retry --
    `capability_unavailable` stays unreachable from this seam (would require a scheduler edit;
    the contract itself calls it "a follow-up, not shipped with this text")."""
    connector = context.job.agent.connector.lower()
    modality = _bundle_contract_modality(context.bundle_dir)
    if connector == "retell_chat":
        from .retell_chat_call_runner import RetellChatCallRunner

        return RetellChatCallRunner(adapter, context)
    if connector in _VOICE_CONNECTORS or (connector == "auto" and modality == "voice"):
        # The understand stage read this off the agent's own instructions, so the contract is the
        # only source. Carried through the process environment because `CallRunnerImpl` is handed a
        # context and a scenario document, neither of which reaches the contract; this is an
        # internal hop, not a knob, and nothing outside sets it.
        declared = _bundle_contract_value(context.bundle_dir, "call_direction")
        if declared:
            os.environ["ALK_CALL_DIRECTION"] = declared
        return CallRunnerImpl(adapter, context)
    # Repository-hosted text targets advertise their concrete HTTP interface in the frozen
    # contract adopted into Bundle V2. Connector-only Vapi/Retell remains on the existing
    # NotWired path and is deliberately not inferred as repository chat.
    if (context.bundle_dir / "contract.json").is_file():
        from .chat_call_runner import HostedChatCallRunner

        return HostedChatCallRunner(adapter, context)
    return NotWiredCallRunner()


# =================================================================================================
# Scenario pre-allocation -- a thin client against endpoints.scenarios (outbound-channels.md v1.3
# Authentication: bearer + X-Harness-Fence, `{"result": {...}}` envelope, job-scoped idempotent).
# Previously unowned; owned by this module now.
# =================================================================================================


class ScenarioPreallocationError(RuntimeError):
    def __init__(self, error: ob.ChannelError | None) -> None:
        self.error = error
        super().__init__(
            "scenario pre-allocation failed" if error is None else error.message
        )


class ScenariosClient:
    """RESOLVED (p13-worker-r2, reports/p13-worker-r2.md CONTRACT NOTES): the Scenario
    Generation Contract (PR #63) documented two paths (`run-tests/provision/` +
    `run-tests/{id}/test-executions/`) and a position-ordered `scenario_ids` response, but the
    platform's actual, live route (futureagi/simulate/views/hosted_harness.py:78-90,
    urls.py:128-132) mints exactly ONE url per attempt -- a DRF detail `@action` with no
    `url_path`, so the router only ever produces `.../scenarios/`, never a `provision/`/`begin/`
    sub-resource. The real dispatch key is a body-level `operation: "provision"|"begin"` field
    (serializers/hosted_harness.py:201-226's `HarnessScenarioOperationSerializer`). This class's
    transport (`_post`) is unchanged -- `provision_path`/`begin_path` are the SAME
    constructor-injectable placeholders as before, now correctly defaulted to an EMPTY suffix (the
    real route needs none) rather than a guessed path segment; `register_with_platform`
    (scenario_source.py) is what adds the `operation` field into each payload before calling
    `.provision()`/`.begin()`, matching this class's existing "operation field in payload" seam
    rather than requiring a change to either method's body. Shares `channel_state` with the other
    three channels (a fence on any one must stop all of them, per outbound.py's own `ChannelState`
    docstring)."""

    def __init__(
        self,
        capabilities: ob.HostedCapabilities,
        transport: ob.Transport | None = None,
        *,
        retry_policy: ob.RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
        channel_state: ob.ChannelState | None = None,
        provision_path: str = "",
        begin_path: str = "",
    ) -> None:
        self._capabilities = capabilities
        self._transport = transport or ob.RequestsTransport()
        self._retry_policy = retry_policy or ob.RetryPolicy()
        self._sleep = sleep
        self._rng = rng
        self._channel_state = channel_state or ob.ChannelState()
        self._provision_path = provision_path
        self._begin_path = begin_path

    def provision(
        self, payload: dict[str, Any], *, deadline: float | None = None
    ) -> dict[str, Any]:
        return self._post(self._provision_path, payload, deadline=deadline)

    def begin(
        self, payload: dict[str, Any], *, deadline: float | None = None
    ) -> dict[str, Any]:
        return self._post(self._begin_path, payload, deadline=deadline)

    def _post(
        self, path_suffix: str, payload: dict[str, Any], *, deadline: float | None
    ) -> dict[str, Any]:
        self._channel_state.check()
        url = f"{self._capabilities.endpoints.scenarios}{path_suffix}"

        def perform(_attempt: int) -> ob.TransportResponse:
            return self._transport.request(
                "POST",
                url,
                headers=self._capabilities.auth_headers(),
                json_body=payload,
            )

        try:
            response, error = ob._perform_with_retry(
                perform,
                retry_policy=self._retry_policy,
                sleep=self._sleep,
                rng=self._rng,
                deadline=deadline,
            )
        except (ob.HostedFencedError, ob.HostedChannelFailedError) as exc:
            self._channel_state.latch(exc)
            raise
        if error is not None or response is None:
            raise ScenarioPreallocationError(error)
        body = response.body if isinstance(response.body, dict) else {}
        result = body.get("result")
        if not isinstance(result, dict):
            raise ScenarioPreallocationError(
                ob.ChannelError(
                    ob.ChannelOutcome.PERMANENT_ITEM,
                    FailureDomain.PLATFORM_SYNC,
                    "scenarios_envelope_invalid",
                    "response body has no {'result': {...}} envelope",
                )
            )
        return result


# =================================================================================================
# OutboundPort adapter -- the real emit pipeline: redact -> capabilities.event_builder() ->
# spool.append -> EventsClient.flush(). Also: baseline_frozen/parallelism_degraded from build.json,
# terminal events (exactly one, last), artifact-before-receipt ordering,
# and RunResult.aborted -> TerminalFailure(infrastructure, running, "world_pool_exhausted").
# =================================================================================================


_TERMINAL_FAILURE_MESSAGE_MAX_CHARS = 4096  # an unbounded `failure.message` can blow
# EVENT_PAYLOAD_MAX_BYTES and hard-reject the WHOLE terminal event; log is the only event type
# that self-truncates. 4KB is ample for a diagnostic message.


def _cap_failure_message(message: str) -> str:
    if len(message) <= _TERMINAL_FAILURE_MESSAGE_MAX_CHARS:
        return message
    marker = "…[truncated]"
    return message[: _TERMINAL_FAILURE_MESSAGE_MAX_CHARS - len(marker)] + marker


# guest-side mirror of outbound-channels.md's artifact level table (Channel 3) -- no module
# owns this table yet (the sealer's own version lives at `artifacts.py::seal_artifacts`, scoped to
# the local-SDK path); this hosted upload path needs its own "guest enforces it first" half.
_ARTIFACT_LEVEL_FORBIDDEN_KINDS: dict[ArtifactLevel, frozenset[ob.ArtifactKind]] = {
    ArtifactLevel.METADATA_ONLY: frozenset(
        {
            ob.ArtifactKind.RECORDING_COMBINED,
            ob.ArtifactKind.RECORDING_STEREO,
            ob.ArtifactKind.RECORDING_CUSTOMER,
            ob.ArtifactKind.RECORDING_ASSISTANT,
            ob.ArtifactKind.TRACE,
            ob.ArtifactKind.TOOL_TRACE,
            ob.ArtifactKind.TRANSCRIPT,
            ob.ArtifactKind.OTHER,
        }
    ),
    ArtifactLevel.TRACES: frozenset(
        {
            ob.ArtifactKind.RECORDING_COMBINED,
            ob.ArtifactKind.RECORDING_STEREO,
            ob.ArtifactKind.RECORDING_CUSTOMER,
            ob.ArtifactKind.RECORDING_ASSISTANT,
            ob.ArtifactKind.OTHER,
        }
    ),
    ArtifactLevel.TRACES_AND_RECORDINGS: frozenset({ob.ArtifactKind.OTHER}),
    ArtifactLevel.FULL: frozenset(),
    # `local-only` is rejected at hosted admission (`local_only_not_hosted`) per the contract --
    # this adapter should never see it for a hosted job; forbid everything as a defensive default.
    ArtifactLevel.LOCAL_ONLY: frozenset(ob.ArtifactKind),
}


class OutboundAdapter:
    """Implements `hosted_scheduler.OutboundPort` plus the extra surface the entrypoint itself
    needs (`stage_changed`, `baseline_frozen`, `parallelism_degraded`, `upload_artifact`,
    `push_manifest`, `emit_terminal`) — hosted_scheduler.py only names the five methods scenario
    code needs; everything else here is this module's own.

    Fencing (HostedFencedError) is caught INTERNALLY by every method, never re-raised: letting it
    escape into `HostedScheduler._emit()` (which catches bare `Exception` and tries to log through
    the very port that just raised) would silently swallow the fence and let the scheduler keep
    working an attempt that can no longer report anything. `is_fenced` is the flag the entrypoint's
    orchestration (and `cancel_requested`) polls instead.
    """

    def __init__(
        self,
        capabilities: ob.HostedCapabilities,
        *,
        events_spool: ob.OutboundSpool,
        events_client: ob.EventsClient,
        results_client: ob.ResultsClient,
        artifacts_client: ob.ArtifactsClient,
        channel_state: ob.ChannelState,
        extra_secret_values: tuple[str, ...] = (),
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        flush_window_seconds: float = ob.FLUSH_WINDOW_SECONDS,
    ) -> None:
        self._capabilities = capabilities
        self._spool = events_spool
        self._events = events_client
        self._results = results_client
        self._artifacts = artifacts_client
        self._channel_state = channel_state
        self._extra_secret_values = extra_secret_values
        self._clock = clock
        # `event_builder`'s own `extra_secret_values` binding is what lets
        # `build_event_record` redact `log.message`/`world_unhealthy.cause`/
        # `baseline_frozen.baseline_ref`/`terminal.failure.{code,message}` for every event this
        # adapter emits -- binding it here, alongside identity, gives Channel 1 full redaction coverage.
        self._event_builder = capabilities.event_builder(
            extra_secret_values=extra_secret_values
        )
        self._stage_started = False
        self._current_stage = HarnessStage.QUEUED
        self._uploaded_digests: set[str] = set()
        self._manifest_entries: list[dict[str, Any]] = []
        self._terminal_emitted = False
        # §0.6 v1.14 (exit code 4): the terminal record's own spool sequence, and whether the
        # platform ever permanently rejected it by name -- `terminal_undelivered` (below) needs to
        # tell "this specific record landed" apart from "some flush somewhere failed."
        self._terminal_sequence: int | None = None
        self._terminal_rejected = False
        self._scenario_counts: dict[str, int] = {
            "passed": 0,
            "failed": 0,
            "errored": 0,
            "skipped": 0,
        }
        self._fenced_error: Exception | None = None
        self._channel_failed_error: Exception | None = None
        # the 120s flush window (§5.5) -- armed once, at whichever comes first: a cancel
        # signal (`arm_flush_window` called explicitly by `run_job`'s `cancel_requested`) or the
        # terminal event (`emit_terminal` below arms it itself, so no caller can forget).
        self._flush_window_seconds = flush_window_seconds
        self._flush_window_start: float | None = None
        # job.artifacts is only known once job.json is parsed, which happens after this
        # adapter is built (capabilities load, and the "no channel on a capabilities failure"
        # contract, must come first) -- `configure_artifacts` below is called once it's available;
        # this default is never actually exercised in practice, just a safe placeholder shape.
        self._artifacts_policy = HarnessArtifactPolicy()
        # `recording_headroom_bytes` stays 0 -- this adapter has no visibility into how many
        # scenarios are still to run (or how large their recordings will be) at construction time,
        # unlike the scheduler; sizing it here would be a guess dressed up as enforcement.
        self._budget_tracker = ob.ArtifactBudgetTracker(
            self._artifacts_policy.max_artifact_bytes
        )
        # `would_admit` (check) and `record` (reserve) must run as one atomic step -- two
        # concurrent scenarios at W>1 racing the same remaining budget could otherwise both pass
        # the check against a snapshot neither has updated yet.
        self._artifact_budget_lock = asyncio.Lock()

    @property
    def is_fenced(self) -> bool:
        return self._fenced_error is not None

    @property
    def terminal_undelivered(self) -> bool:
        """The terminal was spooled (`emit_terminal` succeeded) but never confirmed delivered: the
        platform permanently rejected the terminal item by name, or the spool's watermark never
        reached the terminal's own sequence at all (channel exhaustion, a dead channel, or the
        flush window running out before delivery). Exit 0 would claim a flush that provably never
        happened. Fencing is checked by the caller first and always wins -- once fenced, whether
        the terminal was ALSO undelivered is moot."""
        if self._terminal_sequence is None or self.is_fenced:
            return False
        return (
            self._terminal_rejected or self._spool.watermark() < self._terminal_sequence
        )

    @property
    def scenario_counts(self) -> dict[str, int]:
        return dict(self._scenario_counts)

    def configure_artifacts(self, policy: HarnessArtifactPolicy) -> None:
        self._artifacts_policy = policy
        self._budget_tracker = ob.ArtifactBudgetTracker(policy.max_artifact_bytes)

    def arm_flush_window(self) -> None:
        if self._flush_window_start is None:
            self._flush_window_start = time.monotonic()

    def deadline(self) -> float | None:
        if self._flush_window_start is None:
            return None
        return self._flush_window_start + self._flush_window_seconds

    def _record_channel_error(self, exc: Exception) -> None:
        if isinstance(exc, ob.HostedFencedError):
            self._fenced_error = self._fenced_error or exc
        else:
            self._channel_failed_error = self._channel_failed_error or exc
        logger.error("outbound channel latched: %s", exc)

    def _guarded(self, fn: Callable[[], Any]) -> Any:
        """Runs one outbound client call. Catches `HostedFencedError`/`HostedChannelFailedError`
        so neither escapes as an ordinary exception (see class docstring)."""
        try:
            self._channel_state.check()
        except (
            ob.HostedFencedError,
            ob.HostedChannelFailedError,
            ob.HostedAttemptSupersededError,
        ) as exc:
            self._record_channel_error(exc)
            return None
        try:
            return fn()
        except (ob.HostedFencedError, ob.HostedChannelFailedError) as exc:
            self._channel_state.latch(exc)
            self._record_channel_error(exc)
            return None

    # -- events -------------------------------------------------------------------------------

    def _emit_event(
        self,
        *,
        stage: HarnessStage,
        type_: ob.OutboundEventType,
        payload: dict[str, Any],
    ) -> None:
        if self.is_fenced:
            return  # "stop emitting" -- no event of any type once fenced.
        if self._terminal_emitted:
            # `_bounded_close()` runs after the terminal is spooled -- an in-flight reconcile
            # inside it can still call back into world_unhealthy/log. v1.3's "terminal ... exactly
            # one, last emitted" is a hard invariant: anything after it is dropped locally, not
            # spooled, rather than silently landing after the event the platform already finalized on.
            # This also drops the rejected-event error log for anything the platform rejects on the
            # SAME flush that carries the terminal -- diagnostic-quality only, since the rejected
            # bytes are still recoverable as a `log`-kind artifact.
            logger.warning(
                "outbound event dropped after terminal: type=%s stage=%s",
                type_.value,
                stage.value,
            )
            return
        event_id = f"event_{uuid.uuid4().hex}"
        record = self._event_builder(
            event_id=event_id,
            emitted_at=self._clock(),
            stage=stage,
            type=type_,
            payload=payload,
        )
        spooled = self._spool.append(record)
        if type_ is ob.OutboundEventType.TERMINAL:
            self._terminal_sequence = spooled.sequence
        self._current_stage = stage

    async def _aemit_event(
        self,
        *,
        stage: HarnessStage,
        type_: ob.OutboundEventType,
        payload: dict[str, Any],
    ) -> None:
        # the spool append fsyncs the file AND its directory -- routed off the event loop so
        # it never stalls every other concurrently-running scenario at W>1.
        await asyncio.to_thread(
            self._emit_event, stage=stage, type_=type_, payload=payload
        )

    def stage_changed(self, to: HarnessStage) -> None:
        frm = self._current_stage.value if self._stage_started else None
        self._stage_started = True
        self._emit_event(
            stage=to,
            type_=ob.OutboundEventType.STAGE_CHANGED,
            payload={"from": frm, "to": to.value},
        )

    def baseline_frozen(self, *, inputs_digest: str, baseline_ref: str) -> None:
        self._emit_event(
            stage=HarnessStage.VALIDATING_ENVIRONMENT,
            type_=ob.OutboundEventType.BASELINE_FROZEN,
            payload={"inputs_digest": inputs_digest, "baseline_ref": baseline_ref},
        )

    def parallelism_degraded(
        self, *, requested: int, effective: int, reason: str
    ) -> None:
        self._emit_event(
            stage=HarnessStage.VALIDATING_ENVIRONMENT,
            type_=ob.OutboundEventType.PARALLELISM_DEGRADED,
            payload={"requested": requested, "effective": effective, "reason": reason},
        )

    def flush_events(
        self, *, deadline: float | None = None
    ) -> ob.EventsFlushResult | None:
        return self._guarded(lambda: self._events.flush(deadline=deadline))

    async def aflush_events(self, *, deadline: float | None = None) -> None:
        result = await asyncio.to_thread(self.flush_events, deadline=deadline)
        # a rejected event's own payload never reaches the platform any other way -- surface
        # it via a `log` event (error) and keep the bytes recoverable as a `log`-kind artifact,
        # keyed off `dropped_records` (captured before the spool physically drops them).
        if result is None or not result.rejected:
            return
        if self._terminal_sequence is not None and any(
            entry.get("sequence") == self._terminal_sequence
            for entry in result.rejected
        ):
            # A permanent-item rejection is never retried -- the spool physically drops the record,
            # so no later flush can ever redeliver it.
            self._terminal_rejected = True
        dropped_by_sequence = {
            record.sequence: record for record in result.dropped_records
        }
        for entry in result.rejected:
            sequence = entry.get("sequence")
            await self.log(
                level="error",
                message=(
                    f"event sequence={sequence} rejected by the platform: "
                    f"{entry.get('code', 'unknown')}: {entry.get('message', '')}"
                ),
            )
            record = dropped_by_sequence.get(sequence)
            if record is not None:
                await self.upload_artifact(record.body, kind=ob.ArtifactKind.LOG)

    # -- OutboundPort (hosted_scheduler.py) ----------------------------------------------------

    async def scenario_started(
        self, *, scenario_key: str, world_index: int, scenario_attempt: int
    ) -> None:
        await self._aemit_event(
            stage=HarnessStage.RUNNING,
            type_=ob.OutboundEventType.SCENARIO_STARTED,
            payload={
                "scenario_key": scenario_key,
                "world_index": world_index,
                "scenario_attempt": scenario_attempt,
            },
        )

    async def scenario_retried(
        self, *, scenario_key: str, from_world: int, to_world: int
    ) -> None:
        await self._aemit_event(
            stage=HarnessStage.RUNNING,
            type_=ob.OutboundEventType.SCENARIO_RETRIED,
            payload={
                "scenario_key": scenario_key,
                "from_world": from_world,
                "to_world": to_world,
            },
        )

    async def world_unhealthy(self, *, world_index: int, cause: str) -> None:
        # world_unhealthy.cause <=200 (WorldUnhealthyPayload hard-rejects over that, so this must
        # truncate BEFORE `_aemit_event`, not rely on the builder's own redaction, which runs
        # after this call and could not shrink an already-too-long string back into budget).
        redacted = ob.redact_outbound_text(cause, self._extra_secret_values)
        if len(redacted) > 200:
            redacted = redacted[:200]
        await self._aemit_event(
            stage=HarnessStage.RUNNING,
            type_=ob.OutboundEventType.WORLD_UNHEALTHY,
            payload={"world_index": world_index, "cause": redacted},
        )

    async def log(self, *, level: str, message: str) -> None:
        await self._aemit_event(
            stage=self._current_stage,
            type_=ob.OutboundEventType.LOG,
            payload={"level": level, "message": message},
        )

    async def receipt(self, receipt: ResultReceipt) -> None:
        if self.is_fenced:
            return
        # counted only once a push is actually attempted -- counting before this point would
        # include receipts that were never pushed (and the counts feed the terminal payload).
        self._scenario_counts[receipt.status] = (
            self._scenario_counts.get(receipt.status, 0) + 1
        )
        call: dict[str, Any] | None = None
        if receipt.call is not None and receipt.call.started_at is not None:
            transcript_artifact = receipt.call.transcript_artifact
            if transcript_artifact is not None:
                bare = transcript_artifact.split(":", 1)[-1]
                if bare not in self._uploaded_digests:
                    # null it rather than shipping a receipt the platform will 422
                    # (`artifact_unknown`) wholesale -- the contract explicitly blesses a null
                    # `transcript_artifact` "named in a log event."
                    await self.log(
                        level="error",
                        message=(
                            f"receipt for {receipt.scenario_key} references un-acked transcript "
                            f"artifact {transcript_artifact}; nulling it"
                        ),
                    )
                    transcript_artifact = None
            recording_artifacts: list[str] = []
            for artifact_id in receipt.call.recording_artifacts:
                bare = artifact_id.split(":", 1)[-1] if artifact_id else None
                if artifact_id and bare not in self._uploaded_digests:
                    await self.log(
                        level="error",
                        message=(
                            f"receipt for {receipt.scenario_key} references un-acked recording "
                            f"artifact {artifact_id}; dropping it"
                        ),
                    )
                    continue
                recording_artifacts.append(artifact_id)
            ended_at = receipt.call.ended_at
            if ended_at is None:
                # outbound.CallSummary.ended_at is a required str -- a call that started but
                # never finished (CallAborted's partial) would otherwise fail build_result_receipt's
                # validation and silently drop the whole receipt (HostedScheduler._emit's blanket
                # except swallows it).
                ended_at = receipt.call.started_at
                await self.log(
                    level="warning",
                    message=(
                        f"receipt for {receipt.scenario_key} has no call.ended_at; substituting "
                        "started_at"
                    ),
                )
            call = {
                "started_at": receipt.call.started_at,
                "ended_at": ended_at,
                "duration_ms": receipt.call.duration_ms,
                "turns": receipt.call.turns,
                "transcript_artifact": transcript_artifact,
                "recording_artifacts": recording_artifacts,
            }
            stop_reason = getattr(receipt.call, "stop_reason", None)
            if stop_reason:
                call["stop_reason"] = stop_reason
        elif receipt.call is not None:
            # `hosted_scheduler.CallSummary.started_at` is `str | None`, but
            # `outbound.CallSummary.started_at` requires a real timestamp -- per the contract a
            # call summary is only present once the call has genuinely started, so a call that
            # never started is omitted here rather than shipped with a value that would fail
            # `build_result_receipt`'s own validation.
            await self.log(
                level="warning",
                message=f"receipt for {receipt.scenario_key} has no call.started_at; omitting call",
            )
        failure: dict[str, Any] | None = None
        if receipt.failure is not None:
            # Redact before capping -- truncating first can cut a secret in half at
            # the boundary and leave exact-substring redaction unable to find the surviving piece.
            redacted_failure_message = ob.redact_outbound_text(
                receipt.failure.message, self._extra_secret_values
            )
            failure = {
                "domain": receipt.failure.domain,
                "stage": receipt.failure.stage,
                "code": receipt.failure.code,
                "message": _cap_failure_message(redacted_failure_message),
            }
        wire = ob.build_result_receipt(
            job_id=self._capabilities.job_id,
            attempt_id=self._capabilities.attempt_id,
            attempt_number=self._capabilities.attempt_number,
            scenario_key=receipt.scenario_key,
            scenario_id=receipt.scenario_id,
            scenario_attempt=receipt.scenario_attempt,
            world_index=receipt.world_index,
            status=receipt.status,
            sub_goals=[
                {"name": g.name, "held": g.held, "reason": g.reason, "judged": g.judged}
                for g in receipt.sub_goals
            ],
            evaluations=[_evaluation_wire(e) for e in receipt.evaluations],
            call=call,
            failure=failure,
            extra_secret_values=self._extra_secret_values,
        )
        push_result = await asyncio.to_thread(
            self._guarded, lambda: self._results.push(wire)
        )
        if push_result is not None and push_result.error is not None:
            # The contract's own obligation for a permanent rejection (e.g. 409 receipt_conflict,
            # 422 artifact_unknown): "the platform keeps the first; guest logs, no retry." `push()`
            # returns this rather than raising, so nothing inspected it before now.
            await self.log(
                level="error",
                message=(
                    f"receipt for {receipt.scenario_key} rejected by the platform: "
                    f"{push_result.error.code}: {push_result.error.message}"
                ),
            )
        await self.aflush_events()

    # -- artifacts (uploaded+acked BEFORE the referencing receipt) --------------------------

    async def upload_artifact(
        self,
        data: bytes,
        *,
        kind: ob.ArtifactKind,
        scenario_key: str | None = None,
        deadline: float | None = None,
    ) -> str | None:
        """Returns the `sha256:<64-hex>` id form the wire uses (`CallSummary.transcript_artifact`,
        `ArtifactManifestEntry.artifact_id`) — never the bare hex `ArtifactsClient.upload` itself
        takes, which is a different, easy-to-mix-up shape (this module's own report notes it)."""
        digest = hashlib.sha256(data).hexdigest()
        if digest in self._uploaded_digests:
            return f"sha256:{digest}"
        if self.is_fenced:
            return None
        # guest-side level admission + budget, both BEFORE the transport is ever touched
        # ("the guest enforces it first").
        forbidden = _ARTIFACT_LEVEL_FORBIDDEN_KINDS.get(
            self._artifacts_policy.level, frozenset()
        )
        if kind in forbidden:
            await self.log(
                level="error",
                message=(
                    f"artifact upload refused: kind={kind.value} forbidden at "
                    f"level={self._artifacts_policy.level.value}"
                ),
            )
            return None
        # check-and-reserve atomically, before the actual (slow, concurrency-safe) upload --
        # see the lock's own comment in __init__. A failed upload below leaves the reservation in
        # place rather than releasing it (`ArtifactBudgetTracker` has no release primitive): a
        # stuck-conservative budget is safe, an under-counted one that lets two racing uploads both
        # pass admission is not.
        async with self._artifact_budget_lock:
            if not self._budget_tracker.would_admit(kind, len(data), digest=digest):
                await self.log(
                    level="error",
                    message=(
                        f"artifact upload refused: budget exhausted (kind={kind.value}, "
                        f"size={len(data)})"
                    ),
                )
                return None
            self._budget_tracker.record(kind, len(data), digest=digest)
        result = await asyncio.to_thread(
            self._guarded,
            lambda: self._artifacts.upload(
                digest, data, kind=kind, scenario_key=scenario_key, deadline=deadline
            ),
        )
        if result is None or result.error is not None:
            code = (
                result.error.code
                if result is not None and result.error is not None
                else "fenced"
            )
            await self.log(
                level="error", message=f"artifact upload failed ({kind.value}): {code}"
            )
            return None
        self._uploaded_digests.add(digest)
        self._manifest_entries.append(
            {
                "artifact_id": f"sha256:{digest}",
                "kind": kind.value,
                "size": len(data),
                "scenario_key": scenario_key,
            }
        )
        return f"sha256:{digest}"

    async def push_manifest(
        self, *, complete: bool, deadline: float | None = None
    ) -> bool:
        if self.is_fenced:
            return False
        wire = ob.build_artifact_manifest(
            job_id=self._capabilities.job_id,
            attempt_id=self._capabilities.attempt_id,
            attempt_number=self._capabilities.attempt_number,
            entries=list(self._manifest_entries),
            complete=complete,
        )
        result = await asyncio.to_thread(
            self._guarded,
            lambda: self._artifacts.push_manifest(wire, deadline=deadline),
        )
        if result is None:
            return False
        if not result.delivered:
            error = result.error
            logger.error(
                "artifact manifest delivery failed: code=%s message=%s",
                error.code if error is not None else "unknown",
                error.message if error is not None else "no response",
            )
            return False
        return True

    async def ensure_terminal_artifacts(
        self,
        *,
        work_directory: Path,
        stage: HarnessStage,
        failure: dict[str, Any] | None,
    ) -> None:
        """Upload the three artifacts a complete platform manifest requires.

        The platform contract requires ``build``, ``result`` and ``log`` even when a run has no
        transcript/recording.  Previously the adapter only uploaded a log when an event was
        rejected, so every otherwise-successful complete manifest was deterministically rejected.
        """
        build_path = work_directory / "build.json"
        build = (
            build_path.read_bytes()
            if build_path.is_file()
            else b'{"status":"build metadata unavailable"}\n'
        )
        result = (
            json.dumps(
                {
                    "stage": stage.value,
                    "failure": failure,
                    "scenario_counts": self.scenario_counts,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
        log = (
            f"hosted harness terminal stage={stage.value}; "
            f"scenario_counts={json.dumps(self.scenario_counts, sort_keys=True)}\n"
        ).encode()
        for body, kind in (
            (build, ob.ArtifactKind.BUILD),
            (result, ob.ArtifactKind.RESULT),
            (log, ob.ArtifactKind.LOG),
        ):
            await self.upload_artifact(body, kind=kind, deadline=self.deadline())

    # -- terminal (exactly one terminal event, last emitted) --------------------------------

    async def emit_terminal(
        self,
        *,
        stage: HarnessStage,
        reason: ob.TerminalReason | None = None,
        failure: dict[str, Any] | None = None,
    ) -> bool:
        """Returns whether a terminal event was actually emitted (False when already emitted, or
        fenced). No caller reads this return value any more -- `drain()`'s own manifest push is
        gated on `is_fenced` instead; kept `bool` since a future caller may still want it."""
        if self._terminal_emitted or self.is_fenced:
            return False
        if failure is not None and isinstance(failure.get("message"), str):
            # redact BEFORE truncating -- the inverse order can cut a secret in half at the 4KB
            # boundary, and exact-substring redaction can no longer find the surviving fragment.
            redacted = ob.redact_outbound_text(
                failure["message"], self._extra_secret_values
            )
            failure = {**failure, "message": _cap_failure_message(redacted)}
        # the latch is set AFTER a successful append (below), not before -- a raise inside
        # `_emit_event` (an oversized payload, an invalid `failure.domain`) must not permanently
        # disable the terminal event.
        self._emit_event(
            stage=stage,
            type_=ob.OutboundEventType.TERMINAL,
            payload={
                "stage": stage.value,
                "reason": reason.value if reason is not None else None,
                "failure": failure,
                "scenario_counts": dict(self._scenario_counts),
            },
        )
        self._terminal_emitted = True
        # arm the flush window HERE, unconditionally -- "120s from the cancel signal / TTL /
        # terminal event," not only when a cancel was separately observed.
        self.arm_flush_window()
        return True

    async def flush_terminal(self, *, deadline: float | None = None) -> bool:
        """`emit_terminal` only appends the terminal record to the LOCAL spool -- a
        caller that pushes something else on the wire right after (skipped receipts) would
        otherwise risk `receipt()`'s own trailing `aflush_events()` delivering the terminal as a
        side effect of pushing THAT receipt, landing the terminal after it on the wire. Same
        bounded loop as `drain()` (a backlog bigger than one `EVENTS_MAX_BATCH` must not
        strand the terminal), stopping short of the manifest push -- that still belongs after
        skipped receipts, not here. Returns whether a fence was observed."""
        while True:
            before = self._spool.watermark()
            await self.aflush_events(deadline=deadline)
            if self.is_fenced or not self._spool.pending_since_watermark():
                break
            if self._spool.watermark() == before or (
                deadline is not None and time.monotonic() >= deadline
            ):
                break
        return self.is_fenced

    async def drain(self, *, complete: bool, deadline: float | None = None) -> bool:
        """Best-effort final delivery: events, then the artifact manifest (Sequencing: "terminal
        event -> receipts (incl. synthesized skipped) -> manifest" -- receipts are already pushed
        individually by `receipt()` as each scenario finishes). Returns whether a fence was
        observed during (or before) this call -- the fence most often lands on the very flush
        that carries the terminal event, so the caller's exit code must come from THIS return
        value, never a fence check taken before drain() ran.

        `aflush_events` delivers at most ONE `EVENTS_MAX_BATCH`-sized batch per call -- a
        backlog bigger than that (the rejected-event logging in `aflush_events` can grow one) would otherwise
        strand the terminal event, the highest sequence, undelivered while still exiting 0. Loops
        until the spool is actually empty, a fence is observed, a flush makes no further progress,
        or the deadline is gone -- whichever comes first.
        """
        while True:
            before = self._spool.watermark()
            await self.aflush_events(deadline=deadline)
            if self.is_fenced or not self._spool.pending_since_watermark():
                break
            if self._spool.watermark() == before or (
                deadline is not None and time.monotonic() >= deadline
            ):
                break
        await self.push_manifest(complete=complete, deadline=deadline)
        return self.is_fenced


def _evaluation_wire(evaluation: Any) -> dict[str, Any]:
    if evaluation.kind == "metric":
        return {
            # `MetricEvaluation.score: float` coerces `1` -> `1.0`; `build_result_receipt`
            # digests the RAW dict before that coercion, so an int here would digest-mismatch
            # against the model's own re-derivation and silently drop the receipt.
            "name": evaluation.name,
            "kind": "metric",
            "score": float(evaluation.score) if evaluation.score is not None else None,
            "reason": evaluation.reason,
        }
    return {
        "name": evaluation.name,
        "kind": "checkpoint",
        "passed": evaluation.passed,
        "reason": evaluation.reason,
    }


# =================================================================================================
# Cancellation -- spine §0 step 7 / outbound-channels.md "Cancellation signal": the gateway writes
# `cancel_path` then sends SIGTERM; the guest stops LAUNCHING new scenarios (not killing what's
# already running) and starts the 120s flush window.
# =================================================================================================


class CancelState:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._sigterm_seen = False

    def note_sigterm(self) -> None:
        self._sigterm_seen = True

    def requested(self) -> bool:
        return self._sigterm_seen or self._path.exists()

    def reason(self) -> ob.TerminalReason | None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        value = raw.get("reason") if isinstance(raw, dict) else None
        try:
            return ob.TerminalReason(value)
        except ValueError:
            return None


def install_sigterm_handler(cancel_state: CancelState) -> Callable[[], None]:
    """Best-effort: `signal.signal` only works on the process's main thread and raises
    `ValueError` anywhere else (e.g. inside a test running on a worker thread) — caught and
    turned into a no-op restore, since `cancel_requested` still works off the file alone."""

    def _handler(signum: int, frame: Any) -> None:
        del signum, frame
        cancel_state.note_sigterm()

    try:
        previous = signal.signal(signal.SIGTERM, _handler)
    except (ValueError, OSError):
        return lambda: None

    def _restore() -> None:
        try:
            signal.signal(signal.SIGTERM, previous)
        except (ValueError, OSError):
            pass

    return _restore


def default_install_sigterm_handler(cancel_state: CancelState) -> Callable[[], None]:
    return install_sigterm_handler(cancel_state)


def _resolve_hosted_public_url(
    capabilities: ob.HostedCapabilities,
    transport: ob.Transport,
    *,
    port: int,
    expires_in_seconds: int,
) -> str:
    endpoint = capabilities.endpoints.ingress
    if not endpoint:
        raise ProcessRuntimeError(
            "provider_lifecycle",
            "spawn_failed",
            "the platform did not grant an ingress capability for provider webhooks",
            domain=FailureDomain.INFRASTRUCTURE,
        )
    try:
        response = transport.request(
            "POST",
            endpoint,
            headers=capabilities.auth_headers(),
            json_body={
                "port": port,
                "expires_in_seconds": expires_in_seconds,
            },
            timeout=30.0,
        )
    except ob.TransportError as exc:
        raise ProcessRuntimeError(
            "provider_lifecycle",
            "spawn_failed",
            f"platform ingress request failed: {exc}",
            domain=FailureDomain.INFRASTRUCTURE,
        ) from exc
    body = response.body or {}
    url = body.get("url")
    if (
        response.status_code != 200
        or not isinstance(url, str)
        or not url.startswith("https://")
    ):
        raise ProcessRuntimeError(
            "provider_lifecycle",
            "spawn_failed",
            f"platform ingress request was rejected with HTTP {response.status_code}",
            domain=FailureDomain.INFRASTRUCTURE,
        )
    return url


# =================================================================================================
# Dependency injection -- every seam a test needs to replace with a fake, gathered in one place so
# `run_job` itself stays pure orchestration.
# =================================================================================================


@dataclass
class HostedEntrypointDeps:
    load_capabilities: Callable[[], ob.HostedCapabilities] = field(
        default=lambda: ob.load_capabilities()
    )
    bundle_source: BundleSource = field(default_factory=DefaultBundleSource)
    scenario_source: ScenarioSource = field(default_factory=NotWiredScenarioSource)
    build_transport: Callable[[], ob.Transport] = field(
        default=lambda: ob.RequestsTransport()
    )
    # Daytona forces the sandbox to a fixed non-root user (svc-control) and ignores os_user
    # overrides, so the guest cannot setuid/chown to the bundle's svc-agent/svc-tools/svc-data
    # users -- every process runs uniformly as svc-control. The bundle may still DECLARE those
    # users (the model validates them); they are simply not enforced at runtime here.
    build_provider: Callable[
        [ob.HostedCapabilities, ob.Transport], WorldProvisioner
    ] = field(
        default=lambda capabilities, transport: ProcessRuntimeProvider(
            user_resolver=lambda _name: None,
            require_declared_user=False,
            public_url_resolver=lambda port, ttl: _resolve_hosted_public_url(
                capabilities, transport, port=port, expires_in_seconds=ttl
            ),
            provider_attempt_id=capabilities.attempt_id,
            provider_expires_at=capabilities.expires_at,
        )
    )
    # The real call runner needs `OutboundAdapter.upload_artifact` to satisfy the invariant that
    # referenced artifacts are uploaded+acked BEFORE the receipt that names them -- the adapter is
    # threaded in once `run_job` has built it, rather than the CallRunner reaching for a global.
    # `CallRunnerContext` carries everything else `CallRunnerImpl` needs (job, bundle_dir,
    # evidence_seam, the target_provider secret map, attempt_number) that the bare
    # `CallRunner.run(scenario, runtime)` protocol has no room for.
    build_call_runner: Callable[["OutboundAdapter", CallRunnerContext], CallRunner] = (
        field(
            default=lambda adapter, context: _default_build_call_runner(
                adapter, context
            )
        )
    )
    build_world_factory: Callable[[Path], WorldFactory] = field(
        default=ProcessWorldFactory
    )
    retry_policy: Callable[[], ob.RetryPolicy] = field(default=lambda: ob.RetryPolicy())
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc))
    cancel_path: Path = field(default_factory=lambda: Path(CANCEL_SIGNAL_PATH))
    secrets_path: Path = field(default_factory=lambda: SECRETS_PATH)
    simulator_secrets_path: Path = field(default_factory=lambda: SIMULATOR_SECRETS_PATH)
    flush_window_seconds: float = ob.FLUSH_WINDOW_SECONDS
    install_sigterm_handler: Callable[[CancelState], Callable[[], None]] = field(
        default=default_install_sigterm_handler
    )
    events_spool_dir_name: str = EVENTS_SPOOL_DIR_NAME
    scenarios_client_kwargs: dict[str, Any] = field(default_factory=dict)

    def build_events_spool(self, work_directory: Path) -> ob.OutboundSpool:
        return ob.OutboundSpool(
            work_directory / self.events_spool_dir_name, "events", sequenced=True
        )

    def build_scenarios_client(
        self,
        capabilities: ob.HostedCapabilities,
        transport: ob.Transport,
        channel_state: ob.ChannelState,
    ) -> ScenariosClient:
        return ScenariosClient(
            capabilities,
            transport,
            channel_state=channel_state,
            **self.scenarios_client_kwargs,
        )

    def peek_secret_values(self) -> tuple[str, ...]:
        return peek_secret_values(self.secrets_path)

    def peek_target_provider_secret_values(
        self, secret_purposes: dict[str, str]
    ) -> dict[str, str]:
        return peek_target_provider_secret_values(self.secrets_path, secret_purposes)

    def peek_simulator_provider_secret_values(
        self, secret_purposes: dict[str, str]
    ) -> dict[str, str]:
        return peek_simulator_provider_secret_values(self.secrets_path, secret_purposes)

    def load_simulator_secret_values(self) -> dict[str, str]:
        return load_simulator_secret_values(self.simulator_secrets_path)


# =================================================================================================
# Scenario-entry validation at fetch (defense against karthik-integration-changes.md K1): the
# Scenario Generation Contract's own model may not carry `scenario_key` (or may hand back some
# other malformed shape) by the time `scenario_source.build()` returns it here, and
# `hosted_scheduler.py` reads `scenario.scenario_key`/`.sub_goals`/`.setup`/`.ready` at its own
# call sites with plain attribute access -- an attribute a pydantic/dataclass model never defined
# raises AttributeError, not a typed failure, deep inside the scheduler with no terminal event and
# a nonzero exit that reads as an infrastructure crash. Checked here with `getattr` (never direct
# attribute access) so a malformed entry is caught at the seam, before the scheduler ever touches
# it -- one bad entry fails the whole job as a typed FAILED terminal instead of crashing the guest.
# =================================================================================================

# No closed-vocabulary code names this defect specifically (the §2e/§2f tables are bundle/process
# concerns, not scenario-content ones) -- `scenario_preallocation_failed` is this module's own
# existing code for "the scenario set is not viable for this attempt," already scoped to stage
# `validating_scenarios`, and is reused here rather than inventing a new one. Domain `environment`
# (not `platform_sync`, its other use here): a malformed entry is a deterministic generation-stage
# content defect, not a transport failure, and fails identically on retry.
_SCENARIO_ENTRY_INVALID_CODE = "scenario_preallocation_failed"


def _validate_scenario_entry(entry: Any, *, index: int) -> str | None:
    """Returns a human-readable defect description, or `None` if `entry` looks usable by
    `hosted_scheduler.py`'s `Scenario` Protocol. Every check is a `getattr` with a default, never
    a direct attribute/index access -- the whole point is to survive a shape that lacks a field
    entirely, not just one that carries a wrong value.
    """
    scenario_key = getattr(entry, "scenario_key", None)
    if not isinstance(scenario_key, str) or not scenario_key:
        return f"scenario[{index}] has no non-empty scenario_key"
    label = f"scenario[{index}] ({scenario_key!r})"
    if not isinstance(getattr(entry, "scenario_id", None), str):
        return f"{label} has no scenario_id"
    if not callable(getattr(entry, "setup", None)):
        return f"{label} has no callable setup()"
    if not callable(getattr(entry, "ready", None)):
        return f"{label} has no callable ready()"
    sub_goals = getattr(entry, "sub_goals", None)
    if not isinstance(sub_goals, Sequence) or isinstance(sub_goals, (str, bytes)):
        return f"{label} has no sub_goals sequence"
    for goal_index, goal in enumerate(sub_goals):
        goal_name = getattr(goal, "name", None)
        if not isinstance(goal_name, str) or not goal_name:
            return f"{label} sub_goal[{goal_index}] has no non-empty name"
        if not callable(getattr(goal, "check", None)):
            return f"{label} sub_goal[{goal_index}] ({goal_name!r}) has no callable check()"
    return None


def _validate_scenarios(scenarios: Sequence[Any]) -> str | None:
    for index, entry in enumerate(scenarios):
        defect = _validate_scenario_entry(entry, index=index)
        if defect is not None:
            return defect
    return None


# =================================================================================================
# Orchestration -- steps 1-8, in order.
# =================================================================================================


async def run_job(
    job_path: Path,
    source: Path,
    output: Path,
    *,
    deps: HostedEntrypointDeps | None = None,
) -> int:
    """The guest's whole `main()` body. Returns the process exit code (§0.6) — `main()` below is
    the only caller that turns this into `SystemExit`, so tests can call this directly and assert
    on the return value."""
    deps = deps or HostedEntrypointDeps()
    work_directory = output.parent

    # This control-process-only channel is loaded before any Stage or CallRunner is constructed.
    # It is separate from secrets.json so platform simulator credentials never acquire the
    # ``target_provider`` purpose and therefore can never enter an agent process.
    simulator_secret_values = deps.load_simulator_secret_values()
    os.environ.update(simulator_secret_values)

    # 1. Boot -- capabilities. CapabilitiesError -> exit non-zero-and-NOT-3, no event (v1.3 table):
    # there is no channel yet to report a terminal event through.
    try:
        capabilities = deps.load_capabilities()
    except ob.CapabilitiesError as exc:
        logger.error("capabilities load failed: %s: %s", exc.code, exc.message)
        return EXIT_BOOT_FAILURE

    # Every line from here on is attributable. Done as early as the id is known, which is
    # immediately after capabilities load.
    configure_runner_logging(getattr(capabilities, "job_id", None))

    channel_state = ob.ChannelState()
    transport = deps.build_transport()
    retry_policy = deps.retry_policy()
    events_spool = deps.build_events_spool(work_directory)
    events_client = ob.EventsClient(
        capabilities,
        events_spool,
        transport,
        retry_policy=retry_policy,
        channel_state=channel_state,
    )
    results_client = ob.ResultsClient(
        capabilities, transport, retry_policy=retry_policy, channel_state=channel_state
    )
    artifacts_client = ob.ArtifactsClient(
        capabilities, transport, retry_policy=retry_policy, channel_state=channel_state
    )
    scenarios_client = deps.build_scenarios_client(
        capabilities, transport, channel_state
    )

    adapter = OutboundAdapter(
        capabilities,
        events_spool=events_spool,
        events_client=events_client,
        results_client=results_client,
        artifacts_client=artifacts_client,
        channel_state=channel_state,
        extra_secret_values=(
            *deps.peek_secret_values(),
            *tuple(simulator_secret_values.values()),
        ),
        clock=deps.clock,
        flush_window_seconds=deps.flush_window_seconds,
    )

    cancel_state = CancelState(deps.cancel_path)
    restore_sigterm = deps.install_sigterm_handler(cancel_state)
    # held outside the try so an exception on any path after this line still lets the
    # `finally` below close whatever was actually provisioned.
    pool: WorldPool | None = None
    call_runner: CallRunner | None = None

    def cancel_requested() -> bool:
        requested = cancel_state.requested() or adapter.is_fenced
        if requested:
            # "120s from the cancel signal / TTL / terminal event" -- whichever comes first;
            # a cancel/fence observed here starts the clock even though the terminal event (which
            # also arms it, unconditionally) may not land until much later.
            adapter.arm_flush_window()
        return requested

    async def _bounded_close() -> None:
        if pool is None:
            return
        remaining = adapter.deadline()
        if remaining is None:
            await pool.close()
            return
        timeout = max(0.0, remaining - time.monotonic())
        try:
            await asyncio.wait_for(pool.close(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "pool.close() did not finish within the remaining flush window (%.1fs); "
                "WorldPool already latches itself closed on entry to close(), so the top-level "
                "finally's own pool.close() call cannot retry the teardown -- the provisioner may "
                "be left not fully torn down until close()'s latch ordering changes",
                timeout,
            )

    async def _finish(
        stage: HarnessStage,
        *,
        reason: ob.TerminalReason | None = None,
        failure: dict[str, Any] | None = None,
        complete: bool,
        scheduler_result: tuple[HostedScheduler, RunResult] | None = None,
    ) -> int:
        """Terminal event -> drain -> bounded close, in that order, for every path that
        reaches a genuine terminal stage -- FAILED (via `_fail`; this now covers the pre-run
        failure branches too, not just post-run ones), CANCELED, an aborted RunResult, COMPLETED.
        Spending close()'s W-engine teardown time BEFORE a single terminal event is queued is
        exactly the inversion this ordering guards against.

        `scheduler_result` is only ever passed by the three call sites reached AFTER
        `scheduler.run()` -- pre-run terminals (`_fail`, the boundary `_canceled()` checks) have
        no `RunResult` and pass nothing, so this stays a no-op there."""
        # Artifact bytes must be uploaded before the terminal-referenced complete manifest.  The
        # terminal event itself remains before receipts and the manifest on the outbound channel.
        await adapter.ensure_terminal_artifacts(
            work_directory=work_directory,
            stage=stage,
            failure=failure,
        )
        await adapter.emit_terminal(stage=stage, reason=reason, failure=failure)
        # (outbound-channels.md v1.3 Sequencing): skipped receipts go out AFTER the terminal
        # event, never before -- placed here so no return path below can skip this call while
        # still delivering the terminal. A fenced result emits nothing further (the scheduler's
        # own no-op covers it too; checked here as well so a fenced run never even attempts it).
        # Best-effort like every other post-terminal emission in this module: a failure here must
        # not undo the terminal already spooled above or change the exit code below.
        if scheduler_result is not None:
            finished_scheduler, run_result = scheduler_result
            if run_result.fenced is None:
                try:
                    # `emit_terminal` above only spools the terminal locally -- flushed to the
                    # wire here, BEFORE the skipped-receipt pushes below, so a receipt's own
                    # trailing flush can never deliver the terminal as a side effect and land it
                    # after that receipt on the wire. Not itself wrapped in the wait_for below --
                    # it already threads the same deadline through every retry it makes, and it
                    # runs first, so its own delivery attempt is never the thing a timeout cuts off.
                    await adapter.flush_terminal(deadline=adapter.deadline())
                    if not adapter.is_fenced:
                        # `emit_skipped_receipts`/`receipt()` have no deadline plumbing of their
                        # own (`push()`/`aflush_events()` run with `deadline=None`) -- a
                        # degraded-but-alive events channel can retry every skipped scenario's
                        # receipt for the full `RetryPolicy` budget, scaling with how many
                        # scenarios were cut short and blowing past the flush window the gateway
                        # tears the sandbox down at. Bounded the same way `_bounded_close` bounds
                        # `pool.close()`: past the deadline, stop trying and fall through to close.
                        remaining = adapter.deadline()
                        timeout = (
                            None
                            if remaining is None
                            else max(0.0, remaining - time.monotonic())
                        )
                        try:
                            await asyncio.wait_for(
                                finished_scheduler.emit_skipped_receipts(run_result),
                                timeout=timeout,
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                "flush window exhausted before emit_skipped_receipts finished; "
                                "remaining scenarios' receipts were not sent"
                            )
                except Exception as exc:  # noqa: BLE001 - post-terminal telemetry, never fatal
                    logger.error("emit_skipped_receipts failed: %s", exc)
        # unlinked AFTER the terminal event, not before -- every terminal path shares the
        # "secrets are no longer needed past this point" rule, but an unlink failure (a read-only
        # or non-owned /run/futureagi) must never cost the one event that proves the job reached a
        # terminal state at all. missing_ok=True still no-ops on paths where the provider's own
        # §4.4 close() already removed the file; a genuine OSError is logged, not raised --
        # deleting an already-unneeded file is best-effort, not load-bearing.
        try:
            deps.secrets_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("secrets.json unlink failed: %s", exc)
        if adapter.is_fenced:
            await _bounded_close()
            return EXIT_FENCED
        # the exit code comes from drain()'s own post-hoc fence check (deadline computed AFTER
        # emit_terminal, which is what arms the flush window), never a stale pre-drain read.
        fenced = await adapter.drain(deadline=adapter.deadline(), complete=complete)
        await _bounded_close()
        if fenced:
            return EXIT_FENCED
        # §0.6 v1.14: the terminal was decided but the final drain could not flush it (the events
        # channel failed) or the platform permanently rejected the terminal item itself -- exit 0
        # would claim a flush that provably never happened and silently lose the run's evidence.
        if adapter.terminal_undelivered:
            return EXIT_TERMINAL_UNDELIVERED
        return EXIT_OK

    async def _canceled(
        *,
        scheduler_result: tuple[HostedScheduler, RunResult] | None = None,
    ) -> int:
        return await _finish(
            HarnessStage.CANCELED,
            reason=cancel_state.reason(),
            complete=False,
            scheduler_result=scheduler_result,
        )

    async def _fail(
        *, domain: FailureDomain, fail_stage: HarnessStage, code: str, message: str
    ) -> int:
        # routed through `_finish` -- terminal event first, pool close (bounded) after, for
        # every pre-run failure branch too, not just the post-run ones `_finish` already covered.
        return await _finish(
            HarnessStage.FAILED,
            failure={
                "domain": domain.value,
                "stage": fail_stage.value,
                "code": code,
                "message": message,
            },
            complete=True,
        )

    try:
        # 1 (cont'd). job.json (§0.2).
        try:
            job = load_job(job_path)
        except Exception as exc:  # noqa: BLE001 - a malformed job.json has no typed error to catch
            # EXIT_CRASHED (not a FAILED terminal) even though a channel now exists -- a
            # malformed job.json means `job.seed`/`job.agent`/etc are not trustworthy enough to
            # build a reportable failure from, and every downstream stage assumes a valid `job`.
            logger.error("job.json invalid: %s", exc)
            return EXIT_CRASHED

        if job.seed is None:
            logger.warning(
                "job.seed is null; spine §1 guarantees a concrete integer -- using 0"
            )
        job_seed = job.seed if job.seed is not None else 0
        parallelism = resolve_parallelism(job)
        secret_purposes = job_secret_purposes(job)
        # `ProcessRuntimeProvider` deletes `secrets.json` on its FIRST `provision()` call, inside
        # `pool.start()` below -- this capture must happen (and does: `job` is only just now
        # available, but `pool.start()` is still ~50 lines further down) strictly BEFORE that
        # point, same constraint `deps.peek_secret_values()` already satisfies for redaction,
        # above at adapter construction. Alias-preserving so `CallRunnerImpl` can pick e.g.
        # `LIVEKIT_API_KEY` out of the map by name.
        target_provider_secret_values = deps.peek_target_provider_secret_values(
            secret_purposes
        )
        # Platform simulator credentials arrive through simulator-secrets.json, not the
        # customer-controlled secrets.json. Preserve that separately loaded channel all the way
        # into CallRunnerContext. Any legacy simulator-purpose refs are merged first so the
        # platform channel wins on alias collisions and cannot be overridden by a submitted job.
        simulator_provider_secret_values = {
            **deps.peek_simulator_provider_secret_values(secret_purposes),
            **simulator_secret_values,
        }
        adapter.configure_artifacts(
            job.artifacts
        )  # level table + budget, now that job.json is known.

        adapter.stage_changed(HarnessStage.VALIDATING_ENVIRONMENT)
        await adapter.aflush_events()

        # Bundle authoring is not this module's (see the class docstrings above) -- injected.
        try:
            manifest, bundle_dir = await asyncio.to_thread(
                deps.bundle_source.load,
                job,
                source=source,
                work_directory=work_directory,
            )
        except BundleUnavailableError as exc:
            return await _fail(
                domain=FailureDomain.ENVIRONMENT,
                fail_stage=HarnessStage.VALIDATING_ENVIRONMENT,
                code=exc.code,
                message=exc.message,
            )

        if (manifest.metadata or {}).get("generic_harness") == "v1":
            from .certification import CertificationGateError, verify_runtime_certification

            try:
                certificate = await asyncio.to_thread(
                    verify_runtime_certification,
                    work_directory / "runtime-validation.json",
                    bundle_digest=manifest.digest,
                    source_digest=manifest.provenance.source_digest,
                )
            except CertificationGateError as exc:
                return await _fail(
                    domain=FailureDomain.ENVIRONMENT,
                    fail_stage=HarnessStage.VALIDATING_ENVIRONMENT,
                    code=exc.code,
                    message=exc.message,
                )
            await adapter.log(
                level="info",
                message=(
                    "generic harness certification accepted: "
                    f"fingerprint={certificate.fingerprint}; "
                    f"source={certificate.source.digest}; "
                    f"bundle={certificate.compiler.bundle_digest}; "
                    f"scenarios={certificate.checks.scenario_setup_ready}; "
                    f"tools={certificate.checks.tool_contract}; "
                    f"limitations={len(certificate.limitations)}"
                ),
            )

        # 2. Preflight -- BEFORE any provision (§2e). `parallelism` is the RAW requested value
        # (never clamped), so an out-of-1..8 W fails HERE with `parallelism_out_of_range`, per
        # §2e.7, rather than being silently laundered into a valid one.
        try:
            await asyncio.to_thread(
                preflight_bundle,
                bundle_dir,
                manifest,
                parallelism=parallelism,
                secret_refs=secret_purposes,
            )
        except PreflightError as exc:
            return await _fail(
                domain=FailureDomain.ENVIRONMENT,
                fail_stage=HarnessStage.VALIDATING_ENVIRONMENT,
                code=exc.code,
                message=exc.message,
            )

        # cancel/fence check at the post-preflight stage boundary.
        if cancel_requested():
            return await _canceled()

        # 4/5. Provision -- ProcessRuntimeProvider, hosted lane never passes
        # require_declared_user=False (the provider defaults it True on its own; the local lane's
        # opt-out is a construction-site concern, not this module's). §4.5b's provider mutex is
        # `WorldPool`'s own `_provider_lock` now (mutation-verified: it serializes
        # provision/reset/close/healthy under one lock) -- wired directly, no extra wrapper.
        provider = deps.build_provider(capabilities, transport)
        pool = WorldPool(
            provider,
            bundle=manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=work_directory,
            instances=parallelism,
            outbound=adapter,
        )
        try:
            await pool.start()
        except (ob.HostedFencedError, ob.HostedAttemptSupersededError):
            # defensive -- nothing today routes a channel error through `pool.start()`, but a
            # fenced attempt must never fall into the bare `Exception` handler below and get a
            # terminal FAILED event synthesized for it.
            await _bounded_close()
            return EXIT_FENCED
        except ob.HostedChannelFailedError as exc:
            # `_fail` closes the pool itself now, AFTER the terminal event (via `_finish`) --
            # closing here first was the same close-before-terminal inversion that the terminal -> drain -> close ordering fixes elsewhere.
            return await _fail(
                domain=FailureDomain.PLATFORM_SYNC,
                fail_stage=HarnessStage.VALIDATING_SCENARIOS,
                code="scenario_preallocation_failed",
                message=str(exc),
            )
        except ProcessRuntimeError as exc:
            # §2f's own CARRIED domain (never the flattened `infrastructure`/"provision_failed"
            # every provisioning failure used to get), stage `building_environment` per §2f.
            if adapter.is_fenced:
                await _bounded_close()
                return EXIT_FENCED
            return await _fail(
                domain=_process_runtime_error_domain(exc),
                fail_stage=HarnessStage.BUILDING_ENVIRONMENT,
                code=_section_2f_code(exc.code),
                message=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 - genuinely untyped -> infrastructure is the honest default
            if adapter.is_fenced:
                await _bounded_close()
                return EXIT_FENCED
            return await _fail(
                domain=FailureDomain.INFRASTRUCTURE,
                fail_stage=HarnessStage.BUILDING_ENVIRONMENT,
                code=_section_2f_code("provision_failed"),
                message=f"provision_failed: {exc}",
            )

        # baseline_frozen + parallelism_degraded from build.json. The whole
        # block is guarded -- a malformed build.json value must degrade to a `log`, never kill a
        # run that has already provisioned real worlds.
        degrade_emitted = False
        try:
            build_output = await asyncio.to_thread(load_build_output, work_directory)
        except WorldFactoryError:
            build_output = {}
        try:
            for store in build_output.get("stores", []):
                if store.get("baseline_reference"):
                    adapter.baseline_frozen(
                        inputs_digest=str(store.get("inputs_digest", "")),
                        baseline_ref=str(store.get("baseline_reference", "")),
                    )
            degrade_reason = build_output.get("degrade_reason")
            if degrade_reason:
                requested = int(
                    build_output.get("requested_parallelism") or parallelism
                )
                effective = int(build_output.get("effective_parallelism") or 1)
                # `ParallelismDegradedPayload` requires `1 <= effective < requested` --
                # `fixed_port` is recorded at `instances == 1` too (provider-side gap), where
                # `effective == requested == 1` is not representable as a degrade at all.
                if effective < requested:
                    adapter.parallelism_degraded(
                        requested=requested,
                        effective=effective,
                        reason=str(degrade_reason),
                    )
                    degrade_emitted = True
                else:
                    await adapter.log(
                        level="warning",
                        message=(
                            f"degrade recorded ({degrade_reason}) with requested==effective=="
                            f"{requested}; no parallelism_degraded event is representable"
                        ),
                    )
        except Exception as exc:  # noqa: BLE001 - malformed build.json must never crash a live run
            await adapter.log(
                level="warning",
                message=f"build.json degrade/baseline block malformed: {exc}",
            )
        # `pool.effective_size` is the ground truth for how many worlds actually exist --
        # if it's short of what was requested and build.json's own `degrade_reason` didn't already
        # announce it (a runtime degrade build.json doesn't record), say so loudly rather
        # than silently.
        if not degrade_emitted and pool.effective_size < parallelism:
            await adapter.log(
                level="warning",
                message=(
                    f"world pool effective_size={pool.effective_size} < requested "
                    f"parallelism={parallelism}, but build.json recorded no representable "
                    "degrade_reason"
                ),
            )
        await adapter.aflush_events()

        # cancel/fence check at the post-provision stage boundary.
        if cancel_requested():
            return await _canceled()

        # 3. Scenario pre-allocation (spine §5 step 3.5). Generation is not this module's; the
        # pre-allocation CLIENT (ScenariosClient) is.
        adapter.stage_changed(HarnessStage.VALIDATING_SCENARIOS)
        await adapter.aflush_events()
        world_factory = deps.build_world_factory(work_directory)
        # An injected `ScenarioSource` (every test, every future caller) always wins -- the real
        # bundle-reading adapter (scenario_source.py) only steps in when the default
        # `NotWiredScenarioSource` is still in place AND the bundle actually carries a `scenarios/`
        # directory (the LAYOUT DECISION's presence test). A bundle without one keeps the existing
        # typed `ScenarioSourceNotWired` failure below -- no regression for a job whose scenarios
        # are not generated yet.
        scenario_source = deps.scenario_source
        if isinstance(scenario_source, NotWiredScenarioSource) and bundle_has_scenarios(
            bundle_dir
        ):
            scenario_source = BundleScenarioSource()
        try:
            scenarios = await scenario_source.build(
                job,
                manifest,
                scenarios_client,
                pool=pool,
                world_factory=world_factory,
                bundle_dir=bundle_dir,
            )
        except (ob.HostedFencedError, ob.HostedAttemptSupersededError):
            # `ScenariosClient._post` re-raises these after latching `channel_state` -- a fence
            # here must exit 3 with no terminal event, never fall through to the generic handler.
            await _bounded_close()
            return EXIT_FENCED
        except ob.HostedChannelFailedError as exc:
            # `ScenariosClient._post` has already latched `channel_state` by the time this branch
            # runs -- `emit_terminal` still spools the terminal locally (it never touches the
            # network), but the drain that would flush it inherits the same latched channel and can
            # never deliver. `_finish` detects exactly this (the terminal's own spool sequence never
            # gets acked) and reports it honestly rather than claiming a flush that cannot happen.
            return await _fail(
                domain=FailureDomain.PLATFORM_SYNC,
                fail_stage=HarnessStage.VALIDATING_SCENARIOS,
                code="scenario_preallocation_failed",
                message=str(exc),
            )
        except (ScenarioSourceNotWired, ScenarioPreallocationError) as exc:
            if adapter.is_fenced:
                await _bounded_close()
                return EXIT_FENCED
            return await _fail(
                domain=FailureDomain.PLATFORM_SYNC,
                fail_stage=HarnessStage.VALIDATING_SCENARIOS,
                code="scenario_preallocation_failed",
                message=str(exc),
            )
        except ScenarioDocumentInvalid as exc:
            # A scenario document that will not even compile is a generation-stage content defect
            # (deterministic on retry), never a transport failure -- same rationale as
            # `_SCENARIO_ENTRY_INVALID_CODE`'s other use below, reused rather than inventing a new
            # code for the same pair of (domain, stage).
            if adapter.is_fenced:
                await _bounded_close()
                return EXIT_FENCED
            return await _fail(
                domain=FailureDomain.ENVIRONMENT,
                fail_stage=HarnessStage.VALIDATING_SCENARIOS,
                code=_SCENARIO_ENTRY_INVALID_CODE,
                message=str(exc),
            )

        # Defense against a malformed scenario entry (K1) reaching the scheduler, which reads
        # `scenario_key`/`sub_goals`/`setup`/`ready` with plain attribute access and would raise
        # AttributeError instead of failing the job cleanly.
        scenario_defect = _validate_scenarios(scenarios)
        if scenario_defect is not None:
            if adapter.is_fenced:
                await _bounded_close()
                return EXIT_FENCED
            return await _fail(
                domain=FailureDomain.ENVIRONMENT,
                fail_stage=HarnessStage.VALIDATING_SCENARIOS,
                code=_SCENARIO_ENTRY_INVALID_CODE,
                message=scenario_defect,
            )

        # cancel/fence check at the post-pre-allocation stage boundary.
        if cancel_requested():
            return await _canceled()

        # 5/6. Scheduler wiring.
        adapter.stage_changed(HarnessStage.RUNNING)
        await adapter.aflush_events()
        call_runner_context = CallRunnerContext(
            job=job,
            bundle_dir=bundle_dir,
            work_directory=work_directory,
            evidence_seam=manifest.runtime.evidence_seam,
            target_provider_secret_values=target_provider_secret_values,
            simulator_provider_secret_values=simulator_provider_secret_values,
            attempt_number=capabilities.attempt_number,
            source_directory=source,
        )
        call_runner = deps.build_call_runner(adapter, call_runner_context)
        scheduler = HostedScheduler(
            pool=pool,
            world_factory=world_factory,
            call_runner=call_runner,
            outbound=adapter,
            job_seed=job_seed,
            cancel_requested=cancel_requested,
        )
        result: RunResult = await scheduler.run(scenarios)

        # 7. Terminal + exit codes. Terminal -> drain -> close (bounded), never close() first.
        if cancel_state.requested():
            return await _canceled(scheduler_result=(scheduler, result))
        if result.aborted is not None:
            # v1.14 §5.4 pass-through: `result.aborted.domain`/`.code` carry straight through, not
            # flattened to a fixed infrastructure/world_pool_exhausted pair -- the scheduler already
            # resolves whether every world failed on the SAME never-retried §2f code (environment
            # or agent domain) or a mixed set (`world_pool_exhausted`/`infrastructure`), and this
            # just reports that verdict unchanged.
            return await _finish(
                HarnessStage.FAILED,
                failure={
                    "domain": result.aborted.domain,
                    "stage": HarnessStage.RUNNING.value,
                    "code": result.aborted.code,
                    "message": result.aborted.message,
                },
                # The scheduler has emitted the errored receipt plus synthesized skipped
                # receipts for every scenario before reaching this branch. "complete" is an
                # evidence-delivery property, not a success flag: only cancellation may submit
                # an intentionally partial manifest.
                complete=True,
                scheduler_result=(scheduler, result),
            )
        # `complete: true` only on a genuine, nothing-cut-short COMPLETED terminal.
        return await _finish(
            HarnessStage.COMPLETED, complete=True, scheduler_result=(scheduler, result)
        )
    finally:
        if pool is not None:
            try:
                await (
                    pool.close()
                )  # idempotent backstop for any path above that missed one.
            except Exception:  # noqa: BLE001 - a finally must never mask the real exit path
                logger.exception("pool.close() failed in the run_job finally backstop")
        if call_runner is not None:
            close_call_runner = getattr(call_runner, "close", None)
            if callable(close_call_runner):
                try:
                    result = close_call_runner()
                    if hasattr(result, "__await__"):
                        await result
                except Exception:  # noqa: BLE001 - cleanup must never mask the real exit path
                    logger.exception("call runner close failed in the run_job finally backstop")
        restore_sigterm()


# =================================================================================================
# CLI -- spine §0 step 5's frozen invocation line:
# `python -m fi.alk.harness.hosted_entrypoint /work/job.json --source /work/source --output
# /work/artifacts`
# =================================================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alk-harness-worker")
    parser.add_argument("job", type=Path, help="typed HarnessJob JSON (/work/job.json)")
    parser.add_argument(
        "--source", required=True, type=Path, help="/work/source checkout root"
    )
    parser.add_argument("--output", required=True, type=Path, help="/work/artifacts")
    args = parser.parse_args(argv)
    return asyncio.run(run_job(args.job, args.source, args.output))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANCEL_SIGNAL_PATH",
    "EXIT_BOOT_FAILURE",
    "EXIT_CRASHED",
    "EXIT_FENCED",
    "EXIT_OK",
    "EXIT_TERMINAL_UNDELIVERED",
    "BundleSource",
    "BundleUnavailableError",
    "CallRunnerNotWired",
    "CancelState",
    "DefaultBundleSource",
    "HostedEntrypointDeps",
    "NotWiredCallRunner",
    "NotWiredScenarioSource",
    "OutboundAdapter",
    "ProcessWorldFactory",
    "ScenarioPreallocationError",
    "ScenarioSource",
    "ScenarioSourceNotWired",
    "ScenariosClient",
    "WorldFactoryError",
    "install_sigterm_handler",
    "job_secret_purposes",
    "load_build_output",
    "load_job",
    "main",
    "peek_secret_values",
    "resolve_parallelism",
    "row_counts_for_capability",
    "run_job",
]
