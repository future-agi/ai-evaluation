"""The hosted lane's `CallRunner` — places one simulated LiveKit voice call and reports what
happened, satisfying `hosted_scheduler.CallRunner` exactly.

Three sub-systems (world-handle-interface.md, hosted-execution-seams.md v1.15 §2a):

1. **Placing the call.** The customer agent is already running INSIDE the Daytona sandbox, as a
   world process the bundle's provisioner spawned (`process_runtime.py`) and registered with
   LiveKit cloud under `LIVEKIT_AGENT_NAME=agent-w{WORLD_INDEX}`-style identity. This runner never
   starts or manages that process. It drives `SimulationRunner` IN-PROCESS with a
   `SimulationSpec` built by `simulator_voice.simulation_spec`, the same builder the local lane
   uses; only the value lookup differs, resolving from job config and the bundle's scenario
   document rather than `HARNESS_*` env vars. Do not rebuild the spec here: the two lanes drifted
   for exactly that reason. The local-only webhook/subprocess plumbing `run/call.py` and
   `run/live.py` use is neither available nor appropriate in the guest.
2. **Collecting evidence.** The bundle declares exactly one `runtime.evidence_seam`:
   `http_tool` or `tool_trace`. `http_tool` has NO guest-side capture surface anywhere in this
   repo today (see `_collect_http_tool_calls`'s docstring — a verified finding, not an assumption)
   and is intentionally left returning zero calls rather than inventing a capture proxy.
   `tool_trace` is read from the world's own postgres database against an unpinned, isolated
   convention (see `_collect_tool_trace_calls`'s docstring). Either way, zero calls captured is
   never fabricated into something else — the scheduler's own `evidence_missing` retry-once policy
   is the contract-correct handling for "no evidence."
3. **Uploading artifacts.** The transcript and any produced recordings are uploaded through the
   adapter's `upload_artifact` (content-addressed, budget/level-gated, returns `None` on refusal —
   never an exception) BEFORE this runner returns, so `CallOutcome.transcript_artifact`/
   `recording_artifacts` only ever carry ids the platform has already acked.
"""

from __future__ import annotations

import atexit
import asyncio
import json
import logging
import os
import stat
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Protocol

from fi import simulate
from fi.simulate.runtime import (
    SimulationSpec,
    new_run_id,
)
from fi.simulate.runtime.report import SimulationReport
from fi.simulate.runtime.run import TestCaseStatus
from fi.simulate.runtime.runner import SimulationRunner

from .background_noise import scenario_source
from .bundle_v2 import EvidenceSeam
from .hosted_scheduler import CallAborted, CallOutcome
from .hosted_scheduler import Scenario as HostedScenario
from .job import ExecutionMode, HarnessJob, ProviderExecutionMode
from .outbound import ArtifactKind, format_rfc3339_millis
from .process_runtime import EnvironmentRuntime
from .scenario import DEFAULT_VOICEMAIL_STYLE, voicemail_enabled
from .voicemail_audio import clip_for
from .simulator_voice import (
    CLEANUP_TIMEOUT_SECONDS,
    CONNECT_TIMEOUT_SECONDS,
    READINESS_TIMEOUT_SECONDS,
    caller_scenario,
    simulation_spec,
    simulator_definition,
)
from .world.errors import WorldUnavailable
from .world.runtime import Call

logger = logging.getLogger(__name__)

# --- credential aliases / config keys a voice job must carry ---

LIVEKIT_API_KEY_ALIAS = "LIVEKIT_API_KEY"
LIVEKIT_API_SECRET_ALIAS = "LIVEKIT_API_SECRET"
LIVEKIT_URL_ALIAS = "LIVEKIT_URL"
DEEPGRAM_API_KEY_ALIAS = "DEEPGRAM_API_KEY"
CARTESIA_API_KEY_ALIAS = "CARTESIA_API_KEY"
GEMINI_API_KEY_ALIAS = "GEMINI_API_KEY"
GOOGLE_API_KEY_ALIAS = "GOOGLE_API_KEY"
GOOGLE_APPLICATION_CREDENTIALS_JSON_ALIAS = "GOOGLE_APPLICATION_CREDENTIALS_JSON"
GOOGLE_APPLICATION_CREDENTIALS_ALIAS = "GOOGLE_APPLICATION_CREDENTIALS"
GOOGLE_CLOUD_PROJECT_ALIAS = "GOOGLE_CLOUD_PROJECT"
GOOGLE_CLOUD_LOCATION_ALIAS = "GOOGLE_CLOUD_LOCATION"
GOOGLE_GENAI_USE_VERTEXAI_ALIAS = "GOOGLE_GENAI_USE_VERTEXAI"
OPENAI_API_KEY_ALIAS = "OPENAI_API_KEY"
VAPI_API_KEY_ALIAS = "VAPI_API_KEY"
RETELL_API_KEY_ALIAS = "RETELL_API_KEY"
SIMULATOR_LLM_PROVIDER_ALIAS = "SIMULATOR_LLM_PROVIDER"
SIMULATOR_LLM_MODEL_ALIAS = "SIMULATOR_LLM_MODEL"
SIMULATOR_STT_PROVIDER_ALIAS = "SIMULATOR_STT_PROVIDER"
SIMULATOR_STT_MODEL_ALIAS = "SIMULATOR_STT_MODEL"
SIMULATOR_TTS_PROVIDER_ALIAS = "SIMULATOR_TTS_PROVIDER"
SIMULATOR_TTS_MODEL_ALIAS = "SIMULATOR_TTS_MODEL"
BACKGROUND_NOISE_ALIAS = "ALK_BACKGROUND_NOISE"
BACKGROUND_NOISE_CATALOG_ALIAS = "ALK_BACKGROUND_NOISE_CATALOG"
BACKGROUND_NOISE_VOLUME_ALIAS = "HARNESS_BACKGROUND_NOISE_VOLUME"
CALL_DIRECTION_ALIAS = "ALK_CALL_DIRECTION"
VOICEMAIL_CLIP_ALIAS = "HARNESS_VOICEMAIL_CLIP"
VOICEMAIL_CLIP_TONE_ALIAS = "HARNESS_VOICEMAIL_CLIP_HAS_TONE"
VOICEMAIL_CLIP_TEXT_ALIAS = "HARNESS_VOICEMAIL_CLIP_TRANSCRIPT"
LIVEKIT_URL_CONFIG_KEY = "livekit_url"
CALL_TIMEOUT_CONFIG_KEY = "voice_call_timeout_seconds"

_SIMULATOR_PLATFORM_ALIAS_MAP = {
    "SIMULATOR_LIVEKIT_URL": LIVEKIT_URL_ALIAS,
    "SIMULATOR_LIVEKIT_API_KEY": LIVEKIT_API_KEY_ALIAS,
    "SIMULATOR_LIVEKIT_API_SECRET": LIVEKIT_API_SECRET_ALIAS,
    "SIMULATOR_DEEPGRAM_API_KEY": DEEPGRAM_API_KEY_ALIAS,
    "SIMULATOR_CARTESIA_API_KEY": CARTESIA_API_KEY_ALIAS,
    "SIMULATOR_GEMINI_API_KEY": GEMINI_API_KEY_ALIAS,
    "SIMULATOR_GOOGLE_API_KEY": GOOGLE_API_KEY_ALIAS,
    "SIMULATOR_GOOGLE_APPLICATION_CREDENTIALS_JSON": (
        GOOGLE_APPLICATION_CREDENTIALS_JSON_ALIAS
    ),
    "SIMULATOR_GOOGLE_CLOUD_PROJECT": GOOGLE_CLOUD_PROJECT_ALIAS,
    "SIMULATOR_GOOGLE_CLOUD_LOCATION": GOOGLE_CLOUD_LOCATION_ALIAS,
    "SIMULATOR_GOOGLE_GENAI_USE_VERTEXAI": GOOGLE_GENAI_USE_VERTEXAI_ALIAS,
    "SIMULATOR_OPENAI_API_KEY": OPENAI_API_KEY_ALIAS,
}

_DEFAULT_CALL_TIMEOUT_SECONDS = 300.0

# sdk_voice.py::build_spec's own phase-overhead constants, reused verbatim so this runner's
# outer budget composes with the SDK's internal one the same way the local template does.
_RUN_SECONDS_PAD_SECONDS = 60.0
# Headroom beyond `spec.execution.timeout.run_seconds` -- SimulationRunner.run() already wraps
# `plugin.run(...)` in its OWN `asyncio.wait_for(..., timeout=spec.execution.timeout.run_seconds)`
# (runner.py) and catches that TimeoutError into a graceful `SimulationReport(status=TIMED_OUT)`.
# This runner's own outer wait_for must stay LARGER than that so the SDK's internal timeout fires
# first in the ordinary case; it only ever fires itself for a genuinely hung SDK (a real post-dial
# machinery failure) -- a runner-owned asyncio.wait_for as the last-resort bound.
_OUTER_WAIT_FOR_PAD_SECONDS = 60.0

# Unpinned by any contract and no producer exists yet. Isolated as one
# constant + two functions (`_clear_tool_trace_calls`, `_collect_tool_trace_calls`) so a real
# producer's disagreement on the name/shape is a one-line change.
_TOOL_TRACE_TABLE = "_alk_tool_trace"

_RESULT_TRUNCATE_CHARS = 2000

# Turns a timed-out call needs before it is worth grading rather than aborting. Low on purpose: the
# question is only whether a conversation happened at all.
_GRADEABLE_AFTER_TIMEOUT_TURNS = 4

# The real engine's zero-turn "agent joined but never spoke" failure codes (engines/livekit.py::
# _conversation_outcome) -- see `_translate_report`'s `is_silent_agent` gate for why these two, and
# only at zero turns, get mapped to a normal CallOutcome instead of a CallAborted.
_SILENT_AGENT_FAILURE_CODES = frozenset(
    {"no_conversation", "conversation_silence_timeout"}
)

# C3 §4.5: the engine's dispatch-ack ladder marks +60s exhaustion with this structured
# `failure.code`. Matched here to pass the marker through on `CallAborted.marker` (never
# string-matched from `failure.message`). Kept as a literal — the engine module that owns it
# requires the optional `livekit` dependency this runner must import without.
_VOICE_DISPATCH_UNACKNOWLEDGED = "voice_dispatch_unacknowledged"


# --- collaborator seams (named, injectable test boundaries) -----------------------------------


class ArtifactUploader(Protocol):
    """Narrow slice of `hosted_entrypoint.OutboundAdapter` -- avoids importing that module here
    (it imports THIS module's factory to wire the real CallRunner; importing it back would be
    circular)."""

    async def upload_artifact(
        self,
        data: bytes,
        *,
        kind: ArtifactKind,
        scenario_key: str | None = None,
        deadline: float | None = None,
    ) -> str | None: ...


PlaceCall = Callable[[SimulationSpec], Awaitable[SimulationReport]]


async def _default_place_call(spec: SimulationSpec) -> SimulationReport:
    return await SimulationRunner().run(spec)


@dataclass(frozen=True)
class CallRunnerContext:
    """Everything `hosted_entrypoint.py`'s `run_job` already has in scope by the wiring point
    (~1662) that the real `CallRunnerImpl` needs but the bare `CallRunner` protocol signature
    (`run(scenario, runtime)`) has no room to carry. Threaded through the EXTENDED
    `build_call_runner(adapter, context)` seam."""

    job: HarnessJob
    bundle_dir: Path
    work_directory: Path
    evidence_seam: EvidenceSeam | None
    target_provider_secret_values: Mapping[str, str]
    attempt_number: int
    source_directory: Path | None = None
    simulator_provider_secret_values: Mapping[str, str] = field(default_factory=dict)


# --- pre-dial validation -----------------------------------------------------------------------


@dataclass(frozen=True)
class _MissingVoiceConfig:
    aliases: tuple[str, ...]
    config_keys: tuple[str, ...]

    def message(self) -> str:
        parts = []
        if self.aliases:
            parts.append("secrets=" + ",".join(self.aliases))
        if self.config_keys:
            parts.append("config=" + ",".join(self.config_keys))
        return "voice_capability_unavailable: missing " + "; ".join(parts)


def _resolve_connector(
    job: HarnessJob, target_provider_secret_values: Mapping[str, str]
) -> str:
    """Pin the job's transport connector from the credentials actually present.

    A fresh one-shot ships ``job.json`` with ``connector="auto"``: the platform only writes the
    authored connector back onto the job *after* authoring, by which time this guest has already
    booted from the un-resolved payload.  Mirror the platform rule so a LiveKit-credentialed
    ``auto`` job dispatches to the target agent instead of the simulator lane with no identity.
    """
    connector = job.agent.connector.strip().lower()
    if connector != "auto":
        return connector
    if job.agent.config.get(
        LIVEKIT_URL_CONFIG_KEY
    ) or target_provider_secret_values.get(LIVEKIT_URL_ALIAS):
        return "livekit"
    if target_provider_secret_values.get(VAPI_API_KEY_ALIAS):
        return "vapi"
    if target_provider_secret_values.get(RETELL_API_KEY_ALIAS):
        return "retell"
    return connector


def _check_config(
    job: HarnessJob,
    target_provider_secret_values: Mapping[str, str],
    simulator_values: Mapping[str, str] | None = None,
) -> _MissingVoiceConfig | None:
    simulator_values = simulator_values or {}

    def simulator_value(alias: str) -> str | None:
        # Hosted runs supply platform-owned simulator credentials in the control process.  The
        # target-provider value remains a backwards-compatible fallback for local SDK callers.
        return simulator_values.get(alias) or (
            target_provider_secret_values.get(alias)
            if job.execution is ExecutionMode.LOCAL
            else None
        )

    config = job.agent.config
    llm_provider = str(
        config.get("simulator_llm_provider")
        or simulator_value(SIMULATOR_LLM_PROVIDER_ALIAS)
        or "google"
    ).lower()
    stt_provider = str(
        config.get("simulator_stt_provider")
        or simulator_value(SIMULATOR_STT_PROVIDER_ALIAS)
        or "deepgram"
    ).lower()
    tts_provider = str(
        config.get("simulator_tts_provider")
        or simulator_value(SIMULATOR_TTS_PROVIDER_ALIAS)
        or "deepgram"
    ).lower()

    connector = _resolve_connector(job, target_provider_secret_values)
    livekit_values = (
        target_provider_secret_values if connector == "livekit" else simulator_values
    )
    required = [LIVEKIT_API_KEY_ALIAS, LIVEKIT_API_SECRET_ALIAS]
    if connector == "vapi":
        required.append(VAPI_API_KEY_ALIAS)
    elif connector == "retell":
        required.append(RETELL_API_KEY_ALIAS)
    if "deepgram" in {stt_provider, tts_provider}:
        if not simulator_value(DEEPGRAM_API_KEY_ALIAS):
            required.append(DEEPGRAM_API_KEY_ALIAS)

    def credential(alias: str) -> str | None:
        if alias in {LIVEKIT_API_KEY_ALIAS, LIVEKIT_API_SECRET_ALIAS}:
            return livekit_values.get(alias)
        if alias in {VAPI_API_KEY_ALIAS, RETELL_API_KEY_ALIAS}:
            return target_provider_secret_values.get(alias)
        return simulator_value(alias)

    missing_aliases = [alias for alias in required if not credential(alias)]

    if llm_provider == "google":
        has_api_key = bool(
            simulator_value(GEMINI_API_KEY_ALIAS)
            or simulator_value(GOOGLE_API_KEY_ALIAS)
        )
        has_vertex_adc = bool(
            (
                simulator_value(GOOGLE_APPLICATION_CREDENTIALS_ALIAS)
                or simulator_value(GOOGLE_APPLICATION_CREDENTIALS_JSON_ALIAS)
            )
            and simulator_value(GOOGLE_CLOUD_PROJECT_ALIAS)
        )
        if not has_api_key and not has_vertex_adc:
            missing_aliases.append(
                f"{GEMINI_API_KEY_ALIAS}_or_{GOOGLE_API_KEY_ALIAS}_or_VERTEX_ADC"
            )
    elif llm_provider == "openai" and not simulator_value(OPENAI_API_KEY_ALIAS):
        missing_aliases.append(OPENAI_API_KEY_ALIAS)

    has_livekit_url = bool(
        config.get(LIVEKIT_URL_CONFIG_KEY) or livekit_values.get(LIVEKIT_URL_ALIAS)
    )
    missing_config_keys = [] if has_livekit_url else [LIVEKIT_URL_CONFIG_KEY]
    if not missing_aliases and not missing_config_keys:
        return None
    return _MissingVoiceConfig(tuple(missing_aliases), tuple(missing_config_keys))


def _canonical_simulator_secrets(values: Mapping[str, str]) -> dict[str, str]:
    """Translate platform-only aliases into the names expected by simulator plugins."""
    return {
        _SIMULATOR_PLATFORM_ALIAS_MAP.get(alias, alias): value
        for alias, value in values.items()
    }


def _dispatch_agent_name(runtime: EnvironmentRuntime) -> str | None:
    """The ONLY place this repo reads the dispatch-identity metadata key, so a
    change to the key name/convention is a one-line adapt. The provisioner
    mirrors the agent process's rendered LIVEKIT_AGENT_NAME here; a bundle
    that declares none (or an ambiguous set) leaves the key absent and the
    caller's typed `CallAborted` below fires."""
    value = runtime.metadata.get("livekit_agent_name")
    return value.strip() if isinstance(value, str) and value.strip() else None


# --- scenario document re-read (the _CompiledScenario the scheduler hands over carries no
# persona/instruction -- scenario_source.py:170-184's deliberately narrow Scenario-protocol
# shape) ------------------------------------------------------------------------------------


class _ScenarioDocumentUnavailable(RuntimeError):
    pass


def _read_scenario_document(bundle_dir: Path, scenario_key: str) -> dict[str, Any]:
    """Re-reads `scenarios/<folder>/scenario.json` from the bundle, matched by the document's OWN
    `scenario_key` field -- never the folder name (`scenario_source.py`'s own convention; the two
    are not guaranteed to match)."""
    root = bundle_dir / "scenarios"
    if not root.is_dir():
        raise _ScenarioDocumentUnavailable(f"no {root} directory in this bundle")
    try:
        children = sorted(root.iterdir())
    except OSError as exc:
        raise _ScenarioDocumentUnavailable(f"cannot list {root}: {exc}") from exc
    for child in children:
        if not child.is_dir():
            continue
        doc_path = child / "scenario.json"
        if not doc_path.is_file():
            continue
        try:
            body = json.loads(doc_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(body, dict) and body.get("scenario_key") == scenario_key:
            instruction = body.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                raise _ScenarioDocumentUnavailable(
                    f"{child.name}/scenario.json has no non-empty instruction"
                )
            return body
    raise _ScenarioDocumentUnavailable(
        f"no scenario.json under {root} carries scenario_key={scenario_key!r}"
    )


# --- deterministic room naming (asserted verbatim by tests/harness/test_call_runner.py). WHY this
# is a PREFIX guarantee, not a full-match one: in managed room_mode, engines/livekit.py::
# _resolve_room_name appends its own `-{invocation_id}-{test_case_id[-12:]}` suffix unless
# `room_name_verbatim` is set (which this runner does not set) -- the scheme below still gives
# every call a unique, deterministic, greppable prefix; only the exact wire-level name is not this
# string verbatim. -------------------------------------------------------------------------------


def _room_name(
    *, job_id: str, attempt_number: int, scenario_key: str, scenario_attempt: int
) -> str:
    return f"harness-{job_id[:8]}-a{attempt_number}-{scenario_key}-s{scenario_attempt}"


def _duration_ms(started_at: datetime, ended_at: datetime) -> int:
    return max(0, int((ended_at - started_at).total_seconds() * 1000))


# --- SimulationSpec construction. Shared with the local lane through simulator_voice; only the
# value lookup is lane-specific. ---------------------------------------------------------------


def _dials_the_person(doc: dict[str, Any]) -> bool:
    """Whether the agent places the call: scenario first, then the environment, then inbound."""
    direction = str(
        doc.get("call_direction") or os.environ.get(CALL_DIRECTION_ALIAS) or "inbound"
    )
    return direction.strip().lower() == "outbound"


def _build_spec(
    *,
    run_id: str,
    room_name: str,
    agent_name: str | None,
    doc: Mapping[str, Any],
    livekit_url: str,
    call_timeout_seconds: float,
    run_seconds: float,
    recordings_root: Path,
    simulator_config: Mapping[str, Any],
    environ: Mapping[str, str],
    connector: str = "livekit",
    provider_target_id: str | None = None,
) -> SimulationSpec:
    """The hosted lane: values come from job config and the bundle's scenario document.

    Lowercase job config wins; provider environment aliases are accepted for compatibility.
    """

    def setting(name: str) -> str:
        return str(simulator_config.get(name.lower()) or environ.get(name) or "")

    simulator = simulator_definition(setting, doc.get("persona"))
    connector = connector.strip().lower()
    provider_agent: simulate.AgentDefinition | None = None
    if connector == "vapi":
        if not provider_target_id:
            raise ValueError("vapi_target_id_unavailable")
        provider_agent = simulate.AgentDefinition(
            name="harness-vapi-target",
            system_prompt=str(
                simulator_config.get("target_system_prompt")
                or "Provider-hosted Vapi target under test."
            ),
            target={
                "provider": "vapi",
                "assistant_id": provider_target_id,
                "api_base_url": str(
                    simulator_config.get("vapi_api_base_url") or "https://api.vapi.ai"
                ),
                "api_key_env": VAPI_API_KEY_ALIAS,
            },
            transport={"kind": "vapi_websocket"},
            provider_evidence={
                "provider": "vapi",
                "call_id_source": "originator_response",
            },
        )
    elif connector == "retell":
        if not provider_target_id:
            raise ValueError("retell_target_id_unavailable")
        provider_agent = simulate.AgentDefinition(
            name="harness-retell-target",
            system_prompt=str(
                simulator_config.get("target_system_prompt")
                or "Provider-hosted Retell target under test."
            ),
            target={
                "provider": "retell",
                "agent_id": provider_target_id,
                "api_url": str(
                    simulator_config.get("retell_api_url")
                    or "https://api.retellai.com/v2/create-web-call"
                ),
                "livekit_url": str(
                    simulator_config.get("retell_livekit_url")
                    or "wss://retell-ai-4ihahnq7.livekit.cloud"
                ),
                "api_key_env": RETELL_API_KEY_ALIAS,
            },
            transport={"kind": "retell_webcall"},
            provider_evidence={
                "provider": "retell",
                "call_id_source": "originator_response",
            },
        )
    # A provider-hosted agent owns termination and can legitimately finish an agent-first call
    # after the fifth message: agent greeting, caller request, agent clarification, caller answer,
    # agent confirmation followed by the provider's end-call tool. Requiring the simulator's
    # sixth acknowledgement after Retell/Vapi has already disconnected misclassifies a complete
    # call as infrastructure failure and prevents the tool trace from being graded. Native
    # LiveKit keeps the stricter six-message floor because our simulator owns that hang-up path.
    min_turn_messages = 5 if connector in {"vapi", "retell"} else 6
    return simulation_spec(
        run_id=run_id,
        room_name=room_name,
        agent_name=agent_name,
        system_prompt=doc["instruction"],
        livekit_url=livekit_url,
        recording_dir=recordings_root / run_id / "recordings",
        scenario=caller_scenario(
            name=str(doc.get("scenario_key") or doc.get("name") or "harness-voice"),
            persona=doc.get("persona"),
            situation=doc["instruction"],
            fixture=doc.get("fixture"),
            tts_provider=simulator.tts.provider,
        ),
        simulator=simulator,
        # An outbound agent dials; the person answers, so the caller opens.
        direction="simulator_first" if _dials_the_person(doc) else "agent_first",
        max_seconds=call_timeout_seconds,
        min_turn_messages=min_turn_messages,
        # Hosted targets can legitimately spend tens of seconds in a provider call or a tool
        # round-trip after the conversation has begun.  The previous 45-second value terminated
        # an otherwise healthy LiveKit call at exactly the watchdog boundary.  Keep a finite
        # liveness guard, but align it with the engine's 60-second conversation-silence backstop.
        agent_first_silence_seconds=60.0,
        run_seconds=run_seconds,
        agent_definition=provider_agent,
    )


# --- evidence collection -----------------------------------------------------------------------


def _find_postgres_endpoint(runtime: EnvironmentRuntime) -> Any | None:
    """Protocol-based lookup, matching `hosted_entrypoint.py::_find_postgres_endpoint`'s own
    already-correct convention -- capability slugs are bundle-author-chosen (`build_endpoints`,
    process_runtime.py:318-339), never a fixed key, so a hardcoded `endpoints["database"]` would
    break for any bundle that names its capability slug differently. Re-implemented locally rather
    than imported: importing from `hosted_entrypoint.py` here would be circular (it imports this
    module's factory)."""
    for endpoint in runtime.endpoints.values():
        if endpoint.protocol == "postgres":
            return endpoint
    return None


def _collect_http_tool_calls(runtime: EnvironmentRuntime) -> tuple[Call, ...]:
    """No guest-side capture surface exists anywhere in this repo for the
    `http_tool` evidence seam. Verified, not assumed: `world/handle.py::HostedWorld.call()`
    raises `WorldUnavailable` unconditionally with a docstring stating the wire format "is not
    pinned anywhere in the contracts yet"; `process_runtime.py`'s own `provision()` signature
    comment says "evidence-seam wiring is out of this phase's scope"; no `TOOLS_API_URL` wiring
    exists in the hosted lane at all (the local lane's `ProvisionedWorld`/`TOOLS_API_URL` mechanism
    lives in `provision.py`/`world/provisioned.py`, out of scope here and inapplicable to the guest
    regardless). Deliberately stopped rather than inventing a capture proxy: a job whose bundle
    declares `evidence_seam: http_tool` reads zero calls every time, which the scheduler's own
    `evidence_missing` retry-once policy turns into the correct, honest outcome -- never a crash,
    never fabricated evidence."""
    del runtime
    return ()


def _clear_tool_trace_calls(dsn: str) -> None:
    """world-handle-interface.md: "setup's tool calls are NOT evidence (the runner clears them
    before the call starts, as the local runner does)" -- the local runner's analog is
    `world.calls = []` right before dialing (`run/simulation.py`). Best-effort: a missing table (no
    producer yet) or any connection error is swallowed, never raised. Clearing
    is housekeeping, not a correctness requirement, while nothing writes this table yet; once a
    real producer lands this stops being a no-op automatically."""
    try:
        import psycopg

        with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as connection:
            connection.execute(f'DELETE FROM "{_TOOL_TRACE_TABLE}"')  # noqa: S608 - fixed identifier, no interpolated user input
    except Exception as exc:  # noqa: BLE001 - best-effort housekeeping only, never a call-blocking failure
        # WHY: never log exc_info / str(exc) here -- a psycopg connection failure embeds the raw
        # DSN (including the world DB password) in its own exception message; only the exception
        # TYPE is safe for a local log line.
        logger.debug(
            "tool_trace clear skipped (table likely absent): %s", type(exc).__name__
        )


def _collect_tool_trace_calls(runtime: EnvironmentRuntime) -> tuple[Call, ...]:
    """`_alk_tool_trace`'s name and column shape are an isolated local
    convention -- unpinned by any contract (the only harness-reserved table anywhere in this
    repo is `_alk_conformance`, unrelated), no producer exists yet. Isolated in this one function
    (+ `_clear_tool_trace_calls`) so a real producer's disagreement on the name/shape is a one-line
    change. Any failure (missing table, connection refused, malformed row) degrades to `()` --
    never a crash, never fabricated evidence, matching `_collect_http_tool_calls`'s stopped
    behavior above."""
    endpoint = _find_postgres_endpoint(runtime)
    if endpoint is None:
        return ()
    try:
        import psycopg

        with psycopg.connect(
            endpoint.address,
            autocommit=True,
            connect_timeout=5,
            options="-c default_transaction_read_only=on",
        ) as connection:
            cursor = connection.execute(
                f'SELECT name, arguments, result, ok, error, at FROM "{_TOOL_TRACE_TABLE}" '  # noqa: S608
                "ORDER BY at ASC"
            )
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description or []]
    except Exception as exc:  # noqa: BLE001 - missing table / connection failure -> no evidence, not a crash
        # WHY: same DSN-in-exception-message risk as `_clear_tool_trace_calls` above -- log only
        # the exception TYPE, never exc_info/str(exc), which can carry the world DB password.
        logger.debug(
            "tool_trace read failed; treating as no evidence: %s", type(exc).__name__
        )
        return ()

    calls: list[Call] = []
    for row in rows:
        record = dict(zip(columns, row, strict=True))
        name = record.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = record.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        ok = bool(record.get("ok", True))
        raw_result = record.get("result")
        if isinstance(raw_result, str):
            result: Any = _truncate(raw_result)
        else:
            # Already parsed JSON (dict/list/etc, psycopg's own jsonb decoding) -- per
            # world-handle-interface.md, only the STRING form is truncated at 2000 chars.
            result = raw_result
        error = _truncate(str(record.get("error") or ""))
        raw_at = record.get("at")
        at = float(raw_at) if isinstance(raw_at, (int, float)) else 0.0
        calls.append(
            Call(
                name=name,
                arguments=arguments,
                result=result,
                ok=ok,
                error=error,
                refused=not ok,
                at=at,
            )
        )
    return tuple(calls)


def _tool_trace_file(runtime: EnvironmentRuntime) -> Path | None:
    raw = runtime.metadata.get("tool_trace_path")
    return Path(raw) if isinstance(raw, str) and raw.strip() else None


def _clear_file_tool_calls(runtime: EnvironmentRuntime) -> None:
    path = _tool_trace_file(runtime)
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.debug(
            "file tool_trace clear failed; continuing without blocking the call"
        )


def _collect_file_tool_calls(runtime: EnvironmentRuntime) -> tuple[Call, ...]:
    path = _tool_trace_file(runtime)
    if path is None or not path.is_file():
        return ()
    calls: list[Call] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        name = record.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = record.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {"raw": arguments}
        if not isinstance(arguments, dict):
            arguments = {}
        is_error = bool(record.get("is_error", False))
        output = record.get("output")
        calls.append(
            Call(
                name=name,
                arguments=arguments,
                result=None if is_error else output,
                ok=not is_error,
                error=str(output) if is_error and output is not None else None,
            )
        )
    return tuple(calls)


def _collect_provider_tool_calls(case: Any) -> tuple[Call, ...]:
    """Translate provider-reported tool evidence into scheduler calls.

    LiveKit's legacy report conversion stores per-case evidence in
    ``result.metadata.evidence``; canonical reports may populate
    ``case.evidence`` directly. Accept both shapes so hosted execution is not
    coupled to the report representation.
    """
    sources: list[Any] = list(getattr(case, "evidence", None) or [])
    result = getattr(case, "result", None)
    result_metadata = getattr(result, "metadata", None)
    if isinstance(result_metadata, Mapping):
        embedded = result_metadata.get("evidence")
        if isinstance(embedded, list):
            sources.extend(embedded)

    calls: list[Call] = []
    for source in sources:
        if hasattr(source, "model_dump"):
            source = source.model_dump(mode="json", exclude_none=True)
        if not isinstance(source, Mapping):
            continue
        metadata = source.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        raw_calls = metadata.get("tool_calls")
        if not isinstance(raw_calls, list):
            continue
        for raw in raw_calls:
            if not isinstance(raw, Mapping):
                continue
            name = raw.get("name")
            if not isinstance(name, str) or not name:
                continue
            arguments: Any = raw.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    arguments = {"raw": arguments}
            if not isinstance(arguments, dict):
                arguments = {}
            ok = bool(raw.get("ok", True))
            raw_at = raw.get("at")
            at = float(raw_at) if isinstance(raw_at, (int, float)) else 0.0
            calls.append(
                Call(
                    name=name,
                    arguments=arguments,
                    result=raw.get("result") if ok else None,
                    ok=ok,
                    error=str(raw.get("error") or ""),
                    refused=not ok,
                    at=at,
                )
            )
    return tuple(calls)


def _truncate(value: str, *, limit: int = _RESULT_TRUNCATE_CHARS) -> str:
    return value if len(value) <= limit else value[:limit]


def _materialize_vertex_adc(
    secret_values: Mapping[str, str],
    work_directory: Path,
    environ: dict[str, str],
) -> Path | None:
    """Materialize caller-lane Vertex credentials for Google ADC.

    API-key auth needs no file. For Vertex, GOOGLE_APPLICATION_CREDENTIALS_JSON is resolved from
    the platform vault and written mode-0600 under the job work directory; the sandbox is
    ephemeral and the file is removed when the guest exits/deletes.
    """
    raw = secret_values.get(GOOGLE_APPLICATION_CREDENTIALS_JSON_ALIAS)
    if not raw or environ.get(GOOGLE_APPLICATION_CREDENTIALS_ALIAS):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CallAborted(
            "voice_capability_unavailable: GOOGLE_APPLICATION_CREDENTIALS_JSON is invalid"
        ) from exc
    if not isinstance(parsed, dict):
        raise CallAborted(
            "voice_capability_unavailable: GOOGLE_APPLICATION_CREDENTIALS_JSON must be an object"
        )
    credential_dir = work_directory / ".caller-credentials"
    credential_dir.mkdir(parents=True, exist_ok=True)
    fd, path_text = tempfile.mkstemp(
        prefix="google-", suffix=".json", dir=credential_dir
    )
    path = Path(path_text)
    try:
        os.write(fd, raw.encode("utf-8"))
    finally:
        os.close(fd)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    environ[GOOGLE_APPLICATION_CREDENTIALS_ALIAS] = str(path)
    return path


# --- the runner ----------------------------------------------------------------------------


class CallRunnerImpl:
    """Satisfies `hosted_scheduler.CallRunner`. See the module docstring for the three
    sub-systems this class implements."""

    def __init__(
        self,
        adapter: ArtifactUploader,
        context: CallRunnerContext,
        *,
        place_call: PlaceCall | None = None,
        environ: dict[str, str] | None = None,
    ) -> None:
        self._adapter = adapter
        self._context = context
        self._place_call = place_call or _default_place_call
        simulator_secret_values = _canonical_simulator_secrets(
            context.simulator_provider_secret_values
        )
        # Local SDK runs remain BYOK and historically carry simulator keys in the one local
        # target map. Hosted runs deliberately do not fall back: their simulator credentials must
        # come from platform configuration and must not be confused with customer-agent keys.
        if context.job.execution is ExecutionMode.LOCAL:
            for alias in (
                LIVEKIT_URL_ALIAS,
                LIVEKIT_API_KEY_ALIAS,
                LIVEKIT_API_SECRET_ALIAS,
                DEEPGRAM_API_KEY_ALIAS,
                CARTESIA_API_KEY_ALIAS,
                GEMINI_API_KEY_ALIAS,
                GOOGLE_API_KEY_ALIAS,
                GOOGLE_APPLICATION_CREDENTIALS_JSON_ALIAS,
                GOOGLE_CLOUD_PROJECT_ALIAS,
                GOOGLE_CLOUD_LOCATION_ALIAS,
                GOOGLE_GENAI_USE_VERTEXAI_ALIAS,
                OPENAI_API_KEY_ALIAS,
                SIMULATOR_LLM_PROVIDER_ALIAS,
                SIMULATOR_LLM_MODEL_ALIAS,
                SIMULATOR_STT_PROVIDER_ALIAS,
                SIMULATOR_STT_MODEL_ALIAS,
                SIMULATOR_TTS_PROVIDER_ALIAS,
                SIMULATOR_TTS_MODEL_ALIAS,
            ):
                if alias not in simulator_secret_values:
                    value = context.target_provider_secret_values.get(alias)
                    if value:
                        simulator_secret_values[alias] = value
        # WHY: the underlying LiveKit engine reads these directly via `os.environ.get(...)` deep
        # inside `engines/livekit.py` / `livekit_models.py` -- they are NOT `SimulationSpec`
        # fields, so there is no other way to hand them over. Exported ONCE here, at construction,
        # not per-call: the values are job-level (the same secret for every scenario/attempt on
        # this job) and W>1 means each world's CallRunner.run() executes inside this SAME guest
        # process but against a per-world sandboxed agent process reached over the network; no
        # other in-process worker races this job-level environment.
        target_environ = os.environ if environ is None else environ
        connector = _resolve_connector(
            context.job, context.target_provider_secret_values
        )
        target_aliases = [VAPI_API_KEY_ALIAS, RETELL_API_KEY_ALIAS]
        if connector == "livekit":
            target_aliases.extend(
                [LIVEKIT_API_KEY_ALIAS, LIVEKIT_API_SECRET_ALIAS, LIVEKIT_URL_ALIAS]
            )
        for alias in target_aliases:
            value = context.target_provider_secret_values.get(alias)
            if value:
                target_environ[alias] = value
        for alias in (
            LIVEKIT_URL_ALIAS,
            LIVEKIT_API_KEY_ALIAS,
            LIVEKIT_API_SECRET_ALIAS,
            DEEPGRAM_API_KEY_ALIAS,
            CARTESIA_API_KEY_ALIAS,
            GEMINI_API_KEY_ALIAS,
            GOOGLE_API_KEY_ALIAS,
            GOOGLE_APPLICATION_CREDENTIALS_JSON_ALIAS,
            GOOGLE_CLOUD_PROJECT_ALIAS,
            GOOGLE_CLOUD_LOCATION_ALIAS,
            GOOGLE_GENAI_USE_VERTEXAI_ALIAS,
            OPENAI_API_KEY_ALIAS,
            SIMULATOR_LLM_PROVIDER_ALIAS,
            SIMULATOR_LLM_MODEL_ALIAS,
            SIMULATOR_STT_PROVIDER_ALIAS,
            SIMULATOR_STT_MODEL_ALIAS,
            SIMULATOR_TTS_PROVIDER_ALIAS,
            SIMULATOR_TTS_MODEL_ALIAS,
            BACKGROUND_NOISE_ALIAS,
            BACKGROUND_NOISE_CATALOG_ALIAS,
            BACKGROUND_NOISE_VOLUME_ALIAS,
        ):
            value = simulator_secret_values.get(alias)
            if value:
                # Platform-owned simulator credentials already present in the hosted control
                # process win.  Target credentials are retained only as the local-SDK fallback.
                target_environ.setdefault(alias, value)
        self._environ = target_environ
        self._adc_path = _materialize_vertex_adc(
            simulator_secret_values,
            context.work_directory,
            target_environ,
        )
        atexit.register(self._cleanup_credentials)
        self._livekit_url = str(
            context.job.agent.config.get(LIVEKIT_URL_CONFIG_KEY)
            or (
                context.target_provider_secret_values.get(LIVEKIT_URL_ALIAS)
                if connector == "livekit"
                else simulator_secret_values.get(LIVEKIT_URL_ALIAS)
            )
            or ""
        )
        self._missing_config = _check_config(
            context.job,
            context.target_provider_secret_values,
            simulator_secret_values,
        )
        self._scenario_attempt_counts: dict[str, int] = {}

    def _cleanup_credentials(self) -> None:
        if self._adc_path is None:
            return
        try:
            self._adc_path.unlink(missing_ok=True)
        except OSError:
            pass
        if self._environ.get(GOOGLE_APPLICATION_CREDENTIALS_ALIAS) == str(
            self._adc_path
        ):
            self._environ.pop(GOOGLE_APPLICATION_CREDENTIALS_ALIAS, None)
        self._adc_path = None

    async def run(
        self,
        scenario: HostedScenario,
        runtime: EnvironmentRuntime,
        *,
        world: Any | None = None,
    ) -> CallOutcome:
        del world  # Voice tools cross the declared evidence seam; they are not response-carried.
        if self._missing_config is not None:
            # Pre-dial: dialing never starts, so no partial -- and never `WorldUnavailable` (that
            # code is reserved by the contract for a world-level capability mismatch, not a
            # job-level voice config gap).
            raise CallAborted(self._missing_config.message())

        connector = _resolve_connector(
            self._context.job, self._context.target_provider_secret_values
        )
        agent_name = _dispatch_agent_name(runtime) if connector == "livekit" else None
        if connector == "livekit" and agent_name is None:
            raise CallAborted(
                "voice_dispatch_identity_unavailable: runtime.metadata['livekit_agent_name'] is "
                f"not set for world {runtime.world_index}"
            )

        try:
            doc = _read_scenario_document(
                self._context.bundle_dir, scenario.scenario_key
            )
        except _ScenarioDocumentUnavailable as exc:
            raise CallAborted(f"voice_scenario_document_unavailable: {exc}") from exc

        scenario_attempt = (
            self._scenario_attempt_counts.get(scenario.scenario_key, 0) + 1
        )
        self._scenario_attempt_counts[scenario.scenario_key] = scenario_attempt
        room_name = _room_name(
            job_id=self._context.job.job_id,
            attempt_number=self._context.attempt_number,
            scenario_key=scenario.scenario_key,
            scenario_attempt=scenario_attempt,
        )

        raw_timeout = self._context.job.agent.config.get(CALL_TIMEOUT_CONFIG_KEY)
        call_timeout_seconds = (
            float(raw_timeout)
            if isinstance(raw_timeout, (int, float))
            else _DEFAULT_CALL_TIMEOUT_SECONDS
        )
        run_seconds = (
            call_timeout_seconds
            + CONNECT_TIMEOUT_SECONDS
            + READINESS_TIMEOUT_SECONDS
            + CLEANUP_TIMEOUT_SECONDS
            + _RUN_SECONDS_PAD_SECONDS
        )

        # The engine reads this from the environment at call time, so it is set per
        # scenario and cleared otherwise rather than leaking into the next call.
        noise = scenario_source(
            doc.get("background_noise"),
            doc.get("fixture"),
            seed=str(doc.get("name") or ""),
        )
        if noise:
            self._environ["HARNESS_BACKGROUND_NOISE"] = noise
        else:
            self._environ.pop("HARNESS_BACKGROUND_NOISE", None)

        # Read the same way and for the same reason as the noise source above: the simulator's
        # instructions are built deep inside simulator_definition, which sees the environment and
        # not this scenario. Set per scenario and cleared otherwise so one outbound scenario cannot
        # frame the next inbound one.
        # A scenario that names its own direction wins. Otherwise the contract's, which the
        # understand stage read off the agent's own instructions and `hosted_entrypoint` puts here
        # for this process. Not an operator setting: whether an agent places calls or answers them
        # is a fact about the agent, so there is nothing for a run to choose.
        direction = (
            str(
                doc.get("call_direction")
                or os.environ.get(CALL_DIRECTION_ALIAS)
                or "inbound"
            )
            .strip()
            .lower()
        )
        if direction == "outbound":
            self._environ["HARNESS_CALL_DIRECTION"] = direction
            awareness = str(doc.get("caller_awareness") or "").strip().lower()
            if awareness:
                self._environ["HARNESS_CALLER_AWARENESS"] = awareness
            else:
                self._environ.pop("HARNESS_CALLER_AWARENESS", None)
            # Cleared otherwise, so one voicemail scenario cannot silence the next caller.
            if (
                voicemail_enabled()
                and str(doc.get("answered_by") or "").strip().lower() == "voicemail"
            ):
                self._environ["HARNESS_ANSWERED_BY"] = "voicemail"
                # Which kind of mailbox, which decides the greeting and whether a tone follows it.
                style = str(doc.get("voicemail_style") or "").strip().lower()
                if style:
                    self._environ["HARNESS_VOICEMAIL_STYLE"] = style
                else:
                    self._environ.pop("HARNESS_VOICEMAIL_STYLE", None)
                # A recorded greeting where the catalogue has one for this style AND language. It
                # replaces the spoken greeting rather than joining it.
                languages = doc.get("languages") or []
                chosen = clip_for(
                    style or DEFAULT_VOICEMAIL_STYLE,
                    str(languages[0]) if languages else "",
                )
                if chosen:
                    self._environ[VOICEMAIL_CLIP_ALIAS] = chosen["source"]
                    self._environ[VOICEMAIL_CLIP_TONE_ALIAS] = (
                        "1" if chosen["has_tone"] else "0"
                    )
                    if chosen.get("transcript"):
                        self._environ[VOICEMAIL_CLIP_TEXT_ALIAS] = chosen["transcript"]
                    else:
                        self._environ.pop(VOICEMAIL_CLIP_TEXT_ALIAS, None)
                else:
                    self._environ.pop(VOICEMAIL_CLIP_ALIAS, None)
                    self._environ.pop(VOICEMAIL_CLIP_TONE_ALIAS, None)
                    self._environ.pop(VOICEMAIL_CLIP_TEXT_ALIAS, None)
            else:
                self._environ.pop("HARNESS_ANSWERED_BY", None)
                self._environ.pop("HARNESS_VOICEMAIL_STYLE", None)
                self._environ.pop(VOICEMAIL_CLIP_ALIAS, None)
                self._environ.pop(VOICEMAIL_CLIP_TONE_ALIAS, None)
                self._environ.pop(VOICEMAIL_CLIP_TEXT_ALIAS, None)
        else:
            self._environ.pop("HARNESS_CALL_DIRECTION", None)
            self._environ.pop("HARNESS_CALLER_AWARENESS", None)
            self._environ.pop("HARNESS_ANSWERED_BY", None)
            self._environ.pop("HARNESS_VOICEMAIL_STYLE", None)
            self._environ.pop(VOICEMAIL_CLIP_ALIAS, None)
            self._environ.pop(VOICEMAIL_CLIP_TONE_ALIAS, None)
            self._environ.pop(VOICEMAIL_CLIP_TEXT_ALIAS, None)

        provider_target_key = {"vapi": "assistant_id", "retell": "agent_id"}.get(
            connector
        )
        provider_target_id: str | None = None
        if provider_target_key and self._context.job.agent.mode in {
            None,
            ProviderExecutionMode.CONNECT_ONLY,
        }:
            provider_target_id = str(
                self._context.job.agent.config.get(provider_target_key) or ""
            ).strip()
        if provider_target_key and self._context.job.agent.mode not in {
            None,
            ProviderExecutionMode.CONNECT_ONLY,
        }:
            dynamic_target = runtime.metadata.get("provider_target_id")
            provider_target_id = (
                dynamic_target.strip()
                if isinstance(dynamic_target, str) and dynamic_target.strip()
                else None
            )

        spec = _build_spec(
            run_id=new_run_id(),
            room_name=room_name,
            connector=connector,
            agent_name=agent_name,
            provider_target_id=provider_target_id,
            doc=doc,
            simulator_config=self._context.job.agent.config,
            environ=self._environ,
            livekit_url=self._livekit_url,
            call_timeout_seconds=call_timeout_seconds,
            run_seconds=run_seconds,
            recordings_root=self._context.work_directory / "voice-calls",
        )

        if self._context.evidence_seam is EvidenceSeam.TOOL_TRACE:
            _clear_file_tool_calls(runtime)
            endpoint = _find_postgres_endpoint(runtime)
            if endpoint is not None:
                _clear_tool_trace_calls(endpoint.address)

        started_at = datetime.now(timezone.utc)
        outer_timeout = run_seconds + _OUTER_WAIT_FOR_PAD_SECONDS
        try:
            report = await asyncio.wait_for(
                self._place_call(spec), timeout=outer_timeout
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError as exc:
            raise CallAborted(
                "voice_call_runner_timeout: place_call exceeded its outer budget "
                f"({outer_timeout:.0f}s)",
                partial=self._timing_only_outcome(started_at),
            ) from exc
        except Exception as exc:  # noqa: BLE001 - post-dial machinery failure, never let it escape raw
            raise CallAborted(
                f"voice_call_runner_crashed: {type(exc).__name__}: {exc}",
                partial=self._timing_only_outcome(started_at),
            ) from exc

        try:
            return await self._translate_report(
                report,
                runtime=runtime,
                scenario_key=scenario.scenario_key,
                started_at=started_at,
            )
        except (CallAborted, WorldUnavailable):
            # `_translate_report`'s own typed control-flow (non-completed status, no test case,
            # agent-never-joined) -- never re-wrap an intentional abort.
            raise
        except Exception as exc:  # noqa: BLE001 - a transcript/recording read or upload surprise
            # must never lose the timing this call already measured (the receipt's `call` field
            # must not be null once the call has genuinely started) by escaping run() raw.
            raise CallAborted(
                f"voice_call_translate_crashed: {type(exc).__name__}: {exc}",
                partial=self._timing_only_outcome(started_at),
            ) from exc

    def _timing_only_outcome(self, started_at: datetime) -> CallOutcome:
        ended_at = datetime.now(timezone.utc)
        return CallOutcome(
            calls=(),
            turns=0,
            started_at=format_rfc3339_millis(started_at),
            ended_at=format_rfc3339_millis(ended_at),
            duration_ms=_duration_ms(started_at, ended_at),
        )

    async def _translate_report(
        self,
        report: SimulationReport,
        *,
        runtime: EnvironmentRuntime,
        scenario_key: str,
        started_at: datetime,
    ) -> CallOutcome:
        case = report.test_cases[0] if report.test_cases else None
        case_started_at = (
            case.started_at
            if case is not None and case.started_at is not None
            else started_at
        )
        ended_at = (
            case.ended_at
            if case is not None and case.ended_at is not None
            else datetime.now(timezone.utc)
        )
        turns = (
            len(case.result.messages)
            if case is not None and case.result is not None
            else 0
        )

        transcript_artifact: str | None = None
        recording_artifacts: list[str] = []
        # Evidence belongs to the call attempt, not only to successful calls.  Collect it before
        # interpreting the simulator status so a timeout/agent failure still carries the exact
        # tool activity in its partial receipt.  Previously the early CallAborted below discarded
        # every tool call from failed calls, making a real upstream tool error indistinguishable
        # from a proxy/transport failure.
        calls = self._collect_calls(runtime) if case is not None else ()
        # Provider-hosted agents execute tools outside the guest process, so their
        # authoritative call evidence is returned by Vapi/Retell after the call.
        # Preserve provider-native controls such as ``end_call`` because scenario
        # checks may verify termination ordering. Fall back to that observed stream
        # when the submitted backend exposes no local trace seam; never infer calls
        # from transcript prose.
        if not calls and case is not None:
            calls = _collect_provider_tool_calls(case)
        if calls:
            tool_trace = "\n".join(
                json.dumps(
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "result": call.result,
                        "ok": call.ok,
                        "error": call.error,
                        "refused": call.refused,
                        "at": call.at,
                    },
                    sort_keys=True,
                    default=str,
                )
                for call in calls
            ).encode("utf-8")
            await self._adapter.upload_artifact(
                tool_trace,
                kind=ArtifactKind.TOOL_TRACE,
                scenario_key=scenario_key,
            )
        if case is not None and case.result is not None:
            result = case.result
            if result.transcript:
                transcript_payload = json.dumps(
                    {
                        "schema_version": "futureagi.call-transcript.v1",
                        "transcript": result.transcript,
                        "messages": result.messages,
                    },
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
                transcript_artifact = await self._adapter.upload_artifact(
                    transcript_payload,
                    kind=ArtifactKind.TRANSCRIPT,
                    scenario_key=scenario_key,
                )
            for path_str, kind in (
                (result.audio_combined_path, ArtifactKind.RECORDING_COMBINED),
                (result.audio_stereo_path, ArtifactKind.RECORDING_STEREO),
                (result.audio_input_path, ArtifactKind.RECORDING_CUSTOMER),
                (result.audio_output_path, ArtifactKind.RECORDING_ASSISTANT),
            ):
                if not path_str:
                    continue
                path = Path(path_str)
                if not path.is_file():
                    continue
                artifact_id = await self._adapter.upload_artifact(
                    path.read_bytes(),
                    kind=kind,
                    scenario_key=scenario_key,
                )
                if artifact_id is not None:
                    recording_artifacts.append(artifact_id)

        base = CallOutcome(
            calls=calls,
            turns=turns,
            started_at=format_rfc3339_millis(case_started_at),
            ended_at=format_rfc3339_millis(ended_at),
            duration_ms=_duration_ms(case_started_at, ended_at),
            transcript_artifact=transcript_artifact,
            recording_artifacts=tuple(recording_artifacts),
        )

        if case is None:
            raise CallAborted(
                "voice_call_no_test_case: SimulationReport carried no test case",
                partial=base,
            )

        if case.status is TestCaseStatus.AGENT_UNAVAILABLE:
            # world-handle-interface.md: "the agent never joined" is a WORLD failure, not a
            # scenario one -- the agent is part of the world, so the scheduler retires it and
            # retries elsewhere. Verified against the engine's own source (engines/livekit.py):
            # this status fires ONLY on a readiness-stage timeout with a session already started
            # but no target dispatched -- exactly "dispatch fails, agent never joins," never a
            # mid-call condition.
            reason = (
                case.failure.message
                if case.failure is not None
                else "agent_unavailable"
            )
            raise WorldUnavailable(f"target agent never joined the room: {reason}")

        # A genuinely silent agent-first call (agent joined, zero conversational turns) reaches
        # the real engine (engines/livekit.py::_conversation_outcome) as FAILED with code
        # "no_conversation" or "conversation_silence_timeout" and zero messages -- never as a
        # COMPLETED case with zero turns (COMPLETED requires >= min_turn_messages AND role
        # alternation, so the engine cannot produce that shape). Scoped to zero turns only: a
        # short-but-nonzero conversation on either code still failed the completion bar for a real
        # reason and must stay a CallAborted below.
        is_silent_agent = (
            case.status is TestCaseStatus.FAILED
            and turns == 0
            and case.failure is not None
            and case.failure.code in _SILENT_AGENT_FAILURE_CODES
        )

        # An intake agent may ask thirty to fifty questions, so a deadline is an ordinary outcome.
        ran_out_of_time = (
            case.status is TestCaseStatus.TIMED_OUT
            and turns >= _GRADEABLE_AFTER_TIMEOUT_TURNS
        )

        if (
            case.status is not TestCaseStatus.COMPLETED
            and not is_silent_agent
            and not ran_out_of_time
        ):
            reason = (
                case.failure.message if case.failure is not None else case.status.value
            )
            # C3 §4.5 step 2 (the one narrow call_runner change): the engine's dispatch-ack ladder
            # surfaces +60s exhaustion via a STRUCTURED marker (`failure.code`), never a substring
            # of `failure.message`. Pass it through on `CallAborted.marker` so the scheduler's
            # code-selection branch emits the `voice_dispatch_unacknowledged` receipt code instead
            # of the generic `call_failed`. No ladder logic lives here.
            marker = (
                case.failure.code
                if case.failure is not None
                and case.failure.code == _VOICE_DISPATCH_UNACKNOWLEDGED
                else None
            )
            raise CallAborted(
                f"voice_call_not_completed: {case.status.value}: {reason}",
                partial=base,
                marker=marker,
            )

        # Never fabricate calls for a call that produced no conversation -- the scheduler's own
        # coverage guarantee turns an empty `calls` tuple into evidence_missing/simulator
        # regardless of turns (hosted_scheduler.py's own unconditioned-on-turns rule).
        calls = () if is_silent_agent else base.calls
        return CallOutcome(
            calls=calls,
            turns=base.turns,
            started_at=base.started_at,
            ended_at=base.ended_at,
            duration_ms=base.duration_ms,
            transcript_artifact=base.transcript_artifact,
            recording_artifacts=base.recording_artifacts,
        )

    def _collect_calls(self, runtime: EnvironmentRuntime) -> tuple[Call, ...]:
        seam = self._context.evidence_seam
        file_calls = _collect_file_tool_calls(runtime)
        if file_calls:
            return file_calls
        if seam is EvidenceSeam.HTTP_TOOL:
            return _collect_http_tool_calls(runtime)
        if seam is EvidenceSeam.TOOL_TRACE:
            return _collect_tool_trace_calls(runtime)
        # Unrecognized/None (should not happen for a `kind: process` bundle past preflight --
        # bundle_v2.py requires `evidence_seam` whenever `kind is PROCESS` -- but degrading rather
        # than crashing keeps this on the scheduler's own evidence_missing path, never a raw
        # exception).
        return ()
