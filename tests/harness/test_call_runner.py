"""`call_runner.py` against in-memory fakes and scripted `SimulationReport`s -- no real LiveKit
call, no real postgres, no monkeypatching LiveKit internals. Matches this repo's own convention
(`test_hosted_entrypoint.py`'s docstring): `asyncio.run` drives every `async def` seam directly.

The test seam is `CallRunnerImpl`'s own boundary: the injectable `place_call(spec) ->
SimulationReport` callable named by the brief. Real production code (`_default_place_call`)
builds the same `SimulationSpec`/`SimulationRunner().run` pair; these tests never construct or
touch a real `SimulationRunner`, `rtc.Room`, or `AgentSession`.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from fi.alk.harness import call_runner as cr
from fi.alk.harness.bundle_v2 import EvidenceSeam
from fi.alk.harness.hosted_scheduler import CallAborted, CallOutcome
from fi.alk.harness.job import (
    AgentConnection,
    ExecutionMode,
    HarnessJob,
    ProviderExecutionMode,
    RepositorySource,
    SourceKind,
    SourceVisibility,
)
from fi.alk.harness.process_runtime import (
    EnvironmentRuntime,
    RuntimeEndpoint,
    RuntimeState,
)
from fi.alk.harness.world.errors import WorldUnavailable
from fi.alk.harness.world.runtime import Call
from fi.simulate.artifacts import ArtifactManifest
from fi.simulate.runtime.failures import FailureStage, SimulationFailure
from fi.simulate.runtime.report import SimulationReport, SimulationTestCaseResult
from fi.simulate.runtime.run import RunStatus
from fi.simulate.runtime.run import TestCaseStatus as CaseStatus
from fi.simulate.runtime.spec import RuntimeIsolation, RuntimeRequirements
from fi.simulate.simulation.models import Persona as SimPersona
from fi.simulate.simulation.models import TestCaseResult as SimTestCaseResult

LIVEKIT_API_KEY = "LIVEKIT_API_KEY"
LIVEKIT_API_SECRET = "LIVEKIT_API_SECRET"
DEEPGRAM_API_KEY = "DEEPGRAM_API_KEY"
GEMINI_API_KEY = "GEMINI_API_KEY"
RETELL_API_KEY = "RETELL_API_KEY"

_ALL_SECRETS = {
    LIVEKIT_API_KEY: "lk-key",
    LIVEKIT_API_SECRET: "lk-secret",
    DEEPGRAM_API_KEY: "dg-key",
    GEMINI_API_KEY: "gm-key",
}
_ALL_CONFIG = {cr.LIVEKIT_URL_CONFIG_KEY: "wss://example.livekit.cloud"}


# =================================================================================================
# Fixtures -- self-contained (this file touches nothing outside itself + call_runner.py).
# =================================================================================================


def _job(
    *,
    connector: str = "livekit",
    config: dict[str, Any] | None = None,
    execution: ExecutionMode = ExecutionMode.HOSTED,
    mode: ProviderExecutionMode | None = None,
) -> HarnessJob:
    return HarnessJob(
        job_id="job-abcdef12-xyz",
        run_id="run-1",
        execution=execution,
        source=(
            RepositorySource(
                kind=SourceKind.LOCAL_REPOSITORY,
                local_path="/tmp/agent",
                visibility=SourceVisibility.PRIVATE,
            )
            if execution is ExecutionMode.LOCAL
            else RepositorySource(
                kind=SourceKind.GITHUB,
                repository="org/repo",
                visibility=SourceVisibility.PUBLIC,
                commit_sha="a" * 40,
            )
        ),
        agent=AgentConnection(connector=connector, mode=mode, config=config or {}),
        scenario_count=1,
        runtime=RuntimeRequirements(
            isolation=RuntimeIsolation.DEDICATED_VM,
            cpu_units=1,
        ),
        seed=1,
    )


def _runtime(
    *,
    metadata: dict[str, Any] | None = None,
    endpoints: dict[str, RuntimeEndpoint] | None = None,
    world_index: int = 0,
) -> EnvironmentRuntime:
    return EnvironmentRuntime(
        runtime_id="rt-1",
        world_index=world_index,
        bundle_digest="sha256:" + "0" * 64,
        state=RuntimeState.READY,
        endpoints=endpoints or {},
        metadata=metadata or {},
    )


def _postgres_endpoint(
    *, address: str = "postgresql://harness:pw@localhost:15001/w0"
) -> RuntimeEndpoint:
    return RuntimeEndpoint(
        capability="database",
        protocol="postgres",
        address=address,
        configuration_name="DATABASE_URL",
    )


def test_file_tool_trace_is_collected_from_runtime_metadata(tmp_path: Path) -> None:
    trace = tmp_path / "agent-tool-calls.jsonl"
    trace.write_text(
        json.dumps(
            {
                "name": "book_ride",
                "arguments": '{"destination":"airport"}',
                "output": {"booking_id": "ride-1"},
                "is_error": False,
            }
        )
        + "\n"
        + json.dumps(
            {
                "name": "charge_card",
                "arguments": {},
                "output": "declined",
                "is_error": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls = cr._collect_file_tool_calls(
        _runtime(metadata={"tool_trace_path": str(trace)})
    )
    assert [call.name for call in calls] == ["book_ride", "charge_card"]
    assert calls[0].arguments == {"destination": "airport"}
    assert calls[0].ok is True
    assert calls[1].ok is False
    assert calls[1].error == "declined"


def test_file_tool_trace_clear_removes_previous_attempt(tmp_path: Path) -> None:
    trace = tmp_path / "agent-tool-calls.jsonl"
    trace.write_text("stale\n", encoding="utf-8")
    cr._clear_file_tool_calls(_runtime(metadata={"tool_trace_path": str(trace)}))
    assert not trace.exists()


def test_provider_reported_tool_call_is_used_when_guest_trace_is_empty(
    tmp_path: Path,
) -> None:
    _job_obj, context = _context(
        tmp_path=tmp_path,
        connector="retell",
        mode=ProviderExecutionMode.PROVIDER_IMPORT,
        config={"agent_id": "source-retell-agent"},
        secrets={RETELL_API_KEY: "retell-key"},
    )
    _write_scenario_doc(context.bundle_dir, scenario_key="k1")

    async def place_call(spec):
        report = _report(
            transcript="user: window\nagent: saved",
            messages=[
                {"role": "user", "content": "window"},
                {"role": "assistant", "content": "saved"},
            ],
        )
        assert report.test_cases[0].result is not None
        report.test_cases[0].result.metadata["evidence"] = [
            {
                "adapter": "retell",
                "metadata": {
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "name": "record_preference",
                            "arguments": {"preference": "window"},
                            "result": {"recorded": True},
                            "ok": True,
                            "at": 4.25,
                        }
                    ]
                },
            }
        ]
        return report

    adapter = FakeAdapter()
    outcome = _run(
        cr.CallRunnerImpl(adapter, context, place_call=place_call),
        _FakeScenario("k1"),
        _runtime(metadata={"provider_target_id": "cloned-retell-agent"}),
    )

    assert len(outcome.calls) == 1
    assert outcome.calls[0].name == "record_preference"
    assert outcome.calls[0].arguments == {"preference": "window"}
    assert outcome.calls[0].result == {"recorded": True}
    assert any(kind is cr.ArtifactKind.TOOL_TRACE for kind, _, _ in adapter.uploads)


def _context(
    *,
    tmp_path: Path,
    connector: str = "livekit",
    config: dict[str, Any] | None = None,
    secrets: dict[str, str] | None = None,
    simulator_secrets: dict[str, str] | None = None,
    execution: ExecutionMode = ExecutionMode.HOSTED,
    evidence_seam: EvidenceSeam | None = EvidenceSeam.HTTP_TOOL,
    attempt_number: int = 1,
    mode: ProviderExecutionMode | None = None,
) -> tuple[HarnessJob, cr.CallRunnerContext]:
    job = _job(
        connector=connector,
        config=config if config is not None else dict(_ALL_CONFIG),
        execution=execution,
        mode=mode,
    )
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    context = cr.CallRunnerContext(
        job=job,
        bundle_dir=bundle_dir,
        work_directory=tmp_path / "work",
        evidence_seam=evidence_seam,
        target_provider_secret_values=secrets
        if secrets is not None
        else dict(_ALL_SECRETS),
        attempt_number=attempt_number,
        simulator_provider_secret_values=(
            simulator_secrets
            if simulator_secrets is not None
            else {
                "SIMULATOR_LIVEKIT_URL": "wss://platform-livekit.example",
                "SIMULATOR_LIVEKIT_API_KEY": "platform-livekit-key",
                "SIMULATOR_LIVEKIT_API_SECRET": "platform-livekit-secret",
                "SIMULATOR_DEEPGRAM_API_KEY": _ALL_SECRETS[DEEPGRAM_API_KEY],
                "SIMULATOR_GEMINI_API_KEY": _ALL_SECRETS[GEMINI_API_KEY],
            }
        ),
    )
    return job, context


def _write_scenario_doc(
    bundle_dir: Path,
    *,
    scenario_key: str,
    folder_name: str | None = None,
    instruction: str = "Cancel order #42.",
    persona: dict[str, Any] | None = None,
    fixture: dict[str, Any] | None = None,
    tests: str = "",
) -> None:
    folder = bundle_dir / "scenarios" / (folder_name or scenario_key)
    folder.mkdir(parents=True, exist_ok=True)
    body = {
        "scenario_key": scenario_key,
        "scenario_id": "",
        "sub_goals": [],
        "instruction": instruction,
        "persona": persona,
        "fixture": fixture or {},
        "tests": tests,
    }
    (folder / "scenario.json").write_text(json.dumps(body), encoding="utf-8")


@dataclass
class _FakeScenario:
    """`CallRunnerImpl.run` reads only `.scenario_key` off the scheduler's `Scenario` protocol."""

    scenario_key: str
    scenario_id: str = ""


@dataclass
class FakeAdapter:
    """Narrow `ArtifactUploader` fake -- records every call, returns a real `sha256:<hex>` id
    unless the kind is in `refuse_kinds` (mirrors the real adapter's null-not-crash contract)."""

    refuse_kinds: frozenset = field(default_factory=frozenset)
    uploads: list[tuple[Any, str | None, bytes]] = field(default_factory=list)

    async def upload_artifact(
        self, data: bytes, *, kind, scenario_key=None, deadline=None
    ) -> str | None:
        if kind in self.refuse_kinds:
            return None
        import hashlib

        digest = hashlib.sha256(data).hexdigest()
        self.uploads.append((kind, scenario_key, data))
        return f"sha256:{digest}"


def _persona() -> SimPersona:
    return SimPersona(persona={"name": "customer"}, situation="s", outcome="o")


def _report(
    *,
    status: RunStatus = RunStatus.COMPLETED,
    case_status: CaseStatus | None = CaseStatus.COMPLETED,
    transcript: str = "hello there",
    messages: list[dict[str, str]] | None = None,
    failure: SimulationFailure | None = None,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
    no_cases: bool = False,
    run_id: str = "sim-run-1",
    result_metadata: dict[str, Any] | None = None,
) -> SimulationReport:
    messages = messages if messages is not None else [{"role": "user", "content": "hi"}]
    started_at = started_at or datetime.now(timezone.utc)
    ended_at = ended_at or (started_at + timedelta(seconds=30))
    cases: list[SimulationTestCaseResult] = []
    if not no_cases:
        assert case_status is not None
        result = SimTestCaseResult(
            persona=_persona(),
            transcript=transcript,
            messages=messages,
            metadata=result_metadata or {},
        )
        cases.append(
            SimulationTestCaseResult(
                test_case_id="tc-1",
                status=case_status,
                persona=_persona(),
                result=result,
                failure=failure,
                started_at=started_at,
                ended_at=ended_at,
            )
        )
    return SimulationReport(
        run_id=run_id,
        spec_hash="hash",
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        test_cases=cases,
        artifacts=ArtifactManifest(run_id=run_id),
    )


def _run(
    runner: cr.CallRunnerImpl, scenario: _FakeScenario, runtime: EnvironmentRuntime
) -> CallOutcome:
    return asyncio.run(runner.run(scenario, runtime))


def _run_expect_abort(
    runner: cr.CallRunnerImpl, scenario: _FakeScenario, runtime: EnvironmentRuntime
) -> CallAborted:
    try:
        asyncio.run(runner.run(scenario, runtime))
    except CallAborted as exc:
        return exc
    raise AssertionError("expected CallAborted, nothing was raised")


def _run_expect_world_unavailable(
    runner: cr.CallRunnerImpl, scenario: _FakeScenario, runtime: EnvironmentRuntime
) -> WorldUnavailable:
    try:
        asyncio.run(runner.run(scenario, runtime))
    except WorldUnavailable as exc:
        return exc
    raise AssertionError("expected WorldUnavailable, nothing was raised")


# =================================================================================================
# Room naming (deterministic scheme, pinned verbatim by this file). WHY only a prefix at the wire:
# engines/livekit.py::_resolve_room_name appends its own `-{invocation_id}-{test_case_id[-12:]}`
# suffix in managed room_mode unless `room_name_verbatim` is set (this runner does not set it), so
# the pinned string below is the deterministic PREFIX every dialed room carries, not the full
# on-the-wire room name.
# =================================================================================================


def test_room_name_matches_the_pinned_deterministic_scheme() -> None:
    name = cr._room_name(
        job_id="abcdef1234567890",
        attempt_number=2,
        scenario_key="cancel-order",
        scenario_attempt=3,
    )
    assert name == "harness-abcdef12-a2-cancel-order-s3"


def test_room_name_uses_only_the_first_eight_chars_of_job_id() -> None:
    short = cr._room_name(
        job_id="ab", attempt_number=1, scenario_key="k", scenario_attempt=1
    )
    assert short == "harness-ab-a1-k-s1"


# =================================================================================================
# Pre-dial validation.
# =================================================================================================


def test_missing_target_provider_secrets_aborts_pre_dial_without_calling_place_call(
    tmp_path: Path,
) -> None:
    called = False

    async def place_call(spec):
        nonlocal called
        called = True
        raise AssertionError("place_call must never be reached")

    _job_obj, context = _context(tmp_path=tmp_path, secrets={})
    runner = cr.CallRunnerImpl(FakeAdapter(), context, place_call=place_call)
    exc = _run_expect_abort(
        runner,
        _FakeScenario("k1"),
        _runtime(metadata={"livekit_agent_name": "agent-w0"}),
    )
    assert exc.partial is None
    assert str(exc).startswith("voice_capability_unavailable: missing")
    assert LIVEKIT_API_KEY in str(exc)
    assert not called


def test_missing_llm_credential_names_the_either_or_pair(tmp_path: Path) -> None:
    _job_obj, context = _context(
        tmp_path=tmp_path,
        simulator_secrets={"SIMULATOR_DEEPGRAM_API_KEY": "dg-key"},
    )
    runner = cr.CallRunnerImpl(FakeAdapter(), context, environ={})
    exc = _run_expect_abort(
        runner,
        _FakeScenario("k1"),
        _runtime(metadata={"livekit_agent_name": "agent-w0"}),
    )
    assert "GEMINI_API_KEY_or_GOOGLE_API_KEY" in str(exc)


def test_google_api_key_alone_satisfies_the_llm_credential_check(
    tmp_path: Path,
) -> None:
    _job_obj, context = _context(
        tmp_path=tmp_path,
        simulator_secrets={
            "SIMULATOR_DEEPGRAM_API_KEY": "dg-key",
            "SIMULATOR_GOOGLE_API_KEY": "g-key",
        },
    )
    runner = cr.CallRunnerImpl(FakeAdapter(), context)
    assert runner._missing_config is None


def test_missing_livekit_url_config_aborts_pre_dial(tmp_path: Path) -> None:
    _job_obj, context = _context(tmp_path=tmp_path, config={})
    runner = cr.CallRunnerImpl(FakeAdapter(), context)
    exc = _run_expect_abort(
        runner,
        _FakeScenario("k1"),
        _runtime(metadata={"livekit_agent_name": "agent-w0"}),
    )
    assert "config=livekit_url" in str(exc)


def test_missing_dispatch_identity_metadata_aborts_pre_dial(tmp_path: Path) -> None:
    """Verified finding: `EnvironmentRuntime.metadata` is always `{}` in this worktree's HEAD --
    this is the realistic default a real hosted run hits today (CONTRACT NOTE 3)."""
    called = False

    async def place_call(spec):
        nonlocal called
        called = True
        raise AssertionError("place_call must never be reached")

    _job_obj, context = _context(tmp_path=tmp_path)
    runner = cr.CallRunnerImpl(FakeAdapter(), context, place_call=place_call)
    exc = _run_expect_abort(runner, _FakeScenario("k1"), _runtime(metadata={}))
    assert "voice_dispatch_identity_unavailable" in str(exc)
    assert "livekit_agent_name" in str(exc)
    assert not called


def test_missing_scenario_document_aborts_pre_dial(tmp_path: Path) -> None:
    _job_obj, context = _context(tmp_path=tmp_path)
    runner = cr.CallRunnerImpl(FakeAdapter(), context)
    exc = _run_expect_abort(
        runner,
        _FakeScenario("no-such-key"),
        _runtime(metadata={"livekit_agent_name": "agent-w0"}),
    )
    assert "voice_scenario_document_unavailable" in str(exc)


def test_scenario_document_matched_by_scenario_key_field_not_folder_name(
    tmp_path: Path,
) -> None:
    """scenario_source.py's own convention: `scenario_key` is a field INSIDE scenario.json, not
    necessarily the folder name -- this runner must match the same way, not assume they agree."""
    _job_obj, context = _context(tmp_path=tmp_path)
    _write_scenario_doc(
        context.bundle_dir,
        scenario_key="the-real-key",
        folder_name="some-other-folder-name",
    )

    captured: dict[str, Any] = {}

    async def place_call(spec):
        captured["spec"] = spec
        return _report()

    runner = cr.CallRunnerImpl(FakeAdapter(), context, place_call=place_call)
    _run(
        runner,
        _FakeScenario("the-real-key"),
        _runtime(metadata={"livekit_agent_name": "agent-w0"}),
    )
    assert captured["spec"] is not None


# =================================================================================================
# Happy path: COMPLETED -> real CallOutcome, artifacts uploaded, dispatch/room wiring correct.
# =================================================================================================


def test_completed_call_uploads_transcript_and_returns_populated_outcome(
    tmp_path: Path,
) -> None:
    _job_obj, context = _context(
        tmp_path=tmp_path, evidence_seam=EvidenceSeam.HTTP_TOOL
    )
    _write_scenario_doc(
        context.bundle_dir, scenario_key="k1", instruction="Cancel order #42."
    )

    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ended = started + timedelta(seconds=45)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    async def place_call(spec):
        return _report(
            transcript="hi\nhello",
            messages=messages,
            started_at=started,
            ended_at=ended,
            result_metadata={"stop_reason": "simulator_end_call"},
        )

    adapter = FakeAdapter()
    runner = cr.CallRunnerImpl(adapter, context, place_call=place_call)
    outcome = _run(
        runner,
        _FakeScenario("k1"),
        _runtime(metadata={"livekit_agent_name": "agent-w0"}),
    )

    assert isinstance(outcome, CallOutcome)
    assert outcome.turns == 2
    assert outcome.duration_ms == 45_000
    assert outcome.transcript_artifact is not None
    assert outcome.transcript_artifact.startswith("sha256:")
    assert outcome.calls == ()  # http_tool: STOPPED, always zero -- CONTRACT NOTE 1
    assert outcome.stop_reason == "simulator_end_call"
    assert len(adapter.uploads) == 1


def test_dispatch_agent_name_and_livekit_url_flow_into_the_built_spec(
    tmp_path: Path,
) -> None:
    _job_obj, context = _context(
        tmp_path=tmp_path,
        config={cr.LIVEKIT_URL_CONFIG_KEY: "wss://custom.livekit.cloud"},
    )
    _write_scenario_doc(
        context.bundle_dir, scenario_key="k1", instruction="Do the thing."
    )

    captured: dict[str, Any] = {}

    async def place_call(spec):
        captured["spec"] = spec
        return _report()

    runner = cr.CallRunnerImpl(FakeAdapter(), context, place_call=place_call)
    _run(
        runner,
        _FakeScenario("k1"),
        _runtime(metadata={"livekit_agent_name": "agent-w0"}),
    )

    spec = captured["spec"]
    agent_definition = spec.environment.config["agent_definition"]
    livekit_runtime = spec.environment.config["livekit_runtime"]
    assert agent_definition["agent_name"] == "agent-w0"
    assert agent_definition["system_prompt"] == "Do the thing."
    # WHY prefix-match, not exact-match: this asserts what THIS runner puts on the spec, which is
    # exactly the pinned deterministic scheme -- but engines/livekit.py::_resolve_room_name appends
    # its own suffix in managed room_mode before the room is actually dialed, so a reader must not
    # take this string as the full on-the-wire room name.
    assert livekit_runtime["room_name"].startswith("harness-job-abcd-a1-k1-s1")
    assert livekit_runtime["url"] == "wss://custom.livekit.cloud/"
    assert (
        spec.environment.config["params"]["agent_first_silence_timeout_seconds"] == 60.0
    )
    assert spec.environment.config["params"]["min_turn_messages"] == 6


@pytest.mark.parametrize(
    ("connector", "target_key", "target_id", "provider_key", "transport"),
    [
        ("vapi", "assistant_id", "assistant-123", "VAPI_API_KEY", "vapi_websocket"),
        ("retell", "agent_id", "agent-123", "RETELL_API_KEY", "retell_webcall"),
    ],
)
def test_provider_connect_only_builds_direct_target_spec(
    tmp_path: Path,
    connector: str,
    target_key: str,
    target_id: str,
    provider_key: str,
    transport: str,
) -> None:
    secrets = {**_ALL_SECRETS, provider_key: "provider-key"}
    _job_obj, context = _context(
        tmp_path=tmp_path,
        connector=connector,
        config={
            cr.LIVEKIT_URL_CONFIG_KEY: "wss://custom.livekit.cloud",
            target_key: target_id,
        },
        secrets=secrets,
    )
    _write_scenario_doc(context.bundle_dir, scenario_key="provider-call")
    captured: dict[str, Any] = {}

    async def place_call(spec):
        captured["spec"] = spec
        return _report()

    runner = cr.CallRunnerImpl(FakeAdapter(), context, place_call=place_call)
    _run(runner, _FakeScenario("provider-call"), _runtime())

    definition = captured["spec"].environment.config["agent_definition"]
    assert definition["target"][target_key] == target_id
    assert definition["target"]["provider"] == connector
    assert definition["transport"]["kind"] == transport
    # Provider-hosted targets own their end-call tool. Five alternating messages are sufficient
    # for a complete agent-first clarification flow; no sixth caller acknowledgement is possible
    # after the provider disconnects.
    assert captured["spec"].environment.config["params"]["min_turn_messages"] == 5


def test_provider_import_calls_runtime_clone_instead_of_source_target(
    tmp_path: Path,
) -> None:
    _job_obj, context = _context(
        tmp_path=tmp_path,
        connector="vapi",
        mode=ProviderExecutionMode.PROVIDER_IMPORT,
        config={
            cr.LIVEKIT_URL_CONFIG_KEY: "wss://custom.livekit.cloud",
            "assistant_id": "source-assistant",
        },
        secrets={**_ALL_SECRETS, "VAPI_API_KEY": "provider-key"},
    )
    _write_scenario_doc(context.bundle_dir, scenario_key="provider-import-call")
    captured: dict[str, Any] = {}

    async def place_call(spec):
        captured["spec"] = spec
        return _report()

    runner = cr.CallRunnerImpl(FakeAdapter(), context, place_call=place_call)
    _run(
        runner,
        _FakeScenario("provider-import-call"),
        _runtime(metadata={"provider_target_id": "cloned-assistant"}),
    )

    definition = captured["spec"].environment.config["agent_definition"]
    assert definition["target"]["assistant_id"] == "cloned-assistant"


def test_scenario_attempt_counter_increments_per_scenario_key_across_retries(
    tmp_path: Path,
) -> None:
    """The scheduler retries the SAME scenario_key (e.g. after evidence_missing) -- successive
    `run()` calls for one key must get distinct room names, or a LiveKit room collision follows."""
    _job_obj, context = _context(tmp_path=tmp_path)
    _write_scenario_doc(context.bundle_dir, scenario_key="k1")
    _write_scenario_doc(context.bundle_dir, scenario_key="k2")

    rooms: list[str] = []

    async def place_call(spec):
        rooms.append(spec.environment.config["livekit_runtime"]["room_name"])
        return _report()

    runner = cr.CallRunnerImpl(FakeAdapter(), context, place_call=place_call)
    runtime = _runtime(metadata={"livekit_agent_name": "agent-w0"})
    _run(runner, _FakeScenario("k1"), runtime)
    _run(runner, _FakeScenario("k2"), runtime)
    _run(runner, _FakeScenario("k1"), runtime)

    assert rooms[0].endswith("-k1-s1")
    assert rooms[1].endswith("-k2-s1")
    assert rooms[2].endswith("-k1-s2")


# =================================================================================================
# Failure semantics -- the three cases the brief pins.
# =================================================================================================


def test_agent_unavailable_status_raises_world_unavailable(tmp_path: Path) -> None:
    """Verified against engines/livekit.py: AGENT_UNAVAILABLE fires ONLY on a readiness-stage
    timeout with a session started but no target dispatched -- exactly "dispatch fails, agent
    never joins," matching the brief's explicit WorldUnavailable case."""
    _job_obj, context = _context(tmp_path=tmp_path)
    _write_scenario_doc(context.bundle_dir, scenario_key="k1")

    async def place_call(spec):
        failure = SimulationFailure(
            stage=FailureStage.READINESS,
            code="agent_unavailable",
            message="Target agent did not become ready",
        )
        return _report(case_status=CaseStatus.AGENT_UNAVAILABLE, failure=failure)

    runner = cr.CallRunnerImpl(FakeAdapter(), context, place_call=place_call)
    exc = _run_expect_world_unavailable(
        runner,
        _FakeScenario("k1"),
        _runtime(metadata={"livekit_agent_name": "agent-w0"}),
    )
    assert "Target agent did not become ready" in str(exc)


def test_non_completed_status_raises_call_aborted_with_partial(tmp_path: Path) -> None:
    _job_obj, context = _context(tmp_path=tmp_path)
    _write_scenario_doc(context.bundle_dir, scenario_key="k1")
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ended = started + timedelta(seconds=12)

    async def place_call(spec):
        failure = SimulationFailure(
            stage=FailureStage.RUNNING, code="case_execution_error", message="boom"
        )
        return _report(
            case_status=CaseStatus.FAILED,
            failure=failure,
            started_at=started,
            ended_at=ended,
            transcript="",
            messages=[],
        )

    runner = cr.CallRunnerImpl(FakeAdapter(), context, place_call=place_call)
    exc = _run_expect_abort(
        runner,
        _FakeScenario("k1"),
        _runtime(metadata={"livekit_agent_name": "agent-w0"}),
    )
    assert exc.partial is not None
    assert exc.partial.duration_ms == 12_000
    assert exc.partial.calls == ()
    assert "boom" in str(exc)


def test_non_completed_tool_trace_call_preserves_evidence_and_uploads_trace(
    tmp_path: Path, monkeypatch
) -> None:
    """A failed voice call must not discard tool evidence collected before the failure."""
    _job_obj, context = _context(
        tmp_path=tmp_path, evidence_seam=EvidenceSeam.TOOL_TRACE
    )
    _write_scenario_doc(context.bundle_dir, scenario_key="k1")
    captured = Call(
        name="get_ride_options",
        arguments={"pickup": "SFO"},
        result={"options": ["economy"]},
        ok=True,
        error=None,
        refused=False,
        at="2026-01-01T00:00:01.000Z",
    )
    monkeypatch.setattr(cr, "_collect_tool_trace_calls", lambda runtime: (captured,))

    async def place_call(spec):
        failure = SimulationFailure(
            stage=FailureStage.RUNNING,
            code="call_timeout",
            message="conversation exceeded its limit",
        )
        return _report(
            case_status=CaseStatus.FAILED,
            failure=failure,
            transcript="user: hello",
            messages=[{"role": "user", "content": "hello"}],
        )

    adapter = FakeAdapter()
    runner = cr.CallRunnerImpl(adapter, context, place_call=place_call)
    exc = _run_expect_abort(
        runner,
        _FakeScenario("k1"),
        _runtime(metadata={"livekit_agent_name": "agent-w0"}),
    )

    assert exc.partial is not None
    assert exc.partial.calls == (captured,)
    trace_uploads = [
        item for item in adapter.uploads if item[0] is cr.ArtifactKind.TOOL_TRACE
    ]
    assert len(trace_uploads) == 1
    assert (
        json.loads(trace_uploads[0][2].decode().splitlines()[0])["name"]
        == "get_ride_options"
    )


def test_no_test_cases_in_report_raises_call_aborted_with_timing_only_partial(
    tmp_path: Path,
) -> None:
    _job_obj, context = _context(tmp_path=tmp_path)
    _write_scenario_doc(context.bundle_dir, scenario_key="k1")

    async def place_call(spec):
        return _report(status=RunStatus.TIMED_OUT, no_cases=True)

    runner = cr.CallRunnerImpl(FakeAdapter(), context, place_call=place_call)
    exc = _run_expect_abort(
        runner,
        _FakeScenario("k1"),
        _runtime(metadata={"livekit_agent_name": "agent-w0"}),
    )
    assert exc.partial is not None
    assert exc.partial.turns == 0
    assert exc.partial.calls == ()


def test_place_call_exception_raises_call_aborted_with_timing_partial_never_raw(
    tmp_path: Path,
) -> None:
    """world-handle-interface.md's partial-call rule: a generic exception must never lose timing
    (the brief: "never let a raw exception escape post-dial")."""
    _job_obj, context = _context(tmp_path=tmp_path)
    _write_scenario_doc(context.bundle_dir, scenario_key="k1")

    async def place_call(spec):
        raise RuntimeError("engine exploded")

    runner = cr.CallRunnerImpl(FakeAdapter(), context, place_call=place_call)
    exc = _run_expect_abort(
        runner,
        _FakeScenario("k1"),
        _runtime(metadata={"livekit_agent_name": "agent-w0"}),
    )
    assert exc.partial is not None
    assert exc.partial.started_at is not None
    assert exc.partial.ended_at is not None
    assert "engine exploded" in str(exc)


def test_translate_report_upload_failure_raises_call_aborted_with_timing_partial_never_raw(
    tmp_path: Path,
) -> None:
    """The same partial-call rule as `place_call` failing above, but for a failure INSIDE
    `_translate_report` itself (a transcript/recording read or an `upload_artifact` surprise) --
    this must also never escape `run()` raw and lose the timing the call already measured."""
    _job_obj, context = _context(tmp_path=tmp_path)
    _write_scenario_doc(context.bundle_dir, scenario_key="k1")

    class RaisingAdapter:
        async def upload_artifact(
            self, data, *, kind, scenario_key=None, deadline=None
        ):
            raise RuntimeError("upload exploded")

    async def place_call(spec):
        return _report(
            transcript="hi",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
            ],
        )

    runner = cr.CallRunnerImpl(RaisingAdapter(), context, place_call=place_call)
    exc = _run_expect_abort(
        runner,
        _FakeScenario("k1"),
        _runtime(metadata={"livekit_agent_name": "agent-w0"}),
    )
    assert exc.partial is not None
    assert exc.partial.started_at is not None
    assert exc.partial.ended_at is not None
    assert exc.partial.calls == ()
    assert "upload exploded" in str(exc)


def test_place_call_outer_timeout_raises_call_aborted_with_timing_partial(
    tmp_path: Path, monkeypatch
) -> None:
    """Forces the runner-owned `asyncio.wait_for` to actually fire (not just the SDK's own
    internal one) by shrinking every phase-overhead constant to a few milliseconds -- avoids a
    multi-minute real sleep in the test suite while still exercising the real timeout code path."""
    monkeypatch.setattr(cr, "CONNECT_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(cr, "READINESS_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(cr, "CLEANUP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(cr, "_RUN_SECONDS_PAD_SECONDS", 0.01)
    monkeypatch.setattr(cr, "_OUTER_WAIT_FOR_PAD_SECONDS", 0.01)

    _job_obj, context = _context(
        tmp_path=tmp_path,
        config={cr.LIVEKIT_URL_CONFIG_KEY: "wss://x", cr.CALL_TIMEOUT_CONFIG_KEY: 0.01},
    )
    _write_scenario_doc(context.bundle_dir, scenario_key="k1")

    async def place_call(spec):
        await asyncio.Event().wait()  # never completes on its own

    runner = cr.CallRunnerImpl(FakeAdapter(), context, place_call=place_call)
    exc = _run_expect_abort(
        runner,
        _FakeScenario("k1"),
        _runtime(metadata={"livekit_agent_name": "agent-w0"}),
    )
    assert "voice_call_runner_timeout" in str(exc)
    assert exc.partial is not None
    assert exc.partial.calls == ()


def test_silent_agent_real_engine_shape_returns_normal_outcome_with_empty_calls(
    tmp_path: Path,
) -> None:
    """Pins the shape the real engine actually produces for a silent agent-first call
    (engines/livekit.py::_conversation_outcome): status FAILED, code "no_conversation", zero
    messages -- never a COMPLETED case with zero turns (COMPLETED requires >= min_turn_messages
    and role alternation, a shape the engine cannot produce for a silent call). Must still surface
    as a NORMAL CallOutcome with the real (zero) turn count, never a WorldUnavailable/CallAborted,
    letting the scheduler's own coverage guarantee turn it into evidence_missing."""
    _job_obj, context = _context(tmp_path=tmp_path)
    _write_scenario_doc(context.bundle_dir, scenario_key="k1")

    async def place_call(spec):
        failure = SimulationFailure(
            stage=FailureStage.RUNNING,
            code="no_conversation",
            message="No conversation turns were committed before the inactivity deadline",
            retryable=True,
        )
        return _report(
            case_status=CaseStatus.FAILED, failure=failure, transcript="", messages=[]
        )

    runner = cr.CallRunnerImpl(FakeAdapter(), context, place_call=place_call)
    outcome = _run(
        runner,
        _FakeScenario("k1"),
        _runtime(metadata={"livekit_agent_name": "agent-w0"}),
    )
    assert isinstance(outcome, CallOutcome)
    assert outcome.turns == 0
    assert outcome.calls == ()
    assert outcome.transcript_artifact is None  # empty transcript -- never uploaded


def test_silent_agent_conversation_silence_timeout_code_also_returns_normal_outcome(
    tmp_path: Path,
) -> None:
    """The engine's other zero-turn silent code (the agent-first silence watchdog firing before
    any turn ever lands) must map the same way as "no_conversation" above."""
    _job_obj, context = _context(tmp_path=tmp_path)
    _write_scenario_doc(context.bundle_dir, scenario_key="k1")

    async def place_call(spec):
        failure = SimulationFailure(
            stage=FailureStage.RUNNING,
            code="conversation_silence_timeout",
            message="Agent-first conversation stalled after it began",
            retryable=True,
        )
        return _report(
            case_status=CaseStatus.FAILED, failure=failure, transcript="", messages=[]
        )

    runner = cr.CallRunnerImpl(FakeAdapter(), context, place_call=place_call)
    outcome = _run(
        runner,
        _FakeScenario("k1"),
        _runtime(metadata={"livekit_agent_name": "agent-w0"}),
    )
    assert isinstance(outcome, CallOutcome)
    assert outcome.turns == 0
    assert outcome.calls == ()


def test_silent_agent_mapping_is_scoped_to_zero_turns_only(tmp_path: Path) -> None:
    """A short-but-nonzero conversation carrying the same failure code must NOT be laundered into
    a normal outcome -- only a genuinely zero-turn silent call qualifies; this stays a CallAborted
    exactly like any other non-completed status."""
    _job_obj, context = _context(tmp_path=tmp_path)
    _write_scenario_doc(context.bundle_dir, scenario_key="k1")
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    async def place_call(spec):
        failure = SimulationFailure(
            stage=FailureStage.RUNNING,
            code="conversation_silence_timeout",
            message="Agent-first conversation stalled after it began",
            retryable=True,
        )
        return _report(
            case_status=CaseStatus.FAILED,
            failure=failure,
            transcript="hi\nhello",
            messages=messages,
        )

    runner = cr.CallRunnerImpl(FakeAdapter(), context, place_call=place_call)
    exc = _run_expect_abort(
        runner,
        _FakeScenario("k1"),
        _runtime(metadata={"livekit_agent_name": "agent-w0"}),
    )
    assert exc.partial is not None
    assert exc.partial.turns == 2


# =================================================================================================
# Evidence collection.
# =================================================================================================


def test_http_tool_seam_always_returns_no_calls() -> None:
    """CONTRACT NOTE 1: STOPPED, verified -- no invented capture proxy."""
    runtime = _runtime(endpoints={"database": _postgres_endpoint()})
    assert cr._collect_http_tool_calls(runtime) == ()


def test_find_postgres_endpoint_matches_by_protocol_not_a_fixed_slug_name() -> None:
    """Corrects the brief's literal `runtime.endpoints["database"]` wording: capability slugs are
    bundle-author-chosen (verified against `build_endpoints`/`_find_postgres_endpoint` in
    hosted_entrypoint.py) -- a bundle naming its slug anything else must still resolve."""
    endpoint = _postgres_endpoint()
    runtime = _runtime(endpoints={"orders_store": endpoint})
    found = cr._find_postgres_endpoint(runtime)
    assert found is endpoint


def test_find_postgres_endpoint_returns_none_when_absent() -> None:
    runtime = _runtime(
        endpoints={
            "queue": RuntimeEndpoint(
                capability="queue", protocol="amqp", address="amqp://x"
            )
        }
    )
    assert cr._find_postgres_endpoint(runtime) is None


class _FakeCursor:
    def __init__(self, rows: list[tuple[Any, ...]], columns: list[str]) -> None:
        self._rows = rows
        self.description = [(name,) for name in columns]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


class _FakeConnection:
    def __init__(self, rows: list[tuple[Any, ...]], columns: list[str]) -> None:
        self._rows = rows
        self._columns = columns
        self.executed: list[str] = []

    def execute(self, statement: str, params: Any = None) -> _FakeCursor:
        self.executed.append(statement)
        return _FakeCursor(self._rows, self._columns)

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        return False


class _FakePsycopg:
    def __init__(self, rows: list[tuple[Any, ...]], columns: list[str]) -> None:
        self._rows = rows
        self._columns = columns
        self.connections: list[_FakeConnection] = []

    def connect(self, dsn: str, **kwargs: Any) -> _FakeConnection:
        connection = _FakeConnection(self._rows, self._columns)
        self.connections.append(connection)
        return connection


class _RaisingPsycopg:
    class Error(Exception):
        pass

    def connect(self, dsn: str, **kwargs: Any) -> Any:
        raise self.Error("connection refused")


def test_tool_trace_translates_rows_and_applies_v1_refused_rule(monkeypatch) -> None:
    """world-handle-interface.md V1 rule: `refused = not ok` (a trace cannot distinguish refusal
    from crash)."""
    columns = ["name", "arguments", "result", "ok", "error", "at"]
    rows = [
        ("lookup_order", {"id": "1"}, {"status": "shipped"}, True, "", 100.5),
        ("cancel_order", {"id": "2"}, None, False, "not found", 101.0),
    ]
    fake = _FakePsycopg(rows, columns)
    monkeypatch.setitem(sys.modules, "psycopg", fake)

    runtime = _runtime(endpoints={"database": _postgres_endpoint()})
    calls = cr._collect_tool_trace_calls(runtime)

    assert calls == (
        Call(
            name="lookup_order",
            arguments={"id": "1"},
            result={"status": "shipped"},
            ok=True,
            error="",
            refused=False,
            at=100.5,
        ),
        Call(
            name="cancel_order",
            arguments={"id": "2"},
            result=None,
            ok=False,
            error="not found",
            refused=True,
            at=101.0,
        ),
    )


def test_tool_trace_read_failure_degrades_to_no_calls_never_crashes(
    monkeypatch,
) -> None:
    """No producer exists yet (CONTRACT NOTE 2) -- a missing table / connection failure must
    degrade to `()`, never raise past this function."""
    monkeypatch.setitem(sys.modules, "psycopg", _RaisingPsycopg())
    runtime = _runtime(endpoints={"database": _postgres_endpoint()})
    assert cr._collect_tool_trace_calls(runtime) == ()


def test_tool_trace_missing_endpoint_degrades_to_no_calls() -> None:
    runtime = _runtime(endpoints={})
    assert cr._collect_tool_trace_calls(runtime) == ()


def test_tool_trace_result_string_form_is_truncated_at_2000_chars(monkeypatch) -> None:
    long_string = "x" * 3000
    columns = ["name", "arguments", "result", "ok", "error", "at"]
    rows = [("t", {}, long_string, True, "", 1.0)]
    monkeypatch.setitem(sys.modules, "psycopg", _FakePsycopg(rows, columns))
    runtime = _runtime(endpoints={"database": _postgres_endpoint()})
    calls = cr._collect_tool_trace_calls(runtime)
    assert len(calls) == 1
    assert len(calls[0].result) == 2000


def test_tool_trace_parsed_json_result_is_not_truncated(monkeypatch) -> None:
    """Per world-handle-interface.md: "result — parsed JSON where the source captured JSON, else
    a string; both result (string form) and error truncated at 2,000 chars" -- truncation applies
    to the STRING form only."""
    big_list = list(range(3000))
    columns = ["name", "arguments", "result", "ok", "error", "at"]
    rows = [("t", {}, big_list, True, "", 1.0)]
    monkeypatch.setitem(sys.modules, "psycopg", _FakePsycopg(rows, columns))
    runtime = _runtime(endpoints={"database": _postgres_endpoint()})
    calls = cr._collect_tool_trace_calls(runtime)
    assert calls[0].result == big_list


def test_tool_trace_clear_is_best_effort_and_never_raises(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "psycopg", _RaisingPsycopg())
    cr._clear_tool_trace_calls("postgresql://x/y")  # must not raise


def test_completed_call_with_tool_trace_seam_collects_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    columns = ["name", "arguments", "result", "ok", "error", "at"]
    rows = [("do_thing", {}, "done", True, "", 5.0)]
    monkeypatch.setitem(sys.modules, "psycopg", _FakePsycopg(rows, columns))

    _job_obj, context = _context(
        tmp_path=tmp_path, evidence_seam=EvidenceSeam.TOOL_TRACE
    )
    _write_scenario_doc(context.bundle_dir, scenario_key="k1")

    async def place_call(spec):
        return _report()

    runtime = _runtime(
        metadata={"livekit_agent_name": "agent-w0"},
        endpoints={"database": _postgres_endpoint()},
    )
    runner = cr.CallRunnerImpl(FakeAdapter(), context, place_call=place_call)
    outcome = _run(runner, _FakeScenario("k1"), runtime)
    assert len(outcome.calls) == 1
    assert outcome.calls[0].name == "do_thing"


# =================================================================================================
# Credential export (WHY: the LiveKit engine reads these via ambient os.environ, not spec fields).
# =================================================================================================


def test_construction_exports_target_provider_secrets_to_environ_once(
    tmp_path: Path,
) -> None:
    fake_environ: dict[str, str] = {}
    _job_obj, context = _context(tmp_path=tmp_path)
    cr.CallRunnerImpl(FakeAdapter(), context, environ=fake_environ)
    assert fake_environ[LIVEKIT_API_KEY] == "lk-key"
    assert fake_environ[LIVEKIT_API_SECRET] == "lk-secret"
    assert fake_environ[DEEPGRAM_API_KEY] == "dg-key"
    assert fake_environ[GEMINI_API_KEY] == "gm-key"


def test_construction_never_exports_secrets_outside_the_target_provider_map(
    tmp_path: Path,
) -> None:
    fake_environ: dict[str, str] = {}
    secrets = dict(_ALL_SECRETS)
    secrets["UNRELATED_ALIAS"] = "should-not-export"
    _job_obj, context = _context(tmp_path=tmp_path, secrets=secrets)
    cr.CallRunnerImpl(FakeAdapter(), context, environ=fake_environ)
    assert "UNRELATED_ALIAS" not in fake_environ


def test_construction_uses_platform_simulator_key_without_exposing_agent_model_key(
    tmp_path: Path,
) -> None:
    fake_environ: dict[str, str] = {}
    target = {
        LIVEKIT_API_KEY: "lk-key",
        LIVEKIT_API_SECRET: "lk-secret",
        "ANTHROPIC_API_KEY": "customer-agent-key",
    }
    simulator = {
        "SIMULATOR_DEEPGRAM_API_KEY": "platform-deepgram-key",
        "SIMULATOR_GEMINI_API_KEY": "platform-gemini-key",
    }
    _job_obj, context = _context(
        tmp_path=tmp_path,
        secrets=target,
        simulator_secrets=simulator,
    )
    cr.CallRunnerImpl(FakeAdapter(), context, environ=fake_environ)
    assert fake_environ[DEEPGRAM_API_KEY] == "platform-deepgram-key"
    assert fake_environ[GEMINI_API_KEY] == "platform-gemini-key"
    assert "ANTHROPIC_API_KEY" not in fake_environ


def test_hosted_run_never_falls_back_to_customer_simulator_keys(tmp_path: Path) -> None:
    fake_environ: dict[str, str] = {}
    target = dict(_ALL_SECRETS)
    _job_obj, context = _context(
        tmp_path=tmp_path,
        secrets=target,
        simulator_secrets={},
    )
    runner = cr.CallRunnerImpl(FakeAdapter(), context, environ=fake_environ)
    assert DEEPGRAM_API_KEY not in fake_environ
    assert GEMINI_API_KEY not in fake_environ
    assert runner._missing_config is not None


def test_local_sdk_remains_byok_for_simulator_credentials(tmp_path: Path) -> None:
    fake_environ: dict[str, str] = {}
    _job_obj, context = _context(
        tmp_path=tmp_path,
        execution=ExecutionMode.LOCAL,
        simulator_secrets={},
    )
    cr.CallRunnerImpl(FakeAdapter(), context, environ=fake_environ)
    assert fake_environ[DEEPGRAM_API_KEY] == "dg-key"
    assert fake_environ[GEMINI_API_KEY] == "gm-key"


def test_platform_simulator_credentials_win_without_replacing_target_livekit(
    tmp_path: Path,
) -> None:
    fake_environ = {
        DEEPGRAM_API_KEY: "platform-deepgram",
        GEMINI_API_KEY: "platform-gemini",
        "GOOGLE_APPLICATION_CREDENTIALS": "/run/futureagi/platform-vertex.json",
        "GOOGLE_CLOUD_PROJECT": "platform-project",
    }
    _job_obj, context = _context(tmp_path=tmp_path)

    runner = cr.CallRunnerImpl(FakeAdapter(), context, environ=fake_environ)

    assert fake_environ[LIVEKIT_API_KEY] == "lk-key"
    assert fake_environ[LIVEKIT_API_SECRET] == "lk-secret"
    assert fake_environ[DEEPGRAM_API_KEY] == "platform-deepgram"
    assert fake_environ[GEMINI_API_KEY] == "platform-gemini"
    assert (
        fake_environ["GOOGLE_APPLICATION_CREDENTIALS"]
        == "/run/futureagi/platform-vertex.json"
    )
    assert runner._missing_config is None


def test_close_quiesces_livekit_objects_before_event_loop_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_environ: dict[str, str] = {}
    _job_obj, context = _context(tmp_path=tmp_path)
    runner = cr.CallRunnerImpl(FakeAdapter(), context, environ=fake_environ)
    collections: list[None] = []
    monkeypatch.setattr(cr.gc, "collect", lambda: collections.append(None))

    async def close_twice() -> None:
        await runner.close()
        await runner.close()

    asyncio.run(close_twice())

    assert collections == [None, None]
    assert runner._closed is True


def test_provider_voice_uses_platform_livekit_without_exposing_customer_livekit(
    tmp_path: Path,
) -> None:
    fake_environ: dict[str, str] = {}
    target = {
        RETELL_API_KEY: "customer-retell-key",
        LIVEKIT_API_KEY: "customer-livekit-must-not-export",
        LIVEKIT_API_SECRET: "customer-livekit-secret-must-not-export",
    }
    simulator = {
        "SIMULATOR_LIVEKIT_URL": "wss://platform-livekit.example",
        "SIMULATOR_LIVEKIT_API_KEY": "platform-livekit-key",
        "SIMULATOR_LIVEKIT_API_SECRET": "platform-livekit-secret",
        "SIMULATOR_DEEPGRAM_API_KEY": "platform-deepgram",
        "SIMULATOR_GEMINI_API_KEY": "platform-gemini",
    }
    _job_obj, context = _context(
        tmp_path=tmp_path,
        connector="retell",
        mode=ProviderExecutionMode.PROVIDER_IMPORT,
        config={"agent_id": "source-agent"},
        secrets=target,
        simulator_secrets=simulator,
    )

    runner = cr.CallRunnerImpl(FakeAdapter(), context, environ=fake_environ)

    assert fake_environ[LIVEKIT_API_KEY] == "platform-livekit-key"
    assert fake_environ[LIVEKIT_API_SECRET] == "platform-livekit-secret"
    assert fake_environ[RETELL_API_KEY] == "customer-retell-key"
    assert runner._livekit_url == "wss://platform-livekit.example"
    assert runner._missing_config is None


def test_auto_connector_resolves_to_livekit_from_target_secrets() -> None:
    job = _job(connector="auto")
    assert (
        cr._resolve_connector(job, {cr.LIVEKIT_URL_ALIAS: "wss://x.livekit.cloud"})
        == "livekit"
    )


def test_auto_connector_resolves_to_livekit_from_config() -> None:
    job = _job(connector="auto", config=dict(_ALL_CONFIG))
    assert cr._resolve_connector(job, {}) == "livekit"


def test_auto_connector_resolves_to_vapi_and_retell_from_target_secrets() -> None:
    job = _job(connector="auto")
    assert cr._resolve_connector(job, {cr.VAPI_API_KEY_ALIAS: "k"}) == "vapi"
    assert cr._resolve_connector(job, {cr.RETELL_API_KEY_ALIAS: "k"}) == "retell"


def test_explicit_connector_is_never_overridden() -> None:
    job = _job(connector="retell")
    assert cr._resolve_connector(job, {cr.LIVEKIT_URL_ALIAS: "wss://x"}) == "retell"


def _timed_out_runner(tmp_path: Path, turns: int):
    """A runner whose call comes back TIMED_OUT after `turns` messages."""
    _job_obj, context = _context(tmp_path=tmp_path)
    _write_scenario_doc(context.bundle_dir, scenario_key="k1")
    messages = [{"role": "user", "content": f"turn {i}"} for i in range(turns)]

    async def place_call(spec):
        return _report(
            status=RunStatus.TIMED_OUT,
            case_status=CaseStatus.TIMED_OUT,
            messages=messages,
            failure=SimulationFailure(
                stage=FailureStage.RUNNING,
                code="conversation_deadline",
                message="Conversation exceeded its deadline",
            ),
        )

    return cr.CallRunnerImpl(FakeAdapter(), context, place_call=place_call)


def test_a_timed_out_call_with_a_real_conversation_is_graded_not_aborted(
    tmp_path: Path,
) -> None:
    """An intake agent may ask thirty to fifty questions, so reaching the deadline is an ordinary
    outcome. A measured 51-turn call lost all three sub-goals to `held: null` because the timeout was
    treated as infrastructure."""
    outcome = _run(
        _timed_out_runner(tmp_path, 8),
        _FakeScenario("k1"),
        _runtime(metadata={"livekit_agent_name": "a-w0"}),
    )
    assert outcome is not None, (
        "a timed-out call that held a conversation must still be graded"
    )


def test_a_timed_out_call_that_never_got_going_still_aborts(tmp_path: Path) -> None:
    """The carve-out salvages evidence; it does not excuse a call that produced none."""
    aborted = _run_expect_abort(
        _timed_out_runner(tmp_path, 1),
        _FakeScenario("k1"),
        _runtime(metadata={"livekit_agent_name": "a-w0"}),
    )
    assert "voice_call_not_completed" in str(aborted)
