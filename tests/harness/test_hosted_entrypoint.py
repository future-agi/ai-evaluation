"""`hosted_entrypoint.py` against in-memory fakes — no real postgres, no real network."""

from __future__ import annotations

import asyncio
import pytest
import contextlib
import hashlib
import json
import os
import random
import stat
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from unittest import mock

from fi.alk.harness import hosted_entrypoint as he
from fi.alk.harness import outbound as ob
from fi.alk.harness.bundle_v2 import (
    BUNDLE_V2_SCHEMA_VERSION,
    EnvironmentBundleV2,
    EvidenceSeam,
    ManagedEngine,
    compute_inputs_digest,
    seal_bundle_v2,
)
from fi.alk.harness.hosted_scheduler import (
    Call,
    CallAborted,
    CallOutcome,
    HostedScheduler,
    ReceiptFailure,
    ResultReceipt,
    RunResult,
    WorldPool,
)
from fi.alk.harness.job import (
    AgentConnection,
    ArtifactLevel,
    ExecutionMode,
    FailureDomain,
    HarnessArtifactPolicy,
    HarnessJob,
    HarnessStage,
    RepositorySource,
    SourceKind,
    SourceVisibility,
)
from fi.alk.harness.process_runtime import (
    EnvironmentRuntime,
    ProcessRuntimeError,
    RuntimeEndpoint,
    RuntimeState,
)
from fi.alk.harness.scenario_source import SCENARIOS_DIRNAME
from fi.simulate.runtime.spec import RuntimeIsolation, RuntimeRequirements, SecretRef

SCHEMA_SQL = b"CREATE TABLE riders (id int);\n"
SEED_SQL = b"INSERT INTO riders VALUES (1);\n"
TARGET_PROVIDER_ALIAS = "LIVEKIT_API_KEY"


# =================================================================================================
# Bundle fixture — mirrors test_process_preflight.py's own helper (not imported: this file is
# self-contained per the "touch only your two new files" rule).
# =================================================================================================


@pytest.fixture(autouse=True)
def _judged_sub_goals_decided_without_a_model(monkeypatch):
    """These tests are about the entrypoint, not about judging.

    A judged sub-goal now goes to a model, so without this every scenario carrying one would make a
    live call and fail on the verdict rather than on what the test is asking about.
    """
    from fi.alk.harness import hosted_scheduler

    async def _held(goal, world, calls):
        return True, f"{goal.name}: stubbed for an entrypoint test"

    monkeypatch.setattr(hosted_scheduler, "_judge", _held)


def _base_manifest_body() -> dict[str, Any]:
    return {
        "schema_version": BUNDLE_V2_SCHEMA_VERSION,
        "name": "demo",
        "runtime": {
            "kind": "process",
            "control_service": "agent",
            "evidence_seam": "http_tool",
        },
        "processes": [
            {
                "name": "postgres",
                "kind": "managed",
                "engine": "postgres",
                "version": "16",
                "user": "svc-data",
                "depends_on": [],
            },
            {
                "name": "agent",
                "kind": "source",
                "working_directory": ".",
                "build_commands": [["pip", "install", "-r", "requirements.txt"]],
                "run_command": ["python", "agent.py"],
                "environment": {
                    "DATABASE_URL": "{{DATABASE_URL}}",
                    "LIVEKIT_AGENT_NAME": "agent-w{{WORLD_INDEX}}",
                },
                "secret_purposes": ["target_provider"],
                "user": "svc-agent",
                "depends_on": ["postgres"],
            },
        ],
        "capabilities": {
            "database": {
                "protocol": "postgres",
                "service": "postgres",
                "configuration_name": "DATABASE_URL",
            },
        },
        "readiness": [],
        "provenance": {
            "source_kind": "repository",
            "repository": "org/repo",
            "source_digest": "c" * 64,
        },
        "metadata": {},
    }


def _write_bundle(root: Path) -> EnvironmentBundleV2:
    root.mkdir(parents=True, exist_ok=True)
    body = _base_manifest_body()
    file_contents = {"db/schema.sql": SCHEMA_SQL, "db/seed.sql": SEED_SQL}
    files: list[dict[str, Any]] = []
    for relative, content in file_contents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
            }
        )
    body["files"] = files
    digest = compute_inputs_digest(
        root,
        ["db/schema.sql"],
        ["db/seed.sql"],
        engine=ManagedEngine.POSTGRES,
        version="16",
    )
    body["seed"] = {
        "stores": [
            {
                "capability": "database",
                "migrations": ["db/schema.sql"],
                "seed_files": ["db/seed.sql"],
                "baseline": {"strategy": "template_database", "inputs_digest": digest},
                "sentinel": {"query": "SELECT count(*) FROM riders", "expected": "1"},
            }
        ]
    }
    body["digest"] = "sha256:" + "0" * 64
    normalized = EnvironmentBundleV2.model_validate(body)
    body["digest"] = seal_bundle_v2(normalized)
    (root / "manifest.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    return EnvironmentBundleV2.model_validate(body)


def _job(
    *,
    connector: str = "vapi",
    parallelism: int = 1,
    artifacts: HarnessArtifactPolicy | None = None,
) -> HarnessJob:
    runtime = RuntimeRequirements(
        isolation=RuntimeIsolation.DEDICATED_VM,
        cpu_units=max(parallelism, 1),
    )
    runtime = runtime.model_copy(update={"parallelism": parallelism})
    return HarnessJob(
        job_id="job-1",
        run_id="run-1",
        execution=ExecutionMode.HOSTED,
        source=RepositorySource(
            kind=SourceKind.GITHUB,
            repository="org/repo",
            visibility=SourceVisibility.PUBLIC,
            commit_sha="a" * 40,
        ),
        agent=AgentConnection(
            connector=connector,
            secret_refs={
                TARGET_PROVIDER_ALIAS: SecretRef(
                    manager="platform-vault", key="secret-id", purpose="target_provider"
                )
            },
        ),
        scenario_count=2,
        seed=1234,
        runtime=runtime,
        **({"artifacts": artifacts} if artifacts is not None else {}),
    )


def _write_job(path: Path, job: HarnessJob) -> None:
    path.write_text(job.model_dump_json(), encoding="utf-8")


def test_hosted_job_accepts_internal_platform_simulator_ref() -> None:
    job = _job()
    refs = dict(job.agent.secret_refs)
    refs["SIMULATOR_DEEPGRAM_API_KEY"] = SecretRef(
        manager="platform-config",
        key="SIMULATOR_DEEPGRAM_API_KEY",
        purpose="simulator_provider",
    )

    validated = HarnessJob.model_validate(
        job.model_dump() | {"agent": job.agent.model_dump() | {"secret_refs": refs}}
    )

    assert (
        validated.agent.secret_refs["SIMULATOR_DEEPGRAM_API_KEY"].purpose
        == "simulator_provider"
    )


def test_hosted_job_rejects_simulator_ref_from_customer_vault() -> None:
    job = _job()
    refs = dict(job.agent.secret_refs)
    refs["SIMULATOR_DEEPGRAM_API_KEY"] = SecretRef(
        manager="platform-vault",
        key="customer-selected",
        purpose="simulator_provider",
    )

    try:
        HarnessJob.model_validate(
            job.model_dump() | {"agent": job.agent.model_dump() | {"secret_refs": refs}}
        )
    except ValueError as exc:
        assert "hosted_secret_manager_unsupported" in str(exc)
    else:
        raise AssertionError("customer-vault simulator ref must be rejected")


# =================================================================================================
# Capabilities fixture.
# =================================================================================================


def _capabilities(*, attempt_id: str = "attempt-1") -> ob.HostedCapabilities:
    base = f"https://platform.example/simulate/api/harness/attempts/{attempt_id}"
    return ob.HostedCapabilities.model_validate(
        {
            "schema_version": ob.CAPABILITIES_SCHEMA_VERSION,
            "job_id": "job-1",
            "attempt_id": attempt_id,
            "attempt_number": 1,
            "fence": "fence-1",
            "expires_at": "2999-01-01T00:00:00.000Z",
            "token": "bearer-token",
            "endpoints": {
                "events": f"{base}/events/",
                "results": f"{base}/results/",
                "artifacts": f"{base}/artifacts/",
                "scenarios": f"{base}/scenarios/",
            },
        }
    )


# =================================================================================================
# FakeTransport — a minimal in-memory platform. Routes on a URL substring, not a full router: this
# module only needs the four channel shapes, not a general HTTP mock.
# =================================================================================================


@dataclass
class FakeTransport:
    fence_after: int | None = (
        None  # 1-based call count at which every further call 403s.
    )
    fence_on_url_substring: str | None = (
        None  # once a URL matches, that call and every later one 403s.
    )
    # fences the specific events POST whose batch carries a `type: "terminal"` record --
    # `emit_terminal()`'s own spool append succeeds, so this reproduces the fence landing on the
    # network flush that delivers the terminal event, not before it.
    fence_on_terminal_event: bool = False
    _fenced: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list)
    event_records: list[dict[str, Any]] = field(default_factory=list)
    receipts: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    artifacts: dict[str, bytes] = field(default_factory=dict)
    manifests: list[dict[str, Any]] = field(default_factory=list)
    scenarios_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, Any] | None = None,
        data: bytes | Any | None = None,
        timeout: float = 30.0,
    ) -> ob.TransportResponse:
        del timeout
        self.calls.append({"method": method, "url": url, "headers": dict(headers)})
        if (
            self.fence_on_url_substring is not None
            and self.fence_on_url_substring in url
        ):
            self._fenced = True
        if (
            self.fence_on_terminal_event
            and "/events/" in url
            and method == "POST"
            and isinstance(data, (bytes, bytearray))
            and b'"type":"terminal"' in bytes(data)
        ):
            self._fenced = True
        if self._fenced or (
            self.fence_after is not None and len(self.calls) >= self.fence_after
        ):
            return ob.TransportResponse(
                status_code=403,
                body={
                    "error": "fenced",
                    "message": "attempt superseded",
                    "retryable": False,
                },
                headers={},
            )
        if "/events/" in url and method == "POST":
            body_bytes = data if isinstance(data, (bytes, bytearray)) else b""
            body = (
                json.loads(body_bytes.decode("utf-8")) if body_bytes else {"events": []}
            )
            events = body.get("events", [])
            self.event_records.extend(events)
            watermark = max((e["sequence"] for e in events), default=0)
            return ob.TransportResponse(
                200, {"acked_through_sequence": watermark, "rejected": []}, {}
            )
        if "/results/" in url and method == "POST" and json_body is not None:
            key = (json_body["job_id"], json_body["scenario_key"])
            existed = key in self.receipts
            self.receipts[key] = json_body
            return ob.TransportResponse(200 if existed else 201, {}, {})
        if url.endswith("/manifest/") and method == "POST" and json_body is not None:
            self.manifests.append(json_body)
            return ob.TransportResponse(200, {}, {})
        if "/artifacts/" in url and method == "PUT":
            digest = url.rstrip("/").rsplit("/", 1)[-1]
            payload = data if isinstance(data, (bytes, bytearray)) else b"".join(data)
            existed = digest in self.artifacts
            self.artifacts[digest] = bytes(payload)
            return ob.TransportResponse(200 if existed else 201, {}, {})
        if "/scenarios/" in url and method == "POST" and json_body is not None:
            # p13: Azain's real router mints exactly ONE url per attempt (a DRF detail `@action`,
            # no `url_path`) -- provision vs begin is a body-level `operation` field, never a URL
            # suffix, so routing here is on `json_body["operation"]`, not `url`.
            self.scenarios_calls.append((url, json_body))
            operation = json_body.get("operation")
            if operation == "provision":
                keys = [p.get("scenario_key") for p in json_body.get("personas", [])]
                scenarios = [
                    {"scenario_key": key, "scenario_id": f"platform-{key}"}
                    for key in keys
                ]
                return ob.TransportResponse(
                    200,
                    {"result": {"run_test_id": "run-test-1", "scenarios": scenarios}},
                    {},
                )
            if operation == "begin":
                return ob.TransportResponse(
                    200,
                    {"result": {"test_execution_id": "exec-1", "scenarios": []}},
                    {},
                )
            # No/unknown `operation` -- exercised by callers (e.g. `FakeScenarioSource`) that only
            # care about the call happening, never the response shape.
            return ob.TransportResponse(200, {"result": {"ok": True}}, {})
        return ob.TransportResponse(
            404,
            {
                "error": "not_found",
                "message": f"unmapped route: {url}",
                "retryable": False,
            },
            {},
        )

    def terminal_events(self) -> list[dict[str, Any]]:
        return [
            record for record in self.event_records if record.get("type") == "terminal"
        ]


# =================================================================================================
# Fake provisioner / world / scenario / call runner.
# =================================================================================================


class FakeProvisioner:
    """`_serialized` records any overlapping call into `overlaps` rather than asserting in-band --
    an in-band `assert not self._busy` fires INSIDE `WorldPool.lease()`'s own
    `except Exception as exc: reset_exc = exc` (a reset/healthy failure is expected there) or
    `_reconcile`'s `except Exception` (a reconcile must never crash the pool), so production
    swallows it and a caller driving this fake through `WorldPool` (rather than calling it
    directly) would never see the failure. Asserting on `overlaps` from the test's OWN frame is
    what actually makes a `_provider_lock` regression observable end-to-end (R7: this is the same
    fake `WorldPool`'s own test suite already exercises this way, now needed here too since
    `hosted_entrypoint.py` no longer wraps the provider in its own lock)."""

    name = "fake-process"  # mirrors `ProcessRuntimeProvider.name`.

    def __init__(self, instances: int = 1, *, always_unhealthy: bool = False) -> None:
        self.instances = instances
        self.always_unhealthy = always_unhealthy  # every healthy() probe fails.
        self.provision_calls = 0
        self.reset_calls = 0
        self.healthy_calls = 0
        self.closed = False
        self.overlaps: list[str] = []
        self._in_flight: list[str] = []
        self._runtimes = {
            i: EnvironmentRuntime(
                runtime_id=f"digest:w{i}",
                world_index=i,
                bundle_digest="digest",
                state=RuntimeState.READY,
                endpoints={},
            )
            for i in range(instances)
        }

    @contextlib.asynccontextmanager
    async def _serialized(self, label: str):
        if self._in_flight:
            self.overlaps.append(f"{self._in_flight[-1]} overlapped {label}")
        self._in_flight.append(label)
        try:
            await asyncio.sleep(0)
            yield
        finally:
            self._in_flight.remove(label)

    async def provision(
        self,
        bundle: Any,
        *,
        source: Path,
        bundle_dir: Path,
        work_directory: Path,
        contract: Any | None = None,
        instances: int = 1,
    ) -> list[EnvironmentRuntime]:
        del bundle, source, bundle_dir, work_directory, contract
        async with self._serialized("provision"):
            self.provision_calls += 1
            return [self._runtimes[i] for i in range(instances)]

    async def reset(self, runtime: EnvironmentRuntime, *, work_directory: Path) -> None:
        del work_directory
        async with self._serialized(f"reset(w{runtime.world_index})"):
            self.reset_calls += 1
            runtime.state = RuntimeState.READY

    async def healthy(
        self, runtime: EnvironmentRuntime, *, work_directory: Path
    ) -> bool:
        del work_directory
        # v1.12 folds `healthy` into the same non-reentrant set as provision/reset/close.
        async with self._serialized(f"healthy(w{runtime.world_index})"):
            self.healthy_calls += 1
            return not self.always_unhealthy

    async def close(self, *, work_directory: Path) -> None:
        del work_directory
        async with self._serialized("close"):
            self.closed = True


class FakeWorld:
    def __init__(self, world_index: int, rng: Any) -> None:
        self.world_index = world_index
        self.rng = rng

    def state(self, table: str | None = None) -> dict[str, list[dict[str, Any]]]:
        del table
        return {}

    def put(
        self, collection: str, record: dict[str, Any], *, key: str = ""
    ) -> dict[str, Any]:
        del collection, key
        return record

    def change(
        self, collection: str, key: str, changes: dict[str, Any], *, by: str = ""
    ) -> int:
        del collection, key, changes, by
        return 1

    def drop(self, collection: str, key: str = "", *, by: str = "") -> int:
        del collection, key, by
        return 1

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> Call:
        raise NotImplementedError

    def query(self, sql: str, params: Any = ()) -> list[dict[str, Any]]:
        del sql, params
        return []

    def read_only(self) -> "FakeWorld":
        return self


class FakeWorldFactory:
    async def create(self, runtime: EnvironmentRuntime, *, rng: Any) -> FakeWorld:
        return FakeWorld(runtime.world_index, rng)


@dataclass
class FakeSubGoal:
    name: str
    should_hold: bool
    # Empty: this fake carries a working check, so it is a CODED sub-goal. A real sub-goal is one
    # or the other -- `deterministic()` is `bool(check)` and scenario_source only marks `judged`
    # when no check file exists -- so a fake that claimed both routed itself to the judge and
    # never ran the check it was given.
    judged: str = ""

    def check(self, world: Any, calls: Any) -> object:
        del world, calls
        return None if self.should_hold else "the agent did not do it"


@dataclass
class FakeScenario:
    scenario_key: str
    scenario_id: str
    sub_goals: list[FakeSubGoal]

    def setup(self, world: Any) -> object:
        del world
        return None

    def ready(self, world: Any) -> object:
        del world
        return None


class FakeCallRunner:
    """Uploads a transcript through the adapter BEFORE returning, and can optionally trip the
    cancel signal the instant a named scenario starts (for the cancel-mid-run test)."""

    def __init__(
        self,
        adapter: he.OutboundAdapter,
        *,
        cancel_path: Path | None = None,
        cancel_on_scenario: str | None = None,
        cancel_reason: str = "user_canceled",
        delay_seconds: float = 0.05,
    ) -> None:
        self._adapter = adapter
        self._cancel_path = cancel_path
        self._cancel_on_scenario = cancel_on_scenario
        self._cancel_reason = cancel_reason
        self._delay_seconds = delay_seconds

    async def run(
        self, scenario: FakeScenario, runtime: EnvironmentRuntime
    ) -> CallOutcome:
        del runtime
        if (
            self._cancel_path is not None
            and scenario.scenario_key == self._cancel_on_scenario
        ):
            self._cancel_path.write_text(
                json.dumps({"reason": self._cancel_reason}), encoding="utf-8"
            )
        await asyncio.sleep(self._delay_seconds)
        transcript = json.dumps(
            [
                {
                    "speaker_role": "assistant",
                    "content": f"hello from {scenario.scenario_key}",
                }
            ]
        ).encode("utf-8")
        artifact_id = await self._adapter.upload_artifact(
            transcript,
            kind=ob.ArtifactKind.TRANSCRIPT,
            scenario_key=scenario.scenario_key,
        )
        now = _rfc3339(datetime.now(timezone.utc))
        return CallOutcome(
            calls=(
                Call(
                    name="tool",
                    arguments={},
                    result="ok",
                    ok=True,
                    error="",
                    refused=False,
                    at=0.0,
                ),
            ),
            turns=1,
            started_at=now,
            ended_at=now,
            duration_ms=10,
            transcript_artifact=artifact_id,
            recording_artifacts=(),
        )


def _rfc3339(value: datetime) -> str:
    return ob.format_rfc3339_millis(value)


class FakeScenarioSource:
    def __init__(self, scenarios: list[FakeScenario]) -> None:
        self._scenarios = scenarios

    async def build(
        self,
        job: HarnessJob,
        bundle: Any,
        scenarios_client: he.ScenariosClient,
        *,
        pool: Any,
        world_factory: Any,
        bundle_dir: Path,
    ) -> list[FakeScenario]:
        # `bundle_dir` (p12: the scenario-source adapter's own seam) is unused by this in-memory
        # fake -- accepted only because `run_job` now forwards it to every `ScenarioSource.build`,
        # injected or not.
        del job, bundle, pool, world_factory, bundle_dir
        # `operation` mirrors what `register_with_platform` (scenario_source.py) really sends --
        # this fake's own `scenario_keys`/`scenario_ids` bodies are otherwise arbitrary (never
        # parsed by `FakeTransport`, which only inspects `operation` to pick a response), kept only
        # so `test_scenarios_channel_uses_bearer_auth_never_api_key` and friends see two distinct,
        # non-empty POST bodies.
        await asyncio.to_thread(
            scenarios_client.provision,
            {
                "operation": "provision",
                "scenario_keys": [s.scenario_key for s in self._scenarios],
            },
        )
        await asyncio.to_thread(
            scenarios_client.begin, {"operation": "begin", "scenario_ids": {}}
        )
        return self._scenarios


# =================================================================================================
# Deps builder.
# =================================================================================================


@dataclass
class Harness:
    tmp: Path
    work: Path
    source: Path
    output: Path
    job_path: Path
    transport: FakeTransport
    provisioner: FakeProvisioner
    deps: he.HostedEntrypointDeps
    bundle_dir: Path


def _build_harness(
    *,
    scenarios: list[FakeScenario],
    fence_after: int | None = None,
    fence_on_url_substring: str | None = None,
    fence_on_terminal_event: bool = False,
    cancel_on_scenario: str | None = None,
    cancel_reason: str = "user_canceled",
    corrupt_bundle: Callable[[Path], None] | None = None,
    instances: int = 1,
    always_unhealthy: bool = False,
    parallelism: int = 1,
    build_output: dict[str, Any] | None = None,
    artifacts: HarnessArtifactPolicy | None = None,
    # p12: leaves `deps.scenario_source` at its real default (`NotWiredScenarioSource`) instead of
    # the usual `FakeScenarioSource` -- for the scenario_source.py wiring tests, which need
    # `run_job` to actually choose between the default and the real `BundleScenarioSource` itself.
    use_default_scenario_source: bool = False,
    # p12: swaps in `_write_bundle_with_scenario_files` for the scenario-source wiring tests, which
    # need real scenario documents hashed into the manifest's `files[]` (§2e `bundle_file_unlisted`
    # otherwise rejects them at preflight) -- every other caller keeps the plain `_write_bundle`.
    bundle_writer: Callable[[Path], Any] = _write_bundle,
) -> Harness:
    tmp = Path(tempfile.mkdtemp(prefix="p10-e2e-"))
    work = tmp / "work"
    source = work / "source"
    output = work / "artifacts"
    bundle_dir = work / he.DEFAULT_BUNDLE_DIR_NAME
    source.mkdir(parents=True, exist_ok=True)
    bundle_writer(bundle_dir)
    if corrupt_bundle is not None:
        corrupt_bundle(bundle_dir)

    job_path = tmp / "job.json"
    _write_job(job_path, _job(parallelism=parallelism, artifacts=artifacts))

    if build_output is not None:
        # `write_build_output` (process_runtime.py) writes here; no test previously did, so
        # the baseline_frozen/parallelism_degraded block ran only its
        # empty-dict fallback in every prior test.
        output.mkdir(parents=True, exist_ok=True)
        (output / "build.json").write_text(json.dumps(build_output), encoding="utf-8")

    capabilities = _capabilities()
    transport = FakeTransport(
        fence_after=fence_after,
        fence_on_url_substring=fence_on_url_substring,
        fence_on_terminal_event=fence_on_terminal_event,
    )
    provisioner = FakeProvisioner(
        instances=instances, always_unhealthy=always_unhealthy
    )

    cancel_path = tmp / "cancel.json"

    holder: dict[str, he.OutboundAdapter] = {}

    def build_call_runner(
        adapter: he.OutboundAdapter, context: he.CallRunnerContext
    ) -> FakeCallRunner:
        holder["adapter"] = adapter
        holder["call_runner_context"] = context
        return FakeCallRunner(
            adapter,
            cancel_path=cancel_path,
            cancel_on_scenario=cancel_on_scenario,
            cancel_reason=cancel_reason,
        )

    deps = he.HostedEntrypointDeps(
        load_capabilities=lambda: capabilities,
        bundle_source=he.DefaultBundleSource(),
        scenario_source=(
            he.NotWiredScenarioSource()
            if use_default_scenario_source
            else FakeScenarioSource(scenarios)
        ),
        build_transport=lambda: transport,
        build_provider=lambda _capabilities, _transport: provisioner,
        build_call_runner=build_call_runner,
        build_world_factory=lambda work_directory: FakeWorldFactory(),
        cancel_path=cancel_path,
        secrets_path=tmp / "secrets.json",
        install_sigterm_handler=lambda cancel_state: lambda: None,
        flush_window_seconds=5.0,
    )
    return Harness(
        tmp=tmp,
        work=work,
        source=source,
        output=output,
        job_path=job_path,
        transport=transport,
        provisioner=provisioner,
        deps=deps,
        bundle_dir=bundle_dir,
    )


def _run(harness: Harness) -> int:
    return asyncio.run(
        he.run_job(harness.job_path, harness.source, harness.output, deps=harness.deps)
    )


def _build_adapter(
    transport: FakeTransport, *, extra_secret_values: tuple[str, ...] = ()
) -> he.OutboundAdapter:
    """Adapter-only fixture for tests that drive `OutboundAdapter` directly (no `run_job`) --
    mirrors `test_redaction_end_to_end_secret_never_crosses_any_channel`'s own inline construction,
    factored out for reuse across other adapter-level tests."""
    capabilities = _capabilities()
    channel_state = ob.ChannelState()
    retry_policy = ob.RetryPolicy()
    tmp = Path(tempfile.mkdtemp(prefix="p10-adapter-"))
    events_spool = ob.OutboundSpool(tmp / "spool", "events", sequenced=True)
    events_client = ob.EventsClient(
        capabilities,
        events_spool,
        transport,
        retry_policy=retry_policy,
        channel_state=channel_state,
    )
    results_client = ob.ResultsClient(
        capabilities,
        transport,
        retry_policy=retry_policy,
        channel_state=channel_state,
    )
    artifacts_client = ob.ArtifactsClient(
        capabilities,
        transport,
        retry_policy=retry_policy,
        channel_state=channel_state,
    )
    return he.OutboundAdapter(
        capabilities,
        events_spool=events_spool,
        events_client=events_client,
        results_client=results_client,
        artifacts_client=artifacts_client,
        channel_state=channel_state,
        extra_secret_values=extra_secret_values,
    )


# =================================================================================================
# Pure-logic unit tests.
# =================================================================================================


def test_resolve_parallelism_reads_the_raw_value_without_clamping() -> None:
    # `RuntimeRequirements.parallelism` now exists -- `resolve_parallelism`
    # must return it RAW; clamping here would make `parallelism_out_of_range` unreachable.
    assert he.resolve_parallelism(_job(parallelism=1)) == 1
    assert he.resolve_parallelism(_job(parallelism=8)) == 8
    # An out-of-range value is preflight's to reject (§2e.7), not this function's to launder --
    # `RuntimeRequirements.parallelism` itself only enforces `ge=1`, so a too-large W passes model
    # validation and must still reach `resolve_parallelism` unclamped.
    assert he.resolve_parallelism(_job(parallelism=99)) == 99


def test_out_of_range_parallelism_is_rejected_by_preflight_not_clamped() -> None:
    # Confirms an out-of-range W reaches a
    # `parallelism_out_of_range` preflight rejection (§2e.7), never a silently clamped W=8 run.
    async def scenario() -> None:
        harness = _build_harness(scenarios=[], parallelism=20)
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK
        assert (
            harness.provisioner.provision_calls == 0
        )  # rejected before any provision, like §2e.
        terminals = harness.transport.terminal_events()
        assert len(terminals) == 1
        payload = terminals[0]["payload"]
        assert payload["failure"]["code"] == "parallelism_out_of_range"
        assert payload["failure"]["domain"] == "environment"
        assert payload["failure"]["stage"] == "validating_environment"

    asyncio.run(scenario())


def test_job_secret_purposes_maps_alias_to_purpose() -> None:
    job = _job()
    assert he.job_secret_purposes(job) == {TARGET_PROVIDER_ALIAS: "target_provider"}


def test_peek_secret_values_reads_without_deleting(
    tmp_path_factory: Path | None = None,
) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="p10-secrets-"))
    path = tmp / "secrets.json"
    path.write_text(
        json.dumps({"LIVEKIT_API_KEY": "sk-super-secret"}), encoding="utf-8"
    )
    values = he.peek_secret_values(path)
    assert values == ("sk-super-secret",)
    assert (
        path.exists()
    )  # non-destructive read -- the provisioner still owns load-and-delete.


def test_peek_secret_values_missing_file_is_empty() -> None:
    assert he.peek_secret_values(Path("/nonexistent/does-not-exist.json")) == ()


def test_load_simulator_secret_values_is_allowlisted_and_destructive(
    tmp_path: Path,
) -> None:
    path = tmp_path / "simulator-secrets.json"
    path.write_text(
        json.dumps(
            {
                "DEEPGRAM_API_KEY": "platform-deepgram",
                "GOOGLE_CLOUD_PROJECT": "platform-project",
                "LIVEKIT_URL": "wss://platform-livekit.example",
                "LIVEKIT_API_KEY": "platform-livekit-key",
                "LIVEKIT_API_SECRET": "platform-livekit-secret",
                "UNRELATED": "must-not-load",
            }
        ),
        encoding="utf-8",
    )

    values = he.load_simulator_secret_values(path)

    assert values == {
        "DEEPGRAM_API_KEY": "platform-deepgram",
        "GOOGLE_CLOUD_PROJECT": "platform-project",
        "LIVEKIT_URL": "wss://platform-livekit.example",
        "LIVEKIT_API_KEY": "platform-livekit-key",
        "LIVEKIT_API_SECRET": "platform-livekit-secret",
    }
    assert not path.exists()


def test_peek_target_provider_secret_values_filters_by_purpose_and_keeps_the_alias() -> (
    None
):
    tmp = Path(tempfile.mkdtemp(prefix="p14-secrets-"))
    path = tmp / "secrets.json"
    path.write_text(
        json.dumps(
            {
                "LIVEKIT_API_KEY": "lk-secret",
                "GITHUB_INSTALLATION_TOKEN": "gh-secret",
            }
        ),
        encoding="utf-8",
    )
    values = he.peek_target_provider_secret_values(
        path,
        {
            "LIVEKIT_API_KEY": "target_provider",
            "GITHUB_INSTALLATION_TOKEN": "source_checkout",
        },
    )
    assert values == {"LIVEKIT_API_KEY": "lk-secret"}
    assert (
        path.exists()
    )  # non-destructive read, same timing contract as peek_secret_values.


def test_peek_target_provider_secret_values_missing_file_is_empty() -> None:
    assert (
        he.peek_target_provider_secret_values(
            Path("/nonexistent/does-not-exist.json"),
            {"LIVEKIT_API_KEY": "target_provider"},
        )
        == {}
    )


def test_peek_target_provider_secret_values_drops_an_alias_with_no_purpose_entry() -> (
    None
):
    tmp = Path(tempfile.mkdtemp(prefix="p14-secrets-"))
    path = tmp / "secrets.json"
    path.write_text(json.dumps({"UNKNOWN_ALIAS": "value"}), encoding="utf-8")
    assert he.peek_target_provider_secret_values(path, {}) == {}


def test_peek_simulator_provider_secret_values_isolated_from_target_provider() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="simulator-secrets-"))
    path = tmp / "secrets.json"
    path.write_text(
        json.dumps(
            {
                "AGENT_ANTHROPIC_API_KEY": "customer-key",
                "SIMULATOR_DEEPGRAM_API_KEY": "platform-key",
            }
        ),
        encoding="utf-8",
    )
    purposes = {
        "AGENT_ANTHROPIC_API_KEY": "target_provider",
        "SIMULATOR_DEEPGRAM_API_KEY": "simulator_provider",
    }
    assert he.peek_simulator_provider_secret_values(path, purposes) == {
        "SIMULATOR_DEEPGRAM_API_KEY": "platform-key"
    }
    assert he.peek_target_provider_secret_values(path, purposes) == {
        "AGENT_ANTHROPIC_API_KEY": "customer-key"
    }


# =================================================================================================
# CallRunner wiring (p14): the extended build_call_runner(adapter, context) seam, and the
# NotWired-stays-the-fallback / real-CallRunnerImpl split on `agent.connector`.
# =================================================================================================


def _call_runner_context(
    *, job: HarnessJob | None = None, evidence_seam: Any = EvidenceSeam.HTTP_TOOL
) -> he.CallRunnerContext:
    return he.CallRunnerContext(
        job=job or _job(),
        bundle_dir=Path("/nonexistent/bundle"),
        work_directory=Path("/nonexistent/work"),
        evidence_seam=evidence_seam,
        target_provider_secret_values={},
        attempt_number=1,
    )


def test_default_build_call_runner_returns_real_runner_for_vapi() -> None:
    runner = he._default_build_call_runner(
        mock.Mock(), _call_runner_context(job=_job(connector="vapi"))
    )
    assert isinstance(runner, he.CallRunnerImpl)


def test_default_build_call_runner_returns_real_runner_for_retell() -> None:
    runner = he._default_build_call_runner(
        mock.Mock(), _call_runner_context(job=_job(connector="retell"))
    )
    assert isinstance(runner, he.CallRunnerImpl)


def test_default_build_call_runner_returns_notwired_for_unresolved_auto() -> None:
    runner = he._default_build_call_runner(
        mock.Mock(), _call_runner_context(job=_job(connector="auto"))
    )
    assert isinstance(runner, he.NotWiredCallRunner)


def test_default_build_call_runner_resolves_auto_voice_contract_to_livekit() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="auto-voice-contract-"))
    (tmp / "contract.json").write_text(
        json.dumps({"modality": "voice"}), encoding="utf-8"
    )
    context = _call_runner_context(job=_job(connector="auto"))
    context = he.CallRunnerContext(
        job=context.job,
        bundle_dir=tmp,
        work_directory=context.work_directory,
        evidence_seam=context.evidence_seam,
        target_provider_secret_values=context.target_provider_secret_values,
        attempt_number=context.attempt_number,
    )
    runner = he._default_build_call_runner(mock.Mock(), context)
    assert isinstance(runner, he.CallRunnerImpl)


def test_default_build_call_runner_returns_a_real_call_runner_impl_for_livekit() -> (
    None
):
    runner = he._default_build_call_runner(
        mock.Mock(), _call_runner_context(job=_job(connector="livekit"))
    )
    assert isinstance(runner, he.CallRunnerImpl)


def test_call_runner_context_is_threaded_with_real_job_bundle_secrets_and_evidence_seam() -> (
    None
):
    """End-to-end through the real `run_job` wiring point (~line 1728): `secret_purposes =
    job_secret_purposes(job)` runs, `deps.peek_target_provider_secret_values` captures the alias
    map BEFORE `pool.start()` deletes `secrets.json`, and the resulting `CallRunnerContext` reaches
    whatever factory `deps.build_call_runner` names -- verified by capturing it, not by asserting
    on `CallRunnerImpl` internals (that belongs to test_call_runner.py).

    `SecretDeletingProvisioner` mirrors the REAL `ProcessRuntimeProvider`'s own lifetime rule
    (process_runtime.py:3535-3544): `secrets.json` is deleted on the provisioner's FIRST
    `provision()` call, which `pool.start()` awaits synchronously. Without this, `FakeProvisioner`
    never touches the file at all, and a capture that happened AFTER `pool.start()` instead of
    before would still see the intact secrets file and still pass -- the deletion is what makes
    the capture's timing actually load-bearing here, not just documented in a comment."""

    class SecretDeletingProvisioner(FakeProvisioner):
        def __init__(self, *args: Any, secrets_path: Path, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._secrets_path = secrets_path

        async def provision(
            self, *args: Any, **kwargs: Any
        ) -> list[EnvironmentRuntime]:
            runtimes = await super().provision(*args, **kwargs)
            self._secrets_path.unlink(missing_ok=True)
            return runtimes

    harness = _build_harness(scenarios=[])
    harness.deps.secrets_path.parent.mkdir(parents=True, exist_ok=True)
    harness.deps.secrets_path.write_text(
        json.dumps({TARGET_PROVIDER_ALIAS: "lk-secret-value"}), encoding="utf-8"
    )
    harness.deps.simulator_secrets_path = harness.tmp / "simulator-secrets.json"
    harness.deps.simulator_secrets_path.write_text(
        json.dumps(
            {
                "DEEPGRAM_API_KEY": "platform-deepgram",
                "GOOGLE_CLOUD_PROJECT": "platform-project",
                "LIVEKIT_URL": "wss://platform-livekit.example",
                "LIVEKIT_API_KEY": "platform-livekit-key",
                "LIVEKIT_API_SECRET": "platform-livekit-secret",
            }
        ),
        encoding="utf-8",
    )
    harness.deps.build_provider = lambda _capabilities, _transport: (
        SecretDeletingProvisioner(instances=1, secrets_path=harness.deps.secrets_path)
    )

    captured: dict[str, he.CallRunnerContext] = {}

    def build_call_runner(
        adapter: he.OutboundAdapter, context: he.CallRunnerContext
    ) -> FakeCallRunner:
        captured["context"] = context
        return FakeCallRunner(adapter, cancel_path=harness.deps.cancel_path)

    harness.deps.build_call_runner = build_call_runner
    code = _run(harness)
    assert code == he.EXIT_OK

    context = captured["context"]
    assert context.job.job_id == "job-1"
    assert context.bundle_dir == harness.bundle_dir
    # `_base_manifest_body()` (this file's own bundle fixture) declares `evidence_seam: "http_tool"`.
    assert context.evidence_seam is EvidenceSeam.HTTP_TOOL
    assert context.attempt_number == 1
    # The load-bearing assertion: the secrets file is genuinely gone by the time `provision()`
    # (inside `pool.start()`) returns -- if the capture had happened AFTER `pool.start()` instead
    # of before, this map would be empty, not the real alias -> value pair.
    assert context.target_provider_secret_values == {
        TARGET_PROVIDER_ALIAS: "lk-secret-value"
    }
    assert context.simulator_provider_secret_values == {
        "DEEPGRAM_API_KEY": "platform-deepgram",
        "GOOGLE_CLOUD_PROJECT": "platform-project",
        "LIVEKIT_URL": "wss://platform-livekit.example",
        "LIVEKIT_API_KEY": "platform-livekit-key",
        "LIVEKIT_API_SECRET": "platform-livekit-secret",
    }
    assert not harness.deps.secrets_path.exists()
    assert not harness.deps.simulator_secrets_path.exists()


def test_row_counts_for_capability_returns_the_matching_store() -> None:
    build_output = {"stores": [{"capability": "database", "row_counts": {"riders": 3}}]}
    assert he.row_counts_for_capability(build_output, "database") == {"riders": 3}


def test_row_counts_for_capability_raises_when_the_capability_is_absent() -> None:
    build_output = {"stores": [{"capability": "other", "row_counts": {}}]}
    try:
        he.row_counts_for_capability(build_output, "database")
    except he.WorldFactoryError:
        pass
    else:
        raise AssertionError("expected WorldFactoryError")


def test_cancel_state_reads_reason_from_file() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="p10-cancel-"))
    path = tmp / "cancel.json"
    state = he.CancelState(path)
    assert state.requested() is False
    path.write_text(json.dumps({"reason": "ttl_exceeded"}), encoding="utf-8")
    assert state.requested() is True
    assert state.reason() is ob.TerminalReason.TTL_EXCEEDED


def test_world_pool_serializes_concurrent_provider_calls_end_to_end() -> None:
    # R7: `hosted_entrypoint.py` used to wrap every provider in `SerializingProvider` before
    # `WorldPool` ever saw it -- removed now that `WorldPool`'s own `_provider_lock` covers
    # provision/reset/healthy/close (mutation-verified in review: `healthy` rides the same
    # non-reentrancy rule as the other three, per v1.12 §4.5b). This repoints the old
    # wrapper-level test at the SAME guarantee, one level up: `pool.start()` (a `provision()` call)
    # racing `pool._reconcile()` (a `provision()` THEN a `healthy()` call) against the bare,
    # unwrapped `FakeProvisioner` this module now wires directly.
    async def scenario() -> None:
        fake = FakeProvisioner(instances=1)
        work = Path(tempfile.mkdtemp(prefix="p10-pool-serial-"))
        pool = WorldPool(
            fake,
            bundle=None,
            source=work,
            bundle_dir=work,
            work_directory=work,
            instances=1,
        )
        await asyncio.gather(pool.start(), pool._reconcile())
        # `overlaps` is populated OUT-OF-BAND by `FakeProvisioner._serialized()` -- an in-band
        # `assert` there would fire inside `_reconcile`'s own `except Exception` (a reconcile must
        # never crash the pool) and never reach this frame, silently passing a broken lock.
        assert fake.overlaps == []
        assert fake.provision_calls >= 1
        await pool.close()

    asyncio.run(scenario())


def test_scenarios_client_provision_unwraps_the_result_envelope() -> None:
    # p13: keyed response (`{"scenarios": [{"scenario_key", "scenario_id"}, ...]}`), matching the
    # real platform's `_provision_response` (services/hosted_harness.py:487-501) -- never a
    # position-ordered `scenario_ids` array.
    capabilities = _capabilities()
    transport = FakeTransport()
    client = he.ScenariosClient(capabilities, transport)
    result = client.provision(
        {
            "operation": "provision",
            "name": "run-1",
            "personas": [{"scenario_key": "a"}, {"scenario_key": "b"}],
        }
    )
    assert result == {
        "run_test_id": "run-test-1",
        "scenarios": [
            {"scenario_key": "a", "scenario_id": "platform-a"},
            {"scenario_key": "b", "scenario_id": "platform-b"},
        ],
    }


def test_scenarios_client_provision_and_begin_hit_the_same_single_url() -> None:
    # p13: Azain's router mints exactly ONE url per attempt (views/hosted_harness.py:78-90's
    # `scenarios` detail `@action`, no `url_path`; urls.py:128-132) -- `provision_path`/
    # `begin_path` default to an EMPTY suffix now (not the old guessed `"provision/"`/`"begin/"`,
    # which 404 against the real router), so both calls land on `capabilities.endpoints.scenarios`
    # itself, discriminated only by the body's `operation` field.
    capabilities = _capabilities()
    transport = FakeTransport()
    client = he.ScenariosClient(capabilities, transport)
    client.provision({"operation": "provision", "name": "run-1", "personas": []})
    client.begin(
        {"operation": "begin", "run_test_id": "run-test-1", "scenario_keys": []}
    )
    urls = [call["url"] for call in transport.calls]
    assert urls == [capabilities.endpoints.scenarios, capabilities.endpoints.scenarios]


def test_scenarios_client_fencing_latches_the_shared_channel_state() -> None:
    capabilities = _capabilities()
    transport = FakeTransport(fence_after=1)
    channel_state = ob.ChannelState()
    client = he.ScenariosClient(capabilities, transport, channel_state=channel_state)
    try:
        client.provision({"operation": "provision", "name": "run-1", "personas": []})
    except ob.HostedFencedError:
        pass
    else:
        raise AssertionError("expected HostedFencedError")
    try:
        channel_state.check()
    except ob.HostedFencedError:
        pass
    else:
        raise AssertionError(
            "channel_state should now be latched for every other channel too"
        )


# =================================================================================================
# Targeted orchestration tests.
# =================================================================================================


def test_capabilities_failure_exits_boot_failure_with_no_channel_and_no_event() -> None:
    async def scenario() -> None:
        tmp = Path(tempfile.mkdtemp(prefix="p10-boot-"))
        work = tmp / "work"
        source = work / "source"
        output = work / "artifacts"
        source.mkdir(parents=True, exist_ok=True)
        job_path = tmp / "job.json"
        _write_job(job_path, _job())

        def _raise() -> ob.HostedCapabilities:
            raise ob.CapabilitiesError("capabilities_file_missing", "no file")

        deps = he.HostedEntrypointDeps(load_capabilities=_raise)
        code = await he.run_job(job_path, source, output, deps=deps)
        assert code == he.EXIT_BOOT_FAILURE
        assert code != he.EXIT_FENCED
        assert not (
            work / he.EVENTS_SPOOL_DIR_NAME
        ).exists()  # no channel was ever built.

    asyncio.run(scenario())


def test_preflight_rejection_reaches_a_failed_terminal_event_before_any_provision() -> (
    None
):
    async def scenario() -> None:
        harness = _build_harness(
            scenarios=[],
            corrupt_bundle=lambda bundle_dir: (
                bundle_dir / "db" / "schema.sql"
            ).unlink(),
        )
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK
        assert harness.provisioner.provision_calls == 0  # "BEFORE any provision"
        terminals = harness.transport.terminal_events()
        assert len(terminals) == 1
        payload = terminals[0]["payload"]
        assert payload["stage"] == "failed"
        assert payload["failure"]["domain"] == "environment"
        assert payload["failure"]["stage"] == "validating_environment"
        assert payload["failure"]["code"] == "bundle_file_missing"

    asyncio.run(scenario())


def test_hosted_fenced_error_stops_emitting_and_exits_3_with_no_terminal_event() -> (
    None
):
    async def scenario() -> None:
        scenarios = [FakeScenario("s1", "platform-s1", [FakeSubGoal("holds", True)])]
        # Fenced mid-attempt (during the scenario's own receipt push), well after provisioning --
        # proves both halves of the contract: nothing further is ever emitted, AND close() still
        # runs (it is unconditional after scheduler.run(), not gated on fencing).
        harness = _build_harness(
            scenarios=scenarios, fence_on_url_substring="/results/"
        )
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_FENCED
        assert harness.transport.terminal_events() == []
        assert harness.provisioner.closed is True  # close() still runs on the way out.

    asyncio.run(scenario())


def test_cancel_mid_run_synthesizes_a_skipped_receipt_for_the_unstarted_scenario() -> (
    None
):
    async def scenario() -> None:
        order: list[str] = []
        scenarios = [
            FakeScenario("first", "platform-first", [FakeSubGoal("holds", True)]),
            FakeScenario("second", "platform-second", [FakeSubGoal("holds", True)]),
        ]

        class OrderTrackingTransport(FakeTransport):
            def request(
                self,
                method: str,
                url: str,
                *,
                headers: dict[str, str],
                json_body: dict[str, Any] | None = None,
                data: bytes | Any | None = None,
                timeout: float = 30.0,
            ) -> ob.TransportResponse:
                response = super().request(
                    method,
                    url,
                    headers=headers,
                    json_body=json_body,
                    data=data,
                    timeout=timeout,
                )
                if (
                    "/events/" in url
                    and method == "POST"
                    and isinstance(data, (bytes, bytearray))
                    and b'"type":"terminal"' in bytes(data)
                ):
                    order.append("terminal_delivered")
                if (
                    "/results/" in url
                    and method == "POST"
                    and json_body is not None
                    and json_body.get("status") == "skipped"
                ):
                    order.append("skipped_receipt_delivered")
                return response

        harness = _build_harness(
            scenarios=scenarios, cancel_on_scenario="first", instances=1
        )
        transport = OrderTrackingTransport()
        harness.deps.build_transport = lambda: transport
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK
        terminals = transport.terminal_events()
        assert len(terminals) == 1
        assert terminals[0]["payload"]["stage"] == "canceled"
        assert terminals[0]["payload"]["reason"] == "user_canceled"
        statuses = {key[1]: body["status"] for key, body in transport.receipts.items()}
        assert statuses.get("first") == "passed"
        assert statuses.get("second") == "skipped"

        # outbound-channels.md v1.3 Sequencing ("terminal event -> skipped receipts ->
        # manifest") -- the skipped receipt for "second" must reach the platform, and strictly
        # after the terminal event, never before or instead of it.
        assert order == ["terminal_delivered", "skipped_receipt_delivered"]

        # the "skipped" receipt body (exact) per outbound-channels.md's own six-field list --
        # not just its `status`.
        skipped_body = transport.receipts[("job-1", "second")]
        assert skipped_body["scenario_attempt"] == 1
        assert skipped_body["world_index"] is None
        assert skipped_body["sub_goals"] == []
        assert skipped_body["evaluations"] == []
        assert skipped_body["call"] is None
        assert skipped_body["failure"] is None

        # a CANCELED run is cut short -- the manifest must say so.
        assert transport.manifests[-1]["complete"] is False

    asyncio.run(scenario())


def test_fenced_run_result_emits_zero_skipped_receipts() -> None:
    # `RunResult.fenced` must gate the entrypoint's own call to
    # `emit_skipped_receipts` -- checked here at the call site itself, not left to the scheduler's
    # own internal no-op. `HostedScheduler.run` is monkeypatched to hand back a fenced `RunResult`
    # regardless of how the real run went, since the real `OutboundAdapter` has no path that lets
    # a channel fence reach `WorldPool.fenced` (it swallows `HostedFencedError` internally --
    # see `OutboundAdapter._guarded`), so this is the only reliable way to exercise the branch
    # end-to-end through `run_job`.
    async def scenario() -> None:
        scenarios = [
            FakeScenario("first", "platform-first", [FakeSubGoal("holds", True)]),
            FakeScenario("second", "platform-second", [FakeSubGoal("holds", True)]),
        ]
        harness = _build_harness(scenarios=scenarios, instances=1)

        fence_exc = ob.HostedFencedError(
            ob.ChannelError(
                ob.ChannelOutcome.FENCED, None, "fence_mismatch", "attempt superseded"
            )
        )
        original_run = HostedScheduler.run
        original_emit = HostedScheduler.emit_skipped_receipts
        emit_calls: list[RunResult] = []

        async def fenced_run(self: HostedScheduler, scns: Any) -> RunResult:
            real = await original_run(self, scns)
            return RunResult(
                receipts=real.receipts, aborted=real.aborted, fenced=fence_exc
            )

        async def counting_emit(self: HostedScheduler, result: RunResult) -> None:
            emit_calls.append(result)
            await original_emit(self, result)

        HostedScheduler.run = fenced_run  # type: ignore[method-assign]
        HostedScheduler.emit_skipped_receipts = counting_emit  # type: ignore[method-assign]
        try:
            code = await he.run_job(
                harness.job_path, harness.source, harness.output, deps=harness.deps
            )
        finally:
            HostedScheduler.run = original_run
            HostedScheduler.emit_skipped_receipts = original_emit

        assert code == he.EXIT_OK
        assert emit_calls == []

    asyncio.run(scenario())


def test_emit_skipped_receipts_failure_does_not_lose_the_terminal_or_the_exit_code() -> (
    None
):
    # a failure inside the post-terminal `scheduler.emit_skipped_receipts` call must be
    # swallowed locally (matching every other best-effort post-terminal emission in this module),
    # never mask the terminal already delivered or flip the exit code.
    async def scenario() -> None:
        scenarios = [
            FakeScenario("first", "platform-first", [FakeSubGoal("holds", True)]),
            FakeScenario("second", "platform-second", [FakeSubGoal("holds", True)]),
        ]
        harness = _build_harness(
            scenarios=scenarios, cancel_on_scenario="first", instances=1
        )

        original_emit = HostedScheduler.emit_skipped_receipts

        async def poisoned_emit(self: HostedScheduler, result: RunResult) -> None:
            raise RuntimeError("synthetic emit_skipped_receipts failure")

        HostedScheduler.emit_skipped_receipts = poisoned_emit  # type: ignore[method-assign]
        try:
            code = await he.run_job(
                harness.job_path, harness.source, harness.output, deps=harness.deps
            )
        finally:
            HostedScheduler.emit_skipped_receipts = original_emit

        assert code == he.EXIT_OK
        terminals = harness.transport.terminal_events()
        assert len(terminals) == 1
        assert terminals[0]["payload"]["stage"] == "canceled"
        assert terminals[0]["payload"]["reason"] == "user_canceled"
        statuses = {
            key[1]: body["status"] for key, body in harness.transport.receipts.items()
        }
        assert statuses.get("first") == "passed"

    asyncio.run(scenario())


def test_cancel_mid_run_with_ttl_exceeded_reports_that_reason() -> None:
    # the exit-code table's CANCELED branch is only ever exercised with `user_canceled`
    # elsewhere in this file -- `ttl_exceeded` is the contract's other legal reason
    # (`ob.TerminalReason`) and goes through the exact same `CancelState.reason()` path.
    async def scenario() -> None:
        scenarios = [
            FakeScenario("first", "platform-first", [FakeSubGoal("holds", True)]),
            FakeScenario("second", "platform-second", [FakeSubGoal("holds", True)]),
        ]
        harness = _build_harness(
            scenarios=scenarios,
            cancel_on_scenario="first",
            cancel_reason="ttl_exceeded",
            instances=1,
        )
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK
        terminals = harness.transport.terminal_events()
        assert len(terminals) == 1
        assert terminals[0]["payload"]["stage"] == "canceled"
        assert terminals[0]["payload"]["reason"] == "ttl_exceeded"

    asyncio.run(scenario())


def test_malformed_job_json_exits_crashed_with_no_terminal_event() -> None:
    # EXIT_CRASHED -- a malformed job.json exits non-zero with no channel of its own
    # kind (capabilities load already succeeded, but nothing typed can describe the failure).
    async def scenario() -> None:
        tmp = Path(tempfile.mkdtemp(prefix="p10-badjob-"))
        work = tmp / "work"
        source = work / "source"
        output = work / "artifacts"
        source.mkdir(parents=True, exist_ok=True)
        job_path = tmp / "job.json"
        job_path.write_text("{this is not valid json", encoding="utf-8")

        capabilities = _capabilities()
        transport = FakeTransport()
        deps = he.HostedEntrypointDeps(
            load_capabilities=lambda: capabilities,
            build_transport=lambda: transport,
            secrets_path=tmp / "secrets.json",
            install_sigterm_handler=lambda cancel_state: lambda: None,
        )
        code = await he.run_job(job_path, source, output, deps=deps)
        assert code == he.EXIT_CRASHED
        assert code != he.EXIT_FENCED
        assert transport.terminal_events() == []

    asyncio.run(scenario())


def test_world_pool_exhaustion_reaches_a_failed_terminal_world_pool_exhausted() -> None:
    # `RunResult.aborted` -> a terminal FAILED with domain `infrastructure`, stage
    # `running`, code `world_pool_exhausted`. Driven for real: the single world never passes its
    # `healthy()` probe, so `WorldPool.lease()` exhausts its reconcile budget and raises
    # `NoWorldsAvailable`, which `HostedScheduler.run()` turns into `RunResult.aborted`.
    async def scenario() -> None:
        scenarios = [FakeScenario("s1", "platform-s1", [FakeSubGoal("holds", True)])]
        harness = _build_harness(
            scenarios=scenarios, instances=1, always_unhealthy=True
        )
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK
        terminals = harness.transport.terminal_events()
        assert len(terminals) == 1
        payload = terminals[0]["payload"]
        assert payload["stage"] == "failed"
        assert payload["failure"]["domain"] == "infrastructure"
        assert payload["failure"]["stage"] == "running"
        assert payload["failure"]["code"] == "world_pool_exhausted"
        # Failure is not cancellation: all errored/skipped receipts and terminal artifacts were
        # emitted, so the evidence manifest is complete even though the outcome failed.
        assert harness.transport.manifests[-1]["complete"] is True
        assert harness.provisioner.closed is True

    asyncio.run(scenario())


def test_fence_landing_on_the_final_drain_still_exits_fenced() -> None:
    # a fence that 403s specifically the events
    # POST carrying the terminal event (not an earlier one) must still exit 3 with no terminal
    # event DELIVERED, never a stale pre-drain fence check that misses it.
    async def scenario() -> None:
        scenarios = [FakeScenario("s1", "platform-s1", [FakeSubGoal("holds", True)])]
        harness = _build_harness(
            scenarios=scenarios, instances=1, fence_on_terminal_event=True
        )
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_FENCED
        assert harness.transport.terminal_events() == []
        assert harness.provisioner.closed is True

    asyncio.run(scenario())


def test_fence_from_scenarios_client_exits_fenced_not_crashed() -> None:
    # `ScenariosClient._post` re-raises `HostedFencedError` after latching --
    # this must reach `run_job`'s typed handler around `scenario_source.build()`, not fall through
    # to a bare `except Exception` (exit 1, and the world pool leaked).
    async def scenario() -> None:
        harness = _build_harness(
            scenarios=[], instances=1, fence_on_url_substring="/scenarios/"
        )
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_FENCED
        assert harness.transport.terminal_events() == []
        assert (
            harness.provisioner.closed is True
        )  # the pool must not leak on this path either.

    asyncio.run(scenario())


def test_scenarios_channel_uses_bearer_auth_never_api_key() -> None:
    # outbound-channels.md calls out bearer + X-Harness-Fence by name for pre-allocation --
    # never `X-Api-Key`.
    async def scenario() -> None:
        scenarios = [FakeScenario("s1", "platform-s1", [FakeSubGoal("holds", True)])]
        harness = _build_harness(scenarios=scenarios, instances=1)
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK
        scenarios_calls = [
            call for call in harness.transport.calls if "/scenarios/" in call["url"]
        ]
        assert scenarios_calls, "expected at least one call against endpoints.scenarios"
        for call in scenarios_calls:
            assert call["headers"].get("Authorization", "").startswith("Bearer ")
            assert "X-Api-Key" not in call["headers"]
            assert "X-Harness-Fence" in call["headers"]

    asyncio.run(scenario())


def test_process_world_factory_raises_when_no_postgres_endpoint() -> None:
    # `ProcessWorldFactory`/`_find_postgres_endpoint` had zero coverage -- only the pure
    # `row_counts_for_capability` helper was unit-tested.
    async def scenario() -> None:
        tmp = Path(tempfile.mkdtemp(prefix="p10-wf-endpoint-"))
        factory = he.ProcessWorldFactory(tmp)
        runtime = EnvironmentRuntime(
            runtime_id="digest:w0",
            world_index=0,
            bundle_digest="digest",
            state=RuntimeState.READY,
            endpoints={},
        )
        try:
            await factory.create(runtime, rng=random.Random(0))
        except he.WorldFactoryError:
            pass
        else:
            raise AssertionError(
                "expected WorldFactoryError for a runtime with no postgres endpoint"
            )

    asyncio.run(scenario())


def test_process_world_factory_raises_when_build_json_has_no_matching_store() -> None:
    async def scenario() -> None:
        tmp = Path(tempfile.mkdtemp(prefix="p10-wf-store-"))
        (tmp / "artifacts").mkdir(parents=True, exist_ok=True)
        (tmp / "artifacts" / "build.json").write_text(
            json.dumps({"stores": [{"capability": "other", "row_counts": {}}]}),
            encoding="utf-8",
        )
        factory = he.ProcessWorldFactory(tmp)
        endpoint = RuntimeEndpoint(
            capability="database",
            protocol="postgres",
            address="postgresql://u:p@localhost/db",
        )
        runtime = EnvironmentRuntime(
            runtime_id="digest:w0",
            world_index=0,
            bundle_digest="digest",
            state=RuntimeState.READY,
            endpoints={"database": endpoint},
        )
        try:
            await factory.create(runtime, rng=random.Random(0))
        except he.WorldFactoryError:
            pass
        else:
            raise AssertionError(
                "expected WorldFactoryError for a build.json with no matching store"
            )

    asyncio.run(scenario())


def test_build_json_two_stores_emit_two_baseline_frozen_events() -> None:
    # Each store in build.json's stores list gets its own baseline_frozen event, not just the first.
    async def scenario() -> None:
        build_output = {
            "stores": [
                {
                    "capability": "database",
                    "baseline_reference": "ref-database",
                    "inputs_digest": "digest-database",
                },
                {
                    "capability": "cache",
                    "baseline_reference": "ref-cache",
                    "inputs_digest": "digest-cache",
                },
            ],
        }
        harness = _build_harness(scenarios=[], instances=1, build_output=build_output)
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK
        baseline_events = [
            record
            for record in harness.transport.event_records
            if record.get("type") == "baseline_frozen"
        ]
        assert len(baseline_events) == 2
        refs = {event["payload"]["baseline_ref"] for event in baseline_events}
        assert refs == {"ref-database", "ref-cache"}

    asyncio.run(scenario())


def test_build_json_degrade_payload_matches_the_recorded_values() -> None:
    # `parallelism_degraded`'s payload must mirror build.json's own requested/effective/reason values exactly.
    async def scenario() -> None:
        build_output = {
            "requested_parallelism": 2,
            "effective_parallelism": 1,
            "degrade_reason": "conformance_gate_failed",
        }
        harness = _build_harness(scenarios=[], instances=1, build_output=build_output)
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK
        degrade_events = [
            record
            for record in harness.transport.event_records
            if record.get("type") == "parallelism_degraded"
        ]
        assert len(degrade_events) == 1
        payload = degrade_events[0]["payload"]
        assert payload == {
            "requested": 2,
            "effective": 1,
            "reason": "conformance_gate_failed",
        }

    asyncio.run(scenario())


def test_build_json_fixed_port_at_w1_does_not_crash() -> None:
    # `requested == effective == 1` with
    # `degrade_reason: fixed_port` is not representable as a `parallelism_degraded` event
    # (`1 <= effective < requested` fails); this must degrade to a `log`, never crash the run.
    async def scenario() -> None:
        build_output = {
            "requested_parallelism": 1,
            "effective_parallelism": 1,
            "degrade_reason": "fixed_port",
        }
        scenarios = [FakeScenario("s1", "platform-s1", [FakeSubGoal("holds", True)])]
        harness = _build_harness(
            scenarios=scenarios, instances=1, build_output=build_output
        )
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK
        assert (
            harness.provisioner.closed is True
        )  # the pool must not be orphaned by a crash.
        degrade_events = [
            record
            for record in harness.transport.event_records
            if record.get("type") == "parallelism_degraded"
        ]
        assert (
            degrade_events == []
        )  # not representable -- a log event carries it instead.
        log_events = [
            record
            for record in harness.transport.event_records
            if record.get("type") == "log"
        ]
        assert any(
            "fixed_port" in record["payload"]["message"] for record in log_events
        )
        # a substring shared with the pydantic error text the blanket `except Exception`
        # would ALSO produce if the `effective < requested` guard were reverted -- this is the one
        # phrase that only the guard's own `else` branch ever writes, so it is what actually tells
        # the guard apart from the catch-all swallowing a ValidationError.
        assert any(
            "no parallelism_degraded event is representable"
            in record["payload"]["message"]
            for record in log_events
        )
        terminals = harness.transport.terminal_events()
        assert len(terminals) == 1
        assert terminals[0]["payload"]["stage"] == "completed"

    asyncio.run(scenario())


def test_e2e_two_scenarios_one_pass_one_fail_reaches_completed_and_exits_0() -> None:
    async def scenario() -> None:
        scenarios = [
            FakeScenario("passing", "platform-passing", [FakeSubGoal("holds", True)]),
            FakeScenario("failing", "platform-failing", [FakeSubGoal("holds", False)]),
        ]
        harness = _build_harness(scenarios=scenarios, instances=1)
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK

        terminals = harness.transport.terminal_events()
        assert len(terminals) == 1
        payload = terminals[0]["payload"]
        assert payload["stage"] == "completed"
        assert payload["reason"] is None
        assert payload["failure"] is None
        assert payload["scenario_counts"] == {
            "passed": 1,
            "failed": 1,
            "errored": 0,
            "skipped": 0,
        }

        statuses = {
            key[1]: body["status"] for key, body in harness.transport.receipts.items()
        }
        assert statuses == {"passing": "passed", "failing": "failed"}

        # both receipts reference an already-uploaded, already-acked transcript artifact.
        for key, body in harness.transport.receipts.items():
            del key
            transcript = body["call"]["transcript_artifact"]
            assert transcript is not None
            assert transcript.split(":", 1)[1] in harness.transport.artifacts

        assert len(harness.transport.manifests) >= 1
        assert harness.transport.manifests[-1]["complete"] is True
        kinds = {entry["kind"] for entry in harness.transport.manifests[-1]["entries"]}
        assert {"build", "result", "log", "transcript"} <= kinds

        assert harness.provisioner.provision_calls >= 1
        assert harness.provisioner.closed is True

        # Scenario pre-allocation (item 3) actually ran against endpoints.scenarios -- p13: a
        # single url, discriminated by the body's `operation` field (never a `/provision/`/
        # `/begin/` url suffix, which 404s against the real platform router).
        assert any(
            body.get("operation") == "provision"
            for _, body in harness.transport.scenarios_calls
        )
        assert any(
            body.get("operation") == "begin"
            for _, body in harness.transport.scenarios_calls
        )

    asyncio.run(scenario())


# =================================================================================================
# Additional tests closing mutation-honesty gaps -- cases where mutating the guarded code paths
# would have passed the suite unnoticed.
# =================================================================================================


def test_pool_close_backstop_runs_even_when_scenario_source_raises_untyped() -> None:
    # deleting the top-level `finally: pool.close()` backstop passed 29/29 -- every prior
    # test's exception path had an explicit close ahead of it. `MemoryError` matches none of
    # `run_job`'s typed handlers around `scenario_source.build()`, so it propagates straight past
    # the `finally` with no explicit close anywhere on this path -- only the backstop can close it.
    async def scenario() -> None:
        class ExplodingScenarioSource:
            async def build(
                self, job, bundle, scenarios_client, *, pool, world_factory, bundle_dir
            ):
                del job, bundle, scenarios_client, pool, world_factory, bundle_dir
                raise MemoryError("boom")

        harness = _build_harness(scenarios=[], instances=1)
        harness.deps.scenario_source = ExplodingScenarioSource()
        raised = False
        try:
            await he.run_job(
                harness.job_path, harness.source, harness.output, deps=harness.deps
            )
        except MemoryError:
            raised = True
        assert raised, "expected the untyped exception to propagate past run_job"
        assert (
            harness.provisioner.closed is True
        )  # only the finally backstop could have done this

    asyncio.run(scenario())


def test_process_runtime_error_uses_the_carried_domain_over_the_fallback_map() -> None:
    # v1.15 §2f: the producer resolves and carries `domain` at the raise site -- `spawn_failed`'s
    # managed/source split is no longer re-derived from a manifest lookup here (mutation: force
    # `_process_runtime_error_domain` back to ignoring `exc.domain` and this test catches it, since
    # the SAME code with two different carried domains would then collapse to one fallback value).
    async def run_case(
        code: str, domain: FailureDomain | None, expected_domain: str
    ) -> None:
        class RaisingProvisioner(FakeProvisioner):
            async def provision(
                self,
                bundle: Any,
                *,
                source: Path,
                bundle_dir: Path,
                work_directory: Path,
                contract: Any | None = None,
                instances: int = 1,
            ) -> list[EnvironmentRuntime]:
                del bundle, source, bundle_dir, work_directory, contract, instances
                raise ProcessRuntimeError(
                    "build", code, "synthetic failure", domain=domain
                )

        harness = _build_harness(scenarios=[], instances=1)
        harness.deps.build_provider = lambda _capabilities, _transport: (
            RaisingProvisioner(instances=1)
        )
        result = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert result == he.EXIT_OK
        terminals = harness.transport.terminal_events()
        assert len(terminals) == 1
        failure = terminals[0]["payload"]["failure"]
        assert failure["domain"] == expected_domain
        assert failure["stage"] == "building_environment"

    # The same code, two different CARRIED domains -- proves the domain is read off the
    # exception, not re-derived from `code`/`process` (the map alone could never distinguish
    # these two cases, since both are `spawn_failed`).
    asyncio.run(
        run_case("spawn_failed", FailureDomain.AGENT, "agent")
    )  # source, carried
    asyncio.run(
        run_case("spawn_failed", FailureDomain.INFRASTRUCTURE, "infrastructure")
    )  # managed, carried
    # No carried domain (as a raise site outside this module's control might produce) -- the
    # closed map is consulted as a fallback only.
    asyncio.run(run_case("build_failed", None, "agent"))
    asyncio.run(run_case("seed_failed", None, "environment"))


def test_scenario_entry_missing_scenario_key_fails_cleanly_never_an_attributeerror() -> (
    None
):
    # karthik-integration-changes.md K1: the Scenario Generation Contract's own model may not
    # carry `scenario_key` yet, and `hosted_scheduler.py` reads `scenario.scenario_key` with plain
    # attribute access -- an entry that lacks the field entirely used to raise AttributeError deep
    # in the scheduler (no terminal event, a crash exit code) instead of failing the job cleanly.
    class ScenarioMissingKey:
        scenario_id = "id-x"
        sub_goals: list[Any] = []

        def setup(self, world: Any) -> object:
            del world
            return None

        def ready(self, world: Any) -> object:
            del world
            return None

    class RawScenarioSource:
        """Returns entries verbatim -- unlike `FakeScenarioSource`, never reads `.scenario_key`
        itself before handing them back, so the entrypoint's own validation is what is under
        test, not this fixture crashing first."""

        def __init__(self, scenarios: list[Any]) -> None:
            self._scenarios = scenarios

        async def build(
            self,
            job: Any,
            bundle: Any,
            scenarios_client: he.ScenariosClient,
            *,
            pool: Any,
            world_factory: Any,
            bundle_dir: Path,
        ) -> list[Any]:
            del job, bundle, scenarios_client, pool, world_factory, bundle_dir
            return self._scenarios

    async def scenario() -> None:
        harness = _build_harness(scenarios=[], instances=1)
        harness.deps.scenario_source = RawScenarioSource([ScenarioMissingKey()])
        result = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert result == he.EXIT_OK
        terminals = harness.transport.terminal_events()
        assert len(terminals) == 1
        failure = terminals[0]["payload"]["failure"]
        assert failure["stage"] == "validating_scenarios"
        assert failure["domain"] == "environment"
        assert "scenario_key" in failure["message"]

    asyncio.run(scenario())


def test_finish_emits_the_terminal_before_closing_the_pool() -> None:
    # moving `_bounded_close()` before `emit_terminal()` inside `_finish` passed 29/29 --
    # nothing recorded the RELATIVE order of the two. This records both against one shared timeline.
    async def scenario() -> None:
        order: list[str] = []
        tmp = Path(tempfile.mkdtemp(prefix="p10-order-"))
        work = tmp / "work"
        source = work / "source"
        output = work / "artifacts"
        bundle_dir = work / he.DEFAULT_BUNDLE_DIR_NAME
        source.mkdir(parents=True, exist_ok=True)
        _write_bundle(bundle_dir)
        job_path = tmp / "job.json"
        _write_job(job_path, _job(parallelism=1))
        capabilities = _capabilities()

        class OrderTrackingTransport(FakeTransport):
            def request(
                self,
                method: str,
                url: str,
                *,
                headers: dict[str, str],
                json_body: dict[str, Any] | None = None,
                data: bytes | Any | None = None,
                timeout: float = 30.0,
            ) -> ob.TransportResponse:
                response = super().request(
                    method,
                    url,
                    headers=headers,
                    json_body=json_body,
                    data=data,
                    timeout=timeout,
                )
                if (
                    "/events/" in url
                    and method == "POST"
                    and isinstance(data, (bytes, bytearray))
                    and b'"type":"terminal"' in bytes(data)
                ):
                    order.append("terminal_delivered")
                return response

        class OrderTrackingProvisioner(FakeProvisioner):
            async def close(self, *, work_directory: Path) -> None:
                await super().close(work_directory=work_directory)
                order.append("pool_closed")

        transport = OrderTrackingTransport()
        provisioner = OrderTrackingProvisioner(instances=1)
        deps = he.HostedEntrypointDeps(
            load_capabilities=lambda: capabilities,
            bundle_source=he.DefaultBundleSource(),
            scenario_source=FakeScenarioSource([]),
            build_transport=lambda: transport,
            build_provider=lambda _capabilities, _transport: provisioner,
            build_world_factory=lambda work_directory: FakeWorldFactory(),
            cancel_path=tmp / "cancel.json",
            secrets_path=tmp / "secrets.json",
            install_sigterm_handler=lambda cancel_state: lambda: None,
            flush_window_seconds=5.0,
        )
        code = await he.run_job(job_path, source, output, deps=deps)
        assert code == he.EXIT_OK
        assert order == ["terminal_delivered", "pool_closed"]

    asyncio.run(scenario())


def test_redaction_end_to_end_secret_never_crosses_any_channel() -> None:
    # dropping the `extra_secret_values` threading (event_builder AND build_result_receipt),
    # or the two pre-existing `redact_outbound_text` calls (world_unhealthy.cause,
    # receipt.failure.message), all passed 29/29 -- nothing pinned redaction on the wire. Drives the
    # secret through a log message, world_unhealthy, a terminal failure, and a receipt failure --
    # the free-text fields the contract names -- and reads what actually reached the transport.
    async def scenario() -> None:
        secret = "sk-live-eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        capabilities = _capabilities()
        transport = FakeTransport()
        channel_state = ob.ChannelState()
        retry_policy = ob.RetryPolicy()
        tmp = Path(tempfile.mkdtemp(prefix="p10-redact-"))
        events_spool = ob.OutboundSpool(tmp / "spool", "events", sequenced=True)
        events_client = ob.EventsClient(
            capabilities,
            events_spool,
            transport,
            retry_policy=retry_policy,
            channel_state=channel_state,
        )
        results_client = ob.ResultsClient(
            capabilities,
            transport,
            retry_policy=retry_policy,
            channel_state=channel_state,
        )
        artifacts_client = ob.ArtifactsClient(
            capabilities,
            transport,
            retry_policy=retry_policy,
            channel_state=channel_state,
        )
        adapter = he.OutboundAdapter(
            capabilities,
            events_spool=events_spool,
            events_client=events_client,
            results_client=results_client,
            artifacts_client=artifacts_client,
            channel_state=channel_state,
            extra_secret_values=(secret,),
        )

        await adapter.log(level="error", message=f"boom: {secret}")
        await adapter.world_unhealthy(world_index=0, cause=f"probe failed: {secret}")
        await adapter.receipt(
            ResultReceipt(
                scenario_key="s1",
                scenario_id="platform-s1",
                scenario_attempt=1,
                world_index=0,
                status="errored",
                sub_goals=(),
                evaluations=(),
                call=None,
                failure=ReceiptFailure(
                    domain="agent",
                    stage="running",
                    code="call_failed",
                    message=f"failed calling out with {secret}",
                ),
            )
        )
        await adapter.emit_terminal(
            stage=HarnessStage.FAILED,
            failure={
                "domain": "infrastructure",
                "stage": "building_environment",
                "code": "provision_failed",
                "message": f"connection failed: {secret}",
            },
        )
        await adapter.drain(complete=True)

        for record in transport.event_records:
            assert secret not in json.dumps(record)
        for body in transport.receipts.values():
            assert secret not in json.dumps(body)

        log_events = [r for r in transport.event_records if r.get("type") == "log"]
        assert any("***" in r["payload"]["message"] for r in log_events)
        world_unhealthy_events = [
            r for r in transport.event_records if r.get("type") == "world_unhealthy"
        ]
        assert any("***" in r["payload"]["cause"] for r in world_unhealthy_events)
        terminal = transport.terminal_events()[0]
        assert "***" in terminal["payload"]["failure"]["message"]
        receipt_body = transport.receipts[("job-1", "s1")]
        assert "***" in receipt_body["failure"]["message"]

    asyncio.run(scenario())


def test_drain_loops_past_a_backlog_larger_than_one_batch_and_still_delivers_the_terminal() -> (
    None
):
    # A backlog bigger
    # than one `EVENTS_MAX_BATCH` (100) previously stranded the terminal event, the highest
    # sequence, while still exiting 0. A call runner that logs 260 chatter events before returning
    # reproduces the same shape end to end.
    async def scenario() -> None:
        class ChattyCallRunner:
            def __init__(self, adapter: he.OutboundAdapter, *, log_count: int) -> None:
                self._adapter = adapter
                self._log_count = log_count

            async def run(
                self, scenario: FakeScenario, runtime: EnvironmentRuntime
            ) -> CallOutcome:
                del runtime
                for i in range(self._log_count):
                    await self._adapter.log(level="info", message=f"chatter {i}")
                now = _rfc3339(datetime.now(timezone.utc))
                return CallOutcome(
                    calls=(
                        Call(
                            name="tool",
                            arguments={},
                            result="ok",
                            ok=True,
                            error="",
                            refused=False,
                            at=0.0,
                        ),
                    ),
                    turns=1,
                    started_at=now,
                    ended_at=now,
                    duration_ms=10,
                    transcript_artifact=None,
                    recording_artifacts=(),
                )

        scenarios = [FakeScenario("s1", "platform-s1", [FakeSubGoal("holds", True)])]
        harness = _build_harness(scenarios=scenarios, instances=1)
        harness.deps.build_call_runner = lambda adapter, context: ChattyCallRunner(
            adapter, log_count=260
        )
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK

        terminals = harness.transport.terminal_events()
        assert len(terminals) == 1  # must not be stranded behind the backlog

        chatter_logs = [
            record
            for record in harness.transport.event_records
            if record.get("type") == "log"
            and "chatter" in record["payload"].get("message", "")
        ]
        assert (
            len(chatter_logs) == 260
        )  # the WHOLE backlog drained, not just the first batch

    asyncio.run(scenario())


def test_call_aborted_with_no_ended_at_still_produces_a_receipt() -> None:
    # `CallAborted.partial.ended_at` is legitimately `None` (the
    # call started but never finished). Before the fix, `build_result_receipt` raised inside
    # `OutboundAdapter.receipt()` (outbound.CallSummary.ended_at is a required str), swallowed by
    # `HostedScheduler._emit`'s blanket `except Exception` -- the scenario reached the wire with NO
    # receipt at all despite `terminal.scenario_counts` claiming one `errored`.
    async def scenario() -> None:
        class AbortingCallRunner:
            async def run(
                self, scenario: FakeScenario, runtime: EnvironmentRuntime
            ) -> CallOutcome:
                del runtime
                now = _rfc3339(datetime.now(timezone.utc))
                raise CallAborted(
                    "ran out of time before the call finished",
                    partial=CallOutcome(
                        calls=(),
                        turns=1,
                        started_at=now,
                        ended_at=None,
                        duration_ms=10,
                        transcript_artifact=None,
                        recording_artifacts=(),
                    ),
                )

        scenarios = [FakeScenario("s1", "platform-s1", [FakeSubGoal("holds", True)])]
        harness = _build_harness(scenarios=scenarios, instances=1)
        harness.deps.build_call_runner = lambda adapter, context: AbortingCallRunner()
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK

        statuses = {
            key[1]: body["status"] for key, body in harness.transport.receipts.items()
        }
        assert statuses.get("s1") == "errored"
        receipt_body = harness.transport.receipts[("job-1", "s1")]
        assert receipt_body["call"] is not None
        assert receipt_body["call"]["ended_at"] == receipt_body["call"]["started_at"]

        terminals = harness.transport.terminal_events()
        assert len(terminals) == 1
        assert terminals[0]["payload"]["scenario_counts"]["errored"] == 1

    asyncio.run(scenario())


# =================================================================================================
# Additional tests: artifact-level admission coverage, terminal-delivery/receipt-rejection/
# message-capping edge cases, and mutation-survivor gaps around cancellation and the CANCELED manifest.
# =================================================================================================


def test_metadata_only_artifact_level_refuses_transcript_upload_end_to_end() -> None:
    # `_ARTIFACT_LEVEL_FORBIDDEN_KINDS` had zero suite coverage -- emptying the table passed
    # every test. Drives a real `metadata-only` job through the default transcript-uploading call
    # runner and reads what actually reached the transport.
    async def scenario() -> None:
        scenarios = [FakeScenario("s1", "platform-s1", [FakeSubGoal("holds", True)])]
        harness = _build_harness(
            scenarios=scenarios,
            instances=1,
            artifacts=HarnessArtifactPolicy(level=ArtifactLevel.METADATA_ONLY),
        )
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK
        # Metadata-only still carries mandatory terminal metadata, but no transcript bytes.
        kinds = {entry["kind"] for entry in harness.transport.manifests[-1]["entries"]}
        assert {"build", "result", "log"} <= kinds
        assert "transcript" not in kinds

        receipt_body = harness.transport.receipts[("job-1", "s1")]
        assert (
            receipt_body["status"] == "passed"
        )  # a refused upload must not error the scenario
        assert receipt_body["call"]["transcript_artifact"] is None

        log_events = [
            r for r in harness.transport.event_records if r.get("type") == "log"
        ]
        assert any(
            r["payload"]["level"] == "error"
            and "kind=transcript" in r["payload"]["message"]
            and "forbidden at level=metadata-only" in r["payload"]["message"]
            for r in log_events
        )

    asyncio.run(scenario())


def test_traces_artifact_level_refuses_recording_upload_end_to_end() -> None:
    # A second shape at a different level -- `traces` allows transcripts but forbids
    # recordings. A custom call runner uploads both so the table's per-kind behaviour is visible,
    # not just its per-level all-or-nothing behaviour.
    async def scenario() -> None:
        class RecordingCallRunner:
            def __init__(self, adapter: he.OutboundAdapter) -> None:
                self._adapter = adapter

            async def run(
                self, scenario: FakeScenario, runtime: EnvironmentRuntime
            ) -> CallOutcome:
                del runtime
                transcript_id = await self._adapter.upload_artifact(
                    b"transcript-bytes",
                    kind=ob.ArtifactKind.TRANSCRIPT,
                    scenario_key=scenario.scenario_key,
                )
                recording_id = await self._adapter.upload_artifact(
                    b"recording-bytes",
                    kind=ob.ArtifactKind.RECORDING_COMBINED,
                    scenario_key=scenario.scenario_key,
                )
                now = _rfc3339(datetime.now(timezone.utc))
                return CallOutcome(
                    calls=(
                        Call(
                            name="tool",
                            arguments={},
                            result="ok",
                            ok=True,
                            error="",
                            refused=False,
                            at=0.0,
                        ),
                    ),
                    turns=1,
                    started_at=now,
                    ended_at=now,
                    duration_ms=10,
                    transcript_artifact=transcript_id,
                    recording_artifacts=(recording_id,) if recording_id else (),
                )

        scenarios = [FakeScenario("s1", "platform-s1", [FakeSubGoal("holds", True)])]
        harness = _build_harness(
            scenarios=scenarios,
            instances=1,
            artifacts=HarnessArtifactPolicy(level=ArtifactLevel.TRACES),
        )
        harness.deps.build_call_runner = lambda adapter, context: RecordingCallRunner(
            adapter
        )
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK

        receipt_body = harness.transport.receipts[("job-1", "s1")]
        assert receipt_body["status"] == "passed"
        assert (
            receipt_body["call"]["transcript_artifact"] is not None
        )  # traces allows transcripts
        assert (
            receipt_body["call"]["recording_artifacts"] == []
        )  # recordings refused at traces

        transcript_digest = receipt_body["call"]["transcript_artifact"].split(":", 1)[1]
        assert transcript_digest in harness.transport.artifacts
        kinds = {entry["kind"] for entry in harness.transport.manifests[-1]["entries"]}
        assert "recording_combined" not in kinds
        assert {"build", "result", "log", "transcript"} <= kinds

        log_events = [
            r for r in harness.transport.event_records if r.get("type") == "log"
        ]
        assert any(
            r["payload"]["level"] == "error"
            and "kind=recording_combined" in r["payload"]["message"]
            and "forbidden at level=traces" in r["payload"]["message"]
            for r in log_events
        )

    asyncio.run(scenario())


def test_secrets_unlink_failure_after_the_terminal_does_not_lose_the_terminal_event() -> (
    None
):
    # A non-writable secrets directory must not cost the terminal event -- the
    # unlink now runs AFTER `emit_terminal` and is wrapped, so an OSError there is logged, not
    # raised past `run_job`.
    async def scenario() -> None:
        scenarios = [FakeScenario("s1", "platform-s1", [FakeSubGoal("holds", True)])]
        harness = _build_harness(scenarios=scenarios, instances=1)
        guard_dir = Path(tempfile.mkdtemp(prefix="p10-ro-"))
        secrets_path = guard_dir / "secrets.json"
        secrets_path.write_text('{"A": "x"}', encoding="utf-8")
        os.chmod(
            guard_dir, stat.S_IRUSR | stat.S_IXUSR
        )  # read-only directory -> unlink raises
        harness.deps.secrets_path = secrets_path
        try:
            code = await he.run_job(
                harness.job_path, harness.source, harness.output, deps=harness.deps
            )
            assert code == he.EXIT_OK
            terminals = harness.transport.terminal_events()
            assert len(terminals) == 1
            assert terminals[0]["payload"]["stage"] == "completed"
        finally:
            os.chmod(guard_dir, stat.S_IRWXU)

    asyncio.run(scenario())


def test_events_channel_dying_on_the_final_drain_exits_terminal_undelivered() -> None:
    # The events channel dies specifically on the flush that carries the terminal
    # (sustained 5xx exhausts the retry budget) -- exit 0 would falsely claim a flush that never
    # happened. `terminal_undelivered` catches this via the spool watermark never reaching the
    # terminal's own sequence.
    async def scenario() -> None:
        class DyingOnTerminalTransport(FakeTransport):
            def request(
                self,
                method: str,
                url: str,
                *,
                headers: dict[str, str],
                json_body: dict[str, Any] | None = None,
                data: bytes | Any | None = None,
                timeout: float = 30.0,
            ) -> ob.TransportResponse:
                if (
                    method == "POST"
                    and "/events/" in url
                    and isinstance(data, (bytes, bytearray))
                    and b'"type":"terminal"' in bytes(data)
                ):
                    self.calls.append(
                        {"method": method, "url": url, "headers": dict(headers)}
                    )
                    return ob.TransportResponse(
                        500,
                        {"error": "server_error", "message": "boom", "retryable": True},
                        {},
                    )
                return super().request(
                    method,
                    url,
                    headers=headers,
                    json_body=json_body,
                    data=data,
                    timeout=timeout,
                )

        scenarios = [FakeScenario("s1", "platform-s1", [FakeSubGoal("holds", True)])]
        harness = _build_harness(scenarios=scenarios, instances=1)
        transport = DyingOnTerminalTransport()
        harness.deps.build_transport = lambda: transport
        harness.deps.retry_policy = lambda: ob.RetryPolicy(
            initial_backoff_seconds=0.0,
            max_backoff_seconds=0.0,
            max_attempts=3,
        )
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_TERMINAL_UNDELIVERED
        assert transport.terminal_events() == []  # never actually reached the platform

    asyncio.run(scenario())


def test_platform_rejecting_the_terminal_event_exits_terminal_undelivered() -> None:
    # A different shape: the platform permanently rejects the terminal item itself (a per-item rejection
    # inside an otherwise-200 response), distinct from a dead channel.
    async def scenario() -> None:
        class RejectingTerminalTransport(FakeTransport):
            def request(
                self,
                method: str,
                url: str,
                *,
                headers: dict[str, str],
                json_body: dict[str, Any] | None = None,
                data: bytes | Any | None = None,
                timeout: float = 30.0,
            ) -> ob.TransportResponse:
                if (
                    method == "POST"
                    and "/events/" in url
                    and isinstance(data, (bytes, bytearray))
                ):
                    body = json.loads(bytes(data).decode("utf-8"))
                    events = body.get("events", [])
                    terminal = [e for e in events if e.get("type") == "terminal"]
                    if terminal:
                        self.calls.append(
                            {"method": method, "url": url, "headers": dict(headers)}
                        )
                        keep = [e for e in events if e.get("type") != "terminal"]
                        self.event_records.extend(keep)
                        rejected = [
                            {
                                "sequence": e["sequence"],
                                "code": "payload_invalid",
                                "message": "nope",
                            }
                            for e in terminal
                        ]
                        watermark = max((e["sequence"] for e in events), default=0)
                        return ob.TransportResponse(
                            200,
                            {"acked_through_sequence": watermark, "rejected": rejected},
                            {},
                        )
                return super().request(
                    method,
                    url,
                    headers=headers,
                    json_body=json_body,
                    data=data,
                    timeout=timeout,
                )

        scenarios = [FakeScenario("s1", "platform-s1", [FakeSubGoal("holds", True)])]
        harness = _build_harness(scenarios=scenarios, instances=1)
        transport = RejectingTerminalTransport()
        harness.deps.build_transport = lambda: transport
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_TERMINAL_UNDELIVERED
        assert (
            transport.terminal_events() == []
        )  # rejected, never landed as a delivered record

    asyncio.run(scenario())


def test_receipt_rejection_by_the_platform_is_logged() -> None:
    # `ResultsClient.push()` returns `ReceiptPushResult(error=...)` on a permanent rejection
    # rather than raising -- nothing inspected the return value before this fix, so the contract's
    # "guest logs, no retry" obligation for e.g. 409 receipt_conflict went unmet.
    async def scenario() -> None:
        class RejectingResultsTransport(FakeTransport):
            def request(
                self,
                method: str,
                url: str,
                *,
                headers: dict[str, str],
                json_body: dict[str, Any] | None = None,
                data: bytes | Any | None = None,
                timeout: float = 30.0,
            ) -> ob.TransportResponse:
                if method == "POST" and "/results/" in url and json_body is not None:
                    self.calls.append(
                        {"method": method, "url": url, "headers": dict(headers)}
                    )
                    return ob.TransportResponse(
                        409,
                        {
                            "error": "receipt_conflict",
                            "message": "digest mismatch",
                            "retryable": False,
                        },
                        {},
                    )
                return super().request(
                    method,
                    url,
                    headers=headers,
                    json_body=json_body,
                    data=data,
                    timeout=timeout,
                )

        scenarios = [FakeScenario("s1", "platform-s1", [FakeSubGoal("holds", True)])]
        harness = _build_harness(scenarios=scenarios, instances=1)
        transport = RejectingResultsTransport()
        harness.deps.build_transport = lambda: transport
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK

        log_events = [r for r in transport.event_records if r.get("type") == "log"]
        assert any(
            r["payload"]["level"] == "error"
            and "s1" in r["payload"]["message"]
            and "receipt_conflict" in r["payload"]["message"]
            for r in log_events
        )

    asyncio.run(scenario())


def test_receipt_failure_message_is_capped_like_the_terminal_message() -> None:
    # `receipt().failure.message` was uncapped while `emit_terminal` caps at 4KB -- an
    # oversized receipt failure message is the most plausible route into a 413 (the same silent-loss
    # shape as a rejected receipt). Adapter-level: no `run_job` needed for a pure formatting check.
    async def scenario() -> None:
        transport = FakeTransport()
        adapter = _build_adapter(transport)
        await adapter.receipt(
            ResultReceipt(
                scenario_key="s1",
                scenario_id="platform-s1",
                scenario_attempt=1,
                world_index=0,
                status="errored",
                sub_goals=(),
                evaluations=(),
                call=None,
                failure=ReceiptFailure(
                    domain="agent",
                    stage="running",
                    code="call_failed",
                    message="y" * 20_000,
                ),
            )
        )
        body = transport.receipts[("job-1", "s1")]
        message = body["failure"]["message"]
        assert len(message) <= he._TERMINAL_FAILURE_MESSAGE_MAX_CHARS
        assert message.endswith("…[truncated]")

    asyncio.run(scenario())


def test_cancel_before_provision_skips_provisioning_entirely() -> None:
    # The pre-provision cancel checkpoint (`hosted_entrypoint.py:1342` area) had no
    # test driving a cancel signal written BEFORE `run_job` starts -- disabling all three
    # checkpoints still passed the whole suite.
    async def scenario() -> None:
        scenarios = [FakeScenario("s1", "platform-s1", [FakeSubGoal("holds", True)])]
        harness = _build_harness(scenarios=scenarios, instances=1)
        harness.deps.cancel_path.write_text(
            json.dumps({"reason": "ttl_exceeded"}), encoding="utf-8"
        )
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK
        assert (
            harness.provisioner.provision_calls == 0
        )  # canceled before provisioning ever ran
        terminals = harness.transport.terminal_events()
        assert len(terminals) == 1
        assert terminals[0]["payload"]["stage"] == "canceled"
        assert terminals[0]["payload"]["reason"] == "ttl_exceeded"

        # a CANCELED run is cut short -- the manifest must say so.
        assert harness.transport.manifests[-1]["complete"] is False

    asyncio.run(scenario())


def test_evaluation_wire_coerces_an_int_score_to_float() -> None:
    # `_evaluation_wire`'s `float(score)` coercion had no direct unit test -- an
    # int score digest-mismatches `build_result_receipt`'s own re-derivation and silently drops the
    # whole receipt.
    class IntScoreEvaluation:
        kind = "metric"
        name = "accuracy"
        score = 1
        reason = "int not float"

    wire = he._evaluation_wire(IntScoreEvaluation())
    assert wire["score"] == 1.0
    assert isinstance(wire["score"], float)


def test_receipt_nulls_an_unacked_transcript_artifact_and_logs_it() -> None:
    # A receipt naming an artifact id this adapter never uploaded (and therefore
    # never acked) must ship `null`/`[]` on the wire, not the un-acked id -- the platform would 422
    # (`artifact_unknown`) the whole receipt otherwise.
    async def scenario() -> None:
        transport = FakeTransport()
        adapter = _build_adapter(transport)

        class UnackedCall:
            started_at = "2026-01-01T00:00:00.000Z"
            ended_at = "2026-01-01T00:00:01.000Z"
            duration_ms = 1000
            turns = 1
            transcript_artifact = (
                "sha256:" + "a" * 64
            )  # never uploaded through this adapter
            recording_artifacts = ("sha256:" + "b" * 64,)

        await adapter.receipt(
            ResultReceipt(
                scenario_key="s1",
                scenario_id="platform-s1",
                scenario_attempt=1,
                world_index=0,
                status="passed",
                sub_goals=(),
                evaluations=(),
                call=UnackedCall(),
                failure=None,
            )
        )
        body = transport.receipts[("job-1", "s1")]
        assert body["call"]["transcript_artifact"] is None
        assert body["call"]["recording_artifacts"] == []

        log_events = [r for r in transport.event_records if r.get("type") == "log"]
        assert any(
            r["payload"]["level"] == "error"
            and "un-acked transcript" in r["payload"]["message"]
            for r in log_events
        )
        assert any(
            r["payload"]["level"] == "error"
            and "un-acked recording" in r["payload"]["message"]
            for r in log_events
        )

    asyncio.run(scenario())


def test_pre_run_provision_failure_emits_the_terminal_before_closing_the_pool() -> None:
    # `test_finish_emits_the_terminal_before_...`
    # only drives the COMPLETED path -- a pre-run failure (`_fail` -> `_finish`) is a structurally
    # different entry into `_finish` and needs its own ordering check.
    async def scenario() -> None:
        order: list[str] = []

        class OrderTrackingTransport(FakeTransport):
            def request(
                self,
                method: str,
                url: str,
                *,
                headers: dict[str, str],
                json_body: dict[str, Any] | None = None,
                data: bytes | Any | None = None,
                timeout: float = 30.0,
            ) -> ob.TransportResponse:
                response = super().request(
                    method,
                    url,
                    headers=headers,
                    json_body=json_body,
                    data=data,
                    timeout=timeout,
                )
                if (
                    "/events/" in url
                    and method == "POST"
                    and isinstance(data, (bytes, bytearray))
                    and b'"type":"terminal"' in bytes(data)
                ):
                    order.append("terminal_delivered")
                return response

        class FailingProvisioner(FakeProvisioner):
            async def provision(
                self,
                bundle: Any,
                *,
                source: Path,
                bundle_dir: Path,
                work_directory: Path,
                contract: Any | None = None,
                instances: int = 1,
            ) -> list[EnvironmentRuntime]:
                del bundle, source, bundle_dir, work_directory, contract, instances
                raise ProcessRuntimeError(
                    "build", "build_failed", "synthetic", process="agent"
                )

            async def close(self, *, work_directory: Path) -> None:
                await super().close(work_directory=work_directory)
                order.append("pool_closed")

        harness = _build_harness(scenarios=[], instances=1)
        transport = OrderTrackingTransport()
        harness.deps.build_transport = lambda: transport
        harness.deps.build_provider = lambda _capabilities, _transport: (
            FailingProvisioner(instances=1)
        )
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK
        assert order[:2] == ["terminal_delivered", "pool_closed"]
        terminals = transport.terminal_events()
        assert len(terminals) == 1
        assert terminals[0]["payload"]["failure"]["domain"] == "agent"

    asyncio.run(scenario())


# =================================================================================================
# Additional tests -- the wire retargeted the fence-on-final-drain test to the pre-drain
# check; flush_terminal's and drain()'s bounded loops now cover for each other; the
# post-terminal wire block had no deadline.
# =================================================================================================


def test_fence_on_the_manifest_push_still_exits_fenced_with_the_terminal_already_delivered() -> (
    None
):
    # `flush_terminal` now delivers the terminal-carrying POST ahead of `drain()`, so a fence
    # on that POST is caught by the PRE-drain `is_fenced` check in `_finish`, not the post-drain
    # `if fenced: return EXIT_FENCED` line -- `test_fence_landing_on_the_final_
    # drain_still_exits_fenced` no longer exercises that check for the reason its own comment
    # claims. A fence that only starts on `/manifest/` lands strictly inside `drain()` itself
    # (drain()'s own `push_manifest` call), after a clean `flush_terminal`, so `drain()`'s return
    # value -- and therefore only the post-drain check -- is what decides the exit code here. Also
    # pins the observation that a fenced attempt can still have already delivered its terminal,
    # deliberately rather than by accident.
    async def scenario() -> None:
        scenarios = [FakeScenario("s1", "platform-s1", [FakeSubGoal("holds", True)])]
        harness = _build_harness(
            scenarios=scenarios,
            instances=1,
            fence_on_url_substring="/manifest/",
        )
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_FENCED
        terminals = harness.transport.terminal_events()
        assert len(terminals) == 1  # delivered before the fence ever landed
        assert terminals[0]["payload"]["stage"] == "completed"
        assert harness.provisioner.closed is True

    asyncio.run(scenario())


def test_drain_alone_delivers_a_pre_run_backlog_larger_than_one_batch() -> None:
    # On a pre-run terminal (`_fail` -> `_finish` with no `scheduler_result`), the wire's
    # `flush_terminal` is never called at all -- `drain()`'s own bounded loop is the ONLY delivery
    # mechanism for whatever accumulated beforehand. `test_drain_loops_past_a_backlog_...` only
    # drives a post-run COMPLETED path, where `flush_terminal` already drains everything first and
    # masks a collapsed `drain()` loop. A `ScenarioSource` that logs a large pre-run backlog before
    # raising a typed pre-allocation failure reproduces the shape end to end, on the one path where
    # collapsing `drain()`'s loop to a single flush cannot be covered for by anything else.
    async def scenario() -> None:
        class ChattyPreRunScenarioSource:
            def __init__(self, *, chatter_count: int) -> None:
                self._chatter_count = chatter_count

            async def build(
                self,
                job: HarnessJob,
                bundle: Any,
                scenarios_client: he.ScenariosClient,
                *,
                pool: Any,
                world_factory: Any,
                bundle_dir: Path,
            ) -> list[FakeScenario]:
                del job, bundle, scenarios_client, world_factory, bundle_dir
                adapter = pool._outbound  # no adapter seam on ScenarioSource itself
                for i in range(self._chatter_count):
                    await adapter.log(level="info", message=f"pre-run chatter {i}")
                raise he.ScenarioPreallocationError(None)

        harness = _build_harness(scenarios=[], instances=1)
        harness.deps.scenario_source = ChattyPreRunScenarioSource(chatter_count=260)
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK

        terminals = harness.transport.terminal_events()
        assert len(terminals) == 1  # must not be stranded behind the pre-run backlog
        assert terminals[0]["payload"]["stage"] == "failed"

        chatter_logs = [
            record
            for record in harness.transport.event_records
            if record.get("type") == "log"
            and "pre-run chatter" in record["payload"].get("message", "")
        ]
        assert (
            len(chatter_logs) == 260
        )  # the whole backlog drained, not just the first batch

    asyncio.run(scenario())


def test_flush_terminal_alone_must_deliver_the_terminal_before_a_skipped_receipt_under_backlog() -> (
    None
):
    # The converse of the previous test -- on a cancel-mid-run path, `flush_terminal` is the
    # ONLY thing standing between "terminal not yet on the wire" and `emit_skipped_receipts`
    # pushing a receipt straight to `/results/` (that push is not gated on event delivery at all).
    # A backlog bigger than one batch, queued before the cancel is even noticed, means a collapsed
    # `flush_terminal` loop delivers only the first batch and leaves the terminal pending -- the
    # skipped receipt for "second" then reaches the platform BEFORE the terminal does, even though
    # `drain()`'s own (intact) loop mops up the rest a moment later. Ordering, not delivery count,
    # is what only `flush_terminal`'s loop can guarantee here.
    async def scenario() -> None:
        order: list[str] = []
        scenarios = [
            FakeScenario("first", "platform-first", [FakeSubGoal("holds", True)]),
            FakeScenario("second", "platform-second", [FakeSubGoal("holds", True)]),
        ]

        class OrderTrackingTransport(FakeTransport):
            def request(
                self,
                method: str,
                url: str,
                *,
                headers: dict[str, str],
                json_body: dict[str, Any] | None = None,
                data: bytes | Any | None = None,
                timeout: float = 30.0,
            ) -> ob.TransportResponse:
                response = super().request(
                    method,
                    url,
                    headers=headers,
                    json_body=json_body,
                    data=data,
                    timeout=timeout,
                )
                if (
                    "/events/" in url
                    and method == "POST"
                    and isinstance(data, (bytes, bytearray))
                    and b'"type":"terminal"' in bytes(data)
                ):
                    order.append("terminal_delivered")
                if (
                    "/results/" in url
                    and method == "POST"
                    and json_body is not None
                    and json_body.get("status") == "skipped"
                ):
                    order.append("skipped_receipt_delivered")
                return response

        class ChattyCancelingCallRunner:
            def __init__(
                self,
                adapter: he.OutboundAdapter,
                *,
                cancel_path: Path,
                cancel_on_scenario: str,
                chatter_count: int,
            ) -> None:
                self._adapter = adapter
                self._cancel_path = cancel_path
                self._cancel_on_scenario = cancel_on_scenario
                self._chatter_count = chatter_count

            async def run(
                self, scenario: FakeScenario, runtime: EnvironmentRuntime
            ) -> CallOutcome:
                del runtime
                if scenario.scenario_key == self._cancel_on_scenario:
                    for i in range(self._chatter_count):
                        await self._adapter.log(level="info", message=f"chatter {i}")
                    self._cancel_path.write_text(
                        json.dumps({"reason": "user_canceled"}), encoding="utf-8"
                    )
                now = _rfc3339(datetime.now(timezone.utc))
                return CallOutcome(
                    calls=(
                        Call(
                            name="tool",
                            arguments={},
                            result="ok",
                            ok=True,
                            error="",
                            refused=False,
                            at=0.0,
                        ),
                    ),
                    turns=1,
                    started_at=now,
                    ended_at=now,
                    duration_ms=10,
                    transcript_artifact=None,
                    recording_artifacts=(),
                )

        harness = _build_harness(
            scenarios=scenarios, cancel_on_scenario="first", instances=1
        )
        transport = OrderTrackingTransport()
        harness.deps.build_transport = lambda: transport
        harness.deps.build_call_runner = lambda adapter, context: (
            ChattyCancelingCallRunner(
                adapter,
                cancel_path=harness.deps.cancel_path,
                cancel_on_scenario="first",
                chatter_count=260,
            )
        )
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK
        statuses = {key[1]: body["status"] for key, body in transport.receipts.items()}
        assert statuses.get("first") == "passed"
        assert statuses.get("second") == "skipped"

        assert order == ["terminal_delivered", "skipped_receipt_delivered"]

    asyncio.run(scenario())


def test_post_terminal_wire_block_is_bounded_by_the_remaining_flush_window() -> None:
    # `emit_skipped_receipts`/`receipt()` push with `deadline=None` -- a degraded-but-alive
    # events channel could previously retry every skipped scenario's receipt for the full
    # `RetryPolicy` budget regardless of how much of the flush window was already spent, well past
    # the point the gateway tears the sandbox down. A tiny window plus a slow
    # `emit_skipped_receipts` reproduces the shape without a real multi-second stall dominating the
    # suite: `asyncio.wait_for`'s own cancellation cuts the sleep short well before it would run out.
    async def scenario() -> None:
        scenarios = [
            FakeScenario("first", "platform-first", [FakeSubGoal("holds", True)]),
            FakeScenario("second", "platform-second", [FakeSubGoal("holds", True)]),
        ]
        harness = _build_harness(
            scenarios=scenarios, cancel_on_scenario="first", instances=1
        )
        harness.deps.flush_window_seconds = 0.2

        original_emit = HostedScheduler.emit_skipped_receipts

        async def slow_emit(self: HostedScheduler, result: RunResult) -> None:
            await asyncio.sleep(2.0)
            await original_emit(self, result)

        HostedScheduler.emit_skipped_receipts = slow_emit  # type: ignore[method-assign]
        started = time.monotonic()
        try:
            code = await he.run_job(
                harness.job_path, harness.source, harness.output, deps=harness.deps
            )
        finally:
            HostedScheduler.emit_skipped_receipts = original_emit
        elapsed = time.monotonic() - started

        assert elapsed < 1.5, (
            f"post-terminal wire block was not bounded: {elapsed:.2f}s"
        )
        assert (
            code == he.EXIT_OK
        )  # the terminal was already delivered before the window ran out
        terminals = harness.transport.terminal_events()
        assert len(terminals) == 1
        assert terminals[0]["payload"]["stage"] == "canceled"

        statuses = {
            key[1]: body["status"] for key, body in harness.transport.receipts.items()
        }
        assert statuses.get("first") == "passed"
        assert (
            "second" not in statuses
        )  # window ran out before the skipped receipt could go

        assert (
            harness.provisioner.closed is True
        )  # the finally backstop still closes the pool

    asyncio.run(scenario())


# =================================================================================================
# p12: scenario_source.py wiring -- item 4. Writes real scenario documents (`folder.py`'s own
# on-disk layout) straight into `harness.bundle_dir`, matching `scenario_source.py`'s
# `SCENARIOS_DIRNAME` constant. Deliberately its own tiny writer rather than importing
# `test_scenario_source.py`'s `_write_scenario` (no cross-test-module private-helper imports).
#
# §2e preflight (`bundle_file_unlisted`) rejects any bundle file absent from the manifest's
# `files[]` -- exactly the CONTRACT QUESTIONS obligation the report calls out for a real Scenario
# Generation Contract bundle author. So a scenario-bearing bundle for these tests cannot just drop
# files under `bundle_dir/scenarios/` after `_write_bundle` has already sealed the manifest; the
# scenario files have to be hashed into `files[]` and the bundle re-sealed, the same way
# `test_process_preflight.py`'s own `_build_bundle(extra_files=...)` does it.
# =================================================================================================


def _scenario_doc_files(
    name: str,
    *,
    scenario_key: str,
    scenario_id: str = "",
    sub_goals: list[str] | None = None,
    setup_code: str = "",
    checks: dict[str, str] | None = None,
) -> dict[str, bytes]:
    """One scenario folder's contents as {relative path: bytes} -- fed to
    `_write_bundle_with_scenario_files` below rather than written straight to disk, so every byte
    can be hashed into the manifest's `files[]` before the bundle is sealed."""
    body = {
        "name": name,
        "scenario_key": scenario_key,
        "scenario_id": scenario_id,
        "sub_goals": sub_goals or [],
    }
    prefix = f"{SCENARIOS_DIRNAME}/{name}"
    files = {f"{prefix}/scenario.json": json.dumps(body).encode("utf-8")}
    if setup_code:
        files[f"{prefix}/setup.py"] = setup_code.encode("utf-8")
    for goal_name, code in (checks or {}).items():
        files[f"{prefix}/checks/{goal_name}.py"] = code.encode("utf-8")
    return files


def _write_bundle_with_scenario_files(
    root: Path, scenario_files: dict[str, bytes]
) -> EnvironmentBundleV2:
    """`_write_bundle`, plus `scenario_files` hashed into `files[]` and the digest re-sealed over
    all of it -- otherwise every one of these trips `bundle_file_unlisted` at preflight, before
    `scenario_source.build()` is ever reached."""
    root.mkdir(parents=True, exist_ok=True)
    body = _base_manifest_body()
    file_contents = {
        "db/schema.sql": SCHEMA_SQL,
        "db/seed.sql": SEED_SQL,
        **scenario_files,
    }
    files: list[dict[str, Any]] = []
    for relative, content in file_contents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
            }
        )
    body["files"] = files
    digest = compute_inputs_digest(
        root,
        ["db/schema.sql"],
        ["db/seed.sql"],
        engine=ManagedEngine.POSTGRES,
        version="16",
    )
    body["seed"] = {
        "stores": [
            {
                "capability": "database",
                "migrations": ["db/schema.sql"],
                "seed_files": ["db/seed.sql"],
                "baseline": {"strategy": "template_database", "inputs_digest": digest},
                "sentinel": {"query": "SELECT count(*) FROM riders", "expected": "1"},
            }
        ]
    }
    body["digest"] = "sha256:" + "0" * 64
    normalized = EnvironmentBundleV2.model_validate(body)
    body["digest"] = seal_bundle_v2(normalized)
    (root / "manifest.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    return EnvironmentBundleV2.model_validate(body)


def test_bundle_without_scenarios_keeps_the_notwired_regression() -> None:
    # item 4: a bundle that does not carry a `scenarios/` directory must keep behaving exactly as
    # it did before this adapter existed -- the default `NotWiredScenarioSource`'s typed failure,
    # never a crash and never a silently-empty run.
    async def scenario() -> None:
        harness = _build_harness(
            scenarios=[], instances=1, use_default_scenario_source=True
        )
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK
        terminals = harness.transport.terminal_events()
        assert len(terminals) == 1
        failure = terminals[0]["payload"]["failure"]
        assert failure["stage"] == "validating_scenarios"
        assert failure["domain"] == "platform_sync"
        assert failure["code"] == "scenario_preallocation_failed"

    asyncio.run(scenario())


def test_default_scenario_source_wires_the_bundle_adapter_when_scenarios_present() -> (
    None
):
    # item 4 + 5c (end to end): the presence test flips the default over to the real
    # `BundleScenarioSource` -- no `FakeScenarioSource` involved anywhere in this test. One
    # deterministic sub_goal that genuinely holds against the fake world, so this proves a real
    # COMPLETED pass, not just that the vacuous-pass guard fired.
    async def scenario() -> None:
        harness = _build_harness(
            scenarios=[],
            instances=1,
            use_default_scenario_source=True,
            bundle_writer=lambda bundle_dir: _write_bundle_with_scenario_files(
                bundle_dir,
                _scenario_doc_files(
                    # `scenario_id` non-empty here on purpose -- see
                    # `test_empty_scenario_id_receipt_is_dropped_by_the_wire_schema` just below for
                    # the (newly discovered, load-bearing) reason an EMPTY one cannot be used to
                    # prove a receipt actually arrives.
                    "passing",
                    scenario_key="passing",
                    scenario_id="platform-passing",
                    sub_goals=["holds"],
                    checks={"holds": "def check(world, calls):\n    return None\n"},
                ),
            ),
        )
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK

        terminals = harness.transport.terminal_events()
        assert len(terminals) == 1
        payload = terminals[0]["payload"]
        assert payload["stage"] == "completed"
        assert payload["failure"] is None

        # terminal last: the terminal record is the final event this run ever pushed.
        assert harness.transport.event_records[-1].get("type") == "terminal"

        statuses = {
            key[1]: body["status"] for key, body in harness.transport.receipts.items()
        }
        assert statuses == {
            "passing": "passed"
        }  # the real scheduler actually ran it, and it held

    asyncio.run(scenario())


def test_empty_scenario_key_from_bundle_document_fails_cleanly_via_existing_validation() -> (
    None
):
    # Mutation table item 2: "empty-key fixture -> typed failure", at the FULL integration level --
    # a hand-written document with an empty `scenario_key` flows verbatim through this adapter
    # (work item 3: never synthesized) into the scheduler's OWN pre-existing defense
    # (`_validate_scenario_entry`, untouched by this task), which must catch it as a typed FAILED
    # terminal rather than an `AttributeError` deep in the scheduler. `test_scenario_source.py`
    # covers the reader's verbatim carry and the mutation that would break it; this is the
    # end-to-end proof that the two halves agree.
    async def scenario() -> None:
        harness = _build_harness(
            scenarios=[],
            instances=1,
            use_default_scenario_source=True,
            bundle_writer=lambda bundle_dir: _write_bundle_with_scenario_files(
                bundle_dir, _scenario_doc_files("s1", scenario_key="", sub_goals=[])
            ),
        )
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK
        terminals = harness.transport.terminal_events()
        assert len(terminals) == 1
        failure = terminals[0]["payload"]["failure"]
        assert failure["stage"] == "validating_scenarios"
        assert failure["domain"] == "environment"
        assert failure["code"] == "scenario_preallocation_failed"
        assert "scenario_key" in failure["message"]

    asyncio.run(scenario())


def test_mutation_adapter_off_makes_the_e2e_test_fail() -> None:
    # Mutation table item 1: "adapter-off -> e2e test fails". Patches `bundle_has_scenarios` (as
    # seen from `hosted_entrypoint.py`'s own namespace, where it was imported) to always report
    # "no scenarios here" -- simulating hard-rule edit (b) never having been made, i.e. the wiring
    # `if isinstance(...) and bundle_has_scenarios(...)` guard permanently failing closed.
    #
    # R1-6 fold-in (p12-review-r1.md LOW finding): the original version of this test asserted the
    # MUTANT's own failure terminal directly -- true, but it never actually ran the `stage ==
    # "completed"` assertion that is the real kill. This version runs the REAL
    # `test_default_scenario_source_wires_the_bundle_adapter_when_scenarios_present` under the
    # patch and checks THAT it fails. `mock.patch.object` restores the original function in its own
    # `finally` regardless of how the inner call ends -- no manual restore bookkeeping needed.
    with mock.patch.object(he, "bundle_has_scenarios", lambda bundle_dir: False):
        try:
            test_default_scenario_source_wires_the_bundle_adapter_when_scenarios_present()
        except AssertionError:
            pass
        else:
            raise AssertionError(
                "adapter-off mutant did not fail "
                "test_default_scenario_source_wires_the_bundle_adapter_when_scenarios_present "
                "(no pytest.raises here -- this file runs stand-alone via TESTS, per its own "
                "module docstring)"
            )

    # Restored: the real test passes again.
    test_default_scenario_source_wires_the_bundle_adapter_when_scenarios_present()


def test_bundle_scenario_id_is_assigned_by_registration_and_receipt_now_delivers() -> (
    None
):
    # p13 UPDATE of the former `test_empty_scenario_id_receipt_is_dropped_by_the_wire_schema`
    # (p12): that test pinned a real gap -- `outbound.py`'s `ResultReceiptDraft` schema requires
    # `scenario_id` non-empty (pydantic `min_length=1`), and before this task nothing ever filled
    # it in, so a bundle-sourced scenario's receipt was silently dropped. Registration
    # (`register_with_platform`, scenario_source.py) now runs between load and the scheduler and
    # OVERWRITES `scenario_id` with the platform-assigned one before `BundleScenarioSource.build`
    # ever returns -- the document is written with `scenario_id=""` here specifically to prove the
    # id on the wire came from the (fake) platform's provision response, not the document. See
    # `test_unregistered_scenario_with_empty_scenario_id_receipt_still_drops_safely` just below for
    # the property this test used to pin, preserved on the path that never registers at all.
    async def scenario() -> None:
        harness = _build_harness(
            scenarios=[],
            instances=1,
            use_default_scenario_source=True,
            bundle_writer=lambda bundle_dir: _write_bundle_with_scenario_files(
                bundle_dir,
                _scenario_doc_files(
                    "passing",
                    scenario_key="passing",
                    scenario_id="",
                    sub_goals=["holds"],
                    checks={"holds": "def check(world, calls):\n    return None\n"},
                ),
            ),
        )
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK

        terminals = harness.transport.terminal_events()
        assert len(terminals) == 1
        payload = terminals[0]["payload"]
        assert payload["stage"] == "completed"
        assert payload["scenario_counts"]["passed"] == 1

        statuses = {
            key[1]: body["status"] for key, body in harness.transport.receipts.items()
        }
        assert statuses == {"passing": "passed"}  # the receipt DELIVERS now -- no drop.

        ((_, body),) = [
            (key, body)
            for key, body in harness.transport.receipts.items()
            if key[1] == "passing"
        ]
        # `FakeTransport`'s fake platform assigns `f"platform-{scenario_key}"` -- confirms the id
        # on the wire is the PLATFORM's, not the document's own (empty) one.
        assert body["scenario_id"] == "platform-passing"

        error_logs = [
            record
            for record in harness.transport.event_records
            if record.get("type") == "log"
            and record["payload"].get("level") == "error"
            and "ResultReceiptDraft" in record["payload"].get("message", "")
        ]
        assert error_logs == []  # no drop, so no drop log either.

    asyncio.run(scenario())


def test_unregistered_scenario_with_empty_scenario_id_receipt_still_drops_safely() -> (
    None
):
    # Preserves the property the pre-p13 pinning test proved: a scenario that reaches the
    # scheduler with an empty `scenario_id` (never pre-allocated) still has its receipt rejected by
    # `ResultReceiptDraft`'s own schema (`min_length=1`) and DROPPED, loudly, rather than crashing
    # the job or silently delivering a receipt the platform would 422 anyway. Through
    # `BundleScenarioSource` this is now unreachable (registration always assigns a real id or the
    # job fails first) -- so this is exercised through `FakeScenarioSource`'s injected-scenario
    # path instead, which never calls `register_with_platform` at all (an "unregistered" scenario
    # source, same shape as a future ScenarioSource that also skips pre-allocation).
    async def scenario() -> None:
        scenarios = [FakeScenario("passing", "", [FakeSubGoal("holds", True)])]
        harness = _build_harness(
            scenarios=scenarios, instances=1
        )  # FakeScenarioSource, as usual
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK

        terminals = harness.transport.terminal_events()
        assert len(terminals) == 1
        payload = terminals[0]["payload"]
        assert payload["stage"] == "completed"  # the job itself does not fail
        assert (
            payload["scenario_counts"]["passed"] == 1
        )  # ...and reports the scenario as passed...

        statuses = {
            key[1]: body["status"] for key, body in harness.transport.receipts.items()
        }
        assert statuses == {}  # ...yet no receipt for it ever reached the platform.

        error_logs = [
            record
            for record in harness.transport.event_records
            if record.get("type") == "log"
            and record["payload"].get("level") == "error"
            and "ResultReceiptDraft" in record["payload"].get("message", "")
        ]
        assert (
            len(error_logs) == 1
        )  # the drop is at least loud, not silent -- but still a drop.

    asyncio.run(scenario())


def test_registration_response_mismatch_reaches_the_typed_platform_sync_terminal() -> (
    None
):
    # p13: a provision response that fails `_scenario_ids_by_key`'s guards (scenario_source.py --
    # here, naming NO scenario_key at all, so every submitted one is "missing") must fail the
    # whole job through the SAME typed `validating_scenarios`/`platform_sync` terminal
    # `run_job`'s existing `except (ScenarioSourceNotWired, ScenarioPreallocationError)` clause
    # already produces for every other pre-allocation failure -- no new except clause needed. The
    # scheduler must never run: no receipt for the scenario ever reaches the platform.
    async def scenario() -> None:
        class MismatchedProvisionTransport(FakeTransport):
            def request(
                self,
                method: str,
                url: str,
                *,
                headers: dict[str, str],
                json_body: dict[str, Any] | None = None,
                data: bytes | Any | None = None,
                timeout: float = 30.0,
            ) -> ob.TransportResponse:
                if (
                    method == "POST"
                    and "/scenarios/" in url
                    and json_body is not None
                    and json_body.get("operation") == "provision"
                ):
                    self.calls.append(
                        {"method": method, "url": url, "headers": dict(headers)}
                    )
                    self.scenarios_calls.append((url, json_body))
                    return ob.TransportResponse(
                        200,
                        {"result": {"run_test_id": "run-test-1", "scenarios": []}},
                        {},
                    )
                return super().request(
                    method,
                    url,
                    headers=headers,
                    json_body=json_body,
                    data=data,
                    timeout=timeout,
                )

        harness = _build_harness(
            scenarios=[],
            instances=1,
            use_default_scenario_source=True,
            bundle_writer=lambda bundle_dir: _write_bundle_with_scenario_files(
                bundle_dir,
                _scenario_doc_files(
                    "passing",
                    scenario_key="passing",
                    scenario_id="",
                    sub_goals=["holds"],
                    checks={"holds": "def check(world, calls):\n    return None\n"},
                ),
            ),
        )
        transport = MismatchedProvisionTransport()
        harness.deps.build_transport = lambda: transport
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK

        terminals = transport.terminal_events()
        assert len(terminals) == 1
        failure = terminals[0]["payload"]["failure"]
        assert failure["stage"] == "validating_scenarios"
        assert failure["domain"] == "platform_sync"
        assert failure["code"] == "scenario_preallocation_failed"

        assert (
            transport.receipts == {}
        )  # the scheduler never ran -- registration failed first
        # `begin` must never have been attempted -- the provision-side guard stops it first.
        assert not any(
            body.get("operation") == "begin" for _, body in transport.scenarios_calls
        )

    asyncio.run(scenario())


def test_injected_scenario_source_always_wins_over_the_bundle_adapter() -> None:
    # item 4: even when the bundle ALSO carries a valid `scenarios/` directory, an explicitly
    # injected `ScenarioSource` must be used untouched -- the presence test only ever applies to
    # the untouched default.
    async def scenario() -> None:
        injected = [
            FakeScenario(
                "from-fake", "platform-from-fake", [FakeSubGoal("holds", True)]
            )
        ]
        harness = _build_harness(
            scenarios=injected,
            instances=1,  # FakeScenarioSource, as usual -- not the default
            bundle_writer=lambda bundle_dir: _write_bundle_with_scenario_files(
                bundle_dir,
                _scenario_doc_files(
                    "from-bundle", scenario_key="from-bundle", sub_goals=[]
                ),
            ),
        )
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK
        statuses = {
            key[1]: body["status"] for key, body in harness.transport.receipts.items()
        }
        assert statuses == {
            "from-fake": "passed"
        }  # the bundle's own scenario never ran

    asyncio.run(scenario())


# =================================================================================================
# R1-1 (CRITICAL, p12-review-r1.md) -- through the REAL `run_job`, on a sealed, preflight-clean
# bundle: a scenario document must never be able to set the guest's own exit code, and an unreadable
# or malformed scenario file must never escape as an unhandled exception with zero terminal events.
# The chmod-000 "unreadable file" trigger is proven directly against `scenario_source.py` in
# `test_scenario_source.py` instead of here -- see that file's R1-1 section docstring for why the
# full bundle/preflight pipeline cannot exercise it without touching `process_preflight.py`.
# =================================================================================================


def test_module_level_sys_exit_zero_in_setup_is_contained_as_a_typed_failure() -> None:
    # Before the fix: `sys.exit(0)` at module level inside `setup.py` propagated as a raw
    # `SystemExit` straight out of `run_job` -- the guest process itself would exit 0 with ZERO
    # terminal events (a "clean terminal that never happened", per spine §0.6), on a bundle whose
    # every byte was hashed into the manifest and §2e preflight passed.
    async def scenario() -> None:
        harness = _build_harness(
            scenarios=[],
            instances=1,
            use_default_scenario_source=True,
            bundle_writer=lambda bundle_dir: _write_bundle_with_scenario_files(
                bundle_dir,
                _scenario_doc_files(
                    "s1",
                    scenario_key="s1",
                    sub_goals=[],
                    setup_code="import sys\nsys.exit(0)\n",
                ),
            ),
        )
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        # The GUEST's own exit code -- EXIT_OK means "a terminal was reached and flushed", the
        # same meaning it carries for any other typed FAILED terminal, never the scenario's own
        # sys.exit(0) leaking through `run_job`'s return value.
        assert code == he.EXIT_OK

        terminals = harness.transport.terminal_events()
        assert (
            len(terminals) == 1
        )  # a terminal event WAS delivered -- not an empty event stream
        payload = terminals[0]["payload"]
        assert payload["stage"] == "failed"
        failure = payload["failure"]
        assert failure["domain"] == "environment"
        assert failure["stage"] == "validating_scenarios"
        assert failure["code"] == "scenario_preallocation_failed"

    asyncio.run(scenario())


def test_module_level_sys_exit_three_in_setup_does_not_hijack_the_guests_exit_code() -> (
    None
):
    # EXIT_FENCED == 3: before the fix, `sys.exit(3)` here was indistinguishable from the guest
    # itself choosing to exit fenced -- the platform would read an ordinary scenario content defect
    # as a fenced/superseded attempt instead.
    async def scenario() -> None:
        harness = _build_harness(
            scenarios=[],
            instances=1,
            use_default_scenario_source=True,
            bundle_writer=lambda bundle_dir: _write_bundle_with_scenario_files(
                bundle_dir,
                _scenario_doc_files(
                    "s1",
                    scenario_key="s1",
                    sub_goals=[],
                    setup_code="import sys\nsys.exit(3)\n",
                ),
            ),
        )
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK
        assert (
            code != he.EXIT_FENCED
        )  # explicit: the scenario's own exit code did not leak through

        terminals = harness.transport.terminal_events()
        assert len(terminals) == 1
        failure = terminals[0]["payload"]["failure"]
        assert failure["domain"] == "environment"
        assert failure["code"] == "scenario_preallocation_failed"

    asyncio.run(scenario())


def test_non_utf8_setup_file_is_contained_as_a_typed_failure_not_an_escape() -> None:
    # Before the fix: a raw `UnicodeDecodeError` from `_read_text`'s `.read_text(encoding="utf-8")`
    # propagated straight out of `run_job` -- per spine §0.6 a non-zero exit with no terminal event
    # reads as `infrastructure` and gets retried to exhaustion, even though this is a deterministic
    # content defect that will never succeed on retry. The non-UTF-8 bytes hash and seal fine (§2e
    # preflight only hashes raw bytes, never decodes) -- only this module's own `.read_text()` call
    # ever attempts to decode them.
    async def scenario() -> None:
        files = _scenario_doc_files("s1", scenario_key="s1", sub_goals=[])
        files[f"{SCENARIOS_DIRNAME}/s1/setup.py"] = (
            b"def setup(world):\n    return '\xff\xfe'\n"
        )
        harness = _build_harness(
            scenarios=[],
            instances=1,
            use_default_scenario_source=True,
            bundle_writer=lambda bundle_dir: _write_bundle_with_scenario_files(
                bundle_dir, files
            ),
        )
        code = await he.run_job(
            harness.job_path, harness.source, harness.output, deps=harness.deps
        )
        assert code == he.EXIT_OK

        terminals = harness.transport.terminal_events()
        assert len(terminals) == 1
        failure = terminals[0]["payload"]["failure"]
        assert failure["domain"] == "environment"
        assert failure["stage"] == "validating_scenarios"
        assert failure["code"] == "scenario_preallocation_failed"

    asyncio.run(scenario())


TESTS = [
    test_resolve_parallelism_reads_the_raw_value_without_clamping,
    test_out_of_range_parallelism_is_rejected_by_preflight_not_clamped,
    test_job_secret_purposes_maps_alias_to_purpose,
    test_peek_secret_values_reads_without_deleting,
    test_peek_secret_values_missing_file_is_empty,
    test_peek_target_provider_secret_values_filters_by_purpose_and_keeps_the_alias,
    test_peek_target_provider_secret_values_missing_file_is_empty,
    test_peek_target_provider_secret_values_drops_an_alias_with_no_purpose_entry,
    test_default_build_call_runner_returns_real_runner_for_vapi,
    test_default_build_call_runner_returns_real_runner_for_retell,
    test_default_build_call_runner_returns_notwired_for_unresolved_auto,
    test_default_build_call_runner_resolves_auto_voice_contract_to_livekit,
    test_default_build_call_runner_returns_a_real_call_runner_impl_for_livekit,
    test_call_runner_context_is_threaded_with_real_job_bundle_secrets_and_evidence_seam,
    test_row_counts_for_capability_returns_the_matching_store,
    test_row_counts_for_capability_raises_when_the_capability_is_absent,
    test_cancel_state_reads_reason_from_file,
    test_world_pool_serializes_concurrent_provider_calls_end_to_end,
    test_scenarios_client_provision_unwraps_the_result_envelope,
    test_scenarios_client_provision_and_begin_hit_the_same_single_url,
    test_scenarios_client_fencing_latches_the_shared_channel_state,
    test_capabilities_failure_exits_boot_failure_with_no_channel_and_no_event,
    test_preflight_rejection_reaches_a_failed_terminal_event_before_any_provision,
    test_hosted_fenced_error_stops_emitting_and_exits_3_with_no_terminal_event,
    test_cancel_mid_run_synthesizes_a_skipped_receipt_for_the_unstarted_scenario,
    test_fenced_run_result_emits_zero_skipped_receipts,
    test_emit_skipped_receipts_failure_does_not_lose_the_terminal_or_the_exit_code,
    test_cancel_mid_run_with_ttl_exceeded_reports_that_reason,
    test_malformed_job_json_exits_crashed_with_no_terminal_event,
    test_world_pool_exhaustion_reaches_a_failed_terminal_world_pool_exhausted,
    test_fence_landing_on_the_final_drain_still_exits_fenced,
    test_fence_from_scenarios_client_exits_fenced_not_crashed,
    test_scenarios_channel_uses_bearer_auth_never_api_key,
    test_process_world_factory_raises_when_no_postgres_endpoint,
    test_process_world_factory_raises_when_build_json_has_no_matching_store,
    test_build_json_two_stores_emit_two_baseline_frozen_events,
    test_build_json_degrade_payload_matches_the_recorded_values,
    test_build_json_fixed_port_at_w1_does_not_crash,
    test_e2e_two_scenarios_one_pass_one_fail_reaches_completed_and_exits_0,
    test_pool_close_backstop_runs_even_when_scenario_source_raises_untyped,
    test_process_runtime_error_uses_the_carried_domain_over_the_fallback_map,
    test_finish_emits_the_terminal_before_closing_the_pool,
    test_redaction_end_to_end_secret_never_crosses_any_channel,
    test_drain_loops_past_a_backlog_larger_than_one_batch_and_still_delivers_the_terminal,
    test_call_aborted_with_no_ended_at_still_produces_a_receipt,
    test_metadata_only_artifact_level_refuses_transcript_upload_end_to_end,
    test_traces_artifact_level_refuses_recording_upload_end_to_end,
    test_secrets_unlink_failure_after_the_terminal_does_not_lose_the_terminal_event,
    test_events_channel_dying_on_the_final_drain_exits_terminal_undelivered,
    test_platform_rejecting_the_terminal_event_exits_terminal_undelivered,
    test_receipt_rejection_by_the_platform_is_logged,
    test_receipt_failure_message_is_capped_like_the_terminal_message,
    test_cancel_before_provision_skips_provisioning_entirely,
    test_evaluation_wire_coerces_an_int_score_to_float,
    test_receipt_nulls_an_unacked_transcript_artifact_and_logs_it,
    test_pre_run_provision_failure_emits_the_terminal_before_closing_the_pool,
    test_fence_on_the_manifest_push_still_exits_fenced_with_the_terminal_already_delivered,
    test_drain_alone_delivers_a_pre_run_backlog_larger_than_one_batch,
    test_flush_terminal_alone_must_deliver_the_terminal_before_a_skipped_receipt_under_backlog,
    test_post_terminal_wire_block_is_bounded_by_the_remaining_flush_window,
    test_bundle_without_scenarios_keeps_the_notwired_regression,
    test_default_scenario_source_wires_the_bundle_adapter_when_scenarios_present,
    test_empty_scenario_key_from_bundle_document_fails_cleanly_via_existing_validation,
    test_mutation_adapter_off_makes_the_e2e_test_fail,
    test_bundle_scenario_id_is_assigned_by_registration_and_receipt_now_delivers,
    test_unregistered_scenario_with_empty_scenario_id_receipt_still_drops_safely,
    test_registration_response_mismatch_reaches_the_typed_platform_sync_terminal,
    test_injected_scenario_source_always_wins_over_the_bundle_adapter,
    test_module_level_sys_exit_zero_in_setup_is_contained_as_a_typed_failure,
    test_module_level_sys_exit_three_in_setup_does_not_hijack_the_guests_exit_code,
    test_non_utf8_setup_file_is_contained_as_a_typed_failure_not_an_escape,
]


if __name__ == "__main__":
    failures = 0
    for test_fn in TESTS:
        started = time.monotonic()
        try:
            test_fn()
        except Exception as exc:  # noqa: BLE001 - a direct-invocation runner, not pytest
            failures += 1
            print(f"FAIL {test_fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {test_fn.__name__} ({time.monotonic() - started:.2f}s)")
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    raise SystemExit(1 if failures else 0)


def test_every_runner_log_line_carries_the_job_id(capsys):
    """Concurrent runs are collected into one log stream, so a line with no job id cannot be
    attributed to a run at all."""
    import logging

    from fi.alk.harness.hosted_entrypoint import configure_runner_logging

    root = logging.getLogger()
    saved = list(root.handlers)
    saved_level = root.level
    try:
        root.handlers = []
        configure_runner_logging("job-abc123")
        logging.getLogger("livekit.agents").warning("target disconnected")
        logging.getLogger("fi.alk.harness.hosted_entrypoint").info("stage finished")
        for handler in root.handlers:
            handler.flush()
    finally:
        root.handlers = saved
        root.setLevel(saved_level)

    err = capsys.readouterr().err
    assert "job=job-abc123 livekit.agents: target disconnected" in err
    assert "job=job-abc123" in err.splitlines()[-1]


def test_runner_logging_falls_back_when_the_job_id_is_unknown(capsys):
    """A boot that fails before the id is known must still produce readable lines."""
    import logging

    from fi.alk.harness.hosted_entrypoint import configure_runner_logging

    root = logging.getLogger()
    saved = list(root.handlers)
    saved_level = root.level
    try:
        root.handlers = []
        configure_runner_logging(None)
        logging.getLogger("boot").error("capabilities load failed")
        for handler in root.handlers:
            handler.flush()
    finally:
        root.handlers = saved
        root.setLevel(saved_level)

    assert "job=-" in capsys.readouterr().err


def test_two_callers_do_not_read_the_same_words_at_the_same_pace():
    """Delivery, not content."""
    from fi.alk.harness.simulator_voice import persona_speech_rate

    marcus = persona_speech_rate({"name": "Marcus Thorne"})
    priya = persona_speech_rate({"name": "Priya Sundaram"})

    assert 0.6 <= marcus <= 2.0
    assert 0.6 <= priya <= 2.0
    assert marcus != priya, "every caller would sound identical"


def test_a_persona_keeps_its_pace_across_reruns():
    """A rate that moves between runs makes two recordings of one scenario incomparable."""
    from fi.alk.harness.simulator_voice import persona_speech_rate

    assert persona_speech_rate({"name": "Marcus Thorne"}) == persona_speech_rate(
        {"name": "Marcus Thorne"}
    )
    assert persona_speech_rate({}) == 1.0
    assert persona_speech_rate(None) == 1.0


def test_the_simulator_definition_carries_the_persona_s_pace():
    from fi.alk.harness.simulator_voice import persona_speech_rate, simulator_definition

    definition = simulator_definition(lambda key: "", persona={"name": "Marcus Thorne"})

    assert definition.tts.speed == persona_speech_rate({"name": "Marcus Thorne"})


def test_the_caller_is_given_a_countable_reason_to_lose_patience():
    """Measured across 16 calls: ONE impatience marker, and that on a hostile do-not-call."""
    from fi.alk.harness.simulator_voice import simulator_instructions

    text = simulator_instructions("outbound", "expecting", "", "")

    assert "roughly ten in a row" in text, "the trigger has to be countable, not a mood"
    # Rule 3 forbids volunteering, and asking how much longer IS volunteering, so without an
    # explicit carve-out the two rules contradict and the earlier one wins. Measured: A/B against
    # the composed prompt produced 0 impatience markers over 14 questions until rule 3 said that
    # what it governs is FACTS, not a question about the call itself.
    assert "This governs FACTS about you" in text
    assert "not stop you asking your own question about the call itself" in text
    assert "how many more" in text
    assert "already gave it" in text, "a repeated question must be named as repeated"
    # The rules that protect the intake flow must survive.
    assert "Answer only what was asked, one fact at a time" in text
    assert "close in ONE turn" in text


def test_the_persona_s_pace_reaches_the_speech_provider(monkeypatch):
    """The definition carrying a speed proves nothing on its own: the provider call has to receive
    it. Cartesia documents speed as valid 0.6 to 2.0 for sonic-3, which is the model we use."""
    from types import SimpleNamespace

    from fi.simulate.agent.definition import TTSConfig
    from fi.simulate.simulation import livekit_models

    captured = {}

    def fake_tts(**kwargs):
        captured.update(kwargs)
        return "tts"

    monkeypatch.setattr(
        livekit_models,
        "_import_plugin",
        lambda name: SimpleNamespace(TTS=fake_tts),
    )
    monkeypatch.setenv("CARTESIA_API_KEY", "not-a-real-key")

    livekit_models._cartesia_tts(
        TTSConfig(provider="cartesia", model="sonic-3", voice="abc", speed=1.12),
        http_session=None,
    )

    assert captured["speed"] == 1.12
    assert captured["voice"] == "abc"
    # Absent because this persona has no recognised emotion, not because emotion is unsupported:
    # probing the live API settled that it is accepted, and it is wired from a validated set.
    assert "emotion" not in captured


def test_a_provider_with_no_speed_setting_is_left_alone(monkeypatch):
    """A persona with no rate must not send speed=None into a provider that would reject it."""
    from types import SimpleNamespace

    from fi.simulate.agent.definition import TTSConfig
    from fi.simulate.simulation import livekit_models

    captured = {}

    def fake_tts(**kwargs):
        captured.update(kwargs)
        return "tts"

    monkeypatch.setattr(
        livekit_models,
        "_import_plugin",
        lambda name: SimpleNamespace(TTS=fake_tts),
    )
    monkeypatch.setenv("CARTESIA_API_KEY", "not-a-real-key")

    livekit_models._cartesia_tts(
        TTSConfig(provider="cartesia", model="sonic-3", voice="abc"), http_session=None
    )

    assert "speed" not in captured


def test_the_delivery_a_persona_was_rendered_with_is_recoverable_from_the_log(monkeypatch, caplog):
    """The only record of what the simulator actually sounded like.

    call_metadata reports conversation_speed 1.0 and a constant voice name on every call whatever
    the simulator was given, so without this line a run's real delivery cannot be checked after the
    fact, and the tests below would be the only evidence that either control was ever applied.
    """
    import logging
    from types import SimpleNamespace

    from fi.simulate.agent.definition import TTSConfig
    from fi.simulate.simulation import livekit_models

    monkeypatch.setattr(
        livekit_models,
        "_import_plugin",
        lambda name: SimpleNamespace(TTS=lambda **kwargs: "tts"),
    )
    monkeypatch.setenv("CARTESIA_API_KEY", "not-a-real-key")

    with caplog.at_level(logging.INFO, logger="fi.simulate.simulation.livekit_models"):
        livekit_models._cartesia_tts(
            TTSConfig(
                provider="cartesia",
                model="sonic-3",
                voice="abc",
                speed=1.12,
                emotion=["anger:low"],
            ),
            http_session=None,
        )

    said = caplog.text
    assert "voice=abc" in said
    assert "speed=1.12" in said
    assert "anger:low" in said


def test_every_emotion_we_can_emit_is_one_cartesia_accepts():
    """Established against the live sonic-3 API: it validates the emotion NAME and the LEVEL
    separately and rejects either being wrong with HTTP 400, so a bad value fails the call rather
    than being ignored. The plugin's own TTSVoiceEmotion vocabulary ("Neutral", "Frustrated",
    "Tired") is rejected outright, which is why nothing here is taken from the plugin's types."""
    from fi.alk.harness.simulator_voice import (
        _CARTESIA_EMOTION_LEVELS,
        _CARTESIA_EMOTION_NAMES,
        _PERSONALITY_EMOTION,
        persona_emotion,
    )

    assert _CARTESIA_EMOTION_NAMES == {
        "anger",
        "positivity",
        "surprise",
        "sadness",
        "curiosity",
    }, "fear and disgust are rejected by the API; do not add them without re-testing"
    assert _CARTESIA_EMOTION_LEVELS == {"lowest", "low", "high", "highest"}

    for _words, emotion in _PERSONALITY_EMOTION:
        name, _, level = emotion.partition(":")
        assert name in _CARTESIA_EMOTION_NAMES, emotion
        assert level in _CARTESIA_EMOTION_LEVELS, emotion

    # And nothing unrecognised invents one.
    assert persona_emotion({"personality": "Something nobody mapped"}) == []
    assert persona_emotion({}) == []
    assert persona_emotion(None) == []


def test_two_personalities_do_not_share_one_emotional_register():
    from fi.alk.harness.simulator_voice import persona_emotion

    assert persona_emotion({"personality": "Warm and chatty"}) == ["positivity:high"]
    assert persona_emotion({"personality": "Professional and formal"}) == ["positivity:low"]
    assert persona_emotion({"personality": "Impatient and abrupt"}) == ["anger:low"]
    assert persona_emotion({"personality": "Curious and sceptical"}) == ["curiosity:high"]


def test_the_persona_s_emotion_reaches_the_speech_provider(monkeypatch):
    from types import SimpleNamespace

    from fi.simulate.agent.definition import TTSConfig
    from fi.simulate.simulation import livekit_models

    captured = {}
    monkeypatch.setattr(
        livekit_models, "_import_plugin",
        lambda name: SimpleNamespace(TTS=lambda **kw: captured.update(kw) or "tts"),
    )
    monkeypatch.setenv("CARTESIA_API_KEY", "not-a-real-key")

    livekit_models._cartesia_tts(
        TTSConfig(provider="cartesia", model="sonic-3", voice="abc",
                  speed=1.05, emotion=["anger:low"]),
        http_session=None,
    )

    assert captured["emotion"] == ["anger:low"]
    assert captured["speed"] == 1.05


def test_a_persona_with_no_recognised_emotion_sends_no_emotion_key(monkeypatch):
    """An empty list must not become emotion=[] on the wire; no control is the provider default."""
    from types import SimpleNamespace

    from fi.simulate.agent.definition import TTSConfig
    from fi.simulate.simulation import livekit_models

    captured = {}
    monkeypatch.setattr(
        livekit_models, "_import_plugin",
        lambda name: SimpleNamespace(TTS=lambda **kw: captured.update(kw) or "tts"),
    )
    monkeypatch.setenv("CARTESIA_API_KEY", "not-a-real-key")

    livekit_models._cartesia_tts(
        TTSConfig(provider="cartesia", model="sonic-3", voice="abc", emotion=[]),
        http_session=None,
    )

    assert "emotion" not in captured


def test_the_closing_turn_must_carry_everything_left_to_say():
    """Rule 9 named THANKS specifically and the model did not generalise."""
    from fi.alk.harness.simulator_voice import simulator_instructions

    text = simulator_instructions("outbound", "expecting", "", "")

    assert "EVERYTHING you still" in text
    for kind in ("a thanks", "a last condition", "a reminder", "a warning", "a caveat"):
        assert kind in text, kind
    # The worked example has to show both shapes, or the rule is abstract.
    assert "make sure it stays off the list. Goodbye." in text
    assert "Say your last point BEFORE the farewell" in text
    # And the rules this must not undo.
    assert "Answer only what was asked, one fact at a time" in text
    assert "After your closing turn you say nothing further" in text
