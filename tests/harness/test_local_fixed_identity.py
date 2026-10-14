import asyncio
import dataclasses
import json
from types import SimpleNamespace

import pytest

from fi.alk.harness import local_certified_runtime as local
from fi.alk.harness.job import HarnessJob
from fi.alk.harness.hosted_scheduler import ReceiptFailure, ResultReceipt
from fi.alk.harness.process_runtime import (
    ProcessRuntimeProvider,
    ProcessRuntimeError,
    default_user_resolver,
    fixed_sandbox_user_resolver,
)


@pytest.fixture
def job(tmp_path):
    return HarnessJob.model_validate({
        "job_id": "local-admission", "run_id": "test-run", "execution": "local",
        "source": {"kind": "local_repository", "local_path": str(tmp_path)},
        "agent": {"connector": "auto"},
        "runtime": {"cpu_units": 4, "memory_mb": 8192, "parallelism": 1},
    })


@pytest.mark.parametrize("fixed_user", [False, True])
def test_local_execution_uses_selected_process_identity(tmp_path, monkeypatch, fixed_user, job):
    monkeypatch.setenv("ALK_FIXED_SANDBOX_USER", "1" if fixed_user else "0")
    monkeypatch.delenv("ALK_SANDBOX_UPLOAD_ROOT", raising=False)
    authoring = tmp_path / "authoring"
    (authoring / "generic-harness" / "certified-bundle").mkdir(parents=True)

    async def validated(*args, **kwargs):
        return None

    monkeypatch.setattr(local, "validate_and_repair", validated)
    monkeypatch.setattr(local, "load_bundle_v2", lambda _: object())
    monkeypatch.setattr(local, "_make_local_seed_files_readable", lambda _: None)

    class ObservedIdentity(Exception):
        pass

    class Provider:
        def __init__(self, **kwargs):
            expected = fixed_sandbox_user_resolver if fixed_user else default_user_resolver
            assert kwargs["user_resolver"] is expected
            raise ObservedIdentity

    monkeypatch.setattr(local, "ProcessRuntimeProvider", Provider)
    with pytest.raises(ObservedIdentity):
        asyncio.run(local.run_local_certified(job, source=tmp_path, authoring=authoring))


@pytest.mark.parametrize("sufficient_capacity", [True, False])
def test_execution_hands_declared_resources_to_admission_and_closes_failed_start(
    tmp_path, monkeypatch, job, sufficient_capacity
):
    monkeypatch.delenv("ALK_SANDBOX_UPLOAD_ROOT", raising=False)
    if not sufficient_capacity:
        job = job.model_copy(update={
            "runtime": job.runtime.model_copy(update={"cpu_units": 1, "memory_mb": 128})
        })
    authoring = tmp_path / "authoring"
    (authoring / "generic-harness" / "certified-bundle").mkdir(parents=True)

    async def validated(*args, **kwargs):
        return None

    monkeypatch.setattr(local, "validate_and_repair", validated)
    monkeypatch.setattr(local, "load_bundle_v2", lambda _: object())
    monkeypatch.setattr(local, "_make_local_seed_files_readable", lambda _: None)
    events = []

    class StartStopped(Exception):
        pass

    class Provider(ProcessRuntimeProvider):
        def __init__(self, **kwargs):
            super().__init__(cpu_observer=lambda: None, mem_observer=lambda: None, **kwargs)

        async def provision(self, bundle, *, work_directory, instances, **kwargs):
            events.append("starting")
            admitted = self._run_admission(instances, work_directory)
            assert admitted == 1
            events.append("admitted")
            raise StartStopped("injected startup failure after admission")

        async def close(self, *, work_directory):
            events.append("closed")

    monkeypatch.setattr(local, "ProcessRuntimeProvider", Provider)
    expected = StartStopped if sufficient_capacity else ProcessRuntimeError
    with pytest.raises(expected) as failure:
        asyncio.run(local.run_local_certified(job, source=tmp_path, authoring=authoring))
    if sufficient_capacity:
        assert events == ["starting", "admitted", "closed"]
    else:
        assert failure.value.code == "resource_capacity_unavailable"
        assert events == ["starting", "closed"]


@pytest.mark.parametrize("status,with_details", [
    ("passed", False), ("failed", False), ("errored", True), ("errored", False),
])
def test_local_run_preserves_scenario_errors_and_evidence(
    tmp_path, monkeypatch, job, status, with_details
):
    monkeypatch.delenv("ALK_SANDBOX_UPLOAD_ROOT", raising=False)
    authoring = tmp_path / "authoring"
    (authoring / "generic-harness" / "certified-bundle").mkdir(parents=True)
    failure = ReceiptFailure(
        domain="simulator", stage="running", code="setup_crashed",
        message="scenario fixture setup crashed",
    ) if with_details else None
    receipt = ResultReceipt(
        scenario_key="scenario-1", scenario_id="scenario-1", scenario_attempt=1,
        world_index=0, status=status, sub_goals=(), evaluations=(), call=None,
        failure=failure,
    )
    events = []

    async def validated(*args, **kwargs):
        pass

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        async def start(self):
            events.append("started")

        async def close(self):
            events.append("closed")

    class Scheduler:
        def __init__(self, *, outbound, **kwargs):
            self.sink = outbound

        async def run(self, scenarios):
            await self.sink.receipt(receipt)
            return SimpleNamespace(aborted=None)

    monkeypatch.setattr(local, "validate_and_repair", validated)
    monkeypatch.setattr(local, "load_bundle_v2", lambda _: SimpleNamespace(
        runtime=SimpleNamespace(evidence_seam=None)
    ))
    monkeypatch.setattr(local, "_make_local_seed_files_readable", lambda _: None)
    monkeypatch.setattr(local, "WorldPool", Pool)
    monkeypatch.setattr(local, "load_scenarios", lambda _: [])
    monkeypatch.setattr(local, "_default_build_call_runner", lambda *args: None)
    monkeypatch.setattr(local, "HostedScheduler", Scheduler)

    if status == "errored":
        with pytest.raises(local.LocalCertifiedRunError) as caught:
            asyncio.run(local.run_local_certified(job, source=tmp_path, authoring=authoring))
        assert caught.value.domain == ("simulator" if with_details else "infrastructure")
        assert caught.value.code == ("setup_crashed" if with_details else "scenario_errored")
        assert str(caught.value) == (
            "scenario fixture setup crashed" if with_details
            else "Scenario scenario-1 errored without failure details"
        )
        assert caught.value.completed_scenarios == 1
    else:
        assert asyncio.run(local.run_local_certified(
            job, source=tmp_path, authoring=authoring
        )) == (0 if status == "passed" else 2, 1)

    run_root, = (authoring / "runs").iterdir()
    expected = json.loads(json.dumps(dataclasses.asdict(receipt)))
    assert json.loads((run_root / "run.json").read_text())["receipts"] == [expected]
    assert json.loads((run_root / "scenario-1" / "result.json").read_text()) == expected
    assert events == ["started", "closed"]
