import asyncio
import json
from pathlib import Path

import pytest

from fi.alk.harness.executor import HarnessExecutor
from fi.alk.harness.job import HarnessJob, HarnessStage
from fi.alk.harness.local_certified_runtime import LocalCertifiedRunError
from fi.alk.harness.process_runtime import ProcessRuntimeError


@pytest.mark.parametrize("partial_run", [False, True])
def test_authored_scenarios_are_not_completed_calls(tmp_path, monkeypatch, partial_run):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"

    async def authored(args):
        Path(args.out).mkdir()
        (Path(args.out) / "scenarios.json").write_text(json.dumps([{"name": "authored"}]))
        return 0

    async def failed_runtime(*args, **kwargs):
        if partial_run:
            raise LocalCertifiedRunError(
                "infrastructure", "runtime_stopped", "stopped after one call",
                completed_scenarios=1,
            )
        raise ProcessRuntimeError(
            "admission", "resource_capacity_unknown", "resource bounds unavailable"
        )

    monkeypatch.setattr("fi.alk.harness.cli._auto", authored)
    monkeypatch.setattr(
        "fi.alk.harness.local_certified_runtime.run_local_certified", failed_runtime
    )
    job = HarnessJob(
        job_id="completed-count", run_id="completed-count-run", execution="local",
        source={"kind": "local_repository", "local_path": str(source)},
        agent={"connector": "auto"}, scenario_count=1,
        metadata={"generic_harness_v1": True},
    )
    status = asyncio.run(HarnessExecutor().run(job, source=source, output=output))
    assert status.stage is HarnessStage.FAILED
    assert status.total_scenarios == 1
    assert status.completed_scenarios == (1 if partial_run else 0)
    assert status.failure.code == ("runtime_stopped" if partial_run else "resource_capacity_unknown")
