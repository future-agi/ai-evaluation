from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from fi.alk.harness import local_certified_runtime as local
from fi.alk.harness.job import HarnessJob
from fi.alk.harness.outbound import ArtifactKind
from fi.alk.harness.process_diagnostics import upload_process_logs
from fi.simulate.agent.wrapper import AgentInput
from fi.simulate.agent.wrappers.http import HTTPAgentWrapper


@pytest.mark.parametrize("failure_stage", ["http", "setup"])
def test_setup_capture_is_redacted_in_later_failure(monkeypatch, failure_stage):
    token = "session-token-only-returned-by-setup"
    setup = [{"path": "/session", "capture": {"session": "credentials"}}]
    if failure_stage == "setup":
        setup.append({"path": "/configure"})
    wrapper = HTTPAgentWrapper(
        endpoint="http://example.invalid/chat",
        secret_values={},
        setup_requests=setup,
    )

    def request(endpoint, **kwargs):
        if endpoint.endswith("/session"):
            return 200, {"credentials": {"token": token}}
        wrapper._response_status = 500
        wrapper._response_text = json.dumps({"error": token})
        return 500, {"error": token}

    monkeypatch.setattr(wrapper, "_request_json", request)
    response = asyncio.run(
        wrapper.call(AgentInput(thread_id="t", messages=[], new_message={}))
    )
    trace = response.metadata["external_agent"]
    assert trace["failure_stage"] == failure_stage
    assert trace["status_code"] == 500
    assert token not in trace["error"]
    assert token not in trace["response_excerpt"]


def test_local_sink_keeps_distinct_process_artifacts(tmp_path):
    logs = tmp_path / "logs" / "process-logs"
    logs.mkdir(parents=True)
    (logs / "first.log").write_text("first process")
    (logs / "second.log").write_text("second process")
    sink = local.LocalCertifiedSink(tmp_path / "run")
    asyncio.run(upload_process_logs(logs.parent, sink))
    assert sorted(path.read_bytes() for path in (sink.root / "run").glob("*.bin")) == [
        b"first.log\nfirst process",
        b"second.log\nsecond process",
    ]
    first = asyncio.run(sink.upload_artifact(b"first", kind=ArtifactKind.LOG))
    second = asyncio.run(sink.upload_artifact(b"second", kind=ArtifactKind.LOG))
    assert first != second
    assert Path(first).read_bytes() == b"first"
    assert Path(second).read_bytes() == b"second"


def test_local_runs_collect_only_their_own_logs_even_with_same_clock(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("ALK_SANDBOX_UPLOAD_ROOT", raising=False)
    authoring = tmp_path / "authoring"
    (authoring / "generic-harness" / "certified-bundle").mkdir(parents=True)
    prior = authoring / "generic-harness" / "process-logs"
    prior.mkdir()
    (prior / "old.log").write_text("old run evidence")
    roots = []
    sinks = []

    class FixedClock:
        @staticmethod
        def now(_zone):
            return datetime(2026, 10, 14, tzinfo=timezone.utc)

    class Provider:
        def __init__(self, **kwargs):
            self.root = kwargs["generic_artifact_root"]
            self.index = len(roots)
            roots.append(self.root)

    class StoppedAfterStart(Exception):
        pass

    class Pool:
        def __init__(self, provider, *, outbound, **kwargs):
            self.provider = provider
            sinks.append(outbound)

        async def start(self):
            raise StoppedAfterStart

        async def close(self):
            logs = self.provider.root / "process-logs"
            logs.mkdir(parents=True, exist_ok=True)
            (logs / f"current-{self.provider.index}.log").write_text(
                f"run {self.provider.index}"
            )

    async def validated(*args, **kwargs):
        pass

    monkeypatch.setattr(local, "datetime", FixedClock)
    monkeypatch.setattr(local, "validate_and_repair", validated)
    from fi.alk.harness.bundle_v2 import RuntimeKindV2

    monkeypatch.setattr(
        local, "load_bundle_v2",
        lambda _: SimpleNamespace(runtime=SimpleNamespace(kind=RuntimeKindV2.PROCESS)),
    )
    monkeypatch.setattr(local, "_make_local_seed_files_readable", lambda _: None)
    monkeypatch.setattr(local, "ProcessRuntimeProvider", Provider)
    monkeypatch.setattr(local, "WorldPool", Pool)
    job = HarnessJob.model_validate(
        {
            "job_id": "local-logs",
            "run_id": "test-run",
            "execution": "local",
            "source": {"kind": "local_repository", "local_path": str(tmp_path)},
            "agent": {"connector": "auto"},
            "runtime": {"cpu_units": 4, "memory_mb": 8192, "parallelism": 1},
        }
    )
    for _ in range(2):
        with pytest.raises(StoppedAfterStart):
            asyncio.run(
                local.run_local_certified(job, source=tmp_path, authoring=authoring)
            )
    assert roots[0] != roots[1]
    assert sinks[0].root != sinks[1].root
    for index, sink in enumerate(sinks):
        assert [path.read_bytes() for path in (sink.root / "run").glob("*.bin")] == [
            f"current-{index}.log\nrun {index}".encode(),
        ]
    assert (prior / "old.log").read_text() == "old run evidence"
