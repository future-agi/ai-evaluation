"""Hosted-runner (plan §9) SDK tests: job contract, offline child run, and the
sink's pre-created-execution routing."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from fi.simulate.hosted.child_entrypoint import main as child_main
from fi.simulate.hosted.job import RunnerMode, StartRunnerJob
from fi.simulate.results.futureagi import FutureAGIResultSink
from fi.simulate.runtime import new_run_id
from fi.simulate.runtime.runner import SimulationRunner
from fi.simulate.runtime.spec import (
    AgentEndpointSpec,
    EnvironmentSpec,
    EvidencePolicy,
    SimulationSpec,
    SimulatorPolicySpec,
)
from fi.simulate.simulation.models import Persona, Scenario

_ECHO_SOURCE = (
    "def reply(input):\n"
    "    msgs = getattr(input, 'messages', None) or []\n"
    "    last = msgs[-1]['content'] if msgs else 'hello'\n"
    "    return f'Thanks for reaching out about: {last}. I can help with that.'\n"
)


def _echo_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    module_path = tmp_path / "hosted_echo_agent.py"
    module_path.write_text(_ECHO_SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return "hosted_echo_agent:reply"


def _chat_spec(target_adapter: str, target_config: dict) -> SimulationSpec:
    scenario = Scenario(
        name="refund-help",
        dataset=[
            Persona(
                persona={"name": "Sam"},
                situation="I was double charged on my invoice.",
                outcome="Get a refund confirmation.",
            )
        ],
    )
    return SimulationSpec(
        run_id=new_run_id(),
        environment=EnvironmentSpec(
            adapter="chat",
            world_kind="conversation",
            config={"max_turns": 4, "min_turns": 2, "modality": "text"},
        ),
        target=AgentEndpointSpec(adapter=target_adapter, config=target_config),
        simulator=SimulatorPolicySpec(adapter="synthetic_user"),
        scenario=scenario,
        evidence=EvidencePolicy(),
    )


def test_chat_runner_job_roundtrips():
    spec = _chat_spec("callable", {"target": "mod:fn"})
    job = StartRunnerJob(job_id="job-1", mode=RunnerMode.CHAT, spec=spec)
    restored = StartRunnerJob.model_validate_json(job.model_dump_json())
    assert restored.job_id == "job-1"
    assert restored.mode is RunnerMode.CHAT
    assert restored.mode.needs_phone is False
    assert restored.spec.environment.adapter == "chat"


def test_voice_runner_job_roundtrips_and_requires_voice_config():
    from fi.simulate.hosted.job import VoiceRunConfig

    voice = VoiceRunConfig(
        agent_definition={
            "name": "vx",
            "system_prompt": "p",
            "transport": {"kind": "sip_outbound"},
        },
        scenario={
            "name": "s",
            "dataset": [{"persona": {"name": "x"}, "situation": "y", "outcome": "z"}],
        },
    )
    job = StartRunnerJob(job_id="v1", mode=RunnerMode.VOICE_SIP, voice=voice)
    restored = StartRunnerJob.model_validate_json(job.model_dump_json())
    assert restored.mode.is_voice is True
    assert restored.mode.needs_phone is True
    assert restored.voice.agent_definition["transport"]["kind"] == "sip_outbound"

    with pytest.raises(Exception):
        StartRunnerJob(job_id="bad", mode=RunnerMode.VOICE_WEBRTC)  # no voice cfg


def test_child_entrypoint_chat_completes_offline(tmp_path, monkeypatch):
    # A callable target runs caller code in-process — allowed here only via the
    # explicit trusted-default escape (an operator-configured local target).
    monkeypatch.setenv("ALK_UNSAFE_INPROCESS_CODE_ACTORS", "true")
    target = _echo_module(tmp_path, monkeypatch)
    run_root = tmp_path / "runs"
    spec = _chat_spec("callable", {"target": target})
    job = StartRunnerJob(
        job_id="job-offline",
        mode=RunnerMode.CHAT,
        spec=spec,
        sink={"root_directory": str(run_root)},
    )
    job_path = tmp_path / "job.json"
    job_path.write_text(job.model_dump_json(), encoding="utf-8")

    # No FI_* creds -> sink records not_configured, run still completes.
    for var in (
        "FI_API_KEY",
        "FI_SECRET_KEY",
        "FI_BASE_URL",
        "FI_TEST_EXECUTION_ID",
        "FI_INTERNAL_SUBMIT_SECRET",
        "ALK_RUNNER_INTERNAL_SECRET",
        "INTERNAL_API_SECRET",
    ):
        monkeypatch.delenv(var, raising=False)

    rc = child_main([str(job_path), "--status-file", str(tmp_path / "status.jsonl")])
    assert rc == 0

    report = json.loads((run_root / spec.run_id / "report.json").read_text())
    assert report["status"] == "completed"
    submission = json.loads((run_root / spec.run_id / "submission.json").read_text())
    assert submission["status"] == "not_configured"


def test_sink_submits_into_pre_created_execution(tmp_path, monkeypatch):
    target = _echo_module(tmp_path, monkeypatch)
    spec = _chat_spec("callable", {"target": target})

    sink = FutureAGIResultSink(
        root=str(tmp_path / "runs"),
        api_url="http://localhost:8000",
        run_test_id="rt-1",
        test_execution_id="te-1",
    )
    report = asyncio.run(
        SimulationRunner().run(spec, target=_import(target), result_sink=sink)
    )

    seen = {"create": [], "batch": [], "batch_counts": [], "result": []}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/test-executions/"):
            seen["create"].append(path)
            return httpx.Response(200, json={"result": {"test_execution_id": "BAD"}})
        if path.endswith("/batch/"):
            seen["batch"].append(path)
            seen["batch_counts"].append(json.loads(request.content)["count"])
            return httpx.Response(
                200,
                json={"result": {"call_execution_ids": ["ce-1"], "has_more": False}},
            )
        if path.endswith("/result/"):
            seen["result"].append(path)
            return httpx.Response(
                200,
                json={
                    "result": {
                        "call_execution_id": "ce-1",
                        "status": "ingested",
                        "eval_dispatched": True,
                    }
                },
            )
        return httpx.Response(404)

    original_client = httpx.Client

    def client_factory(**kwargs):
        kwargs.setdefault("transport", httpx.MockTransport(handler))
        return original_client(**kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    monkeypatch.setenv("FI_API_KEY", "k")
    monkeypatch.setenv("FI_SECRET_KEY", "s")

    outcome = sink.submit(report)

    assert outcome["status"] == "submitted"
    # Pre-created execution -> create endpoint never hit; batch targets te-1.
    assert seen["create"] == []
    assert any("te-1" in path for path in seen["batch"])
    assert seen["batch_counts"] == [len(report.test_cases)]
    assert seen["result"], "expected a result PATCH per test case"


def _canonical_case(index: int):
    from fi.simulate.runtime import TestCaseStatus
    from fi.simulate.runtime.report import SimulationTestCaseResult
    from fi.simulate.simulation.models import Persona as _Persona
    from fi.simulate.simulation.models import TestCaseResult

    persona = _Persona(
        persona={"name": f"Caller {index}"}, situation="s", outcome="o"
    )
    return SimulationTestCaseResult(
        test_case_id=f"tc-{index}",
        status=TestCaseStatus.COMPLETED,
        persona=persona,
        result=TestCaseResult(persona=persona, transcript="hi", messages=[]),
    )


def _mock_streaming_client(monkeypatch, seen, *, result_status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/test-executions/"):
            seen["create"].append(path)
            return httpx.Response(200, json={"result": {"test_execution_id": "BAD"}})
        if path.endswith("/batch/"):
            seen["batch"].append(path)
            return httpx.Response(
                200,
                json={
                    "result": {
                        "call_execution_ids": ["ce-0", "ce-1", "ce-2"],
                        "has_more": False,
                    }
                },
            )
        if path.endswith("/status/"):
            seen.setdefault("status", []).append(path)
            return httpx.Response(200, json={"result": {"updated": True}})
        if path.endswith("/result/"):
            seen["result"].append(path)
            if result_status >= 400:
                return httpx.Response(result_status, json={"detail": "down"})
            return httpx.Response(200, json={"result": {"status": "ingested"}})
        return httpx.Response(404)

    original_client = httpx.Client

    def client_factory(**kwargs):
        kwargs.setdefault("transport", httpx.MockTransport(handler))
        return original_client(**kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)


def test_sink_streams_each_case_and_finalizes_submitted(tmp_path, monkeypatch):
    from types import SimpleNamespace

    sink = FutureAGIResultSink(
        root=str(tmp_path / "runs"),
        api_url="http://localhost:8000",
        run_test_id="rt-1",
        test_execution_id="te-1",
    )
    spec = _chat_spec("callable", {"target": "x:y"})
    sink.prepare(spec)

    seen = {"create": [], "batch": [], "result": []}
    _mock_streaming_client(monkeypatch, seen)
    monkeypatch.delenv("FI_API_KEY", raising=False)
    monkeypatch.delenv("FI_SECRET_KEY", raising=False)
    monkeypatch.setenv("INTERNAL_API_SECRET", "internal-service-secret")

    # Hosted gate opens; rows allocated up front, create endpoint untouched.
    assert sink.begin_stream(spec) is True
    assert seen["create"] == []
    assert any("te-1" in path for path in seen["batch"])

    cases = [_canonical_case(i) for i in range(3)]
    # Stream slots 0 and 2 as they "finish"; slot 1 is left for finalize.
    sink.submit_case(0, cases[0])
    sink.submit_case(2, cases[2])
    assert len(seen["result"]) == 2

    report = SimpleNamespace(run_id="r", report_hash="h", test_cases=cases)
    submission = sink.finalize_stream(report)

    # finalize reconciles the missed slot -> three PATCHes total, all submitted.
    assert len(seen["result"]) == 3
    assert submission["status"] == "submitted"
    assert submission["streamed"] is True
    assert sorted(submission["submitted_call_executions"]) == [
        "ce-0",
        "ce-1",
        "ce-2",
    ]
    assert submission["failed_call_executions"] == []
    # submission.json on disk says submitted — child_entrypoint gates on this.
    on_disk = json.loads((sink.run_directory / "submission.json").read_text())
    assert on_disk["status"] == "submitted"


def test_finalize_reports_failed_when_every_case_fails_to_submit(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    sink = FutureAGIResultSink(
        root=str(tmp_path / "runs"),
        api_url="http://localhost:8000",
        run_test_id="rt-1",
        test_execution_id="te-1",
    )
    spec = _chat_spec("callable", {"target": "x:y"})
    sink.prepare(spec)

    seen = {"create": [], "batch": [], "result": []}
    _mock_streaming_client(monkeypatch, seen, result_status=500)
    monkeypatch.setenv("FI_API_KEY", "k")
    monkeypatch.setenv("FI_SECRET_KEY", "s")

    assert sink.begin_stream(spec) is True
    cases = [_canonical_case(i) for i in range(3)]
    for i, case in enumerate(cases):
        sink.submit_case(i, case)
    # Backend rejects every PATCH -> nothing landed.
    assert sink._streamed_indices == set()

    report = SimpleNamespace(run_id="r", report_hash="h", test_cases=cases)
    submission = sink.finalize_stream(report)

    # A total submission failure must surface as FAILED, not a green empty job —
    # child_entrypoint would otherwise report COMPLETED with zero results.
    assert submission["status"] == "failed"
    assert submission["submitted_call_executions"] == []
    assert len(submission["failed_call_executions"]) == 3
    assert submission["failed_call_executions"][0]["status_code"] == 500


def test_runner_streaming_callback_patches_by_index(tmp_path, monkeypatch):
    from fi.simulate.simulation.models import Persona, TestCaseResult

    sink = FutureAGIResultSink(
        root=str(tmp_path / "runs"),
        api_url="http://localhost:8000",
        run_test_id="rt-1",
        test_execution_id="te-1",
    )
    spec = _chat_spec("callable", {"target": "x:y"})
    sink.prepare(spec)

    seen = {"create": [], "batch": [], "result": []}
    _mock_streaming_client(monkeypatch, seen)
    monkeypatch.setenv("FI_API_KEY", "k")
    monkeypatch.setenv("FI_SECRET_KEY", "s")

    # The runner's real closures: (on_case_start, on_case_complete). The complete
    # closure does begin_stream + legacy->canonical + to_thread PATCH; the start
    # closure PATCHes the row ONGOING when its case begins.
    on_case_start, on_case_complete = SimulationRunner()._begin_streaming(
        sink, spec, None
    )
    assert on_case_complete is not None
    assert on_case_start is not None  # sink exposes case_started

    persona = Persona(persona={"name": "C0"}, situation="s", outcome="o")
    legacy = TestCaseResult(
        persona=persona, transcript="hi", messages=[], metadata={"status": "completed"}
    )
    asyncio.run(on_case_start(0))
    asyncio.run(on_case_complete(0, legacy))

    assert any(path.endswith("/status/") for path in seen.get("status", []))
    assert any(path.endswith("/result/") for path in seen["result"])
    assert 0 in sink._streamed_indices


def test_begin_stream_is_noop_for_local_run_without_execution(tmp_path):
    # No pre-created test_execution_id -> local/chat run keeps the batch-at-end
    # path; streaming never activates.
    sink = FutureAGIResultSink(
        root=str(tmp_path / "runs"),
        api_url="http://localhost:8000",
        run_test_id="rt-1",
    )
    spec = _chat_spec("callable", {"target": "x:y"})
    sink.prepare(spec)
    assert sink.begin_stream(spec) is False
    assert sink._streaming is False


def test_result_sink_prefers_customer_api_pair_over_internal_bearer():
    from fi.simulate.results.futureagi import _open_client

    with _open_client(
        "http://localhost:8000",
        "customer-key",
        "customer-secret",
        "runner-internal-secret",
    ) as client:
        assert client.headers["x-api-key"] == "customer-key"
        assert client.headers["x-secret-key"] == "customer-secret"
        assert "Authorization" not in client.headers


def test_result_sink_uses_internal_bearer_without_customer_api_pair():
    from fi.simulate.results.futureagi import _open_client

    with _open_client(
        "http://localhost:8000", None, None, "runner-internal-secret"
    ) as client:
        assert client.headers["Authorization"] == "Bearer runner-internal-secret"
        assert "x-api-key" not in client.headers
        assert "x-secret-key" not in client.headers


def test_sink_uploads_stereo_recording_and_sets_url(tmp_path, monkeypatch):
    import wave

    import numpy as np

    from fi.simulate.runtime import TestCaseStatus
    from fi.simulate.runtime.report import SimulationTestCaseResult
    from fi.simulate.simulation.models import Persona, TestCaseResult

    sink = FutureAGIResultSink(
        root=str(tmp_path / "runs"),
        api_url="http://localhost:8000",
        run_test_id="rt-1",
        test_execution_id="te-1",
    )
    spec = _chat_spec("callable", {"target": "x:y"})
    sink.prepare(spec)

    stereo_path = tmp_path / "stereo.wav"
    with wave.open(str(stereo_path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(
            np.array([1000, 2000, 1000, 2000], dtype=np.int16).tobytes()
        )

    seen = {"batch": [], "recording": [], "result": []}
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/batch/"):
            seen["batch"].append(path)
            return httpx.Response(
                200,
                json={"result": {"call_execution_ids": ["ce-0"], "has_more": False}},
            )
        if path.endswith("/recording/"):
            seen["recording"].append(path)
            return httpx.Response(
                200, json={"result": {"recording_url": "https://cdn/stereo.wav"}}
            )
        if path.endswith("/result/"):
            seen["result"].append(path)
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"result": {"status": "ingested"}})
        return httpx.Response(404)

    original_client = httpx.Client

    def client_factory(**kwargs):
        kwargs.setdefault("transport", httpx.MockTransport(handler))
        return original_client(**kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    monkeypatch.setenv("FI_API_KEY", "k")
    monkeypatch.setenv("FI_SECRET_KEY", "s")

    assert sink.begin_stream(spec) is True

    persona = Persona(persona={"name": "C0"}, situation="s", outcome="o")
    case = SimulationTestCaseResult(
        test_case_id="tc-0",
        status=TestCaseStatus.COMPLETED,
        persona=persona,
        result=TestCaseResult(
            persona=persona,
            transcript="hi",
            messages=[],
            audio_stereo_path=str(stereo_path),
        ),
    )
    sink.submit_case(0, case)

    # Only the stereo file exists -> exactly one /recording/ POST, and the PATCH
    # carries stereo_recording_url (combined/output/input are absent).
    assert len(seen["recording"]) == 1
    assert captured["payload"].get("stereo_recording_url") == "https://cdn/stereo.wav"
    assert "recording_url" not in captured["payload"]


def test_livekit_run_stamps_provider_marker_for_role_resolution():
    from fi.simulate.results.futureagi import _build_result_payload
    from fi.simulate.runtime import TestCaseStatus
    from fi.simulate.runtime.report import SimulationTestCaseResult
    from fi.simulate.simulation.models import Persona, TestCaseResult

    persona = Persona(persona={"name": "C"}, situation="s", outcome="o")
    result = TestCaseResult(
        persona=persona,
        transcript="hi",
        messages=[],
        metadata={"engine": "livekit"},
    )
    case = SimulationTestCaseResult(
        test_case_id="tc",
        status=TestCaseStatus.COMPLETED,
        persona=persona,
        result=result,
    )

    payload = _build_result_payload(case)

    # A truthy livekit marker lets the platform SpeakerRoleResolver detect the
    # provider as LiveKit even for a black-box target with no usage evidence, so
    # eval transcript labels don't fall back to the VAPI-inbound (swapped) map.
    pcd = payload.get("provider_call_data") or {}
    assert pcd.get("livekit")


def _import(ref: str):
    import importlib

    module_name, _, attr = ref.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def test_child_cancel_propagates_to_run_task(tmp_path, monkeypatch):
    # Regression: cancelling _execute left run_task alive, so asyncio.run's
    # shutdown waited on the engine forever and the child leaked past SIGTERM.
    from fi.simulate.hosted import child_entrypoint as ce

    spec = _chat_spec("callable", {"target": "mod:fn"})
    job = StartRunnerJob(
        job_id="job-cancel",
        mode=RunnerMode.CHAT,
        spec=spec,
        sink={"root_directory": str(tmp_path / "runs")},
    )
    reporter = ce._StatusReporter("job-cancel", tmp_path / "status.jsonl")
    state: dict[str, bool] = {}

    class _ParkedRunner:
        async def run(self, *args, **kwargs):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                state["run_task_cancelled"] = True
                raise

    monkeypatch.setattr(ce, "SimulationRunner", _ParkedRunner)
    monkeypatch.setattr(ce, "resolve_chat_target", lambda _spec: object())

    async def scenario() -> None:
        task = asyncio.create_task(ce._execute(job, reporter))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert state.get("run_task_cancelled") is True
    statuses = [
        json.loads(line)
        for line in (tmp_path / "status.jsonl").read_text().splitlines()
    ]
    assert statuses[-1]["phase"] == "canceled"
