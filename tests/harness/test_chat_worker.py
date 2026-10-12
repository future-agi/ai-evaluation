from __future__ import annotations

import asyncio
import base64
import io
import json
import os
from types import SimpleNamespace

import pytest

from test_chat_call_runner import (
    _Adapter,
    _ToolWorld,
    _context,
    _single_exchange,
)

from fi.alk.harness import chat_call_runner, chat_worker
from fi.alk.harness.hosted_scheduler import CallAborted
from fi.alk.harness.process_runtime import (
    EnvironmentRuntime,
    RuntimeEndpoint,
    RuntimeState,
)
from fi.simulate.agent.wrapper import AgentResponse


def test_isolated_chat_worker_uses_authored_scenario_for_selected_trial(
    tmp_path, monkeypatch
):
    context = _context(tmp_path)

    class Wrapper:
        def __init__(self, **_kwargs):
            pass

        async def call(self, _input):
            return AgentResponse(content="Your account is active.")

    runtime = EnvironmentRuntime(
        runtime_id="runtime-1",
        world_index=0,
        bundle_digest="sha256:" + "a" * 64,
        state=RuntimeState.READY,
        endpoints={
            "target_http": RuntimeEndpoint(
                capability="target_http",
                protocol="http",
                address="http://agent.invalid",
            )
        },
    )
    payload = {
        "job": context.job.model_dump(mode="json"),
        "bundle_dir": str(context.bundle_dir),
        "work_directory": str(context.work_directory),
        "source_directory": None,
        "target_secrets": {},
        "attempt_number": 1,
        "runtime": runtime.model_dump(mode="json"),
        "scenario_key": "trial-key",
        "source_scenario_key": "one",
        "scenario_id": "trial-id",
    }
    result_path = tmp_path / "worker-result.json"
    monkeypatch.setattr(chat_call_runner, "HTTPAgentWrapper", Wrapper)
    monkeypatch.setattr(chat_call_runner, "_tool_world", lambda *_args: _ToolWorld())
    monkeypatch.setattr(chat_call_runner, "_drive_conversation", _single_exchange)
    monkeypatch.setattr(chat_worker.sys, "stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(chat_worker.sys, "argv", ["chat_worker", str(result_path)])

    asyncio.run(chat_worker.main())

    result = json.loads(result_path.read_text())
    assert "error" not in result
    transcript = json.loads(
        base64.b64decode(result["artifacts"][0]["data"], validate=True)
    )
    assert transcript["messages"] == [
        {"role": "user", "content": "Look up account ACC-2048"},
        {"role": "assistant", "content": "Your account is active."},
    ]


def test_parallel_chat_workers_keep_routing_private_and_upload_in_parent(
    tmp_path, monkeypatch
):
    context = _context(tmp_path)
    adapter = _Adapter()
    monkeypatch.setenv("HARNESS_PLATFORM_API_KEY", "parent-only")
    monkeypatch.setenv("AGENTCC_API_KEY", "simulator-model-key")
    monkeypatch.setenv("AGENTCC_BASE_URL", "https://gateway.example")
    monkeypatch.setenv("ALK_CLAUDE_GATEWAY_API_KEY", "simulator-claude-key")
    monkeypatch.setenv("ALK_CLAUDE_GATEWAY_URL", "https://claude.example")
    original = dict(os.environ)
    payloads = []

    async def run():
        both = asyncio.Event()

        async def worker(module, payload, *, environ, work_directory):
            assert "HARNESS_PLATFORM_API_KEY" not in environ
            assert environ["AGENTCC_API_KEY"] == "simulator-model-key"
            assert environ["AGENTCC_BASE_URL"] == "https://gateway.example"
            assert environ["ALK_CLAUDE_GATEWAY_API_KEY"] == "simulator-claude-key"
            assert environ["ALK_CLAUDE_GATEWAY_URL"] == "https://claude.example"
            payloads.append(payload)
            if len(payloads) == 2:
                both.set()
            await asyncio.wait_for(both.wait(), 3)
            return {
                "outcome": {
                    "calls": [],
                    "turns": 1,
                    "started_at": None,
                    "ended_at": None,
                    "duration_ms": 1,
                    "transcript_artifact": "0",
                    "recording_artifacts": [],
                },
                "artifacts": [
                    {
                        "id": "0",
                        "kind": "transcript",
                        "data": base64.b64encode(
                            payload["scenario_key"].encode()
                        ).decode(),
                    }
                ],
            }

        monkeypatch.setattr(chat_worker, "run_json_worker", worker)
        runner = chat_worker.IsolatedChatCallRunner(adapter, context)
        outcomes = await asyncio.gather(
            *(
                runner.run(
                    SimpleNamespace(
                        scenario_key=f"trial-{index}",
                        source_scenario_key=f"authored-{index}",
                        scenario_id=f"id-{index}",
                    ),
                    EnvironmentRuntime(
                        runtime_id=f"runtime-{index}",
                        world_index=index,
                        bundle_digest="sha256:" + "a" * 64,
                        state=RuntimeState.READY,
                    ),
                )
                for index in range(2)
            )
        )
        assert all(
            outcome.transcript_artifact == "transcript-1" for outcome in outcomes
        )

    asyncio.run(run())
    assert sorted(adapter.uploads) == [b"trial-0", b"trial-1"]
    assert {payload["runtime"]["world_index"] for payload in payloads} == {0, 1}
    assert {
        (payload["scenario_key"], payload["source_scenario_key"])
        for payload in payloads
    } == {("trial-0", "authored-0"), ("trial-1", "authored-1")}
    assert dict(os.environ) == original


def test_isolated_chat_runner_preserves_structured_worker_failure(
    tmp_path, monkeypatch
):
    context = _context(tmp_path)

    async def worker(*_args, **_kwargs):
        return {
            "error": "chat_target_failed: upstream unavailable",
            "error_code": "target_agent_failed",
            "error_marker": "target_unavailable",
        }

    monkeypatch.setattr(chat_worker, "run_json_worker", worker)
    runner = chat_worker.IsolatedChatCallRunner(_Adapter(), context)

    with pytest.raises(CallAborted) as error:
        asyncio.run(
            runner.run(
                SimpleNamespace(
                    scenario_key="trial-key",
                    source_scenario_key="one",
                    scenario_id="trial-id",
                ),
                EnvironmentRuntime(
                    runtime_id="runtime-1",
                    world_index=0,
                    bundle_digest="sha256:" + "a" * 64,
                    state=RuntimeState.READY,
                ),
            )
        )

    assert str(error.value) == (
        "chat_worker_failed: chat_target_failed: upstream unavailable"
    )
    assert error.value.code == "target_agent_failed"
    assert error.value.marker == "target_unavailable"


def test_chat_worker_returns_call_aborted_details_only(tmp_path, monkeypatch):
    context = _context(tmp_path)
    runtime = EnvironmentRuntime(
        runtime_id="runtime-1",
        world_index=0,
        bundle_digest="sha256:" + "a" * 64,
        state=RuntimeState.READY,
    )
    payload = {
        "job": context.job.model_dump(mode="json"),
        "bundle_dir": str(context.bundle_dir),
        "work_directory": str(context.work_directory),
        "source_directory": None,
        "target_secrets": {},
        "attempt_number": 1,
        "runtime": runtime.model_dump(mode="json"),
        "scenario_key": "trial-key",
        "source_scenario_key": "one",
        "scenario_id": "trial-id",
    }
    result_path = tmp_path / "worker-error.json"

    async def fail(*_args, **_kwargs):
        raise CallAborted(
            "chat target did not start",
            code="target_agent_failed",
            marker="target_unavailable",
        )

    monkeypatch.setattr(chat_call_runner.HostedChatCallRunner, "run", fail)
    monkeypatch.setattr(chat_worker.sys, "stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(chat_worker.sys, "argv", ["chat_worker", str(result_path)])

    asyncio.run(chat_worker.main())

    assert json.loads(result_path.read_text()) == {
        "error": "chat target did not start",
        "error_code": "target_agent_failed",
        "error_marker": "target_unavailable",
    }
