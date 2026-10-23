from __future__ import annotations

import asyncio
import base64
import dataclasses
import io
import json
import os
from datetime import datetime
from types import SimpleNamespace
from unittest import mock

import pytest

from test_chat_call_runner import (
    _Adapter,
    _ToolWorld,
    _context,
    _single_exchange,
)

from fi.alk.harness import chat_call_runner, chat_worker
from fi.alk.harness.run.conversation import FINISHED, Exchange, Transcript
from fi.alk.harness.hosted_scheduler import CallAborted
from fi.alk.harness.process_runtime import (
    EnvironmentRuntime,
    RuntimeEndpoint,
    RuntimeState,
)
from fi.alk.harness.usage import UsageDenied, failure_domain_for_code
from fi.simulate.agent.wrapper import AgentResponse


class _Reporter:
    """Records the order of usage checks, worker starts and usage records."""

    def __init__(self, *, deny: bool = False) -> None:
        self.deny = deny
        self.events: list[tuple[str, object]] = []

    def check(self, action):
        self.events.append(("check", action))
        if self.deny:
            raise UsageDenied({"reason": "limit_reached"})

    def record(self, **fields):
        self.events.append(("record", fields))


def _in_process_worker(tmp_path, events):
    """Run the real worker entry point with only the environment the parent hands it."""

    async def worker(module, payload, *, environ, work_directory):
        assert module == "fi.alk.harness.chat_worker"
        events.append(("worker", payload["scenario_key"]))
        result_path = tmp_path / f"worker-result-{len(events)}.json"
        with (
            mock.patch.dict(os.environ, environ, clear=True),
            mock.patch.object(
                chat_worker.sys, "stdin", io.StringIO(json.dumps(payload))
            ),
            mock.patch.object(
                chat_worker.sys, "argv", ["chat_worker", str(result_path)]
            ),
        ):
            await chat_worker.main()
        return json.loads(result_path.read_text())

    return worker


def _metered_run(tmp_path, monkeypatch, reporter, conversation):
    class Wrapper:
        def __init__(self, **_kwargs):
            pass

    context = dataclasses.replace(_context(tmp_path), usage_reporter=reporter)
    monkeypatch.setattr(chat_call_runner, "HTTPAgentWrapper", Wrapper)
    monkeypatch.setattr(chat_call_runner, "_tool_world", lambda *_args: _ToolWorld())
    monkeypatch.setattr(chat_call_runner, "_drive_conversation", conversation)
    monkeypatch.setattr(
        chat_worker, "run_json_worker", _in_process_worker(tmp_path, reporter.events)
    )
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
    return asyncio.run(
        chat_worker.IsolatedChatCallRunner(_Adapter(), context).run(
            SimpleNamespace(
                scenario_key="trial-key",
                source_scenario_key="one",
                scenario_id="trial-id",
            ),
            runtime,
        )
    )


async def _conversation_with_tokens(_target, scenario, _contract, _bundle_dir):
    return Transcript(
        exchanges=[
            Exchange("customer", scenario.instruction),
            Exchange("agent", "Done."),
        ],
        ended=FINISHED,
        simulator_input_tokens=120,
        simulator_output_tokens=30,
    )


@pytest.mark.parametrize(
    ("funding", "events"),
    [
        ("platform", ["check", "worker", "record"]),
        ("customer", ["worker", "record"]),
    ],
)
def test_isolated_chat_call_is_checked_before_the_worker_and_billed_after_it(
    tmp_path, monkeypatch, funding, events
):
    monkeypatch.setenv("ALK_SIMULATOR_FUNDING", funding)
    reporter = _Reporter()

    _metered_run(tmp_path, monkeypatch, reporter, _conversation_with_tokens)

    assert [kind for kind, _ in reporter.events] == events
    (record,) = [fields for kind, fields in reporter.events if kind == "record"]
    assert isinstance(record.pop("occurred_at"), datetime)
    assert record == {
        "action": "text_call",
        "scenario_key": "trial-key",
        "amount": 150,
        "funding": funding,
        "outcome": "completed",
        "failure_domain": None,
    }


def test_failed_isolated_chat_call_bills_the_tokens_spent_before_it_failed(
    tmp_path, monkeypatch
):
    reporter = _Reporter()

    async def fail_midway(_target, scenario, _contract, _bundle_dir):
        error = ConnectionError("agent hung up")
        error.partial_transcript = Transcript(
            exchanges=[Exchange("customer", scenario.instruction)],
            simulator_input_tokens=40,
            simulator_output_tokens=2,
        )
        raise error

    with pytest.raises(CallAborted) as aborted:
        _metered_run(tmp_path, monkeypatch, reporter, fail_midway)

    assert aborted.value.code == "target_agent_failed"
    (record,) = [fields for kind, fields in reporter.events if kind == "record"]
    assert record["amount"] == 42
    assert record["outcome"] == "failed"
    assert record["failure_domain"] == failure_domain_for_code("target_agent_failed")


def test_denied_usage_stops_an_isolated_chat_call_before_the_worker_starts(
    tmp_path, monkeypatch
):
    reporter = _Reporter(deny=True)

    with pytest.raises(CallAborted) as aborted:
        _metered_run(tmp_path, monkeypatch, reporter, _conversation_with_tokens)

    assert aborted.value.code == "usage_exhausted"
    assert reporter.events == [("check", "text_call")]


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
        "usage": [],
    }
