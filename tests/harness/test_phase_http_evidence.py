import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from fi.alk.harness.hosted_scheduler import HostedScheduler, _run_phase
from fi.alk.harness.local_certified_runtime import LocalCertifiedSink
from fi.alk.harness.scenario_source import _compile_entry
import test_observation_http_sessions as sessions
import test_hosted_scheduler as scheduler_tests
from test_process_observations import hosted

application = sessions.application
phase = sessions.phase


@pytest.mark.parametrize("sink_kind", ["local", "hosted_port"])
@pytest.mark.parametrize("broken", [False, True])
def test_real_scheduler_retains_each_phase_without_private_http_data(
    application, phase, tmp_path, sink_kind, broken, capfd
):
    address, secrets, _ = application
    world = hosted(tmp_path, {"app": address})
    sink = (
        LocalCertifiedSink(tmp_path / "retained")
        if sink_kind == "local"
        else scheduler_tests.FakeOutbound()
    )
    scenario = scheduler_tests.FakeScenario(
        "private-session", "", requires_tool_evidence=False
    )
    scenario.setup = _compile_entry(
        sessions.SETUP + "    print(response)\n", label="setup", entry="setup"
    )
    scenario.ready = _compile_entry(sessions.READY, label="ready", entry="ready")
    source = "def check(world, calls):\n    assert world.request('app', 'GET', '/private/ready')['status'] == 200\n"
    if broken:
        source += "    raise ValueError('deliberate checker fault')\n"
    goal = scheduler_tests.FakeSubGoal("authenticated", None)
    goal.check = _compile_entry(source, label="check", entry="check")
    scenario.sub_goals = [goal]

    class Factory:
        async def create(self, runtime, *, rng):
            return world

    async def run():
        pool, _ = scheduler_tests._pool(1, outbound=sink)
        await pool.start()
        scheduler = HostedScheduler(
            pool=pool,
            world_factory=Factory(),
            call_runner=scheduler_tests.FakeCallRunner(
                {"private-session": scheduler_tests._call_outcome()}
            ),
            outbound=sink,
            job_seed=0,
        )
        try:
            return await scheduler.run([scenario])
        finally:
            await pool.close()

    result = asyncio.run(run())
    assert result.receipts[0].status == ("errored" if broken else "passed")
    if broken:
        assert result.receipts[0].failure.code == "check_broken"
    if sink_kind == "local":
        events = [
            json.loads(line)
            for line in (tmp_path / "retained/events.jsonl").read_text().splitlines()
        ]
        logs = [event["payload"] for event in events if event["type"] == "log"]
    else:
        logs = [value for kind, value in sink.events if kind == "log"]
    evidence = [
        json.loads(item["message"].removeprefix("native_http "))
        for item in logs
        if item["message"].startswith("native_http ")
    ]
    assert [item["phase"] for item in evidence] == ["setup", "setup", "ready", "check"]
    assert all(item["http"]["status"] == 200 for item in evidence)
    assert all(
        item["scenario_key"] == "private-session"
        and item["scenario_attempt"] == 1
        and item["world_index"] == 0
        for item in evidence
    )
    assert evidence[-1]["check"] == "authenticated"
    captured = capfd.readouterr()
    public = json.dumps(logs) + captured.out + captured.err
    assert all(secret not in public for secret in secrets.values())
    assert "/private/ready" not in public


def test_evidence_sink_failure_preserves_checker_result(application, phase, tmp_path):
    address, _, _ = application
    world = hosted(tmp_path, {"app": address})

    async def broken_sink(**kwargs):
        raise OSError("sink unavailable")

    fn = _compile_entry(
        "def check(world):\n    world.request('app', 'GET', '/missing')\n    return 'task failed'\n",
        label="failure",
        entry="check",
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = asyncio.run(
            _run_phase(
                fn,
                world.read_only(),
                phase="check",
                timeout=15,
                executor=executor,
                audit_log=broken_sink,
            )
        )
    assert result.value == "task failed"
    assert result.failure is None


def test_metadata_sink_preserves_hosted_fence(application, phase, tmp_path):
    from fi.alk.harness.outbound import HostedFencedError, ChannelError, ChannelOutcome

    address, _, _ = application
    world = hosted(tmp_path, {"app": address})
    fence = HostedFencedError(
        ChannelError(ChannelOutcome.FENCED, None, "fence_mismatch", "expired")
    )

    async def fenced_sink(**kwargs):
        raise fence

    fn = _compile_entry(
        "def check(world):\n    world.request('app', 'GET', '/missing')\n",
        label="fence",
        entry="check",
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(HostedFencedError) as caught:
            asyncio.run(
                _run_phase(
                    fn,
                    world.read_only(),
                    phase="check",
                    timeout=15,
                    executor=executor,
                    audit_log=fenced_sink,
                )
            )
    assert caught.value is fence
