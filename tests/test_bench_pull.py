"""Tests for the pull / RL control mode (agent drives a simulated env via reset/step)."""

from __future__ import annotations

from pathlib import Path

import pytest

from fi.alk import bench
from fi.alk.bench import _pull

ROOT = Path(__file__).parent.parent
PULL_SUITE = ROOT / "examples" / "bench_suites" / "pull_starter.json"


def test_reference_policy_solves_all() -> None:
    r = bench.run_bench(PULL_SUITE, {"type": "reference"}, control_mode="pull",
                        evidence_class="local_gate", emit_telemetry=False)
    assert r["modalities"] == ["rl"]
    assert r["aggregate"]["pass_rate"] == 1.0
    assert all(row["verdict"] == "pass" for row in r["per_task"])
    assert all(row["world_kind"] == "env" for row in r["per_task"])


def test_noop_policy_fails_all() -> None:
    r = bench.run_bench(PULL_SUITE, {"type": "noop"}, control_mode="pull", emit_telemetry=False)
    assert all(row["verdict"] == "fail" for row in r["per_task"])
    assert r["aggregate"]["pass_rate"] == 0.0


def test_custom_callable_policy() -> None:
    # a callable obs->action policy: move right (solves reach_target).
    r = bench.run_bench(PULL_SUITE, lambda obs: "right", control_mode="pull",
                        max_tasks=1, emit_telemetry=False)
    assert r["per_task"][0]["verdict"] == "pass"


def test_pull_requires_pull_control_mode() -> None:
    with pytest.raises(bench.BenchError):
        bench.run_bench(PULL_SUITE, {"type": "reference"}, control_mode="artifact_in",
                        submission={}, emit_telemetry=False)


def test_pull_requires_agent() -> None:
    with pytest.raises(bench.BenchError):
        bench.run_bench(PULL_SUITE, None, control_mode="pull", emit_telemetry=False)


def test_unknown_env_is_void() -> None:
    suite = {
        "kind": "agent-learning.bench-suite.v1", "control": "pull", "name": "x",
        "tasks": [{"id": "t", "instruction": "i", "env": {"kind": "nope"},
                   "guards": {"min_guard_count": 1}}],
    }
    r = bench.run_bench(suite, {"type": "reference"}, control_mode="pull", emit_telemetry=False)
    assert r["per_task"][0]["verdict"] == "void"
    assert r["aggregate"]["scored"] == 0


def test_envs_are_deterministic_and_solvable() -> None:
    for kind in ("reach_target", "guess_number"):
        env = _pull.ENVIRONMENTS[kind]()
        state, obs = env.reset({})
        done = False
        steps = 0
        while not done and steps < 60:
            state, obs, reward, done, info = env.step(state, str(env.optimal_action(obs)))
            steps += 1
        assert info.get("reached") is True, f"{kind} optimal policy did not solve it"
