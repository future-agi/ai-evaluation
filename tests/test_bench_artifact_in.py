"""Tests for the bench artifact_in coding lane (phase 15B).

Covers the code-tests verifier (accepts gold, fails broken / fake-success /
timeout), the coding suite loader/validation, the artifact_in run path, and the
``bench_contract_readiness`` release gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fi.alk import bench, trinity
from fi.alk.bench import _coding
from fi.alk.bench._codeexec import run_code_tests

ROOT = Path(__file__).parent.parent
SUITE = ROOT / "examples" / "bench_suites" / "coding_starter.json"

_CHECKS = (
    "import solution\n\n"
    "def check_a():\n    assert solution.f(2) == 4\n\n"
    "def check_b():\n    assert solution.f(3) == 9\n"
)


def test_verifier_accepts_correct() -> None:
    r = run_code_tests("def f(x):\n    return x * x\n", _CHECKS)
    assert r["result"]["scalar"] == 1.0
    assert all(r["result"]["pass_fail"].values())


def test_verifier_fails_wrong() -> None:
    r = run_code_tests("def f(x):\n    return 0\n", _CHECKS)
    assert r["result"]["scalar"] == 0.0
    assert not any(r["result"]["pass_fail"].values())


def test_verifier_fails_fake_success_noop() -> None:
    # No entrypoint defined; just prints success -> the held-out oracle fails it.
    r = run_code_tests("print('done!')\n", _CHECKS)
    assert r["result"]["scalar"] == 0.0


def test_verifier_enforces_timeout() -> None:
    r = run_code_tests("import time\ndef f(x):\n    time.sleep(30)\n    return x*x\n", _CHECKS, timeout_s=2.0)
    assert r["result"]["scalar"] == 0.0
    assert r["raw"]["timed_out"] is True


def test_verifier_rejects_unknown_sandbox_and_language() -> None:
    assert "unknown sandbox" in run_code_tests("x=1", _CHECKS, sandbox="vm")["result"]["explanation"]
    assert "unsupported language" in run_code_tests("x=1", _CHECKS, language="rust")["result"]["explanation"]


def test_suite_loads_and_validates() -> None:
    suite = _coding.load_coding_suite(SUITE)
    assert suite["kind"] == _coding.BENCH_SUITE_KIND
    assert len(suite["tasks"]) >= 3


def test_suite_validation_rejects_malformed() -> None:
    base = {"kind": _coding.BENCH_SUITE_KIND, "name": "x", "tasks": []}
    with pytest.raises(_coding.CodingSuiteError):
        _coding.load_coding_suite(base)  # no tasks
    missing_field = {
        "kind": _coding.BENCH_SUITE_KIND,
        "tasks": [{"id": "t", "instruction": "i", "checks": "c"}],  # no reference_solution
    }
    with pytest.raises(_coding.CodingSuiteError):
        _coding.load_coding_suite(missing_field)
    no_guards = {
        "kind": _coding.BENCH_SUITE_KIND,
        "tasks": [
            {"id": "t", "instruction": "i", "checks": "c", "reference_solution": "s"}
        ],
    }
    with pytest.raises(_coding.CodingSuiteError):
        _coding.load_coding_suite(no_guards)  # missing guards.min_guard_count


def test_artifact_in_reference_all_pass() -> None:
    suite = _coding.load_coding_suite(SUITE)
    ref = _coding.reference_submission(suite)
    res = bench.run_bench(
        SUITE, control_mode="artifact_in", submission=ref,
        evidence_class="local_gate", emit_telemetry=False,
    )
    assert res["control_mode"] == "artifact_in"
    assert res["modalities"] == ["coding"]
    assert res["aggregate"]["pass_rate"] == 1.0
    for row in res["per_task"]:
        assert row["verdict"] == "pass"
        assert row["execution_class"] == "executable"
        assert row["overclaim"] is False


def test_artifact_in_broken_and_missing() -> None:
    suite = _coding.load_coding_suite(SUITE)
    ref = _coding.reference_submission(suite)
    broken = dict(ref)
    first = suite["tasks"][0]["id"]
    broken[first] = "def nope():\n    return 1\n"
    del broken[suite["tasks"][1]["id"]]  # missing submission -> void
    res = bench.run_bench(
        SUITE, control_mode="artifact_in", submission=broken, emit_telemetry=False,
    )
    by = {r["task_id"]: r for r in res["per_task"]}
    assert by[first]["verdict"] == "fail"
    assert by[suite["tasks"][1]["id"]]["verdict"] == "void"


def test_artifact_in_requires_submission() -> None:
    with pytest.raises(bench.BenchError):
        bench.run_bench(SUITE, control_mode="artifact_in", emit_telemetry=False)


def test_coding_suite_rejects_push_mode() -> None:
    with pytest.raises(bench.BenchError):
        bench.run_bench(SUITE, {"type": "scripted"}, control_mode="push", emit_telemetry=False)


def test_artifact_in_is_deterministic() -> None:
    suite = _coding.load_coding_suite(SUITE)
    ref = _coding.reference_submission(suite)
    a = bench.run_bench(SUITE, control_mode="artifact_in", submission=ref, emit_telemetry=False)
    b = bench.run_bench(SUITE, control_mode="artifact_in", submission=ref, emit_telemetry=False)
    sa = {r["task_id"]: r["result"]["scalar"] for r in a["per_task"]}
    sb = {r["task_id"]: r["result"]["scalar"] for r in b["per_task"]}
    assert sa == sb


def test_bench_contract_gate_clean() -> None:
    st = trinity._release_bench_contract_status(ROOT)
    assert st["kind"] == "agent-learning.bench-contract-readiness.v1"
    for bucket in (
        "missing_files",
        "suite_errors",
        "reference_pass_errors",
        "discrimination_errors",
        "determinism_errors",
        "oracle_held_out_errors",
        "guard_errors",
        "command_graded_errors",
        "pull_errors",
        "voice_errors",
    ):
        assert st[bucket] == [], f"{bucket}: {st[bucket]}"


def _good_artifact() -> dict:
    return {
        "kind": "agent-learning.coding-benchmark-example.v1",
        "gate_evidence": {
            "reference_pass": {"all_reference_solutions_pass": True},
            "discrimination": {"broken_candidate_fails": True, "fake_success_noop_fails": True},
            "determinism": {"scores_identical_across_runs": True},
            "oracle_held_out": {"checks_not_in_reference": True},
            "guard_presence": {"all_tasks_have_guards": True},
            "honesty": {"no_executable_overclaim": True},
            "command_graded": {
                "reference_all_pass": True, "wrong_all_fail": True, "forge_all_fail": True,
            },
            "pull": {"reference_solves_all": True, "noop_fails_all": True},
            "voice": {"reference_all_pass": True, "bad_all_fail": True},
        },
    }


# Each mutation -> the bucket that MUST fire. A gate that cannot fail is worthless;
# the gate audits the example's self-reported evidence, so prove every bucket bites.
_BUCKET_FIRES = [
    ("kind", lambda a: a.update(kind="wrong"), "suite_errors"),
    (
        "reference_pass",
        lambda a: a["gate_evidence"]["reference_pass"].update(all_reference_solutions_pass=False),
        "reference_pass_errors",
    ),
    (
        "broken_candidate",
        lambda a: a["gate_evidence"]["discrimination"].update(broken_candidate_fails=False),
        "discrimination_errors",
    ),
    (
        "noop",
        lambda a: a["gate_evidence"]["discrimination"].update(fake_success_noop_fails=False),
        "discrimination_errors",
    ),
    (
        "determinism",
        lambda a: a["gate_evidence"]["determinism"].update(scores_identical_across_runs=False),
        "determinism_errors",
    ),
    (
        "oracle",
        lambda a: a["gate_evidence"]["oracle_held_out"].update(checks_not_in_reference=False),
        "oracle_held_out_errors",
    ),
    (
        "guards",
        lambda a: a["gate_evidence"]["guard_presence"].update(all_tasks_have_guards=False),
        "guard_errors",
    ),
    (
        "overclaim",
        lambda a: a["gate_evidence"]["honesty"].update(no_executable_overclaim=False),
        "guard_errors",
    ),
    (
        "command_forge",
        lambda a: a["gate_evidence"]["command_graded"].update(forge_all_fail=False),
        "command_graded_errors",
    ),
    (
        "pull_noop",
        lambda a: a["gate_evidence"]["pull"].update(noop_fails_all=False),
        "pull_errors",
    ),
    (
        "voice_bad",
        lambda a: a["gate_evidence"]["voice"].update(bad_all_fail=False),
        "voice_errors",
    ),
]


@pytest.mark.parametrize("label,mutate,bucket", _BUCKET_FIRES, ids=[c[0] for c in _BUCKET_FIRES])
def test_bench_contract_gate_buckets_fire(monkeypatch, label, mutate, bucket) -> None:
    artifact = _good_artifact()
    mutate(artifact)
    monkeypatch.setattr(trinity, "_exec_example_run", lambda *a, **k: (artifact, None))
    st = trinity._release_bench_contract_status(ROOT)
    assert st[bucket], f"expected {bucket} to fire for mutation {label!r}"


def test_bench_contract_gate_fires_on_run_error(monkeypatch) -> None:
    monkeypatch.setattr(trinity, "_exec_example_run", lambda *a, **k: ({}, "boom"))
    st = trinity._release_bench_contract_status(ROOT)
    assert st["suite_errors"]


# --- review fixes: validation, infra-void, fatal paths, edge cases ---


def test_run_bench_rejects_bad_sandbox_and_evidence_class() -> None:
    suite = _coding.load_coding_suite(SUITE)
    ref = _coding.reference_submission(suite)
    with pytest.raises(bench.BenchError):
        bench.run_bench(SUITE, control_mode="artifact_in", submission=ref,
                        sandbox="vm", emit_telemetry=False)
    with pytest.raises(bench.BenchError):
        bench.run_bench(SUITE, control_mode="artifact_in", submission=ref,
                        evidence_class="totally_made_up", emit_telemetry=False)


def test_docker_unavailable_is_infra_void_not_fail(monkeypatch) -> None:
    # BH-03/BH-13: a missing Docker daemon must VOID (lane never ran), never report
    # a correct agent as 0% — and must never raise. Credential-free (monkeypatched).
    import fi.alk.bench._docker as dk

    monkeypatch.setattr(dk, "docker_available", lambda: False)
    suite = _coding.load_coding_suite(SUITE)
    ref = _coding.reference_submission(suite)
    res = bench.run_bench(SUITE, control_mode="artifact_in", submission=ref,
                          sandbox="docker", emit_telemetry=False)
    assert all(r["verdict"] == "void" for r in res["per_task"])
    assert res["aggregate"]["void"] == len(res["per_task"])
    assert res["aggregate"]["scored"] == 0
    # void rows carry the forced live_lane stamp (stamping precedes any run) + an error
    for r in res["per_task"]:
        assert r["evidence_class"] == "live_lane"
        assert "infra" in (r.get("error") or "")


def test_run_code_tests_docker_unavailable_returns_honest_failure(monkeypatch) -> None:
    import fi.alk.bench._docker as dk

    monkeypatch.setattr(dk, "docker_available", lambda: False)
    r = run_code_tests("def f(x):\n    return x\n", _CHECKS, sandbox="docker")
    assert r["result"]["scalar"] == 0.0
    assert r["raw"].get("infra_error") is True
    assert "docker unavailable" in r["result"]["explanation"]


def test_docker_argv_has_hardening_flags() -> None:
    # BH-04/BH-12: credential-free assertion of the isolation flags (a docker-gated
    # test would never run on no-docker CI).
    from fi.alk.bench._docker import _build_docker_argv

    argv = _build_docker_argv("name", "img", "256m", "1.0", "print(1)")
    for token in ("--network", "none", "--cap-drop", "ALL",
                  "--security-opt", "no-new-privileges",
                  "--read-only", "/tmp:size=16m,nosuid", "--user", "65534:65534"):
        assert token in argv, f"missing {token!r}"


def test_codeexec_fatal_paths() -> None:
    # BH-06: honest-failure branches in the subprocess runner.
    bad_import = "import does_not_exist_xyz\n"  # checks import a missing module
    r = run_code_tests("x = 1\n", bad_import)
    assert r["result"]["scalar"] == 0.0
    assert "checks_import_failed" in r["result"]["explanation"]

    no_checks = "import solution\nVALUE = 1\n"  # no check_* callables
    r = run_code_tests("def f(x):\n    return x\n", no_checks)
    assert r["result"]["scalar"] == 0.0
    assert "no check_" in r["result"]["explanation"]

    # candidate hard-exits at import -> runner emits nothing parseable (exit 0)
    osexit = "import os\nos._exit(0)\n"
    r = run_code_tests(osexit, _CHECKS)
    assert r["result"]["scalar"] == 0.0
    assert "no parseable result" in r["result"]["explanation"]


def test_void_row_schema_and_edge_cases() -> None:
    # BH-20 + BH-21: void-row shape, max_tasks clamp.
    suite = _coding.load_coding_suite(SUITE)
    ref = dict(_coding.reference_submission(suite))
    first = suite["tasks"][0]["id"]
    del ref[first]  # omit one submission -> void
    res = bench.run_bench(SUITE, control_mode="artifact_in", submission=ref,
                          emit_telemetry=False)
    void = next(r for r in res["per_task"] if r["task_id"] == first)
    assert void["verdict"] == "void"
    assert void["result"]["scalar"] is None
    assert void["result"]["components"] == {}
    assert void["result"]["pass_fail"] == {}
    assert "error" in void
    assert "execution_class" in void and "evidence_class" in void

    # max_tasks=0 -> empty, no ZeroDivision in aggregate
    empty = bench.run_bench(SUITE, control_mode="artifact_in",
                            submission=_coding.reference_submission(suite),
                            max_tasks=0, emit_telemetry=False)
    assert empty["per_task"] == []
    assert empty["aggregate"]["count"] == 0
    assert empty["aggregate"]["pass_rate"] == 0.0


def test_per_task_timeout_override() -> None:
    # BH-21(c): a task-level timeout_s overrides the default (subprocess, fast).
    suite = {
        "kind": _coding.BENCH_SUITE_KIND,
        "name": "to",
        "language": "python",
        "tasks": [{
            "id": "slow",
            "instruction": "n/a",
            "checks": "import solution\n\ndef check_x():\n    assert solution.f() == 1\n",
            "reference_solution": "def f():\n    return 1\n",
            "timeout_s": 1,
            "guards": {"min_guard_count": 1},
        }],
    }
    slow = "import time\ndef f():\n    time.sleep(30)\n    return 1\n"
    rows = _coding.run_coding_artifact_in(suite, {"slow": slow})
    assert rows[0]["verdict"] == "fail"
    assert rows[0]["raw"]["timed_out"] is True
