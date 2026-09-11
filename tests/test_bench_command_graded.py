"""Tests for the hardened command/artifact-graded coding lane (artifact-graded).

The verdict is the held-out grader's exit code (not candidate stdout) and the
grader runs AFTER the candidate, so this lane is structurally robust to the two
PR-review vulns: verdict forgery (BH-01) and oracle reads (BH-02). Multi-language
(Python + bash) via the shipped command suite. Docker tests are opt-in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fi.alk import bench
from fi.alk.bench import _coding
from fi.alk.bench._docker import docker_available
from fi.alk.bench._grader import run_command_graded

ROOT = Path(__file__).parent.parent
COMMAND_SUITE = ROOT / "examples" / "bench_suites" / "coding_command_starter.json"

# A minimal command-graded task: grader runs the candidate as a subprocess on a
# held-out case and checks stdout; verdict = grader exit code.
_GRADER_PY = (
    "import json, os, subprocess, sys\n"
    "GD = os.environ['GRADER_DIR']\n"
    "p = subprocess.run([sys.executable, 'solution.py'], input='2 3',\n"
    "                   capture_output=True, text=True, timeout=5)\n"
    "ok = p.stdout.strip() == '5'\n"
    "json.dump({'score': 1 if ok else 0}, open(os.path.join(GD, 'reward.json'), 'w'))\n"
    "sys.exit(0 if ok else 1)\n"
)
_TASK = {
    "id": "sum",
    "instruction": "read two ints, print sum",
    "grader_cmd": 'python3 "$GRADER_DIR/grade.py"',
    "grader_files": {"grade.py": _GRADER_PY},
}
_CORRECT = {"solution.py": "import sys\na,b=map(int,sys.stdin.read().split())\nprint(a+b)\n"}


def test_grader_accepts_correct_rejects_wrong_and_forge() -> None:
    assert run_command_graded(_TASK, _CORRECT)["result"]["scalar"] == 1.0
    assert run_command_graded(_TASK, {"solution.py": "print(0)\n"})["result"]["scalar"] == 0.0
    # FORGE: print a winning reward to stdout. Verdict is the grader's exit code,
    # not candidate stdout -> still fails.
    forged = {"solution.py": 'print("{\\"score\\": 1}")\n'}
    assert run_command_graded(_TASK, forged)["result"]["scalar"] == 0.0


def test_grader_rejects_unknown_sandbox() -> None:
    r = run_command_graded(_TASK, _CORRECT, sandbox="vm")
    assert r["result"]["scalar"] == 0.0
    assert r["raw"].get("infra_error") is True


def test_command_suite_loads_and_is_multi_language() -> None:
    suite = _coding.load_coding_suite(COMMAND_SUITE)
    langs = {t.get("language") for t in suite["tasks"]}
    assert {"python", "bash"} <= langs


def test_command_suite_reference_passes_and_broken_fails() -> None:
    suite = _coding.load_coding_suite(COMMAND_SUITE)
    ref = _coding.reference_submission(suite)
    # reference is a {path: content} map per task (command-graded)
    assert all(isinstance(v, dict) for v in ref.values())
    res = bench.run_bench(COMMAND_SUITE, control_mode="artifact_in", submission=ref,
                          evidence_class="local_gate", emit_telemetry=False)
    assert res["aggregate"]["pass_rate"] == 1.0
    assert all(r["verdict"] == "pass" for r in res["per_task"])

    broken = {t["id"]: {p: "print('x')\n" if p.endswith(".py") else "echo x\n"
                        for p in t["reference_files"]} for t in suite["tasks"]}
    rb = bench.run_bench(COMMAND_SUITE, control_mode="artifact_in", submission=broken,
                         emit_telemetry=False)
    assert all(r["verdict"] == "fail" for r in rb["per_task"])


def test_command_suite_validation_rejects_missing_grader() -> None:
    bad = {
        "kind": _coding.BENCH_SUITE_KIND,
        "grading": "command",
        "tasks": [{"id": "t", "instruction": "i", "grader_files": {"g": "x"},
                   "reference_files": {"s": "y"}, "guards": {"min_guard_count": 1}}],
    }
    with pytest.raises(_coding.CodingSuiteError):
        _coding.load_coding_suite(bad)  # missing grader_cmd


@pytest.mark.skipif(not docker_available(), reason="docker daemon unavailable")
def test_command_graded_docker_reference_passes_and_forge_fails() -> None:
    suite = _coding.load_coding_suite(COMMAND_SUITE)
    ref = _coding.reference_submission(suite)
    res = bench.run_bench(COMMAND_SUITE, control_mode="artifact_in", submission=ref,
                          sandbox="docker", evidence_class="live_lane", emit_telemetry=False)
    assert all(r["verdict"] == "pass" for r in res["per_task"])
    assert all(r["evidence_class"] == "live_lane" for r in res["per_task"])
    # forge attempt in the hardened lane still fails
    r = run_command_graded(_TASK, {"solution.py": 'print("{\\"score\\": 1}")\n'}, sandbox="docker")
    assert r["result"]["scalar"] == 0.0
