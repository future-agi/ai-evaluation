"""Opt-in tests for the Docker code-exec lane (phase 15E).

These run ONLY when a Docker daemon is reachable (skipped otherwise), so the
credential-free suite still passes on machines without Docker. The Docker lane is
never a release-gate prerequisite — the ``bench_contract_readiness`` gate uses the
subprocess lane on trusted shipped code.
"""

from __future__ import annotations

import pytest

from fi.alk import bench
from fi.alk.bench import _coding
from fi.alk.bench._codeexec import run_code_tests
from fi.alk.bench._docker import docker_available

pytestmark = pytest.mark.skipif(
    not docker_available(), reason="docker daemon unavailable"
)

_CHECKS = (
    "import solution\n\n"
    "def check_a():\n    assert solution.f(2) == 4\n\n"
    "def check_b():\n    assert solution.f(3) == 9\n"
)


def test_docker_accepts_correct() -> None:
    r = run_code_tests("def f(x):\n    return x * x\n", _CHECKS, sandbox="docker")
    assert r["result"]["scalar"] == 1.0
    assert r["raw"]["sandbox"] == "docker"
    assert r["raw"]["network"] == "none"
    assert r["raw"]["exit_code"] == 0


def test_docker_fails_wrong_and_noop() -> None:
    assert run_code_tests("def f(x):\n    return 0\n", _CHECKS, sandbox="docker")["result"]["scalar"] == 0.0
    assert run_code_tests("print('done')\n", _CHECKS, sandbox="docker")["result"]["scalar"] == 0.0


def test_docker_enforces_timeout() -> None:
    slow = "import time\ndef f(x):\n    time.sleep(60)\n    return x * x\n"
    r = run_code_tests(slow, _CHECKS, sandbox="docker", timeout_s=3.0)
    assert r["result"]["scalar"] == 0.0
    assert r["raw"]["timed_out"] is True


def test_docker_blocks_network() -> None:
    # --network none must block egress: online() raises -> returns False -> check passes.
    checks = "import solution\n\ndef check_blocked():\n    assert solution.online() is False\n"
    cand = (
        "def online():\n"
        "    import socket\n"
        "    try:\n"
        "        socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "        return True\n"
        "    except Exception:\n"
        "        return False\n"
    )
    r = run_code_tests(cand, checks, sandbox="docker", timeout_s=20.0)
    assert r["result"]["scalar"] == 1.0, "network was NOT blocked under --network none"


def test_docker_rows_are_live_lane() -> None:
    suite_path = "examples/bench_suites/coding_starter.json"
    ref = _coding.reference_submission(_coding.load_coding_suite(suite_path))
    res = bench.run_bench(
        suite_path, control_mode="artifact_in", submission=ref,
        sandbox="docker", evidence_class="captured_fixture", max_tasks=1,
        emit_telemetry=False,
    )
    row = res["per_task"][0]
    # untrusted live execution is never mislabeled as a fixture, even if asked.
    assert row["evidence_class"] == "live_lane"
    assert row["overclaim"] is False
