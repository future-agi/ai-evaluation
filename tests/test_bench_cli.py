"""CLI coverage for ``agent-learn bench`` (review BH-07).

Exercises arg dispatch, the agent/submission resolution, exit-code mapping, and
the ``bench``/``benchmark`` aliases against real shipped fixtures. Credential-free
(``--no-telemetry`` on success paths).
"""

from __future__ import annotations

from pathlib import Path

from fi.alk.cli import main

ROOT = Path(__file__).parent.parent
CODING = str(ROOT / "examples" / "bench_suites" / "coding_starter.json")
TASKS = str(ROOT / "examples" / "task_datasets" / "support_starter.json")


def test_bench_artifact_in_reference_exit_0(tmp_path) -> None:
    out = tmp_path / "r.json"
    code = main([
        "bench", CODING, "--mode", "artifact_in", "--reference",
        "--no-telemetry", "--quiet", "-o", str(out),
    ])
    assert code == 0
    assert out.exists()


def test_bench_benchmark_alias_exit_0(tmp_path) -> None:
    code = main([
        "benchmark", CODING, "--mode", "artifact_in", "--reference",
        "--no-telemetry", "--quiet", "-o", str(tmp_path / "r.json"),
    ])
    assert code == 0


def test_bench_artifact_in_without_submission_exit_1() -> None:
    assert main(["bench", CODING, "--mode", "artifact_in", "--no-telemetry", "--quiet"]) == 1


def test_bench_push_without_agent_exit_1() -> None:
    assert main(["bench", TASKS, "--mode", "push", "--no-telemetry", "--quiet"]) == 1


def test_bench_bad_agent_json_exit_1() -> None:
    assert main(["bench", TASKS, "--agent", "{not json", "--no-telemetry", "--quiet"]) == 1


def test_bench_pull_not_implemented_exit_2() -> None:
    # pull on a task dataset is staged -> NotImplementedError -> exit 2
    assert main([
        "bench", TASKS, "--agent", '{"type":"scripted","content":"x"}',
        "--mode", "pull", "--no-telemetry", "--quiet",
    ]) == 2
