"""Unit tests for the unified bench harness facade (phase 15A).

Fast + deterministic: the engine is bypassed via the injectable ``runner`` seam,
so these exercise the contract glue (unified Result projection, modality mapping,
control-mode dispatch, honesty-field passthrough) without any network or model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from fi.alk import bench, tasks

ROOT = Path(__file__).parent.parent
SUITE = ROOT / "examples" / "task_datasets" / "support_starter.json"

_AGENT = {"type": "scripted", "content": "stub"}


def _fake_runner(
    metric_averages: Mapping[str, float], eval_score: float = 0.8
) -> Callable[[Mapping[str, Any], Mapping[str, Any]], dict]:
    def runner(_task: Mapping[str, Any], _agent: Mapping[str, Any]) -> dict:
        return {
            "summary": {
                "metric_averages": dict(metric_averages),
                "evaluation_score": eval_score,
            }
        }

    return runner


def _run(**kwargs: Any) -> dict:
    runner = _fake_runner({"task_completion": 0.9, "tool_selection_accuracy": 1.0})
    return bench.run_bench(
        SUITE, _AGENT, runner=runner, emit_telemetry=False, **kwargs
    )


def test_push_mode_shape() -> None:
    res = _run(control_mode="push")
    assert res["kind"] == bench.BENCH_RESULT_KIND
    assert res["control_mode"] == "push"
    assert res["per_task"], "expected per-task rows"
    assert set(res["modalities"]) <= {"text", "tool", "coding", "voice", "computer_use", "unknown"}
    assert "aggregate" in res
    # never emits telemetry into the result when disabled
    assert "telemetry" not in res


def test_unified_result_projection() -> None:
    res = _run(control_mode="push", max_tasks=1)
    row = res["per_task"][0]
    result = row["result"]
    assert set(result) == {"scalar", "components", "pass_fail", "explanation"}
    assert isinstance(result["components"], dict)
    # the fake runner's metrics flow into components
    assert result["components"].get("task_completion") == pytest.approx(0.9)
    assert result["pass_fail"]["verdict"] in (True, False)
    assert result["explanation"] in ("objective", "evaluation_score_fallback")


def test_honesty_fields_preserved() -> None:
    row = _run(control_mode="push", max_tasks=1)["per_task"][0]
    for key in ("execution_class", "evidence_class", "overclaim", "modality", "world_kind"):
        assert key in row
    assert row["overclaim"] is False  # captured_fixture default is honest


def test_overclaim_tripwire_passthrough() -> None:
    # A typed_only task stamped with a live evidence_class MUST be flagged.
    res = _run(control_mode="push", evidence_class="live_lane")
    typed_only = [r for r in res["per_task"] if r["execution_class"] in ("typed_only", "fixture")]
    assert typed_only, "support_starter ships a typed-only (browser) task"
    assert all(r["overclaim"] is True for r in typed_only)
    executable = [r for r in res["per_task"] if r["execution_class"] == "executable"]
    assert all(r["overclaim"] is False for r in executable)


def test_modality_mapping() -> None:
    assert bench.modality_for_world_kind("conversation") == "text"
    assert bench.modality_for_world_kind("tool_api") == "tool"
    assert bench.modality_for_world_kind("code_exec") == "coding"
    assert bench.modality_for_world_kind("voice_telephony") == "voice"
    assert bench.modality_for_world_kind("browser") == "computer_use"
    assert bench.modality_for_world_kind("nonsense") == "unknown"


def test_load_bench_suite_accepts_path_and_mapping() -> None:
    from_path = bench.load_bench_suite(SUITE)
    assert from_path["tasks"]
    from_mapping = bench.load_bench_suite(from_path)
    assert from_mapping["name"] == from_path["name"]
    with pytest.raises(bench.BenchError):
        bench.load_bench_suite(123)  # type: ignore[arg-type]


def test_staged_modes_raise_clearly() -> None:
    # pull is not implemented yet; artifact_in on a *task dataset* is a usage
    # error (it requires a coding bench suite) -> BenchError, not NotImplemented.
    with pytest.raises(NotImplementedError):
        _run(control_mode="pull")
    with pytest.raises(bench.BenchError):
        _run(control_mode="artifact_in")


def test_unknown_mode_raises_bench_error() -> None:
    with pytest.raises(bench.BenchError):
        _run(control_mode="bogus")


def test_projection_is_pure_relabel_of_engine_result() -> None:
    # The bench result must not alter the underlying scores — same scalar as the
    # engine's per-task score (the facade re-badges, never re-scores).
    runner = _fake_runner({"task_completion": 0.9, "tool_selection_accuracy": 1.0})
    engine = tasks.run_benchmark(
        tasks.load_task_dataset(SUITE), _AGENT, runner=runner, emit_telemetry=False, max_tasks=2
    )
    badged = bench.run_bench(SUITE, _AGENT, runner=runner, emit_telemetry=False, max_tasks=2)
    for e, b in zip(engine["per_task"], badged["per_task"]):
        assert e["score"] == b["result"]["scalar"]
        assert e["verdict"] == b["verdict"]
