"""RSI loop — close the loop: dataset -> optimize -> verify HELD-OUT.

The honest RSI guard (advisor's bar): the optimizer's winner must beat the
baseline on tasks it NEVER optimized against (the held-out test split), measured
on the discriminating objective score (not the metric it climbed). Uses an
injected deterministic runner so the loop mechanics are tested without the engine;
one real-engine integration run proves it closes end-to-end credential-free.
"""

from __future__ import annotations

import pytest

from fi.alk import tasks


def _task(task_id: str) -> dict:
    return {
        "id": task_id,
        "title": task_id,
        "world": {"kind": "conversation"},
        "difficulty": "easy",
        "objective": {
            "source": "declared",
            "evals": [{"eval": "task_success", "weight": 1.0, "anchor": True}],
            "guards": {"sentinel_rows": [{"id": "s"}], "min_guard_count": 1},
        },
        "scenario": {"name": task_id, "kind": "task",
                     "dataset": [{"persona": {"name": "P"}, "situation": "s", "outcome": "o"}]},
        "verification": {"checks": [{"type": "contains", "value": "x"}], "threshold": 0.5},
    }


def _dataset() -> dict:
    return tasks.compile_task_dataset({
        "name": "rsi-mini",
        "tasks": [_task("tr1"), _task("tr2"), _task("te1"), _task("te2")],
        "splits": {"train": ["tr1", "tr2"], "test": ["te1", "te2"]},
    })


def _runner_for(scores_by_content: dict):
    """Deterministic runner: the agent's `content` (the search variable) maps to a
    fixed per-content score, so a clearly-better config exists to be found."""

    def _run(task, agent):  # noqa: ANN001
        s = scores_by_content[agent["content"]]
        return {"status": "passed" if s >= 0.5 else "failed",
                "summary": {"evaluation_score": s, "evaluation_passed": s >= 0.5,
                            "metric_averages": {"task_completion": s}}}

    return _run


def test_rsi_loop_finds_winner_and_verifies_held_out() -> None:
    ds = _dataset()
    # "good" content scores 0.9 everywhere; "bad" 0.2. Baseline = "bad".
    runner = _runner_for({"good": 0.9, "bad": 0.2})
    report = tasks.optimize_against_dataset(
        ds,
        base_agent={"type": "scripted", "content": "bad"},
        search_space={"agent.content": ["bad", "good"]},
        runner=runner,
    )
    assert report["kind"] == tasks.AGENT_LEARNING_RSI_REPORT_KIND
    assert report["winner"]["assignment"]["agent.content"] == "good"
    ho = report["held_out"]
    assert ho["verified"] is True
    assert ho["baseline_mean_score"] == 0.2
    assert ho["winner_mean_score"] == 0.9
    assert ho["lift"] == pytest.approx(0.7)
    assert ho["improved"] is True


def test_rsi_no_improvement_is_honest_null() -> None:
    ds = _dataset()
    # every config scores the same -> no real lift; must report improved=False.
    runner = _runner_for({"a": 0.6, "b": 0.6})
    report = tasks.optimize_against_dataset(
        ds, base_agent={"type": "scripted", "content": "a"},
        search_space={"agent.content": ["a", "b"]}, runner=runner,
    )
    assert report["held_out"]["lift"] == 0.0
    assert report["held_out"]["improved"] is False


def test_rsi_without_splits_flags_not_held_out() -> None:
    ds = tasks.compile_task_dataset({"name": "nosplit", "tasks": [_task("t1"), _task("t2")]})
    runner = _runner_for({"good": 0.9, "bad": 0.2})
    report = tasks.optimize_against_dataset(
        ds, base_agent={"type": "scripted", "content": "bad"},
        search_space={"agent.content": ["bad", "good"]}, runner=runner,
    )
    assert report["held_out"]["verified"] is False  # train==test, NOT held-out
    assert "not held-out" in report["held_out"]["test_split"]


def test_rsi_candidate_cap_truncates_and_flags() -> None:
    ds = _dataset()
    runner = _runner_for({f"c{i}": 0.5 for i in range(5)})
    report = tasks.optimize_against_dataset(
        ds, base_agent={"type": "scripted", "content": "c0"},
        search_space={"agent.content": [f"c{i}" for i in range(5)]},
        max_candidates=2, runner=runner,
    )
    assert report["candidates_evaluated"] == 2
    assert report["candidates_truncated"] is True


@pytest.mark.integration
def test_rsi_loop_closes_on_real_engine() -> None:
    """End-to-end credential-free: a content-grounded task where a name-dropping
    answer genuinely scores higher than 'no'. The winner must verify on held-out."""
    def mk(task_id):
        t = _task(task_id)
        t["objective"]["evals"] = [
            {"eval": "task_success", "weight": 1.0, "anchor": True},
            {"eval": "goal_progress", "weight": 0.6, "anchor": True},
        ]
        t["scenario"]["dataset"] = [{
            "persona": {"name": "D"},
            "situation": "Where is the refund policy and what is the window?",
            "outcome": "States the refund policy location and a 30-day window.",
        }]
        t["verification"] = {"checks": [{"type": "contains", "value": "policy"}], "threshold": 0.5}
        return t

    ds = tasks.compile_task_dataset({
        "name": "rsi-real",
        "tasks": [mk("tr1"), mk("te1")],
        "splits": {"train": ["tr1"], "test": ["te1"]},
    })
    report = tasks.optimize_against_dataset(
        ds,
        base_agent={"type": "scripted", "content": "no"},
        search_space={"agent.content": [
            "no",
            "Our refund policy is at /help/refunds; refunds within 30 days of purchase.",
        ]},
    )
    # the good content must win and verify on the held-out test task
    assert "refund policy" in report["winner"]["assignment"]["agent.content"]
    assert report["held_out"]["verified"] is True
    assert report["held_out"]["improved"] is True
    assert report["held_out"]["lift"] > 0.2
