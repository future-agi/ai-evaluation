"""Unit 14 (BBG U14) — build_practice_loop_manifest (the trainer facade)."""
from __future__ import annotations

import pytest

from fi.alk import loss as L
from fi.alk.optimize import build_practice_loop_manifest


def _objective(source="declared"):
    p = {"evals": [{"eval": "agent_report", "weight": 1.0}], "source": source}
    if source == "declared":
        p["guards"] = {"sentinel_rows": ["row_g"], "min_guard_count": 1}
    return L.compile_objective(p)


def _sim(source="declared"):
    return {"version": "sha256:simv", "inline": {
        "kind": "agent-learning.simulation.v1", "name": "s", "version": "sha256:simv",
        "world": {"kind": "conversation"}, "scenarios": [{"cast": []}],
        "objective": _objective(source),
    }}


def _build(**over):
    kw = dict(
        name="pl", simulation=_sim(), base_agent={"provider": "custom", "instructions": "x"},
        search_space={"agent.instructions": ["a", "b"]}, eval_budget=64, seed=42,
    )
    kw.update(over)
    return build_practice_loop_manifest(**kw)


def test_delegation_passes_whole_agent_validators():
    m = _build()
    assert m["version"] == "agent-learning.practice-loop.v1"
    assert m["whole_agent"]["eval_budget"] == 64
    assert m["optimization"]["ranking_source"] == "evaluation_suite"


def test_ru1_defaults_materialized_and_echoed():
    m = _build()
    p = m["practice"]
    assert p["budget_plan"] == [0.25, 0.35, 0.25, 0.15]
    assert p["review_ratio"] == 0.25
    assert p["zpd"] == {"band": [0.2, 0.7], "k": 8, "icc_floor": 0.5}
    assert p["scaffold_fade"]["intensities"] == [1.0, 0.5, 0.0]
    assert p["store"]["active_cap"] == 64
    assert p["inner_operator"]["backend"] == "society"
    assert p["schedule"]["intervals"] == [1, 2, 4, 8, 16]


def test_seed_mandatory():
    with pytest.raises((ValueError, TypeError)):
        _build(seed=None)


def test_budget_mandatory():
    with pytest.raises(ValueError):
        _build(eval_budget=0)


def test_unguarded_objective_rejected():
    with pytest.raises(ValueError, match="objective_guards_missing"):
        _build(simulation=_sim(source="derived"))


def test_no_objective_rejected():
    sim = {"version": "sha256:x", "inline": {"kind": "agent-learning.simulation.v1",
            "name": "s", "world": {"kind": "conversation"}, "scenarios": [{"cast": []}]}}
    with pytest.raises(ValueError, match="objective_guards_missing"):
        _build(simulation=sim)


def test_fade_not_ending_0_rejected():
    with pytest.raises(ValueError, match="end at 0.0"):
        _build(scaffold_fade={"intensities": [1.0, 0.5]})


def test_inner_operator_outside_canon_rejected():
    with pytest.raises(ValueError, match="inner_operator.backend"):
        _build(inner_operator={"backend": "nonexistent_backend"})


def test_no_objective_kwarg_exists():
    """ARCH §2d field table has no objective row; the kwarg must not exist."""
    import inspect
    sig = inspect.signature(build_practice_loop_manifest)
    assert "objective" not in sig.parameters
