"""Unit 13 (BBG U13) — the six-phase driver + determinism + calibration."""
from __future__ import annotations

import json

import pytest

from fi.alk import loss as L
from fi.alk.practice import _calibrate
from fi.alk.practice._trainer import PracticeRefusal, run_practice_loop

STRIP = ("created_at", "started_at", "completed_at", "duration_s", "timing")


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in STRIP}
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    return obj


def _objective(source="declared"):
    payload = {"evals": [{"eval": "agent_report", "weight": 1.0}], "source": source}
    if source == "declared":
        payload["guards"] = {"sentinel_rows": ["row_g"], "min_guard_count": 1}
    return L.compile_objective(payload)


def _manifest(tmp_path, source="declared", eval_budget=50, **over):
    sim = {
        "kind": "agent-learning.simulation.v1", "name": "s",
        "version": "sha256:simv",
        "scenarios": [{"scenario": {"name": "s", "coverage": {"intents": ["a", "b"]}},
                       "cast": [{"persona": "sha256:p", "role": "user"}], "weight": 1.0}],
        "world": {"kind": "conversation"},
        "objective": _objective(source),
    }
    m = {
        "name": "pl", "simulation": {"version": "sha256:simv", "inline": sim},
        "eval_budget": eval_budget, "seed": 7, "max_rounds": 2,
        "search_space": {"agent.instructions": ["x"]},
        "store": {"path": str(tmp_path / "records.jsonl"), "active_cap": 64},
    }
    m.update(over)
    return m


def test_seeded_two_run_determinism(tmp_path):
    """Identical seed + fixtures ⇒ byte-identical phase artifacts after strip."""
    def scorer(cell):
        return {"scalar": 0.5, "verdict": "fail" if cell.get("intent") == "a" else "pass",
                "evidence_class": "local_gate"}
    m1 = _manifest(tmp_path / "a", eval_budget=50)
    m2 = _manifest(tmp_path / "b", eval_budget=50)
    r1 = run_practice_loop(m1, cell_scorer=scorer, repeat_scorer=lambda s, seed: 0.5)
    r2 = run_practice_loop(m2, cell_scorer=scorer, repeat_scorer=lambda s, seed: 0.5)
    # store paths differ; strip them and compare round artifacts.
    a = _strip({k: v for k, v in r1.items() if k != "budget_ledger"})
    b = _strip({k: v for k, v in r2.items() if k != "budget_ledger"})
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_budget_exhaustion_stops(tmp_path):
    m = _manifest(tmp_path, eval_budget=2, max_rounds=10)
    r = run_practice_loop(m, cell_scorer=lambda c: {"scalar": 0.0, "verdict": "fail", "evidence_class": "local_gate"})
    assert r["stop_reason"] == "budget_exhausted"


def test_derived_objective_refusal(tmp_path):
    m = _manifest(tmp_path, source="derived")
    with pytest.raises(L.ObjectiveError, match="objective_guards_missing"):
        run_practice_loop(m)


def test_undeclared_budget_refusal(tmp_path):
    m = _manifest(tmp_path)
    m["eval_budget"] = None
    with pytest.raises(PracticeRefusal, match="budget_undeclared"):
        run_practice_loop(m)


def test_headline_never_best_found(tmp_path):
    m = _manifest(tmp_path)
    r = run_practice_loop(m)
    assert "retention_and_transfer_at_equal_budget" in r
    assert "best_found" not in json.dumps(r)


def test_budget_conservation(tmp_path):
    m = _manifest(tmp_path, eval_budget=50)
    r = run_practice_loop(m, cell_scorer=lambda c: {"scalar": 0.0, "verdict": "fail", "evidence_class": "local_gate"})
    led = r["budget_ledger"]
    assert sum(led["by_phase"].values()) == led["consumed"] <= led["total"]
    for art_round in r["rounds"]:
        assert "budget_consumed" in art_round["report"]


# --- calibration -----------------------------------------------------------
def test_calibrate_learned():
    rec = _calibrate.calibrate_cell({"intent": "a"}, score=0.9, fork_entropy=0.1,
                                    divergence_step=None, icc=0.8, repeats=8)
    assert rec["verdict"] == "learned"


def test_calibrate_fluent_not_learned():
    rec = _calibrate.calibrate_cell({"intent": "a"}, score=0.9, fork_entropy=0.6,
                                    divergence_step=2, icc=0.8, repeats=8)
    assert rec["verdict"] == "fluent_not_learned"


def test_calibrate_plateaued():
    rec = _calibrate.calibrate_cell({"intent": "a"}, score=0.5, fork_entropy=0.1,
                                    divergence_step=None, icc=0.8, repeats=8, prior_score=0.5)
    assert rec["verdict"] == "plateaued"
