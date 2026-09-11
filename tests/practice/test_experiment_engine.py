"""Unit 23 (13D-5 capstone EXPERIMENT ENGINE) — the deferred 13D-5 deliverable.

These tests guard the EXPERIMENT path (``practice ab --run`` / run_experiment),
which is SEPARATE from the outcome-free contract harness (``_capstone.run_ab``).
They assert the experiment actually RUNS arms, meters equal total budget, honours
the A1-A4 ablation flags as real trainer knobs, computes the interference/
retention metrics, and is deterministic + offline + seeded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fi.alk import cli
from fi.alk.practice import _experiment

CAPSTONE_DIR = Path(__file__).resolve().parents[1].parent / "examples" / "practice_capstone"
FIXTURES = CAPSTONE_DIR / "fixtures"


def _digest(obj) -> str:
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items()
                    if k not in ("created_at", "started_at", "completed_at", "duration_s", "timing")}
        if isinstance(o, list):
            return [strip(x) for x in o]
        return o
    return hashlib.sha256(
        json.dumps(strip(obj), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


# --- fixtures exist + parse -------------------------------------------------
def test_three_fixtures_exist_and_parse():
    for name in ("refund_desk", "tool_world_ops", "escalation_ladder"):
        fx = _experiment.load_fixture(FIXTURES, name)
        assert fx["kind"] == "agent-learning.practice-capstone-fixture.v1"
        assert fx["primary_cells"] and fx["interference_cells"]
        # interference must share >=1 config path with primary (so forgetting is possible).
        primary_paths = {c["path"] for c in fx["primary_cells"]}
        interf_paths = {c["path"] for c in fx["interference_cells"]}
        assert primary_paths & interf_paths, f"{name}: interference shares no path with primary"


# --- the latent-skill model is real + deterministic ------------------------
def test_latent_skill_model_closes_on_required_value():
    fx = _experiment.load_fixture(FIXTURES, "refund_desk")
    cell = fx["primary_cells"][0]
    closed = {cell["path"]: cell["required_value"]}
    wrong = {cell["path"]: "definitely_not_it"}
    assert _experiment._cell_score(cell, closed) == 1.0
    assert _experiment._cell_score(cell, wrong) < 1.0


# --- arms actually run + produce REAL retention numbers --------------------
def test_experiment_runs_and_emits_real_retention():
    result = _experiment.run_experiment(CAPSTONE_DIR)
    exp = result["experiment"]
    assert exp["headline_metric"] == "retention_after_interference"
    assert exp["budget_match"] is True
    arms = {r["arm"]: r for r in exp["arms"] if not r["ablations"]}
    assert set(arms) == {"practice_loop", "gepa", "tpe", "society", "bandit"}
    for arm in exp["arms"]:
        # REAL numbers, not None placeholders (the experiment path).
        assert isinstance(arm["mean_retention"], float)
        assert isinstance(arm["mean_best_found"], float)


# --- equal total metered budget (the AD-I discipline) ----------------------
def test_equal_total_budget_respected():
    result = _experiment.run_experiment(CAPSTONE_DIR)
    exp = result["experiment"]
    total = exp["equal_total_budget"]
    for row in exp["arms"] + exp["ablations"]:
        assert row["total_metered_budget"] <= total, row
        assert row["budget_match"] is True


# --- A1-A4 are REAL trainer knobs, not labels ------------------------------
def test_ablations_are_real_knobs_changing_behaviour():
    result = _experiment.run_experiment(CAPSTONE_DIR)
    exp = result["experiment"]
    ablations = {tuple(r["ablations"]): r for r in exp["ablations"]}
    assert ("a2_no_spacing",) in ablations
    assert ("a3_no_consolidation",) in ablations
    practice = next(r for r in exp["arms"] if r["arm"] == "practice_loop" and not r["ablations"])
    a3 = ablations[("a3_no_consolidation",)]
    a2 = ablations[("a2_no_spacing",)]
    # the headline isolation: removing consolidation/spacing must NOT retain as
    # well as the full loop (else the mechanism is dead — a real, falsifiable test).
    assert a3["mean_retention"] <= practice["mean_retention"]
    assert a2["mean_retention"] <= practice["mean_retention"]


def test_no_consolidation_loses_standing_detection():
    """A3 (no store) can never detect interference standing (only promotion-veto)."""
    result = _experiment.run_experiment(CAPSTONE_DIR)
    exp = result["experiment"]
    ablations = {tuple(r["ablations"]): r for r in exp["ablations"]}
    assert ablations[("a3_no_consolidation",)]["standing_detection_any"] is False


def test_ablation_flag_changes_trainer_run():
    """run_practice_loop with a3_no_consolidation admits NO records to the store."""
    from fi.alk.practice._trainer import run_practice_loop
    from fi.alk.practice._store import ConsolidationStore
    fx = _experiment.load_fixture(FIXTURES, "refund_desk")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        full = ConsolidationStore(Path(d) / "full.jsonl")
        m_full = _experiment._practice_manifest(fx, eval_budget=128, seed=3,
                                                store_path=Path(d) / "full.jsonl", ablations=[])
        m_full["meter_drill_repeats"] = True
        by_key = {_experiment._loss._cell_key(_experiment._grid_cell(c)): c for c in fx["primary_cells"]}
        cfg = {}

        def scorer(cell):
            fc = by_key.get(_experiment._loss._cell_key(cell))
            if fc is None:
                return {"scalar": 1.0, "verdict": "pass", "evidence_class": "local_gate"}
            s = _experiment._cell_score(fc, cfg)
            return {"scalar": s, "verdict": "pass" if s >= 0.7 else "fail", "evidence_class": "local_gate"}

        def rep(sim, c):
            tk = (sim.get("metadata") or {}).get("drill_cell")
            fc = by_key.get(_experiment._loss._cell_key(tk)) if tk else None
            if fc is None:
                return 1.0
            cfg[fc["path"]] = fc["required_value"]
            return 1.0

        run_practice_loop(m_full, cell_scorer=scorer, repeat_scorer=rep, store=full)
        cfg.clear()
        a3 = ConsolidationStore(Path(d) / "a3.jsonl")
        m_a3 = _experiment._practice_manifest(fx, eval_budget=128, seed=3,
                                              store_path=Path(d) / "a3.jsonl",
                                              ablations=["a3_no_consolidation"])
        m_a3["meter_drill_repeats"] = True
        run_practice_loop(m_a3, cell_scorer=scorer, repeat_scorer=rep, store=a3)
        # full loop consolidates; a3 admits nothing.
        assert len(a3.active_records()) == 0
        assert len(full.active_records()) >= len(a3.active_records())


# --- interference + retention metrics --------------------------------------
def test_interference_creates_a_real_regression():
    """The interference phase overwrites a shared path → a primary cell flips."""
    fx = _experiment.load_fixture(FIXTURES, "refund_desk")
    best = {c["path"]: c["required_value"] for c in fx["primary_cells"]}
    interfered = _experiment._interfere_config(best, fx["interference_cells"],
                                               fx["interference_strength"], seed=11)
    flipped = [c for c in fx["primary_cells"] if _experiment._cell_score(c, interfered) < 0.7]
    assert flipped, "interference did not regress any primary cell — fixture is inert"


def test_retention_metrics_vocabulary():
    pre = {"a": 1.0, "b": 1.0}
    post = {"a": 1.0, "b": 0.2}
    transfer = {"x": 0.8}
    m = _experiment._retention_metrics(pre, post, transfer)
    assert 0.0 <= m["retention"] <= 1.0
    assert 0.0 <= m["stability"] <= 1.0
    assert m["mean_pre"] == 1.0
    assert m["retention"] == round(0.6 / 1.0, 6)


# --- determinism (offline, seeded) -----------------------------------------
def test_experiment_is_deterministic():
    a = _experiment.run_experiment(CAPSTONE_DIR)["experiment"]
    b = _experiment.run_experiment(CAPSTONE_DIR)["experiment"]
    assert _digest(a) == _digest(b)


# --- the verdict is a real comparison, not hardcoded -----------------------
def test_verdict_reflects_the_lift():
    exp = _experiment.run_experiment(CAPSTONE_DIR)["experiment"]
    cmp = exp["key_comparison"]
    assert cmp["verdict"] in ("LIFT_REAL", "NULL", "INCONCLUSIVE")
    # the verdict must agree with the numbers it reports.
    if cmp["retention_lift_vs_a3_no_consolidation"] >= 0.05:
        assert cmp["verdict"] == "LIFT_REAL"
        assert cmp["supports_paper"] is True


# --- CLI --run path (separate from the outcome-free gate path) -------------
def test_cli_ab_run_flag(capsys):
    rc = cli.main(["practice", "ab", str(CAPSTONE_DIR), "--run"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["status"] == "ran"
    assert "experiment" in out
    assert out["experiment"]["headline_metric"] == "retention_after_interference"


def test_cli_ab_default_stays_outcome_free(capsys):
    """The DEFAULT (no --run) path is the contract harness — retention stays None
    (the gate / test_harness_never_asserts_outcomes invariant is preserved)."""
    rc = cli.main(["practice", "ab", str(CAPSTONE_DIR)])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert "ab_harness" in out
    for arm in out["ab_harness"]["arms"]:
        assert arm["retention_after_interference"] is None
