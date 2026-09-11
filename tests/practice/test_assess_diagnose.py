"""Unit 10 (BBG U10) — ASSESS battery + DIAGNOSE composition."""
from __future__ import annotations

from fi.alk import loss as L
from fi.alk.practice import _assess, _diagnose
from fi.alk.practice._budget import BudgetMeter


def _objective():
    return L.compile_objective({
        "evals": [{"eval": "agent_report", "weight": 1.0}],
        "source": "declared",
        "guards": {"sentinel_rows": ["row_g"], "min_guard_count": 1},
    })


def _simulation(intents=None, personas=("sha256:p1",)):
    scenario = {"name": "s"}
    if intents:
        scenario["coverage"] = {"intents": intents}
    return {
        "name": "sim",
        "scenarios": [{
            "scenario": scenario,
            "cast": [{"persona": p, "role": "user"} for p in personas],
            "weight": 1.0,
        }],
    }


def test_grid_enumeration_counts():
    sim = _simulation(intents=["a", "b"], personas=("sha256:p1", "sha256:p2"))
    cells = _assess._grid_cells(sim)
    assert len(cells) == 4  # 2 intents × 2 personas × 1 perturbation


def test_degenerate_single_cell():
    cells = _assess._grid_cells({"scenarios": []})
    assert len(cells) == 1


def test_assess_meter_charging():
    sim = _simulation(intents=["a", "b"])
    obj = _objective()
    meter = BudgetMeter(100)
    n = 0

    def scorer(cell):
        nonlocal n
        n += 1
        return {"scalar": 1.0, "verdict": "pass", "evidence_class": "local_gate"}

    report = _assess.assess(sim, obj, meter=meter, round_no=0, seed=42,
                            cell_scorer=scorer, repeats=3)
    # 2 cells × 3 repeats = 6 charges
    assert meter.consumed == 6
    assert n == 6
    assert report["kind"] == "agent-learning.practice-report.v1"
    assert report["grid"]["cells_total"] == 2


def test_assess_loss_matches_loss_directly():
    sim = _simulation(intents=["a"])
    obj = _objective()
    meter = BudgetMeter(10)
    report = _assess.assess(sim, obj, meter=meter, round_no=0, seed=1,
                            cell_scorer=lambda c: {"scalar": 0.0, "verdict": "fail", "evidence_class": "local_gate"})
    assert report["loss_report"]["cells"][0]["loss"] == 1.0


def test_deficit_ranking_determinism():
    sim = _simulation(intents=["a", "b"])
    obj = _objective()
    meter = BudgetMeter(10)
    scores = {"a": 0.2, "b": 0.9}

    def scorer(cell):
        s = scores.get(cell.get("intent"), 0.5)
        return {"scalar": s, "verdict": "pass" if s >= 0.7 else "fail", "evidence_class": "local_gate"}

    report = _assess.assess(sim, obj, meter=meter, round_no=0, seed=1, cell_scorer=scorer)
    deficits = _diagnose.diagnose(report, search_space={"agent.instructions": ["x"]})
    # both cells have loss > 0; ranked by loss DESC ⇒ 'a' (loss 0.8) before 'b' (loss 0.1).
    assert deficits["kind"] == "agent-learning.practice-deficits.v1"
    intents = [d["cell"]["intent"] for d in deficits["deficits"]]
    assert intents[0] == "a"  # highest loss ranks first (deterministic)
    assert intents == ["a", "b"]


def test_diagnose_search_paths_narrowing():
    sim = _simulation(intents=["a"])
    obj = _objective()
    meter = BudgetMeter(10)
    report = _assess.assess(sim, obj, meter=meter, round_no=0, seed=1,
                            cell_scorer=lambda c: {"scalar": 0.0, "verdict": "fail", "evidence_class": "local_gate"})
    search_space = {"agent.instructions": ["x"], "tools.lookup.config": ["y"]}
    deficits = _diagnose.diagnose(report, search_space=search_space,
                                  layer_hint={_diagnose._cell_hash(report["loss_report"]["cells"][0]["cell"]): "tool_interface"})
    paths = deficits["deficits"][0]["search_paths"]
    # tool_interface layer narrows to the tools path only
    assert "tools.lookup.config" in paths
    assert "agent.instructions" not in paths
