"""#2 — real cell_scorer for the 13D practice loop (audit gap fix).

The practice loop's cell_scorer defaulted to all-pass (scalar 1.0, verdict pass),
so the loop measured nothing. make_cell_scorer bridges engine metric_averages ->
objective_score -> the {scalar,verdict,evidence_class} cell shape, so the loop
(and code-RSI built on it) measures REAL fitness. Deterministic tests via the
runner seam + one real-engine integration.
"""

from __future__ import annotations

import pytest

from fi.alk import tasks

OBJ = {"source": "declared", "evals": [
    {"eval": "task_success", "weight": 1.0, "anchor": True}],
    "guards": {"sentinel_rows": [{"id": "s"}], "min_guard_count": 1}}


def _runner(score_by_persona: dict):
    def run(cell, agent):  # noqa: ANN001
        s = score_by_persona.get(str(cell.get("persona")), 0.9)
        return {"summary": {"metric_averages": {"task_completion": s}}}
    return run


def test_cell_scorer_returns_real_shape() -> None:
    cs = tasks.make_cell_scorer(agent={"type": "scripted"}, objective=OBJ,
                                runner=_runner({"p1": 0.9}))
    out = cs({"intent": "i", "persona": "p1"})
    assert out["scalar"] == 0.9
    assert out["verdict"] == "pass"
    assert out["evidence_class"] == "captured_fixture"
    assert out["eval"] == "agent_report"


def test_cell_scorer_fails_below_threshold() -> None:
    cs = tasks.make_cell_scorer(agent={"type": "scripted"}, objective=OBJ,
                                threshold=0.5, runner=_runner({"bad": 0.2}))
    out = cs({"intent": "i", "persona": "bad"})
    assert out["scalar"] == 0.2
    assert out["verdict"] == "fail"   # NOT all-pass — the whole point


def test_cell_scorer_discriminates_cells() -> None:
    cs = tasks.make_cell_scorer(agent={"type": "scripted"}, objective=OBJ,
                                runner=_runner({"good": 0.95, "bad": 0.1}))
    assert cs({"persona": "good"})["verdict"] == "pass"
    assert cs({"persona": "bad"})["verdict"] == "fail"


def test_cell_scorer_is_not_the_allpass_default() -> None:
    # a bad cell must NOT score 1.0/pass (the no-op default the loop shipped with)
    cs = tasks.make_cell_scorer(agent={"type": "scripted"}, objective=OBJ,
                                runner=_runner({"x": 0.0}))
    out = cs({"persona": "x"})
    assert out["scalar"] != 1.0 and out["verdict"] == "fail"


@pytest.mark.integration
def test_cell_scorer_real_engine() -> None:
    cs = tasks.make_cell_scorer(
        agent={"type": "scripted", "content": "The refund policy is at /help/refunds."},
        objective=OBJ,
        scenario={"name": "r", "kind": "task", "dataset": [{"persona": {"name": "D"},
                  "situation": "Where is the refund policy?", "outcome": "States the policy."}]},
        threshold=0.3)
    out = cs({"intent": "refund", "persona": "D"})
    assert out["verdict"] in ("pass", "fail")
    assert isinstance(out["scalar"], float)
    assert out["metric_averages"]  # real metrics came back
