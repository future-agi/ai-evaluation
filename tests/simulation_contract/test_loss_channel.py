"""Unit 5 (BBG U5 / ARCH §2c) — the declared objective channel."""
from __future__ import annotations

import pytest

from fi.alk import loss as L
from fi.alk.loss import ObjectiveError


def _declared(evals=None, **over):
    base = {
        "evals": evals or [{"eval": "agent_report", "weight": 1.0}],
        "source": "declared",
        "guards": {"sentinel_rows": ["row_aaaa"], "min_guard_count": 1},
    }
    base.update(over)
    return base


def test_guard_mandatory_rejection():
    with pytest.raises(ObjectiveError, match="objective_guards_missing"):
        L.compile_objective({"evals": [{"eval": "agent_report"}], "source": "declared", "guards": {}})


def test_declared_with_guards_ok():
    obj = L.compile_objective(_declared())
    assert obj["kind"] == L.AGENT_LEARNING_OBJECTIVE_KIND
    assert obj["source"] == "declared"
    assert obj["version"].startswith("sha256:")


def test_derived_accepted_for_replication():
    obj = L.compile_objective({"evals": [{"eval": "agent_report"}], "source": "derived"})
    assert obj["source"] == "derived"


def test_refuse_derived_for_training():
    obj = L.compile_objective({"evals": [{"eval": "agent_report"}], "source": "derived"})
    with pytest.raises(ObjectiveError, match="objective_guards_missing"):
        L.refuse_derived_for_training(obj)
    # declared passes
    L.refuse_derived_for_training(L.compile_objective(_declared()))


def test_refuse_none_objective():
    with pytest.raises(ObjectiveError):
        L.refuse_derived_for_training(None)


def test_content_hash_stability():
    a = L.compile_objective(_declared())
    b = L.compile_objective(_declared())
    assert a["version"] == b["version"]


def test_verdict_row_admissibility():
    row = L.verdict_row(
        eval_ref="agent_report",
        cell={"intent": "resolve"},
        scalar=0.9,
        verdict="pass",
        evidence_class="local_gate",
    )
    assert row["admissible"] is True
    # unstable is never admissible
    row_u = L.verdict_row(eval_ref="x", cell={}, scalar=0.5, verdict="unstable", evidence_class="local_gate")
    assert row_u["admissible"] is False


def test_loss_composition_unstable_and_void():
    obj = L.compile_objective(_declared())
    rows = [
        L.verdict_row(eval_ref="agent_report", cell={"intent": "a"}, scalar=1.0, verdict="pass", evidence_class="local_gate"),
        L.verdict_row(eval_ref="agent_report", cell={"intent": "b"}, scalar=0.5, verdict="unstable", evidence_class="local_gate"),
        L.verdict_row(eval_ref="agent_report", cell={"intent": "c"}, scalar=0.0, verdict="void", evidence_class="local_gate"),
    ]
    report = L.loss_report(obj, rows, budget_consumed=3)
    by_cell = {c["cell"]["intent"]: c for c in report["cells"]}
    assert by_cell["a"]["loss"] == 0.0
    assert by_cell["b"]["unstable_mass"] == 1.0
    assert by_cell["c"]["void_count"] == 1
    assert report["budget_consumed"] == 3


def test_scalar_projection_arithmetic():
    obj = L.compile_objective(_declared(evals=[
        {"eval": "q", "weight": 3.0}, {"eval": "s", "weight": 1.0},
    ]))
    rows = [
        L.verdict_row(eval_ref="q", cell={"intent": "x"}, scalar=1.0, verdict="pass", evidence_class="local_gate"),
        L.verdict_row(eval_ref="s", cell={"intent": "x"}, scalar=0.0, verdict="fail", evidence_class="local_gate"),
    ]
    report = L.loss_report(obj, rows)
    # cell x: weighted mean of (1.0*3 + 0.0*1)/(3+1) = 0.75
    assert report["cells"][0]["loss"] == pytest.approx(0.25)
    assert report["scalar"] == pytest.approx(0.75)


def test_conjunction_open_cells():
    obj = L.compile_objective(_declared())
    rows = [
        L.verdict_row(eval_ref="agent_report", cell={"intent": "ok"}, scalar=1.0, verdict="pass", evidence_class="local_gate"),
        L.verdict_row(eval_ref="agent_report", cell={"intent": "bad"}, scalar=0.0, verdict="fail", evidence_class="local_gate"),
    ]
    report = L.loss_report(obj, rows)
    assert report["conjunction"]["closed"] is False
    assert any(c.get("intent") == "bad" for c in report["conjunction"]["open_cells"])


def test_objective_metric_weights_view():
    obj = L.compile_objective(_declared(evals=[
        {"eval": "world_contract", "weight": 4.0}, {"eval": "framework_trace", "weight": 3.0},
    ]))
    view = L.objective_metric_weights(obj)
    assert view == {"world_contract": 4.0, "framework_trace": 3.0}
