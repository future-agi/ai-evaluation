"""Unit 12 (BBG U12) — scoped update + the D7 enforcement point."""
from __future__ import annotations

import inspect

from fi.alk.practice import _update
from fi.alk.practice._budget import BudgetMeter
from fi.alk.practice._store import ConsolidationStore, build_record


def _store_with_records(tmp_path, decks):
    store = ConsolidationStore(tmp_path / "records.jsonl")
    for i, deck in enumerate(decks):
        rec = build_record(
            lesson={"kind": "config_patch", "payload": {}, "applies_to_paths": []},
            source_justification={}, deck=list(deck), cells=[f"c{i}"],
            created_round=0, seed=42,
        )
        # make them NOT due (schedule state must be irrelevant to the sweep)
        rec["schedule"]["due_round"] = 99999
        store.admit(rec)
    return store


def _deficit():
    return {"cell": {"intent": "a"}, "harness_layer": "verification"}


def test_locality_breach_recorded_not_blocked(tmp_path):
    store = _store_with_records(tmp_path, [])
    rec = _update.update(
        _deficit(),
        allowed_layer="verification",
        allowed_paths=["evaluation.threshold"],
        proposals=[
            {"patch": {"evaluation.threshold": 0.8}, "justification": {"hetu": "x"}},
            {"patch": {"agent.model": "gpt"}, "justification": {"hetu": "y"}},  # out-of-layer
        ],
        store=store, frozen_rows=["f1"], replay_row=lambda r: True,
        meter=BudgetMeter(100),
    )
    assert len(rec["locality_breaches"]) == 1
    assert rec["locality_breaches"][0]["recorded_as"] == "asiddha"
    assert rec["locality_breaches"][0]["path"] == "agent.model"
    # the in-layer proposal is the selected candidate
    assert rec["selected_candidate"]["patch"] == {"evaluation.threshold": 0.8}


def test_promotion_sweep_replays_full_deck_zero_due(tmp_path):
    """D7 unit twin of gate clause (c): the sweep replays the FULL union even
    though ZERO records are due (schedule state is irrelevant)."""
    store = _store_with_records(tmp_path, [("row_a", "row_b"), ("row_c",)])
    sweep = _update.promotion_sweep(store, frozen_rows=["frozen_1"], replay_row=lambda r: True)
    assert set(sweep["rows_replayed"]) == {"frozen_1", "row_a", "row_b", "row_c"}
    assert sweep["row_count"] == 4
    assert sweep["all_closed"] is True


def test_veto_propagation(tmp_path):
    store = _store_with_records(tmp_path, [("row_a",)])
    sweep = _update.promotion_sweep(
        store, frozen_rows=["frozen_1"],
        replay_row=lambda r: r != "row_a",  # row_a flips
    )
    assert sweep["veto"] is True
    assert sweep["vetoed_rows"] == ["row_a"]
    assert sweep["hetvabhasa_class"] == "badhita"


def test_operator_slice_charged_to_meter(tmp_path):
    store = _store_with_records(tmp_path, [])
    meter = BudgetMeter(100)
    _update.update(
        _deficit(), allowed_layer="verification", allowed_paths=["evaluation.threshold"],
        proposals=[{"patch": {"evaluation.threshold": 0.8}, "justification": {}}],
        store=store, frozen_rows=[], replay_row=lambda r: True,
        meter=meter, budget_fraction=0.1,
    )
    assert meter.consumed >= 10  # the 0.1 slice was charged


def test_update_module_never_imports_schedule():
    """The 13D-D7 structural boundary: _update.py never imports _schedule."""
    src = inspect.getsource(_update)
    assert "import _schedule" not in src
    assert "from ._schedule" not in src
    assert "due_reviews" not in src


def test_sweep_full_deck_charges_meter(tmp_path):
    store = _store_with_records(tmp_path, [("row_a", "row_b")])
    meter = BudgetMeter(100)
    sweep = _update.promotion_sweep(store, frozen_rows=["f1"], replay_row=lambda r: True, meter=meter)
    assert meter.consumed == sweep["row_count"]  # one charge per replayed row
