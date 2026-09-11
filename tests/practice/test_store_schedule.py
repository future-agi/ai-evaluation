"""Unit 9 (BBG U9) — consolidation store + T1-T7 schedule + D7 boundary."""
from __future__ import annotations


from fi.alk.practice import _schedule
from fi.alk.practice._store import ConsolidationStore, build_record, record_id


def _record(round_no=0, interval=1, ladder="episodic", deck=("row_a",), cells=("c1",)):
    return build_record(
        lesson={"kind": "instruction_block", "payload": {"text": "x"}, "applies_to_paths": ["agent.instructions"]},
        source_justification={"pratijna": "improve", "hetu": "drill cell"},
        deck=list(deck),
        cells=list(cells),
        created_round=round_no,
        seed=42,
        interval_rounds=interval,
        ladder_state=ladder,
    )


# --- id recipe agreement ---------------------------------------------------
def test_id_recipe_agreement_with_frozen_row():
    from fi.alk.optimize import _sorted_json_digest as opt
    body = {"x": 1, "y": [2, 3]}
    rid = record_id(body)
    assert rid.startswith("lesson_")
    assert rid[len("lesson_"):] == opt(body)[:16]


# --- T-rows ----------------------------------------------------------------
def test_t1_pass_expands_interval():
    rec = _record(interval=2)
    out = _schedule.transition(rec, "review_pass", round_no=10)
    assert out["schedule"]["interval_rounds"] == 4
    assert out["schedule"]["due_round"] == 14
    assert out["schedule"]["consecutive_failures"] == 0


def test_interval_ladder_walk_caps_at_16():
    rec = _record(interval=1)
    r = 0
    for expected in (2, 4, 8, 16, 16):
        rec = _schedule.transition(rec, "review_pass", round_no=r)
        assert rec["schedule"]["interval_rounds"] == expected
        r += rec["schedule"]["interval_rounds"]


def test_t2_demote_above_episodic():
    rec = _record(interval=8, ladder="skill")
    out = _schedule.transition(rec, "review_fail", round_no=5)
    assert out["ladder_state"] == "instruction"
    assert out["schedule"]["interval_rounds"] == 1
    assert out["schedule"]["due_round"] == 6


def test_demotion_chain_to_retired():
    rec = _record(interval=8, ladder="skill")
    rec = _schedule.transition(rec, "review_fail", round_no=1)   # skill->instruction
    assert rec["ladder_state"] == "instruction"
    rec = _schedule.transition(rec, "review_fail", round_no=2)   # T4: failures>=2 retire
    assert rec["schedule"]["status"] == "retired"
    assert rec["schedule"]["retired_reason"] == "repeated_failure"


def test_t3_fail_at_episodic_retires():
    rec = _record(ladder="episodic")
    out = _schedule.transition(rec, "review_fail", round_no=3)
    assert out["schedule"]["status"] == "retired"
    assert out["schedule"]["retired_reason"] == "repeated_failure"


def test_t5_obsolescence_retires():
    rec = _record()
    out = _schedule.transition(rec, "obsolete", round_no=4)
    assert out["schedule"]["status"] == "retired"
    assert out["schedule"]["retired_reason"] == "obsolete"


def test_t7_retired_terminal():
    rec = _record()
    retired = _schedule.transition(rec, "obsolete", round_no=1)
    again = _schedule.transition(retired, "review_pass", round_no=2)
    assert again["schedule"]["status"] == "retired"


# --- store -----------------------------------------------------------------
def test_append_only_audit(tmp_path):
    store = ConsolidationStore(tmp_path / "records.jsonl")
    rec = _record()
    store.admit(rec)
    for i in range(3):
        rec = _schedule.transition(rec, "review_pass", round_no=i)
        store.update_record(rec)
    snapshots = store._read_snapshots()
    assert len(snapshots) == 4  # 1 admit + 3 transitions


def test_cap_refusal_leaves_records_untouched(tmp_path):
    store = ConsolidationStore(tmp_path / "records.jsonl", active_cap=2)
    a = _record(deck=("row_a",))
    b = _record(deck=("row_b",))
    c = _record(deck=("row_c",))
    assert store.admit(a)["admitted"]
    assert store.admit(b)["admitted"]
    out = store.admit(c)
    assert out["admitted"] is False
    assert out["status"] == "cap_deferred"
    # standing records untouched
    assert len(store.active_records()) == 2


def test_full_deck_ignores_schedule_state(tmp_path):
    """D7: full_deck is the union regardless of due/not-due (zero due ⇒ unchanged)."""
    store = ConsolidationStore(tmp_path / "records.jsonl")
    a = _record(deck=("row_a", "row_b"))
    a["schedule"]["due_round"] = 9999  # not due
    store.admit(a)
    deck = store.full_deck(frozen_rows=["frozen_1"])
    assert set(deck) == {"frozen_1", "row_a", "row_b"}
    # zero records due ⇒ union still includes the active record's deck
    due = _schedule.due_reviews(store.active_records(), round_no=0)
    assert due == []
    assert set(store.full_deck(frozen_rows=["frozen_1"])) == {"frozen_1", "row_a", "row_b"}


def test_due_reviews_deterministic_order(tmp_path):
    store = ConsolidationStore(tmp_path / "records.jsonl")
    a = _record(deck=("row_a",))
    a["schedule"]["due_round"] = 5
    b = _record(deck=("row_b",))
    b["schedule"]["due_round"] = 3
    store.admit(a)
    store.admit(b)
    due = _schedule.due_reviews(store.active_records(), round_no=10)
    assert [r["schedule"]["due_round"] for r in due] == [3, 5]


def test_schedule_module_has_no_promotion_path():
    """The 13D-D7 structural boundary: _schedule.py never calls full_deck (the
    promotion-row source) — it exposes only due_reviews."""
    import inspect
    src = inspect.getsource(_schedule)
    # the only reference to full_deck is inside the docstring narration; there is
    # no call site. Assert no executable reference (full_deck( ).
    assert "full_deck(" not in src
    assert "_update" not in src  # never imports the promotion invoker
