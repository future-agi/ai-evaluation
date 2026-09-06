"""The harness's own spend, and that every session feeds it."""

from __future__ import annotations

import json

from fi.alk.harness import spend
from fi.alk.harness.backends import SessionSpec, StageDone
from fi.alk.harness.session import Stage, Turn


def _fresh(tmp_path):
    spend._stages.clear()
    spend.journal_to(tmp_path / "cost.json")
    return tmp_path / "cost.json"


def test_the_total_is_every_stage_added_up(tmp_path):
    journal = _fresh(tmp_path)
    spend.record("understand-agent", 0.0123, turns=4, models={"m"})
    spend.record("write-scenarios:slice-a", 0.5, turns=30, models={"m"})
    spend.record("write-scenarios:slice-b", 0.25, turns=20, models={"m"})

    body = json.loads(journal.read_text(encoding="utf-8"))
    assert body["total_usd"] == 0.7623
    assert [entry["stage"] for entry in body["stages"]] == [
        "understand-agent",
        "write-scenarios:slice-a",
        "write-scenarios:slice-b",
    ]


def test_a_turn_nobody_could_price_is_reported_rather_than_dropped(tmp_path):
    """An unpriced turn makes the total a floor, and a floor presented as a total is a wrong bill."""
    journal = _fresh(tmp_path)
    spend.record("build-environment", 0.02, turns=3, models={"m"})
    spend.record("build-environment", None, turns=1, models={"m"})

    body = json.loads(journal.read_text(encoding="utf-8"))
    assert body["total_usd"] == 0.02
    assert body["unpriced_turns"] == 1


def test_the_file_is_current_after_every_turn(tmp_path):
    """The platform reads this while the sandbox lives, so it cannot be written only at the end."""
    journal = _fresh(tmp_path)
    spend.record("understand-agent", 0.1, turns=1)
    assert json.loads(journal.read_text(encoding="utf-8"))["total_usd"] == 0.1
    spend.record("understand-agent", 0.2, turns=1)
    assert json.loads(journal.read_text(encoding="utf-8"))["total_usd"] == 0.3


def test_a_session_reports_its_own_cost_without_being_asked(tmp_path):
    """The hook is in `Stage`, so a stage added later is counted with no new call site."""
    journal = _fresh(tmp_path)
    stage = Stage(SessionSpec(system_prompt="x"), name="write-scenarios:slice-c")
    stage._events(
        StageDone(outcome="success", turns=7, cost_usd=0.4, models={"gemini"}),
        Turn(),
    )

    body = json.loads(journal.read_text(encoding="utf-8"))
    assert body["total_usd"] == 0.4
    assert body["stages"][0]["stage"] == "write-scenarios:slice-c"
    assert body["stages"][0]["turns"] == 7
    assert body["stages"][0]["models"] == ["gemini"]


def test_the_tokens_behind_a_price_are_kept(tmp_path):
    """A dollar figure with no units cannot be audited, and this one is a bill."""
    journal = _fresh(tmp_path)
    spend.record("understand-agent", 0.02, turns=2, models={"m"}, tokens_in=5000, tokens_out=1200)
    spend.record("understand-agent", 0.01, turns=1, models={"m"}, tokens_in=2000, tokens_out=300)

    entry = json.loads(journal.read_text(encoding="utf-8"))["stages"][0]
    assert entry["tokens_in"] == 7000
    assert entry["tokens_out"] == 1500
    assert entry["usd"] == 0.03


def test_a_session_reports_its_tokens_too(tmp_path):
    journal = _fresh(tmp_path)
    stage = Stage(SessionSpec(system_prompt="x"), name="build-environment")
    stage._events(
        StageDone(
            outcome="success",
            turns=3,
            cost_usd=0.5,
            models={"gemini"},
            tokens_in=9000,
            tokens_out=2500,
        ),
        Turn(),
    )

    entry = json.loads(journal.read_text(encoding="utf-8"))["stages"][0]
    assert (entry["tokens_in"], entry["tokens_out"]) == (9000, 2500)


def test_a_price_that_has_run_out_refuses_instead_of_billing_it(monkeypatch):
    """gemini-3.7-flash is introductory pricing that doubles on 2027-01-01.

    A hardcoded table cannot know that, so the table carries the last day it is good for and a
    figure past that day is withheld: `unpriced_turns` shows a gap, which somebody notices, where a
    stale price bills confidently and nobody does.
    """
    from datetime import date

    from fi.alk.harness.backends import vertex_gemini

    class _Before(date):
        @classmethod
        def today(cls):
            return date(2026, 12, 31)

    class _After(date):
        @classmethod
        def today(cls):
            return date(2027, 1, 1)

    monkeypatch.setattr(vertex_gemini, "date", _Before)
    assert vertex_gemini.priced("gemini-3.7-flash", 1_000_000, 1_000_000) == 4.5

    monkeypatch.setattr(vertex_gemini, "date", _After)
    assert vertex_gemini.priced("gemini-3.7-flash", 1_000_000, 1_000_000) is None
    assert vertex_gemini.priced("a-model-nobody-listed", 1_000, 1_000) is None
