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
    """gemini-3.7-flash is introductory pricing that doubles on 2027-01-01."""
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


def test_both_backends_report_the_same_units():
    """One backend giving auditable tokens and the other not is a ledger you cannot reconcile."""
    from fi.alk.harness.backends.claude import _tokens

    assert _tokens({"m": {"inputTokens": 900, "outputTokens": 120}}) == {
        "tokens_in": 900,
        "tokens_out": 120,
    }
    assert _tokens(
        {"a": {"input_tokens": 5, "output_tokens": 6}, "b": {"input_tokens": 7, "output_tokens": 8}}
    ) == {"tokens_in": 12, "tokens_out": 14}
    assert _tokens(None) == {"tokens_in": 0, "tokens_out": 0}
    assert _tokens({"m": object()}) == {"tokens_in": 0, "tokens_out": 0}


def test_the_ledger_carries_input_the_provider_served_from_cache(tmp_path):
    """Two reruns of the same authoring produced byte-identical ledgers ($3.533797, identical
    token counts) four times apart in wall clock. That is what provider caching looks like when
    nothing records it, and it means a rerun's ledger overstates what was actually billed."""
    spend._stages.clear()
    spend.record("understand-agent", 1.0, turns=2, tokens_in=1000, tokens_out=50, tokens_cached=800)
    spend.record("understand-agent", 1.0, turns=1, tokens_in=500, tokens_out=25, tokens_cached=400)

    stage = spend.snapshot()["stages"][0]
    assert stage["tokens_in"] == 1500
    assert stage["tokens_cached"] == 1200, "the cached share has to be visible to be reconciled"
    assert stage["tokens_out"] == 75


def test_cached_tokens_default_to_zero_for_a_backend_that_does_not_report_them(tmp_path):
    spend._stages.clear()
    spend.record("write-scenarios", 0.5, turns=1, tokens_in=100, tokens_out=10)

    assert spend.snapshot()["stages"][0]["tokens_cached"] == 0


def test_the_default_model_is_one_we_can_price():
    """A run on an unpriced model reports a total that understates the bill."""
    from fi.alk.harness.backends.vertex_gemini import DEFAULT_MODEL, PRICES_PER_MILLION

    assert DEFAULT_MODEL in PRICES_PER_MILLION, (
        f"{DEFAULT_MODEL} has no entry in PRICES_PER_MILLION, so every run on it reports "
        "less than it cost"
    )


def test_every_harness_stage_feeds_the_one_ledger():
    """Parallel scenario writers and the suite review each open their own session."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "src" / "fi" / "alk" / "harness"
    # rglob, not glob: a second call site added under backends/ or world/ has to fail this too.
    recorded = sorted(
        path.name
        for path in root.rglob("*.py")
        if "spend.record(" in path.read_text(encoding="utf-8")
    )
    assert recorded == ["session.py"], (
        "spend must be recorded in exactly one place; a second call site double-counts or drifts, "
        f"found: {recorded}"
    )


def test_the_price_table_agrees_with_the_platform_model_table():
    """Prices are copied from the platform's litellm table, so drift is a silent mis-bill."""
    import json
    from pathlib import Path

    from fi.alk.harness.backends.vertex_gemini import PRICES_PER_MILLION

    # The platform checked out beside this repo, or nowhere: never an absolute path, which would
    # carry one machine's layout into a public repo and skip for everyone else.
    table = (
        Path(__file__).resolve().parents[3]
        / "future-agi"
        / "agentcc-gateway"
        / "internal"
        / "modeldb"
        / "litellm.json"
    )
    if not table.exists():
        import pytest

        pytest.skip("platform checkout not beside this repo")

    models = json.loads(table.read_text(encoding="utf-8"))
    for name, (want_in, want_out, _good_until) in PRICES_PER_MILLION.items():
        entry = models.get(f"vertex_ai/{name}") or models.get(name)
        assert entry, f"{name} is priced here but absent from the platform table"
        assert round(entry["input_cost_per_token"] * 1_000_000, 6) == want_in, name
        assert round(entry["output_cost_per_token"] * 1_000_000, 6) == want_out, name
