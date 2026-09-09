"""Judged sub-goals are decided by a model, against the world the call left behind."""

from __future__ import annotations

import asyncio

from fi.alk.harness import judge as judge_module


class _Goal:
    def __init__(self, judged: str = "nothing observable shows intent") -> None:
        self.name = "removal_honoured"
        self.what = "the caller's removal request is recorded"
        self.judged = judged


class _World:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self.rows = rows if rows is not None else [{"id": 1, "opted_out": True}]
        self.queried: list[str] = []

    def state(self, table: str | None = None) -> dict:
        return {"contacts": self.rows} if table is None else {table: self.rows}

    def query(self, sql: str, params=()) -> list[dict]:
        self.queried.append(sql)
        return self.rows


class _RejectingWorld(_World):
    def query(self, sql: str, params=()) -> list[dict]:
        self.queried.append(sql)
        raise RuntimeError("relation sqlite_master does not exist")


def _drive(monkeypatch, decision: dict, *, world=None, raises: bool = False):
    """Run the judge with the model replaced by a scripted decide() call."""
    world = world or _World()

    class _Stage:
        def __init__(self, spec, name=""):
            self.spec, self.name = spec, name

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def say(self, _prompt):
            if raises:
                raise RuntimeError("provider exploded")
            tools = {t.name: t for t in self.spec.servers["world"].tools}
            if decision.get("look"):
                await tools["inspect_world"].handler({"table": "contacts"})
                await tools["query_world"].handler({"sql": "select 1"})
            await tools["decide"].handler(decision)
            return None

    monkeypatch.setattr(judge_module, "Stage", _Stage)
    return asyncio.run(judge_module.judge(_Goal(), world, [])), world


def test_a_judged_sub_goal_can_fail(monkeypatch):
    """The whole point: before this, every judged sub-goal passed before anything looked."""
    (held, why), _ = _drive(
        monkeypatch, {"passed": False, "explanation": "contacts row 1 still has opted_out false"}
    )
    assert held is False
    assert "opted_out" in why


def test_a_judged_sub_goal_can_pass_with_its_explanation(monkeypatch):
    (held, why), _ = _drive(
        monkeypatch, {"passed": True, "explanation": "contacts row 1 shows opted_out true"}
    )
    assert held is True
    assert why == "contacts row 1 shows opted_out true"


def test_the_judge_reads_the_live_world_before_deciding(monkeypatch):
    """It is given the final state, not a snapshot: the query has to reach the world handle."""
    (held, _), world = _drive(
        monkeypatch, {"passed": True, "explanation": "row seen", "look": True}
    )
    assert held is True
    assert world.queried == ["select 1"]


def test_bad_model_sql_is_returned_as_recoverable_tool_feedback(monkeypatch):
    world = _RejectingWorld()
    (held, _), world = _drive(
        monkeypatch,
        {"passed": True, "explanation": "contacts row seen", "look": True},
        world=world,
    )
    assert held is True
    assert world.queried == ["select 1"]


def test_an_undecided_judge_reports_unjudged_not_failed(monkeypatch):
    """A judge that cannot tell is not evidence against the agent; the platform skips held None."""
    (held, why), _ = _drive(
        monkeypatch, {"undecided": True, "explanation": "no table records intent"}
    )
    assert held is None
    assert "intent" in why


def test_a_verdict_with_no_explanation_is_refused(monkeypatch):
    """An explanation-free verdict is the placeholder again, so decide() refuses it."""
    (held, why), _ = _drive(monkeypatch, {"passed": True, "explanation": "  "})
    assert held is None
    assert "without a verdict" in why


def test_a_judge_that_raises_never_fails_the_agent(monkeypatch):
    (held, why), _ = _drive(monkeypatch, {"passed": True, "explanation": "x"}, raises=True)
    assert held is None
    assert "could not run" in why


def test_the_judge_model_is_changeable(monkeypatch):
    monkeypatch.setenv(judge_module.JUDGE_MODEL_ALIAS, "gemini-3.8-flash")
    assert judge_module.judge_model() == "gemini-3.8-flash"
    monkeypatch.delenv(judge_module.JUDGE_MODEL_ALIAS)
    assert judge_module.judge_model()  # falls back to the harness model, whatever it is


def test_the_judge_needs_no_environment_to_pick_a_priced_model(monkeypatch):
    """With nothing set anywhere the judge still runs, on a model the ledger can price.

    The override must stay an override: a deployment that sets none of these has to work, or the
    judge would need a new env var to be usable at all.
    """
    from fi.alk.harness.backends import vertex_gemini

    for name in (judge_module.JUDGE_MODEL_ALIAS, "ALK_HARNESS_MODEL", "ALK_HARNESS"):
        monkeypatch.delenv(name, raising=False)

    model = judge_module.judge_model()

    assert model == vertex_gemini.DEFAULT_MODEL
    assert model in vertex_gemini.PRICES_PER_MILLION, (
        f"the judge would run unpriced on {model}, so its spend would be missing from the ledger"
    )


def test_a_long_cell_is_trimmed_rather_than_flooding_the_judge():
    trimmed = judge_module._short({"blob": "x" * 5000})
    assert len(str(trimmed)) < 1000
    assert "5000 chars" in str(trimmed)


def test_the_judge_is_given_what_was_said_not_only_what_was_done():
    """A claim about wording has to reach the transcript, or it can only ever be undecided.

    Three drivethru, three insurance and two hotel scenarios errored on `judge_undecided` for
    exactly this reason: the sub-goal asked whether the agent said something and the judge was
    handed tool calls alone.
    """
    said = [
        {"role": "user", "content": "one large fries please"},
        {"role": "assistant", "content": "That is one large fries. Anything else?"},
    ]
    rendered = judge_module._transcript(said)
    assert "user: one large fries please" in rendered
    assert "assistant: That is one large fries. Anything else?" in rendered


def test_a_long_transcript_keeps_its_tail_where_a_readback_would_be():
    said = [{"role": "assistant", "content": "x" * 200} for _ in range(400)]
    said.append({"role": "assistant", "content": "your order is one large fries"})
    rendered = judge_module._transcript(said)
    assert len(rendered) <= judge_module._TRANSCRIPT_LIMIT + 64
    assert "your order is one large fries" in rendered
    assert rendered.startswith("... [earlier turns trimmed]")


def test_a_session_with_no_transcript_says_so_rather_than_inventing_one():
    assert judge_module._transcript([]) == ""
