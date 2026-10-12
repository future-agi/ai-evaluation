"""Judged sub-goals are decided by a model, against the world the call left behind."""

from __future__ import annotations

import asyncio

import pytest

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
        def __init__(self, spec, name="", overheard=True):
            self.spec, self.name, self.overheard = spec, name, overheard

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


def test_a_judge_that_will_not_commit_is_unjudged_and_says_nothing_internal(monkeypatch):
    (held, why), _ = _drive(
        monkeypatch, {"undecided": True, "explanation": "no table records intent"}
    )
    assert held is None
    assert why == ""


def test_a_verdict_given_as_text_is_not_read_as_a_pass(monkeypatch):
    (held, _), _ = _drive(monkeypatch, {"passed": "false", "explanation": "the agent refused"})
    assert held is None


def test_a_verdict_with_no_explanation_is_refused(monkeypatch):
    """An explanation-free verdict is the placeholder again, so decide() refuses it."""
    (held, why), _ = _drive(monkeypatch, {"passed": True, "explanation": "  "})
    assert held is None
    assert why == ""


def test_a_judge_that_raises_never_fails_the_agent(monkeypatch):
    (held, why), _ = _drive(monkeypatch, {"passed": True, "explanation": "x"}, raises=True)
    assert held is None
    assert why == ""


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
    from fi.alk.harness.backends import resolve, vertex_gemini

    for name in (judge_module.JUDGE_MODEL_ALIAS, "ALK_HARNESS_MODEL", "ALK_HARNESS"):
        monkeypatch.delenv(name, raising=False)

    backend = resolve()
    model = judge_module.judge_model()

    assert model == backend.default_model
    if backend.name == "vertex-gemini":
        assert model in vertex_gemini.PRICES_PER_MILLION, (
            f"the judge would run unpriced on {model}, so its spend would be missing "
            "from the ledger"
        )


def test_a_long_cell_is_trimmed_rather_than_flooding_the_judge():
    trimmed = judge_module._short({"blob": "x" * 5000})
    assert len(str(trimmed)) < 1000
    assert "5000 chars" in str(trimmed)


def test_the_judge_is_given_what_was_said_not_only_what_was_done():
    """A claim about wording has to reach the transcript, or it can only ever be undecided."""
    said = [{"role": "user", "content": "one large fries"}] * 400
    said.append({"role": "assistant", "content": "your order is one large fries"})
    rendered = judge_module._transcript(said)
    assert "Customer: one large fries" in rendered
    assert "Agent: your order is one large fries" in rendered
    assert "your order is one large fries" in rendered
    assert len(rendered) <= judge_module._TRANSCRIPT_LIMIT + 8


def test_a_judging_stage_is_not_overheard_in_the_chat() -> None:
    from pathlib import Path as _P

    source = _P("src/fi/alk/harness/judge.py").read_text(encoding="utf-8")
    assert 'name="judge-sub-goals", overheard=False' in source

    session = _P("src/fi/alk/harness/session.py").read_text(encoding="utf-8")
    assert "if self._overheard:" in session
    assert "self.channel.waiting() if self._overheard else []" in session


def test_the_retry_carries_the_evidence_and_can_settle_the_claim(monkeypatch):
    prompts: list[str] = []

    class _Stage:
        def __init__(self, spec, name="", overheard=True):
            self.spec = spec

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def say(self, prompt):
            prompts.append(prompt)
            if len(prompts) == 2:
                tools = {t.name: t for t in self.spec.servers["world"].tools}
                await tools["decide"].handler(
                    {"passed": True, "explanation": "the agent confirmed the booking"}
                )

    class _Scenario:
        instruction = "the caller books a ride to the airport"
        tests = "the agent confirms before booking"

    monkeypatch.setattr(judge_module, "Stage", _Stage)
    said = [{"role": "assistant", "content": "shall I book it?"}]
    calls = [type("C", (), {"name": f"step_{i}", "arguments": {}, "result": None})() for i in range(30)]
    held, why = asyncio.run(
        judge_module.judge(_Goal(), _World(), calls, messages=said, scenario=_Scenario())
    )
    assert (held, why) == (True, "the agent confirmed the booking")
    assert "books a ride to the airport" in prompts[0]
    assert "step_29" in prompts[0]
    assert "shall I book it?" in prompts[1]


def test_with_no_actions_the_judge_is_told_the_conversation_is_the_evidence(monkeypatch):
    prompts: list[str] = []

    class _Stage:
        def __init__(self, spec, name="", overheard=True):
            self.spec = spec

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def say(self, prompt):
            prompts.append(prompt)
            tools = {t.name: t for t in self.spec.servers["world"].tools}
            await tools["decide"].handler({"passed": True, "explanation": "it read the fee first"})

    monkeypatch.setattr(judge_module, "Stage", _Stage)
    asyncio.run(judge_module.judge(_Goal(), _World(rows=[]), []))
    assert "only the conversation is observable" in prompts[0]
    call = type("C", (), {"name": "lookup", "arguments": {}, "result": None})()
    asyncio.run(judge_module.judge(_Goal(), _World(), [call]))
    assert "only the conversation is observable" not in prompts[1]


def test_the_criteria_decide_a_situation_that_did_not_arise():
    text = judge_module._INSTRUCTIONS
    assert "what the verdict is when its situation does not arise" in text
    assert "whether the agent's own behaviour kept it from arising" in text
    assert "never come up" not in text and "decide false" not in text


def test_the_customer_is_context_and_never_judged():
    text = judge_module._INSTRUCTIONS
    assert "The customer is not being evaluated" in text
    assert "Never hold the agent to more than" in text


def test_an_answer_with_nothing_to_check_it_against_is_never_called_accurate():
    assert "never call what the agent said accurate" in judge_module._INSTRUCTIONS


def test_the_judge_is_shown_the_customer_side_and_the_agent_instructions():
    from types import SimpleNamespace

    scenario = SimpleNamespace(presented={"situation": "Asks for a refund.", "outcome": "Refund policy."})
    side = judge_module._customer_side(scenario)
    assert "for context only" in side and "Asks for a refund." in side
    assert judge_module._customer_side(None) == ""


def test_a_one_sentence_sub_goal_is_read_as_what_the_agent_must_do() -> None:
    from fi.alk.harness.catalogue import criteria_text

    labelled = "Applies when: x.\nPass when: y.\nFail when: z.\nIf it does not arise: Pass."
    assert criteria_text(labelled) == labelled
    read = criteria_text("Confirms the address back")
    assert read.startswith("Pass when: Confirms the address back") and "If it does not arise:" in read


def test_a_caller_outside_the_accented_language_meets_an_accented_level_without_an_accent() -> None:
    from fi.alk.harness.scenario import Persona, Scenario, _condition_the_call_lacks

    persona = Persona(name="A B", languages=["French"], accent="British")
    assert persona.accent == "Neutral"
    scenario = Scenario(name="s", use_case="u", persona=persona, coverage={"interface": "non_native"})
    assert _condition_the_call_lacks(scenario) == ""
    english = Scenario(
        name="t", use_case="u", persona=Persona(name="C D", languages=["English"]), coverage={"interface": "accented"}
    )
    assert "accent not set" in _condition_the_call_lacks(english)


# (persona languages, agent languages, override, expected STT language)
_STT_CASES = [
    # English callers keep their own variant; the agent's languages never move them.
    (["English"], "", "", "en-US"),
    (["en-GB"], "", "", "en-GB"),
    (["English"], "Arabic", "", "en-US"),
    # Covered by the multilingual model: always multi, so English and the caller's language are heard.
    (["Spanish"], "", "", "multi"),
    (["Spanish"], "Spanish", "", "multi"),
    (["Spanish"], "English", "", "multi"),
    (["French"], "", "", "multi"),
    (["Hindi"], "", "", "multi"),
    (["es"], "", "", "multi"),
    # Not covered: the caller's own language unless the agent declares languages without it.
    (["Arabic"], "", "", "ar"),
    (["Arabic"], "Arabic", "", "ar"),
    (["Arabic"], "Arabic,English", "", "ar"),
    (["Arabic"], "English", "", "multi"),
    (["Arabic"], "english", "", "multi"),
    (["Arabic"], " English , French ", "", "multi"),
    (["Arabic"], "en", "", "multi"),
    (["Polish"], "English", "", "multi"),
    (["ko"], "", "", "ko"),
    (["ko"], "Korean", "", "ko"),
    # A declared list holding anything unrecognised is ignored as a whole.
    (["Arabic"], "Englsh", "", "ar"),
    (["Arabic"], "English (US)", "", "ar"),
    (["Arabic"], "English,Klingon", "", "ar"),
    (["Arabic"], "English;French", "", "ar"),
    (["Arabic"], " , ", "", "ar"),
    # Several persona languages, or none, are unchanged by the agent's list.
    (["Arabic", "English"], "English", "", "multi"),
    ([], "English", "", "en"),
    # An explicit override always wins.
    (["Arabic"], "English", "ar", "ar"),
    (["Spanish"], "", "es", "es"),
]


@pytest.mark.parametrize(("languages", "agent", "override", "expected"), _STT_CASES)
def test_the_simulator_transcribes_the_agent_in_the_language_it_will_hear(
    languages: list[str], agent: str, override: str, expected: str
) -> None:
    from fi.alk.harness.simulator_voice import persona_stt_language

    assert persona_stt_language({"languages": languages}, override, agent) == expected


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("multi", ("deepgram", "nova-3", "multi")),
        ("es", ("deepgram", "nova-3", "multi")),
        ("en-US", ("deepgram", "nova-3", "multi")),
        ("ar", ("deepgram", "nova-3", "ar")),
        ("ko", ("deepgram", "nova-3", "ko")),
        ("", ("deepgram", "nova-3", "en-US")),
    ],
)
def test_the_transcriber_sends_multi_only_where_the_model_covers_it(
    language: str, expected: tuple[str, str, str]
) -> None:
    from fi.alk.harness.simulator_voice import transcriber_for

    assert transcriber_for(language) == expected


def test_the_simulator_definition_reads_the_agents_languages_from_its_settings() -> None:
    from fi.alk.harness.simulator_voice import simulator_definition

    def heard(settings: dict[str, str]) -> str:
        definition = simulator_definition(lambda key: settings.get(key, ""), {"languages": ["Arabic"]})
        stt = definition.stt
        return stt["language"] if isinstance(stt, dict) else stt.language

    assert heard({}) == "ar"
    assert heard({"ALK_AGENT_LANGUAGES": "English"}) == "multi"
    assert heard({"ALK_AGENT_LANGUAGES": "Englsh"}) == "ar"
    assert heard({"ALK_AGENT_LANGUAGES": "English", "SIMULATOR_STT_LANGUAGE": "ar"}) == "ar"


def test_the_agents_declared_languages_reach_the_call_environment(tmp_path) -> None:
    import json

    from fi.alk.harness import call_runner
    from fi.alk.harness.hosted_entrypoint import _bundle_contract_list

    (tmp_path / "contract.json").write_text(json.dumps({"agent_languages": ["English", " ", "Arabic"]}))
    assert _bundle_contract_list(tmp_path, "agent_languages") == ["English", "Arabic"]
    assert _bundle_contract_list(tmp_path, "missing") == []
    (tmp_path / "contract.json").write_text(json.dumps({"agent_languages": "English"}))
    assert _bundle_contract_list(tmp_path, "agent_languages") == []
    (tmp_path / "contract.json").write_text("{not json")
    assert _bundle_contract_list(tmp_path, "agent_languages") == []
    (tmp_path / "contract.json").unlink()
    assert _bundle_contract_list(tmp_path, "agent_languages") == []
    # The call runner forwards only *_ALIAS environment names to the call process.
    assert call_runner.AGENT_LANGUAGES_ALIAS == "ALK_AGENT_LANGUAGES"


def test_the_contract_records_the_agents_languages_and_older_contracts_still_load() -> None:
    from fi.alk.harness.contract import AgentContract

    base = {"agent": "a", "summary": "s"}
    assert AgentContract.model_validate(base).agent_languages == []
    assert AgentContract.model_validate({**base, "agent_languages": "English"}).agent_languages == ["English"]
    assert AgentContract.model_validate({**base, "agent_languages": ["English", "Hindi"]}).agent_languages == [
        "English",
        "Hindi",
    ]
