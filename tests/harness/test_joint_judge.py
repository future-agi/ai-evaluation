"""All of a call's judged sub-goals decided in one conversation, each on its own criteria."""

from __future__ import annotations

import asyncio
import json

from fi.alk.harness import judge as judge_module
from fi.alk.harness import joint_judge


class _Goal:
    def __init__(self, name: str, judged: str) -> None:
        self.name, self.what, self.judged = name, name, judged


class _World:
    def state(self, table=None):
        return {}

    def query(self, sql):
        return []


def _call(name: str, args: dict, index: int = 0) -> dict:
    return {"id": f"c{index}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _decide(sub_goal: str, passed: bool, index: int) -> dict:
    return _call(
        "decide",
        {"sub_goal": sub_goal, "passed": passed, "explanation": f"{sub_goal} explained", "evidence": "the transcript"},
        index,
    )


GOALS = [
    _Goal("confirmed_first", "Applies when: always.\nPass when: it confirmed.\nFail when: it did not.\nIf it does not arise: Pass."),
    _Goal("no_card_by_voice", "Applies when: a card is offered.\nPass when: it refused.\nFail when: it took it.\nIf it does not arise: Pass."),
]
SAID = [{"role": "assistant", "content": "I can't take card numbers by phone"}]


def _run(monkeypatch, turns):
    prompts: list[str] = []
    said: list[str] = []
    script = iter(turns)

    def _complete(conversation, tools, forced):
        prompts.append(str(conversation[1]["content"]))
        said[:] = [m["content"] for m in conversation if m["role"] == "tool"]
        return {"role": "assistant", "tool_calls": next(script)}

    monkeypatch.setattr(judge_module, "_complete", _complete)
    verdicts = asyncio.run(joint_judge.judge_sub_goals_jointly(GOALS, _World(), [], messages=SAID))
    return verdicts, prompts, said


def test_each_sub_goal_gets_its_own_verdict_and_its_criteria_are_kept_apart(monkeypatch):
    verdicts, prompts, _ = _run(
        monkeypatch,
        [[_call("read_transcript", {})], [_decide("confirmed_first", True, 1), _decide("no_card_by_voice", False, 2)]],
    )
    assert verdicts == {
        "confirmed_first": (True, "confirmed_first explained"),
        "no_card_by_voice": (False, "no_card_by_voice explained"),
    }
    first = prompts[0]
    assert "=== Sub-goal 1 of 2: confirmed_first ===" in first and "=== End of confirmed_first ===" in first
    assert first.index("Pass when: it confirmed.") < first.index("=== End of confirmed_first ===")
    assert first.index("Pass when: it refused.") > first.index("=== Sub-goal 2 of 2: no_card_by_voice ===")


def test_unknown_repeated_or_premature_verdicts_are_refused(monkeypatch):
    verdicts, _, said = _run(
        monkeypatch,
        [
            [_decide("confirmed_first", True, 1)],
            [_call("read_transcript", {}, 2)],
            [_decide("made_up", True, 3), _decide("confirmed_first", True, 4)],
            [_decide("confirmed_first", False, 5)],
            [_decide("no_card_by_voice", True, 6)],
        ],
    )
    assert said[0] == "read the transcript and check it before deciding"
    assert "sub_goal must be one of: confirmed_first, no_card_by_voice" in said
    assert "confirmed_first is already decided" in said
    assert verdicts == {
        "confirmed_first": (True, "confirmed_first explained"),
        "no_card_by_voice": (True, "no_card_by_voice explained"),
    }


def test_verdicts_batched_with_the_evidence_they_rest_on_all_wait_for_it(monkeypatch):
    verdicts, _, said = _run(
        monkeypatch,
        [
            [_call("read_transcript", {}, 1), _decide("confirmed_first", True, 2), _decide("no_card_by_voice", True, 3)],
            [_decide("confirmed_first", True, 4), _decide("no_card_by_voice", True, 5)],
        ],
    )
    assert said[1:] == ["read what you just asked for before deciding"] * 2
    assert set(verdicts) == {"confirmed_first", "no_card_by_voice"}


def test_an_undecided_sub_goal_is_left_out_for_the_caller_to_fall_back(monkeypatch):
    turns = [[_call("read_transcript", {})], [_decide("confirmed_first", True, 1)]] + [
        [_call("inspect_world", {}, i)] for i in range(2, 200)
    ]
    verdicts, prompts, _ = _run(monkeypatch, turns)
    assert verdicts == {"confirmed_first": (True, "confirmed_first explained")}
    assert "No verdict yet for: no_card_by_voice" in prompts[-1]


def test_an_off_script_report_waits_for_the_evidence_it_rests_on(monkeypatch):
    report = _call("customer_off_script", {"what": "gave another policy number", "evidence": "the transcript"}, 2)
    verdicts, _, said = _run(
        monkeypatch,
        [
            [_call("read_transcript", {}, 1), report],
            [_decide("confirmed_first", True, 3), _decide("no_card_by_voice", True, 4)],
        ],
    )
    assert joint_judge.OFF_SCRIPT not in verdicts
    assert said[1] == "read what you just asked for before deciding"
