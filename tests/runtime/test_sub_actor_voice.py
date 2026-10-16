from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("livekit")

from fi.simulate.simulation.engines.sub_actor_voice import (
    SubActorVoice,
    sub_actor_messages,
    with_sub_actor_turns,
)


class _Stream:
    def __init__(self, items):
        self.items = items

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def __aiter__(self):
        async def gen():
            for item in self.items:
                yield item

        return gen()


class _Model:
    def __init__(self, text: str):
        self.text = text

    def chat(self, **_kwargs):
        return _Stream([SimpleNamespace(delta=SimpleNamespace(content=self.text))])


class _Tts:
    def synthesize(self, text: str):
        return _Stream([SimpleNamespace(frame=f"frame:{text}")])


def _voice(kind: str = "mid_journey", turn: int = 2, lines: str = "Hurry up.\nAsk for the extra.\nThird.", max_lines: int = 2):
    played: list = []

    async def play(frames, volume):
        played.append((frames, round(volume, 3)))

    ticks = iter(range(100, 200))
    voice = SubActorVoice(
        actor={
            "id": "subactor_bystander",
            "name": "Bystander",
            "role": "someone near the caller",
            "behaviour": {"prompt": "impatient", "loudness_db": -20, "max_lines": max_lines},
            "trigger": {"kind": kind, "turn": turn},
        },
        model=_Model(lines),
        tts=_Tts(),
        play=play,
        clock=lambda: float(next(ticks)),
    )
    return voice, played


def test_it_enters_after_the_agent_turn_its_trigger_names_and_only_once():
    async def scenario():
        voice, played = _voice(turn=2)
        voice.start()
        voice.agent_turn_ended(1)
        await asyncio.sleep(0)
        assert played == []
        voice.agent_turn_ended(2)
        await voice._delivery
        voice.agent_turn_ended(2)
        voice.agent_turn_ended(3)
        await asyncio.sleep(0)
        return voice, played

    voice, played = asyncio.run(scenario())
    assert [frames for frames, _ in played] == [["frame:Hurry up."], ["frame:Ask for the extra."]]
    assert played[0][1] == 0.1
    assert [event["text"] for event in voice.events()] == ["Hurry up.", "Ask for the extra."]
    assert all(event["sub_actor"] == "subactor_bystander" for event in voice.events())


def test_always_enters_after_the_first_agent_turn():
    voice, _played = _voice(kind="always")
    assert not voice.due(0) and voice.due(1)


def test_a_failed_preparation_leaves_it_silent_without_failing_the_call():
    async def scenario():
        voice, played = _voice()

        class Broken:
            def chat(self, **_kwargs):
                raise RuntimeError("model down")

        voice.model = Broken()
        voice.start()
        voice.agent_turn_ended(2)
        await voice._delivery
        await voice.close()
        return voice, played

    voice, played = asyncio.run(scenario())
    assert played == [] and voice.events() == []


def test_what_it_said_is_placed_in_the_transcript_where_it_was_heard():
    events = [{"sub_actor": "s", "name": "Bystander", "kind": "spoke", "text": "Hurry up.", "at": 15.0, "ended_at": 16.0}]
    messages = [
        {"role": "assistant", "content": "Hello", "started_speaking_at": 10.0},
        {"role": "user", "content": "Hi", "started_speaking_at": 20.0},
    ]
    merged = with_sub_actor_turns(messages, events)
    assert [one["role"] for one in merged] == ["assistant", "sub_actor", "user"]
    assert merged[1]["name"] == "Bystander" and merged[1]["content"] == "Hurry up."
    assert with_sub_actor_turns(messages, []) == messages
    assert sub_actor_messages([{"kind": "spoke", "text": "", "at": 1}]) == []
