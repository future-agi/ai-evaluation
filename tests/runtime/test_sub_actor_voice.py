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


def _voice(
    kind: str = "mid_journey",
    turn: int = 2,
    lines: str = "Hurry up.\nAsk for the extra.\nThird.",
    max_lines: int = 2,
    channel: str = "audio_mix",
    **trigger,
):
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
            "trigger": {"kind": kind, "turn": turn, **trigger},
            "channel": channel,
        },
        model=_Model(lines),
        tts=_Tts(),
        play=play,
        say=lambda frames, text, name: _said(played, frames, text),
        tools_called=lambda: ["lookup", "book"],
        seed="run-1",
        clock=lambda: float(next(ticks)),
    )
    voice.pause, voice.chance = (0.0, 0.0), 1.0
    return voice, played


async def _said(played, frames, text):
    played.append(("said", text))


def test_it_enters_after_the_agent_turn_its_trigger_names_and_only_once():
    async def scenario():
        voice, played = _voice(turn=2)
        voice.start()
        await voice.on_agent_turn(1)
        await asyncio.gather(*voice._tasks)
        assert played == []
        await voice.on_agent_turn(2)
        await voice.on_agent_turn(2)
        await voice.on_agent_turn(3)
        await asyncio.gather(*voice._tasks)
        return voice, played

    voice, played = asyncio.run(scenario())
    assert [frames for frames, _ in played] == [["frame:Hurry up."], ["frame:Ask for the extra."]]
    assert played[0][1] == 0.1
    assert [event["text"] for event in voice.events()] == ["Hurry up.", "Ask for the extra."]
    assert all(event["sub_actor"] == "subactor_bystander" for event in voice.events())


def test_a_voice_the_provider_cannot_speak_falls_back_to_its_second_voice():
    class Unknown:
        def synthesize(self, text: str):
            raise RuntimeError("voice not found")

    async def scenario():
        voice, played = _voice(kind="always", channel="audio_handoff")
        voice.tts, voice.fallback_tts = Unknown(), _Tts()
        assert await voice.on_agent_turn(1) is True
        return voice, played

    voice, played = asyncio.run(scenario())
    assert played == [("said", "Hurry up.")]
    assert [event["text"] for event in voice.events()] == ["Hurry up."]


def test_a_background_voice_says_one_line_per_agent_turn_not_all_at_once():
    async def scenario():
        voice, played = _voice(turn=1)
        voice.start()
        await voice.on_agent_turn(1)
        await asyncio.gather(*voice._tasks)
        first = len(played)
        await voice.on_agent_turn(2)
        await asyncio.gather(*voice._tasks)
        return first, played

    first, played = asyncio.run(scenario())
    assert first == 1
    assert [frames for frames, _ in played] == [["frame:Hurry up."], ["frame:Ask for the extra."]]


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
        await voice.on_agent_turn(2)
        await asyncio.gather(*voice._tasks)
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


def _run(voice, *turns):
    held = []

    async def scenario():
        voice.start()
        for number, text in turns:
            held.append(await voice.on_agent_turn(number, text))
            await asyncio.gather(*voice._tasks)
        await voice.close()

    asyncio.run(scenario())
    return held


def test_someone_who_takes_the_line_speaks_one_line_per_agent_turn_for_its_turns():
    voice, played = _voice(kind="mid_journey", turn=1, channel="audio_handoff", lines="Let me talk to them.", max_lines=2)
    held = _run(voice, (1, "How can I help?"), (2, "Sure."), (3, "Anything else?"))
    assert played == [("said", "Let me talk to them."), ("said", "Let me talk to them.")]
    assert held == [True, True, False]
    assert [event["kind"] for event in voice.events()] == ["spoke", "spoke"]


def test_a_keyword_brings_a_sub_actor_in_only_when_it_is_said():
    voice, played = _voice(kind="on_keyword", channel="audio_handoff", lines="I can help.", max_lines=1, words=["manager"])
    _run(voice, (1, "Let me check that."), (2, "I will get my manager."))
    assert played == [("said", "I can help.")]


def test_a_tool_call_brings_in_a_sub_actor_waiting_for_it():
    voice, played = _voice(kind="on_tool", channel="audio_handoff", lines="Add mine too.", max_lines=1, tool="book")
    _run(voice, (1, "Booking that now."))
    assert played == [("said", "Add mine too.")]

    voice, played = _voice(kind="on_tool", channel="audio_handoff", lines="Add mine too.", max_lines=1, tool="refund")
    _run(voice, (1, "Booking that now."))
    assert played == []


def test_a_probability_trigger_is_drawn_once_from_the_run_seed():
    from fi.simulate.simulation.sub_actors import drawn

    entered = [drawn(f"run-{n}", "s", 0.5) for n in range(200)]
    assert 60 < sum(entered) < 140
    assert drawn("run-7", "s", 0.5) == drawn("run-7", "s", 0.5)
    voice, _played = _voice(kind="probability", turn=1, p=0.999999)
    assert voice.due(1)
    voice, _played = _voice(kind="probability", turn=1, p=0.000001)
    assert not voice.due(1)


def test_each_committed_agent_turn_is_counted_once_across_the_call():
    from fi.simulate.simulation.engines.sub_actor_voice import SubActorsOnCall

    voice, played = _voice(kind="mid_journey", turn=2, channel="audio_handoff", lines="Mine now.", max_lines=1)
    call = SubActorsOnCall([voice])

    async def scenario():
        return [await call.agent_turn("one"), await call.agent_turn("two"), await call.agent_turn("three")]

    assert asyncio.run(scenario()) == [False, True, False]
    assert played == [("said", "Mine now.")]


def test_a_menu_bound_always_answers_before_anyone_speaks():
    from fi.simulate.simulation.engines.sub_actor_voice import SubActorsOnCall

    voice, played = _voice(kind="always", channel="ivr", lines="Press one for sales.", max_lines=2)
    call = SubActorsOnCall([voice])
    assert call.answers_first()

    async def scenario():
        await call.open_line()
        return [await call.agent_turn("one"), await call.agent_turn("two")]

    assert asyncio.run(scenario()) == [True, False]
    assert played == [("said", "Press one for sales."), ("said", "Press one for sales.")]


def test_a_forwarded_agent_turn_reaches_the_sub_actors_and_the_person_stays_silent_while_they_hold_the_line():
    from test_livekit_engine import _FakeReplySession, _FakeTranscriptionReader

    from fi.simulate.simulation.engines import livekit

    heard = []

    async def holds(text):
        heard.append(text)
        return text.startswith("transfer")

    async def exercise():
        session = _FakeReplySession()
        await livekit._forward_target_transcription(
            _FakeTranscriptionReader("Let me check."), session, sub_actor_turn=holds
        )
        await livekit._forward_target_transcription(
            _FakeTranscriptionReader("transfer you now"), session, sub_actor_turn=holds
        )
        return session

    session = asyncio.run(exercise())
    assert heard == ["Let me check.", "transfer you now"]
    assert session.reply_inputs == ["Let me check."]
    assert ("user", "transfer you now") in session.history_adds


def test_a_heard_agent_turn_reaches_the_sub_actors_once_and_silences_the_person_while_they_hold_the_line():
    from livekit.agents.llm import ChatMessage
    from test_livekit_engine import _scenario

    from fi.simulate.simulation.engines import livekit

    heard = []

    async def holds(text):
        heard.append(text)
        return text == "hold"

    async def exercise():
        agent = livekit._TestRunnerAgent(persona=_scenario().dataset[0], instructions="Be a customer.")
        agent._sub_actor_turn = holds
        await agent.on_user_turn_completed(None, SimpleNamespace(text_content="pass"))
        with pytest.raises(livekit.StopResponse):
            await agent.on_user_turn_completed(None, ChatMessage(role="user", content=["hold"]))

    asyncio.run(exercise())
    assert heard == ["pass", "hold"]


def test_someone_who_gives_the_phone_back_lets_the_person_speak_next():
    voice, played = _voice(kind="mid_journey", turn=1, channel="audio_handoff", lines="Here they are. [BACK]", max_lines=3)
    held = _run(voice, (1, "What is the code?"), (2, "Thanks."))
    assert played == [("said", "Here they are.")]
    assert held == [True, False]


def test_a_line_that_cannot_be_heard_never_leaves_the_person_silent():
    class Silent:
        def synthesize(self, text: str):
            raise RuntimeError("voice not found")

    voice, played = _voice(kind="mid_journey", turn=1, channel="audio_handoff", lines="Mine now.", max_lines=3)
    voice.tts, voice.fallback_tts = Silent(), None
    held = _run(voice, (1, "Hello?"), (2, "Anyone there?"))
    assert played == [] and held == [False, False]


def test_a_sub_actor_too_slow_to_speak_gives_the_line_back(monkeypatch):
    from fi.simulate.simulation.engines import sub_actor_voice
    from fi.simulate.simulation.engines.sub_actor_voice import SubActorsOnCall

    monkeypatch.setattr(sub_actor_voice, "_TURN_SECONDS", 0.05)
    voice, _played = _voice(kind="mid_journey", turn=1, channel="audio_handoff", lines="Mine now.", max_lines=3)

    async def hangs(*_args, **_kwargs):
        await asyncio.sleep(10)

    voice._write_lines = hangs

    async def scenario():
        call = SubActorsOnCall([voice])
        return [await call.agent_turn("Hello?"), await call.agent_turn("Anyone?")]

    assert asyncio.run(scenario()) == [False, False]
    assert voice.entering.turns_left == 0
