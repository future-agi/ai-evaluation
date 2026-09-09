"""A mailbox speaks first and is never waited for."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("livekit")

from fi.simulate.simulation.engines import livekit


def test_the_watchdog_clears_a_slow_first_turn_without_being_generous():
    """It only sees a turn once the session commits it, and measured agent turn latency on a real
    run was 4292ms and 3947ms, so a bound near five seconds would talk over the greeting."""
    assert livekit._OPEN_INSTEAD_AFTER_SECONDS == 8.0
    assert (
        livekit._OPEN_INSTEAD_AFTER_SECONDS < livekit._NO_CONVERSATION_TIMEOUT_SECONDS
    )


def test_a_mailbox_is_recognised_from_the_calls_own_lane(monkeypatch):
    monkeypatch.delenv("HARNESS_ANSWERED_BY", raising=False)
    assert not livekit._answered_by_voicemail()

    monkeypatch.setenv("HARNESS_ANSWERED_BY", "voicemail")
    assert livekit._answered_by_voicemail()

    monkeypatch.setenv("HARNESS_ANSWERED_BY", " VoiceMail ")
    assert livekit._answered_by_voicemail()

    monkeypatch.setenv("HARNESS_ANSWERED_BY", "person")
    assert not livekit._answered_by_voicemail()


def test_the_person_still_opens_a_call_the_agent_never_starts():
    """The watchdog is what stops two voice agents waiting for each other, so it has to fire when
    nobody has spoken and stay quiet when somebody has."""
    opened: list[str] = []

    class Agent:
        def open_conversation(self) -> None:
            opened.append("opened")

    empty = type("S", (), {"history": type("H", (), {"items": []})()})()
    asyncio.run(
        livekit._open_if_nobody_speaks_first(empty, Agent(), timeout_seconds=0.05)
    )
    assert opened == ["opened"]

    spoken = type(
        "S",
        (),
        {
            "history": type(
                "H",
                (),
                {
                    "items": [
                        type(
                            "M",
                            (),
                            {
                                "type": "message",
                                "role": "assistant",
                                "text_content": "hello there",
                                "created_at": 1.0,
                                "interrupted": False,
                            },
                        )()
                    ]
                },
            )()
        },
    )()
    opened.clear()
    asyncio.run(
        livekit._open_if_nobody_speaks_first(spoken, Agent(), timeout_seconds=0.05)
    )
    assert opened == []


def test_a_mailbox_call_carries_no_ambience(monkeypatch):
    """Nothing stands behind a mailbox, so a scenario asking for a room must not get one: it would
    tell the agent a person is there when the point of the scenario is that none is."""
    started: dict = {}

    class _Player:
        def __init__(self, **kwargs) -> None:
            started["ambient"] = kwargs.get("ambient_sound")

        async def start(self, **kwargs) -> None:
            return None

        def play(self, audio):
            started.setdefault("played", []).append(audio)
            return object()

    monkeypatch.setattr(livekit, "BackgroundAudioPlayer", _Player)
    monkeypatch.setenv("HARNESS_ANSWERED_BY", "voicemail")
    monkeypatch.setenv("HARNESS_VOICEMAIL_STYLE", "personal")
    monkeypatch.setenv("HARNESS_BACKGROUND_NOISE", "CITY_AMBIENCE")
    monkeypatch.delenv("HARNESS_VOICEMAIL_CLIP", raising=False)

    agent = livekit._TestRunnerAgent.__new__(livekit._TestRunnerAgent)
    session = type("S", (), {"history": type("H", (), {"items": []})()})()
    asyncio.run(agent._maybe_start_background_audio(object(), session))

    assert started["ambient"] is None
    # The tone still plays: that is the one thing a mailbox does have.
    assert getattr(agent, "_voicemail_tone_task", None) is not None
    agent._voicemail_tone_task.cancel()


def test_a_mailbox_call_is_not_held_to_a_conversation_floor(monkeypatch):
    """A mailbox plays one greeting and records, so eight alternating messages are unreachable
    however well the agent behaves. Held to the conversation floor, every correct voicemail call was
    reported as an infrastructure failure, retried, and errored."""
    monkeypatch.setenv("HARNESS_ANSWERED_BY", "voicemail")
    assert livekit._turn_requirements(8) == (1, False)

    monkeypatch.setenv("HARNESS_ANSWERED_BY", "person")
    assert livekit._turn_requirements(8) == (8, True)


def test_a_two_turn_mailbox_call_completes_however_it_ended(monkeypatch):
    monkeypatch.setenv("HARNESS_ANSWERED_BY", "voicemail")
    messages = [
        {"role": "assistant", "content": "Hi. It's me. Leave a message."},
        {
            "role": "user",
            "content": "This is Uber, we could not reach you about your booking.",
        },
    ]
    for reason in (
        "conversation_silence_timeout",
        "room_disconnected",
        "target_disconnected",
    ):
        outcome = livekit._conversation_outcome(reason, messages, min_turn_messages=8)
        assert outcome.status == livekit.TestCaseStatus.COMPLETED, reason
        assert outcome.failure is None

    # A mailbox that never played is still a fault, and so is the same call answered by a person.
    empty = livekit._conversation_outcome(
        "no_conversation", messages, min_turn_messages=8
    )
    assert empty.failure.code == "no_conversation"
    monkeypatch.setenv("HARNESS_ANSWERED_BY", "person")
    person = livekit._conversation_outcome(
        "room_disconnected", messages, min_turn_messages=8
    )
    assert person.status == livekit.TestCaseStatus.FAILED


def test_a_recorded_greeting_is_spoken_as_the_mailboxs_own_turn(monkeypatch):
    """Mixed under the call it was audible and invisible: the transcript showed only the agent, so
    the call read as one where nobody answered and the evals judging the mailbox had nothing to
    read."""
    said: dict = {}

    class _Session:
        history = type("H", (), {"items": []})()

        def say(self, text, **kwargs):
            said["text"] = text
            said["audio"] = kwargs.get("audio")
            said["interruptible"] = kwargs.get("allow_interruptions")
            return object()

    class _Player:
        def __init__(self, **kwargs) -> None:
            pass

        async def start(self, **kwargs) -> None:
            return None

        def play(self, audio):
            said.setdefault("played", []).append(audio)
            return object()

    monkeypatch.setattr(livekit, "BackgroundAudioPlayer", _Player)
    monkeypatch.setenv("HARNESS_ANSWERED_BY", "voicemail")
    monkeypatch.setenv("HARNESS_VOICEMAIL_STYLE", "carrier")
    monkeypatch.setenv("HARNESS_VOICEMAIL_CLIP", "/tmp/greeting.wav")
    monkeypatch.setenv(
        "HARNESS_VOICEMAIL_CLIP_TRANSCRIPT", "No one is available to take your call."
    )

    agent = livekit._TestRunnerAgent.__new__(livekit._TestRunnerAgent)
    asyncio.run(agent._maybe_start_background_audio(object(), _Session()))

    assert said["text"] == "No one is available to take your call."
    assert said["audio"] is not None
    # Nothing the agent says cuts a recording short; a mailbox is not listening.
    assert said["interruptible"] is False
    agent._voicemail_tone_task.cancel()


def test_the_mailbox_cuts_the_line_once_it_stops_recording(monkeypatch):
    """A mailbox is not a party to the call: it does not ask whether anybody is there and it does not
    wait. Nothing was ending these calls, so one ran another minute and a half
    after the agent had already left its message."""
    monkeypatch.setattr(livekit, "_VOICEMAIL_RECORD_SECONDS", 0.01)
    agent = livekit._TestRunnerAgent.__new__(livekit._TestRunnerAgent)
    agent._end_requested = asyncio.Event()
    agent._voicemail_greeting = None
    agent._voicemail_tone_task = None

    asyncio.run(agent._close_mailbox_after_recording())
    assert agent._end_requested.is_set()


def test_the_recording_window_starts_after_the_greeting_and_the_tone(monkeypatch):
    monkeypatch.setattr(livekit, "_VOICEMAIL_RECORD_SECONDS", 0.01)
    order: list[str] = []

    async def tone() -> None:
        await asyncio.sleep(0.02)
        order.append("tone")

    class _Greeting:
        def __await__(self):
            async def wait():
                order.append("greeting")
                return None

            return wait().__await__()

    async def drive() -> None:
        agent = livekit._TestRunnerAgent.__new__(livekit._TestRunnerAgent)
        agent._end_requested = asyncio.Event()
        agent._voicemail_greeting = _Greeting()
        agent._voicemail_tone_task = asyncio.create_task(tone())
        await agent._close_mailbox_after_recording()
        order.append("cut")
        assert agent._end_requested.is_set()

    asyncio.run(drive())
    assert order == ["greeting", "tone", "cut"]


def test_a_recorded_greeting_opens_the_call_by_itself(monkeypatch):
    """A clip greeting followed by the persona's own line greets twice, the second time in
    another voice naming somebody else, so one mailbox answered as two people."""
    said: list = []

    class _Session:
        def say(self, text, **kwargs):
            said.append(text)
            return object()

        def generate_reply(self):
            said.append("<generated>")

    agent = livekit._TestRunnerAgent.__new__(livekit._TestRunnerAgent)
    agent._session = _Session()
    agent._persona = type(
        "P", (), {"persona": {"initial_message": "Hi, this is Liam."}}
    )()

    # No recording: the persona's own greeting opens the call, as it always did.
    agent._voicemail_greeting = None
    agent.open_conversation()
    assert said == ["Hi, this is Liam."]

    # With one, the clip has already greeted and nothing further is said.
    agent._voicemail_greeting = object()
    agent.open_conversation()
    assert said == ["Hi, this is Liam."]


def test_a_clip_without_words_is_heard_but_never_invents_a_turn(monkeypatch):
    """Inventing a line would put words in the transcript that the audio never says, and the evals
    that judge how a mailbox was handled would then judge those words."""
    played: list = []
    said: list = []

    class _Session:
        history = type("H", (), {"items": []})()

        def say(self, text, **kwargs):
            said.append(text)
            return object()

    class _Player:
        def __init__(self, **kwargs) -> None:
            pass

        async def start(self, **kwargs) -> None:
            return None

        def play(self, audio):
            played.append(audio)
            return object()

    monkeypatch.setattr(livekit, "BackgroundAudioPlayer", _Player)
    monkeypatch.setenv("HARNESS_ANSWERED_BY", "voicemail")
    monkeypatch.setenv("HARNESS_VOICEMAIL_STYLE", "carrier")
    monkeypatch.setenv("HARNESS_VOICEMAIL_CLIP", "/tmp/greeting.wav")
    monkeypatch.delenv("HARNESS_VOICEMAIL_CLIP_TRANSCRIPT", raising=False)

    agent = livekit._TestRunnerAgent.__new__(livekit._TestRunnerAgent)
    asyncio.run(agent._maybe_start_background_audio(object(), _Session()))

    assert said == []
    assert len(played) == 1
    # Nothing was committed, so the persona's own greeting still opens the call.
    assert agent._voicemail_greeting is None
    agent._voicemail_tone_task.cancel()
    agent._mailbox_close_task.cancel()


def test_a_mailbox_speaks_once_and_never_answers_the_agent(monkeypatch):
    """Asking a model for silence does not get silence: the mailbox
    greeted, the agent replied, and the mailbox said "Alright, thank you, bye." One turn is allowed
    because a mailbox with no recording greets through this path; a second never is."""
    reached: list[str] = []

    async def _base_llm_node(self, chat_ctx, tools, model_settings):
        reached.append("model")
        yield "a reply the model wanted to give"

    monkeypatch.setattr(livekit.Agent, "llm_node", _base_llm_node, raising=False)
    agent = livekit._TestRunnerAgent.__new__(livekit._TestRunnerAgent)

    async def drain():
        return [chunk async for chunk in agent.llm_node(None, [], None)]

    monkeypatch.setenv("HARNESS_ANSWERED_BY", "voicemail")
    # The greeting, which a mailbox with no recording has to be able to speak.
    assert asyncio.run(drain()) == ["a reply the model wanted to give"]
    # Everything after it, which is where the sign-off came from.
    assert asyncio.run(drain()) == []
    assert asyncio.run(drain()) == []
    assert reached == ["model"]

    # A recording has already greeted, so not even the first turn is allowed.
    recorded = livekit._TestRunnerAgent.__new__(livekit._TestRunnerAgent)
    recorded._voicemail_greeting = object()

    async def drain_recorded():
        return [chunk async for chunk in recorded.llm_node(None, [], None)]

    reached.clear()
    assert asyncio.run(drain_recorded()) == []
    assert reached == []

    # A person still gets the model on every turn.
    monkeypatch.setenv("HARNESS_ANSWERED_BY", "person")
    person = livekit._TestRunnerAgent.__new__(livekit._TestRunnerAgent)

    async def drain_person():
        return [chunk async for chunk in person.llm_node(None, [], None)]

    assert asyncio.run(drain_person()) == ["a reply the model wanted to give"]
    assert asyncio.run(drain_person()) == ["a reply the model wanted to give"]


def test_a_full_mailbox_still_gets_its_recording_timer(monkeypatch):
    """FULL is the one style with no tone, so the early return that skips a call wanting no ambience
    used to skip the timer with it: a measured full-mailbox call ran 90 seconds with the agent asking
    "Is anyone there?" until the watchdog ended it."""

    class _Player:
        def __init__(self, **kwargs) -> None:
            pass

        async def start(self, **kwargs) -> None:
            return None

        def play(self, audio):
            return object()

    monkeypatch.setattr(livekit, "BackgroundAudioPlayer", _Player)
    monkeypatch.setenv("HARNESS_ANSWERED_BY", "voicemail")
    monkeypatch.setenv("HARNESS_VOICEMAIL_STYLE", "full")
    monkeypatch.delenv("HARNESS_BACKGROUND_NOISE", raising=False)
    monkeypatch.delenv("HARNESS_VOICEMAIL_CLIP", raising=False)

    # A full mailbox has no tone, which is the whole point of the style.
    assert livekit._voicemail_tone_style() == ""

    agent = livekit._TestRunnerAgent.__new__(livekit._TestRunnerAgent)
    session = type("S", (), {"history": type("H", (), {"items": []})()})()
    asyncio.run(agent._maybe_start_background_audio(object(), session))

    assert agent._voicemail_tone_task is None
    assert agent._mailbox_close_task is not None
    agent._mailbox_close_task.cancel()


def test_the_mailbox_timer_is_cancelled_with_the_call(monkeypatch):
    """Every task this method starts has to be cancelled on the way out, or a suite leaks one per
    scenario, which is what the comment on `_stop_background_audio` exists to prevent."""

    async def forever() -> None:
        await asyncio.sleep(3600)

    async def drive() -> None:
        agent = livekit._TestRunnerAgent.__new__(livekit._TestRunnerAgent)
        agent._mailbox_close_task = asyncio.create_task(forever())
        agent._background_noise_file = None
        await agent._stop_background_audio()
        assert agent._mailbox_close_task is None

    asyncio.run(drive())


def _said(role: str, content: str = "something") -> dict[str, str]:
    return {"role": role, "content": content}


def test_caller_may_hang_up_once_the_target_stops_replying() -> None:
    """The floor counts messages, and the caller's own filler counts toward it.

    A caller refused the tool talks to fill the silence and eventually buys its own permission,
    which is the opposite of what the floor is for. Measured on a real call: the agent finished,
    the caller said goodbye five times over thirty-seven seconds.
    """
    # One unanswered turn is the caller finishing a thought; the agent may still be about to reply.
    one_unanswered_turn = [
        _said("assistant"),
        _said("user"),
        _said("assistant"),
        _said("user"),
    ]
    assert livekit._target_has_gone_quiet(one_unanswered_turn) is False

    # Two in a row means nobody is replying.
    agent_gone_quiet = one_unanswered_turn + [_said("user")]
    assert livekit._target_has_gone_quiet(agent_gone_quiet) is True

    # And replying resets it.
    assert (
        livekit._target_has_gone_quiet(agent_gone_quiet + [_said("assistant")]) is False
    )


def test_a_caller_that_has_never_heard_the_agent_may_not_hang_up() -> None:
    """A call where only the caller ever spoke is a failed call, not a finished one."""
    assert livekit._target_has_gone_quiet([_said("user"), _said("user")]) is False
    assert livekit._target_has_gone_quiet([]) is False


def test_an_outbound_agent_makes_the_caller_the_one_receiving(monkeypatch) -> None:
    """call_type is the AGENT's direction and it picks which half of the role block is written.

    Hardcoded to inbound, an outbound run told the caller "You are MAKING this call" and then
    contradicted itself further down with the harness's own instructions.
    """
    from fi.simulate.simulation.voice_prompt import build_voice_simulator_prompt
    from fi.simulate.simulation.models import Persona

    persona = Persona(
        persona={"name": "Marcus"}, situation="You are expecting a call.", outcome=""
    )
    receiving = build_voice_simulator_prompt(persona, call_type="outbound")
    assert "You are RECEIVING this call" in receiving
    assert "You are MAKING this call" not in receiving

    placing = build_voice_simulator_prompt(persona, call_type="inbound")
    assert "You are MAKING this call" in placing


def test_the_closing_rule_supplies_no_words_to_say() -> None:
    """Whatever the prompt quotes is spoken back verbatim, and in English."""
    from fi.simulate.simulation.voice_prompt import build_voice_simulator_prompt
    from fi.simulate.simulation.models import Persona

    text = build_voice_simulator_prompt(
        Persona(persona={"name": "Marcus"}, situation="You want a refund.", outcome=""),
        call_type="outbound",
    )
    assert "Alright, thanks, bye" not in text
    assert "endCall" in text


def test_a_finished_conversation_settles_faster_than_a_stalled_one() -> None:
    """Three rounds of prompt work failed to make the caller hang up reliably, so the engine
    stops waiting a full minute for a call that is plainly over. Measured across 21 calls on six
    agents, every call the caller had to end sat at 36 to 43 seconds of dead air."""
    window = livekit._settled_silence_window(0.0, livekit._SILENCE_BACKSTOP_SECONDS)
    assert window < livekit._SILENCE_BACKSTOP_SECONDS
    assert window > 8.0


def test_the_settle_window_outlasts_the_agent_it_is_waiting_on() -> None:
    """A fixed window is what made this wrong the first time: it was set from one agent's 4.3s
    worst case, while another agent was measured at 25.2s. Anything shorter than the agent's own
    reply time cuts it off mid-answer and the transcript reads as the agent failing."""
    backstop = livekit._SILENCE_BACKSTOP_SECONDS

    fast = livekit._settled_silence_window(3.0, backstop)
    slow = livekit._settled_silence_window(25.2, backstop)

    assert fast == livekit._SETTLED_SILENCE_FLOOR_SECONDS
    assert slow > 25.2, "the window must clear the slowest reply this call actually saw"
    assert slow > fast, "a slower agent has to be given longer, not the same"
    # An early settle can never wait longer than the real backstop, or it is not an early settle.
    assert livekit._settled_silence_window(10_000.0, backstop) == backstop


def test_the_settle_window_never_undercuts_the_floor() -> None:
    """An unmeasured call (agent never answered) settles at the floor, not at zero."""
    assert (
        livekit._settled_silence_window(0.0, livekit._SILENCE_BACKSTOP_SECONDS)
        == livekit._SETTLED_SILENCE_FLOOR_SECONDS
    )


def test_silence_backstop_is_not_a_normal_turn_gap() -> None:
    """Silence is a failure backstop, not evidence of successful completion."""
    assert livekit._SILENCE_BACKSTOP_SECONDS >= 90.0
