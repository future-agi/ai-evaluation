"""The tone a mailbox plays once its greeting is done, which is the part a session cannot speak."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("livekit")

from fi.simulate.simulation.engines import livekit


class _Player:
    def __init__(self) -> None:
        self.played: list = []

    def play(self, audio):
        self.played.append(audio)
        return object()


class _Session:
    """Only `history.items`, which is all `_session_messages` reads."""

    def __init__(self, items=()) -> None:
        self.history = type("History", (), {"items": list(items)})()

    def spoke(self, role: str, text: str) -> None:
        self.history.items.append(
            type(
                "Item",
                (),
                {"type": "message", "role": role, "text_content": text, "interrupted": False},
            )()
        )


class _Mailbox:
    """Only what `_play_voicemail_tone` reaches for."""

    def __init__(self, player=None) -> None:
        self._background_player = player


def _play(mailbox, session, style: str = "personal") -> None:
    asyncio.run(livekit._TestRunnerAgent._play_voicemail_tone(mailbox, style, session))


def _quick(monkeypatch) -> None:
    monkeypatch.setattr(livekit, "_VOICEMAIL_TONE_GAP_SECONDS", 0.0)
    monkeypatch.setattr(livekit, "_VOICEMAIL_TONE_WAIT_SECONDS", 0.5)


def test_the_tone_plays_after_the_mailbox_has_spoken(monkeypatch):
    _quick(monkeypatch)
    player = _Player()
    session = _Session()
    session.spoke("assistant", "You've reached Dana, leave a message after the tone.")

    _play(_Mailbox(player), session)

    assert len(player.played) == 1
    assert player.played[0].volume == livekit._VOICEMAIL_TONE_VOLUME


def test_a_mailbox_that_never_speaks_plays_nothing_and_gives_up(monkeypatch):
    """Bounded, so a silent mailbox cannot leave this pending for the length of the call."""
    _quick(monkeypatch)
    player = _Player()

    _play(_Mailbox(player), _Session())

    assert player.played == []


def test_the_agents_own_speech_is_not_mistaken_for_the_greeting(monkeypatch):
    """A tone owed to the mailbox must not fire because the target agent spoke first."""
    _quick(monkeypatch)
    player = _Player()
    session = _Session()
    session.spoke("user", "Hello? Am I speaking with Dana?")

    _play(_Mailbox(player), session)

    assert player.played == []


def test_a_full_mailbox_has_no_tone_at_all(monkeypatch):
    """It never invites a message, so a tone there would point the agent at nothing."""
    _quick(monkeypatch)
    player = _Player()
    session = _Session()
    session.spoke("assistant", "This mailbox is full and cannot accept new messages.")

    _play(_Mailbox(player), session, style="full")

    assert player.played == []


def test_each_style_has_its_own_tone_and_full_has_none():
    shapes = livekit._VOICEMAIL_TONE_BY_STYLE
    assert set(shapes) == {"personal", "carrier", "operator"}
    assert "full" not in shapes
    # The operator announcement is the long one, which is what a careless agent talks over.
    assert shapes["operator"][1] > shapes["personal"][1] > shapes["carrier"][1]


def test_a_missing_player_is_survived(monkeypatch):
    _quick(monkeypatch)
    session = _Session()
    session.spoke("assistant", "Leave a message.")

    _play(_Mailbox(None), session)


def test_the_generated_frame_is_the_length_and_rate_asked_for():
    frame = livekit._tone_frame(1000.0, 0.25)
    # 48000 specifically, not merely whatever the constant says. The mixer behind ``play`` reads
    # frames at its own rate without resampling them, so a tone built at any other rate reaches the
    # agent at the wrong pitch and a fraction of its length. Measured in a real room before this was
    # pinned: 1000Hz built at 24000 arrived as 2000Hz lasting half as long.
    assert frame.sample_rate == 48000
    assert livekit._BACKGROUND_MIXER_RATE == 48000
    assert frame.num_channels == 1
    assert frame.samples_per_channel == int(48000 * 0.25)
    # Faded at both ends, so it neither clicks in nor clicks out.
    import array

    samples = array.array("h")
    samples.frombytes(bytes(frame.data))
    assert abs(samples[0]) < 200
    assert abs(samples[-1]) < 200
    assert max(abs(one) for one in samples) > 20000


def test_the_style_is_read_from_the_environment_and_a_person_gets_no_tone(monkeypatch):
    monkeypatch.setenv("HARNESS_ANSWERED_BY", "voicemail")
    monkeypatch.setenv("HARNESS_VOICEMAIL_STYLE", "carrier")
    assert livekit._voicemail_tone_style() == "carrier"

    monkeypatch.setenv("HARNESS_VOICEMAIL_STYLE", "full")
    assert livekit._voicemail_tone_style() == ""

    monkeypatch.delenv("HARNESS_VOICEMAIL_STYLE")
    assert livekit._voicemail_tone_style() == "personal"

    monkeypatch.setenv("HARNESS_ANSWERED_BY", "person")
    assert livekit._voicemail_tone_style() == ""


def test_frames_from_a_slower_voice_are_resampled_for_the_mixer():
    import array

    from livekit import rtc

    # What Deepgram aura returns for the bystander line.
    body = array.array("h", [1000] * 2400).tobytes()
    source = rtc.AudioFrame(
        data=body, sample_rate=24000, num_channels=1, samples_per_channel=2400
    )
    resampled = livekit._frame_at_mixer_rate(source)
    assert resampled.sample_rate == 48000
    # Twice the samples for the same wall-clock length, which is what stops it playing double speed.
    assert 4700 < resampled.samples_per_channel < 4900


def test_a_frame_already_at_the_mixer_rate_is_left_alone():
    frame = livekit._tone_frame(1000.0, 0.1)
    assert livekit._frame_at_mixer_rate(frame) is frame
