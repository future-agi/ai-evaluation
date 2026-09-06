"""A second voice in the room is spoken by the caller's own TTS, mixed into the caller's side."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("livekit")

from livekit import rtc

from fi.simulate.simulation.engines import livekit


def _voice_frame(samples: int = 2400, rate: int = 24000) -> rtc.AudioFrame:
    """One frame at the rate Deepgram aura returns, which is not the rate the mixer reads."""
    import array

    return rtc.AudioFrame(
        data=array.array("h", [1000] * samples).tobytes(),
        sample_rate=rate,
        num_channels=1,
        samples_per_channel=samples,
    )


class _Player:
    def __init__(self) -> None:
        self.played: list = []

    def play(self, audio):
        self.played.append(audio)
        return object()


class _Tts:
    """Stands in for whichever TTS this call configured. `synthesize` yields frame-carrying events,
    which is the shape the background player accepts besides a file path."""

    def __init__(self, frames: list) -> None:
        self._frames = frames
        self.asked: list[str] = []

    def synthesize(self, text: str):
        self.asked.append(text)
        frames = self._frames

        class _Stream:
            def __aiter__(self):
                async def gen():
                    for one in frames:
                        yield type("Event", (), {"frame": one})()

                return gen()

        return _Stream()


class _Caller:
    """Only what `_say_bystander_line` reaches for, so no LiveKit session is needed."""

    def __init__(self, player=None, tts=None) -> None:
        self._background_player = player
        self.tts = tts


def _speak(caller, line: str = "Mum, are we nearly there") -> None:
    asyncio.run(livekit._TestRunnerAgent._say_bystander_line(caller, line))


def test_the_line_is_synthesised_and_mixed_in_under_the_caller(monkeypatch):
    monkeypatch.setattr(livekit, "_BYSTANDER_AFTER_SECONDS", 0.0)
    player, tts = _Player(), _Tts([_voice_frame(), _voice_frame()])
    _speak(_Caller(player, tts))

    assert tts.asked == ["Mum, are we nearly there"]
    assert len(player.played) == 1
    config = player.played[0]
    assert config.volume == livekit._BYSTANDER_VOLUME

    # The player is handed frames, not a file, which is why this needs no audio asset.
    async def drain():
        return [frame async for frame in config.source]

    played = asyncio.run(drain())
    assert len(played) == 2
    # At the rate the mixer reads rather than the rate the voice arrived at. The mixer reinterprets
    # whatever it is given at its own rate, so an unconverted frame speaks at double speed.
    assert [frame.sample_rate for frame in played] == [48000, 48000]
    assert all(frame.samples_per_channel > 4700 for frame in played)


def test_it_stays_quieter_than_the_caller_and_louder_than_the_room():
    """Speech reaches 15000 to 23000 of 32768 on real calls and the ambience clip measures near 770
    at volume 2.0, so a voice from the back seat sits between the two."""
    assert 0.0 < livekit._BYSTANDER_VOLUME < 1.0
    assert livekit._BYSTANDER_AFTER_SECONDS > livekit._OPEN_INSTEAD_AFTER_SECONDS


def test_nothing_to_speak_through_is_not_a_failed_call(monkeypatch):
    monkeypatch.setattr(livekit, "_BYSTANDER_AFTER_SECONDS", 0.0)
    _speak(_Caller(None, _Tts(["f1"])))
    _speak(_Caller(_Player(), None))


def test_a_tts_that_will_not_speak_is_not_a_failed_call(monkeypatch):
    monkeypatch.setattr(livekit, "_BYSTANDER_AFTER_SECONDS", 0.0)

    class _Broken:
        def synthesize(self, text: str):
            raise RuntimeError("no voice")

    _speak(_Caller(_Player(), _Broken()))


def test_a_voice_that_fails_midway_never_reaches_the_mixer(monkeypatch):
    """The player consumes the iterator from its mixer task, and that mixer also carries the
    ambience, so a half-synthesised line must fail here rather than there."""
    monkeypatch.setattr(livekit, "_BYSTANDER_AFTER_SECONDS", 0.0)

    class _FailsMidway:
        def synthesize(self, text: str):
            class _Stream:
                def __aiter__(self):
                    async def gen():
                        yield type("Event", (), {"frame": "f1"})()
                        raise RuntimeError("stream died")

                    return gen()

            return _Stream()

    player = _Player()
    _speak(_Caller(player, _FailsMidway()))
    assert player.played == []


def test_a_silent_voice_publishes_nothing(monkeypatch):
    monkeypatch.setattr(livekit, "_BYSTANDER_AFTER_SECONDS", 0.0)
    player = _Player()
    _speak(_Caller(player, _Tts([])))
    assert player.played == []


def test_cancelling_the_call_cancels_the_pending_line():
    """Teardown must not leave a task that speaks into a closed room."""
    started: list[str] = []

    async def scenario() -> None:
        caller = _Caller(_Player(), _Tts(["f1"]))
        caller._bystander_task = asyncio.create_task(
            livekit._TestRunnerAgent._say_bystander_line(caller, "later")
        )
        started.append("scheduled")
        await asyncio.sleep(0)
        await livekit._TestRunnerAgent._stop_background_audio(caller)
        assert caller._bystander_task is None

    asyncio.run(scenario())
    assert started == ["scheduled"]
