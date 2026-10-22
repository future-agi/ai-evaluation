"""Prepare persona-authored interjections off the live turn's critical path."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import re
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
from livekit import rtc
from livekit.agents import llm as livekit_llm

logger = logging.getLogger(__name__)

_MIN_TARGET_SPEECH_SECONDS = 0.4
_PREPARE_INSTRUCTIONS = (
    "This call is selected to test a realistic caller interruption. Write what "
    "THIS caller would naturally say while the agent is still explaining the "
    "next step. Use the caller persona's goal, constraints, and emotional state "
    "rather than a generic line. If hurried, impatient, frustrated, doubtful, "
    "or needing a correction, express that in ONE brief sentence (at most 8 "
    "words). A brief acknowledgement or filler can fit too; not every "
    "interruption needs a new request. Vary the kind and wording from this "
    "caller's earlier interruptions. Restating a need is okay when natural, "
    "but do not just repeat the caller's immediately previous request; add "
    "a genuine reaction, urgency, or correction if you restate it. A short "
    "backchannel or filled pause is fine when that suits the moment, but do "
    "not default to one on every turn. "
    "Do not guess the agent's next words or invent services, facts, names or "
    "policies. Never restate a credential, code, or PIN value. Do not claim a "
    "mismatch unless one is established. Do not talk about confirming details "
    "that have not been spoken yet. Unspoken goals may be introduced as new "
    "requests, not as existing facts. Avoid stock clarifications. "
    "You are the caller, not the agent: do not finish the agent's sentence, "
    "ask the agent's questions for them, or imitate their service response. "
    "Your line must stand alone as something this caller would say. "
    "Never output the private SILENCE hold marker. "
    "Output NONE only if interrupting would be completely out of character. "
    "Output only the sentence or NONE, without quotes or explanation."
)

_NUMBER_WORDS = frozenset(
    "zero one two three four five six seven eight nine oh cero uno dos tres cuatro "
    "cinco seis siete ocho nueve".split()
)


def _unsafe_candidate(
    phrase: str,
    previous_interjections: list[str],
    recent_caller_turns: list[str] | None = None,
) -> bool:
    """Reject credentials and near-duplicate interjections, not all restatements."""
    if re.match(r"^\s*SILENCE\b", phrase, re.I):
        return True
    if re.search(r"\d", phrase):
        return True
    words = re.findall(r"\b\w+\b", phrase.lower())
    consecutive_numbers = 0
    for word in words:
        consecutive_numbers = consecutive_numbers + 1 if word in _NUMBER_WORDS else 0
        if consecutive_numbers >= 3:
            return True
    for turn in previous_interjections:
        previous_words = re.findall(r"\b\w+\b", turn.lower())
        if words == previous_words:
            return True
        shared = set(words) & set(previous_words)
        union = set(words) | set(previous_words)
        if (
            len(words) >= 4
            and len(previous_words) >= 4
            and len(shared) >= 3
            and len(shared) / len(union) >= 0.55
        ):
            return True
    if len(words) >= 4:
        for turn in recent_caller_turns or []:
            prior = re.findall(r"\b\w+\b", turn.lower())
            if any(prior[index : index + len(words)] == words for index in range(len(prior))):
                return True
    return False


def selected_call_indices(run_id: str, count: int, rate: float) -> set[int]:
    """Select calls stably; hosted one-case runs use a Bernoulli draw.

    Hosted phone calls invoke the engine once per scenario.  Rounding 20% of
    one call to an integer would silently disable every hosted interruption.
    Multi-case local runs can instead select an exact share of the batch.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError("caller barge-in rate must be between 0 and 1")
    if count <= 0 or rate == 0:
        return set()
    seed = int.from_bytes(hashlib.sha256(run_id.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    if count < 5:
        return {index for index in range(count) if rng.random() < rate}
    return set(rng.sample(range(count), min(count, round(count * rate))))


async def complete_text(model: Any, context: Any) -> str:
    """The model's whole reply to this context, with no tools."""
    parts: list[str] = []
    async with model.chat(chat_ctx=context, tools=[], tool_choice="none") as stream:
        async for chunk in stream:
            content = getattr(getattr(chunk, "delta", None), "content", None)
            if isinstance(content, str):
                parts.append(content)
    return "".join(parts)


async def synthesize_frames(tts: Any, text: str) -> list[Any]:
    """The audio frames this voice speaks the text in."""
    async with tts.synthesize(text) as stream:
        return [item.frame async for item in stream]


def caller_language(persona: Any) -> str | None:
    data = getattr(persona, "persona", {}) or {}
    language = data.get("language") or data.get("languages")
    if not language:
        identity = getattr(persona, "identity", None)
        language = getattr(identity, "language", None)
    if isinstance(language, list):
        language = language[0] if language else None
    if not language:
        return "en"
    normalized = str(language).strip().lower()
    if normalized.startswith(("en", "english")):
        return "en"
    if normalized.startswith(("es", "spanish", "español")):
        return "es"
    return None  # Never inject English into a known non-English persona.


class _RoomInterjectionAudio:
    """Write interjections through the caller's existing RoomIO microphone track."""

    def __init__(self, output: Any):
        self.output = output
        self.sample_rate: int | None = None
        self.num_channels: int | None = None
        self._lock = asyncio.Lock()
        self._playing = False

    async def prepare(self, frame: rtc.AudioFrame) -> None:
        async with self._lock:
            if self.sample_rate is not None:
                if (
                    frame.sample_rate != self.sample_rate
                    or frame.num_channels != self.num_channels
                ):
                    raise ValueError("interjection TTS format changed within call")
                return
            expected = getattr(self.output, "sample_rate", None)
            if expected is not None and frame.sample_rate != expected:
                raise ValueError("interjection TTS sample rate differs from caller output")
            self.sample_rate = frame.sample_rate
            self.num_channels = frame.num_channels

    async def play(self, frames: list[rtc.AudioFrame], on_started: Any) -> None:
        if self.sample_rate is None:
            raise RuntimeError("interjection output not prepared")
        async with self._lock:
            self._playing = True
            try:
                for index, frame in enumerate(frames):
                    await self.output.capture_frame(frame)
                    if index == 0:
                        await on_started()
                self.output.flush()
                await self.output.wait_for_playout()
            except asyncio.CancelledError:
                self.output.clear_buffer()
                raise
            finally:
                self._playing = False

    def clear(self) -> None:
        # The output also carries the caller's own replies; only cut an interjection.
        if self._playing:
            self.output.clear_buffer()

    async def close(self) -> None:
        self.clear()


class CallerBargeIn:
    """One selected call; multiple target turns may receive an interjection."""

    def __init__(
        self,
        *,
        session: Any,
        agent: Any,
        model: Any,
        tts: Any,
        audio_output: Any,
        language: str | None,
        seed: str,
        delay: float = 0.0,
        cooldown: float = 10.0,
        propensity: float = 0.4,
    ):
        self.session = session
        self.agent = agent
        self.model = model
        self.tts = tts
        self.audio_output = audio_output
        self.language = language
        self.delay = delay
        self.cooldown = cooldown
        self.propensity = max(0.0, min(1.0, propensity))
        self._random = random.Random(
            int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")
        )
        self._pending: asyncio.Task[None] | None = None
        self._preparation: asyncio.Task[None] | None = None
        self._candidate: tuple[str, list[Any]] | None = None
        self._play_task: asyncio.Task[None] | None = None
        self._finishing_task: asyncio.Task[None] | None = None
        self._target_audio_task: asyncio.Task[None] | None = None
        self._target_audio_stream: rtc.AudioStream | None = None
        self._audio_gate_enabled = False
        self._await_target_audio = False
        self._audio_speech_elapsed = 0.0
        self._audio_voiced_elapsed = 0.0
        self._audio_silence_elapsed = 0.0
        self._target_active = False
        self._await_target_text = False
        self._gate_scheduled = False
        self._target_started_at = 0.0
        self._dialogue_started = False
        self._turn_index = 0
        self._last_attempt_at = 0.0
        self._closed = False
        self._audio_measured = False
        self._diagnostics: dict[str, int] = {}
        self.events: list[dict[str, Any]] = []

    def _count(self, reason: str) -> None:
        self._diagnostics[reason] = self._diagnostics.get(reason, 0) + 1

    def start(self) -> None:
        pass  # The first preparation starts after a real dialogue turn.

    @property
    def audio_gate_enabled(self) -> bool:
        return self._audio_gate_enabled

    def attach_target_audio(self, track: rtc.Track) -> bool:
        """Observe the real target waveform; never use VAD hangover as speech."""
        if self._closed or self._target_audio_task is not None:
            return False
        try:
            stream = rtc.AudioStream(track, sample_rate=8000, num_channels=1)
        except Exception as exc:
            logger.warning("target audio monitor unavailable: %s", type(exc).__name__)
            return False
        self._target_audio_stream = stream
        self._audio_gate_enabled = True
        self._target_audio_task = asyncio.create_task(
            self._monitor_target_audio(stream), name="caller-barge-in-target-audio"
        )
        return True

    async def _monitor_target_audio(self, stream: rtc.AudioStream) -> None:
        try:
            async for event in stream:
                self.target_audio_frame(event.frame)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._audio_gate_enabled = False
            self._count("target_audio_monitor_error")
            logger.warning("target audio monitor failed: %s", type(exc).__name__)
        finally:
            await stream.aclose()

    def target_audio_frame(self, frame: rtc.AudioFrame) -> None:
        if not self._target_active or not self._await_target_audio:
            return
        samples = np.frombuffer(frame.data, dtype="<i2").astype(np.float32)
        if not samples.size:
            return
        duration = frame.samples_per_channel / frame.sample_rate
        voiced = float(np.sqrt(np.mean(samples * samples))) >= 250.0
        if voiced:
            self._audio_silence_elapsed = 0.0
            self._audio_speech_elapsed += duration
            self._audio_voiced_elapsed += duration
        else:
            self._audio_silence_elapsed += duration
            if self._audio_silence_elapsed >= 0.35:
                self._audio_speech_elapsed = 0.0
                self._audio_voiced_elapsed = 0.0
            elif self._audio_speech_elapsed:
                self._audio_speech_elapsed += duration
        if (
            voiced
            and self._audio_speech_elapsed >= 4.0
            and self._audio_voiced_elapsed >= 2.2
            and not self._gate_scheduled
        ):
            self._gate_scheduled = True
            self._count("gate_sustained_audio")
            self._schedule_gate()

    def dialogue_started(self) -> None:
        """The first real target turn completed; notices/greetings are not candidates."""
        self._dialogue_started = True

    def prepare_next(self) -> None:
        """Generate a fresh optional interjection while the caller is otherwise idle."""
        if self._closed or not self._dialogue_started or self.language is None:
            return
        if self._target_active:
            return
        if self._candidate is not None or (
            self._preparation is not None and not self._preparation.done()
        ):
            return
        self._count("preparation_started")
        self._candidate = None
        self._preparation = asyncio.create_task(
            self._prepare(), name="caller-barge-in-prepare"
        )

    async def _prepare(self) -> None:
        try:
            context = livekit_llm.ChatContext.empty()
            context.add_message(role="system", content=str(self.agent.instructions))
            history = getattr(self.session, "history", None)
            previous_messages = [
                item
                for item in getattr(history, "items", [])
                if getattr(item, "type", None) == "message"
            ][-8:] or self.agent.chat_ctx.messages()[-8:]
            recent_caller_turns = [
                str(getattr(message, "text_content", ""))
                for message in previous_messages
                if getattr(message, "role", None) == "assistant"
            ][-2:]
            for message in previous_messages:
                context.items.append(message)
            if self.events:
                context.add_message(
                    role="user",
                    content=(
                        "Earlier interruptions in this call, which must not be repeated: "
                        + " | ".join(event["text"] for event in self.events[-3:])
                    ),
                )
            else:
                context.add_message(
                    role="user",
                    content=(
                        "For this caller's first interruption, prefer a useful "
                        "question, correction, or expression of their need over "
                        "a bare acknowledgement. Later interruptions can be "
                        "brief backchannels when natural."
                    ),
                )
            context.add_message(role="user", content=_PREPARE_INSTRUCTIONS)
            for attempt in range(3):
                phrase = (await complete_text(self.model, context)).strip().strip('"\u201c\u201d')
                if not phrase or phrase.upper() == "NONE":
                    self._count("preparation_declined")
                    return
                if len(phrase.split()) > 15 or len(phrase) > 120 or _unsafe_candidate(
                    phrase,
                    [event["text"] for event in self.events],
                    recent_caller_turns,
                ):
                    self._count("preparation_rejected_unsafe")
                    if attempt == 2:
                        return
                    context.add_message(role="assistant", content=phrase)
                    context.add_message(
                        role="user",
                        content=(
                            "That interruption is too similar to an earlier line "
                            "or otherwise unsuitable. Choose a genuinely different "
                            "reaction that fits this caller and the conversation, "
                            "or output NONE. Do not repeat the same point."
                        ),
                    )
                    continue
                break
            frames = await synthesize_frames(self.tts, phrase)
            if frames and not self._closed:
                await self.audio_output.prepare(frames[0])
                self._candidate = (phrase, frames)
                self._count("preparation_ready")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # Optional behavior must not fail the call.
            self._count("preparation_error")
            logger.warning("caller interjection preparation failed: %s", type(exc).__name__)

    def target_started(
        self, *, wait_for_text: bool = False, wait_for_audio: bool = False
    ) -> None:
        if self._closed or self._target_active:
            return
        self._target_active = True
        self._await_target_text = wait_for_text
        self._await_target_audio = wait_for_audio
        self._gate_scheduled = False
        self._audio_speech_elapsed = 0.0
        self._audio_voiced_elapsed = 0.0
        self._audio_silence_elapsed = 0.0
        self._target_started_at = time.monotonic()
        self._turn_index += 1
        self._count("target_started")
        if not wait_for_text and not wait_for_audio:
            self._schedule_gate()

    def target_text(self, partial: str) -> None:
        """Wait for an actual explanation, not a brief filler or tool preamble."""
        if not self._target_active or not self._await_target_text or self._gate_scheduled:
            return
        words = re.findall(r"\b\w+\b", partial.lower())
        if len(words) < 8:
            return
        lowered = partial.lower().lstrip()
        if len(words) < 13 and lowered.startswith(
            ("let me check", "one moment", "just a second", "hold on", "i understand")
        ):
            return
        self._gate_scheduled = True
        self._schedule_gate()

    def _schedule_gate(self) -> None:
        if not self._dialogue_started:
            self._count("gate_before_dialogue")
        elif self.language is None:
            self._count("gate_unknown_language")
        elif self._pending is not None:
            self._count("gate_pending")
        elif self._finishing_task is not None:
            self._count("gate_finishing")
        elif self._play_task is not None and not self._play_task.done():
            self._count("gate_playing")
        elif time.monotonic() - self._last_attempt_at < self.cooldown:
            self._count("gate_cooldown")
        else:
            self._count("gate_eligible")
            self._pending = asyncio.create_task(
                self._try_barge(), name="caller-barge-in-decision"
            )

    def target_ended(self) -> None:
        short_noninteractive_turn = (
            self._await_target_text or self._await_target_audio
        ) and not self._gate_scheduled
        self._target_active = False
        self._await_target_text = False
        self._await_target_audio = False
        if self._pending is not None:
            self._count("target_ended_with_pending")
            self._pending.cancel()
            self._pending = None
        if self._play_task is not None and not self._play_task.done():
            self._finishing_task = self._play_task
        # Keep a prepared line through a short filler ("One moment") because
        # the real explanation normally follows. Otherwise the context changed.
        if not short_noninteractive_turn:
            self._candidate = None

    @property
    def interjected_current_turn(self) -> bool:
        """Whether the caller has already spoken during the latest target turn.

        The overlap flag is diagnostic only: if playback began just after the
        target stopped, it is still a caller turn and must not be duplicated.
        """
        now = time.time()
        return any(
            event.get("target_turn") == self._turn_index
            and event.get("attempted_at") is not None
            and (
                event.get("stopped_at") is None
                or now - float(event["stopped_at"]) <= 1.25
            )
            for event in self.events
        )

    async def wait_for_interjection(self) -> None:
        """Finish the short interjection before handing the completed turn to LiveKit."""
        task = self._finishing_task
        if task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=4.0)
        except (asyncio.TimeoutError, RuntimeError):
            task.cancel()
            self.audio_output.clear()
        finally:
            if self._finishing_task is task:
                self._finishing_task = None

    async def _try_barge(self) -> None:
        try:
            await asyncio.sleep(
                max(
                    self.delay,
                    _MIN_TARGET_SPEECH_SECONDS
                    - (time.monotonic() - self._target_started_at),
                )
            )
            if (
                self._closed
                or not self._target_active
                or self.language is None
            ):
                self._count("target_ended_before_gate")
                return
            if self._preparation is not None and not self._preparation.done():
                try:
                    await asyncio.wait_for(asyncio.shield(self._preparation), timeout=3.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    self._count("preparation_not_ready")
                    return
            if not self._target_active or self._candidate is None:
                self._count("no_candidate")
                return
            # Never queue this behind the simulator's own ongoing sentence:
            # that would play after the target stopped and look like a barge-in
            # in the plan while sounding like an ordinary late reply.
            if getattr(self.session, "agent_state", "listening") == "speaking":
                self._count("caller_busy")
                return
            if getattr(self.session, "current_speech", None) is not None:
                self._count("caller_speech_pending")
                return
            # A selected call may interrupt more than once, but not every
            # explanation should be cut short.
            if self.events and self._random.random() > 0.2 + 0.85 * self.propensity:
                self._count("turn_probability_skipped")
                return
            phrase, frames = self._candidate
            self._candidate = None
            event = {
                "kind": "persona_interjection",
                "text": phrase,
                "target_turn": self._turn_index,
                "attempted_at": None,
                "target_active_at_schedule": True,
                "speech_state_overlap": False,
                "clip_duration_s": round(
                    sum(float(getattr(frame, "duration", 0.3)) for frame in frames), 3
                ),
            }
            self._play_task = asyncio.create_task(
                self._play(phrase, frames, event), name="caller-barge-in-audio"
            )
            self._last_attempt_at = time.monotonic()
            self._count("attempted")
            self.events.append(event)
        except asyncio.CancelledError:
            return
        finally:
            if self._pending is asyncio.current_task():
                self._pending = None

    async def _play(
        self, phrase: str, frames: list[Any], event: dict[str, Any]
    ) -> None:
        history_message: Any | None = None

        async def on_started() -> None:
            nonlocal history_message
            event["attempted_at"] = time.time()
            event["speech_state_overlap"] = self._target_active
            # Direct audio must also appear in the simulator's conversational
            # memory and the report, or the next model turn will forget it.
            try:
                history = getattr(self.session, "history", None)
                message = None
                if history is not None:
                    message = history.add_message(role="assistant", content=phrase)
                chat_ctx = self.agent.chat_ctx.copy()
                if message is not None:
                    if not any(item.id == message.id for item in chat_ctx.items):
                        chat_ctx.items.append(message)
                else:
                    message = chat_ctx.add_message(role="assistant", content=phrase)
                history_message = message
                if message is not None:
                    message.metrics = {
                        **(getattr(message, "metrics", None) or {}),
                        "started_speaking_at": event["attempted_at"],
                    }
                await self.agent.update_chat_ctx(chat_ctx)
            except Exception as exc:
                self._count("history_update_error")
                logger.warning(
                    "caller interjection history update failed: %s", type(exc).__name__
                )

        try:
            await self.audio_output.play(frames, on_started)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._count("playback_error")
            logger.warning("caller interjection playback failed: %s", type(exc).__name__)
        finally:
            if history_message is not None and event["attempted_at"] is not None:
                stopped_at = time.time()
                event["stopped_at"] = stopped_at
                history_message.metrics = {
                    **(getattr(history_message, "metrics", None) or {}),
                    "stopped_speaking_at": stopped_at,
                }

    def simulator_state_changed(self, state: str) -> None:
        if state in {"listening", "speaking"} and not self._target_active:
            self.prepare_next()

    async def close(self) -> None:
        self._closed = True
        self.target_ended()
        if self._target_audio_task is not None:
            self._target_audio_task.cancel()
            await asyncio.gather(self._target_audio_task, return_exceptions=True)
            self._target_audio_task = None
        if self._preparation is not None and not self._preparation.done():
            self._preparation.cancel()
            try:
                await asyncio.wait_for(self._preparation, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        if self._play_task is not None and not self._play_task.done():
            self._play_task.cancel()
            await asyncio.gather(self._play_task, return_exceptions=True)
        await self.audio_output.close()
        self._finishing_task = None

    def measure_audio(
        self, stereo_path: str | None, recording_started_at: float | None
    ) -> None:
        """Check actual dual-channel energy near each scheduled interjection."""
        if not stereo_path or recording_started_at is None:
            return
        try:
            with wave.open(str(Path(stereo_path)), "rb") as recording:
                if recording.getnchannels() != 2 or recording.getsampwidth() != 2:
                    return
                self._audio_measured = True
                rate = recording.getframerate()
                for event in self.events:
                    if event["attempted_at"] is None:
                        event["audio_overlap_ms"] = 0
                        continue
                    start = max(
                        0, round((event["attempted_at"] - recording_started_at) * rate)
                    )
                    if start >= recording.getnframes():
                        event["audio_overlap_ms"] = 0
                        continue
                    recording.setpos(start)
                    window_s = max(1.0, min(3.5, event["clip_duration_s"] + 0.8))
                    raw = recording.readframes(
                        min(round(window_s * rate), recording.getnframes() - start)
                    )
                    stereo = np.frombuffer(raw, dtype="<i2").reshape(-1, 2)
                    width = max(1, round(rate * 0.02))
                    stereo = stereo[: len(stereo) // width * width]
                    if not len(stereo):
                        event["audio_overlap_ms"] = 0
                        continue
                    energy = np.sqrt(
                        np.mean(
                            stereo.reshape(-1, width, 2).astype(np.float32) ** 2, axis=1
                        )
                    )
                    threshold = np.maximum(250.0, energy.max(axis=0) * 0.06)
                    active = (energy > threshold).all(axis=1)
                    event["audio_overlap_ms"] = round(
                        int(active.sum()) * width / rate * 1000
                    )
        except (OSError, ValueError, EOFError) as exc:
            logger.warning(
                "could not measure caller overlap from recording: %s",
                type(exc).__name__,
            )

    def summary(self) -> dict[str, Any]:
        return {
            "selected": True,
            "language": self.language,
            "audio_gate_enabled": self._audio_gate_enabled,
            "attempted": len(self.events),
            "speech_state_overlaps": sum(
                bool(event["speech_state_overlap"]) for event in self.events
            ),
            "audio_verified_overlaps": (
                sum(
                    bool(event.get("speech_state_overlap"))
                    and (event.get("audio_overlap_ms") or 0) >= 80
                    for event in self.events
                )
                if self._audio_measured
                and all("audio_overlap_ms" in event for event in self.events)
                else None
            ),
            "events": list(self.events),
            "diagnostics": dict(self._diagnostics),
        }
