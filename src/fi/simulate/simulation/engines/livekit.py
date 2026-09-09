from __future__ import annotations

import array
import asyncio
import json
import math
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any, AsyncIterable
from uuid import uuid4

try:
    from livekit import api, rtc
    from livekit.agents import (
        Agent,
        AgentSession,
        AudioConfig,
        BackgroundAudioPlayer,
        RunContext,
        function_tool,
        metrics,
    )
    from livekit.agents.utils.audio import audio_frames_from_file
    from livekit.agents.voice.background_audio import BuiltinAudioClip
    from livekit.agents.types import (
        ATTRIBUTE_TRANSCRIPTION_TRACK_ID,
        TOPIC_TRANSCRIPTION,
    )
    from livekit.agents.voice import ModelSettings
    from livekit.agents.voice.io import TimedString
    from livekit.agents.voice.room_io import AudioInputOptions, RoomOptions
    from livekit.api import AccessToken, VideoGrants
    from livekit.plugins import silero
    from livekit.protocol.sip import (
        CreateSIPDispatchRuleRequest,
        DeleteSIPDispatchRuleRequest,
        ListSIPDispatchRuleRequest,
        SIPDispatchRule,
        SIPDispatchRuleDirect,
    )
except ImportError as exc:
    raise ImportError(
        "LiveKit mode requires the 'livekit' optional dependency"
    ) from exc

from datetime import datetime, timezone

from fi.simulate._logging import redacted_exc_info
from fi.simulate.agent.definition import (
    AgentDefinition,
    LiveKitSimulatorRuntime,
    LLMConfig,
    ProviderEvidenceConfig,
    RetellTargetConfig,
    SimulatorAgentDefinition,
    STTConfig,
    TelephonyTransport,
    TTSConfig,
    VapiTargetConfig,
    VoiceProviderTarget,
)
from fi.simulate.artifacts.manifest import ArtifactManifestEntry
from fi.simulate.evidence.base import EvidenceSourceSummary
from fi.simulate.evidence.providers import (
    EvidenceContext,
    ProviderConfigError,
    ProviderFetchResult,
    RetellEvidenceSource,
    VapiEvidenceSource,
)
from fi.simulate.endpoints.originators import (
    CallOriginator,
    build_call_originator,
    finalize_originator,
)
from fi.simulate.simulation.bridge import LiveKitAudioBridge
from fi.simulate.simulation.bridge.audio import PCMResampler
from fi.simulate.simulation.livekit_models import LiveKitModels, build_livekit_models
from fi.simulate.recording.room_recorder import (
    RoomRecorder,
    mix_recordings,
    mix_recordings_stereo,
)
from fi.simulate.runtime import (
    FailureStage,
    SimulationFailure,
    TestCaseStatus,
    derive_test_case_id,
    new_run_id,
)
from fi.simulate.simulation.engines.base import BaseEngine
from fi.simulate.simulation.generator import ScenarioGenerator
from fi.simulate.simulation.models import Persona, Scenario, TestCaseResult, TestReport
from fi.simulate.simulation.voice_prompt import CallType, build_voice_simulator_prompt

logger = logging.getLogger(__name__)
_SAFE_ROOM = re.compile(r"[^A-Za-z0-9_.-]+")
# On conversation end, wait up to this long for the party still finishing its
# own turn to commit it (a LiveKit turn lands in history only after its TTS
# finishes playing), then delete the room so neither side keeps talking into a
# call the other has already left.
_FINAL_TURN_COMMIT_WAIT_SECONDS = 30.0
# The hosted platform inflates ``cleanup_timeout`` to carry the whole run
# budget (observed 1470s); as a per-step cleanup bound it must stay capped.
_MAX_CLEANUP_TIMEOUT_SECONDS = 60.0
# A dead LiveKit signal connection can leave any one SDK cleanup await pending
# indefinitely. The case-level deadline is still the outer bound, but no
# single best-effort operation may consume it all and starve every cleanup that
# follows. Session close gets longer because it drains several SDK activities.
_CLEANUP_STEP_TIMEOUT_SECONDS = 8.0
_SESSION_CLEANUP_TIMEOUT_SECONDS = 15.0
_BACKGROUND_AUDIO_CLEANUP_TIMEOUT_SECONDS = 5.0
_NO_CONVERSATION_TIMEOUT_SECONDS = 120.0
# How long the side that was meant to speak first is given before the simulated person speaks
# instead. Both sides are voice agents waiting to be addressed, so when the one that placed the call
# says nothing the call is silence until a deadline discards it, and nothing was learned about
# either side. Kept well under the timeout above, which is what abandons a call nobody started.
#
# Eight seconds: the smallest bound that cannot pre-empt a slow first turn.
_OPEN_INSTEAD_AFTER_SECONDS = 8.0
# Frequency and length per kind of mailbox. FULL has no entry: it invites no message.
_VOICEMAIL_TONE_BY_STYLE: dict[str, tuple[float, float]] = {
    "personal": (1000.0, 0.40),
    "carrier": (1400.0, 0.33),
    "operator": (440.0, 0.52),
}
_DEFAULT_VOICEMAIL_STYLE = "personal"
# Loud enough to be unmistakable against speech, which reaches 15000 to 23000 of 32768.
_VOICEMAIL_TONE_VOLUME = 0.8
# A mailbox plays one greeting and then records, so the eight-message conversation floor is
# unreachable however well the agent behaves, and holding it there errored every voicemail call.
_VOICEMAIL_MIN_TURN_MESSAGES = 1
# How long a mailbox records before cutting the line, from the end of the tone or greeting. Without a
# bound the call runs to the silence watchdog with the agent still talking into a machine.
_VOICEMAIL_RECORD_SECONDS = 40.0
# Resamplers into the mixer's rate, one per source rate, kept because ``ratecv`` is stateful.
_MIXER_RESAMPLERS: dict[tuple[int, int], PCMResampler] = {}
# How long after the mailbox stops speaking the tone comes. A real system leaves a beat.
_VOICEMAIL_TONE_GAP_SECONDS = 0.7
# How long to wait for the mailbox to say anything before giving up on the tone. Bounded so a
# mailbox that never speaks cannot leave this task pending for the length of the call.
_VOICEMAIL_TONE_WAIT_SECONDS = 40.0
# The mixer reinterprets frames at this rate rather than resampling them, so anything published
# through it must be produced here or it plays at the wrong pitch and length.
_BACKGROUND_MIXER_RATE = 48000
# Each web case drives a full voice pipeline (STT/LLM/TTS + LiveKit conns) in one
# child; too many starve the pod's CPU. This is an OPS CEILING on the
# config-driven ``max_parallel_cases`` (not a replacement for it) — tune
# ``ALK_VOICE_MAX_CASE_CONCURRENCY`` to the pod's cores. Caps web cases only.
_VOICE_MAX_CASE_CONCURRENCY_DEFAULT = 4


def _simulator_participant_identity(persona: Persona, test_case_id: str) -> str:
    """Give repository agents the scenario caller ANI through a standard identity seam.

    LiveKit token metadata is not exposed consistently across every SDK/agent version. The
    harness therefore uses the identity convention already understood by repository voice
    agents: ``fagi-simulator-phone-<digits>-...``. A persona without a fixture-derived phone
    keeps the legacy anonymous identity.
    """
    definition = persona.persona if isinstance(persona.persona, dict) else {}
    metadata = definition.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    digits = re.sub(r"\D", "", str(metadata.get("caller_phone") or ""))
    suffix = test_case_id[-12:]
    if 7 <= len(digits) <= 15:
        return f"fagi-simulator-phone-{digits}-{suffix}"
    return f"fagi-simulator-{suffix}"


def _voice_max_case_concurrency() -> int:
    raw = os.environ.get("ALK_VOICE_MAX_CASE_CONCURRENCY", "").strip()
    if not raw:
        return _VOICE_MAX_CASE_CONCURRENCY_DEFAULT
    try:
        value = int(raw)
    except ValueError:
        return _VOICE_MAX_CASE_CONCURRENCY_DEFAULT
    return value if value >= 1 else _VOICE_MAX_CASE_CONCURRENCY_DEFAULT


_silero_vad: Any | None = None
_silero_vad_guard = threading.Lock()


def _load_silero_vad_sync() -> Any:
    """One shared VAD per process; per-case ``VAD.load()`` ran a synchronous
    model load on the event loop for every concurrent case."""
    global _silero_vad
    with _silero_vad_guard:
        if _silero_vad is None:
            _silero_vad = silero.VAD.load()
    return _silero_vad


@dataclass(frozen=True)
class _TargetParticipant:
    identity: str
    sid: str
    audio_track_sid: str
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class _CaseOutcome:
    status: TestCaseStatus
    transcript: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    failure: SimulationFailure | None = None
    audio_input_path: str | None = None
    audio_output_path: str | None = None
    audio_combined_path: str | None = None
    audio_stereo_path: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    evidence: list[EvidenceSourceSummary] = field(default_factory=list)
    provider_artifacts: list[ArtifactManifestEntry] = field(default_factory=list)


def _dispatch_metadata_json(agent_definition) -> str:
    """Metadata for the target agent's LiveKit dispatch.

    EMPTY by default: a target agent built from a LiveKit template branches on
    ``ctx.job.metadata`` and treats any non-empty payload as an outbound/no-greet
    job, so it never publishes an audio track and readiness times out
    (``agent_unavailable``). Only a target explicitly built to consume dispatch
    metadata sets ``agent_definition.dispatch_metadata``.
    """
    meta = getattr(agent_definition, "dispatch_metadata", None)
    return json.dumps(meta, sort_keys=True) if meta else ""


def _resolve_target_profile(kind: str):
    """Look up the target adapter's profile — the factory that replaced the
    engine's ``transport.kind`` branching. Unknown kinds fail loudly, which is
    what makes it safe to open ``TelephonyTransport.kind`` from a Literal to a
    free string later."""
    from fi.simulate.endpoints.profiles import get_profile

    profile = get_profile(kind)
    if profile is None:
        raise ValueError(f"unsupported_transport_kind: {kind}")
    return profile


def _simulator_turn_handling(
    *,
    vad: object | None,
    allow_interruptions: bool | None = None,
    min_endpointing_delay: float | None = None,
    max_endpointing_delay: float | None = None,
) -> dict[str, object]:
    return {
        "turn_detection": "vad" if vad is not None else "stt",
        # A short delay fires inside a sentence, on a comma or a breath, so the caller treats a pause
        # as the end of the turn, talks over the agent and then repeats itself for want of an answer.
        "endpointing": {
            "mode": "fixed",
            "min_delay": min_endpointing_delay or 0.9,
            "max_delay": max_endpointing_delay or 3.0,
        },
        # A real caller interrupts, but only over something long enough to be worth interrupting.
        "interruption": {
            "enabled": (True if allow_interruptions is None else allow_interruptions),
            "discard_audio_if_uninterruptible": True,
            "min_duration": 0.6,
        },
        "preemptive_generation": {"enabled": True},
    }


class _TestRunnerAgent(Agent):
    def __init__(
        self,
        persona: Persona,
        *,
        min_turn_messages: int = 0,
        **kwargs,
    ):
        turn_handling = kwargs.setdefault(
            "turn_handling",
            _simulator_turn_handling(vad=kwargs.get("vad")),
        )
        super().__init__(**kwargs)
        self._persona = persona
        self._min_turn_messages = min_turn_messages
        self._session_turn_handling = turn_handling
        self._session: AgentSession | None = None
        self._end_requested = asyncio.Event()
        self._end_speech_handle: Any | None = None
        self._usage_collector = metrics.ModelUsageCollector()

    @function_tool(
        name="endCall",
        # Nothing quotable and nothing English-specific: wording here comes back out as speech.
        description=(
            "Ends the call. Nothing else ends it and no one else ends it for you. "
            "Use it once you have nothing further."
        ),
    )
    async def end_call(self, ctx: RunContext) -> str:
        if self._session is None:
            logger.warning("endCall refused: no session yet")
            return "Continue the conversation before ending the call."
        messages = _session_messages(self._session)
        floor, alternation_required = _turn_requirements(self._min_turn_messages)
        below_floor = len(messages) < floor or (
            alternation_required and not _has_role_alternation(messages)
        )
        if below_floor and _target_has_gone_quiet(messages):
            below_floor = False
        if below_floor:
            # Whether the caller ever reached for this tool, and why it was turned away, is the
            # difference between a simulator that will not hang up and one that was not allowed to.
            logger.warning(
                "endCall refused: %d messages, floor %d, alternating=%s",
                len(messages),
                floor,
                _has_role_alternation(messages),
            )
            # "Not yet" rather than "stop asking", or the caller never retries the tool.
            return (
                f"Not yet: {len(messages)} of {floor} messages so far and both speakers must "
                "have spoken. Keep the conversation going, then call endCall again."
            )
        logger.warning("endCall accepted after %d messages", len(messages))
        # The tool runs inside the same SpeechHandle that carries the model's
        # natural closing sentence. Remember that exact handle before waking
        # the outer runner so it cannot snapshot history in the brief interval
        # before TTS starts and ``session.current_speech`` becomes non-None.
        self._end_speech_handle = ctx.speech_handle
        self._end_requested.set()
        return "Conversation ended."

    async def wait_for_end_speech(self) -> None:
        if self._end_speech_handle is not None:
            await self._end_speech_handle

    @property
    def started_session(self) -> AgentSession | None:
        return self._session

    @property
    def end_requested(self) -> asyncio.Event:
        return self._end_requested

    @property
    def model_usage(self) -> list[dict[str, object]]:
        return [
            usage.model_dump(mode="json")
            for usage in sorted(
                self._usage_collector.flatten(),
                key=lambda usage: (usage.type, usage.provider, usage.model),
            )
        ]

    async def start_session(
        self,
        room: rtc.Room,
        *,
        participant_kinds: list | None = None,
        participant_identity: str | None = None,
    ) -> AgentSession:
        session = AgentSession(
            stt=self.stt,
            llm=self.llm,
            tts=self.tts,
            vad=self.vad,
            turn_handling=self._session_turn_handling,
        )
        self._session = session
        session.on(
            "metrics_collected",
            lambda event: self._usage_collector.collect(event.metrics),
        )
        default_kinds = [
            rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
            getattr(
                rtc.ParticipantKind,
                "PARTICIPANT_KIND_AGENT",
                rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
            ),
            rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
        ]
        room_kwargs: dict = {
            "audio_input": AudioInputOptions(
                pre_connect_audio=False,
                pre_connect_audio_timeout=3.0,
            ),
            # Enabled to build RoomIO's TranscriptSynchronizer, which aligns the
            # spoken transcript to audio playback. On an interruption the
            # recorded turn is then truncated to what was actually said instead
            # of the full LLM text (playback-timing estimate — works with any
            # TTS, unlike use_tts_aligned_transcript which needs word timing our
            # Deepgram/Gemini voices don't emit and would drop the turn). The
            # simulator's transcription is published to the room as a harmless
            # side effect (our target-transcription handler filters by identity).
            "text_output": True,
            "close_on_disconnect": False,
            "delete_room_on_close": False,
            "participant_kinds": participant_kinds or default_kinds,
        }
        if participant_identity:
            room_kwargs["participant_identity"] = participant_identity
        await session.start(
            self,
            room=room,
            room_options=RoomOptions(**room_kwargs),
        )
        await self._maybe_start_background_audio(room, session)
        return session

    async def _maybe_start_background_audio(
        self, room: "rtc.Room", session: "AgentSession"
    ) -> None:
        """Mix caller-side ambient noise under the simulated caller, if the run asked for it.

        Off unless HARNESS_BACKGROUND_NOISE names a source: a LiveKit builtin clip name, or an
        http(s) URL to an ambient file. Any failure is swallowed, because a call without ambience is
        preferable to a dropped one.
        """
        source = os.environ.get("HARNESS_BACKGROUND_NOISE", "").strip()
        # A mailbox needs this method for its tone and its recording timer, and FULL has no tone.
        tone_style = _voicemail_tone_style()
        if _answered_by_voicemail() and source:
            # Nothing stands behind a recording, and a room behind one gives the game away.
            logger.info("mailbox answered, so ambience is dropped (noise %r)", source)
            source = ""
        if not source and not tone_style and not _answered_by_voicemail():
            return

        try:
            # 2.0, not the 0.3 this used to default to. Measured in an isolated two-participant
            # room, the office clip peaks at 119 of 32768 at 0.3, which is below the noise floor of
            # speech near 15000: the ambience played and nobody could hear it. At 2.0 the same clip
            # measures 752 to 789 on real calls, which is audible under a voice without masking it.
            volume = float(os.environ.get("HARNESS_BACKGROUND_NOISE_VOLUME", "2.0"))
            clip_source: Any = None
            if source.startswith(("http://", "https://")):
                clip_source = await asyncio.to_thread(_downloaded_audio, source)
                if not clip_source:
                    return
                self._background_noise_file = clip_source
            elif source:
                clip_source = getattr(BuiltinAudioClip, source, None)
                if clip_source is None:
                    logger.warning(
                        "background audio clip %r is not one LiveKit ships", source
                    )
                    return
            # A player is created even with no ambience clip, because a mailbox tone needs a
            # published track whether or not this scenario also asked for a room.
            player = (
                BackgroundAudioPlayer(
                    ambient_sound=AudioConfig(clip_source, volume=volume)
                )
                if clip_source is not None
                else BackgroundAudioPlayer()
            )
            await player.start(room=room, agent_session=session)
            self._background_player = player
        except Exception:
            logger.warning("background audio not started", exc_info=True)
            return
        # Spoken as its own turn, so the transcript shows what the agent heard.
        recorded = os.environ.get("HARNESS_VOICEMAIL_CLIP", "").strip()
        if recorded.startswith(("http://", "https://")):
            # Catalogue clips are meant to be served from object storage rather than shipped in the
            # image, and a URL cannot be decoded in place.
            recorded = await asyncio.to_thread(_downloaded_audio, recorded) or ""
        said = os.environ.get("HARNESS_VOICEMAIL_CLIP_TRANSCRIPT", "").strip()
        if recorded and said:
            try:
                self._voicemail_greeting = session.say(
                    said,
                    audio=audio_frames_from_file(recorded),
                    allow_interruptions=False,
                )
            except Exception:
                logger.warning("recorded mailbox greeting not played", exc_info=True)
        elif recorded:
            # No words for it, so it cannot be a turn: an invented line would put words in the
            # transcript that the audio never says, and an eval would judge those words.
            logger.warning("mailbox clip has no transcript; playing it without a turn")
            try:
                player.play(AudioConfig(recorded, volume=1.0))
            except Exception:
                logger.warning("recorded mailbox greeting not played", exc_info=True)
        if tone_style:
            self._voicemail_tone_task = asyncio.create_task(
                self._play_voicemail_tone(tone_style, session)
            )
        if _answered_by_voicemail():
            self._mailbox_close_task = asyncio.create_task(
                self._close_mailbox_after_recording()
            )

    _voicemail_greeting: Any = None
    _voicemail_tone_task: Any = None
    _mailbox_close_task: Any = None

    async def _close_mailbox_after_recording(self) -> None:
        """Stop recording and cut the line, the way a mailbox does.

        A mailbox is not a party to the call. It never says goodbye, it never asks whether anybody is
        there, and it does not wait: it records for as long as it records and then hangs up. Nothing
        here was ending these calls, so they ran to the silence watchdog with the agent talking into
        a machine long after it had left its message.

        The clock starts once the greeting and the tone are done where there are either, and at call
        start otherwise, which is the FULL mailbox: it invites no message, so the window it gets is
        generous rather than precise.
        """
        try:
            if self._voicemail_greeting is not None:
                await self._voicemail_greeting
            tone = self._voicemail_tone_task
            if tone is not None:
                try:
                    await tone
                except Exception:
                    # The tone failing is not a reason to record for ever.
                    logger.warning(
                        "mailbox tone failed before the recording timer", exc_info=True
                    )
            await asyncio.sleep(_VOICEMAIL_RECORD_SECONDS)
            logger.info(
                "mailbox stopped recording after %ss and cut the line",
                _VOICEMAIL_RECORD_SECONDS,
            )
            self._end_requested.set()
        except asyncio.CancelledError:
            raise
        except Exception:
            # A mailbox that fails to hang up leaves the watchdog to end the call, which is the
            # behaviour this replaces rather than a new failure.
            logger.warning("mailbox recording timer failed", exc_info=True)

    async def _play_voicemail_tone(self, style: str, session: "AgentSession") -> None:
        """Play the tone a mailbox plays once its greeting has finished.

        The greeting comes either from a catalogue recording or from this session speaking the
        persona's opening line. The tone is the part neither can carry, and without it a greeting that
        says "leave a message after the tone" asks the agent to wait for something that never comes.

        Generated rather than fetched: a sine burst is a sine burst, and an asset would be a
        download, a licence and a catalogue for something twelve lines of arithmetic produce. Handed
        over eagerly rather than lazily, since the player consumes the iterator inside its mixer
        task and a failure there can take the ambience down with it.
        """
        shape = _VOICEMAIL_TONE_BY_STYLE.get(style)
        if shape is None:
            return
        hz, seconds = shape
        try:
            # After the greeting, not at a guessed offset: wait for the mailbox's own first
            # committed turn, which is this session's assistant role, then leave a beat.
            recorded = os.environ.get("HARNESS_VOICEMAIL_CLIP", "").strip()
            if recorded:
                # Awaiting the greeting's own handle is exact where a duration would be a guess.
                if self._voicemail_greeting is not None:
                    await self._voicemail_greeting
            else:
                loop = asyncio.get_running_loop()
                deadline = loop.time() + _VOICEMAIL_TONE_WAIT_SECONDS
                while loop.time() < deadline:
                    if any(
                        message["content"]
                        for message in _session_messages(session)
                        if message["role"] == "assistant"
                    ):
                        break
                    await asyncio.sleep(0.2)
                else:
                    logger.warning(
                        "mailbox said nothing in %ss; no tone",
                        _VOICEMAIL_TONE_WAIT_SECONDS,
                    )
                    return
            await asyncio.sleep(_VOICEMAIL_TONE_GAP_SECONDS)
            player = getattr(self, "_background_player", None)
            if player is None:
                return
            # A recording that ends with its own tone replaces this one rather than preceding it.
            # Two beeps is worse than one, and the catalogue records which clips carry theirs.
            if os.environ.get("HARNESS_VOICEMAIL_CLIP_HAS_TONE", "").strip() == "1":
                return
            spoken = [_tone_frame(hz, seconds)]

            async def frames() -> Any:
                for frame in spoken:
                    yield frame

            player.play(AudioConfig(frames(), volume=_VOICEMAIL_TONE_VOLUME))
        except asyncio.CancelledError:
            raise
        except Exception:
            # A mailbox without its tone is a weaker test, never a failed call.
            logger.warning("voicemail tone not played", exc_info=True)

    async def _stop_background_audio(self) -> None:
        """Close the ambience player and remove any clip downloaded for it.

        Without this the mixer task, its audio source and the published track outlive the call,
        and a suite leaks one of each (plus a temp file) per scenario.
        """
        for name in ("_voicemail_tone_task", "_mailbox_close_task"):
            pending = getattr(self, name, None)
            if pending is not None:
                setattr(self, name, None)
                if not pending.done():
                    pending.cancel()
        player = getattr(self, "_background_player", None)
        if player is not None:
            self._background_player = None
            try:
                await player.aclose()
            except Exception:
                logger.warning("background audio not closed cleanly", exc_info=True)
        downloaded = getattr(self, "_background_noise_file", None)
        if downloaded:
            self._background_noise_file = None
            try:
                Path(downloaded).unlink(missing_ok=True)
            except OSError:
                logger.warning("background audio clip not removed: %s", downloaded)

    def open_conversation(self) -> None:
        if self._session is None:
            raise RuntimeError("simulator_session_not_started")
        if self._voicemail_greeting is not None:
            # A recording has already greeted, and a mailbox does not greet twice: a spoken line on
            # top of the clip is one mailbox answering in two voices.
            return
        initial_message = self._persona.persona.get("initial_message")
        if isinstance(initial_message, str) and initial_message.strip():
            self._session.say(initial_message.strip())
            return
        self._session.generate_reply()

    _mailbox_greeted: bool = False

    async def llm_node(self, chat_ctx, tools, model_settings):
        """A mailbox speaks once and then never again, counted here rather than asked of the model.

        One turn is allowed, not none, because a mailbox without a recording greets through this path.
        Where a recording has already greeted, no turn is allowed at all.
        """
        if _answered_by_voicemail():
            if self._mailbox_greeted or self._voicemail_greeting is not None:
                return
            self._mailbox_greeted = True
        async for chunk in super().llm_node(chat_ctx, tools, model_settings):
            yield chunk

    async def transcription_node(
        self,
        text: AsyncIterable[str | TimedString],
        model_settings: ModelSettings,
    ):
        async for chunk in text:
            logger.debug(
                "Simulator transcription chunk",
                extra={"timed": isinstance(chunk, TimedString)},
            )
            yield chunk


class LiveKitEngine(BaseEngine):
    async def run(
        self,
        agent_definition: AgentDefinition | None = None,
        livekit_runtime: LiveKitSimulatorRuntime | None = None,
        scenario: Scenario | None = None,
        simulator: SimulatorAgentDefinition | None = None,
        num_scenarios: int = 1,
        topic: str | None = None,
        record_audio: bool = False,
        recorder_sample_rate: int = 8000,
        recorder_join_delay: float = 0.2,
        min_turn_messages: int = 8,
        max_seconds: float = 45.0,
        connect_timeout: float = 15.0,
        readiness_timeout: float = 30.0,
        cleanup_timeout: float = 30.0,
        conversation_direction: str = "simulator_first",
        agent_first_silence_timeout_seconds: float = 120.0,
        recording_root: str | Path = "recordings",
        recording_case_directory: str | Path | None = None,
        run_id: str | None = None,
        max_concurrency: int = 1,
        on_case_complete: Callable[[int, TestCaseResult], Awaitable[None]]
        | None = None,
        on_case_start: Callable[[int], Awaitable[None]] | None = None,
        **kwargs,
    ) -> TestReport:
        if agent_definition is None:
            raise ValueError("LiveKitEngine requires 'agent_definition'.")
        runtime = _resolve_livekit_runtime(agent_definition, livekit_runtime)
        if conversation_direction not in {"simulator_first", "agent_first"}:
            raise ValueError(
                "conversation_direction must be simulator_first or agent_first"
            )
        if agent_first_silence_timeout_seconds <= 0:
            raise ValueError("agent_first_silence_timeout_seconds must be positive")
        if scenario is None:
            generator = ScenarioGenerator(
                agent_definition,
                llm_config=(
                    simulator.llm
                    if simulator is not None
                    else _default_simulator_llm_config()
                ),
            )
            if topic is None:
                simulator_context = (
                    simulator.instructions
                    if simulator and simulator.instructions
                    else ""
                )
                topic = (
                    simulator_context
                    or agent_definition.system_prompt
                    or "customer support scenarios"
                ).strip()
            personas = await generator.generate(
                topic=topic,
                num_personas=num_scenarios,
            )
            scenario = Scenario(name="Generated Scenario", dataset=personas)
        transport = agent_definition.transport or TelephonyTransport()
        profile = _resolve_target_profile(transport.kind)
        # Computed before the room-name checks below: a serial (concurrency-1)
        # run is what makes reusing one fixed room across cases safe, so the
        # verbatim check needs this value, not just the dataset size.
        # Cases run concurrently up to ``max_concurrency`` (bounded by the
        # customer agent's own session capacity). SIP legs stay serial: a run
        # leases a single DID, so overlapping calls would collide. Ask the
        # resolved profile rather than re-branching on ``transport.kind``.
        case_concurrency = (
            1
            if profile.is_sip
            else max(
                1,
                min(
                    int(max_concurrency or 1),
                    _voice_max_case_concurrency(),
                    len(scenario.dataset),
                ),
            )
        )
        if (
            runtime.room_name_verbatim
            and len(scenario.dataset) != 1
            and case_concurrency != 1
        ):
            raise ValueError(
                "room_name_verbatim_requires_serial_cases: a fixed room hosts "
                "one case at a time; run with max_concurrency=1"
            )
        if (
            runtime.room_mode == "external"
            and len(scenario.dataset) > 1
            and not _has_room_template(runtime.room_name)
        ):
            raise ValueError(
                "external_room_template_required: concurrent-safe multi-case runs "
                "need {run_id}, {test_case_id}, or {index} in room_name"
            )
        if not profile.uses_external_room and runtime.room_mode != "managed":
            raise ValueError("managed_transport_requires_managed_room")
        if (
            profile.receives_inbound_call
            and len(scenario.dataset) > 1
            and not _has_room_template(runtime.room_name)
            and not runtime.room_name_verbatim
        ):
            raise ValueError(
                "sip_inbound_room_template_required: multi-case inbound runs "
                "need {run_id} or {test_case_id} in room_name"
            )
        cleanup_timeout = min(cleanup_timeout, _MAX_CLEANUP_TIMEOUT_SECONDS)
        current_run_id = run_id or new_run_id()
        if recording_case_directory is not None and len(scenario.dataset) != 1:
            raise ValueError(
                "recording_case_directory requires a single-persona scenario"
            )
        invocation_id = uuid4().hex[:12]
        report = TestReport()

        case_semaphore = asyncio.Semaphore(case_concurrency)

        async def _run_case(index: int, persona: Persona) -> TestCaseResult:
            async with case_semaphore:
                # Mark this case's row ONGOING the moment it claims a concurrency
                # slot — cases still queued behind the semaphore stay PENDING.
                # Best-effort and engine-agnostic; a failed ping never fails the
                # case (the backend gates the update on PENDING).
                if on_case_start is not None:
                    try:
                        await on_case_start(index)
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "voice case start callback failed",
                            exc_info=redacted_exc_info(exc),
                            extra={
                                "run_id": current_run_id,
                                "case_index": index,
                            },
                        )
                persona_ref = persona.version or persona.content_hash()
                test_case_id = derive_test_case_id(
                    current_run_id,
                    persona_ref,
                    index,
                )
                room_name = _resolve_room_name(
                    runtime,
                    run_id=current_run_id,
                    test_case_id=test_case_id,
                    index=index,
                    invocation_id=invocation_id,
                )
                case_directory = (
                    Path(recording_case_directory)
                    if recording_case_directory is not None
                    else Path(recording_root) / current_run_id / test_case_id
                )
                try:
                    outcome = await self._run_single_test_case(
                        agent_definition,
                        runtime,
                        persona,
                        simulator,
                        run_id=current_run_id,
                        test_case_id=test_case_id,
                        invocation_id=invocation_id,
                        room_name=room_name,
                        case_directory=case_directory,
                        record_audio=record_audio,
                        recorder_sample_rate=recorder_sample_rate,
                        recorder_join_delay=recorder_join_delay,
                        min_turn_messages=min_turn_messages,
                        max_seconds=max_seconds,
                        connect_timeout=connect_timeout,
                        readiness_timeout=readiness_timeout,
                        cleanup_timeout=cleanup_timeout,
                        conversation_direction=conversation_direction,
                        agent_first_silence_timeout_seconds=agent_first_silence_timeout_seconds,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    # One case crashing must neither sink the batch nor leave a
                    # hole that shifts the positional result-to-CallExecution
                    # mapping — emit a dense failed case in its slot.
                    logger.error(
                        "voice case crashed",
                        exc_info=redacted_exc_info(exc),
                        extra={
                            "run_id": current_run_id,
                            "test_case_id": test_case_id,
                        },
                    )
                    failure = SimulationFailure(
                        stage=FailureStage.RUNNING,
                        code="case_execution_error",
                        message=f"{type(exc).__name__}: {exc}",
                        retryable=False,
                    )
                    result = TestCaseResult(
                        persona=persona,
                        transcript="",
                        messages=[],
                        metadata={
                            "engine": "livekit",
                            "run_id": current_run_id,
                            "test_case_id": test_case_id,
                            "invocation_id": invocation_id,
                            "status": TestCaseStatus.FAILED.value,
                            "room_name": room_name,
                            "room_mode": runtime.room_mode,
                            "failure": failure.model_dump(
                                mode="json", exclude_none=True
                            ),
                        },
                    )
                else:
                    metadata = {
                        "engine": "livekit",
                        "run_id": current_run_id,
                        "test_case_id": test_case_id,
                        "invocation_id": invocation_id,
                        "status": outcome.status.value,
                        "room_name": room_name,
                        "room_mode": runtime.room_mode,
                        **outcome.metadata,
                    }
                    if outcome.failure is not None:
                        metadata["failure"] = outcome.failure.model_dump(
                            mode="json",
                            exclude_none=True,
                        )
                    if outcome.evidence:
                        metadata["evidence"] = [
                            item.model_dump(mode="json", exclude_none=True)
                            for item in outcome.evidence
                        ]
                    if outcome.provider_artifacts:
                        metadata["provider_artifacts"] = [
                            entry.model_dump(mode="json", exclude_none=True)
                            for entry in outcome.provider_artifacts
                        ]
                    result = TestCaseResult(
                        persona=persona,
                        transcript=outcome.transcript,
                        messages=outcome.messages,
                        metadata=metadata,
                        audio_input_path=outcome.audio_input_path,
                        audio_output_path=outcome.audio_output_path,
                        audio_combined_path=outcome.audio_combined_path,
                        audio_stereo_path=outcome.audio_stereo_path,
                    )

            # Stream the finished case AFTER releasing the semaphore — a slow
            # result PATCH (recording upload) must not hold a concurrency slot.
            # A streaming error never fails the case; finalize reconciles it.
            if on_case_complete is not None:
                try:
                    await on_case_complete(index, result)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "voice case stream callback failed",
                        exc_info=redacted_exc_info(exc),
                        extra={
                            "run_id": current_run_id,
                            "test_case_id": test_case_id,
                        },
                    )
            return result

        # ``gather`` preserves argument order regardless of completion order, so
        # ``report.results`` stays in dataset order — the positional contract the
        # FutureAGI sink relies on to map results to pre-allocated CallExecutions.
        results = await asyncio.gather(
            *(
                _run_case(index, persona)
                for index, persona in enumerate(scenario.dataset)
            )
        )
        report.results.extend(results)
        return report

    async def _run_single_test_case(
        self,
        agent_definition: AgentDefinition,
        runtime: LiveKitSimulatorRuntime,
        persona: Persona,
        simulator: SimulatorAgentDefinition | None,
        *,
        run_id: str,
        test_case_id: str,
        invocation_id: str,
        room_name: str,
        case_directory: Path,
        record_audio: bool,
        recorder_sample_rate: int,
        recorder_join_delay: float,
        min_turn_messages: int,
        max_seconds: float,
        connect_timeout: float,
        readiness_timeout: float,
        cleanup_timeout: float,
        conversation_direction: str,
        agent_first_silence_timeout_seconds: float,
    ) -> _CaseOutcome:
        # Teardown is a run of independent steps that each used to take the full
        # ``cleanup_timeout``. Ten of them at up to sixty seconds is six hundred seconds of
        # cleanup against a run budget of five hundred and seventy, so one slow teardown spent
        # the whole budget and the case was discarded as a timeout with its conversation already
        # finished. Share one deadline across the run instead, started at the first cleanup that
        # actually waits, so every path gets the same bound however it got there.
        _cleanup_started: list[float] = []

        def _cleanup_budget(
            cap: float = _CLEANUP_STEP_TIMEOUT_SECONDS,
        ) -> float:
            if not _cleanup_started:
                _cleanup_started.append(time.monotonic())
            spent = time.monotonic() - _cleanup_started[0]
            remaining = max(0.1, cleanup_timeout - spent)
            return min(cap, remaining)

        api_key = os.environ.get(runtime.api_key_env)
        api_secret = os.environ.get(runtime.api_secret_env)
        if not api_key or not api_secret:
            return _failure_outcome(
                TestCaseStatus.FAILED,
                FailureStage.PREPARING,
                "livekit_credentials_missing",
                f"{runtime.api_key_env} and {runtime.api_secret_env} are required",
            )
        simulator_identity = _simulator_participant_identity(persona, test_case_id)
        recorder_identity = f"fagi-recorder-{test_case_id[-12:]}"
        room = rtc.Room()
        models: LiveKitModels | None = None
        recorder: RoomRecorder | None = None
        customer_agent: _TestRunnerAgent | None = None
        session: AgentSession | None = None
        api_client: api.LiveKitAPI | None = None
        target: _TargetParticipant | None = None
        target_transcription_mode = False
        target_transcription_handler_registered = False
        target_transcription_tasks: set[asyncio.Task[None]] = set()
        # Set the moment the conversation ends. A target transcription stream
        # still in flight at that point is the target's final utterance; it must
        # be recorded into the transcript, but WITHOUT triggering another
        # simulator reply (the call is over).
        conversation_ended = asyncio.Event()
        # Every target utterance, captured straight off its transcription stream
        # independently of the simulator session. Once the session starts
        # draining it rejects new input ("speech scheduling is paused"), so a
        # target closing delivered after the simulator is done never reaches the
        # chat context. These are merged into the report so the trailing target
        # turn is never lost.
        captured_target_turns: list[dict[str, Any]] = []
        # agent_first (target greets first): the target can publish its greeting
        # transcription before the main handler is registered post-readiness, and
        # the LiveKit client DROPS a text-stream header that arrives with no
        # handler. So for managed external-room agent_first we register an early
        # buffer handler right after connect, defer the target dispatch until the
        # buffer is live, and drain the buffered streams through the (unchanged,
        # unconditional) main handler once the target is selected.
        pending_target_transcriptions: list[tuple["rtc.TextStreamReader", str]] = []
        target_dispatch_deferred = False
        _MAX_BUFFERED_TARGET_STREAMS = 16
        managed_room_owned = runtime.room_mode == "managed"
        room_connected = False
        cleanup_errors: list[str] = []
        outcome: _CaseOutcome | None = None
        sip_dispatch_rule_id: str | None = None
        sip_dispatch_rule_created = False
        call_originator: CallOriginator | None = None
        provider_call_id: str | None = None
        provider_termination_source: str | None = None
        reconciled_call_ids: list[str] = []
        caller_verification: str | None = None
        audio_bridge: LiveKitAudioBridge | None = None
        bridge_task: asyncio.Task[None] | None = None
        case_started_at = datetime.now(timezone.utc)
        transport = agent_definition.transport or TelephonyTransport()
        profile = _resolve_target_profile(transport.kind)
        provider_target = agent_definition.target
        effective_target_identity = agent_definition.target_participant_identity
        effective_readiness_timeout = (
            transport.readiness_timeout_seconds
            if profile.receives_inbound_call
            and transport.readiness_timeout_seconds is not None
            else readiness_timeout
        )
        sip_answer_timeout = transport.answer_timeout_seconds or max(
            connect_timeout, 60.0
        )

        def buffer_target_transcription(
            reader: "rtc.TextStreamReader",
            participant_identity: str,
        ) -> None:
            # Early (pre-readiness) handler for managed external-room agent_first.
            # ``target.audio_track_sid`` isn't known yet, so attribute by the same
            # exclusion ``_wait_for_target_audio`` uses — anything that is not the
            # simulator or recorder is target-worthy. The main handler re-applies
            # the strict track/identity filter when draining, so over-buffering is
            # safe.
            nonlocal target_transcription_mode
            pid = str(participant_identity)
            if pid in (simulator_identity, recorder_identity):
                return
            if len(pending_target_transcriptions) >= _MAX_BUFFERED_TARGET_STREAMS:
                logger.warning(
                    "target_transcription_buffer_full: dropping stream (buffered=%d)",
                    len(pending_target_transcriptions),
                )
                return
            pending_target_transcriptions.append((reader, pid))
            # Kill the duplicate-response race at the source: the target is
            # speaking, so disable the simulator's STT now — otherwise STT would
            # also transcribe the greeting and emit a second, duplicate reply.
            # Dispatch is deferred until after ``session.start()``, so a buffered
            # stream implies a live session.
            if not target_transcription_mode and session is not None:
                session.input.set_audio_enabled(False)
                session.clear_user_turn()
                target_transcription_mode = True

        try:
            if managed_room_owned:
                api_client = api.LiveKitAPI(
                    _api_url(str(runtime.url)),
                    api_key,
                    api_secret,
                )
                if not profile.places_outbound_call:
                    # The pool's dispatch rule stays live for the whole run, so a
                    # stray inbound call can re-create this room at any moment
                    # between cases — drain it before trusting it as ours.
                    if runtime.room_name_verbatim:
                        try:
                            polls = await asyncio.wait_for(
                                _ensure_room_absent(api_client, room_name),
                                timeout=connect_timeout,
                            )
                            logger.info(
                                "leased room drained",
                                extra={
                                    "run_id": run_id,
                                    "test_case_id": test_case_id,
                                    "room_name": room_name,
                                    "polls": polls,
                                },
                            )
                        except asyncio.TimeoutError:
                            outcome = _failure_outcome(
                                TestCaseStatus.TIMED_OUT,
                                FailureStage.PREPARING,
                                "livekit_room_drain_timeout",
                                "The leased simulator room was still occupied when its drain deadline passed",
                                retryable=True,
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "leased room drain failed",
                                exc_info=redacted_exc_info(exc),
                                extra={
                                    "run_id": run_id,
                                    "test_case_id": test_case_id,
                                    "room_name": room_name,
                                },
                            )
                            outcome = _failure_outcome(
                                TestCaseStatus.FAILED,
                                FailureStage.PREPARING,
                                "livekit_room_drain_failed",
                                "Could not clear the leased simulator room before the call",
                                details=_safe_provider_error_details(
                                    exc, operation="room_drain"
                                ),
                            )
                    if outcome is None:
                        try:
                            await asyncio.wait_for(
                                api_client.room.create_room(
                                    api.CreateRoomRequest(name=room_name)
                                ),
                                timeout=connect_timeout,
                            )
                        except asyncio.TimeoutError:
                            outcome = _failure_outcome(
                                TestCaseStatus.TIMED_OUT,
                                FailureStage.PREPARING,
                                "livekit_room_create_timeout",
                                "LiveKit room creation exceeded its deadline",
                                retryable=True,
                            )
                        except Exception as exc:
                            logger.warning(
                                "LiveKit room creation failed",
                                exc_info=redacted_exc_info(exc),
                                extra={
                                    "run_id": run_id,
                                    "test_case_id": test_case_id,
                                    "room_name": room_name,
                                },
                            )
                            outcome = _failure_outcome(
                                TestCaseStatus.FAILED,
                                FailureStage.PREPARING,
                                "livekit_room_create_failed",
                                "Failed to create the LiveKit room",
                                details=_safe_provider_error_details(
                                    exc, operation="room_create"
                                ),
                            )
                if outcome is None and profile.uses_external_room:
                    # Defer the target dispatch until AFTER the early buffer
                    # handler is registered and the session is live (both
                    # directions): a native target may greet the moment it
                    # joins even when the simulator is meant to open, and the
                    # LiveKit client drops a text-stream header that arrives
                    # with no handler — the greeting would be lost.
                    target_dispatch_deferred = True
                elif outcome is None and profile.receives_inbound_call:
                    try:
                        (
                            sip_dispatch_rule_id,
                            sip_dispatch_rule_created,
                        ) = await asyncio.wait_for(
                            _ensure_sip_inbound_dispatch(
                                api_client,
                                transport=transport,
                                room_name=room_name,
                            ),
                            timeout=connect_timeout,
                        )
                    except asyncio.TimeoutError:
                        outcome = _failure_outcome(
                            TestCaseStatus.TIMED_OUT,
                            FailureStage.PREPARING,
                            "sip_inbound_dispatch_timeout",
                            "SIP inbound dispatch provisioning exceeded its deadline",
                            retryable=True,
                        )
                    except Exception as exc:
                        logger.warning(
                            "SIP inbound dispatch provisioning failed",
                            exc_info=redacted_exc_info(exc),
                            extra={
                                "run_id": run_id,
                                "test_case_id": test_case_id,
                                "room_name": room_name,
                            },
                        )
                        outcome = _failure_outcome(
                            TestCaseStatus.FAILED,
                            FailureStage.PREPARING,
                            "sip_inbound_dispatch_failed",
                            "Failed to provision SIP inbound dispatch",
                            details=_safe_provider_error_details(
                                exc, operation="sip_dispatch"
                            ),
                        )
            if outcome is not None:
                return outcome
            # The target resolves who is calling from participant attributes or metadata.
            # Without the persona's number every scenario looks like the same demo rider and
            # the agent looks up the wrong account, which reads as an agent bug.
            caller_phone = str(
                (persona.persona.get("metadata") or {}).get("caller_phone") or ""
            ).strip()
            builder = (
                AccessToken(api_key, api_secret)
                .with_identity(simulator_identity)
                .with_grants(VideoGrants(room_join=True, room=room_name))
            )
            if caller_phone:
                builder = builder.with_attributes(
                    {"harness.callerPhone": caller_phone}
                ).with_metadata(json.dumps({"caller_phone": caller_phone}))
            token = builder.to_jwt()
            await asyncio.wait_for(
                room.connect(str(runtime.url), token),
                timeout=connect_timeout,
            )
            room_connected = True
            if profile.uses_external_room:
                # Buffer any target greeting that arrives before readiness; the
                # LiveKit client drops a text-stream header with no handler.
                room.register_text_stream_handler(
                    TOPIC_TRANSCRIPTION, buffer_target_transcription
                )
                target_transcription_handler_registered = True
            if record_audio:
                recorder = RoomRecorder(
                    url=str(runtime.url),
                    api_key=api_key,
                    api_secret=api_secret,
                    room_name=room_name,
                    identity=recorder_identity,
                    sample_rate=recorder_sample_rate,
                    output_dir=case_directory / "audio",
                    join_delay_s=recorder_join_delay,
                )
                await asyncio.wait_for(
                    recorder.start(),
                    timeout=connect_timeout,
                )
            customer_agent, models = await self._create_customer_agent(
                persona,
                simulator,
                # The AGENT's direction; it picks which half of the role block the caller gets.
                call_type=(
                    "outbound"
                    if os.environ.get("HARNESS_CALL_DIRECTION", "").strip().lower()
                    == "outbound"
                    else "inbound"
                ),
                # `name` is an identity for dispatch, not a label for the caller to hear.
                agent_name=agent_definition.description,
                min_turn_messages=min_turn_messages,
            )
            setup = getattr(self, "_last_simulator_setup", {}) or {}
            _record_simulator_setup(
                case_directory,
                persona=persona,
                instructions=setup.get("instructions", ""),
                llm_config=setup.get("llm_config"),
                stt_config=setup.get("stt_config"),
                tts_config=setup.get("tts_config"),
                extra={
                    "room_name": room_name,
                    "agent_name": agent_definition.name,
                    "test_case_id": test_case_id,
                    "run_id": run_id,
                    "conversation_direction": conversation_direction,
                    "allow_interruptions": setup.get("allow_interruptions"),
                    "min_endpointing_delay": setup.get("min_endpointing_delay"),
                    "max_endpointing_delay": setup.get("max_endpointing_delay"),
                    "use_tts_aligned_transcript": setup.get(
                        "use_tts_aligned_transcript"
                    ),
                },
            )
            sip_participant_identity: str | None = None
            bridge_identity: str | None = None
            if profile.places_outbound_call:
                identity_template = (
                    transport.participant_identity
                    or "sip-caller-{invocation_id}-{test_case_id}"
                )
                sip_participant_identity = identity_template.format(
                    test_case_id=test_case_id,
                    run_id=run_id,
                    invocation_id=invocation_id,
                )
                if effective_target_identity is None:
                    effective_target_identity = sip_participant_identity
            elif profile.uses_web_audio_bridge:
                bridge_identity = (
                    f"fagi-{profile.bridge_provider}-bridge-{test_case_id[-12:]}"
                )
                effective_target_identity = bridge_identity
            session_participant_kinds = None
            session_participant_identity: str | None = None
            if profile.joins_as_sip_participant:
                session_participant_kinds = [rtc.ParticipantKind.PARTICIPANT_KIND_SIP]
                session_participant_identity = (
                    effective_target_identity or sip_participant_identity
                )
            session = await asyncio.wait_for(
                customer_agent.start_session(
                    room,
                    participant_kinds=session_participant_kinds,
                    participant_identity=session_participant_identity,
                ),
                timeout=connect_timeout,
            )
            if target_dispatch_deferred:
                # Session + early buffer handler are live; now dispatch the target
                # so its greeting stream is captured, not dropped.
                assert api_client is not None
                dispatch_agent_name = (
                    agent_definition.agent_name or agent_definition.name
                )
                try:
                    await asyncio.wait_for(
                        api_client.agent_dispatch.create_dispatch(
                            api.CreateAgentDispatchRequest(
                                agent_name=dispatch_agent_name,
                                room=room_name,
                                metadata=_dispatch_metadata_json(agent_definition),
                            )
                        ),
                        timeout=connect_timeout,
                    )
                except asyncio.TimeoutError:
                    outcome = _failure_outcome(
                        TestCaseStatus.TIMED_OUT,
                        FailureStage.PREPARING,
                        "livekit_dispatch_timeout",
                        "Target agent dispatch exceeded its deadline",
                        retryable=True,
                    )
                    return outcome
                except Exception as exc:
                    logger.warning(
                        "LiveKit target dispatch failed",
                        exc_info=redacted_exc_info(exc),
                        extra={
                            "run_id": run_id,
                            "test_case_id": test_case_id,
                            "room_name": room_name,
                        },
                    )
                    outcome = _failure_outcome(
                        TestCaseStatus.FAILED,
                        FailureStage.PREPARING,
                        "livekit_dispatch_failed",
                        "Failed to dispatch the target agent",
                        details=_safe_provider_error_details(
                            exc, operation="agent_dispatch"
                        ),
                    )
                    return outcome
                logger.info(
                    "livekit_target_dispatched agent=%s room=%s run=%s case=%s",
                    dispatch_agent_name,
                    room_name,
                    run_id,
                    test_case_id,
                )
            if profile.uses_web_audio_bridge:
                try:
                    connector = profile.build_connector(
                        provider_target,
                        conversation_direction=conversation_direction,
                    )
                    audio_bridge = LiveKitAudioBridge(
                        url=str(runtime.url),
                        api_key=api_key,
                        api_secret=api_secret,
                        room_name=room_name,
                        identity=bridge_identity or "fagi-provider-bridge",
                        connector=connector,
                    )
                    await asyncio.wait_for(
                        audio_bridge.connect(), timeout=connect_timeout
                    )
                    provider_call_id = audio_bridge.call_id
                    bridge_task = asyncio.create_task(audio_bridge.run())
                except asyncio.TimeoutError:
                    outcome = _failure_outcome(
                        TestCaseStatus.TIMED_OUT,
                        FailureStage.PREPARING,
                        "web_bridge_start_timeout",
                        "Provider web call creation exceeded its deadline",
                        retryable=True,
                    )
                    return outcome
                except Exception as exc:
                    logger.warning(
                        "Provider web bridge creation failed",
                        exc_info=redacted_exc_info(exc),
                        extra={
                            "run_id": run_id,
                            "test_case_id": test_case_id,
                            "transport": transport.kind,
                        },
                    )
                    outcome = _failure_outcome(
                        TestCaseStatus.FAILED,
                        FailureStage.PREPARING,
                        "web_bridge_start_failed",
                        "Failed to start the provider web bridge",
                        details=_safe_provider_error_details(
                            exc, operation="web_bridge_start"
                        ),
                    )
                    return outcome
            if profile.places_outbound_call and api_client is not None:
                try:
                    logger.info(
                        "sip_outbound_dialing",
                        extra={
                            "run_id": run_id,
                            "test_case_id": test_case_id,
                            "room_name": room_name,
                        },
                    )
                    await asyncio.wait_for(
                        api_client.sip.create_sip_participant(
                            api.CreateSIPParticipantRequest(
                                sip_trunk_id=transport.sip_trunk_id,
                                sip_number=transport.sip_number,
                                sip_call_to=transport.sip_call_to,
                                room_name=room_name,
                                participant_identity=sip_participant_identity,
                                wait_until_answered=True,
                                play_ringtone=True,
                            )
                        ),
                        timeout=sip_answer_timeout,
                    )
                except asyncio.TimeoutError:
                    outcome = _failure_outcome(
                        TestCaseStatus.TIMED_OUT,
                        FailureStage.PREPARING,
                        "sip_answer_timeout",
                        "Outbound SIP call was not answered before the deadline",
                        retryable=True,
                    )
                    return outcome
                except Exception as exc:
                    logger.warning(
                        "SIP dial failed",
                        exc_info=redacted_exc_info(exc),
                        extra={
                            "run_id": run_id,
                            "test_case_id": test_case_id,
                            "room_name": room_name,
                        },
                    )
                    outcome = _failure_outcome(
                        TestCaseStatus.FAILED,
                        FailureStage.PREPARING,
                        "sip_dial_failed",
                        "Failed to dial the SIP participant",
                        details=_safe_provider_error_details(exc, operation="sip_dial"),
                    )
                    return outcome
            if profile.receives_inbound_call:
                logger.info(
                    "sip_inbound_ready",
                    extra={
                        "run_id": run_id,
                        "test_case_id": test_case_id,
                        "room_name": room_name,
                        "sip_dispatch_rule_id": sip_dispatch_rule_id,
                        "sip_dispatch_rule_created": sip_dispatch_rule_created,
                    },
                )
            if runtime.room_name_verbatim and profile.receives_inbound_call:
                unexpected = _unexpected_participants(
                    room,
                    simulator_identity=simulator_identity,
                    recorder_identity=recorder_identity,
                )
                if unexpected:
                    logger.warning(
                        "leased room occupied before dial",
                        extra={
                            "run_id": run_id,
                            "test_case_id": test_case_id,
                            "room_name": room_name,
                            "unexpected_participants": len(unexpected),
                        },
                    )
                    outcome = _failure_outcome(
                        TestCaseStatus.FAILED,
                        FailureStage.PREPARING,
                        "livekit_room_occupied",
                        "Another participant was already in the leased simulator room before the call was placed",
                        retryable=True,
                    )
                    return outcome
            if transport.inbound_call_originator is not None:
                name = transport.inbound_call_originator
                try:
                    call_originator = build_call_originator(transport)
                    originated_call = await asyncio.wait_for(
                        call_originator.start(), timeout=connect_timeout
                    )
                    provider_call_id = originated_call.call_id
                except asyncio.TimeoutError:
                    outcome = _failure_outcome(
                        TestCaseStatus.TIMED_OUT,
                        FailureStage.PREPARING,
                        f"{name}_call_start_timeout",
                        f"{name.capitalize()} call creation exceeded its deadline",
                        retryable=True,
                    )
                    return outcome
                except Exception as exc:
                    logger.warning(
                        f"{name.capitalize()} call creation failed",
                        exc_info=redacted_exc_info(exc),
                        extra={
                            "run_id": run_id,
                            "test_case_id": test_case_id,
                        },
                    )
                    outcome = _failure_outcome(
                        TestCaseStatus.FAILED,
                        FailureStage.PREPARING,
                        f"{name}_call_start_failed",
                        f"Failed to start the {name.capitalize()} call",
                        details=_safe_provider_error_details(
                            exc, operation=f"{name}_call_start"
                        ),
                    )
                    return outcome
            target = await _wait_for_target_audio(
                room,
                excluded_identities={simulator_identity, recorder_identity},
                target_identity=effective_target_identity,
                timeout=effective_readiness_timeout,
            )
            if (
                runtime.room_name_verbatim
                and profile.receives_inbound_call
                and transport.originator_from_number
            ):
                verdict = _caller_matches(
                    target.attributes, transport.originator_from_number
                )
                if verdict is False:
                    logger.warning(
                        "leased room wrong caller",
                        extra={
                            "run_id": run_id,
                            "test_case_id": test_case_id,
                            "expected_digits": len(
                                _number_digits(transport.originator_from_number)
                            ),
                            "observed_digits": len(
                                _number_digits(
                                    target.attributes.get(
                                        _SIP_REMOTE_NUMBER_ATTRIBUTE, ""
                                    )
                                )
                            ),
                        },
                    )
                    caller_verification = "mismatch"
                    raise _LeasedRoomCallerMismatch()
                elif verdict is None:
                    logger.warning(
                        "leased room caller unverified",
                        extra={"run_id": run_id, "test_case_id": test_case_id},
                    )
                    caller_verification = "unverified"
                else:
                    caller_verification = "matched"
            logger.info(
                "livekit_target_joined identity=%s sid=%s track=%s run=%s case=%s",
                target.identity,
                target.sid,
                target.audio_track_sid,
                run_id,
                test_case_id,
            )
            # RoomIO auto-links to the first participant that joined — the
            # recorder, which publishes no audio — so the simulator's STT never
            # hears the target. Re-point it at the target readiness selected.
            target_room_io = getattr(session, "room_io", None)
            if target_room_io is not None:
                target_room_io.set_participant(target.identity)

            # Swap the early buffer handler for the authoritative one. The
            # unregister → def → register sequence has NO await between the
            # unregister and register, so no header can land unhandled in the gap
            # (LiveKit allows one handler per topic and raises on double-register).
            if target_transcription_handler_registered:
                room.unregister_text_stream_handler(TOPIC_TRANSCRIPTION)
                target_transcription_handler_registered = False

            # The target's mic track stays open for the whole call, so STT never
            # finalizes a turn; the agent's authoritative turns arrive on its
            # lk.transcription stream instead. Consume that stream and feed each
            # completed target utterance into the simulator as a user turn.
            def on_target_transcription(
                reader: "rtc.TextStreamReader",
                participant_identity: str,
            ) -> None:
                nonlocal target_transcription_mode
                attrs = reader.info.attributes or {}
                transcribed_track_id = attrs.get(ATTRIBUTE_TRANSCRIPTION_TRACK_ID)
                if transcribed_track_id:
                    if transcribed_track_id != target.audio_track_sid:
                        return
                elif str(participant_identity) != target.identity:
                    return
                # First target transcription means the target is speaking — stop
                # the redundant simulator STT so it cannot emit duplicate turns.
                if not target_transcription_mode:
                    session.input.set_audio_enabled(False)
                    session.clear_user_turn()
                    target_transcription_mode = True
                task = asyncio.create_task(
                    _forward_target_transcription(
                        reader,
                        session,
                        conversation_ended=conversation_ended,
                        captured_target_turns=captured_target_turns,
                    )
                )
                target_transcription_tasks.add(task)
                task.add_done_callback(target_transcription_tasks.discard)

            room.register_text_stream_handler(
                TOPIC_TRANSCRIPTION, on_target_transcription
            )
            target_transcription_handler_registered = True

            # Drain greeting streams buffered before readiness through the
            # authoritative handler (it re-applies the strict track/identity
            # filter, so over-buffered non-target streams are dropped here).
            buffered_streams = list(pending_target_transcriptions)
            pending_target_transcriptions.clear()
            for buffered_reader, buffered_identity in buffered_streams:
                on_target_transcription(buffered_reader, buffered_identity)

            opener: asyncio.Task[None] | None = None
            if conversation_direction == "simulator_first" or _answered_by_voicemail():
                # A mailbox speaks first and needs no watchdog to break a mutual silence.
                customer_agent.open_conversation()
            else:
                # The agent placed this call and should speak first. If it does not, the person
                # answers rather than both sides waiting for each other.
                opener = asyncio.create_task(
                    _open_if_nobody_speaks_first(
                        session,
                        customer_agent,
                        timeout_seconds=_OPEN_INSTEAD_AFTER_SECONDS,
                    )
                )
            try:
                stop_reason = await _wait_for_conversation_end(
                    room,
                    session,
                    customer_agent=customer_agent,
                    target_identity=target.identity,
                    timeout=max_seconds,
                    conversation_direction=conversation_direction,
                    agent_first_silence_timeout_seconds=agent_first_silence_timeout_seconds,
                    provider_task=bridge_task,
                )
            finally:
                # However the conversation ended, including badly, the watchdog goes with it: a
                # pending task at loop close is noise in the log of every call.
                if opener is not None and not opener.done():
                    opener.cancel()
            logger.info(
                "livekit_conversation_ended stop_reason=%s run=%s case=%s",
                stop_reason,
                run_id,
                test_case_id,
            )
            # End the call cleanly. First let the party that just spoke commit its
            # own final turn — a LiveKit turn only lands in history once its TTS
            # finishes — bounded so we do not wait on the other side. We do NOT
            # wait for the target's trailing speech: once the conversation has
            # ended, the target talking on is monologuing into a call the other
            # side left.
            conversation_ended.set()
            if stop_reason == "simulator_end_call":
                wait_for_end_speech = getattr(
                    customer_agent,
                    "wait_for_end_speech",
                    None,
                )
                try:
                    if callable(wait_for_end_speech):
                        await asyncio.wait_for(
                            wait_for_end_speech(),
                            timeout=_FINAL_TURN_COMMIT_WAIT_SECONDS,
                        )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Simulator closing speech did not finish before cleanup",
                        extra={"run_id": run_id, "test_case_id": test_case_id},
                    )
            _loop = asyncio.get_running_loop()
            _commit_deadline = _loop.time() + _FINAL_TURN_COMMIT_WAIT_SECONDS
            while _loop.time() < _commit_deadline:
                try:
                    if session.current_speech is None:
                        break
                except Exception:  # noqa: BLE001
                    break
                await asyncio.sleep(0.2)
            # Delete the room so the target agent can't keep monologuing into a
            # dead call (its audio would be recorded but is untranscribable once
            # the simulator has left) — the recording then ends when the call
            # actually ends, matching the transcript.
            if api_client is not None and managed_room_owned:
                try:
                    await asyncio.wait_for(
                        api_client.room.delete_room(
                            api.DeleteRoomRequest(room=room_name)
                        ),
                        timeout=_cleanup_budget(),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "LiveKit room delete on conversation end failed",
                        exc_info=redacted_exc_info(exc),
                    )
            messages = _canonical_report_messages(session)
            messages = _merge_captured_target_turns(messages, captured_target_turns)
            outcome = _conversation_outcome(
                stop_reason,
                messages,
                min_turn_messages=min_turn_messages,
            )
        except asyncio.TimeoutError:
            stage = (
                FailureStage.READINESS
                if session is not None and target is None
                else FailureStage.PREPARING
            )
            if stage == FailureStage.READINESS and profile.receives_inbound_call:
                code = "sip_inbound_no_participant"
                message = "No inbound SIP participant joined before deadline"
            elif stage == FailureStage.READINESS:
                code = "agent_unavailable"
                message = "Target agent did not become ready"
            else:
                code = "livekit_connect_timeout"
                message = "LiveKit setup exceeded its deadline"
            status = (
                TestCaseStatus.AGENT_UNAVAILABLE
                if stage == FailureStage.READINESS
                else TestCaseStatus.TIMED_OUT
            )
            outcome = _failure_outcome(
                status,
                stage,
                code,
                message,
                retryable=True,
            )
        except _LeasedRoomCallerMismatch:
            # Unlike the pre-dial failures above, this path has a billed call and
            # a joined target; the recordings, provider evidence and metadata
            # update below all run after the try, so this must not return early.
            outcome = _failure_outcome(
                TestCaseStatus.FAILED,
                FailureStage.READINESS,
                "livekit_room_wrong_caller",
                "The participant that answered is not calling from the customer's configured number",
                retryable=True,
            )
        except Exception as exc:
            logger.error(
                "LiveKit test case failed",
                exc_info=redacted_exc_info(exc),
                extra={
                    "run_id": run_id,
                    "test_case_id": test_case_id,
                    "exception_type": type(exc).__name__,
                },
            )
            outcome = _failure_outcome(
                TestCaseStatus.FAILED,
                FailureStage.RUNNING if session is not None else FailureStage.PREPARING,
                "livekit_case_failed",
                "LiveKit test case failed",
                details={"exception_type": type(exc).__name__},
            )
        finally:
            # The ambience belongs to the caller agent, not the engine. Guarded because teardown
            # must never be the reason a case fails.
            if customer_agent is not None:
                try:
                    # Bounded like every other teardown step. Closing the ambience player unpublishes
                    # its track, and when the room's signal client has already died, that wait never
                    # returns: the SDK loops on resume and restart while this await sits here, and the
                    # case never completes, so the whole run is discarded on its deadline with a
                    # finished conversation inside it. Measured: two calls ended on endCall at 15 and
                    # 17 messages and both reported 570004ms and no test case.
                    await asyncio.wait_for(
                        customer_agent._stop_background_audio(),
                        timeout=_cleanup_budget(
                            _BACKGROUND_AUDIO_CLEANUP_TIMEOUT_SECONDS
                        ),
                    )
                except Exception:
                    logger.warning("background audio not closed cleanly", exc_info=True)
            if target_transcription_handler_registered:
                room.unregister_text_stream_handler(TOPIC_TRANSCRIPTION)
            pending_target_transcriptions.clear()
            pending_transcriptions = list(target_transcription_tasks)
            for pending in pending_transcriptions:
                pending.cancel()
            if pending_transcriptions:
                # Cancelled above, but a task blocked reading a stream whose connection is gone does
                # not observe the cancellation, so this is bounded too.
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*pending_transcriptions, return_exceptions=True),
                        timeout=_cleanup_budget(),
                    )
                except Exception as exc:  # noqa: BLE001 - teardown never fails a case
                    logger.warning("transcription tasks did not stop cleanly: %s", exc)
            session_to_close = session or (
                getattr(customer_agent, "started_session", None)
                if customer_agent is not None
                else None
            )
            if session_to_close is not None:
                try:
                    await _close_agent_session(
                        session_to_close,
                        timeout=_cleanup_budget(_SESSION_CLEANUP_TIMEOUT_SECONDS),
                    )
                except Exception as exc:
                    _record_cleanup_error(
                        cleanup_errors,
                        exc,
                        "session_close",
                        run_id,
                        test_case_id,
                    )
            if models is not None:
                try:
                    await asyncio.wait_for(
                        models.aclose(),
                        timeout=_cleanup_budget(),
                    )
                except Exception as exc:
                    _record_cleanup_error(
                        cleanup_errors,
                        exc,
                        "models_close",
                        run_id,
                        test_case_id,
                    )
            if recorder is not None:
                try:
                    await asyncio.wait_for(
                        recorder.aclose(),
                        timeout=_cleanup_budget(),
                    )
                except Exception as exc:
                    _record_cleanup_error(
                        cleanup_errors,
                        exc,
                        "recorder_close",
                        run_id,
                        test_case_id,
                    )
            if audio_bridge is not None:
                try:
                    await asyncio.wait_for(
                        audio_bridge.aclose(), timeout=_cleanup_budget()
                    )
                    if bridge_task is not None:
                        await asyncio.wait_for(bridge_task, timeout=_cleanup_budget())
                except Exception as exc:
                    _record_cleanup_error(
                        cleanup_errors,
                        exc,
                        "bridge_close",
                        run_id,
                        test_case_id,
                    )
            if room_connected:
                try:
                    await asyncio.wait_for(room.disconnect(), timeout=_cleanup_budget())
                except Exception as exc:
                    _record_cleanup_error(
                        cleanup_errors,
                        exc,
                        "room_disconnect",
                        run_id,
                        test_case_id,
                    )
            if call_originator is not None:
                # Guarded like every other cleanup step below: a future escape
                # from the helper (today it never raises) must not skip the
                # dispatch-rule/room/api-client teardown that follows.
                try:
                    finalize_result = await finalize_originator(
                        call_originator,
                        provider_call_id=provider_call_id,
                        originator_name=transport.inbound_call_originator,
                        case_started_at=case_started_at,
                        cleanup_timeout=_cleanup_budget(),
                    )
                    for operation, exc in finalize_result.cleanup_errors:
                        _record_cleanup_error(
                            cleanup_errors, exc, operation, run_id, test_case_id
                        )
                    if finalize_result.termination_source:
                        provider_termination_source = finalize_result.termination_source
                    reconciled_call_ids = finalize_result.reconciled_call_ids
                    if provider_call_id is None and reconciled_call_ids:
                        # We stopped a billed call we believe is ours, so
                        # evidence fetches its record instead of reporting
                        # "not matched" after searching nothing.
                        provider_call_id = reconciled_call_ids[0]
                    # Written here too (not only in the metadata.update below)
                    # because a start-failure path returns from inside the try,
                    # bypassing that block entirely.
                    if outcome is not None:
                        outcome.metadata["reconciled_call_ids"] = reconciled_call_ids
                        if finalize_result.termination_source:
                            outcome.metadata["provider_termination_source"] = (
                                finalize_result.termination_source
                            )
                except Exception as exc:
                    _record_cleanup_error(
                        cleanup_errors,
                        exc,
                        f"{transport.inbound_call_originator}_call_finalize",
                        run_id,
                        test_case_id,
                    )
            if (
                api_client is not None
                and sip_dispatch_rule_id
                and sip_dispatch_rule_created
            ):
                try:
                    await asyncio.wait_for(
                        _delete_sip_dispatch_rule(api_client, sip_dispatch_rule_id),
                        timeout=_cleanup_budget(),
                    )
                except Exception as exc:
                    if not _is_not_found(exc):
                        _record_cleanup_error(
                            cleanup_errors,
                            exc,
                            "sip_dispatch_delete",
                            run_id,
                            test_case_id,
                        )
            if api_client is not None and managed_room_owned:
                try:
                    await asyncio.wait_for(
                        api_client.room.delete_room(
                            api.DeleteRoomRequest(room=room_name)
                        ),
                        timeout=_cleanup_budget(),
                    )
                except Exception as exc:
                    if not _is_not_found(exc):
                        _record_cleanup_error(
                            cleanup_errors,
                            exc,
                            "room_delete",
                            run_id,
                            test_case_id,
                        )
            if api_client is not None:
                try:
                    await api_client.aclose()
                except Exception as exc:
                    _record_cleanup_error(
                        cleanup_errors,
                        exc,
                        "api_close",
                        run_id,
                        test_case_id,
                    )
            # Written here too (not only in the metadata.update below) because a
            # pre-dial return (e.g. the occupancy check) exits from inside the
            # try, bypassing that block entirely.
            if outcome is not None and outcome.metadata.get("cleanup_status") is None:
                outcome.metadata["cleanup_status"] = (
                    "failed" if cleanup_errors else "completed"
                )
                outcome.metadata["cleanup_errors"] = cleanup_errors
        if outcome is None:
            outcome = _failure_outcome(
                TestCaseStatus.FAILED,
                FailureStage.FINALIZING,
                "livekit_outcome_missing",
                "LiveKit test case ended without an outcome",
            )
        if recorder is not None:
            _attach_recordings(
                outcome,
                recorder,
                simulator_identity=simulator_identity,
                target_identity=target.identity if target is not None else None,
                target_track_sid=(
                    target.audio_track_sid if target is not None else None
                ),
                case_directory=case_directory,
                sample_rate=recorder_sample_rate,
            )
            if recorder.errors:
                cleanup_errors.extend(
                    f"recording:{type(error).__name__}" for error in recorder.errors
                )
        if agent_definition.provider_evidence is not None:
            provider_summary, provider_artifacts = await _collect_provider_evidence(
                config=agent_definition.provider_evidence,
                transport=transport,
                run_id=run_id,
                test_case_id=test_case_id,
                case_directory=case_directory,
                started_at=case_started_at,
                target=target,
                provider_call_id_hint=provider_call_id,
                provider_api_key=_target_api_key(provider_target),
                provider_api_base_url=_target_evidence_base_url(provider_target),
                termination_source=provider_termination_source,
            )
            if provider_summary is not None:
                outcome.evidence.append(provider_summary)
                if provider_call_id is None:
                    resolved_call_id = provider_summary.metadata.get("call_id")
                    if resolved_call_id:
                        provider_call_id = str(resolved_call_id)
                _recover_successful_provider_end_call(outcome, provider_summary)
            outcome.provider_artifacts.extend(provider_artifacts)
        outcome.metadata.update(
            {
                "simulator_participant_identity": simulator_identity,
                "target_participant_identity": (
                    target.identity if target is not None else None
                ),
                "target_participant_sid": target.sid if target is not None else None,
                "target_audio_track_sid": (
                    target.audio_track_sid if target is not None else None
                ),
                "target_participant_attributes": (
                    dict(target.attributes) if target is not None else {}
                ),
                "cleanup_status": "failed" if cleanup_errors else "completed",
                "cleanup_errors": cleanup_errors,
                "sip_dispatch_rule_id": sip_dispatch_rule_id,
                "sip_dispatch_rule_created": sip_dispatch_rule_created,
                "target_provider": (
                    provider_target.provider if provider_target is not None else None
                ),
                "provider_call_id": provider_call_id,
                "reconciled_call_ids": reconciled_call_ids,
                "caller_verification": caller_verification,
                "vapi_call_id": (
                    provider_call_id
                    if profile.evidence_provider == "vapi"
                    or transport.inbound_call_originator == "vapi"
                    else None
                ),
                "retell_call_id": (
                    provider_call_id
                    if profile.evidence_provider == "retell"
                    or transport.inbound_call_originator == "retell"
                    else None
                ),
                "simulator_model_usage": (
                    customer_agent.model_usage
                    if customer_agent is not None
                    and hasattr(customer_agent, "model_usage")
                    else []
                ),
            }
        )
        logger.info(
            "livekit_case_outcome status=%s stop_reason=%s failure=%s run=%s case=%s",
            outcome.status.value,
            outcome.metadata.get("stop_reason"),
            outcome.failure.code if outcome.failure is not None else None,
            run_id,
            test_case_id,
        )
        return outcome

    async def _create_customer_agent(
        self,
        persona: Persona,
        simulator: SimulatorAgentDefinition | None,
        *,
        call_type: CallType = "inbound",
        agent_name: str | None = None,
        min_turn_messages: int = 0,
    ) -> tuple[_TestRunnerAgent, LiveKitModels]:
        customer_prompt = build_voice_simulator_prompt(
            persona,
            call_type=call_type,
            agent_name=agent_name,
            additional_instructions=(
                simulator.instructions if simulator is not None else None
            ),
            default_language=(
                simulator.stt.language if simulator is not None else None
            ),
            variables={"instruction": persona.situation or ""},
            # Delivery cues are Cartesia only. Passing the provider here rather than reading it
            # inside the prompt keeps the decision where the provider is actually known.
            tts_provider=(simulator.tts.provider if simulator is not None else None),
        )
        if simulator is None:
            voice_provider = os.environ.get(
                "SIMULATOR_VOICE_PROVIDER", "openai"
            ).lower()
            llm_config = _default_simulator_llm_config()
            stt_config = STTConfig(
                provider=voice_provider,
                model=os.environ.get("SIMULATOR_STT_MODEL", "gpt-4o-mini-transcribe"),
            )
            tts_config = TTSConfig(
                provider=voice_provider,
                model=os.environ.get("SIMULATOR_TTS_MODEL", "gpt-4o-mini-tts"),
                voice=os.environ.get("SIMULATOR_TTS_VOICE_ID", "alloy"),
            )
            instructions = customer_prompt
            allow_interruptions = None
            min_endpointing_delay = None
            max_endpointing_delay = None
            use_aligned_transcript = None
        else:
            llm_config = simulator.llm
            stt_config = simulator.stt
            tts_config = simulator.tts
            instructions = customer_prompt
            allow_interruptions = simulator.allow_interruptions
            min_endpointing_delay = simulator.min_endpointing_delay
            max_endpointing_delay = simulator.max_endpointing_delay
            use_aligned_transcript = simulator.use_tts_aligned_transcript
        # Per-persona voice: a persona may carry a ``voice`` (or ``voice_id``)
        # attribute so different simulated customers sound different. Deepgram
        # encodes the voice in the model name (``aura-*``); every other provider
        # uses the dedicated ``voice`` field. Falls back to the simulator/global
        # default when the persona does not specify one.
        persona_attrs = getattr(persona, "persona", None)
        persona_voice = (
            (persona_attrs.get("voice") or persona_attrs.get("voice_id"))
            if isinstance(persona_attrs, dict)
            else None
        )
        if persona_voice:
            field = "model" if tts_config.provider == "deepgram" else "voice"
            tts_config = tts_config.model_copy(update={field: str(persona_voice)})
        models = await build_livekit_models(
            llm_config=llm_config,
            stt_config=stt_config,
            tts_config=tts_config,
        )
        vad = await asyncio.to_thread(_load_silero_vad_sync)
        self._last_simulator_setup = {
            "instructions": instructions,
            "llm_config": llm_config,
            "stt_config": stt_config,
            "tts_config": tts_config,
            "allow_interruptions": allow_interruptions,
            "min_endpointing_delay": min_endpointing_delay,
            "max_endpointing_delay": max_endpointing_delay,
            "use_tts_aligned_transcript": use_aligned_transcript,
        }
        agent = _TestRunnerAgent(
            persona=persona,
            min_turn_messages=min_turn_messages,
            stt=models.stt,
            llm=models.llm,
            tts=models.tts,
            vad=vad,
            instructions=instructions,
            turn_handling=_simulator_turn_handling(
                vad=vad,
                allow_interruptions=allow_interruptions,
                min_endpointing_delay=min_endpointing_delay,
                max_endpointing_delay=max_endpointing_delay,
            ),
            use_tts_aligned_transcript=use_aligned_transcript,
        )
        return agent, models


async def _wait_for_target_audio(
    room: rtc.Room,
    *,
    excluded_identities: set[str],
    target_identity: str | None,
    timeout: float,
) -> _TargetParticipant:
    ready = asyncio.Event()
    selected: _TargetParticipant | None = None

    def inspect_room(*_args) -> None:
        nonlocal selected
        selected = _find_target_audio(
            room,
            excluded_identities=excluded_identities,
            target_identity=target_identity,
        )
        if selected is not None:
            ready.set()

    room.on("participant_connected", inspect_room)
    room.on("track_published", inspect_room)
    room.on("track_subscribed", inspect_room)
    inspect_room()
    try:
        await asyncio.wait_for(ready.wait(), timeout=timeout)
    finally:
        _remove_room_listener(room, "participant_connected", inspect_room)
        _remove_room_listener(room, "track_published", inspect_room)
        _remove_room_listener(room, "track_subscribed", inspect_room)
    if selected is None:
        raise asyncio.TimeoutError
    return selected


async def _forward_target_transcription(
    reader: "rtc.TextStreamReader",
    session: "AgentSession",
    *,
    conversation_ended: "asyncio.Event | None" = None,
    captured_target_turns: list[dict[str, Any]] | None = None,
) -> None:
    # Receiver-side wall clock — same clock domain as the simulator's
    # ChatMessage.metrics, and the target's transcript IO is playback-synced
    # (TranscriptSynchronizer), so stream-open ~= speech start and read_all()
    # completion ~= speech end. Timestamps embedded in the stream are the
    # sender's (laptop) clock; skew there would corrupt the derived latencies.
    started_at = time.time()
    try:
        transcript = (await reader.read_all()).strip()
        stopped_at = time.time()
        if not transcript:
            return
        # Capture the target's turn independently of the simulator session FIRST.
        # Once the session drains it rejects new input ("speech scheduling is
        # paused"), so a closing delivered after the simulator is done never
        # reaches the chat context. This list is merged into the report so the
        # trailing target turn survives regardless of session state.
        if captured_target_turns is not None:
            captured_target_turns.append(
                {
                    "content": transcript,
                    "started_speaking_at": started_at,
                    "stopped_speaking_at": stopped_at,
                }
            )
        # Only elicit a simulator response while the conversation is live; once
        # it has ended the target's turn is recorded but the simulator stays
        # silent. The turn MUST travel through ``generate_reply(user_input=...)``:
        # the reply pipeline reads the agent's own chat context, not
        # ``session.history``, so a turn only added to the history is invisible
        # to the simulator LLM (it answers as if it heard nothing). The pipeline
        # then persists the message into both contexts once the reply schedules.
        if conversation_ended is None or not conversation_ended.is_set():
            try:
                session.generate_reply(user_input=transcript)
            except RuntimeError:
                # Session is already closing; the turn is captured above.
                pass
            else:
                return
        # Conversation over (or the session rejected the reply): record the
        # turn on the transcript without eliciting a response.
        try:
            session.history.add_message(role="user", content=transcript)
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to consume target transcription stream",
            exc_info=redacted_exc_info(exc),
        )


def _find_target_audio(
    room: rtc.Room,
    *,
    excluded_identities: set[str],
    target_identity: str | None,
) -> _TargetParticipant | None:
    candidates: list[tuple[int, int, _TargetParticipant]] = []
    agent_kind = getattr(
        rtc.ParticipantKind,
        "PARTICIPANT_KIND_AGENT",
        None,
    )
    for participant in room.remote_participants.values():
        identity = str(participant.identity)
        if identity in excluded_identities:
            continue
        if target_identity is not None and identity != target_identity:
            continue
        priority = 0 if getattr(participant, "kind", None) == agent_kind else 1
        for publication in participant.track_publications.values():
            if getattr(publication, "kind", None) != rtc.TrackKind.KIND_AUDIO:
                continue
            # Agents may publish ambient music/noise alongside their synthesized speech. The
            # first LiveKit publication is not necessarily the conversational track (the
            # official drive-thru example publishes ``background_audio``). Prefer ordinary
            # speech/microphone tracks so STT, transcription filtering, and recording all bind
            # to the same semantic stream.
            track_name = str(getattr(publication, "name", "") or "").lower()
            background = any(
                marker in track_name
                for marker in ("background", "ambient", "music", "sound_effect")
            )
            track_priority = 1 if background else 0
            attrs = dict(getattr(participant, "attributes", {}) or {})
            candidates.append(
                (
                    priority,
                    track_priority,
                    _TargetParticipant(
                        identity=identity,
                        sid=str(participant.sid),
                        audio_track_sid=str(publication.sid),
                        attributes={
                            str(key): str(value) for key, value in attrs.items()
                        },
                    ),
                )
            )
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            item[0],
            item[1],
            item[2].identity,
            item[2].audio_track_sid,
        ),
    )[0][2]


# A run ends naturally when the simulator calls ``endCall``; this is only the
# backstop for a conversation that has genuinely stalled or already finished but
# never hung up. Kept long so normal turn-gaps (STT endpoint + LLM + TTS latency)
# never trip it — the run is never cut off at a message count.
_SILENCE_BACKSTOP_SECONDS = 90.0

# Mutual silence in a conversation both sides joined is a finished call, not a stalled one. A
# thinking agent is working rather than silent, so the timer holds while either side is busy: no
# fixed window fits both a 4.3s and a 25.2s reply. The measured fallback covers providers that
# report no thinking state, stretching to the slowest reply this call has seen.
_SETTLED_SILENCE_FLOOR_SECONDS = 12.0
_SETTLED_LATENCY_MULTIPLE = 2.0

# LiveKit reports OUR SIMULATED CALLER as "assistant" and the TARGET AGENT as "user", because the
# caller is this session's agent and the target connects as the remote party. The published
# transcript swaps them (see _canonical_report_messages), so session-native code must never reuse
# the published convention. Named here because reading it the wrong way round is silent: a check
# still runs, still passes its tests, and watches the wrong side of the call.
_CALLER = "assistant"
_TARGET = "user"

# AgentState describes THIS SESSION'S AGENT, our caller; UserState describes the target. UserState
# has no "thinking", so this pair stops us cutting off our own caller mid-thought and cannot see a
# target composing a reply. The measured window below is what protects a slow target.
_AGENT_BUSY_STATES = frozenset({"initializing", "thinking", "speaking"})
_USER_BUSY_STATES = frozenset({"speaking"})


def _either_side_busy(session: Any) -> bool:
    """Whether work is in flight, as opposed to a conversation that has gone quiet."""
    return (
        getattr(session, "agent_state", None) in _AGENT_BUSY_STATES
        or getattr(session, "user_state", None) in _USER_BUSY_STATES
    )


def _settled_silence_window(observed_agent_reply: float, backstop_seconds: float) -> float:
    """How long silence must last before a settled call is treated as over."""
    return min(
        backstop_seconds,
        max(_SETTLED_SILENCE_FLOOR_SECONDS, observed_agent_reply * _SETTLED_LATENCY_MULTIPLE),
    )


def _turn_gap_seconds(
    previous: dict[str, Any], current: dict[str, Any]
) -> float | None:
    """Silence between one turn finishing and the next starting, in seconds.

    Uses the real audio timing the transport reports and falls back to the wall-clock stamp for
    text-only turns. Returns None when neither side is timed, so a caller can tell "no gap" from
    "not measurable" rather than reading an absent measurement as zero.
    """
    start = current.get("started_speaking_at") or current.get("created_at") or None
    end = previous.get("stopped_speaking_at") or previous.get("created_at") or None
    if not start or not end:
        return None
    gap = float(start) - float(end)
    return gap if gap >= 0 else None


def _observed_agent_reply_seconds(messages: list[dict[str, Any]]) -> float:
    """The slowest reply this agent has actually produced on this call.

    Read from the transport rather than tracked against the poll loop's own clock, so it is also
    correct for history that arrives in bulk (a resume, a reconnect) where there was no live
    transition to observe. Prefers LiveKit's reported end-to-end latency and falls back to the gap
    between the caller finishing and the agent starting, which is the wait a listener would hear.
    """
    slowest = 0.0
    previous: dict[str, Any] | None = None
    for message in messages:
        if not message.get("content"):
            continue
        if message.get("role") == _TARGET:
            reported = message.get("e2e_latency")
            if reported:
                slowest = max(slowest, float(reported))
            elif previous is not None and previous.get("role") == _CALLER:
                gap = _turn_gap_seconds(previous, message)
                if gap is not None:
                    slowest = max(slowest, gap)
        previous = message
    return slowest


async def _wait_for_conversation_end(
    room: rtc.Room,
    session: AgentSession,
    *,
    customer_agent: _TestRunnerAgent,
    target_identity: str,
    timeout: float,
    conversation_direction: str,
    agent_first_silence_timeout_seconds: float,
    provider_task: asyncio.Task[None] | None = None,
) -> str:
    closed = asyncio.Event()
    target_disconnected = asyncio.Event()
    room_disconnected = asyncio.Event()

    def on_close(_event) -> None:
        closed.set()

    def on_participant_disconnected(participant) -> None:
        if str(participant.identity) == target_identity:
            target_disconnected.set()

    def on_room_disconnected(*_args) -> None:
        room_disconnected.set()

    session.on("close", on_close)
    room.on("participant_disconnected", on_participant_disconnected)
    # A native target commonly hangs up by DELETING the room (the LiveKit
    # hangup recipe); the simulator then sees a room disconnect, not a
    # participant_disconnected, and without this watcher the case idled
    # through the silence backstop before ending.
    room.on("disconnected", on_room_disconnected)
    # The target may have left in the gap between readiness and this
    # registration — the event is gone, so recheck presence once.
    remote_participants = getattr(room, "remote_participants", None)
    if isinstance(remote_participants, dict) and not any(
        str(participant.identity) == target_identity
        for participant in remote_participants.values()
    ):
        target_disconnected.set()
    tasks = {
        "closed": asyncio.create_task(closed.wait()),
        "target_disconnected": asyncio.create_task(target_disconnected.wait()),
        "room_disconnected": asyncio.create_task(room_disconnected.wait()),
        "simulator_end_call": asyncio.create_task(customer_agent.end_requested.wait()),
        "conversation_stalled": asyncio.create_task(
            _wait_for_conversation_silence(
                session,
                # A stub agent in a test carries no floor; absent means never settle early.
                min_turn_messages=int(
                    getattr(customer_agent, "_min_turn_messages", 0) or 0
                ),
            )
        ),
        "closing_loop": asyncio.create_task(_wait_for_closing_loop(session)),
        "no_conversation": asyncio.create_task(
            _wait_for_conversation_never_started(
                session,
                timeout_seconds=_NO_CONVERSATION_TIMEOUT_SECONDS,
            )
        ),
    }
    if conversation_direction == "agent_first":
        tasks["conversation_silence_timeout"] = asyncio.create_task(
            _wait_for_agent_first_silence(
                session,
                timeout_seconds=agent_first_silence_timeout_seconds,
            )
        )
    if provider_task is not None:
        tasks["provider_disconnected"] = provider_task
    try:
        done, pending = await asyncio.wait(
            set(tasks.values()),
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        owned_pending = {
            task
            for name, task in tasks.items()
            if task in pending and name != "provider_disconnected"
        }
        for task in owned_pending:
            task.cancel()
        if owned_pending:
            await asyncio.gather(*owned_pending, return_exceptions=True)
        if not done:
            return "timeout"
        # A crashed monitor is also "done"; it must not count as its condition.
        completed: set[str] = set()
        monitor_failures: dict[str, BaseException] = {}
        for name, task in tasks.items():
            if task not in done or task.cancelled():
                continue
            exc = task.exception()
            if exc is None:
                completed.add(name)
            else:
                monitor_failures[name] = exc
        for name, exc in monitor_failures.items():
            logger.warning(
                "conversation end monitor failed",
                exc_info=redacted_exc_info(exc),
                extra={"monitor": name, "target_identity": target_identity},
            )
        # A bridge task error is still a real provider-side disconnect.
        if "provider_disconnected" in monitor_failures:
            completed.add("provider_disconnected")
        for reason in (
            "simulator_end_call",
            "target_disconnected",
            "room_disconnected",
            "no_conversation",
            # A farewell loop is a finished conversation, so it outranks the silence backstop
            # that would otherwise report the same call as a stall.
            "closing_loop",
            "conversation_silence_timeout",
            "conversation_stalled",
            "provider_disconnected",
            "closed",
        ):
            if reason in completed:
                return "session_closed" if reason == "closed" else reason
        if monitor_failures:
            return "monitor_failed"
        return "session_closed"
    finally:
        _remove_room_listener(session, "close", on_close)
        _remove_room_listener(
            room,
            "participant_disconnected",
            on_participant_disconnected,
        )
        _remove_room_listener(room, "disconnected", on_room_disconnected)


_CLOSING_PHRASES = (
    "goodbye",
    "bye",
    "take care",
    "have a great day",
    "have a good day",
    "have a wonderful day",
    "you too",
)

_CLOSING_EXCHANGE_LIMIT = 4


# Words a farewell is allowed to be made of. Anything outside this set is substance, whatever the
# turn's length: "yes it is, bye" is an answer and ending on it would cut a live call short.
_CLOSING_FILLER = frozenset(
    """
    a again alright and bye byebye care cheers day drive evening fine good goodbye great
    have later lovely morning much nice night ok okay perfect right safe see so soon sounds
    speak sure take talk thank thanks then to tomorrow too well wonderful you your
    """.split()
)


def _is_closing_only(text: str) -> bool:
    """Whether a turn is nothing but a farewell.

    Deliberately narrow: a turn that closes AND carries anything else (a question, a fact, a
    correction) is still conversation, and ending on it would cut a live call short.

    Decided on whether every word is farewell filler rather than on a word count. A cap of six
    words classified "Sounds great, thanks. Talk tomorrow. Bye." as a farewell and "Sounds good,
    talk to you then. Bye." as conversation, purely because the second has one more word, and the
    engine then asked the caller for two further turns and got two more goodbyes.
    """
    stripped = "".join(
        character.lower() if character.isalnum() or character.isspace() else " "
        for character in (text or "")
    ).split()
    if not stripped or len(stripped) > 12:
        return False
    joined = " ".join(stripped)
    if not any(phrase in joined for phrase in _CLOSING_PHRASES):
        return False
    return not (set(stripped) - _CLOSING_FILLER)


def _stop_any_further_speech(session: Any) -> None:
    """Cancel anything already in flight, so the farewell is the last thing said.

    Noticing the farewell only stops us asking for the NEXT turn. A reply already being generated
    still plays, which is how "Take care." arrived after a correct goodbye on a measured call. The
    farewell itself is already in history, meaning its own audio finished, so there is nothing of
    the caller's left to cut off here.
    """
    try:
        session.interrupt(force=True)
    except Exception:  # noqa: BLE001 - nothing in flight, or a session already shutting down
        logger.debug("nothing to interrupt when the call was closed", exc_info=True)


async def _wait_for_closing_loop(
    session: AgentSession,
    *,
    limit: int = _CLOSING_EXCHANGE_LIMIT,
) -> None:
    """Finish once both sides are only trading farewells.

    A simulator that does not reach for ``endCall`` leaves the target answering goodbye with
    goodbye until the deadline. One such call ran seventy-six turns, held its worker past the
    world pool's patience and cost the rest of that job its worlds, so this ends the call on the
    evidence already in the transcript rather than waiting for a timeout that arrives too late.
    """
    while True:
        messages = _session_messages(session)
        spoken = [
            message for message in messages if (message.get("content") or "").strip()
        ]
        # The caller's own farewell is the end of the call from its side, so there is no reason to
        # ask it for another turn. Waiting for a loop of farewells is what produced "Talk
        # tomorrow. Bye." followed by "Take care." and then "Bye." -- three closings where the
        # first was already correct. Rule 10 of the caller's prompt says exactly this, and an
        # instruction cannot enforce it: the model only speaks again because it was asked to.
        if (
            _turns_from_each_side(spoken) >= 1
            and spoken
            and spoken[-1].get("role") == _CALLER
            and _is_closing_only(str(spoken[-1].get("content") or ""))
        ):
            logger.info("the caller said goodbye, ending the call")
            _stop_any_further_speech(session)
            return
        tail = spoken[-limit:]
        if len(tail) == limit and all(
            _is_closing_only(str(message.get("content") or "")) for message in tail
        ):
            logger.info(
                "closing loop: last %d turns were farewells only, ending the call",
                limit,
            )
            _stop_any_further_speech(session)
            return
        # A turn lands in history only after its TTS finishes, so every poll interval between the
        # farewell committing and this noticing is time in which the caller can be asked for
        # another turn. Measured: one trailing turn survived at a one-second poll.
        await asyncio.sleep(0.25)


async def _wait_for_conversation_silence(
    session: AgentSession,
    *,
    quiet_seconds: float = _SILENCE_BACKSTOP_SECONDS,
    min_turn_messages: int = 0,
) -> None:
    """Finish only after a long, genuine stretch of mutual silence.

    The simulator ends a call by calling ``endCall`` once the scenario is done;
    this is only the backstop for a conversation that has actually stalled (or
    already finished but never hung up). It deliberately does **not** look at the
    message count — a run is never cut off at a floor, it runs as long as turns
    keep flowing. The timer resets on every new message and while either side is
    speaking, so only a real ``quiet_seconds`` gap of nothing ends the call.

    Parks until the first non-empty turn: a call where nobody ever spoke is the
    ``no_conversation`` monitor's condition, and this backstop firing first
    mislabeled dead calls as merely settled.
    """
    last_signature: tuple[tuple[str, str], ...] | None = None
    stable_since: float | None = None
    loop = asyncio.get_running_loop()
    while True:
        messages = _session_messages(session)
        signature = tuple((message["role"], message["content"]) for message in messages)
        if not any(message["content"] for message in messages):
            last_signature = signature
            stable_since = None
            await asyncio.sleep(0.1)
            continue
        now = loop.time()
        participant_busy = _either_side_busy(session)
        floor, _ = _turn_requirements(min_turn_messages)
        # Far enough in for the measured window to beat the fixed one. A third of the floor is a
        # threshold, not a derived figure: enough turns to have timed a reply, well short of done.
        settled = min_turn_messages > 0 and _turns_from_each_side(messages) >= max(2, floor // 3)
        effective_quiet = (
            _settled_silence_window(_observed_agent_reply_seconds(messages), quiet_seconds)
            if settled
            else quiet_seconds
        )
        if participant_busy:
            stable_since = None
        elif stable_since is None or signature != last_signature:
            stable_since = now
        elif now - stable_since >= effective_quiet:
            return
        last_signature = signature
        await asyncio.sleep(0.1)


async def _wait_for_conversation_never_started(
    session: AgentSession,
    *,
    timeout_seconds: float,
) -> None:
    """Completes only when no non-empty turn has ever been committed; parks
    forever (until cancelled) once the conversation has actually started."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while loop.time() < deadline:
        if any(message["content"] for message in _session_messages(session)):
            await asyncio.Event().wait()
        await asyncio.sleep(0.5)


def _voicemail_tone_style() -> str:
    """The style whose tone this call plays; empty for a person, and for a full mailbox by design."""
    if not _answered_by_voicemail():
        return ""
    style = (
        os.environ.get("HARNESS_VOICEMAIL_STYLE", "").strip().lower()
        or _DEFAULT_VOICEMAIL_STYLE
    )
    return style if style in _VOICEMAIL_TONE_BY_STYLE else ""


def _downloaded_audio(source: str) -> str | None:
    """A local copy of a remote audio file, or None: a call heard in the clear beats a dropped one."""
    import tempfile
    import urllib.request

    try:
        suffix = ".mp3" if ".mp3" in source else ".ogg" if ".ogg" in source else ".wav"
        with urllib.request.urlopen(source, timeout=15) as response:
            data = response.read()
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        handle.write(data)
        handle.close()
        return handle.name
    except Exception:
        return None


def _frame_at_mixer_rate(frame: "rtc.AudioFrame") -> "rtc.AudioFrame":
    """The same audio at the mixer's rate, which reinterprets rather than resamples what it is given."""
    if frame.sample_rate == _BACKGROUND_MIXER_RATE:
        return frame
    resampler = _MIXER_RESAMPLERS.get((frame.sample_rate, frame.num_channels))
    if resampler is None:
        resampler = PCMResampler(
            from_rate=frame.sample_rate,
            to_rate=_BACKGROUND_MIXER_RATE,
            channels=frame.num_channels,
        )
        _MIXER_RESAMPLERS[(frame.sample_rate, frame.num_channels)] = resampler
    converted = resampler.convert(bytes(frame.data))
    return rtc.AudioFrame(
        data=converted,
        sample_rate=_BACKGROUND_MIXER_RATE,
        num_channels=frame.num_channels,
        samples_per_channel=len(converted) // (2 * frame.num_channels),
    )


def _tone_frame(hz: float, seconds: float) -> "rtc.AudioFrame":
    """One frame of sine, faded in and out: a burst at full amplitude clicks and a detector hears the click."""
    total = int(_BACKGROUND_MIXER_RATE * seconds)
    fade = max(1, int(_BACKGROUND_MIXER_RATE * 0.01))
    samples = array.array("h")
    for index in range(total):
        gain = min(1.0, index / fade, (total - index) / fade)
        samples.append(
            int(
                32767
                * 0.9
                * gain
                * math.sin(2 * math.pi * hz * index / _BACKGROUND_MIXER_RATE)
            )
        )
    return rtc.AudioFrame(
        data=samples.tobytes(),
        sample_rate=_BACKGROUND_MIXER_RATE,
        num_channels=1,
        samples_per_channel=total,
    )


def _answered_by_voicemail() -> bool:
    """Whether a mailbox answered rather than a person; set per scenario by the call runner."""
    return os.environ.get("HARNESS_ANSWERED_BY", "").strip().lower() == "voicemail"


async def _open_if_nobody_speaks_first(
    session: AgentSession,
    customer_agent: Any,
    *,
    timeout_seconds: float,
) -> None:
    """Have the simulated person open the conversation when the other side never does.

    Only for a call the agent was supposed to start. It opens exactly the way a simulator-first call
    does, through ``open_conversation``, so the person's own initial message is used where the
    persona has one. Returns as soon as anybody speaks, which is the ordinary case.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while loop.time() < deadline:
        if any(message["content"] for message in _session_messages(session)):
            return
        await asyncio.sleep(0.2)
    if any(message["content"] for message in _session_messages(session)):
        return
    logger.warning(
        "no first turn after %ss; the simulated person opens instead", timeout_seconds
    )
    try:
        customer_agent.open_conversation()
    except Exception:  # noqa: BLE001 - a call that cannot be opened is the case's own failure
        logger.warning("the simulated person could not open the call", exc_info=True)


async def _wait_for_agent_first_silence(
    session: AgentSession,
    *,
    timeout_seconds: float,
) -> None:
    last_signature: tuple[tuple[str, str], ...] = ()
    last_change = asyncio.get_running_loop().time()
    while True:
        messages = _session_messages(session)
        signature = tuple((message["role"], message["content"]) for message in messages)
        # A turn lands in history only after its TTS finishes, so an in-flight utterance longer
        # than the timeout must count as activity -- and so must an agent that is still thinking,
        # or the caller opens over the top of a reply that was on its way.
        if signature != last_signature or _either_side_busy(session):
            last_signature = signature
            last_change = asyncio.get_running_loop().time()
        roles = {message["role"] for message in messages if message["content"]}
        if {"user", "assistant"}.issubset(
            roles
        ) and asyncio.get_running_loop().time() - last_change >= timeout_seconds:
            return
        await asyncio.sleep(0.1)


def _session_messages(session: AgentSession) -> list[dict[str, Any]]:
    """Return normalized transcript messages with real per-item speech timing.

    Each dict carries:
      role, content: str
      started_speaking_at, stopped_speaking_at: float | None
          Real audio timing from ``ChatMessage.metrics`` (seconds since epoch).
          See livekit.agents.llm.chat_context.MetricsReport.
      created_at: float
          Fallback wall-clock stamp from ``ChatMessage.created_at`` (used when
          the metrics timestamps are missing, e.g. text-only turns).
      interrupted: bool
      e2e_latency: float | None
          Agent-side turn latency, when reported by LiveKit.

    Downstream code turns these into millisecond offsets so the platform can
    recompute WPM, talk-ratio and interruption counts with real overlap data.
    """
    messages: list[dict[str, Any]] = []
    for item in session.history.items:
        if getattr(item, "type", None) != "message":
            continue
        role = getattr(item, "role", None)
        text = getattr(item, "text_content", None)
        if role is None or text is None:
            continue
        interrupted = bool(getattr(item, "interrupted", False))
        created_at = float(getattr(item, "created_at", 0.0) or 0.0)
        metrics = getattr(item, "metrics", None) or {}
        started_speaking_at = _maybe_float(metrics.get("started_speaking_at"))
        stopped_speaking_at = _maybe_float(metrics.get("stopped_speaking_at"))
        e2e_latency = _maybe_float(metrics.get("e2e_latency"))
        current: dict[str, Any] = {
            "role": str(role),
            "content": str(text),
            "created_at": created_at,
            "started_speaking_at": started_speaking_at,
            "stopped_speaking_at": stopped_speaking_at,
            "interrupted": interrupted,
            "e2e_latency": e2e_latency,
        }
        if messages and messages[-1]["role"] == current["role"]:
            previous = messages[-1]
            previous_text = previous["content"]
            if current["content"].startswith(previous_text):
                # Newer emission extends the previous partial — keep the
                # earliest start we saw, adopt the latest stop.
                current["started_speaking_at"] = (
                    previous.get("started_speaking_at")
                    or current["started_speaking_at"]
                )
                current["created_at"] = previous["created_at"] or created_at
                messages[-1] = current
            elif previous_text.startswith(current["content"]):
                previous["interrupted"] = previous.get("interrupted") or interrupted
                previous["stopped_speaking_at"] = (
                    previous.get("stopped_speaking_at")
                    or current["stopped_speaking_at"]
                )
            elif previous.get("interrupted") or interrupted:
                previous["content"] = f"{previous_text} {current['content']}".strip()
                previous["interrupted"] = interrupted
                previous["stopped_speaking_at"] = current[
                    "stopped_speaking_at"
                ] or previous.get("stopped_speaking_at")
            else:
                messages.append(current)
            continue
        messages.append(current)
    return messages


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _canonical_report_messages(session: AgentSession) -> list[dict[str, Any]]:
    """Emit report messages with roles remapped to the test-agent perspective.

    LiveKit reports our simulator as ``assistant`` and the target agent as
    ``user`` (the SDK connects with role ``agent``); we swap those so the
    downstream platform sees:
      role="user"      → simulator / customer
      role="assistant" → agent-under-test
    which matches the CallTranscript convention.

    Timing anchors (``started_speaking_at`` / ``stopped_speaking_at``) travel
    through unchanged so the platform can derive ms offsets.
    """
    role_map = {"assistant": "user", "user": "assistant"}
    messages: list[dict[str, Any]] = []
    for source in _session_messages(session):
        messages.append(
            {
                "role": role_map.get(source["role"], source["role"]),
                "content": source["content"],
                "created_at": source.get("created_at"),
                "started_speaking_at": source.get("started_speaking_at"),
                "stopped_speaking_at": source.get("stopped_speaking_at"),
                "interrupted": source.get("interrupted", False),
                "e2e_latency": source.get("e2e_latency"),
            }
        )
    return messages


def _merge_captured_target_turns(
    messages: list[dict[str, Any]],
    captured_target_turns: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Restore native target-turn timing, and append any target turn that never
    reached the session history.

    Native target turns are fed to the simulator via ``generate_reply(
    user_input=text)`` — a text input with no audio metrics — so their report
    entries carry a start but no ``stopped_speaking_at``: zero-duration turns
    that leave bot WPM, latency, and talk-ratio unpopulated. Each captured turn
    carries receiver-side wall-clock timing (see ``_forward_target_transcription``);
    here we (a) patch it onto the matching ``assistant`` turns missing a real
    stop, and (b) append the trailing turn delivered after the simulator drained
    ("speech scheduling is paused"). One turn can arrive as several partial or
    extended emissions, so match by containment and aggregate min-start/max-stop.
    Only populated by the native transcription handler — VAPI/Retell are untouched.
    """
    if not captured_target_turns:
        return messages

    def _matching(text: str) -> list[dict[str, Any]]:
        result = []
        for captured in captured_target_turns:
            cap_text = (captured.get("content") or "").strip()
            if cap_text and (text in cap_text or cap_text in text):
                result.append(captured)
        return result

    # (a) Fill timing onto existing assistant turns that lack a real stop; never
    #     override genuine audio metrics if livekit-agents ever populates them.
    for message in messages:
        if message.get("role") != "assistant":
            continue
        text = (message.get("content") or "").strip()
        if not text:
            continue
        started = message.get("started_speaking_at")
        stopped = message.get("stopped_speaking_at")
        if (
            isinstance(started, (int, float))
            and isinstance(stopped, (int, float))
            and stopped > started
        ):
            continue
        matched = _matching(text)
        starts = [
            c["started_speaking_at"]
            for c in matched
            if isinstance(c.get("started_speaking_at"), (int, float))
        ]
        stops = [
            c["stopped_speaking_at"]
            for c in matched
            if isinstance(c.get("stopped_speaking_at"), (int, float))
        ]
        if starts:
            message["started_speaking_at"] = min(starts)
            if not isinstance(message.get("created_at"), (int, float)):
                message["created_at"] = min(starts)
        if stops:
            message["stopped_speaking_at"] = max(stops)

    # (b) Append target turns that never reached the report at all.
    assistant_texts = [
        (m.get("content") or "").strip()
        for m in messages
        if m.get("role") == "assistant" and m.get("content")
    ]

    def _already_present(text: str) -> bool:
        return any(text in existing or existing in text for existing in assistant_texts)

    last_ts = 0.0
    for m in messages:
        for key in ("stopped_speaking_at", "started_speaking_at", "created_at"):
            value = m.get(key)
            if isinstance(value, (int, float)) and value > last_ts:
                last_ts = value

    merged = list(messages)
    for offset, captured in enumerate(captured_target_turns, start=1):
        text = (captured.get("content") or "").strip()
        if not text or _already_present(text):
            continue
        started = captured.get("started_speaking_at")
        if not isinstance(started, (int, float)):
            started = (last_ts + offset) if last_ts else None
        stopped = captured.get("stopped_speaking_at")
        if not isinstance(stopped, (int, float)):
            stopped = started
        merged.append(
            {
                "role": "assistant",
                "content": text,
                "created_at": started,
                "started_speaking_at": started,
                "stopped_speaking_at": stopped,
                "interrupted": False,
                "e2e_latency": None,
            }
        )
        assistant_texts.append(text)
    return merged


def _caller_never_spoke(messages: list[dict[str, Any]]) -> bool:
    """Whether the simulated caller's turns exist as text with no audio behind them.

    Speech synthesis that fails still leaves the caller's line in the transcript, so a mute
    simulator and a silent agent produce the same stall unless the missing audio is read directly.
    """

    def timed(message: dict[str, Any]) -> bool:
        return isinstance(message.get("started_speaking_at"), (int, float))

    spoken = [
        message
        for message in messages
        if message.get("role") == "user" and (message.get("content") or "").strip()
    ]
    if not spoken or any(timed(message) for message in spoken):
        return False
    # Only the agent's turns carrying timing makes the caller's missing timing evidence of
    # silence rather than a transcript that simply does not record when anyone spoke.
    return any(
        timed(message) for message in messages if message.get("role") == "assistant"
    )


def _recover_successful_provider_end_call(
    outcome: _CaseOutcome,
    provider_summary: EvidenceSourceSummary,
) -> None:
    """Keep a clean provider hangup separate from scenario correctness.

    Retell can disconnect immediately after its ``end_call`` tool succeeds, before the final
    synthesized farewell is committed into LiveKit's transcript.  A minimum-turn guard may have
    provisionally classified that as an incomplete conversation.  Provider evidence is the
    authoritative lifecycle signal here: promote the call to completed and let scenario checks
    report any business-goal failure.  Requiring both roles protects genuine mute/no-conversation
    failures from being hidden by a malformed provider trace.
    """
    if outcome.failure is None or outcome.failure.code not in {
        "insufficient_conversation",
        "target_disconnected",
        "room_disconnected",
    }:
        return
    if not _has_role_alternation(outcome.messages):
        return
    calls = provider_summary.metadata.get("tool_calls")
    if not isinstance(calls, list):
        return
    ended_cleanly = any(
        isinstance(call, dict)
        and str(call.get("name") or "").strip().lower() == "end_call"
        and call.get("ok") is not False
        for call in calls
    )
    if not ended_cleanly:
        return
    outcome.status = TestCaseStatus.COMPLETED
    outcome.failure = None
    outcome.metadata["provider_end_call_recovered"] = True


def _turn_requirements(min_turn_messages: int) -> tuple[int, bool]:
    """The turn floor and whether alternation is required: a mailbox is held to its greeting alone."""
    if _answered_by_voicemail():
        return min(min_turn_messages, _VOICEMAIL_MIN_TURN_MESSAGES), False
    return min_turn_messages, True


def _has_role_alternation(messages: list[dict[str, Any]]) -> bool:
    roles = {msg.get("role") for msg in messages if msg.get("content")}
    return "user" in roles and "assistant" in roles


def _turns_from_each_side(messages: list[dict[str, Any]]) -> int:
    """How many turns the quieter speaker took; a total is inflated by one side's own filler."""
    spoken = [msg for msg in messages if msg.get("content")]
    return min(
        sum(1 for msg in spoken if msg.get("role") == "assistant"),
        sum(1 for msg in spoken if msg.get("role") == "user"),
    )


# Two unanswered turns: one can be the caller finishing a thought, two means nobody is replying.
_QUIET_AFTER_UNANSWERED_TURNS = 2


def _target_has_gone_quiet(messages: list[dict[str, Any]]) -> bool:
    """Whether the agent has stopped replying, so the floor can never be reached honestly.

    The floor counts messages, and the caller's own turns count toward it, so a caller that is
    refused the tool talks to fill the silence and eventually buys its own permission. That is the
    opposite of what the floor is for. When the agent has spoken and then stopped, the caller is
    allowed to hang up instead.
    """
    spoken = [message for message in messages if message.get("content")]
    if not any(message.get("role") == "assistant" for message in spoken):
        return False
    trailing = 0
    for message in reversed(spoken):
        if message.get("role") != "user":
            break
        trailing += 1
    return trailing >= _QUIET_AFTER_UNANSWERED_TURNS


def _conversation_outcome(
    stop_reason: str,
    messages: list[dict[str, str]],
    *,
    min_turn_messages: int,
) -> _CaseOutcome:
    transcript = "\n".join(
        f"{message['role']}: {message['content']}" for message in messages
    )
    if (
        stop_reason in {"target_disconnected", "room_disconnected"}
        and _has_role_alternation(messages)
        and _has_natural_terminal_exchange(messages)
    ):
        # A provider target can deliberately end a short call before the generated
        # minimum-turn budget (for example, by accepting a caller's request to hang
        # up).  The transport still completed successfully.  Keep call lifecycle
        # separate from business-goal correctness: the scenario checks/evals decide
        # whether ending early was acceptable instead of reporting a false
        # connectivity failure.
        return _CaseOutcome(
            status=TestCaseStatus.COMPLETED,
            transcript=transcript,
            messages=messages,
            metadata={
                "stop_reason": stop_reason,
                "short_terminal_exchange": True,
            },
        )
    if (
        stop_reason == "conversation_silence_timeout"
        and len(messages) >= min_turn_messages
        and _has_role_alternation(messages)
        and _has_natural_terminal_exchange(messages)
    ):
        # Agent-first calls use a short silence watchdog because the tested
        # agent owns the opening turn.  A simulator can occasionally omit its
        # endCall tool even after both sides have clearly closed the call.  Do
        # not turn a fully recorded farewell/transfer into an infrastructure
        # failure merely because the now-idle room remained open.  Evaluation
        # still decides whether the agent actually completed the requested
        # business action.
        return _CaseOutcome(
            status=TestCaseStatus.COMPLETED,
            transcript=transcript,
            messages=messages,
            metadata={
                "stop_reason": stop_reason,
                "terminal_exchange_recovered": True,
            },
        )
    if stop_reason == "timeout":
        return _failure_outcome(
            TestCaseStatus.TIMED_OUT,
            FailureStage.RUNNING,
            "conversation_timeout",
            "Conversation exceeded its deadline",
            transcript=transcript,
            messages=messages,
            retryable=True,
        )
    if stop_reason == "conversation_silence_timeout" and _caller_never_spoke(messages):
        # The target sat in real silence because nothing was ever spoken at it. Retrying cannot
        # put a voice back on the line, so fail fast and name the synthesis rather than spending
        # the attempt budget reporting the agent as stalled.
        return _failure_outcome(
            TestCaseStatus.FAILED,
            FailureStage.RUNNING,
            "simulator_tts_silent",
            "Simulated caller produced transcript text but no audio",
            transcript=transcript,
            messages=messages,
            retryable=False,
            details={"stop_reason": stop_reason, "turn_count": str(len(messages))},
        )
    stalled = {
        "conversation_silence_timeout",
        "conversation_stalled",
        "session_closed",
        "no_conversation",
        "monitor_failed",
    }
    if _answered_by_voicemail():
        # Silence after a mailbox greeting is the call's natural end, not a stall.
        stalled.discard("conversation_silence_timeout")
    if stop_reason in stalled:
        code = stop_reason
        message = {
            "conversation_silence_timeout": (
                "Agent-first conversation stalled after it began"
            ),
            "conversation_stalled": (
                "Conversation produced no new speech for the stall deadline"
            ),
            "session_closed": (
                "Conversation session closed before a natural end condition"
            ),
            "no_conversation": (
                "No conversation turns were committed before the inactivity deadline"
            ),
            "monitor_failed": (
                "Conversation end monitoring failed before a natural end condition"
            ),
        }[stop_reason]
        return _failure_outcome(
            TestCaseStatus.FAILED,
            FailureStage.RUNNING,
            code,
            message,
            transcript=transcript,
            messages=messages,
            retryable=True,
        )
    floor, alternation_required = _turn_requirements(min_turn_messages)
    if len(messages) < floor or (
        alternation_required and not _has_role_alternation(messages)
    ):
        code = (
            stop_reason
            if stop_reason in {"target_disconnected", "room_disconnected"}
            else "insufficient_conversation"
        )
        return _failure_outcome(
            TestCaseStatus.FAILED,
            FailureStage.RUNNING,
            code,
            "Conversation ended before the required alternating turns completed",
            transcript=transcript,
            messages=messages,
            retryable=stop_reason
            in {"target_disconnected", "room_disconnected", "session_closed"},
            details={
                "stop_reason": stop_reason,
                "turn_count": str(len(messages)),
                "minimum_turn_count": str(floor),
            },
        )
    return _CaseOutcome(
        status=TestCaseStatus.COMPLETED,
        transcript=transcript,
        messages=messages,
        metadata={"stop_reason": stop_reason},
    )


def _has_natural_terminal_exchange(messages: list[dict[str, str]]) -> bool:
    """Recognize only explicit terminal language near the end of a call.

    This deliberately avoids broad sentiment or short-answer heuristics.  A
    normal unanswered question must remain a silence failure.  The two safe
    cases are an explicit farewell, or a transfer handoff followed by the
    caller's acknowledgement.
    """
    tail = [
        (
            str(message.get("role") or "").lower(),
            str(message.get("content") or "").strip().lower(),
        )
        for message in messages[-4:]
        if str(message.get("content") or "").strip()
    ]
    if not tail:
        return False
    farewell_markers = (
        "goodbye",
        "bye",
        "take care",
        "have a great day",
        "have a good day",
        "have a nice day",
    )
    if any(marker in text for _role, text in tail for marker in farewell_markers):
        return True

    for index, (role, text) in enumerate(tail[:-1]):
        if role != "assistant" or "transfer" not in text:
            continue
        if not any(marker in text for marker in ("now", "connect", "please wait")):
            continue
        next_role, acknowledgement = tail[index + 1]
        if next_role == "user" and acknowledgement.rstrip(".! ") in {
            "ok",
            "okay",
            "alright",
            "please do",
            "thank you",
            "thanks",
        }:
            return True
    return False


def _failure_outcome(
    status: TestCaseStatus,
    stage: FailureStage,
    code: str,
    message: str,
    *,
    transcript: str = "",
    messages: list[dict[str, str]] | None = None,
    retryable: bool = False,
    details: dict[str, str] | None = None,
) -> _CaseOutcome:
    return _CaseOutcome(
        status=status,
        transcript=transcript,
        messages=messages or [],
        failure=SimulationFailure(
            stage=stage,
            code=code,
            message=message,
            retryable=retryable,
            provider="livekit",
            details=details or {},
        ),
    )


def _record_simulator_setup(
    case_directory: Path,
    *,
    persona: Persona,
    instructions: str,
    llm_config: Any,
    stt_config: Any,
    tts_config: Any,
    turn_handling: Any = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write the exact prompt and voice settings this call is about to use.

    Reconstructing either one afterwards from a transcript is guesswork, and the simulator's
    prompt is what decides how the caller behaves. Written before the call connects so it
    survives a run that dies mid-conversation.
    """

    def settings(config: Any) -> Any:
        if config is None:
            return None
        for method in ("model_dump", "dict"):
            dump = getattr(config, method, None)
            if callable(dump):
                try:
                    return dump()
                except Exception:  # noqa: BLE001 - never fail a call over logging
                    pass
        return str(config)

    try:
        case_directory.mkdir(parents=True, exist_ok=True)
        (case_directory / "simulator-prompt.txt").write_text(
            instructions or "", encoding="utf-8"
        )
        payload = {
            "persona": settings(persona),
            "simulator_system_prompt": instructions or "",
            "llm": settings(llm_config),
            "stt": settings(stt_config),
            "tts": settings(tts_config),
            "turn_handling": settings(turn_handling),
        }
        payload.update(extra or {})
        (case_directory / "simulator-setup.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except Exception as error:  # noqa: BLE001 - logging must never break a run
        logger.warning("could not record simulator setup: %s", error)


def _attach_recordings(
    outcome: _CaseOutcome,
    recorder: RoomRecorder,
    *,
    simulator_identity: str,
    target_identity: str | None,
    target_track_sid: str | None,
    case_directory: Path,
    sample_rate: int,
) -> None:
    simulator_paths = recorder.paths_for_participant(simulator_identity)
    target_paths = (
        recorder.paths_for_participant(
            target_identity,
            track_sid=target_track_sid,
        )
        if target_identity is not None
        else []
    )
    # ``audio_track_sid`` identifies the track used to establish target
    # readiness, but it is not necessarily the track that remains published for
    # the conversation.  Agents using ``BackgroundAudioPlayer`` publish more
    # than one audio track and can replace the initially selected publication.
    # Recording is evidence of the whole participant, so fall back to all of the
    # target participant's tracks instead of silently producing no artifact.
    if target_identity is not None and not target_paths:
        target_paths = recorder.paths_for_participant(target_identity)

    # The simulator normally publishes with ``simulator_identity``.  Retain a
    # conservative fallback for SDKs that expose the local publication under a
    # different participant identity: only use it when there is exactly one
    # non-target publishing participant, so another caller can never be folded
    # into the customer channel accidentally.
    if not simulator_paths:
        non_target_identities = {
            record.participant_identity
            for record in recorder.records
            if record.participant_identity != target_identity
        }
        if len(non_target_identities) == 1:
            simulator_paths = recorder.paths_for_participant(
                next(iter(non_target_identities))
            )
    audio_directory = case_directory / "audio"
    input_path = _collapse_recordings(
        simulator_paths,
        audio_directory / "simulator.wav",
        sample_rate=sample_rate,
    )
    output_path = _collapse_recordings(
        target_paths,
        audio_directory / "target.wav",
        sample_rate=sample_rate,
    )
    combined_path = mix_recordings(
        [path for path in (input_path, output_path) if path is not None],
        audio_directory / "combined.wav",
        sample_rate=sample_rate,
    )
    stereo_path = mix_recordings_stereo(
        [path for path in (input_path,) if path is not None],
        [path for path in (output_path,) if path is not None],
        audio_directory / "stereo.wav",
        sample_rate=sample_rate,
    )
    outcome.audio_input_path = str(input_path) if input_path is not None else None
    outcome.audio_output_path = str(output_path) if output_path is not None else None
    outcome.audio_combined_path = (
        str(combined_path) if combined_path is not None else None
    )
    outcome.audio_stereo_path = str(stereo_path) if stereo_path is not None else None
    outcome.metadata["recording_tracks"] = [
        {
            "participant_identity": record.participant_identity,
            "participant_sid": record.participant_sid,
            "track_sid": record.track_sid,
            "path": str(record.path),
            "start_offset_frames": record.start_offset_frames,
        }
        for record in recorder.records
    ]
    outcome.metadata["recording_diagnostics"] = {
        "simulator_identity": simulator_identity,
        "target_identity": target_identity,
        "target_track_sid": target_track_sid,
        "simulator_track_count": len(simulator_paths),
        "target_track_count": len(target_paths),
        "recorder_error_types": [type(error).__name__ for error in recorder.errors],
    }
    if combined_path is None:
        logger.warning(
            "LiveKit call completed without captured audio tracks",
            extra={
                "simulator_identity": simulator_identity,
                "target_identity": target_identity,
                "target_track_sid": target_track_sid,
                "recorded_track_count": len(recorder.records),
                "recorder_error_types": [
                    type(error).__name__ for error in recorder.errors
                ],
            },
        )
    speech_starts = [
        float(message["started_speaking_at"])
        for message in outcome.messages
        if isinstance(message.get("started_speaking_at"), (int, float))
    ]
    if recorder.recording_started_at is not None and speech_starts:
        outcome.metadata["recording_offset_ms"] = max(
            0,
            round((min(speech_starts) - recorder.recording_started_at) * 1000),
        )


def _collapse_recordings(
    paths: list[Path],
    destination: Path,
    *,
    sample_rate: int,
) -> Path | None:
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]
    return mix_recordings(paths, destination, sample_rate=sample_rate)


def _target_api_key(target: VoiceProviderTarget | None) -> str | None:
    if target is None:
        return None
    return os.environ.get(target.api_key_env) or None


def _target_evidence_base_url(target: VoiceProviderTarget | None) -> str | None:
    if isinstance(target, VapiTargetConfig):
        return str(target.api_base_url).rstrip("/")
    if isinstance(target, RetellTargetConfig):
        parsed = urlsplit(str(target.api_url))
        return f"{parsed.scheme}://{parsed.netloc}"
    return None


def _default_simulator_llm_config() -> LLMConfig:
    return LLMConfig(
        provider=os.environ.get("SIMULATOR_LLM_PROVIDER", "openai"),
        model=os.environ.get("SIMULATOR_LLM_MODEL", "gpt-4o-mini"),
        temperature=0.6,
    )


def _resolve_livekit_runtime(
    agent_definition: AgentDefinition,
    runtime: LiveKitSimulatorRuntime | None,
) -> LiveKitSimulatorRuntime:
    if runtime is not None:
        return runtime
    if agent_definition.url is None or not agent_definition.room_name:
        raise ValueError(
            "livekit_runtime_required: provide LiveKitSimulatorRuntime or legacy "
            "AgentDefinition url and room_name"
        )
    return LiveKitSimulatorRuntime(
        url=agent_definition.url,
        room_name=agent_definition.room_name,
        room_mode=agent_definition.room_mode,
    )


def _resolve_room_name(
    runtime: LiveKitSimulatorRuntime,
    *,
    run_id: str,
    test_case_id: str,
    index: int,
    invocation_id: str,
) -> str:
    rendered = runtime.room_name.format(
        run_id=run_id,
        test_case_id=test_case_id,
        index=index,
        invocation_id=invocation_id,
    )
    if runtime.room_mode == "external" or getattr(runtime, "room_name_verbatim", False):
        return rendered
    prefix = _SAFE_ROOM.sub("-", rendered).strip("-._") or "simulation"
    suffix_parts = []
    if invocation_id not in prefix:
        suffix_parts.append(invocation_id)
    if test_case_id not in prefix:
        suffix_parts.append(test_case_id[-12:])
    suffix = "-" + "-".join(suffix_parts) if suffix_parts else ""
    return f"{prefix[: 255 - len(suffix)]}{suffix}"


def _has_room_template(room_name: str) -> bool:
    return any(
        marker in room_name for marker in ("{run_id}", "{test_case_id}", "{index}")
    )


def _api_url(url: str) -> str:
    if url.startswith("wss://"):
        return "https://" + url.removeprefix("wss://")
    if url.startswith("ws://"):
        return "http://" + url.removeprefix("ws://")
    return url


def _remove_room_listener(room: rtc.Room, event: str, listener) -> None:
    try:
        room.off(event, listener)
    except (AttributeError, ValueError):
        logger.debug("LiveKit listener was already removed", extra={"event": event})


async def _close_agent_session(session: AgentSession, *, timeout: float) -> None:
    """Close a session without abandoning teardown on the event loop.

    A shielded, timed-out ``aclose`` used to keep running after the case had
    returned. Repeating that in a soak test accumulated SDK activities until
    the guest process failed. Graceful close gets a bounded opportunity; after
    that, cancel and reap it because room and process teardown are independent.
    """
    close_session = getattr(session, "aclose", None)
    if close_session is None:
        session.shutdown(drain=False)
        return
    close_task = asyncio.create_task(close_session())
    try:
        await asyncio.wait_for(asyncio.shield(close_task), timeout=timeout)
    except asyncio.TimeoutError:
        close_task.cancel()
        try:
            await asyncio.wait_for(close_task, timeout=1.0)
        except (Exception, asyncio.CancelledError):
            if not close_task.done():
                close_task.add_done_callback(_consume_background_task_result)
        raise


def _consume_background_task_result(task: asyncio.Task) -> None:
    try:
        task.result()
    except (Exception, asyncio.CancelledError):
        pass


def _is_not_found(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    return str(getattr(code, "value", code)).lower() in {
        "not_found",
        "404",
    }


def _record_cleanup_error(
    errors: list[str],
    exc: Exception,
    operation: str,
    run_id: str,
    test_case_id: str,
) -> None:
    errors.append(f"{operation}:{type(exc).__name__}")
    logger.error(
        "LiveKit cleanup operation failed",
        exc_info=redacted_exc_info(exc),
        extra={
            "run_id": run_id,
            "test_case_id": test_case_id,
            "operation": operation,
            "exception_type": type(exc).__name__,
        },
    )


_LIVEKIT_INBOUND_TRUNK_ENV = "LIVEKIT_INBOUND_TRUNK_ID"


def _safe_provider_error_details(
    exc: Exception, *, operation: str
) -> dict[str, object]:
    """Extract sanitized error attributes for report failures.

    Never returns the exception message; only structural fields that are
    known to be safe from LiveKit/Twirp exception classes.
    """

    code = getattr(exc, "code", None)
    if code is not None:
        code_value = getattr(code, "value", None)
        if code_value is None and not isinstance(code, (str, int)):
            code_value = str(code)
        else:
            code_value = code_value if code_value is not None else code
    else:
        code_value = None
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    details: dict[str, object] = {
        "operation": operation,
        "exception_type": type(exc).__name__,
    }
    if code_value is not None:
        details["provider_code"] = code_value
    if status is not None:
        try:
            details["http_status"] = int(status)
        except (TypeError, ValueError):
            details["http_status"] = str(status)
    metadata = getattr(exc, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("sip_status_code", "sip_status", "sip-code"):
            value = metadata.get(key)
            if value is not None:
                details["sip_status_code"] = str(value)
                break
    return details


async def _ensure_sip_inbound_dispatch(
    api_client: api.LiveKitAPI,
    *,
    transport: TelephonyTransport,
    room_name: str,
) -> tuple[str, bool]:
    """Return ``(sip_dispatch_rule_id, created_by_sdk)``.

    When ``transport.dispatch_rule_name`` is supplied the SDK verifies
    the rule exists and reuses it. Otherwise the SDK provisions a
    per-run direct rule bound to ``LIVEKIT_INBOUND_TRUNK_ID`` that routes
    incoming calls into ``room_name`` — the same room the local
    simulator has already joined — and returns its id so the caller can
    tear it down.
    """

    existing = await api_client.sip.list_sip_dispatch_rule(ListSIPDispatchRuleRequest())
    if transport.dispatch_rule_name:
        for rule in existing.items:
            if rule.name != transport.dispatch_rule_name:
                continue
            direct = (
                getattr(rule.rule, "dispatch_rule_direct", None) if rule.rule else None
            )
            direct_room = getattr(direct, "room_name", "") if direct is not None else ""
            if not direct_room:
                raise RuntimeError(
                    "sip_inbound_rule_mismatch: "
                    f"{transport.dispatch_rule_name} is not a direct rule"
                )
            if direct_room != room_name:
                raise RuntimeError(
                    "sip_inbound_rule_mismatch: "
                    f"{transport.dispatch_rule_name} targets a different room"
                )
            return rule.sip_dispatch_rule_id, False
        raise RuntimeError(f"sip_inbound_rule_missing: {transport.dispatch_rule_name}")
    trunk_id = os.environ.get(_LIVEKIT_INBOUND_TRUNK_ENV)
    if not trunk_id:
        raise RuntimeError(
            f"sip_inbound_trunk_missing: set {_LIVEKIT_INBOUND_TRUNK_ENV}"
        )
    for rule in existing.items:
        if trunk_id and trunk_id in rule.trunk_ids:
            raise RuntimeError(
                "sip_inbound_route_conflict: existing dispatch rule "
                f"{rule.sip_dispatch_rule_id} already covers this trunk"
            )
    rule_name = f"sim-inbound-{room_name[-24:]}"
    resp = await api_client.sip.create_sip_dispatch_rule(
        CreateSIPDispatchRuleRequest(
            rule=SIPDispatchRule(
                dispatch_rule_direct=SIPDispatchRuleDirect(
                    room_name=room_name,
                ),
            ),
            trunk_ids=[trunk_id],
            hide_phone_number=False,
            name=rule_name,
        )
    )
    return resp.sip_dispatch_rule_id, True


async def _ensure_room_absent(
    api_client: api.LiveKitAPI, room_name: str, *, poll_interval: float = 0.5
) -> int:
    """Make ``room_name`` absent before the caller (re-)creates it.

    The pool's dispatch rule stays live for the whole run, so an inbound call
    can re-create this room between our own delete and our next poll — hence
    the re-delete inside the loop rather than a single delete-then-poll. No
    internal deadline: the caller bounds this with ``asyncio.wait_for``.
    """

    try:
        await api_client.room.delete_room(api.DeleteRoomRequest(room=room_name))
    except Exception as exc:  # noqa: BLE001
        if not _is_not_found(exc):
            raise
    polls = 0
    while True:
        resp = await api_client.room.list_rooms(api.ListRoomsRequest(names=[room_name]))
        polls += 1
        if not resp.rooms:
            return polls
        try:
            await api_client.room.delete_room(api.DeleteRoomRequest(room=room_name))
        except Exception as exc:  # noqa: BLE001
            if not _is_not_found(exc):
                raise
        await asyncio.sleep(poll_interval if polls < 4 else 5.0)


def _unexpected_participants(
    room, *, simulator_identity: str, recorder_identity: str
) -> set[str]:
    return {str(p.identity) for p in room.remote_participants.values()} - {
        simulator_identity,
        recorder_identity,
    }


# LiveKit: the other party's number on a SIP participant (the caller, for an
# inbound call). sip.trunkPhoneNumber is OUR number — never use it.
_SIP_REMOTE_NUMBER_ATTRIBUTE = "sip.phoneNumber"


def _number_digits(value) -> str:
    text = str(value or "")
    if text.startswith("sip:"):
        text = text[len("sip:") :]
        text = text.split("@", 1)[0]
        text = text.split(";", 1)[0]
    return re.sub(r"\D", "", text)


def _caller_matches(attributes, expected: str | None) -> bool | None:
    if not expected:
        return None
    observed = attributes.get(_SIP_REMOTE_NUMBER_ATTRIBUTE) if attributes else None
    if not observed:
        return None
    expected_digits = _number_digits(expected)
    observed_digits = _number_digits(observed)
    if len(expected_digits) < 7 or len(observed_digits) < 7:
        return None
    longer, shorter = (
        (expected_digits, observed_digits)
        if len(expected_digits) >= len(observed_digits)
        else (observed_digits, expected_digits)
    )
    return longer.endswith(shorter)


class _LeasedRoomCallerMismatch(Exception):
    """Module-private: raised by the leased-room caller check (step 4a), caught
    by the case's top-level try so the post-try recordings/evidence/metadata
    block still runs for a call that was billed and answered."""


async def _delete_sip_dispatch_rule(api_client: api.LiveKitAPI, rule_id: str) -> None:
    await api_client.sip.delete_sip_dispatch_rule(
        DeleteSIPDispatchRuleRequest(sip_dispatch_rule_id=rule_id)
    )


async def _collect_provider_evidence(
    *,
    config: ProviderEvidenceConfig,
    transport: TelephonyTransport,
    run_id: str,
    test_case_id: str,
    case_directory: Path,
    started_at: datetime,
    target: _TargetParticipant | None,
    provider_call_id_hint: str | None = None,
    provider_api_key: str | None = None,
    provider_api_base_url: str | None = None,
    termination_source: str | None = None,
) -> tuple[EvidenceSourceSummary | None, list[ArtifactManifestEntry]]:
    call_id_hint = provider_call_id_hint
    caller_phone = transport.sip_number if transport.kind == "sip_outbound" else None
    callee_phone = transport.sip_call_to if transport.kind == "sip_outbound" else None
    if target is not None:
        if call_id_hint is None and config.participant_attribute:
            call_id_hint = target.attributes.get(config.participant_attribute)
        caller_phone = caller_phone or (
            target.attributes.get("sip.from")
            or target.attributes.get("sip.fromUser")
            or target.attributes.get("sip.callerNumber")
        )
        callee_phone = callee_phone or (
            target.attributes.get("sip.to")
            or target.attributes.get("sip.toUser")
            or target.attributes.get("sip.calledNumber")
        )
    context = EvidenceContext(
        run_id=run_id,
        test_case_id=test_case_id,
        case_directory=case_directory,
        started_at=started_at,
        call_id_hint=call_id_hint,
        caller_phone=caller_phone,
        callee_phone=callee_phone,
        termination_source=termination_source,
    )
    try:
        if config.provider == "vapi":
            adapter = VapiEvidenceSource(
                config,
                api_key=provider_api_key,
                api_base_url=provider_api_base_url,
            )
        elif config.provider == "retell":
            adapter = RetellEvidenceSource(
                config,
                api_key=provider_api_key,
                api_base_url=provider_api_base_url,
            )
        else:
            raise ProviderConfigError(
                f"unsupported_provider_evidence: {config.provider}"
            )
    except ProviderConfigError as exc:
        summary = EvidenceSourceSummary(
            source_id=f"{config.provider}:unconfigured",
            adapter=config.provider,
            evidence_class=_EVIDENCE_PROVIDER_REPORTED,
            available=False,
            redactions=["auth", "phone_e164"],
            metadata={"provider": config.provider, "reason": str(exc)},
        )
        return summary, []
    try:
        await adapter.connect(context)
        result: ProviderFetchResult = await adapter.fetch_final()
    except Exception as exc:  # noqa: BLE001 — provider failures are first-class evidence
        logger.warning(
            "Provider evidence adapter failed",
            exc_info=redacted_exc_info(exc),
            extra={
                "provider": config.provider,
                "run_id": run_id,
                "test_case_id": test_case_id,
            },
        )
        summary = EvidenceSourceSummary(
            source_id=f"{config.provider}:error",
            adapter=config.provider,
            evidence_class=_EVIDENCE_PROVIDER_REPORTED,
            available=False,
            redactions=["auth", "phone_e164"],
            metadata={
                "provider": config.provider,
                "reason": "adapter_exception",
                "exception_type": type(exc).__name__,
            },
        )
        return summary, []
    finally:
        try:
            await adapter.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Provider evidence adapter close failed",
                extra={
                    "provider": config.provider,
                    "exception_type": type(exc).__name__,
                },
            )
    return result.summary, result.artifacts


# Import lazily to avoid a module-import cycle with ProviderConfigError above.
from fi.simulate.evidence.base import EvidenceClass as _EvidenceClass  # noqa: E402

_EVIDENCE_PROVIDER_REPORTED = _EvidenceClass.PROVIDER_REPORTED
