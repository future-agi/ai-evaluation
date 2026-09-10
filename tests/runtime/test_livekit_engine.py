from __future__ import annotations

import asyncio
import json
import logging
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("livekit")

from fi.simulate.agent.definition import AgentDefinition, TelephonyTransport
from fi.simulate.recording.room_recorder import (
    RecordedTrack,
    mix_recordings,
    mix_recordings_stereo,
)
from fi.simulate.runtime import TestCaseStatus as CaseStatus
from fi.simulate.simulation import bridge as _bridge
from fi.simulate.simulation.engines import livekit
from fi.simulate.simulation.engines.livekit import (
    LiveKitEngine,
    _voice_max_case_concurrency,
)
from fi.simulate.simulation import livekit_models
from fi.simulate.simulation.models import Persona, Scenario


def _agent(**updates) -> AgentDefinition:
    values = {
        "name": "support-agent",
        "url": "wss://livekit.example.com",
        "room_name": "support-room",
        "system_prompt": "Help the caller.",
    }
    values.update(updates)
    return AgentDefinition(**values)


def _scenario(count: int = 1) -> Scenario:
    return Scenario(
        name="voice",
        dataset=[
            Persona(
                persona={"name": f"Caller {index}"},
                situation="I need help.",
                outcome="The issue is resolved.",
            )
            for index in range(count)
        ],
    )


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 8000) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.astype(np.int16).tobytes())


def test_simulator_identity_carries_fixture_phone_for_repository_agents() -> None:
    persona = Persona(
        persona={
            "name": "Noor",
            "metadata": {"caller_phone": "+1 (415) 555-0107"},
        },
        situation="Cancel my ride.",
        outcome="The ride is cancelled.",
    )
    assert (
        livekit._simulator_participant_identity(persona, "case_aaaaaaaaaaaa")
        == "fagi-simulator-phone-14155550107-aaaaaaaaaaaa"
    )


def test_simulator_identity_preserves_legacy_shape_without_valid_phone() -> None:
    persona = Persona(
        persona={"name": "Caller", "metadata": {"caller_phone": "unknown"}},
        situation="I need help.",
        outcome="The issue is resolved.",
    )
    assert (
        livekit._simulator_participant_identity(persona, "case_bbbbbbbbbbbb")
        == "fagi-simulator-bbbbbbbbbbbb"
    )


def test_managed_room_names_are_unique_per_run_and_case() -> None:
    agent = _agent(room_mode="managed", agent_name="support-agent")

    first = livekit._resolve_room_name(
        agent,
        run_id="run_a",
        test_case_id="case_aaaaaaaaaaaa",
        index=0,
        invocation_id="invocation-a",
    )
    second = livekit._resolve_room_name(
        agent,
        run_id="run_a",
        test_case_id="case_bbbbbbbbbbbb",
        index=1,
        invocation_id="invocation-a",
    )

    assert first != second
    assert first.startswith("support-room-")


def test_external_multi_case_run_requires_room_template() -> None:
    with pytest.raises(ValueError, match="external_room_template_required"):
        asyncio.run(
            LiveKitEngine().run(
                agent_definition=_agent(),
                scenario=_scenario(2),
            )
        )


def test_verbatim_room_name_requires_serial_cases(monkeypatch) -> None:
    # A stray env override of the max-case-concurrency default would let this
    # test pass for the wrong reason (concurrency already 1), so make it explicit.
    monkeypatch.delenv("ALK_VOICE_MAX_CASE_CONCURRENCY", raising=False)
    runtime = livekit.LiveKitSimulatorRuntime(
        url="wss://livekit.example.com",
        room_name="sim-slot-03",
        room_mode="managed",
        room_name_verbatim=True,
    )

    with pytest.raises(
        ValueError,
        match="room_name_verbatim_requires_serial_cases",
    ):
        asyncio.run(
            LiveKitEngine().run(
                agent_definition=_agent(),
                livekit_runtime=runtime,
                scenario=_scenario(2),
                max_concurrency=2,
            )
        )


def test_verbatim_sip_inbound_multi_persona_reuses_the_same_room(
    monkeypatch,
) -> None:
    runtime = livekit.LiveKitSimulatorRuntime(
        url="wss://livekit.example.com",
        room_name="sim-slot-03",
        room_mode="managed",
        room_name_verbatim=True,
    )
    agent = _agent(
        room_mode="managed", transport=TelephonyTransport(kind="sip_inbound")
    )
    engine = LiveKitEngine()
    captured_room_names: list[str] = []

    async def _fake_run_case(*_args, **kwargs):
        captured_room_names.append(kwargs["room_name"])
        return livekit._CaseOutcome(status=CaseStatus.COMPLETED)

    monkeypatch.setattr(engine, "_run_single_test_case", _fake_run_case)

    report = asyncio.run(
        engine.run(
            agent_definition=agent,
            livekit_runtime=runtime,
            scenario=_scenario(3),
            max_concurrency=2,
        )
    )

    assert captured_room_names == ["sim-slot-03"] * 3
    assert len(report.results) == 3


def test_verbatim_multi_persona_sip_inbound_skips_room_template_requirement(
    monkeypatch,
) -> None:
    runtime = livekit.LiveKitSimulatorRuntime(
        url="wss://livekit.example.com",
        room_name="sim-slot-03",  # no {test_case_id}/{run_id} template
        room_mode="managed",
        room_name_verbatim=True,
    )
    agent = _agent(
        room_mode="managed", transport=TelephonyTransport(kind="sip_inbound")
    )
    engine = LiveKitEngine()

    async def _fake_run_case(*_args, **kwargs):
        return livekit._CaseOutcome(status=CaseStatus.COMPLETED)

    monkeypatch.setattr(engine, "_run_single_test_case", _fake_run_case)

    try:
        report = asyncio.run(
            engine.run(
                agent_definition=agent,
                livekit_runtime=runtime,
                scenario=_scenario(3),
            )
        )
    except ValueError as exc:
        pytest.fail(f"sip_inbound_room_template_required unexpectedly raised: {exc}")

    assert len(report.results) == 3


def test_drain_failure_is_a_failure_of_that_case_only(monkeypatch) -> None:
    runtime = livekit.LiveKitSimulatorRuntime(
        url="wss://livekit.example.com",
        room_name="sim-slot-03",
        room_mode="managed",
        room_name_verbatim=True,
    )
    agent = _agent(
        room_mode="managed", transport=TelephonyTransport(kind="sip_inbound")
    )
    engine = LiveKitEngine()
    calls = {"count": 0}

    async def _fake_run_case(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return livekit._failure_outcome(
                CaseStatus.TIMED_OUT,
                livekit.FailureStage.PREPARING,
                "livekit_room_drain_timeout",
                "The leased simulator room was still occupied when its drain deadline passed",
                retryable=True,
            )
        return livekit._CaseOutcome(status=CaseStatus.COMPLETED)

    monkeypatch.setattr(engine, "_run_single_test_case", _fake_run_case)

    report = asyncio.run(
        engine.run(
            agent_definition=agent,
            livekit_runtime=runtime,
            scenario=_scenario(3),
        )
    )

    assert len(report.results) == 3
    assert report.results[0].metadata["status"] == CaseStatus.TIMED_OUT.value
    assert report.results[0].metadata["failure"]["code"] == "livekit_room_drain_timeout"
    assert report.results[1].metadata["status"] == CaseStatus.COMPLETED.value
    assert report.results[2].metadata["status"] == CaseStatus.COMPLETED.value


class _FakeApiError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def test_ensure_room_absent_tolerates_not_found_on_initial_delete() -> None:
    calls: list[tuple] = []

    async def delete_room(request):
        calls.append(("delete", request.room))
        raise _FakeApiError("not_found")

    async def list_rooms(request):
        calls.append(("list", tuple(request.names)))
        return SimpleNamespace(rooms=[])

    api_client = SimpleNamespace(
        room=SimpleNamespace(delete_room=delete_room, list_rooms=list_rooms)
    )

    polls = asyncio.run(livekit._ensure_room_absent(api_client, "sim-slot-01"))

    assert polls == 1
    assert calls == [("delete", "sim-slot-01"), ("list", ("sim-slot-01",))]


def test_ensure_room_absent_redeletes_when_still_listed_once(monkeypatch) -> None:
    calls: list[tuple] = []
    list_responses = [
        SimpleNamespace(rooms=[SimpleNamespace(name="sim-slot-01")]),
        SimpleNamespace(rooms=[]),
    ]
    real_sleep = asyncio.sleep

    async def _fake_sleep(seconds):
        # Real backoff sleeps would cost CI 1.5s across these two tests for no
        # assertion value; yield instead so the poll loop still awaits.
        await real_sleep(0)

    monkeypatch.setattr(livekit.asyncio, "sleep", _fake_sleep)

    async def delete_room(request):
        calls.append(("delete", request.room))

    async def list_rooms(request):
        return list_responses.pop(0)

    api_client = SimpleNamespace(
        room=SimpleNamespace(delete_room=delete_room, list_rooms=list_rooms)
    )

    polls = asyncio.run(livekit._ensure_room_absent(api_client, "sim-slot-01"))

    assert polls == 2
    assert calls == [("delete", "sim-slot-01"), ("delete", "sim-slot-01")]


def test_ensure_room_absent_redeletes_when_still_listed_twice(monkeypatch) -> None:
    calls: list[tuple] = []
    list_responses = [
        SimpleNamespace(rooms=[SimpleNamespace(name="sim-slot-01")]),
        SimpleNamespace(rooms=[SimpleNamespace(name="sim-slot-01")]),
        SimpleNamespace(rooms=[]),
    ]
    real_sleep = asyncio.sleep

    async def _fake_sleep(seconds):
        # Real backoff sleeps would cost CI 1.5s across these two tests for no
        # assertion value; yield instead so the poll loop still awaits.
        await real_sleep(0)

    monkeypatch.setattr(livekit.asyncio, "sleep", _fake_sleep)

    async def delete_room(request):
        calls.append(("delete", request.room))

    async def list_rooms(request):
        return list_responses.pop(0)

    api_client = SimpleNamespace(
        room=SimpleNamespace(delete_room=delete_room, list_rooms=list_rooms)
    )

    polls = asyncio.run(livekit._ensure_room_absent(api_client, "sim-slot-01"))

    assert polls == 3
    assert len(calls) == 3


def test_ensure_room_absent_propagates_non_not_found_delete_error() -> None:
    async def delete_room(request):
        raise _FakeApiError("internal")

    async def list_rooms(request):
        raise AssertionError("list_rooms should not be called")

    api_client = SimpleNamespace(
        room=SimpleNamespace(delete_room=delete_room, list_rooms=list_rooms)
    )

    with pytest.raises(_FakeApiError):
        asyncio.run(livekit._ensure_room_absent(api_client, "sim-slot-01"))


def test_ensure_room_absent_times_out_and_backs_off(monkeypatch) -> None:
    sleep_calls: list[float] = []
    real_sleep = asyncio.sleep

    async def _fake_sleep(seconds):
        sleep_calls.append(seconds)
        # A non-yielding stub would never let wait_for's own timeout fire.
        await real_sleep(0)

    monkeypatch.setattr(livekit.asyncio, "sleep", _fake_sleep)

    async def delete_room(_request):
        return None

    async def list_rooms(_request):
        return SimpleNamespace(rooms=[SimpleNamespace(name="sim-slot-01")])

    api_client = SimpleNamespace(
        room=SimpleNamespace(delete_room=delete_room, list_rooms=list_rooms)
    )

    async def _run():
        await asyncio.wait_for(
            livekit._ensure_room_absent(api_client, "sim-slot-01"),
            timeout=0.2,
        )

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(_run())

    assert sleep_calls[:5] == [0.5, 0.5, 0.5, 5.0, 5.0]


def test_unexpected_participants_excludes_simulator_and_recorder() -> None:
    room = SimpleNamespace(
        remote_participants={
            "sim": SimpleNamespace(identity="fagi-simulator-abc"),
            "rec": SimpleNamespace(identity="fagi-recorder-abc"),
            "other": SimpleNamespace(identity="sip-caller-xyz"),
        }
    )

    result = livekit._unexpected_participants(
        room,
        simulator_identity="fagi-simulator-abc",
        recorder_identity="fagi-recorder-abc",
    )

    assert result == {"sip-caller-xyz"}


def test_unexpected_participants_empty_when_only_ours() -> None:
    room = SimpleNamespace(
        remote_participants={
            "sim": SimpleNamespace(identity="fagi-simulator-abc"),
            "rec": SimpleNamespace(identity="fagi-recorder-abc"),
        }
    )

    result = livekit._unexpected_participants(
        room,
        simulator_identity="fagi-simulator-abc",
        recorder_identity="fagi-recorder-abc",
    )

    assert result == set()


@pytest.mark.parametrize(
    "attribute_value, expected",
    [
        ("+14155551234", True),
        ("14155551234", True),
        ("4155551234", True),
        ("5551234", True),
        ("sip:+14155551234@trunk.example.com", True),
        ("sip:+14155551234@trunk;transport=udp", True),
        ("+14155559999", False),
        ("555", None),
    ],
)
def test_caller_matches_suffix_rule(attribute_value, expected) -> None:
    attributes = {"sip.phoneNumber": attribute_value}
    assert livekit._caller_matches(attributes, "+14155551234") is expected


def test_caller_matches_missing_attribute_is_unverified() -> None:
    assert livekit._caller_matches({}, "+14155551234") is None


def test_caller_matches_no_expected_number_is_unverified() -> None:
    assert livekit._caller_matches({"sip.phoneNumber": "+14155551234"}, None) is None


def test_caller_matches_national_format_mismatch_is_documented_limitation() -> None:
    # A national-format caller ID (0-prefixed instead of the country code) is a
    # known false mismatch, not a false match — loud and retryable.
    assert (
        livekit._caller_matches({"sip.phoneNumber": "0612345678"}, "+33612345678")
        is False
    )


def test_number_digits_strips_sip_uri_user_part_and_params() -> None:
    assert livekit._number_digits("sip:+1415@host9;x=1") == "1415"


def test_missing_credentials_is_typed_failure_not_transcript(monkeypatch) -> None:
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)

    report = asyncio.run(
        LiveKitEngine().run(
            agent_definition=_agent(),
            scenario=_scenario(),
            run_id="run_missing_credentials",
        )
    )

    result = report.results[0]
    assert result.transcript == ""
    assert result.metadata["status"] == CaseStatus.FAILED.value
    assert result.metadata["failure"]["code"] == "livekit_credentials_missing"


def test_recording_case_directory_is_used_without_repeating_run_and_case(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured = {}
    engine = LiveKitEngine()

    async def _fake_run_case(*_args, **kwargs):
        captured["case_directory"] = kwargs["case_directory"]
        return livekit._CaseOutcome(status=CaseStatus.COMPLETED)

    monkeypatch.setattr(engine, "_run_single_test_case", _fake_run_case)
    case_directory = tmp_path / "recordings"

    asyncio.run(
        engine.run(
            agent_definition=_agent(),
            scenario=_scenario(),
            run_id="run_direct_recordings",
            recording_case_directory=case_directory,
        )
    )

    assert captured["case_directory"] == case_directory


def test_recording_case_directory_rejects_multi_persona_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a single-persona scenario"):
        asyncio.run(
            LiveKitEngine().run(
                agent_definition=_agent(room_name="room-{test_case_id}"),
                scenario=_scenario(2),
                recording_case_directory=tmp_path / "recordings",
            )
        )


def test_default_customer_agent_supports_elevenlabs(monkeypatch) -> None:
    monkeypatch.setenv("SIMULATOR_VOICE_PROVIDER", "elevenlabs")
    monkeypatch.setenv("SIMULATOR_LLM_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("SIMULATOR_STT_MODEL", "scribe_v2_realtime")
    monkeypatch.setenv("SIMULATOR_TTS_MODEL", "eleven_flash_v2_5")
    monkeypatch.setenv("SIMULATOR_TTS_VOICE_ID", "voice-id")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.delenv("ELEVEN_API_KEY", raising=False)

    fake_openai = SimpleNamespace(
        LLM=lambda **kw: ("llm", kw),
        STT=lambda **kw: ("stt", kw),
        TTS=lambda **kw: ("tts", kw),
    )
    fake_elevenlabs = SimpleNamespace(
        STT=lambda **kw: ("stt", kw), TTS=lambda **kw: ("tts", kw)
    )

    def _fake_import(name):
        return {"openai": fake_openai, "elevenlabs": fake_elevenlabs}[name]

    monkeypatch.setattr(livekit_models, "_import_plugin", _fake_import)
    monkeypatch.setattr(
        livekit.silero,
        "VAD",
        SimpleNamespace(load=lambda: "vad"),
    )
    monkeypatch.setattr(
        livekit,
        "_TestRunnerAgent",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    agent, models = asyncio.run(
        LiveKitEngine()._create_customer_agent(
            _scenario().dataset[0],
            None,
        )
    )

    assert agent.llm == (
        "llm",
        {"model": "gpt-5.4-mini", "temperature": 0.6},
    )
    assert agent.stt == (
        "stt",
        {
            "api_key": "test-key",
            "http_session": models.http_session,
            "model_id": "scribe_v2_realtime",
            "server_vad": {
                "vad_silence_threshold_secs": 0.8,
                "vad_threshold": 0.4,
                "min_speech_duration_ms": 100,
                "min_silence_duration_ms": 500,
            },
        },
    )
    assert agent.tts == (
        "tts",
        {
            "api_key": "test-key",
            "http_session": models.http_session,
            "model": "eleven_flash_v2_5",
            "voice_id": "voice-id",
        },
    )
    assert models.http_session is not None
    asyncio.run(models.aclose())


def test_two_ten_case_suites_do_not_share_room_names(monkeypatch) -> None:
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
    engine = LiveKitEngine()
    agent = _agent(room_mode="managed", agent_name="support-agent")

    async def run_suites():
        return await asyncio.gather(
            engine.run(
                agent_definition=agent,
                scenario=_scenario(10),
                run_id="run_a",
            ),
            engine.run(
                agent_definition=agent,
                scenario=_scenario(10),
                run_id="run_b",
            ),
        )

    first, second = asyncio.run(run_suites())
    first_rooms = {result.metadata["room_name"] for result in first.results}
    second_rooms = {result.metadata["room_name"] for result in second.results}

    assert len(first_rooms) == 10
    assert len(second_rooms) == 10
    assert first_rooms.isdisjoint(second_rooms)


def _role_content(messages: list[dict]) -> list[dict]:
    """Project canonical report messages to just role+content.

    ``_canonical_report_messages`` enriches each message with voice-timing
    metadata (created_at, started/stopped_speaking_at, interrupted, e2e_latency);
    these tests assert the role-perspective + interruption-merge behavior, which
    lives entirely in role/content.
    """
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def test_report_messages_use_target_perspective_roles() -> None:
    session = SimpleNamespace(
        history=SimpleNamespace(
            items=[
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    text_content="Simulator opens the call.",
                ),
                SimpleNamespace(
                    type="message",
                    role="user",
                    text_content="Target agent responds.",
                ),
            ]
        )
    )

    assert _role_content(livekit._canonical_report_messages(session)) == [
        {"role": "user", "content": "Simulator opens the call."},
        {"role": "assistant", "content": "Target agent responds."},
    ]


def test_report_messages_merge_interrupted_same_role_fragments() -> None:
    session = SimpleNamespace(
        history=SimpleNamespace(
            items=[
                SimpleNamespace(
                    type="message",
                    role="user",
                    text_content="Your parcel is",
                    interrupted=True,
                ),
                SimpleNamespace(
                    type="message",
                    role="user",
                    text_content="still in transit.",
                ),
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    text_content="Thanks.",
                ),
            ]
        )
    )

    assert _role_content(livekit._canonical_report_messages(session)) == [
        {"role": "assistant", "content": "Your parcel is still in transit."},
        {"role": "user", "content": "Thanks."},
    ]


def test_report_messages_preserve_distinct_same_role_turns() -> None:
    session = SimpleNamespace(
        history=SimpleNamespace(
            items=[
                SimpleNamespace(
                    type="message",
                    role="user",
                    text_content="First update.",
                    interrupted=False,
                ),
                SimpleNamespace(
                    type="message",
                    role="user",
                    text_content="Second update.",
                    interrupted=False,
                ),
            ]
        )
    )

    assert _role_content(livekit._canonical_report_messages(session)) == [
        {"role": "assistant", "content": "First update."},
        {"role": "assistant", "content": "Second update."},
    ]


def test_target_audio_selection_uses_explicit_identity() -> None:
    audio_kind = livekit.rtc.TrackKind.KIND_AUDIO
    room = SimpleNamespace(
        remote_participants={
            "other": SimpleNamespace(
                identity="other-agent",
                sid="participant-other",
                track_publications={
                    "track-other": SimpleNamespace(
                        sid="track-other",
                        kind=audio_kind,
                    )
                },
            ),
            "target": SimpleNamespace(
                identity="target-agent",
                sid="participant-target",
                track_publications={
                    "track-target": SimpleNamespace(
                        sid="track-target",
                        kind=audio_kind,
                    )
                },
            ),
        }
    )

    selected = livekit._find_target_audio(
        room,
        excluded_identities=set(),
        target_identity="target-agent",
    )

    assert selected is not None
    assert selected.identity == "target-agent"
    assert selected.audio_track_sid == "track-target"


def test_target_audio_selection_ignores_background_track_from_same_agent() -> None:
    audio_kind = livekit.rtc.TrackKind.KIND_AUDIO
    room = SimpleNamespace(
        remote_participants={
            "target": SimpleNamespace(
                identity="drive-thru-agent",
                sid="participant-target",
                track_publications={
                    "ambient": SimpleNamespace(
                        sid="track-ambient",
                        name="background_audio",
                        kind=audio_kind,
                    ),
                    "speech": SimpleNamespace(
                        sid="track-speech",
                        name="roomio_audio",
                        kind=audio_kind,
                    ),
                },
            )
        }
    )

    selected = livekit._find_target_audio(
        room,
        excluded_identities=set(),
        target_identity="drive-thru-agent",
    )

    assert selected is not None
    assert selected.audio_track_sid == "track-speech"


def test_recording_mix_uses_only_explicit_paths(tmp_path: Path) -> None:
    first = tmp_path / "simulator.wav"
    second = tmp_path / "target.wav"
    unrelated = tmp_path / "unrelated.wav"
    _write_wav(first, np.array([1000, 1000], dtype=np.int16))
    _write_wav(second, np.array([2000, 2000], dtype=np.int16))
    _write_wav(unrelated, np.array([30000, 30000], dtype=np.int16))

    destination = tmp_path / "combined.wav"
    mix_recordings([first, second], destination, sample_rate=8000)

    with wave.open(str(destination), "rb") as wav_file:
        samples = np.frombuffer(
            wav_file.readframes(wav_file.getnframes()),
            dtype=np.int16,
        )
    assert samples.tolist() == [3000, 3000]


def test_stereo_mix_interleaves_and_zero_pads(tmp_path: Path) -> None:
    left = tmp_path / "simulator.wav"
    right = tmp_path / "target.wav"
    _write_wav(left, np.array([1000, 1000], dtype=np.int16))
    _write_wav(right, np.array([2000], dtype=np.int16))

    destination = tmp_path / "stereo.wav"
    result = mix_recordings_stereo([left], [right], destination, sample_rate=8000)

    assert result == destination
    with wave.open(str(destination), "rb") as wav_file:
        assert wav_file.getnchannels() == 2
        samples = np.frombuffer(
            wav_file.readframes(wav_file.getnframes()),
            dtype=np.int16,
        )
    assert samples.tolist() == [1000, 2000, 1000, 0]


def test_stereo_mix_sums_each_side_independently(tmp_path: Path) -> None:
    left_a = tmp_path / "sim_a.wav"
    left_b = tmp_path / "sim_b.wav"
    right = tmp_path / "target.wav"
    _write_wav(left_a, np.array([100, 100], dtype=np.int16))
    _write_wav(left_b, np.array([200, 200], dtype=np.int16))
    _write_wav(right, np.array([500, 500], dtype=np.int16))

    destination = tmp_path / "stereo.wav"
    mix_recordings_stereo([left_a, left_b], [right], destination, sample_rate=8000)

    with wave.open(str(destination), "rb") as wav_file:
        samples = np.frombuffer(
            wav_file.readframes(wav_file.getnframes()),
            dtype=np.int16,
        )
    assert samples.tolist() == [300, 500, 300, 500]


def test_stereo_mix_leaves_missing_side_silent(tmp_path: Path) -> None:
    right = tmp_path / "target.wav"
    _write_wav(right, np.array([2000, 2000], dtype=np.int16))

    destination = tmp_path / "stereo.wav"
    result = mix_recordings_stereo([], [right], destination, sample_rate=8000)

    assert result == destination
    with wave.open(str(destination), "rb") as wav_file:
        assert wav_file.getnchannels() == 2
        samples = np.frombuffer(
            wav_file.readframes(wav_file.getnframes()),
            dtype=np.int16,
        )
    assert samples.tolist() == [0, 2000, 0, 2000]


def test_attach_recordings_falls_back_to_all_target_tracks(tmp_path: Path) -> None:
    simulator = tmp_path / "simulator.wav"
    target_speech = tmp_path / "target-speech.wav"
    target_ambient = tmp_path / "target-ambient.wav"
    for path, samples in (
        (simulator, np.array([100, 100], dtype=np.int16)),
        (target_speech, np.array([200, 200], dtype=np.int16)),
        (target_ambient, np.array([50, 50], dtype=np.int16)),
    ):
        _write_wav(path, samples)

    records = (
        RecordedTrack("sim", "p1", "sim-track", simulator),
        RecordedTrack("target", "p2", "speech-track", target_speech),
        RecordedTrack("target", "p2", "ambient-track", target_ambient),
    )
    recorder = SimpleNamespace(
        records=records,
        errors=(),
        recording_started_at=None,
        paths_for_participant=lambda identity, track_sid=None: [
            record.path
            for record in records
            if record.participant_identity == identity
            and (track_sid is None or record.track_sid == track_sid)
        ],
    )
    outcome = livekit._CaseOutcome(status=CaseStatus.COMPLETED)

    livekit._attach_recordings(
        outcome,
        recorder,
        simulator_identity="sim",
        target_identity="target",
        target_track_sid="replaced-readiness-track",
        case_directory=tmp_path / "case",
        sample_rate=8000,
    )

    assert outcome.audio_combined_path is not None
    assert outcome.audio_output_path is not None
    assert outcome.metadata["recording_diagnostics"]["target_track_count"] == 2


def test_attach_recordings_uses_unambiguous_simulator_identity_fallback(
    tmp_path: Path,
) -> None:
    simulator = tmp_path / "sdk-local-track.wav"
    target = tmp_path / "target.wav"
    _write_wav(simulator, np.array([100, 100], dtype=np.int16))
    _write_wav(target, np.array([200, 200], dtype=np.int16))
    records = (
        RecordedTrack("sdk-local-identity", "p1", "sim-track", simulator),
        RecordedTrack("target", "p2", "target-track", target),
    )
    recorder = SimpleNamespace(
        records=records,
        errors=(),
        recording_started_at=None,
        paths_for_participant=lambda identity, track_sid=None: [
            record.path
            for record in records
            if record.participant_identity == identity
            and (track_sid is None or record.track_sid == track_sid)
        ],
    )
    outcome = livekit._CaseOutcome(status=CaseStatus.COMPLETED)

    livekit._attach_recordings(
        outcome,
        recorder,
        simulator_identity="expected-simulator-identity",
        target_identity="target",
        target_track_sid="target-track",
        case_directory=tmp_path / "case",
        sample_rate=8000,
    )

    assert outcome.audio_input_path == str(simulator)
    assert outcome.metadata["recording_diagnostics"]["simulator_track_count"] == 1


def test_room_recorder_token_is_hidden_subscribe_only() -> None:
    import jwt

    from fi.simulate.recording.room_recorder import RoomRecorder

    recorder = RoomRecorder(
        url="wss://livekit.example.com",
        api_key="key",
        api_secret="secret-0123456789abcdef0123456789abcdef",
        room_name="room-1",
        identity="fagi-recorder-abc123",
    )
    claims = jwt.decode(
        recorder._build_token(),
        "secret-0123456789abcdef0123456789abcdef",
        algorithms=["HS256"],
    )
    video = claims["video"]
    assert video["room"] == "room-1"
    assert video["roomJoin"] is True
    assert video["hidden"] is True
    assert video["recorder"] is True
    assert video["canPublish"] is False
    assert video["canPublishData"] is False
    assert video["canUpdateOwnMetadata"] is False
    assert video.get("canSubscribe", True) is True


def test_stereo_mix_returns_none_when_both_sides_empty(tmp_path: Path) -> None:
    destination = tmp_path / "stereo.wav"
    result = mix_recordings_stereo([], [], destination, sample_rate=8000)

    assert result is None
    assert not destination.exists()


def test_livekit_api_url_normalizes_websocket_schemes() -> None:
    assert livekit._api_url("wss://lk.example.com") == "https://lk.example.com"
    assert livekit._api_url("ws://localhost:7880") == "http://localhost:7880"


def test_provider_evidence_uses_explicit_target_api_configuration(monkeypatch) -> None:
    monkeypatch.setenv("TARGET_VAPI_KEY", "vapi-secret")
    vapi_target = livekit.VapiTargetConfig(
        assistant_id="assistant_123",
        api_base_url="https://vapi.example",
        api_key_env="TARGET_VAPI_KEY",
    )
    retell_target = livekit.RetellTargetConfig(
        agent_id="agent_123",
        api_url="https://retell.example/v2/create-web-call",
    )

    assert livekit._target_api_key(vapi_target) == "vapi-secret"
    assert livekit._target_evidence_base_url(vapi_target) == "https://vapi.example"
    assert livekit._target_evidence_base_url(retell_target) == "https://retell.example"


def test_managed_case_dispatches_waits_and_cleans_up(monkeypatch) -> None:
    calls = []
    audio_kind = livekit.rtc.TrackKind.KIND_AUDIO

    class FakeRoom:
        def __init__(self):
            self.remote_participants = {
                "target": SimpleNamespace(
                    identity="target-agent",
                    sid="participant-target",
                    track_publications={
                        "track-target": SimpleNamespace(
                            sid="track-target",
                            kind=audio_kind,
                        )
                    },
                )
            }
            self.listeners = {}

        async def connect(self, url, token):
            calls.append(("connect", url, token))

        async def disconnect(self):
            calls.append(("disconnect",))

        def on(self, event, callback=None):
            self.listeners.setdefault(event, []).append(callback)
            return callback

        def off(self, event, callback):
            self.listeners.get(event, []).remove(callback)

        def register_text_stream_handler(self, topic, handler=None):
            self.listeners.setdefault(("text_stream", topic), []).append(handler)
            return handler

        def unregister_text_stream_handler(self, topic):
            self.listeners.pop(("text_stream", topic), None)

    class FakeRoomService:
        async def create_room(self, request):
            calls.append(("create_room", request.name))

        async def delete_room(self, request):
            calls.append(("delete_room", request.room))

    class FakeDispatchService:
        async def create_dispatch(self, request):
            calls.append(
                ("dispatch", request.agent_name, request.room, request.metadata)
            )

    class FakeApiClient:
        def __init__(self):
            self.room = FakeRoomService()
            self.agent_dispatch = FakeDispatchService()

        async def aclose(self):
            calls.append(("api_close",))

    class FakeAccessToken:
        def __init__(self, _key, _secret):
            pass

        def with_identity(self, identity):
            calls.append(("identity", identity))
            return self

        def with_grants(self, _grants):
            return self

        def to_jwt(self):
            return "token"

    class FakeSession:
        def __init__(self):
            self.history = SimpleNamespace(
                items=[
                    SimpleNamespace(
                        type="message",
                        role="user",
                        text_content="Hello",
                    ),
                    SimpleNamespace(
                        type="message",
                        role="assistant",
                        text_content="Resolved",
                    ),
                ]
            )

        def on(self, event, callback):
            if event == "close":
                asyncio.get_running_loop().call_soon(callback, None)

        def shutdown(self, *, drain=True):
            calls.append(("shutdown", drain))

        async def wait_for_inactive(self):
            calls.append(("inactive",))

    class FakeCustomerAgent:
        def __init__(self):
            self.end_requested = asyncio.Event()
            self.end_requested.set()

        async def start_session(self, _room, **_kwargs):
            return FakeSession()

        def open_conversation(self):
            calls.append(("open",))

    room = FakeRoom()
    api_client = FakeApiClient()
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    monkeypatch.setattr(livekit.rtc, "Room", lambda: room)
    monkeypatch.setattr(livekit.api, "LiveKitAPI", lambda *_args: api_client)
    monkeypatch.setattr(livekit, "AccessToken", FakeAccessToken)
    engine = LiveKitEngine()

    async def _fake_create(_persona, _simulator, **_kwargs):
        return FakeCustomerAgent(), None

    monkeypatch.setattr(engine, "_create_customer_agent", _fake_create)

    report = asyncio.run(
        engine.run(
            agent_definition=_agent(
                room_mode="managed",
                agent_name="registered-agent",
                target_participant_identity="target-agent",
            ),
            scenario=_scenario(),
            run_id="run_managed",
            min_turn_messages=2,
        )
    )

    result = report.results[0]
    room_name = result.metadata["room_name"]
    assert result.metadata["status"] == CaseStatus.COMPLETED.value
    assert result.metadata["target_participant_identity"] == "target-agent"
    dispatch = next(call for call in calls if call[0] == "dispatch")
    assert dispatch[1:3] == ("registered-agent", room_name)
    # Dispatch metadata is empty by default (d13f8c2): a template-built target
    # treats any payload as an outbound/no-greet job and never publishes audio.
    # Metadata forwarding is covered by test_dispatch_metadata_* below; only a
    # target that opts in via AgentDefinition.dispatch_metadata gets a payload.
    assert dispatch[3] == ""
    assert ("delete_room", room_name) in calls
    assert ("open",) in calls


def test_simulator_subscribes_to_sip_participant_audio(monkeypatch) -> None:
    captured = {}

    class FakeSession:
        def __init__(self, **kwargs):
            captured["session_options"] = kwargs

        def on(self, event, callback):
            captured[event] = callback

        async def start(self, _agent, *, room, room_options):
            captured["room_options"] = room_options

    monkeypatch.setattr(livekit, "AgentSession", FakeSession)
    agent = livekit._TestRunnerAgent(
        persona=_scenario().dataset[0],
        instructions="Be a customer.",
    )
    asyncio.run(agent.start_session(SimpleNamespace()))

    assert (
        livekit.rtc.ParticipantKind.PARTICIPANT_KIND_SIP
        in captured["room_options"].participant_kinds
    )
    assert captured["room_options"].close_on_disconnect is False
    # text_output enabled so RoomIO builds the TranscriptSynchronizer that
    # truncates an interrupted turn to what was actually spoken.
    assert captured["room_options"].text_output is True
    assert captured["room_options"].audio_input.pre_connect_audio is False
    assert "turn_handling" in captured["session_options"]
    assert "allow_interruptions" not in captured["session_options"]
    assert "min_endpointing_delay" not in captured["session_options"]


def test_simulator_collects_normalized_model_usage(monkeypatch) -> None:
    from livekit.agents.metrics.base import Metadata

    captured = {}

    class FakeSession:
        def __init__(self, **_kwargs):
            pass

        def on(self, event, callback):
            captured[event] = callback

        async def start(self, _agent, *, room, room_options):
            pass

    monkeypatch.setattr(livekit, "AgentSession", FakeSession)
    agent = livekit._TestRunnerAgent(
        persona=_scenario().dataset[0],
        instructions="Be a customer.",
    )
    asyncio.run(agent.start_session(SimpleNamespace()))
    captured["metrics_collected"](
        SimpleNamespace(
            metrics=livekit.metrics.LLMMetrics(
                label="llm",
                request_id="request-1",
                timestamp=1.0,
                duration=0.2,
                ttft=0.1,
                cancelled=False,
                completion_tokens=5,
                prompt_tokens=7,
                prompt_cached_tokens=2,
                total_tokens=12,
                tokens_per_second=25.0,
                metadata=Metadata(
                    model_provider="google",
                    model_name="gemini-test",
                ),
            )
        )
    )

    assert agent.model_usage == [
        {
            "type": "llm_usage",
            "provider": "google",
            "model": "gemini-test",
            "input_tokens": 7,
            "input_cached_tokens": 2,
            "input_audio_tokens": 0,
            "input_cached_audio_tokens": 0,
            "input_text_tokens": 0,
            "input_cached_text_tokens": 0,
            "input_image_tokens": 0,
            "input_cached_image_tokens": 0,
            "output_tokens": 5,
            "output_audio_tokens": 0,
            "output_text_tokens": 0,
            "session_duration": 0.0,
        }
    ]


def test_open_conversation_generates_opener_without_reading_situation() -> None:
    calls = []
    agent = livekit._TestRunnerAgent(
        persona=_scenario().dataset[0],
        instructions="Be a customer.",
    )
    agent._session = SimpleNamespace(
        generate_reply=lambda **kwargs: calls.append(kwargs)
    )

    agent.open_conversation()

    assert calls == [{}]


def test_end_call_waits_for_minimum_balanced_conversation() -> None:
    class FakeSession:
        history = SimpleNamespace(
            items=[
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    text_content="Hello.",
                )
            ]
        )

        async def aclose(self):
            raise AssertionError("session must remain open")

    agent = livekit._TestRunnerAgent(
        persona=_scenario().dataset[0],
        instructions="Be a customer.",
        min_turn_messages=2,
    )
    agent._session = FakeSession()

    result = asyncio.run(agent.end_call(None))

    assert "at least 2 messages" in result
    assert not agent.end_requested.is_set()


def test_end_call_signals_runner_after_minimum_balanced_conversation() -> None:
    class FakeSpeechHandle:
        waited = False

        def __await__(self):
            async def wait():
                self.waited = True

            return wait().__await__()

    class FakeSession:
        history = SimpleNamespace(
            items=[
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    text_content="Hello.",
                ),
                SimpleNamespace(
                    type="message",
                    role="user",
                    text_content="Goodbye.",
                ),
            ]
        )

        async def aclose(self):
            raise AssertionError("the outer runner owns session teardown")

    agent = livekit._TestRunnerAgent(
        persona=_scenario().dataset[0],
        instructions="Be a customer.",
        min_turn_messages=2,
    )
    agent._session = FakeSession()
    speech_handle = FakeSpeechHandle()

    result = asyncio.run(agent.end_call(SimpleNamespace(speech_handle=speech_handle)))
    asyncio.run(agent.wait_for_end_speech())

    assert result == "Conversation ended."
    assert agent.end_requested.is_set()
    assert speech_handle.waited is True


def test_minimum_messages_is_a_floor_not_a_stop_trigger() -> None:
    calls = []

    class FakeSession:
        history = SimpleNamespace(
            items=[
                SimpleNamespace(type="message", role="assistant", text_content="Hello"),
                SimpleNamespace(type="message", role="user", text_content="Resolved"),
            ]
        )

        def on(self, _event, _callback):
            return None

        def shutdown(self, *, drain=True):
            calls.append(("shutdown", drain))

        async def wait_for_inactive(self):
            calls.append(("inactive",))

    class FakeRoom:
        def on(self, _event, _callback):
            return None

        def off(self, _event, _callback):
            return None

    async def run() -> str:
        end_requested = asyncio.Event()

        async def end_naturally() -> None:
            await asyncio.sleep(0.01)
            end_requested.set()

        task = asyncio.create_task(end_naturally())
        try:
            return await livekit._wait_for_conversation_end(
                FakeRoom(),
                FakeSession(),
                customer_agent=SimpleNamespace(end_requested=end_requested),
                target_identity="target-agent",
                timeout=1,
                conversation_direction="simulator_first",
                agent_first_silence_timeout_seconds=30,
            )
        finally:
            await task

    reason = asyncio.run(run())

    assert reason == "simulator_end_call"
    assert calls == []


def test_conversation_end_returns_settled_when_silence_backstop_fires(
    monkeypatch,
) -> None:
    # Wiring check: task-dict key -> reason tuple -> returned string. With no
    # endCall and no disconnect, a fired silence backstop ends as
    # "conversation_settled" (which classifies COMPLETED).
    class FakeRoom:
        def on(self, _event, _callback):
            return None

        def off(self, _event, _callback):
            return None

    class FakeSession:
        history = SimpleNamespace(
            items=[
                SimpleNamespace(type="message", role="assistant", text_content="Hi"),
                SimpleNamespace(type="message", role="user", text_content="Hello"),
            ]
        )

        def on(self, _event, _callback):
            return None

    async def _immediate_silence(session, **kwargs):
        return None

    monkeypatch.setattr(livekit, "_wait_for_conversation_silence", _immediate_silence)

    async def run() -> str:
        return await livekit._wait_for_conversation_end(
            FakeRoom(),
            FakeSession(),
            customer_agent=SimpleNamespace(end_requested=asyncio.Event()),
            target_identity="target-agent",
            timeout=1,
            conversation_direction="simulator_first",
            agent_first_silence_timeout_seconds=30,
        )

    assert asyncio.run(run()) == "conversation_settled"


def test_provider_disconnect_can_end_a_balanced_conversation() -> None:
    class FakeSession:
        history = SimpleNamespace(
            items=[
                SimpleNamespace(type="message", role="assistant", text_content="One"),
                SimpleNamespace(type="message", role="user", text_content="Two"),
                SimpleNamespace(type="message", role="assistant", text_content="Three"),
                SimpleNamespace(type="message", role="user", text_content="Four"),
                SimpleNamespace(type="message", role="assistant", text_content="Five"),
                SimpleNamespace(type="message", role="user", text_content="Six"),
            ]
        )

        def on(self, _event, _callback):
            return None

    class FakeRoom:
        def on(self, _event, _callback):
            return None

        def off(self, _event, _callback):
            return None

    async def provider_finished() -> None:
        await asyncio.sleep(0.01)

    async def run() -> str:
        return await livekit._wait_for_conversation_end(
            FakeRoom(),
            FakeSession(),
            customer_agent=SimpleNamespace(end_requested=asyncio.Event()),
            target_identity="target-agent",
            timeout=1,
            conversation_direction="simulator_first",
            agent_first_silence_timeout_seconds=30,
            provider_task=asyncio.create_task(provider_finished()),
        )

    reason = asyncio.run(run())

    assert reason == "provider_disconnected"
    outcome = livekit._conversation_outcome(
        reason,
        livekit._session_messages(FakeSession()),
        min_turn_messages=6,
    )
    assert outcome.status == CaseStatus.COMPLETED


def test_conversation_silence_backstop_ends_after_quiet_grace() -> None:
    # Ends purely on a silence gap — no message-count floor. A single message is
    # enough; the backstop only cares that nothing new has happened.
    session = SimpleNamespace(
        history=SimpleNamespace(
            items=[
                SimpleNamespace(type="message", role="assistant", text_content="Hello"),
            ]
        )
    )

    asyncio.run(
        asyncio.wait_for(
            livekit._wait_for_conversation_silence(session, quiet_seconds=0.01),
            timeout=1,
        )
    )


def test_conversation_silence_backstop_does_not_fire_at_message_floor() -> None:
    # The old behaviour ended the call the moment it hit the floor + a short lull.
    # Now a floor-length exchange with a sub-backstop lull keeps running.
    session = SimpleNamespace(
        history=SimpleNamespace(
            items=[
                SimpleNamespace(type="message", role="assistant", text_content="Hi"),
                SimpleNamespace(type="message", role="user", text_content="Hello"),
                SimpleNamespace(type="message", role="assistant", text_content="More?"),
                SimpleNamespace(type="message", role="user", text_content="Yes"),
                SimpleNamespace(type="message", role="assistant", text_content="Go on"),
                SimpleNamespace(type="message", role="user", text_content="Sure"),
            ]
        )
    )

    async def run() -> bool:
        task = asyncio.create_task(
            livekit._wait_for_conversation_silence(session, quiet_seconds=5.0)
        )
        await asyncio.sleep(0.05)
        done = task.done()
        task.cancel()
        return done

    assert asyncio.run(run()) is False


def test_conversation_silence_waits_until_speech_has_finished() -> None:
    session = SimpleNamespace(
        agent_state="speaking",
        user_state="listening",
        history=SimpleNamespace(
            items=[
                SimpleNamespace(type="message", role="assistant", text_content="Hello"),
                SimpleNamespace(type="message", role="user", text_content="Resolved"),
            ]
        ),
    )

    async def run() -> None:
        task = asyncio.create_task(
            livekit._wait_for_conversation_silence(session, quiet_seconds=0.01)
        )
        await asyncio.sleep(0.02)
        assert not task.done()
        session.agent_state = "listening"
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(run())


def test_conversation_settled_reason_classifies_completed() -> None:
    messages = [
        {"role": "assistant", "content": "One"},
        {"role": "user", "content": "Two"},
        {"role": "assistant", "content": "Three"},
        {"role": "user", "content": "Four"},
        {"role": "assistant", "content": "Five"},
        {"role": "user", "content": "Six"},
    ]
    outcome = livekit._conversation_outcome(
        "conversation_settled", messages, min_turn_messages=6
    )
    assert outcome.status == CaseStatus.COMPLETED
    assert outcome.metadata["stop_reason"] == "conversation_settled"


def test_conversation_timeout_does_not_start_session_teardown() -> None:
    calls = []

    class FakeSession:
        history = SimpleNamespace(items=[])

        def on(self, _event, _callback):
            return None

        def shutdown(self, *, drain=True):
            calls.append(("shutdown", drain))

    class FakeRoom:
        def on(self, _event, _callback):
            return None

        def off(self, _event, _callback):
            return None

    reason = asyncio.run(
        livekit._wait_for_conversation_end(
            FakeRoom(),
            FakeSession(),
            customer_agent=SimpleNamespace(end_requested=asyncio.Event()),
            target_identity="target-agent",
            timeout=0.01,
            conversation_direction="simulator_first",
            agent_first_silence_timeout_seconds=30,
        )
    )

    assert reason == "timeout"
    assert calls == []


def test_session_cleanup_timeout_does_not_cancel_livekit_close_task() -> None:
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    close_finished = asyncio.Event()

    class FakeSession:
        async def aclose(self):
            close_started.set()
            await allow_close.wait()
            close_finished.set()

    async def run() -> None:
        with pytest.raises(asyncio.TimeoutError):
            await livekit._close_agent_session(FakeSession(), timeout=0.01)
        assert close_started.is_set()
        assert not close_finished.is_set()
        allow_close.set()
        await asyncio.wait_for(close_finished.wait(), timeout=1)

    asyncio.run(run())


def test_safe_provider_error_details_include_sip_status_metadata() -> None:
    error = SimpleNamespace(
        code="failed_precondition",
        status=412,
        metadata={"sip_status_code": "486", "private": "do-not-copy"},
    )

    details = livekit._safe_provider_error_details(
        error,
        operation="sip_dial",
    )

    assert details == {
        "operation": "sip_dial",
        "exception_type": "SimpleNamespace",
        "provider_code": "failed_precondition",
        "http_status": 412,
        "sip_status_code": "486",
    }


@pytest.mark.parametrize(
    ("stop_reason", "messages", "failure_code"),
    [
        (
            "session_closed",
            [
                {"role": "assistant", "content": "One"},
                {"role": "user", "content": "Two"},
                {"role": "assistant", "content": "Three"},
                {"role": "user", "content": "Four"},
                {"role": "assistant", "content": "Five"},
                {"role": "user", "content": "Six"},
            ],
            "session_closed",
        ),
        (
            "conversation_silence_timeout",
            [
                {"role": "assistant", "content": "One"},
                {"role": "user", "content": "Two"},
                {"role": "assistant", "content": "Three"},
                {"role": "user", "content": "Four"},
                {"role": "assistant", "content": "Five"},
                {"role": "user", "content": "Six"},
            ],
            "conversation_silence_timeout",
        ),
    ],
)
def test_failed_stop_reasons_never_report_completed(
    stop_reason: str,
    messages: list[dict[str, str]],
    failure_code: str,
) -> None:
    outcome = livekit._conversation_outcome(
        stop_reason,
        messages,
        min_turn_messages=6,
    )

    assert outcome.status == CaseStatus.FAILED
    assert outcome.failure is not None
    assert outcome.failure.code == failure_code


@pytest.mark.parametrize(
    "messages",
    [
        [
            {"role": "assistant", "content": "How can I help?"},
            {"role": "user", "content": "Please book a ride."},
            {"role": "assistant", "content": "Your ride is booked."},
            {"role": "user", "content": "No, nothing else."},
            {"role": "assistant", "content": "Have a great day, goodbye!"},
            {"role": "user", "content": "Thanks, bye."},
        ],
        [
            {"role": "assistant", "content": "I cannot book this account."},
            {"role": "user", "content": "Can somebody help me?"},
            {"role": "assistant", "content": "I will transfer you to support."},
            {"role": "user", "content": "Please do."},
            {"role": "assistant", "content": "I am transferring you now. Please wait."},
            {"role": "user", "content": "Okay."},
        ],
    ],
)
def test_agent_first_silence_after_terminal_exchange_is_completed(
    messages: list[dict[str, str]],
) -> None:
    outcome = livekit._conversation_outcome(
        "conversation_silence_timeout",
        messages,
        min_turn_messages=6,
    )

    assert outcome.status == CaseStatus.COMPLETED
    assert outcome.failure is None
    assert outcome.metadata == {
        "stop_reason": "conversation_silence_timeout",
        "terminal_exchange_recovered": True,
    }


def test_farewell_only_turns_are_recognised_as_a_closing_loop() -> None:
    # Shape taken from a call that ran 76 turns trading goodbyes and cost the job its worlds.
    assert livekit._is_closing_only("Goodbye.")
    assert livekit._is_closing_only("Alright, thanks, bye!")
    assert livekit._is_closing_only("Have a great day!")


def test_a_turn_carrying_content_is_not_a_closing() -> None:
    # A farewell that also asks something is still live conversation.
    assert not livekit._is_closing_only("Goodbye, but can you resend the receipt first?")
    assert not livekit._is_closing_only("Yes, please book it.")
    assert not livekit._is_closing_only("")


def test_closing_loop_reports_the_call_as_completed() -> None:
    outcome = livekit._conversation_outcome(
        "closing_loop",
        [
            {"role": "assistant", "content": "Your ride is booked."},
            {"role": "user", "content": "Thanks, bye!"},
            {"role": "assistant", "content": "Goodbye!"},
            {"role": "user", "content": "Goodbye."},
            {"role": "assistant", "content": "Goodbye!"},
            {"role": "user", "content": "Goodbye."},
        ],
        min_turn_messages=6,
    )

    assert outcome.status == CaseStatus.COMPLETED
    assert outcome.failure is None


def test_silence_with_untimed_caller_turns_names_the_synthesis() -> None:
    # Shape taken from a run whose speech synthesis was out of credit: the caller's line reaches
    # the transcript, but nothing was ever spoken, so the agent's silence is not the agent's fault.
    outcome = livekit._conversation_outcome(
        "conversation_silence_timeout",
        [
            {
                "role": "assistant",
                "content": "Hi Dana, where should the driver pick you up?",
                "started_speaking_at": 100.0,
                "stopped_speaking_at": 104.0,
            },
            {
                "role": "user",
                "content": "Hi, I'd like to book a ride from 88 King Street.",
                "started_speaking_at": None,
                "stopped_speaking_at": None,
            },
        ],
        min_turn_messages=6,
    )

    assert outcome.status == CaseStatus.FAILED
    assert outcome.failure is not None
    assert outcome.failure.code == "simulator_tts_silent"
    assert outcome.failure.retryable is False


def test_silence_with_audible_caller_turns_still_reports_a_stall() -> None:
    outcome = livekit._conversation_outcome(
        "conversation_silence_timeout",
        [
            {
                "role": "assistant",
                "content": "Hi Dana, where should the driver pick you up?",
                "started_speaking_at": 100.0,
                "stopped_speaking_at": 104.0,
            },
            {
                "role": "user",
                "content": "I'm at 333 O'Farrell Street.",
                "started_speaking_at": 105.0,
                "stopped_speaking_at": 109.0,
            },
        ],
        min_turn_messages=6,
    )

    assert outcome.status == CaseStatus.FAILED
    assert outcome.failure is not None
    assert outcome.failure.code == "conversation_silence_timeout"


def test_unsupported_provider_lists_supported_options() -> None:
    from fi.simulate.agent.definition import LLMConfig, STTConfig, TTSConfig

    with pytest.raises(
        ValueError, match="Unsupported LiveKit STT provider: 'nope'"
    ) as exc_info:
        asyncio.run(
            livekit_models.build_livekit_models(
                llm_config=LLMConfig(),
                stt_config=STTConfig(provider="nope"),
                tts_config=TTSConfig(),
            )
        )
    assert "Supported:" in str(exc_info.value)


def test_google_tts_uses_linear16_for_non_streaming_synthesis(monkeypatch) -> None:
    from google.cloud import texttospeech

    from fi.simulate.agent.definition import TTSConfig

    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/google.json")
    monkeypatch.setattr(
        livekit_models,
        "_import_plugin",
        lambda _name: SimpleNamespace(TTS=lambda **kwargs: kwargs),
    )

    tts = livekit_models._google_tts(
        TTSConfig(provider="google", voice="en-US-Chirp3-HD-Kore"),
        None,
    )

    assert tts["audio_encoding"] == texttospeech.AudioEncoding.LINEAR16


def test_google_stt_defers_requests_until_vad_detects_speech(monkeypatch) -> None:
    from fi.simulate.agent.definition import STTConfig

    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/google.json")
    monkeypatch.setattr(
        livekit_models,
        "_import_plugin",
        lambda _name: SimpleNamespace(STT=lambda **kwargs: kwargs),
    )

    stt = livekit_models._google_stt(
        STTConfig(provider="google", language="en-US, es-ES"),
        None,
    )

    assert stt["languages"] == ["en-US", "es-ES"]
    assert stt["use_streaming"] is False


def test_gemini_three_defaults_vertex_location_to_global(monkeypatch) -> None:
    from fi.simulate.agent.definition import LLMConfig

    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/google.json")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "project")
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("VERTEX_LOCATION", raising=False)
    monkeypatch.setattr(
        livekit_models,
        "_import_plugin",
        lambda _name: SimpleNamespace(LLM=lambda **kwargs: kwargs),
    )

    llm = livekit_models._google_llm(
        LLMConfig(provider="google", model="gemini-3.6-flash")
    )

    assert llm["location"] == "global"


def test_cartesia_stt_and_tts_use_configured_models(monkeypatch) -> None:
    from fi.simulate.agent.definition import STTConfig, TTSConfig

    monkeypatch.setenv("CARTESIA_API_KEY", "cartesia-key")
    plugin = SimpleNamespace(
        STT=lambda **kwargs: ("stt", kwargs),
        TTS=lambda **kwargs: ("tts", kwargs),
    )
    monkeypatch.setattr(livekit_models, "_import_plugin", lambda _name: plugin)

    stt = livekit_models._cartesia_stt(
        STTConfig(provider="cartesia", model="ink-2", language="en"),
        "session",
    )
    tts = livekit_models._cartesia_tts(
        TTSConfig(provider="cartesia", model="sonic-3", voice="voice-id"),
        "session",
    )

    assert stt == (
        "stt",
        {
            "api_key": "cartesia-key",
            "http_session": "session",
            "model": "ink-2",
            "language": "en",
        },
    )
    assert tts == (
        "tts",
        {
            "api_key": "cartesia-key",
            "http_session": "session",
            "model": "sonic-3",
            "voice": "voice-id",
        },
    )


class _FakeRoomAudio:
    def __init__(self, target_identity: str) -> None:
        audio_kind = livekit.rtc.TrackKind.KIND_AUDIO
        self.remote_participants = {
            "target": SimpleNamespace(
                identity=target_identity,
                sid="participant-target",
                track_publications={
                    "track-target": SimpleNamespace(sid="track-target", kind=audio_kind)
                },
            )
        }

    async def connect(self, url, token):
        pass

    async def disconnect(self):
        pass

    def on(self, event, callback=None):
        return callback

    def off(self, event, callback):
        pass

    def register_text_stream_handler(self, topic, handler=None):
        return handler

    def unregister_text_stream_handler(self, topic):
        pass


class _FakeSipSession:
    def __init__(self) -> None:
        self.history = SimpleNamespace(
            items=[
                SimpleNamespace(type="message", role="user", text_content="hi"),
                SimpleNamespace(type="message", role="assistant", text_content="ok"),
            ]
        )

    def on(self, event, callback):
        if event == "close":
            asyncio.get_running_loop().call_soon(callback, None)

    def shutdown(self, *, drain=True):
        pass

    async def wait_for_inactive(self):
        pass


class _FakeCustomerAgent:
    def __init__(self) -> None:
        self.end_requested = asyncio.Event()
        self.end_requested.set()

    async def start_session(self, _room, **_kwargs):
        return _FakeSipSession()

    def open_conversation(self):
        pass


def _fake_access_token():
    class _Tok:
        def __init__(self, *_a):
            pass

        def with_identity(self, _i):
            return self

        def with_grants(self, _g):
            return self

        def to_jwt(self):
            return "t"

    return _Tok


def _install_engine_fakes(monkeypatch, calls, target_identity="target-agent"):
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")

    class _Sip:
        async def create_sip_participant(self, request):
            calls.append(
                (
                    "sip_dial",
                    request.sip_trunk_id,
                    request.sip_number,
                    request.sip_call_to,
                    request.room_name,
                    request.participant_identity,
                    request.wait_until_answered,
                )
            )

    class _Room:
        async def create_room(self, request):
            calls.append(("create_room", request.name))

        async def delete_room(self, request):
            calls.append(("delete_room", request.room))

    class _Dispatch:
        async def create_dispatch(self, request):
            calls.append(("dispatch", request.agent_name, request.room))

    class _Api:
        def __init__(self):
            self.room = _Room()
            self.agent_dispatch = _Dispatch()
            self.sip = _Sip()

        async def aclose(self):
            pass

    room = _FakeRoomAudio(target_identity)
    monkeypatch.setattr(livekit.rtc, "Room", lambda: room)
    monkeypatch.setattr(livekit.api, "LiveKitAPI", lambda *_a: _Api())
    monkeypatch.setattr(livekit, "AccessToken", _fake_access_token())
    engine = LiveKitEngine()

    async def _fake_create(_p, _s, **_kwargs):
        return _FakeCustomerAgent(), None

    monkeypatch.setattr(engine, "_create_customer_agent", _fake_create)
    return engine


def test_sip_outbound_dials_per_case_room_and_identity(monkeypatch) -> None:
    calls: list = []
    engine = _install_engine_fakes(monkeypatch, calls, target_identity="sip-target")
    agent = _agent(
        room_mode="managed",
        room_name="sdk-suite-{test_case_id}",
        target_participant_identity="sip-target",
        transport={
            "kind": "sip_outbound",
            "sip_trunk_id": "ST_test",
            "sip_number": "+12068956991",
            "sip_call_to": "+14155551234",
        },
    )
    report = asyncio.run(
        engine.run(
            agent_definition=agent,
            scenario=_scenario(2),
            run_id="run_sip",
            min_turn_messages=2,
        )
    )

    dials = [call for call in calls if call[0] == "sip_dial"]
    assert len(dials) == 2
    room_names = [call[4] for call in dials]
    identities = [call[5] for call in dials]
    assert len(set(room_names)) == 2
    assert len(set(identities)) == 2
    assert all(call[1] == "ST_test" for call in dials)
    assert all(call[2] == "+12068956991" for call in dials)
    assert all(call[3] == "+14155551234" for call in dials)
    assert all(call[6] is True for call in dials)
    for result in report.results:
        assert result.metadata["status"] == CaseStatus.COMPLETED.value

    assert not [call for call in calls if call[0] == "dispatch"]


def test_sip_outbound_api_failure_yields_typed_sip_dial_failed(monkeypatch) -> None:
    calls: list = []
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    class _Sip:
        async def create_sip_participant(self, request):
            raise RuntimeError("dial refused")

    class _Room:
        async def create_room(self, request):
            calls.append(("create_room", request.name))

        async def delete_room(self, request):
            calls.append(("delete_room", request.room))

    class _Api:
        def __init__(self):
            self.room = _Room()
            self.agent_dispatch = SimpleNamespace(create_dispatch=lambda _r: None)
            self.sip = _Sip()

        async def aclose(self):
            pass

    monkeypatch.setattr(livekit.rtc, "Room", lambda: _FakeRoomAudio("sip-target"))
    monkeypatch.setattr(livekit.api, "LiveKitAPI", lambda *_a: _Api())
    monkeypatch.setattr(livekit, "AccessToken", _fake_access_token())
    engine = LiveKitEngine()

    async def _fake_create(_p, _s, **_kwargs):
        return _FakeCustomerAgent(), None

    monkeypatch.setattr(engine, "_create_customer_agent", _fake_create)

    agent = _agent(
        room_mode="managed",
        transport={
            "kind": "sip_outbound",
            "sip_trunk_id": "ST_test",
            "sip_number": "+12068956991",
            "sip_call_to": "+14155551234",
        },
    )
    report = asyncio.run(
        engine.run(
            agent_definition=agent,
            scenario=_scenario(),
            run_id="run_sip_fail",
        )
    )
    result = report.results[0]
    assert result.metadata["status"] == CaseStatus.FAILED.value
    failure = result.metadata["failure"]
    assert failure["code"] == "sip_dial_failed"
    assert "dial refused" not in json.dumps(failure)
    assert failure["details"]["exception_type"] == "RuntimeError"


def test_sip_inbound_timeout_yields_typed_no_participant(monkeypatch) -> None:
    calls: list = []
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    class _Room:
        async def create_room(self, request):
            calls.append(("create_room", request.name))
            calls_room_name.append(request.name)

        async def delete_room(self, request):
            calls.append(("delete_room", request.room))

    calls_room_name: list[str] = []

    class _SipStub:
        async def list_sip_dispatch_rule(self, _request):
            expected_room = calls_room_name[-1] if calls_room_name else ""
            item = SimpleNamespace(
                name="inbound-rule",
                sip_dispatch_rule_id="SD_reused",
                trunk_ids=["ST_test_inbound"],
                rule=SimpleNamespace(
                    dispatch_rule_direct=SimpleNamespace(room_name=expected_room)
                ),
            )
            return SimpleNamespace(items=[item])

        async def create_sip_dispatch_rule(self, _request):
            raise AssertionError("dispatch_rule_name reuse must not create a new rule")

        async def delete_sip_dispatch_rule(self, _request):
            raise AssertionError("reused dispatch rule must not be deleted")

        async def create_sip_participant(self, _request):
            return None

    class _Api:
        def __init__(self):
            self.room = _Room()
            self.agent_dispatch = SimpleNamespace(create_dispatch=lambda _r: None)
            self.sip = _SipStub()

        async def aclose(self):
            pass

    class _EmptyRoom:
        remote_participants: dict = {}

        async def connect(self, url, token):
            pass

        async def disconnect(self):
            pass

        def on(self, event, callback=None):
            return callback

        def off(self, event, callback):
            pass

    monkeypatch.setattr(livekit.rtc, "Room", lambda: _EmptyRoom())
    monkeypatch.setattr(livekit.api, "LiveKitAPI", lambda *_a: _Api())
    monkeypatch.setattr(livekit, "AccessToken", _fake_access_token())
    engine = LiveKitEngine()

    async def _fake_create(_p, _s, **_kwargs):
        session = _FakeSipSession()

        class _Agent:
            async def start_session(self, _room, **_kwargs):
                return session

            def open_conversation(self):
                pass

        return _Agent(), None

    monkeypatch.setattr(engine, "_create_customer_agent", _fake_create)

    agent = _agent(
        room_mode="managed",
        transport={
            "kind": "sip_inbound",
            "dispatch_rule_name": "inbound-rule",
            "readiness_timeout_seconds": 0.05,
        },
    )
    report = asyncio.run(
        engine.run(
            agent_definition=agent,
            scenario=_scenario(),
            run_id="run_sip_in",
            readiness_timeout=5.0,
        )
    )
    result = report.results[0]
    assert result.metadata["status"] == CaseStatus.AGENT_UNAVAILABLE.value
    assert result.metadata["failure"]["code"] == "sip_inbound_no_participant"
    assert result.metadata["sip_dispatch_rule_id"] == "SD_reused"
    assert result.metadata["sip_dispatch_rule_created"] is False


def test_cleanup_logging_redacts_exception_details(caplog) -> None:
    secret = "-".join(("provider", "secret", "value"))
    errors = []

    try:
        raise RuntimeError(secret)
    except RuntimeError as exc:
        with caplog.at_level(
            logging.ERROR,
            logger="fi.simulate.simulation.engines.livekit",
        ):
            livekit._record_cleanup_error(
                errors,
                exc,
                "disconnect",
                "run_redaction",
                "case_redaction",
            )

    assert errors == ["disconnect:RuntimeError"]
    assert secret not in caplog.text
    assert "RuntimeError: details redacted" in caplog.text


@pytest.mark.parametrize(
    (
        "transport_kind",
        "connector_name",
        "identity_prefix",
        "target",
        "conversation_direction",
        "first_message_mode",
    ),
    [
        (
            "vapi_websocket",
            "VapiWebSocketConnector",
            "fagi-vapi-bridge-",
            {
                "provider": "vapi",
                "assistant_id": "assistant_123",
                "api_key_env": "TARGET_PROVIDER_KEY",
            },
            "simulator_first",
            "assistant-waits-for-user",
        ),
        (
            "vapi_websocket",
            "VapiWebSocketConnector",
            "fagi-vapi-bridge-",
            {
                "provider": "vapi",
                "assistant_id": "assistant_123",
                "api_key_env": "TARGET_PROVIDER_KEY",
            },
            "agent_first",
            "assistant-speaks-first",
        ),
        (
            "retell_webcall",
            "RetellWebCallConnector",
            "fagi-retell-bridge-",
            {
                "provider": "retell",
                "agent_id": "agent_123",
                "api_key_env": "TARGET_PROVIDER_KEY",
            },
            "simulator_first",
            None,
        ),
    ],
)
def test_web_bridge_joins_as_target_without_sip(
    monkeypatch,
    transport_kind,
    connector_name,
    identity_prefix,
    target,
    conversation_direction,
    first_message_mode,
) -> None:
    calls: list[tuple] = []
    engine = _install_engine_fakes(monkeypatch, calls)

    class _Bridge:
        def __init__(self, **kwargs):
            calls.append(("bridge_init", kwargs["room_name"], kwargs["identity"]))
            self.call_id = "call_web_123"
            self._closed = asyncio.Event()

        async def connect(self):
            calls.append(("bridge_connect",))

        async def run(self):
            await self._closed.wait()

        async def aclose(self):
            calls.append(("bridge_close",))
            self._closed.set()

    async def _wait_for_target(
        _room,
        *,
        excluded_identities,
        target_identity,
        timeout,
    ):
        calls.append(("target_wait", target_identity))
        return livekit._TargetParticipant(
            identity=target_identity,
            sid="bridge-participant",
            audio_track_sid="bridge-track",
        )

    # Connector construction moved into endpoints.profiles (slice 3); it imports
    # the connector from simulation.bridge, so patch from_target on that class.
    connector_type = getattr(_bridge, connector_name)
    received_targets = []
    connector_kwargs = []

    def _from_target(_cls, provider_target, **kwargs):
        received_targets.append(provider_target)
        connector_kwargs.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        connector_type,
        "from_target",
        classmethod(_from_target),
    )
    monkeypatch.setattr(livekit, "LiveKitAudioBridge", _Bridge)
    monkeypatch.setattr(livekit, "_wait_for_target_audio", _wait_for_target)

    report = asyncio.run(
        engine.run(
            agent_definition=_agent(
                room_mode="managed",
                room_name="sdk-web-{test_case_id}",
                transport={"kind": transport_kind},
                target=target,
            ),
            scenario=_scenario(),
            run_id="run_web_bridge",
            min_turn_messages=2,
            conversation_direction=conversation_direction,
        )
    )

    result = report.results[0]
    assert result.metadata["status"] == CaseStatus.COMPLETED.value
    assert result.metadata["target_provider"] == target["provider"]
    assert result.metadata["provider_call_id"] == "call_web_123"
    assert result.metadata["target_participant_identity"].startswith(identity_prefix)
    assert received_targets[0].provider == target["provider"]
    assert connector_kwargs == (
        [{"first_message_mode": first_message_mode}]
        if first_message_mode is not None
        else [{}]
    )
    assert ("bridge_connect",) in calls
    assert ("bridge_close",) in calls
    assert not [call for call in calls if call[0] in {"dispatch", "sip_dial"}]


# ---------------------------------------------------------------------------
# Bounded-concurrent case execution (per-run concurrency + ordering)
# ---------------------------------------------------------------------------


def _concurrency_probe(monkeypatch, *, fail_index: int | None = None):
    """Stub ``_run_single_test_case`` to record in-flight concurrency and use
    inverted per-case delays so completion order differs from dataset order."""
    engine = LiveKitEngine()
    state = {"in_flight": 0, "max_in_flight": 0}

    async def _fake_run_case(_agent, _runtime, persona, _simulator, **kwargs):
        name = persona.persona["name"]
        index = int(name.rsplit(" ", 1)[1])
        state["in_flight"] += 1
        state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        try:
            if fail_index is not None and index == fail_index:
                raise RuntimeError("boom")
            # Inverted delay: earlier cases finish later, so completion order
            # would scramble results if the loop relied on completion order.
            await asyncio.sleep(0.02 * (10 - index))
            return livekit._CaseOutcome(
                status=CaseStatus.COMPLETED,
                metadata={"case_name": name},
            )
        finally:
            state["in_flight"] -= 1

    monkeypatch.setattr(engine, "_run_single_test_case", _fake_run_case)
    return engine, state


def test_cases_run_concurrently_and_preserve_dataset_order(monkeypatch) -> None:
    engine, state = _concurrency_probe(monkeypatch)
    agent = _agent(room_mode="managed", agent_name="support-agent")

    report = asyncio.run(
        engine.run(
            agent_definition=agent,
            scenario=_scenario(6),
            run_id="run_concurrent",
            max_concurrency=5,
        )
    )

    # Web cases overlap but are clamped to the voice ceiling even though the
    # platform requested 5 — one 4-CPU child cannot drive more pipelines.
    assert state["max_in_flight"] == _voice_max_case_concurrency()
    # Despite inverted completion order, results stay in dataset order.
    order = [r.persona.persona["name"] for r in report.results]
    assert order == [f"Caller {i}" for i in range(6)]


def test_default_concurrency_is_sequential(monkeypatch) -> None:
    engine, state = _concurrency_probe(monkeypatch)
    agent = _agent(room_mode="managed", agent_name="support-agent")

    report = asyncio.run(
        engine.run(
            agent_definition=agent,
            scenario=_scenario(4),
            run_id="run_serial_default",
        )
    )

    assert state["max_in_flight"] == 1
    order = [r.persona.persona["name"] for r in report.results]
    assert order == [f"Caller {i}" for i in range(4)]


def test_sip_transport_forces_serial_even_with_high_concurrency(monkeypatch) -> None:
    engine, state = _concurrency_probe(monkeypatch)
    agent = _agent(
        room_mode="managed",
        room_name="sdk-suite-{test_case_id}",
        transport={
            "kind": "sip_outbound",
            "sip_trunk_id": "ST_test",
            "sip_number": "+12068956991",
            "sip_call_to": "+14155551234",
        },
    )

    asyncio.run(
        engine.run(
            agent_definition=agent,
            scenario=_scenario(4),
            run_id="run_sip_serial",
            max_concurrency=5,
        )
    )

    # A run leases one DID — telephone cases must never overlap.
    assert state["max_in_flight"] == 1


def test_case_crash_yields_dense_failed_result_without_shifting_order(
    monkeypatch,
) -> None:
    engine, _state = _concurrency_probe(monkeypatch, fail_index=2)
    agent = _agent(room_mode="managed", agent_name="support-agent")

    report = asyncio.run(
        engine.run(
            agent_definition=agent,
            scenario=_scenario(5),
            run_id="run_crash",
            max_concurrency=5,
        )
    )

    # No hole: every dataset slot has a result, still in order.
    assert len(report.results) == 5
    order = [r.persona.persona["name"] for r in report.results]
    assert order == [f"Caller {i}" for i in range(5)]
    # The crashed case is a typed failure in its own slot; others complete.
    statuses = [r.metadata["status"] for r in report.results]
    assert statuses[2] == CaseStatus.FAILED.value
    assert report.results[2].metadata["failure"]["code"] == "case_execution_error"
    assert all(statuses[i] == CaseStatus.COMPLETED.value for i in (0, 1, 3, 4))


def test_on_case_complete_streams_every_index_including_failed_slot(
    monkeypatch,
) -> None:
    # The streaming hook must fire once per dataset slot — success AND the dense
    # FAILED slot — so no case is silently dropped from the platform.
    engine, _state = _concurrency_probe(monkeypatch, fail_index=2)
    agent = _agent(room_mode="managed", agent_name="support-agent")

    streamed: list[tuple[int, str, str]] = []
    lock = asyncio.Lock()

    async def _on_case_complete(index, case):
        async with lock:
            streamed.append(
                (index, case.persona.persona["name"], case.metadata["status"])
            )

    report = asyncio.run(
        engine.run(
            agent_definition=agent,
            scenario=_scenario(5),
            run_id="run_stream",
            max_concurrency=5,
            on_case_complete=_on_case_complete,
        )
    )

    # Exactly one callback per dataset slot; indices complete and unique.
    assert sorted(i for i, _, _ in streamed) == [0, 1, 2, 3, 4]
    by_index = {i: (name, status) for i, name, status in streamed}
    # Every streamed case matches its finalized report slot (name + status).
    for i, result in enumerate(report.results):
        assert by_index[i][0] == result.persona.persona["name"]
        assert by_index[i][1] == result.metadata["status"]
    assert by_index[2][1] == CaseStatus.FAILED.value


def test_on_case_complete_error_never_fails_the_case(monkeypatch) -> None:
    # A raising stream callback must not sink the case — the run still completes
    # and every slot is present (finalize reconciles the un-streamed ones).
    engine, _state = _concurrency_probe(monkeypatch)
    agent = _agent(room_mode="managed", agent_name="support-agent")

    async def _boom(index, case):
        raise RuntimeError("sink down")

    report = asyncio.run(
        engine.run(
            agent_definition=agent,
            scenario=_scenario(3),
            run_id="run_stream_err",
            max_concurrency=3,
            on_case_complete=_boom,
        )
    )

    assert len(report.results) == 3
    assert [r.persona.persona["name"] for r in report.results] == [
        f"Caller {i}" for i in range(3)
    ]


def test_dispatch_metadata_empty_by_default():
    # A real target agent flips to outbound/no-greet on any dispatch metadata,
    # so the default must be an empty string (not our simulation context).
    assert livekit._dispatch_metadata_json(_agent()) == ""
    assert (
        livekit._dispatch_metadata_json(SimpleNamespace(dispatch_metadata=None)) == ""
    )
    assert livekit._dispatch_metadata_json(SimpleNamespace(dispatch_metadata={})) == ""


def test_dispatch_metadata_forwarded_when_set():
    out = livekit._dispatch_metadata_json(
        SimpleNamespace(dispatch_metadata={"b": 2, "a": 1})
    )
    assert out == json.dumps({"a": 1, "b": 2}, sort_keys=True)


def test_crashed_monitor_is_not_mistaken_for_its_condition(monkeypatch) -> None:
    # Regression: a watcher that raised counted as "done" and was mapped to
    # conversation_silence_timeout, reporting a dead call as merely "stalled".
    class FakeRoom:
        def on(self, _event, _callback):
            return None

        def off(self, _event, _callback):
            return None

    class FakeSession:
        history = SimpleNamespace(
            items=[
                SimpleNamespace(type="message", role="assistant", text_content="Hi"),
                SimpleNamespace(type="message", role="user", text_content="Hello"),
            ]
        )

        def on(self, _event, _callback):
            return None

    async def _crash(session, **kwargs):
        raise RuntimeError("engine is closed")

    async def _park(session, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(livekit, "_wait_for_agent_first_silence", _crash)
    monkeypatch.setattr(livekit, "_wait_for_conversation_silence", _park)

    async def run() -> str:
        return await livekit._wait_for_conversation_end(
            FakeRoom(),
            FakeSession(),
            customer_agent=SimpleNamespace(end_requested=asyncio.Event()),
            target_identity="target-agent",
            timeout=1,
            conversation_direction="agent_first",
            agent_first_silence_timeout_seconds=30,
        )

    reason = asyncio.run(run())

    assert reason == "monitor_failed"
    outcome = livekit._conversation_outcome(reason, [], min_turn_messages=4)
    assert outcome.status == CaseStatus.FAILED
    assert outcome.failure.code == "monitor_failed"


def test_dead_call_with_no_turns_ends_as_no_conversation(monkeypatch) -> None:
    # Regression: a target that greets then hangs up unseen (VAPI
    # silence-timed-out) left the case running for the full talk budget with an
    # empty transcript.
    class FakeRoom:
        def on(self, _event, _callback):
            return None

        def off(self, _event, _callback):
            return None

    class FakeSession:
        history = SimpleNamespace(items=[])

        def on(self, _event, _callback):
            return None

    monkeypatch.setattr(livekit, "_NO_CONVERSATION_TIMEOUT_SECONDS", 0.05)

    async def run() -> str:
        return await livekit._wait_for_conversation_end(
            FakeRoom(),
            FakeSession(),
            customer_agent=SimpleNamespace(end_requested=asyncio.Event()),
            target_identity="target-agent",
            timeout=5,
            conversation_direction="agent_first",
            agent_first_silence_timeout_seconds=30,
        )

    reason = asyncio.run(run())

    assert reason == "no_conversation"
    outcome = livekit._conversation_outcome(reason, [], min_turn_messages=4)
    assert outcome.status == CaseStatus.FAILED
    assert outcome.failure.code == "no_conversation"


def test_no_conversation_guard_parks_once_turns_exist() -> None:
    session = SimpleNamespace(
        history=SimpleNamespace(
            items=[
                SimpleNamespace(type="message", role="user", text_content="Hi"),
            ]
        )
    )

    async def run() -> None:
        task = asyncio.create_task(
            livekit._wait_for_conversation_never_started(
                session,
                timeout_seconds=0.05,
            )
        )
        done, _pending = await asyncio.wait({task}, timeout=0.3)
        assert not done
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


class _FakeTranscriptionReader:
    def __init__(self, text: str) -> None:
        self._text = text
        self.info = SimpleNamespace(attributes={})

    async def read_all(self) -> str:
        return self._text


class _FakeReplySession:
    def __init__(self, *, reply_error: Exception | None = None) -> None:
        self.reply_inputs: list[object] = []
        self.history_adds: list[tuple[str, str]] = []
        self._reply_error = reply_error
        session = self

        class _History:
            def add_message(self, *, role: str, content: str) -> None:
                session.history_adds.append((role, content))

        self.history = _History()

    def generate_reply(self, *, user_input=None):
        if self._reply_error is not None:
            raise self._reply_error
        self.reply_inputs.append(user_input)


def test_target_transcription_feeds_simulator_llm_context() -> None:
    # Regression: the target's turns were added to session.history (the
    # transcript context) with a bare generate_reply(); the reply pipeline
    # reads the AGENT's chat context, so the simulator LLM never saw a single
    # target turn and answered every one with "I can't hear you".
    session = _FakeReplySession()
    captured: list[dict] = []

    asyncio.run(
        livekit._forward_target_transcription(
            _FakeTranscriptionReader("  How can I help you today? "),
            session,
            conversation_ended=asyncio.Event(),
            captured_target_turns=captured,
        )
    )

    assert session.reply_inputs == ["How can I help you today?"]
    # The pipeline persists the turn into both contexts itself; a manual
    # history add here would double the transcript entry.
    assert session.history_adds == []
    assert len(captured) == 1
    assert captured[0]["content"] == "How can I help you today?"
    assert captured[0]["started_speaking_at"] <= captured[0]["stopped_speaking_at"]


def test_target_transcription_after_end_records_without_reply() -> None:
    session = _FakeReplySession()
    ended = asyncio.Event()
    ended.set()
    captured: list[dict] = []

    asyncio.run(
        livekit._forward_target_transcription(
            _FakeTranscriptionReader("Goodbye!"),
            session,
            conversation_ended=ended,
            captured_target_turns=captured,
        )
    )

    assert session.reply_inputs == []
    assert session.history_adds == [("user", "Goodbye!")]
    assert [c["content"] for c in captured] == ["Goodbye!"]


def test_target_transcription_falls_back_to_history_when_session_closing() -> None:
    session = _FakeReplySession(reply_error=RuntimeError("scheduling is paused"))
    captured: list[dict] = []

    asyncio.run(
        livekit._forward_target_transcription(
            _FakeTranscriptionReader("One last thing."),
            session,
            conversation_ended=asyncio.Event(),
            captured_target_turns=captured,
        )
    )

    assert session.history_adds == [("user", "One last thing.")]
    assert [c["content"] for c in captured] == ["One last thing."]


def test_merge_patches_target_turn_missing_stop_timing() -> None:
    # Native target turns reach history via generate_reply(user_input=) with no
    # audio metrics, so stop == start (zero duration) -> bot WPM/latency dead.
    messages = [
        {
            "role": "user",
            "content": "hi",
            "started_speaking_at": 1.0,
            "stopped_speaking_at": 2.0,
        },
        {
            "role": "assistant",
            "content": "hello there how can i help",
            "started_speaking_at": 3.0,
            "stopped_speaking_at": 3.0,
        },
    ]
    captured = [
        {
            "content": "hello there how can i help",
            "started_speaking_at": 2.5,
            "stopped_speaking_at": 5.0,
        }
    ]
    merged = livekit._merge_captured_target_turns(messages, captured)
    assert len(merged) == 2  # patched in place, not appended
    agent = next(m for m in merged if m["role"] == "assistant")
    assert agent["started_speaking_at"] == 2.5
    assert agent["stopped_speaking_at"] == 5.0


def test_merge_aggregates_partial_target_emissions() -> None:
    # One turn arrives as several partial/extended emissions -> min-start/max-stop.
    messages = [
        {
            "role": "assistant",
            "content": "can we look at options for that",
            "started_speaking_at": 10.0,
            "stopped_speaking_at": 10.0,
        },
    ]
    captured = [
        {
            "content": "can we look",
            "started_speaking_at": 9.0,
            "stopped_speaking_at": 9.5,
        },
        {
            "content": "can we look at options for that",
            "started_speaking_at": 9.2,
            "stopped_speaking_at": 12.0,
        },
    ]
    merged = livekit._merge_captured_target_turns(messages, captured)
    assert merged[0]["started_speaking_at"] == 9.0
    assert merged[0]["stopped_speaking_at"] == 12.0


def test_merge_does_not_override_real_target_timing() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "genuine turn",
            "started_speaking_at": 4.0,
            "stopped_speaking_at": 6.0,
        },
    ]
    captured = [
        {
            "content": "genuine turn",
            "started_speaking_at": 1.0,
            "stopped_speaking_at": 99.0,
        }
    ]
    merged = livekit._merge_captured_target_turns(messages, captured)
    assert merged[0]["started_speaking_at"] == 4.0
    assert merged[0]["stopped_speaking_at"] == 6.0


def test_merge_appends_trailing_target_turn_with_real_timing() -> None:
    messages = [
        {
            "role": "user",
            "content": "bye",
            "started_speaking_at": 5.0,
            "stopped_speaking_at": 6.0,
        }
    ]
    captured = [
        {
            "content": "take care now",
            "started_speaking_at": 7.0,
            "stopped_speaking_at": 8.0,
        }
    ]
    merged = livekit._merge_captured_target_turns(messages, captured)
    assert len(merged) == 2
    trailing = merged[-1]
    assert trailing["role"] == "assistant"
    assert trailing["content"] == "take care now"
    assert trailing["started_speaking_at"] == 7.0
    assert trailing["stopped_speaking_at"] == 8.0


class _FakeEventRoom:
    def __init__(self, participants: dict | None = None) -> None:
        self.listeners: dict[str, list] = {}
        if participants is not None:
            self.remote_participants = participants

    def on(self, event, callback=None):
        self.listeners.setdefault(event, []).append(callback)
        return callback

    def off(self, event, callback):
        self.listeners.get(event, []).remove(callback)

    def fire(self, event, *args) -> None:
        for callback in list(self.listeners.get(event, [])):
            callback(*args)


_BALANCED_HISTORY = SimpleNamespace(
    items=[
        SimpleNamespace(type="message", role="assistant", text_content="One"),
        SimpleNamespace(type="message", role="user", text_content="Two"),
        SimpleNamespace(type="message", role="assistant", text_content="Three"),
        SimpleNamespace(type="message", role="user", text_content="Four"),
        SimpleNamespace(type="message", role="assistant", text_content="Five"),
        SimpleNamespace(type="message", role="user", text_content="Six"),
    ]
)


def test_room_disconnect_ends_conversation() -> None:
    # A native target commonly hangs up by deleting the room; the simulator
    # sees a room disconnect, not a participant_disconnected, and the case
    # idled through the silence backstop before ending.
    room = _FakeEventRoom(participants={"t": SimpleNamespace(identity="target-agent")})

    class FakeSession:
        history = _BALANCED_HISTORY

        def on(self, _event, _callback):
            return None

    async def run() -> str:
        loop = asyncio.get_running_loop()
        loop.call_later(0.05, room.fire, "disconnected", "ROOM_DELETED")
        return await livekit._wait_for_conversation_end(
            room,
            FakeSession(),
            customer_agent=SimpleNamespace(end_requested=asyncio.Event()),
            target_identity="target-agent",
            timeout=5,
            conversation_direction="simulator_first",
            agent_first_silence_timeout_seconds=30,
        )

    reason = asyncio.run(run())

    assert reason == "room_disconnected"
    assert room.listeners["disconnected"] == []
    completed = livekit._conversation_outcome(
        reason,
        livekit._session_messages(SimpleNamespace(history=_BALANCED_HISTORY)),
        min_turn_messages=6,
    )
    assert completed.status == CaseStatus.COMPLETED
    failed = livekit._conversation_outcome(reason, [], min_turn_messages=4)
    assert failed.status == CaseStatus.FAILED
    assert failed.failure.code == "room_disconnected"
    assert failed.failure.retryable is True


def test_target_absent_at_watch_start_counts_as_disconnected() -> None:
    # The target can leave between readiness and listener registration; the
    # event is gone by then, so presence is rechecked once.
    room = _FakeEventRoom(participants={})

    class FakeSession:
        history = _BALANCED_HISTORY

        def on(self, _event, _callback):
            return None

    async def run() -> str:
        return await livekit._wait_for_conversation_end(
            room,
            FakeSession(),
            customer_agent=SimpleNamespace(end_requested=asyncio.Event()),
            target_identity="target-agent",
            timeout=5,
            conversation_direction="simulator_first",
            agent_first_silence_timeout_seconds=30,
        )

    assert asyncio.run(run()) == "target_disconnected"


def test_silence_backstop_parks_until_first_turn() -> None:
    # A call where nobody ever spoke belongs to the no_conversation monitor;
    # the backstop firing first mislabeled dead calls as merely settled.
    session = SimpleNamespace(history=SimpleNamespace(items=[]))

    async def run() -> None:
        task = asyncio.create_task(
            livekit._wait_for_conversation_silence(session, quiet_seconds=0.01)
        )
        done, _pending = await asyncio.wait({task}, timeout=0.3)
        assert not done
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_agent_first_silence_ignores_inflight_speech() -> None:
    # A turn lands in history only after its TTS finishes, so a long in-flight
    # utterance used to trip the agent-first stall watcher mid-speech.
    session = SimpleNamespace(
        agent_state="speaking",
        user_state="listening",
        history=SimpleNamespace(
            items=[
                SimpleNamespace(type="message", role="assistant", text_content="Hi"),
                SimpleNamespace(type="message", role="user", text_content="Hello"),
            ]
        ),
    )

    async def run() -> None:
        task = asyncio.create_task(
            livekit._wait_for_agent_first_silence(session, timeout_seconds=0.05)
        )
        await asyncio.sleep(0.2)
        assert not task.done()
        session.agent_state = "listening"
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(run())


def _order_probe_engine(monkeypatch, calls, *, dispatch_error=None):
    audio_kind = livekit.rtc.TrackKind.KIND_AUDIO

    class FakeRoom:
        def __init__(self):
            self.remote_participants = {
                "target": SimpleNamespace(
                    identity="target-agent",
                    sid="participant-target",
                    track_publications={
                        "track-target": SimpleNamespace(
                            sid="track-target", kind=audio_kind
                        )
                    },
                )
            }
            self.listeners = {}

        async def connect(self, url, token):
            calls.append("connect")

        async def disconnect(self):
            calls.append("disconnect")

        def on(self, event, callback=None):
            self.listeners.setdefault(event, []).append(callback)
            return callback

        def off(self, event, callback):
            self.listeners.get(event, []).remove(callback)

        def register_text_stream_handler(self, topic, handler=None):
            calls.append("register_text_handler")
            return handler

        def unregister_text_stream_handler(self, topic):
            pass

    class _Rooms:
        async def create_room(self, request):
            calls.append("create_room")

        async def delete_room(self, request):
            calls.append("delete_room")

    class _Dispatch:
        async def create_dispatch(self, request):
            if dispatch_error is not None:
                raise dispatch_error
            calls.append("dispatch")

    class _Api:
        def __init__(self):
            self.room = _Rooms()
            self.agent_dispatch = _Dispatch()

        async def aclose(self):
            pass

    class FakeSession:
        history = SimpleNamespace(
            items=[
                SimpleNamespace(type="message", role="user", text_content="hi"),
                SimpleNamespace(type="message", role="assistant", text_content="ok"),
            ]
        )

        def on(self, event, callback):
            return None

        def shutdown(self, *, drain=True):
            pass

    class FakeCustomerAgent:
        def __init__(self):
            self.end_requested = asyncio.Event()
            self.end_requested.set()

        async def start_session(self, _room, **_kwargs):
            calls.append("start_session")
            return FakeSession()

        def open_conversation(self):
            calls.append("open")

    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    monkeypatch.setattr(livekit.rtc, "Room", lambda: FakeRoom())
    monkeypatch.setattr(livekit.api, "LiveKitAPI", lambda *_a: _Api())
    monkeypatch.setattr(livekit, "AccessToken", _fake_access_token())
    engine = LiveKitEngine()

    async def _fake_create(_p, _s, **_kwargs):
        return FakeCustomerAgent(), None

    monkeypatch.setattr(engine, "_create_customer_agent", _fake_create)
    return engine


def test_managed_dispatch_waits_for_session_and_buffer_handler(monkeypatch) -> None:
    # Dispatching before the session + buffer handler are live loses a native
    # target's greeting for BOTH directions: the LiveKit client drops a
    # text-stream header that arrives with no registered handler.
    calls: list = []
    engine = _order_probe_engine(monkeypatch, calls)

    report = asyncio.run(
        engine.run(
            agent_definition=_agent(
                room_mode="managed",
                agent_name="registered-agent",
                target_participant_identity="target-agent",
            ),
            scenario=_scenario(),
            run_id="run_dispatch_order",
            min_turn_messages=2,
        )
    )

    assert report.results[0].metadata["status"] == CaseStatus.COMPLETED.value
    assert calls.index("connect") < calls.index("register_text_handler")
    assert calls.index("register_text_handler") < calls.index("dispatch")
    assert calls.index("start_session") < calls.index("dispatch")


def test_dispatch_failure_is_typed_preparing_failure(monkeypatch) -> None:
    calls: list = []
    engine = _order_probe_engine(
        monkeypatch,
        calls,
        dispatch_error=RuntimeError("twirp: agent not registered"),
    )

    report = asyncio.run(
        engine.run(
            agent_definition=_agent(
                room_mode="managed",
                agent_name="registered-agent",
                target_participant_identity="target-agent",
            ),
            scenario=_scenario(),
            run_id="run_dispatch_fail",
            min_turn_messages=2,
        )
    )

    metadata = report.results[0].metadata
    assert metadata["status"] == CaseStatus.FAILED.value
    assert metadata["failure"]["code"] == "livekit_dispatch_failed"
    assert metadata["failure"]["stage"] == "preparing"
    assert "delete_room" in calls
