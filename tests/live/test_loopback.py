"""Phase 9A unit 1 — tick-driven deterministic PCM loopback (machinery tier).

No extras, no env flags, no network — pure stdlib + numpy. Proves: byte-identical
determinism under seed; only-two-PCM-streams contract (9A-D3); loud missing-fixture
refusal; tick/rate provenance; and that the two streams feed
``derive_channel_evidence`` unmodified (the reuse seam, NOT a rebuild).
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from fi.alk.live import _loopback
from fi.alk.live._stats import derive_channel_evidence

_TURNS = [
    {"user": "Hello, can you hear me clearly on this call?"},
    {"user": "Great, please confirm my appointment for tomorrow morning."},
    {"user": "And send me the confirmation to my new account."},
]


def _write_wav(path: Path, samples: np.ndarray, *, sample_rate: int = 24000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())
    return path


def test_loopback_determinism_byte_identical(tmp_path):
    wav = _write_wav(
        tmp_path / "user.wav",
        0.5 * np.sin(2 * np.pi * 220 * np.arange(24000) / 24000.0).astype(np.float32),
    )
    a = _loopback.run_loopback_roundtrip(_TURNS, user_wav=wav, seed=1142)
    b = _loopback.run_loopback_roundtrip(_TURNS, user_wav=wav, seed=1142)
    assert np.array_equal(a["user_pcm"], b["user_pcm"])
    assert np.array_equal(a["agent_pcm"], b["agent_pcm"])
    assert a["provenance"] == b["provenance"]
    # a different seed produces a different agent stream (synthesis fallback path)
    c = _loopback.run_loopback_roundtrip(_TURNS, user_wav=wav, seed=99)
    assert not np.array_equal(a["agent_pcm"], c["agent_pcm"])


def test_loopback_produces_only_two_pcm_streams():
    result = _loopback.run_loopback_roundtrip(_TURNS, seed=7)
    assert set(result) == {"user_pcm", "agent_pcm", "provenance"}
    assert isinstance(result["user_pcm"], np.ndarray)
    assert isinstance(result["agent_pcm"], np.ndarray)
    # NO channels block, NO derived metrics — the loopback does not compute them
    assert "channels" not in result
    assert "derived" not in result


def test_loopback_missing_fixture_refuses_loud(tmp_path):
    missing = tmp_path / "does_not_exist.wav"
    with pytest.raises(_loopback.LoopbackFixtureMissing) as exc:
        _loopback.run_loopback_roundtrip(_TURNS, user_wav=missing, seed=3)
    assert exc.value.missing[1] == str(missing)
    # turn_id is named
    assert exc.value.missing[0] is not None


def test_loopback_tick_and_rate_provenance():
    result = _loopback.run_loopback_roundtrip(
        _TURNS, seed=5, tick_ms=200.0, sample_rate=24000
    )
    prov = result["provenance"]
    assert prov["tick_ms"] == 200.0
    assert prov["sample_rate"] == 24000
    assert prov["seed"] == 5
    assert prov["buffer_policy"] == "clear_truncate"
    assert prov["tick_count"] >= len(_TURNS)
    # per-tick PCM length == int(sample_rate * tick_ms / 1000)
    tick_samples = int(24000 * 200.0 / 1000.0)
    assert tick_samples == 4800
    # synthesis fallback renders each turn in whole-tick multiples
    assert result["user_pcm"].size % tick_samples == 0


def test_loopback_feeds_derive_channel_evidence():
    result = _loopback.run_loopback_roundtrip(_TURNS, seed=11, sample_rate=24000)
    channels = derive_channel_evidence(
        result["user_pcm"], result["agent_pcm"], sample_rate=24000
    )
    # the §1.2 keys are produced — proving the reuse seam, not a rebuild
    for key in (
        "barge_in_latency_ms",
        "overlap_total_ms",
        "overlap_segments",
        "post_interrupt_recovery_turns",
        "ttfb_ms",
        "frame_ms",
        "energy_threshold_db",
    ):
        assert key in channels


def test_loopback_seed_required():
    with pytest.raises(TypeError):
        _loopback.run_loopback_roundtrip(_TURNS)  # type: ignore[call-arg]


def test_loopback_decodes_8bit_pcm(tmp_path):
    # 8-bit unsigned PCM is a common sub-format `wave` exposes
    path = tmp_path / "u8.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = (np.sin(2 * np.pi * 200 * np.arange(4800) / 24000.0) * 0.4 + 0.0)
    u8 = np.clip((samples * 127.0) + 128.0, 0, 255).astype(np.uint8)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(1)
        wav.setframerate(24000)
        wav.writeframes(u8.tobytes())
    result = _loopback.run_loopback_roundtrip([{"user": "hi"}], user_wav=path, seed=1)
    assert result["user_pcm"].size == 4800
