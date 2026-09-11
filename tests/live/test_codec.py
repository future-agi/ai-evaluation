"""Phase 9A unit 3 — pure-numpy codec-survival stage (machinery tier).

No extras, no env flags, no network. Proves: G.711 μ-law/A-law round-trip
reproducibility; 8 kHz band-limit; seeded Gilbert-Elliott; default-ON vs ``none``
opt-out; text-rung raise; computed ``phone_survival`` field presence; and the
post-v1 Opus auto-skip via ``CodecUnsupportedError`` (G.711 never raises).
"""

from __future__ import annotations

import numpy as np
import pytest

from fi.alk.live import _codec


def _tone(hz: float, n: int = 24000, rate: int = 24000, amp: float = 0.5) -> np.ndarray:
    return (amp * np.sin(2 * np.pi * hz * np.arange(n) / rate)).astype(np.float32)


def test_g711_ulaw_roundtrip_reproducible():
    x = _tone(220)
    a = _codec.g711_ulaw_roundtrip(x)
    b = _codec.g711_ulaw_roundtrip(x)
    assert np.array_equal(a, b)
    # round-trip RMS error within the expected μ-law quantization band
    err = float(np.sqrt(((a - x) ** 2).mean()))
    assert err < 0.1


def test_g711_alaw_roundtrip_reproducible():
    x = _tone(300)
    a = _codec.g711_alaw_roundtrip(x)
    b = _codec.g711_alaw_roundtrip(x)
    assert np.array_equal(a, b)
    err = float(np.sqrt(((a - x) ** 2).mean()))
    assert err < 0.15


def test_resample_8k_band_limit():
    # a 6 kHz tone (above the 4 kHz telephony band) should be attenuated
    high = _tone(6000, n=24000, rate=24000)
    out = _codec.resample_8k(high, source_rate=24000)
    assert out.size == 8000  # 24k -> 8k decimation
    assert float(np.sqrt((out ** 2).mean())) < float(np.sqrt((high ** 2).mean()))


def test_gilbert_elliott_seeded_reproducible():
    x = _tone(440, n=8000, rate=8000)
    a, rec_a = _codec.gilbert_elliott_loss(x, sample_rate=8000, seed=1142)
    b, rec_b = _codec.gilbert_elliott_loss(x, sample_rate=8000, seed=1142)
    assert np.array_equal(a, b)
    assert rec_a["loss_realized"] == rec_b["loss_realized"]
    assert rec_a["loss_avg"] == 0.02
    assert rec_a["burst_ms"] == 100.0
    # a different seed differs
    c, _ = _codec.gilbert_elliott_loss(x, sample_rate=8000, seed=7)
    assert not np.array_equal(a, c)


def test_codec_profile_default_on_and_none_optout():
    user = _tone(220)
    agent = _tone(330)
    u, a, rec = _codec.apply_codec_profile(
        user, agent, profile="g711_ulaw_8k_ge", seed=5, sample_rate=24000
    )
    assert rec["applied"] is True
    assert rec["codec"] == "g711_ulaw"
    assert rec["resampled_to_hz"] == 8000
    assert u.size == 8000 and a.size == 8000
    # a computed phone_survival arrives via score_codec_survival (channel_simulated)
    ps = _codec.score_codec_survival(
        user, agent, codec="g711_ulaw", packet_loss="gilbert_elliott", seed=5
    )
    assert ps["tier"] == "channel_simulated"
    # none opt-out: no-op, no codec record applied
    u2, a2, rec2 = _codec.apply_codec_profile(
        user, agent, profile="none", seed=5, sample_rate=24000
    )
    assert rec2["applied"] is False
    assert np.array_equal(u2, user)


def test_codec_text_rung_raises():
    for fn in (
        lambda: _codec.g711_ulaw_roundtrip("hello"),
        lambda: _codec.g711_alaw_roundtrip("hello"),
        lambda: _codec.resample_8k("hello", source_rate=24000),
        lambda: _codec.gilbert_elliott_loss("hello", seed=1),
    ):
        with pytest.raises(ValueError):
            fn()


def test_phone_survival_computed_field_presence():
    user = _tone(220)
    agent = _tone(330)
    ps = _codec.score_codec_survival(
        user, agent, codec="g711_ulaw", packet_loss="gilbert_elliott", seed=3
    )
    assert ps["status"] in ("survives", "partial", "dies", "untested")
    assert ps["tier"] == "channel_simulated"
    # the three computed-evidence fields present at channel_simulated
    for k in ("pre_channel_success", "post_channel_success", "band_energy_lt_4khz"):
        assert k in ps
    # NOT a flat token — structured schema with reason
    assert "reason" in ps


def test_codec_unsupported_raises_for_opus_when_absent():
    user = _tone(220)
    agent = _tone(330)
    with pytest.raises(_codec.CodecUnsupportedError) as exc:
        _codec.apply_codec_profile(
            user, agent, profile="opus_nb_8k_ge", seed=1, sample_rate=24000
        )
    assert exc.value.codec == "opus_nb"
    assert "voice-codecs" in exc.value.install
    # G.711 never raises
    _codec.apply_codec_profile(user, agent, profile="g711_ulaw_8k_ge", seed=1, sample_rate=24000)


def test_facade_exports():
    from fi.alk.live import score_codec_survival, CodecUnsupportedError

    assert callable(score_codec_survival)
    assert issubclass(CodecUnsupportedError, RuntimeError)
