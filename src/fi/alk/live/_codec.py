"""Phase 9A unit 3 — pure-numpy codec-survival stage (the Phase-12-reserved home).

ARCH §2.2 / decisions 9A-A2 (G.711 μ-law/A-law PURE-NUMPY v1, ZERO new dep;
Opus-NB/AMR a post-v1 build-dep extra, auto-skip), 9A-A3 (module home), 9A-A7
(no neural codec; registry-extensible), 9A-A11 (default-ON at rung-2), 9A-A12
(facade ``score_codec_survival`` / ``CodecUnsupportedError``), 9A-A13 (computed
``phone_survival`` 4 frozen + 3 computed fields), 9A-D4.

Imports: numpy + STDLIB ONLY. G.711 μ-law/A-law are vectorized numpy companding
tables — NOT ``audioop`` (deprecated 3.11, REMOVED 3.13 per PEP 594; the kit's
dev interpreter is 3.14 → ``audioop`` cannot back it). 8 kHz resample is
pure-numpy decimation (no scipy). Gilbert-Elliott packet loss is seeded numpy.
The codec/packet operators raise on text-rung input exactly as the ``_perturb``
acoustic operators do (a codec round-trip on a transcript is a contract error).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# --- closed-vocabulary constants (mirrored in trinity.py, cross-pinned by a
# unit test — the GUNA_AXES cross-pin pattern; trinity is the gate-pinned home).
V1_VOICE_CODECS = ("g711_ulaw", "g711_alaw", "opus_nb", "amr_nb")
# g711_* = v1 pure-numpy; opus_nb/amr_nb = post-v1 build-dep, auto-skip.
_V1_NUMPY_CODECS = ("g711_ulaw", "g711_alaw")
_POST_V1_CODECS = ("opus_nb", "amr_nb")
V1_VOICE_PACKET_LOSS_MODELS = ("gilbert_elliott",)
V1_VOICE_CODEC_PROFILES = (
    "g711_ulaw_8k_ge",
    "g711_alaw_8k_ge",
    "opus_nb_8k_ge",
    "amr_nb_8k_ge",
    "none",
)
# named bundle → (codec, packet_loss_model)
_PROFILE_BUNDLE = {
    "g711_ulaw_8k_ge": ("g711_ulaw", "gilbert_elliott"),
    "g711_alaw_8k_ge": ("g711_alaw", "gilbert_elliott"),
    "opus_nb_8k_ge": ("opus_nb", "gilbert_elliott"),
    "amr_nb_8k_ge": ("amr_nb", "gilbert_elliott"),
}
_TELEPHONY_RATE = 8000

# the post-v1 install path named in the auto-skip refusal (9A-A2).
_POST_V1_EXTRA = "agent-learning-kit[voice-codecs]"


class CodecUnsupportedError(RuntimeError):
    """Raised when a requested codec is in ``V1_VOICE_CODECS`` but its build-dep
    extra is absent (the post-v1 ``opus_nb``/``amr_nb`` path). Callers can
    ``except CodecUnsupportedError`` to auto-skip exactly as the framework lanes
    skip on a missing extra (the ``LANE_EXTRAS`` discipline). G.711 / packet-loss
    never raise (numpy, always available)."""

    def __init__(self, message: str, *, codec: str, install: str) -> None:
        super().__init__(message)
        self.codec = codec
        self.install = install


def _require_pcm(pcm: Any, *, where: str) -> np.ndarray:
    """Type-guard the input as numpy PCM; a text/str input raises a ValueError
    mirroring ``_perturb.py``'s rung-wall message (the §3.4 generalization)."""

    if isinstance(pcm, (str, bytes)):
        raise ValueError(
            f"{where} needs a real audio channel (rung 2 loopback transport or "
            "above); a text/transcript input is a contract error"
        )
    arr = np.asarray(pcm, dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def resample_8k(pcm: np.ndarray, *, source_rate: int) -> np.ndarray:
    """24 kHz → 8 kHz telephony band-limit via pure-numpy anti-alias + decimation
    (NO scipy). Anti-alias by a simple moving-average low-pass at the target
    Nyquist (4 kHz), then decimate to ``target_rate=8000``. Deterministic."""

    samples = _require_pcm(pcm, where="resample_8k")
    if source_rate <= 0:
        raise ValueError("source_rate must be positive")
    if samples.size == 0 or source_rate == _TELEPHONY_RATE:
        return samples
    factor = source_rate / float(_TELEPHONY_RATE)
    if factor <= 1.0:
        # upsampling is out of scope for the telephony band-limit; pass through.
        return samples
    # anti-alias: moving average window ~ decimation factor (cuts > 4 kHz energy)
    window = max(int(round(factor)), 1)
    if window > 1:
        kernel = np.ones(window, dtype=np.float32) / float(window)
        samples = np.convolve(samples, kernel, mode="same").astype(np.float32)
    n_out = int(samples.size / factor)
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    idx = (np.arange(n_out, dtype=np.float64) * factor).astype(np.int64)
    idx = np.clip(idx, 0, samples.size - 1)
    return samples[idx].astype(np.float32, copy=False)


# --- G.711 μ-law companding (vectorized numpy, ITU-T G.711) -----------------
_MU = 255.0
_ALAW_A = 87.6


def g711_ulaw_roundtrip(pcm: np.ndarray) -> np.ndarray:
    """μ-law companding round-trip via vectorized numpy (NOT audioop): encode
    (linear → μ-law 8-bit code) then decode (μ-law code → linear). Lossy and
    deterministic — the v1 telephony codec."""

    x = _require_pcm(pcm, where="g711_ulaw_roundtrip")
    if x.size == 0:
        return x
    x = np.clip(x, -1.0, 1.0)
    # encode: μ-law compression → quantize to 8-bit code
    sign = np.sign(x)
    magnitude = np.log1p(_MU * np.abs(x)) / np.log1p(_MU)
    code = np.round(magnitude * 127.0).astype(np.int32)  # 8-bit magnitude quant
    code = np.clip(code, 0, 127)
    # decode: μ-law expansion of the quantized code
    mag_q = code.astype(np.float32) / 127.0
    decoded = sign * (np.expm1(mag_q * np.log1p(_MU)) / _MU)
    return decoded.astype(np.float32, copy=False)


def g711_alaw_roundtrip(pcm: np.ndarray) -> np.ndarray:
    """A-law companding round-trip, same shape as the μ-law path."""

    x = _require_pcm(pcm, where="g711_alaw_roundtrip")
    if x.size == 0:
        return x
    x = np.clip(x, -1.0, 1.0)
    sign = np.sign(x)
    ax = np.abs(x)
    ln_a = 1.0 + np.log(_ALAW_A)
    low = ax < (1.0 / _ALAW_A)
    compressed = np.where(
        low,
        (_ALAW_A * ax) / ln_a,
        (1.0 + np.log(np.clip(_ALAW_A * ax, 1e-12, None))) / ln_a,
    )
    code = np.clip(np.round(compressed * 127.0).astype(np.int32), 0, 127)
    comp_q = code.astype(np.float32) / 127.0
    # A-law expansion (inverse)
    thresh = 1.0 / ln_a
    decoded_mag = np.where(
        comp_q < thresh,
        (comp_q * ln_a) / _ALAW_A,
        np.exp(comp_q * ln_a - 1.0) / _ALAW_A,
    )
    decoded = sign * decoded_mag
    return decoded.astype(np.float32, copy=False)


def gilbert_elliott_loss(
    pcm: np.ndarray,
    *,
    loss_avg: float = 0.02,
    burst_ms: float = 100.0,
    sample_rate: int = _TELEPHONY_RATE,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Two-state burst packet loss (the τ-Voice default 2 %/100 ms recipe,
    R§2.2). Pure-numpy, seeded via ``np.random.default_rng(seed)`` →
    reproducible under seed. Returns the degraded PCM + a record
    ``{model, loss_avg, burst_ms, seed, loss_realized}`` for the artifact/replay.
    """

    x = _require_pcm(pcm, where="gilbert_elliott_loss")
    if x.size == 0 or loss_avg <= 0:
        return x, {
            "model": "gilbert_elliott",
            "loss_avg": float(loss_avg),
            "burst_ms": float(burst_ms),
            "seed": int(seed),
            "loss_realized": 0.0,
        }
    rng = np.random.default_rng(seed)
    frame_samples = max(int(sample_rate * 20.0 / 1000.0), 1)  # 20 ms frames
    n_frames = max(int(np.ceil(x.size / frame_samples)), 1)
    # two-state Markov chain: G(ood) and B(ad). Mean burst length = burst_ms/20ms
    # frames ⇒ p(B→G) = 1/burst_frames; steady-state loss = loss_avg ⇒ derive
    # p(G→B) from the balance equation π_B = loss_avg.
    burst_frames = max(burst_ms / 20.0, 1.0)
    p_bg = 1.0 / burst_frames  # leave Bad
    # π_B = p_gb / (p_gb + p_bg) = loss_avg  ⇒  p_gb = loss_avg * p_bg / (1-loss_avg)
    p_gb = (loss_avg * p_bg) / max(1.0 - loss_avg, 1e-9)
    degraded = x.copy()
    bad = False
    lost_frames = 0
    for f in range(n_frames):
        if bad:
            if rng.random() < p_bg:
                bad = False
        else:
            if rng.random() < p_gb:
                bad = True
        if bad:
            start = f * frame_samples
            degraded[start : start + frame_samples] = 0.0
            lost_frames += 1
    record = {
        "model": "gilbert_elliott",
        "loss_avg": float(loss_avg),
        "burst_ms": float(burst_ms),
        "seed": int(seed),
        "loss_realized": round(lost_frames / float(n_frames), 6),
    }
    return degraded.astype(np.float32, copy=False), record


def _codec_roundtrip(pcm: np.ndarray, *, codec: str) -> np.ndarray:
    if codec == "g711_ulaw":
        return g711_ulaw_roundtrip(pcm)
    if codec == "g711_alaw":
        return g711_alaw_roundtrip(pcm)
    if codec in _POST_V1_CODECS:
        raise CodecUnsupportedError(
            f"codec {codec!r} is a post-v1 build-dep extra and is not installed; "
            f"install {_POST_V1_EXTRA} to enable it (v1 ships g711_ulaw/g711_alaw "
            "as the required pure-numpy codecs)",
            codec=codec,
            install=_POST_V1_EXTRA,
        )
    raise ValueError(f"unknown codec {codec!r}; expected one of {V1_VOICE_CODECS}")


def _apply_channel(
    pcm: np.ndarray, *, codec: str, packet_loss: str, seed: int, sample_rate: int
) -> tuple[np.ndarray, dict]:
    """Resample → codec round-trip → packet loss, returning degraded PCM + the
    codec_round_trip record (UI-UX §3.3 shape)."""

    if packet_loss not in V1_VOICE_PACKET_LOSS_MODELS:
        raise ValueError(
            f"packet_loss_model {packet_loss!r} must be one of "
            f"{V1_VOICE_PACKET_LOSS_MODELS}"
        )
    resampled = resample_8k(pcm, source_rate=sample_rate)
    coded = _codec_roundtrip(resampled, codec=codec)
    degraded, loss_record = gilbert_elliott_loss(
        coded, sample_rate=_TELEPHONY_RATE, seed=seed
    )
    record = {
        "codec": codec,
        "resampled_to_hz": _TELEPHONY_RATE,
        "source_rate_hz": int(sample_rate),
        "packet_loss": loss_record,
        "seed": int(seed),
    }
    return degraded, record


def apply_codec_profile(
    user_pcm: np.ndarray,
    agent_pcm: np.ndarray,
    *,
    profile: str,
    seed: int,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply a named codec profile (codec + resample + packet-loss bundle) to
    both streams. ``profile='none'`` is a no-op (clean-PCM loopback). Raises
    ``CodecUnsupportedError`` for ``opus_nb_8k_ge``/``amr_nb_8k_ge`` when the
    extra is absent (post-v1). Returns degraded (user_pcm, agent_pcm) + the
    codec_round_trip record (UI-UX §3.3 shape)."""

    if profile not in V1_VOICE_CODEC_PROFILES:
        raise ValueError(
            f"codec_profile {profile!r} must be one of {V1_VOICE_CODEC_PROFILES}"
        )
    if profile == "none":
        return (
            _require_pcm(user_pcm, where="apply_codec_profile"),
            _require_pcm(agent_pcm, where="apply_codec_profile"),
            {"profile": "none", "applied": False},
        )
    codec, packet_loss = _PROFILE_BUNDLE[profile]
    user_deg, user_rec = _apply_channel(
        user_pcm, codec=codec, packet_loss=packet_loss, seed=seed, sample_rate=sample_rate
    )
    agent_deg, agent_rec = _apply_channel(
        agent_pcm,
        codec=codec,
        packet_loss=packet_loss,
        seed=seed + 1,
        sample_rate=sample_rate,
    )
    record = {
        "profile": profile,
        "applied": True,
        "codec": codec,
        "packet_loss_model": packet_loss,
        "resampled_to_hz": _TELEPHONY_RATE,
        "source_rate_hz": int(sample_rate),
        "user": user_rec,
        "agent": agent_rec,
        "seed": int(seed),
    }
    return user_deg, agent_deg, record


def _band_energy_lt_4khz(pcm: np.ndarray, *, sample_rate: int) -> float:
    """Fraction of signal energy below 4 kHz the telephony codec preserves
    (CodecAttack <4 kHz framing, R§2.2)."""

    x = _require_pcm(pcm, where="band_energy")
    if x.size < 2:
        return 0.0
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sample_rate))
    total = float(spectrum.sum())
    if total <= 0:
        return 0.0
    low = float(spectrum[freqs < 4000.0].sum())
    return round(low / total, 6)


def _success_score(pcm: np.ndarray) -> float:
    """A reproducible proxy success score for the channel pre/post twins: RMS
    energy normalized to [0,1]. The clean twin scores higher than the degraded
    twin (the channel attenuates / drops frames), so the pre→post delta is the
    survival evidence. Deterministic — no model call."""

    x = _require_pcm(pcm, where="success_score")
    if x.size == 0:
        return 0.0
    rms = float(np.sqrt((x.astype(np.float64) ** 2).mean()))
    return round(min(rms * 2.0, 1.0), 6)


def score_codec_survival(
    user_pcm: np.ndarray,
    agent_pcm: np.ndarray,
    *,
    codec: str,
    packet_loss: str,
    seed: int,
    sample_rate: int = 24000,
    pre_channel_success: float | None = None,
) -> dict:
    """Re-validate an acoustic claim through the telephony channel; return the
    COMPUTED ``phone_survival`` object (9A-A13). The Phase-12 frozen 4 fields
    (``status``/``tier``/``scope_label?``/``reason``) keep their vocabulary
    unchanged; 3 OPTIONAL computed-evidence fields
    (``pre_channel_success``/``post_channel_success``/``band_energy_lt_4khz``)
    are present ONLY when ``tier ∈ {channel_simulated, channel_live}``."""

    _require_pcm(user_pcm, where="score_codec_survival")  # type-guard the user side
    agent = _require_pcm(agent_pcm, where="score_codec_survival")
    # the clean twin success (BEFORE the channel) — measured on the agent side
    # (the side carrying the claim under test) unless supplied by the caller.
    pre = (
        float(pre_channel_success)
        if pre_channel_success is not None
        else _success_score(agent)
    )
    agent_deg, channel_record = _apply_channel(
        agent, codec=codec, packet_loss=packet_loss, seed=seed, sample_rate=sample_rate
    )
    post = _success_score(agent_deg)
    band = _band_energy_lt_4khz(agent_deg, sample_rate=_TELEPHONY_RATE)

    # status derives from the pre→post delta
    if pre <= 0:
        status = "untested"
    else:
        retained = post / pre if pre > 0 else 0.0
        if retained >= 0.85:
            status = "survives"
        elif retained >= 0.4:
            status = "partial"
        else:
            status = "dies"

    return {
        "status": status,
        "tier": "channel_simulated",
        "reason": (
            f"codec={codec} packet_loss={packet_loss} "
            f"loss_realized={channel_record['packet_loss']['loss_realized']} "
            f"pre={pre} post={post} (8 kHz telephony channel, simulated)"
        ),
        "pre_channel_success": round(pre, 6),
        "post_channel_success": round(post, 6),
        "band_energy_lt_4khz": band,
    }
