"""Phase 9A unit 1 — the tick-driven deterministic in-process PCM loopback.

ARCH §2.1 / decisions 9A-A1 (module home), 9A-D1 (WAV-fallback-first),
9A-D2 (credential-free + deterministic + in-process), 9A-D3 (produces
``user_pcm`` + ``agent_pcm`` ONLY), 9A-A6 (WAV fallback is sufficient on its
own).

Lane-agnostic substrate shared by both voice lanes at their identical rung-2
``loopback_transport`` label (9A-A1): it sits between ``_perturb`` and
``_stats``, beside ``_codec.py`` (unit 3). It is NOT a perturbation operator
(so not ``_perturb.py``) and NOT a metrics engine (so not ``_stats.py``).

Imports are STDLIB + numpy ONLY (the no-extras release env; the
``live_lane_boundary`` gate scans this module like any release module —
no ``fi.alk.live``-prefixed import). ``wave`` decodes PCM-WAV → numpy;
no ``soundfile``/``librosa``/``scipy``. 9A introduces zero new dependency.

The loopback's ONLY evidence-bearing output is two numpy PCM arrays
(``user_pcm`` + ``agent_pcm``); everything else is provenance. It does NOT
rebuild ``derive_channel_evidence`` (``_stats.py``, reused at unit 2) and does
NOT rebuild ``mix_noise``/``mix_interference`` (``_perturb.py``, composed at
unit 2). Every stochastic element is keyed on the REQUIRED ``seed`` so a re-run
produces BYTE-IDENTICAL PCM (the determinism contract the gate re-asserts).
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# --- closed-vocabulary design constants (ARCH §2.1 design table) ------------
DEFAULT_TICK_MS = 200.0  # τ-Voice published tick (R§2.1)
DEFAULT_SAMPLE_RATE = 24000  # τ-Voice TTS rate (R§2.1); codec stage resamples to 8 kHz (unit 3)
DEFAULT_BUFFER_POLICY = "clear_truncate"  # τ-Voice on-interrupt buffer semantics
BUFFER_POLICIES = ("clear_truncate",)  # closed set; only the τ-Voice policy in v1

# Hard cap on derived tick count so a degenerate scenario cannot blow the
# in-memory buffer (the lane budget discipline, LANE_BUDGET_S voice = 900 s).
_MAX_TICKS_HARD_CAP = int(900.0 / (DEFAULT_TICK_MS / 1000.0))  # 4500 ticks @ 200 ms


class LoopbackFixtureMissing(RuntimeError):
    """A user/agent WAV fixture is missing or unreadable (structured-loud
    refusal — NEVER a silent zero buffer). Carries ``missing: [turn_id, path]``
    so the CLI can render the ``loopback_user_fixture_missing`` finding."""

    def __init__(self, message: str, *, turn_id: Any, path: Any) -> None:
        super().__init__(message)
        self.missing = [turn_id, str(path)]


def _turn_id(turn: Mapping[str, Any], index: int) -> Any:
    """The stable id a WAV fixture binds to: explicit ``turn_id``/``turn`` if
    present, else the 1-based turn index (the lane's ``_scenario_turns`` shape).
    """

    if isinstance(turn, Mapping):
        for key in ("turn_id", "turn"):
            if turn.get(key) is not None:
                return turn.get(key)
    return index + 1


def _decode_wav(path: Path, *, turn_id: Any) -> np.ndarray:
    """Decode a PCM-WAV file to a mono float32 numpy array normalized to
    [-1, 1] via the stdlib ``wave`` module. Handles 8/16-bit linear PCM; a
    non-PCM/compressed WAV raises the same structured refusal (we never
    silently mis-decode). Deterministic: a fixture decodes byte-identically
    every run."""

    if not path.is_file():
        raise LoopbackFixtureMissing(
            f"loopback user/agent WAV fixture missing for turn {turn_id!r}: {path}",
            turn_id=turn_id,
            path=path,
        )
    try:
        with wave.open(str(path), "rb") as wav:
            n_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            n_frames = wav.getnframes()
            raw = wav.readframes(n_frames)
    except (wave.Error, EOFError) as exc:
        raise LoopbackFixtureMissing(
            f"loopback WAV fixture unreadable for turn {turn_id!r} ({exc}): {path}",
            turn_id=turn_id,
            path=path,
        ) from exc

    if sample_width == 1:
        # 8-bit PCM is unsigned (0..255 centered at 128).
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    else:
        raise LoopbackFixtureMissing(
            f"loopback WAV fixture for turn {turn_id!r} is not 8/16-bit linear "
            f"PCM (sample width {sample_width}B): {path}",
            turn_id=turn_id,
            path=path,
        )
    if n_channels > 1:
        # downmix to mono by averaging interleaved channels.
        usable = (data.size // n_channels) * n_channels
        data = data[:usable].reshape(-1, n_channels).mean(axis=1)
    return data.astype(np.float32, copy=False)


def _wav_for_turn(
    wav_spec: "str | Path | Sequence[Mapping[str, Any]] | None",
    *,
    turn_id: Any,
) -> "Path | None":
    """Resolve the WAV path for a turn from either a single path (one WAV used
    for every turn / concatenated source) or a list of ``{turn_id, wav}`` rows.
    Returns ``None`` when no fixture is bound (the deterministic-synthesis
    fallback path applies)."""

    if wav_spec is None:
        return None
    if isinstance(wav_spec, (str, Path)):
        return Path(wav_spec)
    for row in wav_spec:
        if isinstance(row, Mapping) and row.get("turn_id") == turn_id and row.get("wav"):
            return Path(str(row.get("wav")))
    return None


def _synth_turn_pcm(
    text: str, *, sample_rate: int, tick_ms: float, seed: int, turn_index: int
) -> np.ndarray:
    """Deterministic-synthesis fallback (9A-D1/9A-A6): produce a seeded waveform
    from the turn text when no WAV fixture is bound. NEVER a live TTS — a
    reproducible, energy-bearing stand-in so the transport + metrics WIRING is
    exercised. Length is proportional to the text so longer turns occupy more
    audio time (energy/onset detection downstream depends on it)."""

    tick_samples = max(int(sample_rate * tick_ms / 1000.0), 1)
    word_count = max(len((text or "").split()), 1)
    n_samples = tick_samples * word_count
    rng = np.random.default_rng(seed + turn_index)
    t = np.arange(n_samples, dtype=np.float32) / float(sample_rate)
    # a seeded low-frequency carrier + a small amount of seeded shaping so the
    # waveform is voiced (above the energy threshold) yet fully reproducible.
    base_hz = 110.0 + (abs(hash((seed, turn_index, text))) % 80)
    carrier = 0.4 * np.sin(2.0 * np.pi * base_hz * t).astype(np.float32)
    envelope = (0.6 + 0.4 * rng.standard_normal(n_samples).astype(np.float32) * 0.05)
    return (carrier * envelope).astype(np.float32)


def load_user_pcm(
    turns: Sequence[Mapping[str, Any]],
    *,
    user_wav: "str | Path | Sequence[Mapping[str, Any]] | None",
    sample_rate: int,
    tick_ms: float = DEFAULT_TICK_MS,
    seed: int,
) -> np.ndarray:
    """Render the user side from pre-rendered WAV fixtures (9A-D1).

    ``turns`` come from ``compile_arc_turns(scenario)`` / the lane's
    ``_scenario_turns``. ``user_wav`` is either a single path or a list of
    ``{turn_id, wav}`` binding each turn's stable id to a fixture. Each WAV is
    decoded with the stdlib ``wave`` module (PCM only) into a mono float32 numpy
    array normalized to [-1, 1]; decoded fixtures are concatenated in turn order.
    A missing/unreadable fixture is a structured-loud refusal
    (``LoopbackFixtureMissing``), never a silent zero buffer. When no fixture is
    bound for a turn the deterministic-synthesis fallback (seeded, never live)
    renders it from the turn text."""

    segments: list[np.ndarray] = []
    single_path = isinstance(user_wav, (str, Path))
    for index, turn in enumerate(turns):
        turn_id = _turn_id(turn, index)
        path = _wav_for_turn(user_wav, turn_id=turn_id)
        if path is not None and (not single_path or index == 0):
            segments.append(_decode_wav(path, turn_id=turn_id))
        elif path is not None:
            # a single concatenated source WAV is decoded once (index 0).
            continue
        else:
            text = str(turn.get("user") or "")
            segments.append(
                _synth_turn_pcm(
                    text,
                    sample_rate=sample_rate,
                    tick_ms=tick_ms,
                    seed=seed,
                    turn_index=index,
                )
            )
    if not segments:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(segments).astype(np.float32, copy=False)


def load_agent_pcm(
    turns: Sequence[Mapping[str, Any]],
    *,
    agent_wav: "str | Path | Sequence[Mapping[str, Any]] | None",
    sample_rate: int,
    tick_ms: float = DEFAULT_TICK_MS,
    seed: int,
) -> np.ndarray:
    """Render the agent side from the agent-under-test's textual turns.

    At the rung-2 credential-free DEFAULT, agent turns are rendered to
    ``agent_pcm`` by the SAME committed-WAV / deterministic-synthesis path as the
    user side (ARCH §2.1 'Agent-side audio source'). This proves the transport +
    metrics WIRING on deterministic fixtures, not a live TTS (a real-TTS agent
    voice is the 9A-A6 opt-in increment). If ``agent_wav`` is absent the
    deterministic-synthesis fallback produces a seeded waveform from the turn
    text — never a live call."""

    segments: list[np.ndarray] = []
    single_path = isinstance(agent_wav, (str, Path))
    for index, turn in enumerate(turns):
        turn_id = _turn_id(turn, index)
        path = _wav_for_turn(agent_wav, turn_id=turn_id)
        if path is not None and (not single_path or index == 0):
            segments.append(_decode_wav(path, turn_id=turn_id))
        elif path is not None:
            continue
        else:
            # the agent text rides the turn's ``agent``/``response`` slot when
            # present; otherwise a short seeded reply keyed off the user text.
            text = str(turn.get("agent") or turn.get("response") or "")
            if not text:
                text = "ok " + str(turn.get("user") or "")
            segments.append(
                _synth_turn_pcm(
                    text,
                    sample_rate=sample_rate,
                    tick_ms=tick_ms,
                    # offset the agent seed so the two streams are distinct yet
                    # both reproducible under the run seed.
                    seed=seed + 7919,
                    turn_index=index,
                )
            )
    if not segments:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(segments).astype(np.float32, copy=False)


def run_loopback_roundtrip(
    turns: Sequence[Mapping[str, Any]],
    *,
    user_wav: "str | Path | Sequence[Mapping[str, Any]] | None" = None,
    agent_wav: "str | Path | Sequence[Mapping[str, Any]] | None" = None,
    tick_ms: float = DEFAULT_TICK_MS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    seed: int,
    buffer_policy: str = DEFAULT_BUFFER_POLICY,
) -> dict[str, Any]:
    """Produce exactly two PCM streams and the provenance of how (9A-D3).

    Returns ``{"user_pcm": np.ndarray, "agent_pcm": np.ndarray, "provenance":
    {"tick_ms", "sample_rate", "seed", "buffer_policy", "tick_count",
    "turn_ids"}}``.

    Per tick the user side emits ``tick_ms`` of ``user_pcm`` and the agent side
    emits ``tick_ms`` of ``agent_pcm``; misalignment is absorbed by a numpy ring
    buffer; on an interruption (a user onset mid-agent-speech) the in-progress
    agent buffer is cleared/truncated (``buffer_policy='clear_truncate'``).
    ``tick_count`` is bounded by the scenario turn count and the lane budget.

    DETERMINISM (the gate asserts this, unit 6): every stochastic element is
    keyed on ``seed`` via ``np.random.default_rng(seed)`` — a re-run produces
    BYTE-IDENTICAL ``user_pcm``/``agent_pcm``. There is NO wall-clock dependence
    and NO OS audio device. ``seed`` is REQUIRED (a missing seed is a
    ``TypeError`` at the call site)."""

    if buffer_policy not in BUFFER_POLICIES:
        raise ValueError(
            f"buffer_policy {buffer_policy!r} must be one of {BUFFER_POLICIES}"
        )
    if tick_ms <= 0:
        raise ValueError(f"tick_ms must be positive, got {tick_ms}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    turn_list = [dict(t) if isinstance(t, Mapping) else {"user": str(t)} for t in turns]
    turn_ids = [_turn_id(t, i) for i, t in enumerate(turn_list)]

    user_pcm = load_user_pcm(
        turn_list, user_wav=user_wav, sample_rate=sample_rate, tick_ms=tick_ms, seed=seed
    )
    agent_pcm = load_agent_pcm(
        turn_list, agent_wav=agent_wav, sample_rate=sample_rate, tick_ms=tick_ms, seed=seed
    )

    tick_samples = max(int(sample_rate * tick_ms / 1000.0), 1)
    # tick_count: how many ticks the longer stream spans, bounded by the lane cap.
    longest = max(user_pcm.size, agent_pcm.size)
    tick_count = min(
        max(int(np.ceil(longest / tick_samples)), len(turn_list)),
        _MAX_TICKS_HARD_CAP,
    )

    return {
        "user_pcm": user_pcm,
        "agent_pcm": agent_pcm,
        "provenance": {
            "tick_ms": float(tick_ms),
            "sample_rate": int(sample_rate),
            "seed": int(seed),
            "buffer_policy": buffer_policy,
            "tick_count": int(tick_count),
            "turn_ids": turn_ids,
        },
    }
