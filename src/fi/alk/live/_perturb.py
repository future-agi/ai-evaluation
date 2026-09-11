"""Kit-native perturbation operators for the ``live_stressed`` sub-lane
(guide §3.6 / PRD §4.2). Imports: stdlib + numpy only.

Operators are deterministic under a recorded seed so stressed runs replay.
Flipping ANY operator stamps the run ``evidence_class="live_stressed"`` and
records the operator list in the ``live_lane.perturbations`` stanza; the run
links its clean twin (``paired_clean_run``).
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

import numpy as np

PERTURBATION_OPERATORS = (
    "noise",
    "interference",
    "asr_error",
    "accent",
    "homophone",
    "code_switch",
    "near_dup",
    "reverb_blend",
)
# Operators applicable to text-rung input (rung 1: TranscriptionFrames /
# scripted user text). Acoustic operators need a real audio channel (rung 2+).
TEXT_RUNG_OPERATORS = ("asr_error", "homophone", "code_switch", "near_dup")
# Acoustic operators applied to the rung-2 loopback PCM channel (Phase-12 12C
# rung-2 / ARCH §2c). They activate ONLY when the lane runs at rung-2 and hands
# ``_perturb`` a real PCM ``np.ndarray``; at text-rung they raise exactly as the
# pre-existing ``noise``/``interference`` did. ``reverb_blend`` is the operator
# the Phase-12 BBG deferred (the AudioHijack reverberation-hiding insight, used
# DEFENSIVELY as a test payload against an agent the user is authorized to test).
ACOUSTIC_RUNG_OPERATORS = ("noise", "interference", "reverb_blend")

_VOWELS = "aeiou"

# Spoken-form pairs whose transcripts diverge (rung-1 stand-in for the
# homophone-divergence surface; defensive test payloads, PRD §2 boundary).
HOMOPHONE_TABLE = {
    "to": "two", "two": "to", "for": "four", "four": "for",
    "right": "write", "write": "right", "buy": "by", "by": "buy",
    "cell": "sell", "sell": "cell", "here": "hear", "hear": "here",
    "new": "knew", "knew": "new", "wait": "weight", "weight": "wait",
    "aloud": "allowed", "allowed": "aloud", "cents": "sense", "sense": "cents",
}

# Phonologically plausible code-switch / pseudo-word substitutions around
# safety-adjacent terms (SpeechJBB 2606.06037 lineage — shipped as TEST
# payloads against the user's own agent, never as evasion guidance).
CODE_SWITCH_TABLE = {
    "password": "passwort", "account": "akaunt", "transfer": "transfèr",
    "delete": "dilit", "confirm": "konfirm", "security": "sekurité",
    "verify": "verefai", "balance": "balans", "cancel": "kansel",
}


def apply_asr_error(text: str, *, rate: float = 0.08, seed: int = 0) -> str:
    """Confusion-matrix style token corruption at a configured rate —
    deterministic under the seed. Mimics common ASR failure modes: dropped
    characters, adjacent transpositions, vowel confusions, duplications."""

    if not text or rate <= 0:
        return text
    rng = random.Random(f"{seed}:{text}")
    tokens = text.split(" ")
    corrupted: list[str] = []
    for token in tokens:
        if len(token) < 2 or rng.random() >= rate:
            corrupted.append(token)
            continue
        mode = rng.randrange(4)
        position = rng.randrange(len(token) - 1)
        if mode == 0:  # drop a character
            corrupted.append(token[:position] + token[position + 1 :])
        elif mode == 1:  # transpose adjacent characters
            corrupted.append(
                token[:position]
                + token[position + 1]
                + token[position]
                + token[position + 2 :]
            )
        elif mode == 2:  # vowel confusion
            replaced = False
            chars = list(token)
            for index, char in enumerate(chars):
                if char.lower() in _VOWELS:
                    replacement = rng.choice(_VOWELS)
                    chars[index] = (
                        replacement.upper() if char.isupper() else replacement
                    )
                    replaced = True
                    break
            corrupted.append("".join(chars) if replaced else token)
        else:  # duplicate a character
            corrupted.append(token[: position + 1] + token[position:])
    return " ".join(corrupted)


def _rewrap_token(original: str, replacement: str) -> str:
    """Re-wrap a stripped-form replacement in the original token's
    punctuation, preserving the case of the first character."""

    start = 0
    end = len(original)
    while start < end and not original[start].isalnum():
        start += 1
    while end > start and not original[end - 1].isalnum():
        end -= 1
    core = original[start:end]
    if core and core[0].isupper():
        replacement = replacement[:1].upper() + replacement[1:]
    return original[:start] + replacement + original[end:]


def apply_homophone_swap(text: str, *, rate: float = 0.15, seed: int = 0) -> str:
    """Swap table-listed tokens for their transcript-divergent twin at the
    configured rate — deterministic under the seed. Case of the first
    character is preserved; punctuation-adjacent tokens are matched on
    their stripped lowercase form and re-wrapped."""

    if not text or rate <= 0:
        return text
    rng = random.Random(f"{seed}:{text}")
    swapped: list[str] = []
    for token in text.split(" "):
        stripped = token.strip("".join(
            char for char in token if not char.isalnum()
        )) if token else token
        key = stripped.lower()
        if key not in HOMOPHONE_TABLE or rng.random() >= rate:
            swapped.append(token)
            continue
        swapped.append(_rewrap_token(token, HOMOPHONE_TABLE[key]))
    return " ".join(swapped)


def apply_code_switch(text: str, *, rate: float = 0.2, seed: int = 0) -> str:
    """Substitute safety-adjacent tokens with their code-switched /
    pseudo-word form (``CODE_SWITCH_TABLE``) at the configured rate —
    deterministic under the seed."""

    if not text or rate <= 0:
        return text
    rng = random.Random(f"{seed}:{text}")
    switched: list[str] = []
    for token in text.split(" "):
        stripped = token.strip("".join(
            char for char in token if not char.isalnum()
        )) if token else token
        key = stripped.lower()
        if key not in CODE_SWITCH_TABLE or rng.random() >= rate:
            switched.append(token)
            continue
        switched.append(_rewrap_token(token, CODE_SWITCH_TABLE[key]))
    return " ".join(switched)


def apply_near_dup(text: str, *, rate: float = 0.1, seed: int = 0) -> str:
    """Streaming-ASR doubled-hypothesis artifact: duplicate a token as an
    adjacent edit-distance-1 variant ("send" -> "send sent") at the
    configured rate; the variant reuses the ``apply_asr_error`` single-token
    corruption modes on the duplicate. Deterministic under the seed."""

    if not text or rate <= 0:
        return text
    rng = random.Random(f"{seed}:{text}")
    duplicated: list[str] = []
    for token in text.split(" "):
        duplicated.append(token)
        if len(token) < 2 or rng.random() >= rate:
            continue
        mode = rng.randrange(4)
        position = rng.randrange(len(token) - 1)
        if mode == 0:  # drop a character
            variant = token[:position] + token[position + 1 :]
        elif mode == 1:  # transpose adjacent characters
            variant = (
                token[:position]
                + token[position + 1]
                + token[position]
                + token[position + 2 :]
            )
        elif mode == 2:  # vowel confusion
            chars = list(token)
            variant = token
            for index, char in enumerate(chars):
                if char.lower() in _VOWELS:
                    replacement = rng.choice(_VOWELS)
                    chars[index] = (
                        replacement.upper() if char.isupper() else replacement
                    )
                    variant = "".join(chars)
                    break
        else:  # duplicate a character
            variant = token[: position + 1] + token[position:]
        duplicated.append(variant)
    return " ".join(duplicated)


def apply_text_perturbations(
    turns: Sequence[Mapping[str, Any]],
    operators: Sequence[str],
    *,
    seed: int = 0,
    asr_error_rate: float = 0.08,
    homophone_rate: float = 0.15,
    code_switch_rate: float = 0.2,
    near_dup_rate: float = 0.1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply text-rung operators to a user turn script. Returns the
    perturbed turns plus the applied-operator records for the
    ``live_lane.perturbations`` stanza. Acoustic operators raise — the rung
    is the gate between voice timing and voice audio evidence."""

    applied: list[dict[str, Any]] = []
    for operator in operators:
        if operator not in PERTURBATION_OPERATORS:
            raise ValueError(
                f"unknown perturbation operator {operator!r}; "
                f"expected one of {PERTURBATION_OPERATORS}"
            )
        if operator not in TEXT_RUNG_OPERATORS:
            raise ValueError(
                f"perturbation operator {operator!r} needs a real audio "
                "channel (rung 2 loopback transport or above); only "
                f"{TEXT_RUNG_OPERATORS} apply to text-rung input"
            )
    perturbed: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        row = dict(turn)
        if "asr_error" in operators and isinstance(row.get("user"), str):
            row["user"] = apply_asr_error(
                row["user"], rate=asr_error_rate, seed=seed + index
            )
        if "homophone" in operators and isinstance(row.get("user"), str):
            row["user"] = apply_homophone_swap(
                row["user"], rate=homophone_rate, seed=seed + index
            )
        if "code_switch" in operators and isinstance(row.get("user"), str):
            row["user"] = apply_code_switch(
                row["user"], rate=code_switch_rate, seed=seed + index
            )
        if "near_dup" in operators and isinstance(row.get("user"), str):
            row["user"] = apply_near_dup(
                row["user"], rate=near_dup_rate, seed=seed + index
            )
        perturbed.append(row)
    if "asr_error" in operators:
        applied.append(
            {"operator": "asr_error", "rate": asr_error_rate, "seed": seed}
        )
    if "homophone" in operators:
        applied.append(
            {"operator": "homophone", "rate": homophone_rate, "seed": seed}
        )
    if "code_switch" in operators:
        applied.append(
            {"operator": "code_switch", "rate": code_switch_rate, "seed": seed}
        )
    if "near_dup" in operators:
        applied.append(
            {"operator": "near_dup", "rate": near_dup_rate, "seed": seed}
        )
    return perturbed, applied


def perturbations_stanza(
    applied: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    paired_clean_run: str | None = None,
) -> dict[str, Any]:
    """The ``live_lane.perturbations`` stanza (guide §3.6): operator list,
    recorded seed, and the clean-twin link (deltas render upstream)."""

    return {
        "operators": [dict(record) for record in applied],
        "seed": seed,
        "paired_clean_run": paired_clean_run,
    }


# --- acoustic operators (rung 2+ — applied to the user PCM channel before
# the framework hears it) -----------------------------------------------------


def _require_pcm_acoustic(pcm: Any, *, where: str) -> np.ndarray:
    """Type-guard the input as numpy PCM; a text/str/bytes input raises the
    rung-wall ValueError (the same discipline ``_codec._require_pcm`` enforces —
    an acoustic operator over a transcript is a contract error)."""

    if isinstance(pcm, (str, bytes)):
        raise ValueError(
            f"{where} needs a real audio channel (rung 2 loopback transport or "
            "above); a text/transcript input is a contract error"
        )
    arr = np.asarray(pcm, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def mix_noise(
    pcm: np.ndarray, *, snr_db: float = 20.0, seed: int = 0
) -> np.ndarray:
    """Mix seeded gaussian noise into a PCM stream at the given SNR (dB)."""

    samples = _require_pcm_acoustic(pcm, where="mix_noise")
    if samples.size == 0:
        return samples
    signal_power = float((samples**2).mean())
    if signal_power == 0:
        return samples
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(noise_power), size=samples.shape)
    return samples + noise


def mix_interference(
    pcm: np.ndarray,
    interference: np.ndarray,
    *,
    level_db: float = -10.0,
) -> np.ndarray:
    """Overlay a competing-speaker waveform at the given relative level."""

    samples = _require_pcm_acoustic(pcm, where="mix_interference")
    competing = _require_pcm_acoustic(interference, where="mix_interference")
    if samples.size == 0 or competing.size == 0:
        return samples
    if competing.size < samples.size:
        repeat_count = int(np.ceil(samples.size / competing.size))
        competing = np.tile(competing, repeat_count)
    competing = competing[: samples.size]
    signal_rms = float(np.sqrt((samples**2).mean()))
    competing_rms = float(np.sqrt((competing**2).mean()))
    if competing_rms == 0 or signal_rms == 0:
        return samples
    target_rms = signal_rms * (10.0 ** (level_db / 20.0))
    return samples + competing * (target_rms / competing_rms)


def apply_reverb_blend(
    pcm: np.ndarray,
    *,
    decay: float = 0.4,
    delay_ms: float = 60.0,
    taps: int = 4,
    sample_rate: int = 24000,
    seed: int = 0,
) -> np.ndarray:
    """Reverberation-blended payload operator (Phase-12 12C rung-2 deferred,
    ARCH §2c — the AudioHijack reverberation-hiding insight, used DEFENSIVELY as
    a test payload). Convolves the PCM with a seeded multi-tap exponential-decay
    impulse response (a synthetic room reverb), then mixes the wet signal back at
    ``decay`` so the original waveform stays present. Deterministic under the
    seed (``np.random.default_rng(seed)`` jitters the tap gains reproducibly);
    raises at text-rung exactly like ``mix_noise``/``mix_interference``."""

    samples = _require_pcm_acoustic(pcm, where="apply_reverb_blend")
    if samples.size == 0 or decay <= 0 or taps < 1:
        return samples.astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)
    delay_samples = max(int(sample_rate * delay_ms / 1000.0), 1)
    ir_len = delay_samples * int(taps) + 1
    impulse = np.zeros(ir_len, dtype=float)
    impulse[0] = 1.0  # the dry direct path
    for tap in range(1, int(taps) + 1):
        position = min(tap * delay_samples, ir_len - 1)
        # exponential decay per tap, jittered reproducibly by the seed
        gain = float(decay**tap) * (0.85 + 0.3 * float(rng.random()))
        impulse[position] += gain
    wet = np.convolve(samples, impulse, mode="full")[: samples.size]
    return wet.astype(np.float32, copy=False)


def apply_acoustic_perturbations(
    pcm: np.ndarray,
    operators: Sequence[str],
    *,
    seed: int = 0,
    interference: np.ndarray | None = None,
    snr_db: float = 20.0,
    interference_level_db: float = -10.0,
    reverb_decay: float = 0.4,
    sample_rate: int = 24000,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Apply rung-2 acoustic operators to a real PCM channel (Phase-12 12C
    rung-2 / ARCH §2c). The sibling of ``apply_text_perturbations`` for the audio
    rung: it walks the operator list, applies each acoustic operator to the PCM
    in registry order, and returns the perturbed PCM plus the applied-operator
    records for the ``live_lane.perturbations`` stanza (the paired-clean
    discipline is identical to the text rung). Text-rung operators raise here —
    the rung wall runs in BOTH directions (a homophone swap over a waveform is a
    contract error just as ``mix_noise`` over a transcript is).

    Deterministic under ``seed``: every stochastic element keys on
    ``np.random.default_rng(seed)`` so a re-run produces a BYTE-IDENTICAL PCM and
    the same records — the determinism the rung-2 gate re-asserts over the
    loopback."""

    samples = _require_pcm_acoustic(pcm, where="apply_acoustic_perturbations")
    for operator in operators:
        if operator not in PERTURBATION_OPERATORS:
            raise ValueError(
                f"unknown perturbation operator {operator!r}; "
                f"expected one of {PERTURBATION_OPERATORS}"
            )
        if operator not in ACOUSTIC_RUNG_OPERATORS:
            raise ValueError(
                f"perturbation operator {operator!r} is a text-rung operator; "
                f"only {ACOUSTIC_RUNG_OPERATORS} apply to the rung-2 PCM channel"
            )
    applied: list[dict[str, Any]] = []
    out = samples
    if "noise" in operators:
        out = mix_noise(out, snr_db=snr_db, seed=seed)
        applied.append({"operator": "noise", "snr_db": snr_db, "seed": seed})
    if "interference" in operators:
        # a seeded synthetic competing speaker when the caller supplies none, so
        # the operator is self-contained and reproducible on the loopback.
        competing = interference
        if competing is None:
            rng = np.random.default_rng(seed + 104729)
            t = np.arange(max(out.size, 1), dtype=float) / float(sample_rate)
            competing = (
                0.5 * np.sin(2.0 * np.pi * 180.0 * t)
                + 0.05 * rng.standard_normal(max(out.size, 1))
            )
        out = mix_interference(out, competing, level_db=interference_level_db)
        applied.append(
            {
                "operator": "interference",
                "level_db": interference_level_db,
                "seed": seed,
            }
        )
    if "reverb_blend" in operators:
        out = apply_reverb_blend(
            out, decay=reverb_decay, sample_rate=sample_rate, seed=seed
        )
        applied.append(
            {"operator": "reverb_blend", "decay": reverb_decay, "seed": seed}
        )
    return out.astype(np.float32, copy=False), applied
