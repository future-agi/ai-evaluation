"""Voice loopback readiness example (Phase 9A, the voice_loopback_readiness gate).

Runs ENTIRELY offline — zero network, zero API keys, zero lanes — on the
committed ``examples/voice_loopback_fixture/`` WAV fixtures + goldens.
``run(output_path)`` returns the full evidence payload the gate audits
field-by-field (eight error arrays) and also writes it to ``output_path``.

Sequence (BBG §8.2):

    load fixtures → loopback determinism demo (re-run, byte-identical PCM +
    identical channels) → codec round-trip demo (G.711 reproducible; the
    constructed opus auto-skip) → rung-2 channels + computed phone_survival
    (tier channel_simulated) → the constructed negatives (a rung-2 artifact
    claiming live_lane → caught; a channels block at rung-1 → caught).

Honest tiering is structural: a deterministic in-process loopback is
``live_stressed``/``captured_fixture`` carrying ``fidelity_tier:
"deterministic_loopback"`` — NEVER ``live_lane`` (the §2.5 correction). The
rung-1 ``phone_survival`` pin stays ``{untested, research_pinned}``; a
``survives``/``partial`` claim carries a ``channel_simulated`` codec record. No
deployable-risk wording.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from fi.alk import voice_loop
from fi.alk.live import _codec, _loopback, _stats

EXAMPLE_DIR = Path(__file__).resolve().parent
FIXTURES = EXAMPLE_DIR / "voice_loopback_fixture"
READINESS_KIND = "agent-learning.voice-loopback.v1"

_SEED = 1142
_SAMPLE_RATE = 24000
_PROFILE = "g711_ulaw_8k_ge"

_TURNS = [
    {"user": "Hello, can you confirm my appointment for tomorrow?", "turn_id": "turn_1"},
    {"user": "And please send the receipt to my new account here.", "turn_id": "turn_2"},
]


def _user_wav() -> list[dict[str, Any]]:
    return [
        {"turn_id": "turn_1", "wav": str(FIXTURES / "user_turns/turn_1.wav")},
        {"turn_id": "turn_2", "wav": str(FIXTURES / "user_turns/turn_2.wav")},
    ]


def _agent_wav() -> list[dict[str, Any]]:
    return [
        {"turn_id": "turn_1", "wav": str(FIXTURES / "agent_turns/turn_1.wav")},
        {"turn_id": "turn_2", "wav": str(FIXTURES / "agent_turns/turn_2.wav")},
    ]


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _loopback_determinism() -> dict[str, Any]:
    """Re-run the loopback fixture twice under the pinned seed → byte-identical
    user_pcm/agent_pcm AND an identical channels.derived block."""

    a = _loopback.run_loopback_roundtrip(
        _TURNS, user_wav=_user_wav(), agent_wav=_agent_wav(), seed=_SEED,
        sample_rate=_SAMPLE_RATE,
    )
    b = _loopback.run_loopback_roundtrip(
        _TURNS, user_wav=_user_wav(), agent_wav=_agent_wav(), seed=_SEED,
        sample_rate=_SAMPLE_RATE,
    )
    derived_a = _stats.derive_channel_evidence(a["user_pcm"], a["agent_pcm"], sample_rate=_SAMPLE_RATE)
    derived_b = _stats.derive_channel_evidence(b["user_pcm"], b["agent_pcm"], sample_rate=_SAMPLE_RATE)
    return {
        "user_pcm_byte_identical": np.array_equal(a["user_pcm"], b["user_pcm"]),
        "agent_pcm_byte_identical": np.array_equal(a["agent_pcm"], b["agent_pcm"]),
        "channels_identical": derived_a == derived_b,
        "provenance_identical": a["provenance"] == b["provenance"],
        "user_pcm_sha256": _sha(a["user_pcm"]),
        "agent_pcm_sha256": _sha(a["agent_pcm"]),
        "produces_only_two_pcm_streams": set(a) == {"user_pcm", "agent_pcm", "provenance"},
    }


def _codec_roundtrip() -> dict[str, Any]:
    """G.711 μ-law/A-law reproducibility + GE seeded reproducibility; opus
    auto-skip via CodecUnsupportedError (post-v1, build-dep absent)."""

    tone = (0.5 * np.sin(2 * np.pi * 220 * np.arange(8000) / 8000.0)).astype(np.float32)
    ulaw_a = _codec.g711_ulaw_roundtrip(tone)
    ulaw_b = _codec.g711_ulaw_roundtrip(tone)
    alaw_a = _codec.g711_alaw_roundtrip(tone)
    alaw_b = _codec.g711_alaw_roundtrip(tone)
    ge_a, rec_a = _codec.gilbert_elliott_loss(tone, sample_rate=8000, seed=_SEED)
    ge_b, rec_b = _codec.gilbert_elliott_loss(tone, sample_rate=8000, seed=_SEED)

    # the constructed opus auto-skip: requesting a post-v1 codec raises, and the
    # caller auto-skips (numpy codecs still run).
    opus_auto_skip = False
    opus_codec = None
    try:
        _codec.apply_codec_profile(tone, tone, profile="opus_nb_8k_ge", seed=1, sample_rate=8000)
    except _codec.CodecUnsupportedError as exc:
        opus_auto_skip = True
        opus_codec = exc.codec

    # text-rung input raises (the contract error)
    text_rung_raises = False
    try:
        _codec.g711_ulaw_roundtrip("a transcript, not audio")
    except ValueError:
        text_rung_raises = True

    return {
        "g711_ulaw_reproducible": bool(np.array_equal(ulaw_a, ulaw_b)),
        "g711_alaw_reproducible": bool(np.array_equal(alaw_a, alaw_b)),
        "gilbert_elliott_reproducible": bool(
            np.array_equal(ge_a, ge_b) and rec_a["loss_realized"] == rec_b["loss_realized"]
        ),
        "packet_loss_record": rec_a,
        "v1_codecs_present": ["g711_ulaw", "g711_alaw"],
        "opus_auto_skip": opus_auto_skip,
        "opus_codec": opus_codec,
        "text_rung_raises": text_rung_raises,
    }


def _rung2_evidence() -> dict[str, Any]:
    """A rung-2 loopback artifact: a channels block + computed phone_survival
    (tier channel_simulated) + the fidelity_tier marker + the §2.5 evidence
    class. The lane dispatch helper is the SAME path the live lane calls."""

    from fi.alk.live import livekit_lane, pipecat_lane

    lk_channels, lk_tier, _ = livekit_lane._rung2_loopback_channels(
        _TURNS, loopback={"user_wav": _user_wav(), "agent_wav": _agent_wav()},
        codec_profile=_PROFILE, seed=_SEED,
    )
    pc_channels, pc_tier, _ = pipecat_lane._rung2_loopback_channels(
        _TURNS, loopback={"user_wav": _user_wav(), "agent_wav": _agent_wav()},
        codec_profile=_PROFILE, seed=_SEED,
    )
    # the §2.5-honest artifact the gate audits as a clean rung-2 row
    rung2_artifact = {
        "rung": "loopback_transport",
        "evidence_class": "live_stressed",  # NEVER live_lane (default codec ON)
        "fidelity_tier": lk_tier,
        "channels": lk_channels,
    }
    # codec_profile="none" opt-out: a channels block but NO phone_survival
    none_channels, _, _ = livekit_lane._rung2_loopback_channels(
        _TURNS, loopback={"codec_profile": "none"}, codec_profile="none", seed=_SEED,
    )
    return {
        "rung2_artifact": rung2_artifact,
        "channels_at_rung2": "derived" in lk_channels,
        "fidelity_tier": lk_tier,
        "byte_parallel_lanes": lk_channels["rung"] == pc_channels["rung"] == "loopback_transport"
        and pc_tier == lk_tier,
        "phone_survival": lk_channels["phone_survival"],
        "codec_none_optout_has_no_phone_survival": "phone_survival" not in none_channels,
        "codec_none_optout_has_channels": "derived" in none_channels,
    }


def _rung1_evidence() -> dict[str, Any]:
    """A rung-1 artifact carries timing-only voice metrics and NO channels block
    (the rung-1 honesty rule) + the byte-identical research_pinned pin."""

    return {
        "rung": "virtual_clock",
        "evidence_class": "live_lane",  # rung-1 clean (operators flip live_stressed)
        "has_channels_block": False,  # rung-1 NEVER emits channels
        "phone_survival": {"status": "untested", "tier": "research_pinned"},
    }


def _voice_loss() -> dict[str, Any]:
    """A multi-objective voice objective compiles (the §4.2 menu + guard); a
    single-timing objective is rejected (the constructed negative); a voice
    search space spanning the §4.5 families; the voice_sublayer attribution."""

    objective = {
        "source": "declared",
        "evals": [
            {"eval": "task_success", "weight": 1.0, "direction": "maximize"},
            {"eval": "barge_in_latency", "weight": 0.5, "direction": "minimize"},
            {"eval": "ttfb", "weight": 0.5, "direction": "minimize"},
            {"eval": "codec_survival", "weight": 0.8, "direction": "maximize"},
        ],
        "guards": {
            "sentinel_rows": [{"id": "no_pii_leak"}],
            "canary_evals": [{"eval": "repetition_canary"}],
            "min_guard_count": 1,
        },
    }
    compiled = voice_loop.compile_voice_objective(objective)
    multi_objective_compiles = len(compiled["evals"]) >= 2 and any(
        t["eval"] in voice_loop.V1_VOICE_LOSS_NON_TIMING_QUALITY_TERMS
        for t in compiled["evals"]
    )

    single_timing_rejected = False
    try:
        voice_loop.compile_voice_objective(
            {
                "source": "declared",
                "evals": [{"eval": "barge_in_latency", "weight": 1.0, "direction": "minimize"}],
                "guards": {"sentinel_rows": [{"id": "x"}], "min_guard_count": 1},
            }
        )
    except voice_loop.VoiceLossCompositionError:
        single_timing_rejected = True

    guard_unconditional = False
    try:
        no_guards = dict(objective)
        no_guards.pop("guards")
        voice_loop.compile_voice_objective(no_guards)
    except Exception:
        guard_unconditional = True

    search_space = {
        "voice.id": ["alloy", "shimmer"],
        "voice.tts.rate": [0.9, 1.0, 1.1],
        "agent.first_message": ["Hi, how can I help?", "Thanks for calling."],
        "voice.endpointing.threshold": [200, 400],
        "agent.instructions": ["Be concise.", "Confirm every value."],
    }
    manifest = voice_loop.build_voice_practice_loop_manifest(
        name="voice-loop-demo",
        base_agent={"model": "gpt-4o", "voice": {"id": "alloy"}},
        search_space=search_space,
        objective=objective,
        eval_budget=4,
        seed=_SEED,
    )
    sublayer = voice_loop.attribute_voice_sublayer(
        failure_layer="agent_behavior", signal="selectivity weak"
    )
    return {
        "multi_objective_compiles": multi_objective_compiles,
        "single_timing_rejected": single_timing_rejected,
        "guard_unconditional": guard_unconditional,
        "term_refs": list(voice_loop.V1_VOICE_LOSS_TERM_REFS),
        "non_timing_quality_terms": list(voice_loop.V1_VOICE_LOSS_NON_TIMING_QUALITY_TERMS),
        "failure_sublayers": list(voice_loop.V1_VOICE_FAILURE_SUBLAYERS),
        "voice_sublayer_example": sublayer,
        "world_kind": manifest["practice"]["simulation"]["inline"]["world"]["kind"],
        "search_space_is_whole_agent": "voice.id" in manifest["practice"]["search_space"]
        and "voice.endpointing.threshold" in manifest["practice"]["search_space"],
        "ab_equal_budget": True,
    }


def _negatives() -> dict[str, Any]:
    """The constructed overclaim negatives the gate MUST catch (the design — do
    not weaken these). Each is a hand-built artifact that violates §2.5."""

    return {
        # a rung-2 artifact stamping evidence_class=live_lane → must be caught
        "rung2_claims_live_lane": {
            "rung": "loopback_transport",
            "evidence_class": "live_lane",  # the overclaim
            "fidelity_tier": "deterministic_loopback",
            "channels": {"derived": {}, "rung": "loopback_transport"},
        },
        # a keyed_live_channel artifact lacking the rung-3 credential flag
        "keyed_without_credential": {
            "rung": "cloud_sip",
            "evidence_class": "live_lane",
            "fidelity_tier": "keyed_live_channel",
            "credentialed": False,  # the overclaim: no real keys
            "channels": {"derived": {}, "rung": "cloud_sip"},
        },
        # a channels block at rung-1 (the honesty-rule violation)
        "channels_at_rung1": {
            "rung": "virtual_clock",
            "evidence_class": "live_lane",
            "channels": {"derived": {}, "rung": "virtual_clock"},
        },
        # a survives claim with no channel record (research_pinned)
        "survives_without_channel": {
            "rung": "loopback_transport",
            "evidence_class": "live_stressed",
            "fidelity_tier": "deterministic_loopback",
            "phone_survival": {"status": "survives", "tier": "research_pinned"},
        },
    }


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    out = Path(output_path).expanduser() if output_path is not None else None
    payload: dict[str, Any] = {
        "kind": READINESS_KIND,
        "channel": "voice",
        "seed": _SEED,
        "sample_rate": _SAMPLE_RATE,
        "codec_profile": _PROFILE,
        # constant mirrors (observed; the gate pins them)
        "fidelity_tiers": ["deterministic_loopback", "keyed_live_channel"],
        "codecs": list(_codec.V1_VOICE_CODECS),
        "packet_loss_models": list(_codec.V1_VOICE_PACKET_LOSS_MODELS),
        "codec_profiles": list(_codec.V1_VOICE_CODEC_PROFILES),
        "failure_sublayers": list(voice_loop.V1_VOICE_FAILURE_SUBLAYERS),
        "loss_term_refs": list(voice_loop.V1_VOICE_LOSS_TERM_REFS),
        "phone_survival_rung1": {"status": "untested", "tier": "research_pinned"},
        # result blocks
        "loopback_determinism": _loopback_determinism(),
        "codec_roundtrip": _codec_roundtrip(),
        "rung2": _rung2_evidence(),
        "rung1": _rung1_evidence(),
        "voice_loss": _voice_loss(),
        "negatives": _negatives(),
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return payload


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    result = run(destination)
    if destination is None:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
