"""LiveKit live lane (3B) — real ``livekit-agents`` AgentSession, opt-in.

Framework imports: NONE at module top (P3-D1). Rung-1 execution happens in
the ``_workers/livekit_worker.py`` subprocess (the only sanctioned top-level
framework import home); this module is importable in the no-extras release
env and the live_lane_boundary gate scans it like any release module.

Rungs (P3-D3): 1 virtual-clock text driver (default, implemented) →
2 loopback real-transport audio → 3 LiveKit Cloud/SIP (``live_credentialed``,
standard LiveKit credential names). Rung 1 is honest about its tier: timing-only voice metrics,
no ``channels`` block, no audio claims (guide §3.5).
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ._contract import lane_budget_s, require_lane_enabled
from ._perturb import apply_text_perturbations, perturbations_stanza
from ._runner import run_worker_once
from ._stats import (
    derive_channel_evidence,
    lane_run_payload,
    primary_transcript_events,
    run_repeated,
)

_WORKERS = Path(__file__).resolve().parent / "_workers"
_RUNG_LABELS = {1: "virtual_clock", 2: "loopback_transport", 3: "cloud_sip"}

# Rung-3 credential names: exactly the names the vendored engine reads
# (engines/livekit.py reads LIVEKIT_API_KEY/LIVEKIT_API_SECRET; the server
# URL arrives via LIVEKIT_URL, P3-D5).
RUNG3_REQUIRED_ENV = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")

_DEFAULT_TURNS = (
    {"user": "Hello, can you hear me?"},
    {"user": "Great - please confirm my appointment for tomorrow."},
)


def _scenario_turns(scenario: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = scenario.get("turns") or scenario.get("user_messages")
    if not raw:
        return [dict(turn) for turn in _DEFAULT_TURNS]
    turns: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            turns.append({"user": item})
        elif isinstance(item, Mapping):
            turns.append(dict(item))
    return turns or [dict(turn) for turn in _DEFAULT_TURNS]


def _voice_timing(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Timing-only voice metrics (the rung-1 honesty tier): per-turn agent
    response latency derived from event timestamps — no audio claims."""

    latencies_ms: list[float] = []
    pending_user_t: float | None = None
    for event in events:
        channel = event.get("channel")
        if channel == "user" and event.get("type") == "message":
            t = event.get("t")
            pending_user_t = float(t) if isinstance(t, (int, float)) else None
        elif channel == "agent" and event.get("type") == "message":
            t = event.get("t")
            if pending_user_t is not None and isinstance(t, (int, float)):
                latencies_ms.append(round((float(t) - pending_user_t) * 1000.0, 3))
            pending_user_t = None
    return {
        "turn_latencies_ms": latencies_ms,
        "mean_turn_latency_ms": (
            round(sum(latencies_ms) / len(latencies_ms), 3) if latencies_ms else None
        ),
    }


def _realtime_state(
    events: Sequence[Mapping[str, Any]], *, rung_label: str
) -> dict[str, Any]:
    items = []
    for index, event in enumerate(events, start=1):
        if event.get("channel") in ("user", "agent", "tool"):
            payload = event.get("payload")
            payload = payload if isinstance(payload, Mapping) else {}
            items.append(
                {
                    "index": index,
                    "channel": event.get("channel"),
                    "item_type": event.get("type"),
                    "text": payload.get("text"),
                }
            )
    return {
        "engine": "live_lane_livekit",
        "rung": rung_label,
        "item_count": len(items),
        "items": items[:200],
    }


def _rung2_loopback_channels(
    turns: Sequence[Mapping[str, Any]],
    *,
    loopback: Optional[Mapping[str, Any]],
    codec_profile: str,
    seed: int,
    acoustic_operators: Sequence[str] = (),
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Phase 9A unit 2 + Phase-12 12C rung-2 — the rung-2 loopback dispatch
    (§2.1 / §2.5 + ARCH §2c).

    Produce the two PCM streams via the deterministic ``_loopback`` round-trip,
    apply the rung-2 ACOUSTIC operators (Phase-12 12C: ``mix_noise`` /
    ``mix_interference`` / ``reverb_blend`` over the user PCM — the attack the
    framework hears) BEFORE the codec stage, apply the default-ON codec
    round-trip (9A-A11) unless ``codec_profile == "none"``, feed the
    ALREADY-BUILT ``derive_channel_evidence`` (REUSED, NOT rebuilt), and return
    the ``channels`` block + the ``fidelity_tier`` marker + the applied acoustic
    operator records (the paired-clean stanza). The loopback module is reached
    via the sanctioned ``from fi.alk import live`` function-body idiom so
    this module stays framework-free and the ``live_lane_boundary`` import
    discipline holds.

    The codec-survival score is computed on the PERTURBED-then-channel signal so
    ``phone_survival`` honestly reflects whether the acoustic attack reproduces
    through the 8 kHz telephony channel (P12-D2): no ``survives``/``partial``
    claim without a codec record."""

    from fi.alk import live  # sanctioned facade idiom (cli.py)

    cfg = dict(loopback or {})
    tick_ms = float(cfg.get("tick_ms", live._loopback.DEFAULT_TICK_MS))
    sample_rate = int(cfg.get("sample_rate", live._loopback.DEFAULT_SAMPLE_RATE))
    loop_seed = int(cfg.get("seed", seed))
    profile = str(cfg.get("codec_profile", codec_profile))

    loop = live._loopback.run_loopback_roundtrip(
        list(turns),
        user_wav=cfg.get("user_wav"),
        agent_wav=cfg.get("agent_wav"),
        tick_ms=tick_ms,
        sample_rate=sample_rate,
        seed=loop_seed,
    )
    user_pcm, agent_pcm = loop["user_pcm"], loop["agent_pcm"]

    # Phase-12 12C rung-2: the acoustic attack rides the USER channel (the side
    # the framework hears). Applied to the CLEAN loopback PCM before the codec
    # stage; deterministic under loop_seed. The agent side is untouched.
    acoustic_applied: list[dict[str, Any]] = []
    attacked_user_pcm = user_pcm
    if acoustic_operators:
        attacked_user_pcm, acoustic_applied = live._perturb.apply_acoustic_perturbations(
            user_pcm,
            list(acoustic_operators),
            seed=loop_seed,
            sample_rate=sample_rate,
        )
    user_pcm = attacked_user_pcm

    codec_record: dict[str, Any] | None = None
    phone_survival: dict[str, Any] | None = None
    if profile != "none":
        user_pcm, agent_pcm, codec_record = live._codec.apply_codec_profile(
            user_pcm, agent_pcm, profile=profile, seed=loop_seed, sample_rate=sample_rate
        )
        codec, packet_loss = live._codec._PROFILE_BUNDLE[profile]
        # the attack rides the USER channel, so re-validate the user side through
        # the channel (the clean user PCM is the pre-channel twin).
        phone_survival = live._codec.score_codec_survival(
            loop["user_pcm"],
            attacked_user_pcm,
            codec=codec,
            packet_loss=packet_loss,
            seed=loop_seed,
            sample_rate=sample_rate,
        )

    derived = derive_channel_evidence(
        user_pcm, agent_pcm, sample_rate=(8000 if profile != "none" else sample_rate)
    )
    channels: dict[str, Any] = {
        "derived": derived,
        "source": "derive_channel_evidence",
        "rung": _RUNG_LABELS[2],
        "fidelity_tier": "deterministic_loopback",
        "seed": loop_seed,
        "loopback_provenance": loop["provenance"],
    }
    if codec_record is not None:
        channels["codec_round_trip"] = codec_record
    if phone_survival is not None:
        channels["phone_survival"] = phone_survival
    if acoustic_applied:
        channels["acoustic_operators"] = acoustic_applied
    return channels, "deterministic_loopback", acoustic_applied


def run_livekit_lane(
    scenario: Mapping[str, Any],
    *,
    rung: int = 1,                  # P3-D3: 1 virtual-clock | 2 loopback transport | 3 cloud/SIP
    repeats: int = 8,
    stressed: bool = False,         # perturbation sub-lane -> evidence_class "live_stressed"
    perturbations: Optional[Sequence[str]] = None,
    seed: int = 0,
    required_env: Optional[Sequence[str]] = None,
    version_requirement: str | None = None,
    budget_s: float | None = None,
    artifacts_dir: str | Path | None = None,
    # Phase 9A (BBG A2): additive optional loopback config consumed ONLY on the
    # rung==2 branch; rung-1/rung-3 callers are unaffected.
    loopback: Optional[Mapping[str, Any]] = None,
    codec_profile: str = "g711_ulaw_8k_ge",
) -> dict[str, Any]:
    require_lane_enabled("livekit")
    if rung >= 3:
        require_lane_enabled("credentialed")
    if rung not in _RUNG_LABELS:
        raise ValueError(f"rung must be one of {sorted(_RUNG_LABELS)}, got {rung}")

    required = tuple(required_env) if required_env is not None else ()
    operators = list(perturbations or (["asr_error"] if stressed else []))
    turns = _scenario_turns(scenario)
    # Phase-12 12C rung-2: split text-rung operators (applied to the turn script)
    # from acoustic operators (applied to the rung-2 loopback PCM). At rung-1 an
    # acoustic operator still raises inside ``apply_text_perturbations`` (the
    # rung wall is unchanged for text-rung input).
    from ._perturb import ACOUSTIC_RUNG_OPERATORS

    acoustic_operators = [op for op in operators if op in ACOUSTIC_RUNG_OPERATORS]
    text_operators = [op for op in operators if op not in ACOUSTIC_RUNG_OPERATORS]
    if rung != 2 and acoustic_operators:
        # acoustic operators require the rung-2 PCM channel; outside it they hit
        # the same rung wall ``apply_text_perturbations`` enforces (no silent
        # acoustic claim before the audio channel exists — ARCH §2c).
        raise ValueError(
            f"acoustic operators {acoustic_operators} need a real audio channel "
            "(rung 2 loopback transport or above); rung "
            f"{rung} ({_RUNG_LABELS[rung]}) is a text-rung tier"
        )
    applied: list[dict[str, Any]] = []
    if text_operators:
        turns, applied = apply_text_perturbations(turns, text_operators, seed=seed)

    # Phase 9A unit 2: the rung wall narrows — rung-2 dispatches into the
    # deterministic loopback (§2.1); rung-3 still raises (the owner live-proof,
    # unit 7). rung-1 is completely untouched (timing-only, NO channels block).
    channels: dict[str, Any] | None = None
    fidelity_tier: str | None = None
    acoustic_applied: list[dict[str, Any]] = []
    if rung == 2:
        channels, fidelity_tier, acoustic_applied = _rung2_loopback_channels(
            turns,
            loopback=loopback,
            codec_profile=codec_profile,
            seed=seed,
            acoustic_operators=acoustic_operators,
        )
        # §2.5 binding correction: a deterministic in-process loopback is
        # NEVER live_lane. Default codec round-trip is ON (9A-A11) → a stressed
        # run → live_stressed; a no-op (codec_profile="none") clean run is also
        # live_stressed at rung-2 (it never claims live_lane). captured_fixture
        # is reached through the capture flow, not here.
        evidence_class = "live_stressed"
    elif rung != 1:
        # rung == 3: unchanged keyed path; still requires the credentialed flag
        # + RUNG3_REQUIRED_ENV; rung-3 lands as the owner live-proof (unit 7).
        raise NotImplementedError(
            f"livekit lane rung {rung} ({_RUNG_LABELS[rung]}) is not "
            "implemented yet; rung 1 (virtual_clock) and rung 2 "
            "(loopback_transport) are the supported tiers — rung 3 (cloud_sip) "
            "is the owner-keyed live-proof lane"
        )
    else:
        evidence_class = "live_stressed" if operators else "live_lane"

    base_dir = (
        Path(artifacts_dir)
        if artifacts_dir is not None
        else Path(tempfile.mkdtemp(prefix="agent-learning-live-livekit-"))
    )
    run_id = uuid.uuid4().hex
    resolved_budget = float(budget_s) if budget_s is not None else lane_budget_s("livekit")
    boot = {
        "type": "boot",
        "lane": "livekit",
        "rung": rung,
        "scenario": {"name": str(scenario.get("name") or "livekit-smoke")},
        "turns": turns,
        "config": {
            "instructions": scenario.get("instructions")
            or "You are a concise, helpful voice agent under test.",
            "responses": scenario.get("responses"),
            "expect": scenario.get("expect"),
        },
    }
    worker = _WORKERS / "livekit_worker.py"

    def _run_once(index: int, transcript: Any) -> dict[str, Any]:
        return run_worker_once(
            worker,
            boot,
            lane="livekit",
            required_env=required,
            cwd=base_dir,
            timeout_s=resolved_budget,
            transcript=transcript,
            version_requirement=version_requirement,
        )

    result = run_repeated(
        _run_once,
        lane="livekit",
        evidence_class=evidence_class,
        repeats=repeats,
        budget_s=budget_s,
        required_env=required,
        artifacts_dir=base_dir,
        run_id=run_id,
        rung=_RUNG_LABELS[rung],
        framework="livekit-agents",
        version_requirement=version_requirement,
    )

    events = primary_transcript_events(result)
    # Normalization rides the existing realtime manifest builder — the run
    # lands in the existing `realtime_trace` state family; the live engine
    # is declared in metadata (guide §3.1).
    from .. import simulate as _simulate

    manifest = _simulate.build_realtime_run_manifest(
        name=f"live-livekit-{run_id[:8]}",
        framework="livekit",
        required_env=required,
        min_turns=1,
        max_turns=max(len(turns), 1),
        metadata={
            "simulation_engine": "live_lane_livekit",
            "live_lane": {"lane": "livekit", "rung": _RUNG_LABELS[rung]},
        },
    )

    payload = lane_run_payload(
        result,
        name=f"live-livekit-{run_id[:8]}",
        scenario=scenario,
        manifest=manifest,
        states={"realtime_trace": _realtime_state(events, rung_label=_RUNG_LABELS[rung])},
        metadata={
            "execution_model": "subprocess",
            "rung": _RUNG_LABELS[rung],
            # rung-1 honesty: timing-only voice metrics, NO channels block
            "voice_timing": _voice_timing(events),
        },
    )
    # the perturbations stanza carries BOTH families (text-rung records + the
    # rung-2 acoustic records); the clean-twin link is filled by the campaign.
    all_applied = list(applied) + list(acoustic_applied)
    if all_applied:
        payload["live_lane"]["perturbations"] = perturbations_stanza(
            all_applied, seed=seed, paired_clean_run=None
        )
    if channels is not None:
        # rung-2: attach the dual-channel evidence + the fidelity marker (§2.5 /
        # 9A-A10). fidelity_tier is a MARKER FIELD, not a new evidence class.
        payload["channels"] = channels
        if isinstance(payload.get("live_lane"), dict):
            payload["live_lane"]["fidelity_tier"] = fidelity_tier
        payload["fidelity_tier"] = fidelity_tier
    return payload
