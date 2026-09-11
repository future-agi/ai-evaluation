"""Opt-in live framework lanes (Phase 3) — facade only.

Imports NOTHING framework-side: the substrate (contract/runner/transcript/
stats/attribution/capture) is stdlib+numpy by construction, lane modules
import frameworks lazily inside function bodies, and ``_workers/`` entry
modules only ever run as scrubbed-env subprocesses (P3-D1). Lanes are extras
+ env-gated markers, NEVER release prerequisites: every lane entry refuses
without its ``AGENT_LEARNING_LIVE_<LANE>=1`` flag.
"""

from __future__ import annotations

import importlib
from typing import Any

_SUBMODULES = {
    "_attribution",
    "_capture",
    "_codec",
    "_contract",
    "_loopback",
    "_perturb",
    "_runner",
    "_stats",
    "_transcript",
    "a2a_lane",
    "langgraph_lane",
    "livekit_lane",
    "mcp_lane",
    "pipecat_lane",
    "voice_redteam",
}

# lane name (LANE_ENV_FLAGS / LANE_EXTRAS key) → (module, entry point)
LANE_RUNNERS = {
    "livekit": ("fi.alk.live.livekit_lane", "run_livekit_lane"),
    "pipecat": ("fi.alk.live.pipecat_lane", "run_pipecat_lane"),
    "langchain": ("fi.alk.live.langgraph_lane", "run_langgraph_lane"),
    "mcp": ("fi.alk.live.mcp_lane", "run_mcp_lane"),
    "a2a": ("fi.alk.live.a2a_lane", "run_a2a_lane"),
}

# public name → home module (resolved lazily so `import fi.alk.live`
# stays trivially cheap and provably framework-free)
_LAZY_EXPORTS = {
    "AGENT_LEARNING_RUN_KIND": "_contract",
    "EVIDENCE_CLASSES": "_contract",
    "RELEASE_ADMISSIBLE_EVIDENCE_CLASSES": "_contract",
    "FAILURE_LAYERS": "_contract",
    "VERDICTS": "_contract",
    "LANE_ENV_FLAGS": "_contract",
    "LANE_EXTRAS": "_contract",
    "LANE_BUDGET_S": "_contract",
    "LANE_BUDGET_S_DEFAULT": "_contract",
    "DEFAULT_REPEATS": "_contract",
    "UNSTABLE_ICC_FLOOR": "_contract",
    "LaneDisabledError": "_contract",
    "LaneSpec": "_contract",
    "LaneRun": "_contract",
    "lane_budget_s": "_contract",
    "require_lane_enabled": "_contract",
    "LANE_SAFE_BASE_ENV": "_runner",
    "LANE_BLOCKED_ENV": "_runner",
    "LaneProcessResult": "_runner",
    "scrubbed_lane_env": "_runner",
    "spawn_lane_subprocess": "_runner",
    "run_worker_once": "_runner",
    "version_ok": "_runner",
    "version_preflight": "_runner",
    "TranscriptRecorder": "_transcript",
    "read_transcript": "_transcript",
    "redact_env_values": "_transcript",
    "TRANSCRIPT_MAX_BYTES_ENV": "_transcript",
    "LaneRunResult": "_stats",
    "run_repeated": "_stats",
    "lane_run_payload": "_stats",
    "icc_and_within_variance": "_stats",
    "divergence_step": "_stats",
    "determinism_metrics": "_stats",
    "derive_channel_evidence": "_stats",
    "FailureAttribution": "_attribution",
    "attribute_failure": "_attribution",
    "CaptureRefusedError": "_capture",
    "CAPTURE_PROVENANCE_FIELDS": "_capture",
    "capture_to_fixture": "_capture",
    "replay_fixture": "_capture",
    "run_voice_escalation_campaign": "voice_redteam",
    "compile_arc_turns": "voice_redteam",
    "timing_fidelity": "voice_redteam",
    "validate_authorization": "voice_redteam",
    "VoiceAuthorizationError": "voice_redteam",
    # Phase-12 12C rung-2: acoustic operators over the loopback PCM channel.
    "apply_acoustic_perturbations": "_perturb",
    "apply_reverb_blend": "_perturb",
    "ACOUSTIC_RUNG_OPERATORS": "_perturb",
    # Phase 9A: codec-survival facade (9A-A12, home _codec) + loopback runner
    "score_codec_survival": "_codec",
    "CodecUnsupportedError": "_codec",
    "run_loopback_roundtrip": "_loopback",
    "LoopbackFixtureMissing": "_loopback",
}


def run_lane(lane: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Dispatch to a lane's entry point by name (the CLI front door's hook).

    The lane module is imported lazily; its entry calls
    ``require_lane_enabled`` first, so a missing env flag refuses before any
    framework import is attempted.
    """

    try:
        module_name, entry_name = LANE_RUNNERS[lane]
    except KeyError:
        known = ", ".join(sorted(LANE_RUNNERS))
        raise ValueError(f"unknown live lane {lane!r}; expected one of: {known}")
    module = importlib.import_module(module_name)
    entry = getattr(module, entry_name)
    return entry(*args, **kwargs)


def capture_fixture(*args: Any, **kwargs: Any) -> Any:
    """Live→fixture demotion (see ``_capture.capture_to_fixture``)."""

    from ._capture import capture_to_fixture as _capture_to_fixture

    return _capture_to_fixture(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(
            f"{__name__}.{_LAZY_EXPORTS[name]}"
        )
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_SUBMODULES, *_LAZY_EXPORTS})


__all__ = [
    "LANE_RUNNERS",
    "capture_fixture",
    "run_lane",
    *sorted(_LAZY_EXPORTS),
]
