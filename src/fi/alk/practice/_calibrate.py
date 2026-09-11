"""Unit 13 (BBG U13 / ARCH §2d phase 6) — CALIBRATE: the learned-gate.

Per cell: learned iff score ≥ floor AND fork-entropy ≤ threshold AND ICC ≥ floor
over k; high-score/high-entropy = fluent_not_learned (stays in rotation);
plateaued/zpd_exited stop rules. Trajectory profiles are post-hoc, never a stop
rule. Emits ``agent-learning.practice-calibration.v1``.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from .._schema import public_payload
from ..live._contract import UNSTABLE_ICC_FLOOR
from ._contract import AGENT_LEARNING_PRACTICE_CALIBRATION_KIND, CALIBRATION_VERDICTS


def calibrate_cell(
    cell: Mapping[str, Any],
    *,
    score: float,
    fork_entropy: float,
    divergence_step: Optional[int],
    icc: float,
    repeats: int,
    score_floor: float = 0.7,
    entropy_threshold: float = 0.3,
    icc_floor: float = UNSTABLE_ICC_FLOOR,
    prior_score: Optional[float] = None,
    in_band: bool = True,
) -> dict:
    """Compute one cell's calibration verdict (synthesis §4(6))."""
    learned = score >= score_floor and fork_entropy <= entropy_threshold and icc >= icc_floor
    if learned:
        verdict = "learned"
        stop_reason = "learned"
    elif score >= score_floor and fork_entropy > entropy_threshold:
        verdict = "fluent_not_learned"  # high-score / high-entropy
        stop_reason = None
    elif not in_band:
        verdict = "zpd_exited"
        stop_reason = "zpd_exited"
    elif prior_score is not None and abs(score - prior_score) < 1e-3:
        verdict = "plateaued"
        stop_reason = "plateaued"
    else:
        verdict = "in_rotation"
        stop_reason = None
    assert verdict in CALIBRATION_VERDICTS
    return {
        "cell": dict(cell),
        "score": round(float(score), 6),
        "fork_entropy": round(float(fork_entropy), 6),
        "divergence_step": divergence_step,
        "icc": round(float(icc), 6),
        "repeats": int(repeats),
        "verdict": verdict,
        "stop_reason": stop_reason,
    }


def calibrate(cells: Sequence[Mapping[str, Any]], *, round_no: int) -> dict:
    """Emit the calibration artifact over a list of pre-computed cell measures."""
    records = [calibrate_cell(**c) if "verdict" not in c else dict(c) for c in cells]
    report = {
        "kind": AGENT_LEARNING_PRACTICE_CALIBRATION_KIND,
        "round": int(round_no),
        "cells": records,
    }
    return public_payload(report, kind=AGENT_LEARNING_PRACTICE_CALIBRATION_KIND)
