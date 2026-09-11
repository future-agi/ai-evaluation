"""Variance math, repeat executor, verdicts for live lanes — pure numpy.

ICC(1) via one-way variance decomposition (ARCH Decision 4 — no scipy);
degenerate zero-variance matrices define ICC := 1.0 (a deterministic green
run must never classify ``unstable``). Determinism metrics are reported
separately from quality scores — DFAH's r=-0.11 forbids conflation (R§1 #15).
"""

from __future__ import annotations

import dataclasses
import math
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .._schema import public_payload
from ._contract import (
    AGENT_LEARNING_RUN_KIND,
    DEFAULT_REPEATS,
    EVIDENCE_CLASSES,
    FAILURE_LAYERS,
    UNSTABLE_ICC_FLOOR,
    LaneRun,
    lane_budget_s,
)
from ._transcript import TranscriptRecorder


def icc_and_within_variance(scores: np.ndarray) -> tuple[float, float]:
    """One-way random-effects ICC over a (n_scenarios, k_repeats) score
    matrix + pooled within-scenario variance (R§1 #16, ICC convergence at
    n=8–16; P3-D2 default k=8).

    ICC = (MS_between - MS_within) / (MS_between + (k-1)·MS_within)
    """

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores must be a 2-D (n_scenarios, k_repeats) matrix")
    n, k = scores.shape
    grand = scores.mean()
    row_means = scores.mean(axis=1)
    ms_between = k * ((row_means - grand) ** 2).sum() / max(n - 1, 1)
    ms_within = ((scores - row_means[:, None]) ** 2).sum() / max(n * (k - 1), 1)
    denominator = ms_between + (k - 1) * ms_within
    if denominator == 0:
        # Degenerate zero-variance matrix (e.g. an all-pass run with
        # byte-identical scores): perfect consistency by definition —
        # ICC := 1.0. Without this rule a perfectly deterministic green
        # run would classify `unstable` (review finding F2).
        return 1.0, float(ms_within)
    return float((ms_between - ms_within) / denominator), float(ms_within)


def divergence_step(step_signatures: Sequence[Sequence[str]]) -> int | None:
    """First step index at which repeated trajectories fork (R§1 #17 — 69%
    fork at step 2, so this is cheap, high-signal evidence). Signatures are
    normalized step strings (tool name + outcome class, no payloads).
    Returns None when all repeats share one trajectory."""

    longest = max((len(s) for s in step_signatures), default=0)
    for index in range(longest):
        prefixes = {tuple(s[: index + 1]) for s in step_signatures}
        if len(prefixes) > 1:
            return index
    return None


def determinism_metrics(
    step_signatures: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Trajectory-distribution metrics, kept strictly separate from quality
    (R§1 #15). Entropy is Shannon entropy in bits over distinct trajectories."""

    trajectories = [tuple(signature) for signature in step_signatures]
    if not trajectories:
        return {"distinct_trajectory_count": 0, "trajectory_entropy": 0.0}
    counts: dict[tuple[str, ...], int] = {}
    for trajectory in trajectories:
        counts[trajectory] = counts.get(trajectory, 0) + 1
    total = len(trajectories)
    entropy = -sum(
        (count / total) * math.log2(count / total) for count in counts.values()
    )
    return {
        "distinct_trajectory_count": len(counts),
        "trajectory_entropy": round(float(entropy), 6),
    }


def step_signature_from_events(
    events: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Normalize a transcript into step strings for divergence detection:
    tool name + outcome class, message marks — never payloads."""

    signature: list[str] = []
    for event in events:
        channel = str(event.get("channel") or "")
        event_type = str(event.get("type") or "")
        payload = event.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        if channel == "tool":
            name = str(payload.get("name") or payload.get("tool") or "tool")
            if payload.get("error") or payload.get("ok") is False:
                outcome = "error"
            else:
                outcome = "ok"
            signature.append(f"tool:{name}:{outcome}")
        elif channel in ("user", "agent"):
            signature.append(f"{channel}:{event_type}")
    return signature


@dataclasses.dataclass
class LaneRunResult:
    lane: str
    evidence_class: str            # stamped at construction; member of EVIDENCE_CLASSES
    repeats: int
    verdict: str                   # "pass" | "fail" | "unstable" | "void"
    per_repeat: list[dict]         # {score, passed, failure_layer, transcript_path}
    icc: float | None
    within_variance: float | None
    divergence_step: int | None
    determinism: dict              # {distinct_trajectory_count, trajectory_entropy}
    quarantined_repeats: int       # lane_infra rows excluded from stats (R§1 #7 validate-then-score)
    required_env: list[str]        # NAMES only, never values
    end_state_diff: dict | None    # before/after snapshot (R§1 #14 Saber)
    # --- run identity + budget mechanics (ARCH §4) — open details, simplest
    # additive fields consistent with the architecture: ---------------------
    run_id: str = ""
    rung: str | int = 1
    framework: str | None = None
    framework_version: str | None = None
    version_requirement: str | None = None
    version_ok: bool | None = None
    repeats_requested: int = 0
    repeats_completed: int = 0
    budget_cap_s: float = 0.0
    budget_spent_s: float = 0.0
    verdict_reason: str | None = None
    findings: list[dict] = dataclasses.field(default_factory=list)
    artifacts_dir: str | None = None

    def to_block(self) -> dict[str, Any]:
        """The ``live_lane`` evidence block of the run.v1 payload."""

        return dataclasses.asdict(self)


def run_repeated(
    run_once: Callable[[int, TranscriptRecorder], dict],
    *,
    lane: str,
    evidence_class: str,
    repeats: int = DEFAULT_REPEATS,  # P3-D2 default; --repeats override upstream
    budget_s: float | None = None,   # None → LANE_BUDGET_S.get(lane, default):
                                     # 600 s default, 900 s voice lanes (P3-D2)
    unstable_icc_floor: float = UNSTABLE_ICC_FLOOR,
    required_env: Sequence[str] = (),
    artifacts_dir: str | Path | None = None,
    run_id: str | None = None,
    rung: str | int = 1,
    framework: str | None = None,
    version_requirement: str | None = None,
) -> LaneRunResult:
    """Repeat executor. lane_infra rows are quarantined (excluded from the
    score matrix AND counted); verifier evidence is mandatory per repeat —
    a repeat with no programmatic/judge/end-state verdict is itself
    lane_infra (R§1 #5: sampling without verification is the documented gap).

    Per-scenario verdict (R§3.4; lane-run exit policy lives in unit 6):
      pass     — every non-quarantined repeat passed AND icc >= floor
                 (zero-variance all-pass runs hit this via ICC := 1.0 above)
      fail     — every non-quarantined repeat failed
      unstable — mixed outcomes, or icc < floor; quarantined like a flaky
                 test with fork evidence attached, never a red/green coin flip
      void     — lane_infra consumed the sample (no scoreable repeats);
                 the ONLY source of `void` (PRD §4.1).
    """

    if evidence_class not in EVIDENCE_CLASSES:
        raise ValueError(f"unknown evidence_class: {evidence_class!r}")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    budget = float(budget_s) if budget_s is not None else lane_budget_s(lane)
    base_dir = (
        Path(artifacts_dir)
        if artifacts_dir is not None
        else Path(tempfile.mkdtemp(prefix=f"agent-learning-live-{lane}-"))
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    resolved_run_id = run_id or uuid.uuid4().hex
    started = time.monotonic()

    rows: list[LaneRun] = []
    findings: list[dict] = []
    framework_name = framework
    framework_version: str | None = None
    version_ok_observed: bool | None = None
    end_state_diff: dict | None = None
    budget_exhausted = False

    for index in range(repeats):
        if time.monotonic() - started >= budget:
            budget_exhausted = True
            break
        transcript = TranscriptRecorder(
            base_dir / f"repeat-{index:02d}.jsonl",
            required_env=required_env,
        )
        try:
            outcome: Mapping[str, Any] = run_once(index, transcript) or {}
        except Exception as exc:  # our machinery failing is lane_infra, never a score
            outcome = {
                "passed": None,
                "score": None,
                "failure_layer": "lane_infra",
                "void_reason": f"lane runner exception: {exc}",
                "detail": f"lane runner exception: {exc}",
            }
        finally:
            summary = transcript.close()

        failure_layer = outcome.get("failure_layer")
        if failure_layer is not None and failure_layer not in FAILURE_LAYERS:
            failure_layer = "lane_infra"
        passed = outcome.get("passed")
        if passed is None and failure_layer is None:
            # No verdict at all → the repeat itself is lane_infra (R§1 #5).
            failure_layer = "lane_infra"
            outcome = dict(outcome)
            outcome.setdefault(
                "void_reason", "no verifier evidence for this repeat"
            )
            outcome.setdefault(
                "detail", "no verifier evidence for this repeat"
            )
        quarantined = failure_layer == "lane_infra"
        score = outcome.get("score")
        if score is None and not quarantined:
            score = 1.0 if passed else 0.0

        version_info = outcome.get("version")
        if isinstance(version_info, Mapping):
            framework_name = framework_name or version_info.get("framework")
            framework_version = framework_version or version_info.get(
                "framework_version"
            )
            if version_info.get("version_ok") is False:
                version_ok_observed = False
                findings.append(
                    {
                        "type": "live_lane_framework_version_mismatch",
                        "level": "error",
                        "repeat": index,
                        "detail": version_info.get("void_reason"),
                    }
                )
            elif version_ok_observed is None:
                version_ok_observed = bool(version_info.get("version_ok"))
        if isinstance(outcome.get("end_state_diff"), Mapping):
            end_state_diff = dict(outcome["end_state_diff"])

        if not summary.get("complete", True):
            findings.append(
                {
                    "type": "live_lane_transcript_truncated",
                    "level": "warning",
                    "repeat": index,
                    "detail": summary.get("truncated"),
                }
            )

        rows.append(
            LaneRun(
                index=index,
                passed=None if quarantined else bool(passed),
                score=None if quarantined else float(score),
                failure_layer=failure_layer,
                quarantined=quarantined,
                evidence_class=evidence_class,
                detail=str(outcome.get("detail") or ""),
                void_reason=outcome.get("void_reason"),
                transcript_path=str(summary.get("path")),
                transcript_complete=bool(summary.get("complete", True)),
                transcript_sha256=summary.get("sha256"),
                step_signature=tuple(outcome.get("step_signature") or ()),
            )
        )

    budget_spent = time.monotonic() - started
    scoreable = [row for row in rows if not row.quarantined]
    quarantined_count = sum(1 for row in rows if row.quarantined)

    if scoreable:
        matrix = np.asarray([[row.score for row in scoreable]], dtype=float)
        icc, within = icc_and_within_variance(matrix)
    else:
        icc, within = None, None

    signatures = [
        list(row.step_signature) for row in scoreable if row.step_signature
    ]
    fork_step = divergence_step(signatures) if signatures else None
    determinism = determinism_metrics(signatures)

    verdict_reason: str | None = None
    if not scoreable:
        verdict = "void"
        verdict_reason = "lane_infra_consumed_sample"
        findings.append(
            {
                "type": "live_lane_infra_void",
                "level": "error",
                "detail": "lane_infra consumed the sample (no scoreable repeats)",
            }
        )
    elif all(row.passed for row in scoreable):
        if icc is not None and icc < unstable_icc_floor:
            verdict = "unstable"
            verdict_reason = "icc_below_floor"
        else:
            verdict = "pass"
    elif all(not row.passed for row in scoreable):
        verdict = "fail"
    else:
        verdict = "unstable"
        verdict_reason = "mixed_outcomes"

    if budget_exhausted and verdict == "pass":
        # Hitting a cap mid-run yields `unstable` with reason budget_exhausted
        # rather than a silently smaller n (ARCH §4 budget mechanics).
        verdict = "unstable"
        verdict_reason = "budget_exhausted"

    if verdict == "unstable":
        findings.append(
            {
                "type": "live_lane_scenario_unstable",
                "level": "warning",
                "detail": {
                    "reason": verdict_reason,
                    "icc": icc,
                    "divergence_step": fork_step,
                },
            }
        )

    return LaneRunResult(
        lane=lane,
        evidence_class=evidence_class,
        repeats=repeats,
        verdict=verdict,
        per_repeat=[row.to_row() for row in rows],
        icc=icc,
        within_variance=within,
        divergence_step=fork_step,
        determinism=determinism,
        quarantined_repeats=quarantined_count,
        required_env=[str(name) for name in required_env],
        end_state_diff=end_state_diff,
        run_id=resolved_run_id,
        rung=rung,
        framework=framework_name,
        framework_version=framework_version,
        version_requirement=version_requirement,
        version_ok=version_ok_observed,
        repeats_requested=repeats,
        repeats_completed=len(rows),
        budget_cap_s=budget,
        budget_spent_s=round(budget_spent, 6),
        verdict_reason=verdict_reason,
        findings=findings,
        artifacts_dir=str(base_dir),
    )


def primary_transcript_events(result: LaneRunResult) -> list[dict[str, Any]]:
    """Events of the first scoreable repeat (falling back to the first row) —
    the transcript the lane normalizes into its state keys."""

    from ._transcript import read_transcript

    rows = [row for row in result.per_repeat if not row.get("quarantined")]
    rows = rows or list(result.per_repeat)
    for row in rows:
        path = row.get("transcript_path")
        if path and Path(str(path)).is_file():
            return read_transcript(str(path))
    return []


def lane_run_payload(
    result: LaneRunResult,
    *,
    name: str | None = None,
    scenario: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
    states: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize a lane run into the standard ``agent-learning.run.v1``
    payload via the existing public envelope, with live-only fields under a
    ``live_lane`` evidence block — same artifact kind, same state keys, plus
    live evidence (the graduation contract, R§3.1)."""

    payload: dict[str, Any] = {
        "kind": AGENT_LEARNING_RUN_KIND,
        "name": str(name or f"live-{result.lane}-run-{result.run_id[:8]}"),
        "evidence_class": result.evidence_class,
        "live_lane": result.to_block(),
        "findings": list(result.findings),
        "summary": {
            "verdict": result.verdict,
            "verdict_reason": result.verdict_reason,
            "repeats": result.repeats,
            "repeats_completed": result.repeats_completed,
            "quarantined_repeats": result.quarantined_repeats,
            "icc": result.icc,
            "divergence_step": result.divergence_step,
        },
    }
    if scenario is not None:
        payload["scenario"] = dict(scenario)
    if manifest is not None:
        payload["manifest"] = dict(manifest)
    if states:
        for state_key, state_value in states.items():
            payload[str(state_key)] = state_value
    if metadata:
        payload["metadata"] = dict(metadata)
    return public_payload(payload, kind=AGENT_LEARNING_RUN_KIND)


# --- dual-channel voice evidence (3B/3C — PRD §4.2 / guide §3.5) -------------


def _activity_mask(
    pcm: np.ndarray,
    *,
    frame_samples: int,
    energy_threshold_db: float,
) -> np.ndarray:
    samples = np.asarray(pcm, dtype=float)
    if samples.size == 0:
        return np.zeros(0, dtype=bool)
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples = samples / peak
    frame_count = int(np.ceil(samples.size / frame_samples))
    padded = np.zeros(frame_count * frame_samples, dtype=float)
    padded[: samples.size] = samples
    frames = padded.reshape(frame_count, frame_samples)
    rms = np.sqrt((frames**2).mean(axis=1))
    with np.errstate(divide="ignore"):
        rms_db = 20.0 * np.log10(np.where(rms > 0, rms, 1e-12))
    return rms_db > energy_threshold_db


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous active [start, end) frame spans."""

    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def derive_channel_evidence(
    user_pcm: np.ndarray,
    agent_pcm: np.ndarray,
    *,
    sample_rate: int,
    frame_ms: float = 20.0,
    energy_threshold_db: float = -40.0,
) -> dict[str, Any]:
    """Compute the ``channels.derived`` block from the two PCM streams —
    never from transcripts (R§3.5): barge-in latency, overlap totals,
    post-interrupt recovery turns, and agent onset (ttfb). Pure-numpy
    energy/onset detection; rung-2+ only (rung 1 has no channels block —
    the rung-1 honesty rule, guide §3.5)."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    frame_samples = max(int(sample_rate * frame_ms / 1000.0), 1)
    user_mask = _activity_mask(
        user_pcm, frame_samples=frame_samples, energy_threshold_db=energy_threshold_db
    )
    agent_mask = _activity_mask(
        agent_pcm, frame_samples=frame_samples, energy_threshold_db=energy_threshold_db
    )
    width = max(len(user_mask), len(agent_mask))
    user_full = np.zeros(width, dtype=bool)
    agent_full = np.zeros(width, dtype=bool)
    user_full[: len(user_mask)] = user_mask
    agent_full[: len(agent_mask)] = agent_mask

    overlap = user_full & agent_full
    overlap_spans = _segments(overlap)
    overlap_total_ms = float(sum(end - start for start, end in overlap_spans)) * frame_ms

    user_spans = _segments(user_full)
    agent_spans = _segments(agent_full)

    # ttfb: first agent onset after the first user utterance ends.
    ttfb_ms: float | None = None
    if user_spans and agent_spans:
        first_user_end = user_spans[0][1]
        for start, _ in agent_spans:
            if start >= first_user_end:
                ttfb_ms = float(start - first_user_end) * frame_ms
                break

    # barge-in: first user onset that lands mid-agent-speech; latency runs
    # until the agent yields (its active span ends).
    barge_in_latency_ms: float | None = None
    barge_frame: int | None = None
    for user_start, _ in user_spans:
        for agent_start, agent_end in agent_spans:
            if agent_start < user_start < agent_end:
                barge_in_latency_ms = float(agent_end - user_start) * frame_ms
                barge_frame = user_start
                break
        if barge_in_latency_ms is not None:
            break

    # recovery: agent speech segments after the interrupt until the first
    # segment that starts clear of user speech (a clean turn).
    post_interrupt_recovery_turns: int | None = None
    if barge_frame is not None:
        turns = 0
        for agent_start, _ in agent_spans:
            if agent_start <= barge_frame:
                continue
            turns += 1
            if not user_full[agent_start]:
                break
        post_interrupt_recovery_turns = turns

    return {
        "barge_in_latency_ms": barge_in_latency_ms,
        "overlap_total_ms": overlap_total_ms,
        "overlap_segments": len(overlap_spans),
        "post_interrupt_recovery_turns": post_interrupt_recovery_turns,
        "ttfb_ms": ttfb_ms,
        "frame_ms": frame_ms,
        "energy_threshold_db": energy_threshold_db,
    }
