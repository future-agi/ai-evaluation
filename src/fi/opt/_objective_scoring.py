"""Canonical objective-anchored scoring — the ONE source of truth for the
benchmark score (B2), the reward-hack detector (B6), the Task objective (B1), AND
the optimizer's candidate scoring (bug #2 fix).

Lives in ``fi`` (the vendored engine) so the engine's optimizer integration can
use it WITHOUT importing the public ``fi.alk`` package — respecting the
vendored_engine_boundary. ``fi.alk.tasks`` re-exports these so its public
call sites are unchanged. Pure functions, no kit/engine cross-deps.

Why this exists: the engine's all-metrics mean (``summary.evaluation_score``) pins
~30/38 metrics at 1.0 and has near-zero dynamic range — a terrible agent and a
good one both score ~0.95. Scoring only the task's DECLARED objective terms gives
real dynamic range (a failing ground-truth anchor actually drops the score).
"""

from __future__ import annotations

from typing import Any, Mapping

# eval-ref -> engine metric_averages key. Exact match wins; this only fills gaps
# where the objective's declared ref differs from the engine's metric name.
METRIC_ALIASES = {
    "task_success": "task_completion",
    "artifact_grounding": "source_grounding",
    "tool_argument_correctness": "tool_argument_schema",
}


def resolve_metric(metrics: Mapping[str, Any], eval_ref: str) -> float | None:
    """Resolve an objective eval-ref to its value in the engine metric averages,
    via exact match then the known alias map. Returns None if unresolved."""

    if eval_ref in metrics:
        try:
            return float(metrics[eval_ref])
        except (TypeError, ValueError):
            return None
    alias = METRIC_ALIASES.get(eval_ref)
    if alias and alias in metrics:
        try:
            return float(metrics[alias])
        except (TypeError, ValueError):
            return None
    return None


def objective_score(metrics: Mapping[str, Any], objective: Mapping[str, Any]) -> dict[str, Any]:
    """Weighted mean over the objective's DECLARED terms, each resolved to a real
    engine metric. Returns {score, terms_resolved, terms_total, per_term}; score
    is None when no declared term resolves to a metric (caller falls back)."""

    terms = [t for t in (objective.get("evals") or []) if isinstance(t, Mapping) and t.get("eval")]
    per_term: dict[str, dict[str, Any]] = {}
    num = 0.0
    den = 0.0
    for term in terms:
        ref = str(term["eval"])
        weight = float(term.get("weight", 1.0))
        val = resolve_metric(metrics, ref)
        per_term[ref] = {"weight": weight, "value": val, "anchor": bool(term.get("anchor"))}
        if val is not None and weight > 0:
            num += weight * val
            den += weight
    resolved = sum(1 for v in per_term.values() if v["value"] is not None)
    score = (num / den) if den > 0 else None
    return {
        "score": round(score, 6) if score is not None else None,
        "terms_resolved": resolved,
        "terms_total": len(terms),
        "per_term": per_term,
    }


def has_declared_anchor_objective(objective: Mapping[str, Any] | None) -> bool:
    """True iff ``objective`` declares >=1 term explicitly marked ``anchor: true``.
    The gate for objective-anchored optimizer scoring: only manifests that DECLARE
    a real anchored objective opt in; structural/hook/legacy manifests (no declared
    objective) keep the engine's existing all-metrics score — no regression."""

    if not isinstance(objective, Mapping):
        return False
    return any(
        isinstance(t, Mapping) and t.get("eval") and t.get("anchor") is True
        for t in (objective.get("evals") or [])
    )
