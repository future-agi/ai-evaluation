"""Unit 5 (BBG U5 / ARCH §2c) — the declared objective channel (evals-as-loss).

The loss is a FIRST-CLASS, versioned artifact built from the EXISTING eval
lineage generalized (never a parallel system). 13D-D8: a declared objective
without Goodhart guards is a validation error; ``unstable`` → calibration
channel; ``void`` → excluded-and-recorded. AD-F: ``source:"derived"`` is valid
for replication but refused wherever an eval becomes a training loss.

VERDICTS / EVIDENCE_CLASSES are IMPORTED from live/_contract.py, never
redeclared (ARCH §1.7). The Persona-rule canonicalization is factored locally
(AD-D) rather than importing the private engine helper.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ._schema import public_payload
from .live._contract import EVIDENCE_CLASSES, RELEASE_ADMISSIBLE_EVIDENCE_CLASSES, VERDICTS

AGENT_LEARNING_OBJECTIVE_KIND = "agent-learning.objective.v1"
AGENT_LEARNING_LOSS_REPORT_KIND = "agent-learning.loss-report.v1"

OBJECTIVE_SOURCES = ("declared", "derived")
AGGREGATION_MODES = ("obligation_cells",)
AGGREGATION_CONJUNCTIONS = ("all_cells_must_close",)
AGGREGATION_PROJECTIONS = ("weighted_mean",)
TERM_SCOPES = ("turn", "episode", "run")
TERM_DIRECTIONS = ("maximize", "minimize")
UNSTABLE_POLICY = "calibration_channel"
EXCLUSION_POLICY = "record"


# --- Persona-rule canonicalization (AD-D), factored locally ----------------
def _round_floats(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, Mapping):
        return {k: _round_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(v) for v in value]
    return value


def _content_hash(payload: Mapping[str, Any]) -> str:
    rounded = _round_floats(dict(payload))
    canonical = json.dumps(rounded, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ObjectiveError(ValueError):
    """Raised when an objective fails the §2c contract."""


def compile_objective(payload: Mapping[str, Any]) -> dict:
    """Validate + stamp an ObjectiveSpec. Guards mandatory when declared (D8)."""
    raw = dict(payload)
    evals = raw.get("evals") or []
    if not isinstance(evals, Sequence) or len(evals) < 1:
        raise ObjectiveError("objective.evals must list >= 1 ObjectiveTerm rows")

    terms: List[dict] = []
    for index, term in enumerate(evals, start=1):
        if not isinstance(term, Mapping):
            raise ObjectiveError(f"objective.evals[{index}] must be a mapping")
        eval_ref = term.get("eval")
        if not eval_ref:
            raise ObjectiveError(f"objective.evals[{index}] requires an 'eval' ref")
        weight = float(term.get("weight", 1.0))
        if weight <= 0:
            raise ObjectiveError(f"objective.evals[{index}].weight must be > 0")
        direction = str(term.get("direction", "maximize"))
        if direction not in TERM_DIRECTIONS:
            raise ObjectiveError(f"objective.evals[{index}].direction not in {TERM_DIRECTIONS}")
        scope = str(term.get("scope", "run"))
        if scope not in TERM_SCOPES:
            raise ObjectiveError(f"objective.evals[{index}].scope not in {TERM_SCOPES}")
        compiled_term = {
            "eval": str(eval_ref),
            "weight": round(weight, 6),
            "direction": direction,
            "threshold": round(float(term.get("threshold", 0.7)), 6),
            "scope": scope,
            "cells": list(term.get("cells") or []),
        }
        # preserve the deterministic ground-truth ANCHOR marker through
        # compilation — it is load-bearing downstream (reward-hack detector +
        # task anchor-coverage), not just an authoring hint.
        if term.get("anchor") is True:
            compiled_term["anchor"] = True
        terms.append(compiled_term)

    aggregation = dict(raw.get("aggregation") or {})
    mode = aggregation.get("mode", "obligation_cells")
    conjunction = aggregation.get("conjunction", "all_cells_must_close")
    projection = aggregation.get("projection", "weighted_mean")
    if mode not in AGGREGATION_MODES:
        raise ObjectiveError(f"aggregation.mode {mode!r} not in {AGGREGATION_MODES}")
    if conjunction not in AGGREGATION_CONJUNCTIONS:
        raise ObjectiveError(f"aggregation.conjunction {conjunction!r} not in {AGGREGATION_CONJUNCTIONS}")
    if projection not in AGGREGATION_PROJECTIONS:
        raise ObjectiveError(f"aggregation.projection {projection!r} not in {AGGREGATION_PROJECTIONS}")

    source = str(raw.get("source", "declared"))
    if source not in OBJECTIVE_SOURCES:
        raise ObjectiveError(f"objective.source {source!r} not in {OBJECTIVE_SOURCES}")

    guards = dict(raw.get("guards") or {})
    if source == "declared":
        sentinel = guards.get("sentinel_rows") or []
        canary = guards.get("canary_evals") or []
        min_count = int(guards.get("min_guard_count", 0))
        if (not sentinel and not canary) or min_count < 1:
            # 13D-D8 — no --no-guards override.
            raise ObjectiveError(
                "objective_guards_missing: a declared objective is a training loss "
                "and MUST carry Goodhart guards (sentinel_rows / canary_evals, "
                "min_guard_count >= 1). There is no override."
            )

    compiled = {
        "kind": AGENT_LEARNING_OBJECTIVE_KIND,
        "evals": terms,
        "aggregation": {"mode": mode, "conjunction": conjunction, "projection": projection},
        "guards": {
            "sentinel_rows": list(guards.get("sentinel_rows") or []),
            "canary_evals": list(guards.get("canary_evals") or []),
            "min_guard_count": int(guards.get("min_guard_count", 0)),
        },
        "source": source,
        "unstable_policy": UNSTABLE_POLICY,
        "exclusion_policy": EXCLUSION_POLICY,
    }
    compiled["version"] = _content_hash({k: v for k, v in compiled.items() if k != "version"})
    return compiled


def refuse_derived_for_training(objective: Optional[Mapping[str, Any]]) -> None:
    """AD-F: raise when a source:'derived' objective is consumed as a training
    loss (called by the trainer + build_practice_loop_manifest)."""
    if objective is None:
        raise ObjectiveError(
            "objective_guards_missing: the simulation declares no objective; a "
            "training target requires a source:'declared' objective with guards"
        )
    if str(objective.get("source", "declared")) == "derived":
        raise ObjectiveError(
            "objective_guards_missing: a source:'derived' objective is valid for "
            "replication but REFUSED as a training loss (AD-F). Declare an "
            "objective with guards."
        )


def verdict_row(
    *,
    eval_ref: str,
    cell: Mapping[str, Any],
    scalar: float,
    verdict: str,
    evidence_class: str,
    fidelity_admissible: bool = True,
    context_admissible_classes: Sequence[str] = RELEASE_ADMISSIBLE_EVIDENCE_CLASSES,
    provenance: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Assemble a §2c verdict object. ``verdict`` is echoed from the stats layer,
    never recomputed (live/_stats.py semantics)."""
    if verdict not in VERDICTS:
        raise ObjectiveError(f"verdict {verdict!r} not in {VERDICTS}")
    if evidence_class not in EVIDENCE_CLASSES:
        raise ObjectiveError(f"evidence_class {evidence_class!r} not in {EVIDENCE_CLASSES}")
    # admissible fallback rule until 13B T2 lands.
    admissible = (
        verdict in ("pass", "fail")
        and evidence_class in tuple(context_admissible_classes)
        and bool(fidelity_admissible)
    )
    return {
        "eval": str(eval_ref),
        "cell": {
            "intent": cell.get("intent"),
            "persona": cell.get("persona"),
            "perturbation": cell.get("perturbation"),
            "obligation": cell.get("obligation"),
        },
        "scalar": round(float(scalar), 6),
        "verdict": verdict,
        "evidence_class": evidence_class,
        "admissible": bool(admissible),
        "provenance": dict(provenance or {}),
    }


def _cell_key(cell: Mapping[str, Any]) -> str:
    return json.dumps(
        {k: cell.get(k) for k in ("intent", "persona", "perturbation", "obligation")},
        sort_keys=True, default=str,
    )


def loss_report(
    objective: Mapping[str, Any],
    verdicts: Sequence[Mapping[str, Any]],
    *,
    budget_consumed: int = 0,
) -> dict:
    """Deterministic composition (§2c): per cell loss = 1 - admissible_score over
    admissible verdicts; unstable → zero gradient + unstable_mass; void →
    excluded + void_count; conjunction = all-cells-must-close; scalar =
    weighted mean of per-cell admissible scores by term weights."""
    weights = {t["eval"]: float(t["weight"]) for t in objective.get("evals", [])}

    cells: Dict[str, dict] = {}
    for row in verdicts:
        key = _cell_key(row.get("cell") or {})
        bucket = cells.setdefault(key, {
            "cell": dict(row.get("cell") or {}),
            "admissible_scores": [],
            "admissible_weights": [],
            "admissible_count": 0,
            "unstable_mass": 0.0,
            "void_count": 0,
            "verdicts": [],
        })
        bucket["verdicts"].append(dict(row))
        verdict = row.get("verdict")
        weight = weights.get(row.get("eval"), 1.0)
        if verdict == "unstable":
            bucket["unstable_mass"] = round(bucket["unstable_mass"] + weight, 6)
            continue
        if verdict == "void":
            bucket["void_count"] += 1
            continue
        if row.get("admissible"):
            bucket["admissible_scores"].append(float(row.get("scalar", 0.0)))
            bucket["admissible_weights"].append(weight)
            bucket["admissible_count"] += 1

    cell_reports: List[dict] = []
    open_cells: List[Any] = []
    scalar_num = 0.0
    scalar_den = 0.0
    for key in sorted(cells):
        bucket = cells[key]
        scores = bucket["admissible_scores"]
        wts = bucket["admissible_weights"]
        if scores and sum(wts) > 0:
            admissible_score = sum(s * w for s, w in zip(scores, wts)) / sum(wts)
        else:
            admissible_score = 0.0
        cell_loss = round(1.0 - admissible_score, 6)
        closed = bool(scores) and admissible_score >= 0.0 and cell_loss <= (1.0 - _cell_threshold(objective))
        if not closed:
            open_cells.append(bucket["cell"])
        scalar_num += admissible_score * sum(wts)
        scalar_den += sum(wts)
        cell_reports.append({
            "cell": bucket["cell"],
            "loss": cell_loss,
            "admissible_count": bucket["admissible_count"],
            "unstable_mass": round(bucket["unstable_mass"], 6),
            "void_count": bucket["void_count"],
            "verdicts": bucket["verdicts"],
        })

    scalar = round(scalar_num / scalar_den, 6) if scalar_den > 0 else 0.0
    guards_block = _guard_outcomes(objective)
    report = {
        "kind": AGENT_LEARNING_LOSS_REPORT_KIND,
        "objective_version": objective.get("version"),
        "cells": cell_reports,
        "conjunction": {"closed": len(open_cells) == 0, "open_cells": open_cells},
        "scalar": scalar,
        "guards": guards_block,
        "budget_consumed": int(budget_consumed),
    }
    return public_payload(report, kind=AGENT_LEARNING_LOSS_REPORT_KIND)


def _cell_threshold(objective: Mapping[str, Any]) -> float:
    terms = objective.get("evals", [])
    if not terms:
        return 0.7
    return round(min(float(t.get("threshold", 0.7)) for t in terms), 6)


def _guard_outcomes(objective: Mapping[str, Any]) -> dict:
    guards = objective.get("guards") or {}
    return {
        "sentinel_rows": list(guards.get("sentinel_rows") or []),
        "canary_evals": list(guards.get("canary_evals") or []),
        "tripped": bool(guards.get("tripped")),
        "voids_report_for_training": bool(guards.get("tripped")),
    }


def objective_metric_weights(objective: Mapping[str, Any]) -> dict:
    """The derived view (§2c lineage alignment): reproduce the legacy optimizer
    profile metric_weights mapping shape. The gate byte-compares this against the
    incumbent hand-written map (derived_view_errors)."""
    weights: Dict[str, float] = {}
    for term in objective.get("evals", []):
        weights[str(term["eval"])] = round(float(term["weight"]), 6)
    return weights
