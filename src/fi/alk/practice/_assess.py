"""Unit 10 (BBG U10 / ARCH §2d phase 1) — ASSESS: battery over the obligation grid.

Runs the battery over ``scenarios × cast × perturbations`` at ScenarioBinding
weights by deriving run manifests per cell (each scored episode charges the
meter), collecting verdict rows via loss.verdict_row, composing loss.loss_report.
Emits ``agent-learning.practice-report.v1`` through public_payload.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional

from .._schema import public_payload
from .. import loss as _loss
from ._budget import BudgetMeter
from ._contract import AGENT_LEARNING_PRACTICE_REPORT_KIND


def _grid_cells(simulation: Mapping[str, Any]) -> List[dict]:
    """Enumerate obligation cells from the P7 CoverageDeclaration vocabulary;
    degenerate single cell when no coverage declared."""
    cells: List[dict] = []
    for binding in simulation.get("scenarios") or []:
        scenario = binding.get("scenario") or {}
        coverage = scenario.get("coverage") or {}
        intents = coverage.get("intents") or [None]
        perturbations = coverage.get("perturbations") or [None]
        for member in binding.get("cast") or []:
            for intent in intents:
                for perturbation in perturbations:
                    cells.append({
                        "intent": intent,
                        "persona": member.get("persona"),
                        "perturbation": perturbation,
                        "obligation": None,
                        "weight": float(binding.get("weight", 1.0)),
                    })
    if not cells:
        cells.append({"intent": None, "persona": None, "perturbation": None,
                      "obligation": None, "weight": 1.0})
    return cells


def assess(
    simulation: Mapping[str, Any],
    objective: Mapping[str, Any],
    *,
    meter: BudgetMeter,
    round_no: int,
    seed: int,
    cell_scorer: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    parent_report_hash: Optional[str] = None,
    repeats: int = 1,
    coverage_source: str = "declared",
) -> dict:
    """Run the battery. ``cell_scorer(cell) -> {scalar, verdict, evidence_class}``
    is the per-cell episode evaluator (injected for determinism/testing; in
    production it derives + runs a run manifest). Each scored episode charges the
    meter."""
    cells = _grid_cells(simulation)
    verdicts: List[dict] = []
    calibration_mass_by_cell: Dict[str, float] = {}
    for cell in cells:
        for _ in range(max(1, int(repeats))):
            meter.charge("assess", 1)
            scored = cell_scorer(cell)
            row = _loss.verdict_row(
                eval_ref=scored.get("eval", "agent_report"),
                cell=cell,
                scalar=float(scored.get("scalar", 0.0)),
                verdict=str(scored.get("verdict", "pass")),
                evidence_class=str(scored.get("evidence_class", "local_gate")),
                fidelity_admissible=bool(scored.get("fidelity_admissible", True)),
                provenance={"round": round_no, "seed": seed},
            )
            verdicts.append(row)
            if row["verdict"] == "unstable":
                key = _loss._cell_key(cell)
                calibration_mass_by_cell[key] = round(
                    calibration_mass_by_cell.get(key, 0.0) + 1.0, 6
                )

    loss_report = _loss.loss_report(objective, verdicts, budget_consumed=meter.consumed)
    report = {
        "kind": AGENT_LEARNING_PRACTICE_REPORT_KIND,
        "round": int(round_no),
        "objective_version": objective.get("version"),
        "loss_report": loss_report,
        "grid": {
            "cells_total": len(cells),
            "cells_assessed": len(cells),
            "coverage_source": coverage_source,
        },
        "calibration_mass_by_cell": calibration_mass_by_cell,
        "budget_consumed": meter.consumed,
        "seed": int(seed),
        "parent": parent_report_hash,
    }
    return public_payload(report, kind=AGENT_LEARNING_PRACTICE_REPORT_KIND)


def practice_report(*args: Any, **kwargs: Any) -> dict:
    """Public alias for the ASSESS report builder (facade export)."""
    return assess(*args, **kwargs)
