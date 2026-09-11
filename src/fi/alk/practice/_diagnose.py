"""Unit 10 (BBG U10 / ARCH §2d phase 2) — DIAGNOSE: pure composition.

Ranks weak cells, attributes each to a harness_layer ∈ HARNESS_LAYERS via
ComponentDiagnosis and relevant_search_paths (narrowing, never widening). Credit
method "layer_scoped" always; "counterfactual_replay" (13C T7) only budget-
permitting (fallback = layer scoping only). Emits
``agent-learning.practice-deficits.v1`` ranked deterministically (loss desc,
tie-break by cell content hash). No new machinery.
"""
from __future__ import annotations

import json
from typing import Any, List, Mapping, Optional

from .._schema import public_payload
from ._contract import AGENT_LEARNING_PRACTICE_DEFICITS_KIND


def _components():
    import importlib
    return importlib.import_module("fi.opt.components")


def _cell_hash(cell: Mapping[str, Any]) -> str:
    return json.dumps(cell, sort_keys=True, default=str)


def diagnose(
    practice_report: Mapping[str, Any],
    *,
    search_space: Mapping[str, Any],
    layer_hint: Optional[Mapping[str, str]] = None,
    allow_counterfactual: bool = False,
) -> dict:
    """Pure composition over the ASSESS report's loss cells. ``layer_hint`` maps a
    cell key → harness_layer (from upstream diagnosis); default 'execution'."""
    components = _components()
    harness_layers = components.HARNESS_LAYERS
    prefixes = components.HARNESS_LAYER_PATH_PREFIXES
    layer_hint = dict(layer_hint or {})

    loss_report = practice_report.get("loss_report") or {}
    cells = loss_report.get("cells") or []
    # rank weak cells: loss desc, tie-break by cell content hash.
    ranked = sorted(
        cells,
        key=lambda c: (-float(c.get("loss", 0.0)), _cell_hash(c.get("cell") or {})),
    )

    deficits: List[dict] = []
    for cell_report in ranked:
        cell = cell_report.get("cell") or {}
        if float(cell_report.get("loss", 0.0)) <= 0.0:
            continue  # closed cells are not deficits
        layer = layer_hint.get(_cell_hash(cell), "execution")
        if layer not in harness_layers:
            layer = "execution"
        # narrowing search paths from the layer's prefixes.
        layer_prefixes = prefixes.get(layer, ())
        narrowed = sorted(
            path for path in search_space
            if any(path == p or path.startswith(f"{p}.") for p in layer_prefixes)
        )
        method = "counterfactual_replay" if allow_counterfactual else "layer_scoped"
        deficits.append({
            "cell": cell,
            "harness_layer": layer,
            "search_paths": narrowed,
            "credit": {"method": method, "rows": []},
            "evidence_rows": cell_report.get("verdicts") or [],
        })

    report = {
        "kind": AGENT_LEARNING_PRACTICE_DEFICITS_KIND,
        "round": practice_report.get("round"),
        "objective_version": practice_report.get("objective_version"),
        "deficits": deficits,
    }
    return public_payload(report, kind=AGENT_LEARNING_PRACTICE_DEFICITS_KIND)
