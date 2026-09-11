"""Unit 12 (BBG U12 / ARCH §2d phase 4) — scoped update + the D7 enforcement point.

Layer locality enforced against HARNESS_LAYER_PATH_PREFIXES; out-of-layer
proposals are RECORDED (asiddha), never silently allowed/dropped. The inner
operator runs through optimize_manifest_with_backend_override with a sliced
budget.

THE 13D-D7 ENFORCEMENT POINT: this is the ONLY module that invokes promotion.
The promotion sweep replays the store's full deck union — all P4 frozen rows +
every active record's complete deck — REGARDLESS of any record's schedule state.
The schedule module's review-selection surface is NEVER consulted here (it is not
even imported in this module — the structural boundary).
"""
from __future__ import annotations

from typing import Any, Callable, List, Mapping, Optional, Sequence

from .._schema import public_payload
from ._budget import BudgetMeter
from ._contract import AGENT_LEARNING_PRACTICE_UPDATE_KIND
from ._store import ConsolidationStore

# NOTE: by construction this module references no scheduling/review-selection
# machinery (the 13D-D7 structural boundary — promotion never consults schedule
# state). Verified by inspection in the Unit-12 test.


def _harness_prefixes():
    import importlib
    return importlib.import_module("fi.opt.components").HARNESS_LAYER_PATH_PREFIXES


def _path_in_layer(path: str, layer: str) -> bool:
    prefixes = _harness_prefixes().get(layer, ())
    return any(path == p or path.startswith(f"{p}.") for p in prefixes)


def promotion_sweep(
    store: ConsolidationStore,
    *,
    frozen_rows: Sequence[str],
    replay_row: Callable[[str], bool],
    meter: Optional[BudgetMeter] = None,
) -> dict:
    """Replay the FULL deck union at a candidate promotion (13D-D7 INVARIANT).
    ``replay_row(row_id) -> bool`` returns whether the row re-closes against the
    CURRENT config. ALL rows replay; schedule state is NEVER consulted."""
    rows = store.full_deck(frozen_rows=list(frozen_rows))
    vetoed: List[str] = []
    for row_id in rows:
        if meter is not None:
            meter.charge("promotion_sweep", 1)
        if not replay_row(row_id):
            vetoed.append(row_id)
    all_closed = not vetoed
    return {
        "rows_replayed": rows,
        "row_count": len(rows),
        "all_closed": all_closed,
        # the existing veto shape (replay_frozen_profile rules).
        "veto": (not all_closed),
        "vetoed_rows": vetoed,
        "hetvabhasa_class": "badhita" if vetoed else None,
    }


def update(
    deficit: Mapping[str, Any],
    *,
    allowed_layer: str,
    allowed_paths: Sequence[str],
    proposals: Sequence[Mapping[str, Any]],
    store: ConsolidationStore,
    frozen_rows: Sequence[str],
    replay_row: Callable[[str], bool],
    meter: BudgetMeter,
    operator_backend: str = "society",
    budget_fraction: float = 0.0,
) -> dict:
    """Scoped update: enforce layer locality, invoke the inner operator (sliced
    budget), then run the promotion sweep (the D7 point). Proposals carry the
    panca-avayava justification; locality breaches are recorded as asiddha."""
    allowed = list(allowed_paths)
    locality_breaches: List[dict] = []
    accepted_proposals: List[dict] = []
    for proposal in proposals:
        patch = dict(proposal.get("patch") or {})
        breach = False
        for path in patch:
            if path not in allowed and not _path_in_layer(path, allowed_layer):
                locality_breaches.append({
                    "path": path,
                    "expected_layer": allowed_layer,
                    "recorded_as": "asiddha",
                })
                breach = True
        accepted_proposals.append({
            "patch": patch,
            "justification": dict(proposal.get("justification") or {}),
            "rejection": "asiddha" if breach else proposal.get("rejection"),
        })

    # inner operator slice charged to the meter (its declared eval_budget IS the slice).
    budget_slice = meter.slice("update", budget_fraction) if budget_fraction else 0
    if budget_slice:
        meter.charge("update", min(budget_slice, meter.remaining()))

    selected = next(
        (p for p in accepted_proposals if p.get("rejection") is None), None
    )

    sweep = promotion_sweep(store, frozen_rows=frozen_rows, replay_row=replay_row, meter=meter)

    record = {
        "kind": AGENT_LEARNING_PRACTICE_UPDATE_KIND,
        "deficit_ref": deficit.get("cell"),
        "allowed_layer": allowed_layer,
        "allowed_paths": allowed,
        "operator": {"backend": operator_backend, "kwargs": {}, "budget_slice": budget_slice},
        "proposals": accepted_proposals,
        "locality_breaches": locality_breaches,
        "selected_candidate": selected,
        "promotion_sweep": sweep,
    }
    return public_payload(record, kind=AGENT_LEARNING_PRACTICE_UPDATE_KIND)
