"""Unit 22 (BBG U22 / RU-7) — the capstone A/B harness.

An EXPERIMENT, not a release gate (gates stay deterministic; nothing here
registers a check). The harness runs the practice loop vs real search backends
at EQUAL TOTAL metered budget (the one meter, AD-I) over kit-local fixtures, and
REFUSES to print a headline unless every arm completed the same declared total
(``headline: null`` + ``ab_budget_mismatch`` otherwise — doctrine #11).

This module builds the harness so it CAN run offline-deterministically; running
the capstone experiment + writing the paper is a separate later task.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .._schema import public_payload
from ._contract import AGENT_LEARNING_PRACTICE_LOOP_KIND

# RU-7: real backend tokens only (the canon tuple stays closed). "greedy" = bandit.
CAPSTONE_ARMS = ("practice_loop", "gepa", "tpe", "society", "bandit")
# manifest-level ablation knobs of the practice arm (never a code fork).
CAPSTONE_ABLATIONS = ("a1_no_zpd", "a2_no_spacing", "a3_no_consolidation", "a4_no_calibration")


def _load_config(manifest_dir: Path) -> dict:
    config_path = manifest_dir / "capstone.json"
    if not config_path.exists():
        raise FileNotFoundError(f"capstone config not found at {config_path}")
    return json.loads(config_path.read_text())


def run_ab(manifest_dir: str | Path) -> dict:
    """Run the A/B harness. Reads ``capstone.json`` declaring the arms and the
    equal total budget; enforces the equal-budget headline rule (doctrine #11).

    The arm execution is offline-deterministic: each arm reports its declared
    total metered budget and a (placeholder until the experiment runs)
    retention_after_interference. Running the experiment itself is a later task;
    this harness validates the equal-budget contract and emits the ab_harness
    block."""
    manifest_dir = Path(manifest_dir)
    config = _load_config(manifest_dir)
    declared_total = int(config.get("eval_budget", 0))
    arms_decl = config.get("arms") or list(CAPSTONE_ARMS)

    arms: List[dict] = []
    budgets: set[int] = set()
    for arm in arms_decl:
        arm_total = int(config.get("arm_budgets", {}).get(arm, declared_total))
        budgets.add(arm_total)
        arms.append({
            "arm": arm,
            "total_metered_budget": arm_total,
            # best_found is printed per arm precisely so a search arm may visibly
            # win best-found while losing retention (the headline).
            "best_found": None,
            "retention_after_interference": None,
        })

    # equal TOTAL metered budget per arm (AD-I) — else headline null + warning.
    budget_match = len(budgets) == 1 and declared_total in budgets
    findings: List[dict] = []
    headline = None
    if not budget_match:
        findings.append({
            "type": "ab_budget_mismatch", "level": "warning",
            "reason": f"arms did not complete the same declared total ({sorted(budgets)} != {declared_total})",
        })
    else:
        headline = {"metric": "retention_after_interference", "by_arm": None,
                    "note": "populated when the experiment runs (a later task)"}

    payload = {
        "kind": AGENT_LEARNING_PRACTICE_LOOP_KIND,
        "ab_harness": {
            "arms": arms,
            "ablations": list(CAPSTONE_ABLATIONS),
            "equal_total_budget": declared_total,
            "budget_match": budget_match,
            "headline": headline,
            "findings": findings,
        },
    }
    return public_payload(payload, kind=AGENT_LEARNING_PRACTICE_LOOP_KIND)["ab_harness"]
