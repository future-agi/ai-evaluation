from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, optimize


REQUIRED_ENV = "AGENT_LEARNING_SDK_TARGET_OPTIMIZATION_KEY"


def _approve_transition() -> dict[str, Any]:
    return {
        "id": "approve_refund",
        "actor": "agent",
        "resource": "refund",
        "action": "approve_refund",
        "required": True,
        "preconditions": {"refund.status": "pending"},
        "effects": {"refund.status": "approved"},
        "postconditions": {"refund.status": "approved"},
        "signals": ["refund_resolution"],
    }


def _base_config() -> dict[str, Any]:
    approve_refund_tool_call = {
        "id": "approve_refund",
        "name": "apply_world_transition",
        "arguments": {"id": "approve_refund"},
    }
    return {
        "agent": {
            "type": "scripted",
            "responses": [
                {
                    "content": "I will apply the refund transition.",
                    "tool_calls": [approve_refund_tool_call],
                }
            ],
        },
        "simulation": {
            "engine": "local_text",
            "min_turns": 1,
            "max_turns": 1,
            "auto_execute_tools": True,
            "environments": [
                {
                    "type": "world_contract",
                    "data": {
                        "name": "target-optimization-refund-world",
                        "actors": ["agent", "customer"],
                        "resources": ["refund"],
                        "initial_state": {
                            "policy": {"can_refund": True},
                            "refund": {"status": "pending"},
                        },
                        "transitions": [],
                        "invariants": [
                            {
                                "id": "policy_allows_refunds",
                                "must": {"policy.can_refund": True},
                            }
                        ],
                        "success_conditions": [
                            {
                                "id": "refund_approved",
                                "must": {"refund.status": "approved"},
                            }
                        ],
                    },
                }
            ],
        },
    }


def _evaluation_config() -> dict[str, Any]:
    return {
        "task_description": (
            "Optimize a target path in a local refund world contract."
        ),
        "expected_result": (
            "The selected world contract allows the scripted agent to approve "
            "the refund."
        ),
        "required_tools": ["apply_world_transition"],
        "available_tools": ["world_contract_status", "apply_world_transition"],
        "success_criteria": [
            "refund transition applied",
            "world contract terminal status is success",
        ],
        "required_world_contract": [
            "world_contract",
            "transition",
            "success_condition",
            "refund",
        ],
        "world_contract_quality": {
            "required_actors": ["agent", "customer"],
            "required_resources": ["refund"],
            "required_transitions": ["approve_refund"],
            "min_completed_transitions": 1,
            "require_all_required_transitions": True,
            "require_all_invariants_pass": True,
            "required_success_conditions": ["refund_approved"],
            "terminal_status": "success",
            "max_violation_count": 0,
            "expected_state": {"refund": {"status": "approved"}},
        },
        "metric_weights": {
            "world_contract_quality": 8.0,
            "world_contract_coverage": 3.0,
            "tool_selection_accuracy": 4.0,
            "task_completion": 1.0,
        },
    }


def build_manifest() -> dict[str, Any]:
    return optimize.build_target_optimization_manifest(
        name="sdk-target-optimization",
        required_env=[REQUIRED_ENV],
        base_config=_base_config(),
        evaluation_config=_evaluation_config(),
        target_candidates={
            "simulation.environments.0.data.transitions": [
                [],
                [_approve_transition()],
            ],
        },
        layers=["world", "environment", "evaluator"],
        target_metadata={
            "cookbook": "sdk-target-optimization",
            "optimized_surface": "world_contract_transition",
        },
    )


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)

    result = optimize.optimize_target(
        name="sdk-target-optimization",
        required_env=[REQUIRED_ENV],
        base_config=_base_config(),
        evaluation_config=_evaluation_config(),
        target_candidates={
            "simulation.environments.0.data.transitions": [
                [],
                [_approve_transition()],
            ],
        },
        layers=["world", "environment", "evaluator"],
        target_metadata={
            "cookbook": "sdk-target-optimization",
            "optimized_surface": "world_contract_transition",
        },
        manifest_path=Path(__file__).with_suffix(".json"),
    )
    if output_path is not None:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return result


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    payload = run(destination)
    if destination is None:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
