from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, simulate


REQUIRED_ENV = "AGENT_LEARNING_SDK_TASK_SIMULATION_KEY"


def build_manifest() -> dict[str, Any]:
    transition = {
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
    return simulate.build_task_run_manifest(
        name="sdk-task-simulation",
        required_env=[REQUIRED_ENV],
        task_description=(
            "Approve the refund by applying the world transition and produce "
            "a complete final state."
        ),
        expected_result=(
            "The refund world transition is applied and the final state is "
            "approved and complete."
        ),
        agent={
            "type": "scripted",
            "responses": [
                {
                    "content": (
                        "First, because I approve the refund by applying the "
                        "refund world transition, I produce a complete final "
                        "state; the transition is applied, approved, and complete."
                    ),
                    "tool_calls": [
                        {
                            "id": "approve_refund",
                            "name": "apply_world_transition",
                            "arguments": {"id": "approve_refund"},
                        }
                    ],
                },
                {
                    "content": (
                        "Next, since I approve the refund by applying the "
                        "refund world transition, I produce a complete final "
                        "state; the transition is applied, approved, and complete."
                    ),
                },
                {
                    "content": (
                        "Finally, therefore I approve the refund by applying "
                        "the refund world transition and produce a complete "
                        "final state; the transition is applied and approved."
                    ),
                },
            ],
        },
        environments=[
            {
                "type": "world_contract",
                "data": {
                    "name": "sdk-task-simulation-refund-world",
                    "actors": ["agent", "customer"],
                    "resources": ["refund"],
                    "initial_state": {"refund": {"status": "pending"}},
                    "transitions": [transition],
                    "success_conditions": [
                        {
                            "id": "refund_approved",
                            "must": {"refund.status": "approved"},
                        }
                    ],
                },
            }
        ],
        required_tools=["apply_world_transition"],
        available_tools=["apply_world_transition", "world_contract_status"],
        success_criteria=[
            "refund world transition is applied",
            "final state is approved and complete",
        ],
        evaluation_config={
            "required_world_contract": [
                "world_contract",
                "transition",
                "success_condition",
                "refund",
            ],
            "world_contract_quality": {
                "required_transitions": ["approve_refund"],
                "min_completed_transitions": 1,
                "require_all_required_transitions": True,
                "required_success_conditions": ["refund_approved"],
                "terminal_status": "success",
                "expected_state": {"refund": {"status": "approved"}},
            },
        },
        threshold=0.85,
        min_turns=3,
        max_turns=3,
    )


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)

    manifest_path = Path(output_path).with_suffix(".manifest.json") if output_path else (
        Path(__file__).with_suffix(".json")
    )
    simulate.write_manifest_file(build_manifest(), manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
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
