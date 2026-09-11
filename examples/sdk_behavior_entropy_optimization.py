from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, optimize


REQUIRED_ENV = "AGENT_LEARNING_SDK_BEHAVIOR_ENTROPY_KEY"


def weak_agent() -> dict[str, Any]:
    return {
        "type": "scripted",
        "name": "repetitive-tool-loop-agent",
        "responses": [
            {
                "content": "I will inspect the same policy source again.",
                "tool_calls": [
                    {
                        "id": "search_policy_1",
                        "name": "search_policy",
                        "arguments": {"query": "refund"},
                    }
                ],
            },
            {
                "content": "I will inspect the same policy source again.",
                "tool_calls": [
                    {
                        "id": "search_policy_2",
                        "name": "search_policy",
                        "arguments": {"query": "refund"},
                    }
                ],
            },
            {
                "content": "The refund decision is approved.",
                "tool_calls": [
                    {
                        "id": "search_policy_3",
                        "name": "search_policy",
                        "arguments": {"query": "refund"},
                    }
                ],
            },
        ],
    }


def balanced_agent() -> dict[str, Any]:
    return {
        "type": "scripted",
        "name": "balanced-behavior-entropy-agent",
        "responses": [
            {
                "content": "First I gather the current refund policy evidence.",
                "tool_calls": [
                    {
                        "id": "retrieve_policy",
                        "name": "retrieve_policy",
                        "arguments": {"query": "current refund policy"},
                    }
                ],
            },
            {
                "content": "Then I verify eligibility against the policy.",
                "tool_calls": [
                    {
                        "id": "check_eligibility",
                        "name": "check_eligibility",
                        "arguments": {"case_id": "refund_2026"},
                    }
                ],
            },
            {
                "content": (
                    "The refund decision is approved using distinct evidence "
                    "and decision tools."
                ),
                "tool_calls": [
                    {
                        "id": "apply_refund_decision",
                        "name": "apply_refund_decision",
                        "arguments": {"decision": "approved"},
                    }
                ],
            },
        ],
    }


def evaluation_config() -> dict[str, Any]:
    return {
        "task_description": (
            "Optimize a support-agent behavior pattern so it reaches the "
            "refund decision without looping on the same action."
        ),
        "expected_result": (
            "The refund decision is approved using distinct evidence and "
            "decision tools."
        ),
        "success_criteria": [
            "refund decision approved",
            "uses distinct evidence and decision tools",
        ],
        "required_tools": [
            "retrieve_policy",
            "check_eligibility",
            "apply_refund_decision",
        ],
        "available_tools": [
            "search_policy",
            "retrieve_policy",
            "check_eligibility",
            "apply_refund_decision",
        ],
        "behavior_entropy_quality": {
            "min_action_entropy": 0.35,
            "min_tool_entropy": 0.35,
            "max_repetition_rate": 0.40,
            "max_loop_rate": 0.20,
            "min_information_gain": 0.30,
            "min_exploration_efficiency": 0.55,
        },
        "metric_weights": {
            "behavior_entropy_quality": 8.0,
            "tool_selection_accuracy": 3.0,
            "task_completion": 1.0,
        },
    }


def build_manifest() -> dict[str, Any]:
    return optimize.build_task_optimization_manifest(
        name="sdk-behavior-entropy-optimization",
        required_env=[REQUIRED_ENV],
        agent_candidates=[weak_agent(), balanced_agent()],
        evaluation_config=evaluation_config(),
        threshold=0.9,
        min_turns=3,
        max_turns=3,
        target_metadata={
            "source": "examples/sdk_behavior_entropy_optimization.py",
            "task_kind": "behavior_entropy_optimization",
            "cookbook": "sdk-behavior-entropy-optimization",
            "research_sources": [
                {
                    "id": "2606.05872",
                    "source": "arxiv:2606.05872",
                    "url": "https://arxiv.org/abs/2606.05872",
                }
            ],
        },
    )


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)

    result = optimize.optimize_manifest(
        build_manifest(),
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
