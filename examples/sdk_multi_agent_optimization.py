from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, optimize


REQUIRED_ENV = "AGENT_LEARNING_SDK_MULTI_AGENT_EXAMPLE_KEY"


def participants() -> dict[str, dict[str, str]]:
    return {
        "planner": {"name": "planner", "role": "task planner"},
        "retriever": {"name": "retriever", "role": "policy evidence retriever"},
        "critic": {"name": "critic", "role": "grounding reviewer"},
    }


def weak_agent() -> dict[str, Any]:
    return {
        "type": "scripted",
        "responses": [
            {"content": "I will solve the refund case alone.", "tool_calls": []},
            {
                "content": "I skipped specialist handoff and review.",
                "tool_calls": [],
            },
            {"content": "Final answer without reconciliation.", "tool_calls": []},
        ],
    }


def strong_agent() -> dict[str, Any]:
    return {
        "type": "scripted",
        "responses": [
            {
                "content": "Inspecting multi-agent room state before routing.",
                "tool_calls": [
                    {
                        "id": "room_status_before",
                        "name": "room_status",
                        "arguments": {},
                    }
                ],
            },
            {
                "content": "Routing evidence collection and requesting grounded review.",
                "tool_calls": [
                    {
                        "id": "handoff_retriever",
                        "name": "handoff",
                        "arguments": {
                            "to": "retriever",
                            "task": (
                                "Collect the current refund policy evidence and "
                                "preserve citation context."
                            ),
                            "reason": "source grounding is required before final answer",
                            "context": {
                                "doc_id": "doc_refund_2026",
                                "world_state": "refund_case_open",
                            },
                        },
                    },
                    {
                        "id": "review_critic",
                        "name": "request_review",
                        "arguments": {
                            "reviewer": "critic",
                            "target": "refund policy answer and handoff evidence",
                            "criteria": [
                                "policy",
                                "handoff",
                                "source",
                            ],
                        },
                    },
                ],
            },
            {
                "content": (
                    "The optimized trace proves planner, retriever, and critic "
                    "roles coordinate through a verifiable room contract: "
                    "handoff was sent to retriever, critic review was "
                    "requested, the final decision was reconciled, and room "
                    "contract evidence was recorded for the approved refund "
                    "answer."
                ),
                "tool_calls": [
                    {
                        "id": "reconcile_answer",
                        "name": "reconcile",
                        "arguments": {
                            "summary": (
                                "approved refund answer reconciled across room "
                                "handoff and critic review"
                            ),
                            "decision": "ship grounded refund decision",
                            "accepted_source": "critic",
                            "conflicts": [],
                            "participants": ["planner", "retriever", "critic"],
                        },
                    },
                    {
                        "id": "room_status_after",
                        "name": "room_status",
                        "arguments": {},
                    },
                ],
            },
        ],
    }


def weak_room() -> dict[str, Any]:
    return {
        "participants": {
            "planner": participants()["planner"],
            "retriever": participants()["retriever"],
        },
        "allow_unknown_roles": True,
        "state": {"case": {"status": "triage"}},
    }


def strong_room() -> dict[str, Any]:
    return {
        "participants": participants(),
        "handoff_contracts": {
            "retriever": {
                "require_reason": True,
                "required_context_keys": ["doc_id", "world_state"],
                "required_task_terms": ["refund policy"],
                "forbidden_terms": ["guess"],
            }
        },
        "expected_handoffs": [
            {
                "to": "retriever",
                "task_contains": "current refund policy",
                "reason_contains": "source grounding",
                "context_keys": ["doc_id", "world_state"],
                "contract_matched": True,
            }
        ],
        "expected_reviews": [
            {
                "reviewer": "critic",
                "target_contains": "refund policy answer",
                "criteria": ["policy", "handoff", "source"],
            }
        ],
        "expected_reconciliation": {
            "summary_contains": "approved refund answer",
            "accepted_source": "critic",
            "conflicts_empty": True,
        },
        "allow_unknown_roles": False,
        "state": {"case": {"status": "resolved"}},
    }


def evaluation_config() -> dict[str, Any]:
    return {
        "task_description": (
            "Optimize a multi-agent coordination loop from weak solo execution "
            "to explicit handoff, review, and reconciliation."
        ),
        "expected_result": (
            "The optimized trace proves planner, retriever, and critic roles "
            "coordinate through a verifiable room contract."
        ),
        "required_tools": [
            "room_status",
            "handoff",
            "request_review",
            "reconcile",
        ],
        "available_tools": [
            "room_status",
            "handoff",
            "send_room_message",
            "request_review",
            "reconcile",
        ],
        "success_criteria": [
            "handoff sent to retriever",
            "critic review requested",
            "final decision reconciled",
            "room contract evidence recorded",
        ],
        "required_multi_agent_trace": [
            "trace",
            "role",
            "contract",
            "handoff",
            "review",
            "reconciliation",
            "state",
        ],
        "required_multi_agent_roles": [
            "planner",
            "retriever",
            "critic",
        ],
        "expected_multi_agent_handoffs": [
            {
                "to": "retriever",
                "task_contains": "current refund policy",
                "reason_contains": "source grounding",
                "context_keys": ["doc_id", "world_state"],
                "contract_matched": True,
            }
        ],
        "expected_multi_agent_reviews": [
            {
                "reviewer": "critic",
                "target_contains": "refund policy answer",
                "criteria": ["policy", "handoff", "source"],
            }
        ],
        "expected_multi_agent_reconciliation": {
            "summary_contains": "approved refund answer",
            "accepted_source": "critic",
            "conflicts_empty": True,
        },
        "metric_weights": {
            "multi_agent_coordination_quality": 8.0,
            "multi_agent_trace_coverage": 4.0,
            "tool_selection_accuracy": 3.0,
            "task_completion": 1.0,
        },
    }


def build_manifest() -> dict[str, Any]:
    return optimize.build_multi_agent_optimization_manifest(
        name="sdk-multi-agent-coordination-optimization",
        required_env=[REQUIRED_ENV],
        participants=participants(),
        agent_candidates=[weak_agent(), strong_agent()],
        room_candidates=[weak_room(), strong_room()],
        evaluation_config=evaluation_config(),
        threshold=0.95,
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
