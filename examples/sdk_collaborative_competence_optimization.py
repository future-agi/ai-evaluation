from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, optimize


REQUIRED_ENV = "AGENT_LEARNING_SDK_COLLABORATIVE_COMPETENCE_KEY"


def participants() -> dict[str, dict[str, str]]:
    return {
        "planner": {"name": "planner", "role": "world-state planner"},
        "retriever": {"name": "retriever", "role": "current evidence retriever"},
        "critic": {"name": "critic", "role": "misalignment and risk reviewer"},
    }


def weak_agent() -> dict[str, Any]:
    return {
        "name": "solo-no-common-ground-agent",
        "type": "scripted",
        "responses": [
            {"content": "I will approve the refund by myself.", "tool_calls": []},
            {
                "content": "No need to model partner intent or update shared state.",
                "tool_calls": [],
            },
            {
                "content": "Refund approved without review or repair.",
                "tool_calls": [],
            },
        ],
    }


def collaborative_agent() -> dict[str, Any]:
    return {
        "name": "collaborative-competence-agent",
        "type": "scripted",
        "responses": [
            {
                "content": (
                    "I am checking the room before routing so planner, "
                    "retriever, and critic start from the same refund goal."
                ),
                "tool_calls": [
                    {
                        "id": "room_status_before",
                        "name": "room_status",
                        "arguments": {},
                    }
                ],
            },
            {
                "content": (
                    "Planner predicts retriever will ground the current "
                    "policy while critic will catch stale-source risk."
                ),
                "tool_calls": [
                    {
                        "id": "handoff_retriever",
                        "name": "handoff",
                        "arguments": {
                            "to": "retriever",
                            "task": (
                                "Collect the current 2026 refund policy "
                                "evidence and preserve citation context."
                            ),
                            "reason": (
                                "shared task state says approval requires "
                                "current source evidence"
                            ),
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
                            "target": "shared refund task state and policy evidence",
                            "criteria": [
                                "common_ground",
                                "partner_intent",
                                "source_freshness",
                                "repair",
                            ],
                        },
                    },
                ],
            },
            {
                "content": (
                    "Collaborative competence trace passed: common ground, "
                    "shared task state, mental models, partner-intent "
                    "predictions, critic repair, value diversity, and final "
                    "reconciliation were all recorded for refund approval."
                ),
                "tool_calls": [
                    {
                        "id": "reconcile_answer",
                        "name": "reconcile",
                        "arguments": {
                            "summary": (
                                "approved refund answer after critic repaired "
                                "stale-source risk"
                            ),
                            "decision": "approve refund using doc_refund_2026",
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
        "messages": [
            {"speaker": "planner", "content": "I can finish without review."}
        ],
        "allow_unknown_roles": True,
        "state": {"case": {"status": "triage"}},
    }


def collaborative_room() -> dict[str, Any]:
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
        "messages": [
            {
                "speaker": "planner",
                "content": "Common ground: approval requires current policy evidence.",
            },
            {
                "speaker": "retriever",
                "content": "I will ground the answer in doc_refund_2026.",
            },
            {
                "speaker": "critic",
                "content": "I will repair stale-source and unsupported-claim risk.",
            },
        ],
        "common_ground": [
            {
                "turn": 1,
                "roles": ["planner", "retriever", "critic"],
                "claim": (
                    "refund approval requires current policy evidence, "
                    "critic review, and explicit reconciliation"
                ),
            }
        ],
        "shared_task_state": {
            "goal": "approve eligible refund",
            "policy_doc": "doc_refund_2026",
            "world_state": "refund_case_open",
            "status": "aligned",
            "open_conflicts": [],
        },
        "mental_models": [
            {
                "role": "planner",
                "self_reasoning": "decompose world state, source evidence, and review",
                "perceived_partner_intent": {
                    "retriever": "ground current policy",
                    "critic": "catch stale-source risk",
                },
                "perceived_team_goal": "approved refund with evidence",
            },
            {
                "role": "retriever",
                "self_reasoning": "retrieve and cite the current policy document",
                "perceived_partner_intent": {
                    "planner": "maintain task state",
                    "critic": "verify source freshness",
                },
                "perceived_team_goal": "source-grounded refund decision",
            },
            {
                "role": "critic",
                "self_reasoning": "find unsupported or stale policy claims",
                "perceived_partner_intent": {
                    "planner": "synthesize final answer",
                    "retriever": "supply evidence",
                },
                "perceived_team_goal": "safe reconciled approval",
            },
        ],
        "intent_predictions": [
            {
                "observer": "planner",
                "partner": "retriever",
                "intent": "ground current policy in doc_refund_2026",
                "validated": True,
            },
            {
                "observer": "planner",
                "partner": "critic",
                "intent": "repair stale-source risk before final approval",
                "validated": True,
            },
        ],
        "repair_moves": [
            {
                "actor": "critic",
                "misalignment": "planner could approve from stale policy memory",
                "repair": "require doc_refund_2026 citation before approval",
                "accepted_by": ["planner", "retriever"],
                "shared_state_update": "policy_doc=doc_refund_2026",
            }
        ],
        "handoffs": [
            {
                "from": "planner",
                "to": "retriever",
                "task": (
                    "Collect the current 2026 refund policy evidence and "
                    "preserve citation context."
                ),
                "reason": "shared task state says approval requires current source evidence",
                "context": {
                    "doc_id": "doc_refund_2026",
                    "world_state": "refund_case_open",
                },
                "contract_status": {"matched": True},
            }
        ],
        "reviews": [
            {
                "reviewer": "critic",
                "target": "shared refund task state and policy evidence",
                "criteria": [
                    "common_ground",
                    "partner_intent",
                    "source_freshness",
                    "repair",
                ],
                "finding": "stale-source risk repaired by requiring doc_refund_2026",
            }
        ],
        "reconciliations": [
            {
                "summary": (
                    "approved refund answer after critic repaired stale-source risk"
                ),
                "decision": "approve refund using doc_refund_2026",
                "accepted_source": "critic",
                "conflicts": [],
                "participants": ["planner", "retriever", "critic"],
            }
        ],
        "value_diversity": {
            "roles": ["planner", "retriever", "critic"],
            "perspectives": ["world_state", "source_grounding", "risk_review"],
            "homogenized": False,
        },
        "expected_handoffs": [
            {
                "to": "retriever",
                "task_contains": "current 2026 refund policy",
                "reason_contains": "shared task state",
                "context_keys": ["doc_id", "world_state"],
                "contract_matched": True,
            }
        ],
        "expected_reviews": [
            {
                "reviewer": "critic",
                "target_contains": "shared refund task state",
                "criteria": [
                    "common_ground",
                    "partner_intent",
                    "source_freshness",
                    "repair",
                ],
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
            "Optimize a multi-agent refund workflow for collaborative "
            "competence, not only task completion."
        ),
        "expected_result": (
            "Collaborative competence trace passed: common ground, shared "
            "task state, mental models, partner-intent predictions, critic "
            "repair, value diversity, and final reconciliation were all "
            "recorded for refund approval."
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
            "common ground established",
            "shared task state maintained",
            "mental models and partner intent recorded",
            "critic repair accepted",
            "value diversity preserved",
            "final decision reconciled",
        ],
        "required_multi_agent_trace": [
            "trace",
            "role",
            "contract",
            "handoff",
            "review",
            "reconciliation",
            "state",
            "common_ground",
            "shared_task_state",
            "mental_model",
            "partner_intent",
            "repair",
            "value_diversity",
        ],
        "required_multi_agent_roles": [
            "planner",
            "retriever",
            "critic",
        ],
        "expected_multi_agent_handoffs": [
            {
                "to": "retriever",
                "task_contains": "current 2026 refund policy",
                "reason_contains": "shared task state",
                "context_keys": ["doc_id", "world_state"],
                "contract_matched": True,
            }
        ],
        "expected_multi_agent_reviews": [
            {
                "reviewer": "critic",
                "target_contains": "shared refund task state",
                "criteria": [
                    "common_ground",
                    "partner_intent",
                    "source_freshness",
                    "repair",
                ],
            }
        ],
        "expected_multi_agent_reconciliation": {
            "summary_contains": "approved refund answer",
            "accepted_source": "critic",
            "conflicts_empty": True,
        },
        "collaborative_competence_quality": {
            "required_roles": ["planner", "retriever", "critic"],
            "min_common_ground_updates": 1,
            "min_mental_model_updates": 3,
            "min_intent_predictions": 2,
            "min_repair_moves": 1,
            "min_participation_roles": 3,
            "require_shared_task_state": True,
            "require_protocol_trace": True,
            "require_handoff": True,
            "require_review": True,
            "require_reconciliation": True,
            "require_balanced_participation": True,
            "require_value_diversity": True,
        },
        "metric_weights": {
            "collaborative_competence_quality": 10.0,
            "multi_agent_coordination_quality": 5.0,
            "multi_agent_trace_coverage": 3.0,
            "tool_selection_accuracy": 2.0,
            "task_completion": 1.0,
        },
    }


def build_manifest() -> dict[str, Any]:
    return optimize.build_multi_agent_optimization_manifest(
        name="sdk-collaborative-competence-optimization",
        required_env=[REQUIRED_ENV],
        participants=participants(),
        agent_candidates=[weak_agent(), collaborative_agent()],
        room_candidates=[weak_room(), collaborative_room()],
        evaluation_config=evaluation_config(),
        threshold=0.95,
        target_metadata={
            "source": "examples/sdk_collaborative_competence_optimization.py",
            "task_kind": "collaborative_competence_optimization",
            "research_sources": [
                {
                    "id": "2606.06399",
                    "source": "arxiv:2606.06399",
                    "url": "https://arxiv.org/abs/2606.06399",
                },
                {
                    "id": "2606.06388",
                    "source": "arxiv:2606.06388",
                    "url": "https://arxiv.org/abs/2606.06388",
                },
                {
                    "id": "2606.05985",
                    "source": "arxiv:2606.05985",
                    "url": "https://arxiv.org/abs/2606.05985",
                },
                {
                    "id": "2606.05670",
                    "source": "arxiv:2606.05670",
                    "url": "https://arxiv.org/abs/2606.05670",
                },
            ],
            "original_synthesis": (
                "Collaborative competence should be optimized from explicit "
                "process evidence: common ground, shared task state, mental "
                "models, partner intent, repair, participation, diversity, "
                "and protocol trace logging."
            ),
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
