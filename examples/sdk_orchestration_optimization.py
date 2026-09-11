from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, optimize


REQUIRED_ENV = "AGENT_LEARNING_SDK_ORCHESTRATION_EXAMPLE_KEY"


def weak_agent() -> dict[str, Any]:
    return {
        "type": "scripted",
        "responses": [
            {
                "content": (
                    "I inspected the refund request but did not apply the "
                    "world transition or collect orchestration evidence."
                ),
                "tool_calls": [],
            }
        ],
    }


def strong_agent() -> dict[str, Any]:
    return {
        "type": "scripted",
        "responses": [
            {
                "content": (
                    "First, because I optimize refund orchestration stack across "
                    "world framework retrieval memory lineage multi agent review "
                    "evidence: optimized stack approves refund, records trace, "
                    "current policy grounding, provenance, critic-reviewed "
                    "reconciliation."
                ),
                "tool_calls": [
                    {
                        "id": "approve_refund",
                        "name": "apply_world_transition",
                        "arguments": {"id": "approve_refund"},
                    },
                    {
                        "id": "framework_status",
                        "name": "framework_trace_status",
                        "arguments": {},
                    },
                ],
            },
            {
                "content": (
                    "Next, since I optimize refund orchestration stack across "
                    "world framework retrieval memory lineage multi agent review "
                    "evidence: optimized stack approves refund, records trace, "
                    "current policy grounding, provenance, critic-reviewed "
                    "reconciliation."
                ),
                "tool_calls": [
                    {
                        "id": "retrieve_policy",
                        "name": "retrieve_documents",
                        "arguments": {"query": "current refund policy"},
                    },
                    {
                        "id": "read_policy",
                        "name": "read_document",
                        "arguments": {"id": "doc_refund_2026"},
                    },
                    {
                        "id": "cite_policy",
                        "name": "cite_sources",
                        "arguments": {
                            "doc_ids": ["doc_refund_2026"],
                            "claim": (
                                "The current refund policy allows approved "
                                "refunds when framework trace, source "
                                "grounding, memory provenance, and critic "
                                "review are recorded."
                            ),
                            "freshness_checked": True,
                        },
                    },
                    {
                        "id": "memory_lineage",
                        "name": "agent_memory_lineage_status",
                        "arguments": {},
                    },
                    {
                        "id": "retrieval_memory",
                        "name": "retrieval_memory_status",
                        "arguments": {},
                    },
                ],
            },
            {
                "content": (
                    "Finally, therefore I optimize refund orchestration stack "
                    "across world framework retrieval memory lineage multi agent "
                    "review evidence: optimized stack approves refund, records "
                    "trace, current policy grounding, provenance, critic-reviewed "
                    "reconciliation. The current refund policy allows approved "
                    "refunds when framework trace, source grounding, memory "
                    "provenance, and critic review are recorded."
                ),
                "tool_calls": [
                    {
                        "id": "room_status",
                        "name": "room_status",
                        "arguments": {},
                    },
                    {
                        "id": "critic_review",
                        "name": "request_review",
                        "arguments": {
                            "reviewer": "critic",
                            "target": "refund orchestration decision",
                            "criteria": ["policy", "memory", "world"],
                        },
                    },
                    {
                        "id": "reconcile",
                        "name": "reconcile",
                        "arguments": {
                            "summary": "approved refund orchestration accepted",
                            "accepted_source": "critic",
                            "conflicts": [],
                            "participants": ["planner", "retriever", "critic"],
                        },
                    },
                ],
            },
        ],
    }


def weak_stack() -> dict[str, Any]:
    return {
        "name": "weak-orchestration-stack",
        "world_contract": {
            "name": "refund-world",
            "actors": ["agent", "customer"],
            "resources": ["refund"],
            "initial_state": {"refund": {"status": "pending"}},
            "transitions": [],
            "success_conditions": [
                {"id": "refund_approved", "must": {"refund.status": "approved"}}
            ],
        },
        "framework_trace": {
            "framework": "langgraph",
            "spans": [],
            "adapter_required_signals": ["planner", "tool", "policy"],
        },
        "retrieval_memory": {
            "documents": [
                {
                    "id": "doc_refund_2025",
                    "title": "Archived refund policy",
                    "content": "Archived policy requires manual review.",
                    "current": False,
                }
            ],
            "require_current": True,
        },
        "agent_memory_lineage": {
            "name": "weak-lineage",
            "target": {"agent": "refund-agent"},
            "stores": [],
            "memories": [],
            "operations": [],
            "lineage": [],
        },
        "multi_agent_room": {
            "participants": {"planner": {"name": "planner", "role": "planner"}},
            "allow_unknown_roles": True,
            "state": {"case": {"status": "triage"}},
        },
    }


def strong_stack() -> dict[str, Any]:
    return {
        "name": "strong-orchestration-stack",
        "world_contract": {
            "name": "refund-world",
            "actors": ["agent", "customer"],
            "resources": ["refund"],
            "initial_state": {"refund": {"status": "pending"}},
            "transitions": [
                {
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
            ],
            "success_conditions": [
                {"id": "refund_approved", "must": {"refund.status": "approved"}}
            ],
        },
        "framework_trace": {
            "framework": "langgraph",
            "spans": [
                {
                    "id": "planner",
                    "name": "planner.invoke",
                    "input": "refund workflow",
                    "output": "approved",
                    "tool_calls": [{"name": "framework_trace_status"}],
                    "signals": ["planner", "tool", "policy"],
                }
            ],
            "adapter_required_signals": ["planner", "tool", "policy"],
            "adapter_required_mappings": {"tool": ["tool_name"]},
        },
        "retrieval_memory": {
            "documents": [
                {
                    "id": "doc_refund_2026",
                    "title": "Current refund policy",
                    "content": (
                        "The current refund policy allows approved refunds "
                        "when framework trace, source grounding, memory "
                        "provenance, and critic review are recorded."
                    ),
                    "current": True,
                }
            ],
            "memory": {"prior_case": "manual_review"},
            "require_current": True,
        },
        "agent_memory_lineage": {
            "name": "refund-memory-lineage",
            "target": {"agent": "refund-agent", "tenant": "tenant_a"},
            "stores": [{"id": "episodic", "type": "vector", "tenant": "tenant_a"}],
            "memories": [
                {
                    "id": "refund_decision",
                    "store": "episodic",
                    "status": "active",
                    "source_ids": ["doc_refund_2026"],
                    "tenant": "tenant_a",
                }
            ],
            "operations": [
                {
                    "id": "read_policy_memory",
                    "operation": "read",
                    "store": "episodic",
                    "memory_id": "refund_decision",
                    "status": "allowed",
                    "policy_decision": "allowed",
                    "trace_id": "memory_trace_read",
                    "attribution": {"source": "doc_refund_2026"},
                },
                {
                    "id": "write_policy_memory",
                    "operation": "write",
                    "store": "episodic",
                    "memory_id": "refund_decision",
                    "status": "allowed",
                    "policy_decision": "allowed",
                    "trace_id": "memory_trace_write",
                    "attribution": {"source": "doc_refund_2026"},
                },
                {
                    "id": "recall_policy_memory",
                    "operation": "recall",
                    "store": "episodic",
                    "memory_id": "refund_decision",
                    "status": "allowed",
                    "policy_decision": "allowed",
                    "trace_id": "memory_trace_recall",
                    "attribution": {"source": "doc_refund_2026"},
                }
            ],
            "lineage": [
                {
                    "from": "doc_refund_2026",
                    "to": "refund_decision",
                    "type": "source_attribution",
                }
            ],
            "policies": {
                "retention": {"status": "enforced"},
                "deletion": {"status": "enforced"},
                "redaction": {"status": "enforced"},
                "tenant_isolation": {"status": "enforced"},
                "audit": {"status": "enforced"},
            },
            "poison_tests": [
                {
                    "id": "canary_poison",
                    "status": "blocked",
                    "canary": "FA_ORCHESTRATION_CANARY",
                }
            ],
            "isolation_tests": [{"id": "tenant_a_boundary", "status": "passed"}],
            "retention_tests": [{"id": "delete_after_retention", "status": "passed"}],
            "observability": {
                "traces": ["memory_trace_read", "memory_trace_write"],
                "logs": ["orchestration-memory-audit.jsonl"],
            },
            "artifacts": [
                {
                    "id": "orchestration-memory-audit",
                    "type": "json",
                    "path": "artifacts/orchestration-memory-audit.json",
                }
            ],
            "required_evidence": [
                "source_attribution",
                "tenant_isolation",
                "audit",
                "retention_policy",
                "deletion_policy",
                "redaction",
                "canary",
            ],
            "required_signals": [
                "memory_lineage",
                "source_attribution",
                "tenant_isolation",
                "audit",
            ],
        },
        "multi_agent_room": {
            "participants": {
                "planner": {"name": "planner", "role": "planner"},
                "retriever": {"name": "retriever", "role": "retriever"},
                "critic": {"name": "critic", "role": "critic"},
            },
            "expected_reviews": [
                {
                    "reviewer": "critic",
                    "target_contains": "refund orchestration",
                    "criteria": ["policy", "memory", "world"],
                }
            ],
            "expected_reconciliation": {
                "summary_contains": "approved refund orchestration",
                "accepted_source": "critic",
                "conflicts_empty": True,
            },
            "allow_unknown_roles": False,
            "state": {"case": {"status": "resolved"}},
        },
    }


def evaluation_config() -> dict[str, Any]:
    return {
        "task_description": (
            "Optimize a refund orchestration stack across world, framework, "
            "retrieval, memory lineage, and multi-agent review evidence."
        ),
        "expected_result": (
            "The optimized orchestration stack approves refund, records "
            "framework trace, current policy grounding, memory provenance, "
            "and critic-reviewed reconciliation."
        ),
        "required_tools": [
            "apply_world_transition",
            "framework_trace_status",
            "retrieve_documents",
            "read_document",
            "cite_sources",
            "agent_memory_lineage_status",
            "retrieval_memory_status",
            "room_status",
            "request_review",
            "reconcile",
        ],
        "available_tools": [
            "apply_world_transition",
            "framework_trace_status",
            "retrieve_documents",
            "read_document",
            "cite_sources",
            "agent_memory_lineage_status",
            "retrieval_memory_status",
            "room_status",
            "request_review",
            "reconcile",
        ],
        "success_criteria": [
            "approves refund",
            "framework trace",
            "current policy grounding",
            "memory provenance",
            "critic-reviewed reconciliation",
        ],
        "required_world_contract": ["world_contract", "transition", "refund"],
        "world_contract_quality": {
            "required_transitions": ["approve_refund"],
            "min_completed_transitions": 1,
            "require_all_required_transitions": True,
            "required_success_conditions": ["refund_approved"],
            "terminal_status": "success",
            "expected_state": {"refund": {"status": "approved"}},
        },
        "required_framework_trace": [
            "framework_trace",
            "langgraph",
            "planner",
            "tool",
            "policy",
            "framework_trace_status",
        ],
        "required_retrieval_memory_trace": [
            "trace",
            "query",
            "document_read",
            "attribution",
            "retrieve_documents",
            "cite_sources",
        ],
        "expected_retrieval_doc_ids": ["doc_refund_2026"],
        "forbidden_retrieval_doc_ids": ["doc_refund_2025"],
        "require_current_retrieval": True,
        "require_source_grounding": True,
        "source_grounding_min_overlap": 0.2,
        "required_agent_memory_lineage": [
            "agent_memory_lineage",
            "source_attribution",
            "tenant_isolation",
            "audit",
            "retention_policy",
            "deletion_policy",
            "redaction",
            "canary",
        ],
        "agent_memory_lineage_quality": {
            "min_store_count": 1,
            "min_memory_count": 1,
            "min_operation_count": 3,
            "min_read_operations": 1,
            "min_write_operations": 1,
            "min_recall_operations": 1,
            "min_observability_hooks": 1,
            "min_artifact_count": 1,
            "max_unattributed_memories": 0,
            "max_open_poisoning": 0,
            "max_isolation_violations": 0,
            "max_retention_violations": 0,
            "max_policy_violations": 0,
            "require_target": True,
            "require_stores": True,
            "require_memory_records": True,
            "require_operations": True,
            "require_lineage": True,
            "require_source_attribution": True,
            "require_tenant_isolation": True,
            "require_audit": True,
            "require_retention_policy": True,
            "require_deletion_policy": True,
            "require_redaction": True,
            "require_canaries": True,
            "require_observability": True,
            "require_artifacts": True,
            "required_operation_types": ["read", "write", "recall"],
            "required_policies": [
                "retention",
                "deletion",
                "redaction",
                "tenant_isolation",
            ],
        },
        "required_multi_agent_trace": [
            "trace",
            "role",
            "review_requested",
            "reconciled",
        ],
        "required_multi_agent_roles": ["planner", "retriever", "critic"],
        "expected_multi_agent_reviews": [
            {
                "reviewer": "critic",
                "target_contains": "refund orchestration",
                "criteria": ["policy", "memory", "world"],
            }
        ],
        "expected_multi_agent_reconciliation": {
            "summary_contains": "approved refund orchestration",
            "accepted_source": "critic",
            "conflicts_empty": True,
        },
        "metric_weights": {
            "world_contract_quality": 8.0,
            "world_contract_coverage": 3.0,
            "framework_trace_coverage": 3.0,
            "retrieval_context_quality": 4.0,
            "retrieval_memory_attribution": 4.0,
            "source_grounding": 3.0,
            "agent_memory_lineage_coverage": 5.0,
            "agent_memory_lineage_quality": 8.0,
            "memory_integrity": 2.0,
            "multi_agent_trace_coverage": 4.0,
            "multi_agent_coordination_quality": 7.0,
            "tool_selection_accuracy": 4.0,
            "task_completion": 2.0,
            "goal_progress": 1.0,
        },
    }


def build_manifest() -> dict[str, Any]:
    return optimize.build_orchestration_optimization_manifest(
        name="sdk-orchestration-optimization",
        required_env=[REQUIRED_ENV],
        agent_candidates=[weak_agent(), strong_agent()],
        stack_candidates=[weak_stack(), strong_stack()],
        evaluation_config=evaluation_config(),
        threshold=0.9,
        min_turns=3,
        max_turns=3,
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
