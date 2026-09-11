from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, optimize


REQUIRED_ENV = "AGENT_LEARNING_SDK_MEMORY_EXAMPLE_KEY"


def weak_candidate() -> dict[str, Any]:
    return {
        "retrieval_memory": {
            "documents": [
                {
                    "id": "doc_refund_2025",
                    "title": "Archived refund policy",
                    "content": "Archived refund guidance requires manual review.",
                    "current": False,
                }
            ],
            "memory": {"prior_case": "manual_review"},
            "require_current": True,
        },
        "agent_memory_lineage": {
            "name": "weak-memory-lineage",
            "target": {"agent": "refund-agent"},
            "stores": [],
            "memories": [],
            "operations": [],
            "lineage": [],
            "policies": {},
            "observability": {},
            "artifacts": [],
        },
    }


def strong_candidate() -> dict[str, Any]:
    return {
        "retrieval_memory": {
            "documents": [
                {
                    "id": "doc_refund_2026",
                    "title": "Current refund policy",
                    "content": (
                        "The current refund policy allows approved refunds "
                        "when source grounding and memory provenance are recorded. "
                        "Retention, deletion, and redaction policies must be enforced."
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
            "stores": [
                {"id": "episodic", "type": "vector", "tenant": "tenant_a"}
            ],
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
                    "trace_id": "memory_trace_1",
                    "attribution": {"source": "doc_refund_2026"},
                },
                {
                    "id": "write_policy_memory",
                    "operation": "write",
                    "store": "episodic",
                    "memory_id": "refund_decision",
                    "status": "allowed",
                    "policy_decision": "allowed",
                    "trace_id": "memory_trace_2",
                    "attribution": {"source": "doc_refund_2026"},
                },
                {
                    "id": "recall_policy_memory",
                    "operation": "recall",
                    "store": "episodic",
                    "memory_id": "refund_decision",
                    "status": "allowed",
                    "policy_decision": "allowed",
                    "trace_id": "memory_trace_3",
                    "attribution": {"source": "doc_refund_2026"},
                },
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
                    "canary": "FA_MEMORY_CANARY",
                }
            ],
            "isolation_tests": [
                {"id": "tenant_a_boundary", "status": "passed"}
            ],
            "retention_tests": [
                {"id": "delete_after_retention", "status": "passed"}
            ],
            "observability": {
                "traces": ["memory_trace_1"],
                "logs": ["memory_audit.jsonl"],
            },
            "artifacts": [
                {
                    "id": "memory-audit",
                    "type": "json",
                    "path": "artifacts/memory-audit.json",
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
    }


def evaluation_config() -> dict[str, Any]:
    return {
        "task_description": (
            "Optimize retrieval freshness and memory lineage for a refund "
            "decision."
        ),
        "expected_result": (
            "The optimized memory harness records current refund policy "
            "grounding, source attribution, memory provenance, and enforced "
            "retention/deletion/redaction policies."
        ),
        "required_tools": [
            "retrieve_documents",
            "read_document",
            "cite_sources",
            "write_memory",
            "retrieval_memory_status",
            "agent_memory_lineage_status",
            "list_memory_lineage_operations",
        ],
        "success_criteria": [
            "current refund policy grounding",
            "source attribution",
            "memory provenance",
            "retention and deletion policies enforced",
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
        "allow_extra_tool_arguments": True,
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
        "metric_weights": {
            "retrieval_memory_attribution": 4.0,
            "retrieval_context_quality": 6.0,
            "source_grounding": 3.0,
            "agent_memory_lineage_coverage": 5.0,
            "agent_memory_lineage_quality": 8.0,
            "memory_integrity": 2.0,
            "tool_selection_accuracy": 2.0,
            "task_completion": 2.0,
        },
    }


def build_manifest() -> dict[str, Any]:
    return optimize.build_memory_optimization_manifest(
        name="sdk-memory-optimization",
        required_env=[REQUIRED_ENV],
        memory_candidates=[weak_candidate(), strong_candidate()],
        evaluation_config=evaluation_config(),
        threshold=0.9,
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
