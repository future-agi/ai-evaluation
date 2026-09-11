import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


TARGET = f"{Path(__file__).resolve()}:LocalFrameworkMemoryGraph"


@dataclass(frozen=True)
class MemoryOperation:
    operation: str
    key: str
    namespace: str
    value: str
    trace_id: str
    thread_id: str
    status: str = "allowed"
    policy_decision: str = "allowed"
    source_ids: list[str] | None = None


@dataclass(frozen=True)
class MemoryCheckpoint:
    checkpoint_id: str
    thread_id: str
    namespace: str
    state_keys: list[str]
    trace_id: str
    status: str = "saved"


@dataclass(frozen=True)
class MemoryStore:
    id: str
    type: str
    namespace: str
    tenant: str


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    key: str
    store: str
    namespace: str
    content: str
    source_ids: list[str]
    status: str = "active"


@dataclass(frozen=True)
class MemoryRetrieval:
    id: str
    query: str
    namespace: str
    thread_id: str
    documents: list[dict[str, Any]]
    doc_ids: list[str]
    freshness_checked: bool = True
    status: str = "returned"


@dataclass(frozen=True)
class MemoryTraceExport:
    content: str
    memory_operations: list[Any]
    checkpoints: list[Any]
    memory_stores: list[Any]
    memory_records: list[Any]
    memory_searches: list[Any]
    memory_policies: dict[str, Any]
    poison_tests: list[dict[str, Any]]
    isolation_tests: list[dict[str, Any]]
    retention_tests: list[dict[str, Any]]
    memory_observability: dict[str, Any]
    memory_artifacts: list[dict[str, Any]]


class LocalFrameworkMemoryGraph:
    """Local LangGraph/Mem0-style memory adapter export for discovery."""

    def run(self, text: str) -> str:
        assert text
        return "Weak memory response without checkpoint or memory lineage evidence."

    async def ainvoke(self, payload: dict[str, Any]) -> MemoryTraceExport:
        assert payload["metadata"]["framework"] == "langgraph"
        namespace = "tenant_refunds"
        thread_id = payload["thread_id"]
        return MemoryTraceExport(
            content=(
                "Framework memory trace adapter approved refund with current "
                "policy recall and governed memory lineage."
            ),
            memory_operations=[
                MemoryOperation(
                    operation="write",
                    key="refund_policy_memory",
                    namespace=namespace,
                    value="Customer is eligible under the current 30 day refund policy.",
                    trace_id="mem-write-1",
                    thread_id=thread_id,
                    source_ids=["refund_policy_doc"],
                ),
                MemoryOperation(
                    operation="read",
                    key="refund_policy_memory",
                    namespace=namespace,
                    value="Recall current refund eligibility.",
                    trace_id="mem-read-1",
                    thread_id=thread_id,
                    source_ids=["refund_policy_doc"],
                ),
                MemoryOperation(
                    operation="recall",
                    key="refund_policy_memory",
                    namespace=namespace,
                    value="Current refund policy approves this request.",
                    trace_id="mem-recall-1",
                    thread_id=thread_id,
                    source_ids=["refund_policy_doc"],
                ),
                MemoryOperation(
                    operation="update",
                    key="refund_policy_memory",
                    namespace=namespace,
                    value="Audit metadata attached after policy recall.",
                    trace_id="mem-update-1",
                    thread_id=thread_id,
                    source_ids=["refund_policy_doc"],
                ),
            ],
            checkpoints=[
                MemoryCheckpoint(
                    checkpoint_id="refund-thread-checkpoint-1",
                    thread_id=thread_id,
                    namespace=namespace,
                    state_keys=["messages", "refund_policy_memory", "tool_results"],
                    trace_id="checkpoint-1",
                )
            ],
            memory_stores=[
                MemoryStore(
                    id="langgraph_store",
                    type="long_term_store",
                    namespace=namespace,
                    tenant=namespace,
                )
            ],
            memory_records=[
                MemoryRecord(
                    id="refund_policy_memory",
                    key="refund_policy_memory",
                    store="langgraph_store",
                    namespace=namespace,
                    content="Customer is eligible under the current 30 day refund policy.",
                    source_ids=["refund_policy_doc"],
                )
            ],
            memory_searches=[
                MemoryRetrieval(
                    id="refund-policy-search",
                    query="current refund eligibility",
                    namespace=namespace,
                    thread_id=thread_id,
                    documents=[
                        {
                            "id": "refund_policy_doc",
                            "title": "Current refund policy",
                            "content": "Refunds are approved within 30 days with receipt.",
                            "current": True,
                        }
                    ],
                    doc_ids=["refund_policy_doc"],
                )
            ],
            memory_policies={
                "tenant_isolation": {"status": "enforced"},
                "audit": {"status": "enforced"},
                "retention": {"status": "enforced"},
                "deletion": {"status": "enforced"},
                "redaction": {"status": "enforced"},
                "canary": {"status": "blocked"},
            },
            poison_tests=[{"id": "refund_memory_canary", "status": "blocked"}],
            isolation_tests=[{"id": "tenant_namespace_boundary", "status": "passed"}],
            retention_tests=[{"id": "retention_delete_after_ttl", "status": "passed"}],
            memory_observability={"traces": ["mem-write-1", "mem-read-1", "mem-recall-1"]},
            memory_artifacts=[{"id": "memory-audit-log", "type": "json"}],
        )


def build_manifest() -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_local_adapter(
        name="sdk-framework-adapter-memory-trace-run",
        framework="langgraph",
        target=TARGET,
        method_candidates=["run", "ainvoke"],
        input_mode_candidates=["text", "dict"],
        discovery_max_candidates=6,
        cases=[
            {
                "id": "framework-memory-refund",
                "input": "Approve the refund through governed memory recall.",
                "expected_contains": ["approved refund"],
                "required_events": [
                    "framework_memory_operation",
                    "framework_memory_checkpoint",
                    "framework_memory_retrieval",
                    "framework_memory_record",
                ],
                "required_state_keys": [
                    "framework_runtime",
                    "framework_memory",
                    "retrieval_memory",
                    "agent_memory_lineage",
                ],
            }
        ],
        auto_evaluation_config=True,
        metadata={"cookbook": "sdk-framework-adapter-memory-trace"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    manifest = build_manifest()
    simulate.write_manifest_file(manifest, manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
    result["framework_adapter_memory_trace_manifest"] = manifest

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    destination = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts") / "sdk-framework-adapter-memory-trace.json"
    )
    run(destination)
