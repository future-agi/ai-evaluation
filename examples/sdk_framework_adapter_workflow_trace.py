import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


TARGET = f"{Path(__file__).resolve()}:LocalLangGraphWorkflowAgent"


@dataclass(frozen=True)
class WorkflowNode:
    id: str
    name: str
    type: str
    input_keys: list[str]
    output_keys: list[str]


@dataclass(frozen=True)
class WorkflowEdge:
    source: str
    target: str
    condition: str = ""


@dataclass(frozen=True)
class WorkflowStep:
    id: str
    name: str
    node: str
    status: str
    superstep: int
    input: dict[str, Any]
    output: dict[str, Any]
    state_delta: dict[str, Any]
    tool_calls: list[dict[str, Any]]


@dataclass(frozen=True)
class WorkflowCheckpoint:
    checkpoint_id: str
    thread_id: str
    checkpoint_ns: str
    superstep: int
    state: dict[str, Any]
    pending_writes: list[dict[str, Any]]
    next_nodes: list[str]


@dataclass(frozen=True)
class WorkflowRoute:
    source: str
    target: str
    condition: str
    selected: str
    reason: str


@dataclass(frozen=True)
class WorkflowInterrupt:
    id: str
    node: str
    reason: str
    resumable: bool
    resolved: bool


@dataclass(frozen=True)
class WorkflowReplay:
    id: str
    from_checkpoint: str
    to_checkpoint: str
    skipped_nodes: list[str]
    rerun_nodes: list[str]
    reason: str


@dataclass(frozen=True)
class WorkflowTraceExport:
    content: str
    framework: str
    workflow_id: str
    thread_id: str
    run_id: str
    workflow_nodes: list[Any]
    workflow_edges: list[Any]
    workflow_steps: list[Any]
    workflow_checkpoints: list[Any]
    route_decisions: list[Any]
    interrupts: list[Any]
    workflow_replay: list[Any]
    pending_writes: list[dict[str, Any]]
    state_history: list[dict[str, Any]]
    workflow_events: list[dict[str, Any]]
    final_state: dict[str, Any]


class LocalLangGraphWorkflowAgent:
    """Local LangGraph-style workflow export for adapter promotion."""

    def run(self, text: str) -> str:
        assert text
        return "Weak workflow response without graph execution evidence."

    async def execute_task(self, payload: dict[str, Any]) -> WorkflowTraceExport:
        assert payload["metadata"]["framework"] == "langgraph"
        return WorkflowTraceExport(
            content=(
                "Workflow trace adapter approved refund after policy route, "
                "checkpoint replay, and human approval."
            ),
            framework="langgraph",
            workflow_id="refund-workflow",
            thread_id="thread-refund-42",
            run_id="run-workflow-001",
            workflow_nodes=[
                WorkflowNode(
                    id="intake",
                    name="intake",
                    type="start",
                    input_keys=["message"],
                    output_keys=["refund_request"],
                ),
                WorkflowNode(
                    id="policy_check",
                    name="policy_check",
                    type="tool_step",
                    input_keys=["refund_request"],
                    output_keys=["policy_result"],
                ),
                WorkflowNode(
                    id="human_review",
                    name="human_review",
                    type="interruptible",
                    input_keys=["policy_result"],
                    output_keys=["approval"],
                ),
                WorkflowNode(
                    id="finalize",
                    name="finalize",
                    type="finish",
                    input_keys=["approval"],
                    output_keys=["decision"],
                ),
            ],
            workflow_edges=[
                WorkflowEdge("intake", "policy_check"),
                WorkflowEdge("policy_check", "human_review", condition="needs_review"),
                WorkflowEdge("human_review", "finalize", condition="approved"),
            ],
            workflow_steps=[
                WorkflowStep(
                    id="step-intake",
                    name="intake",
                    node="intake",
                    status="completed",
                    superstep=1,
                    input={"message": payload["input"]},
                    output={"refund_request": "refund-42"},
                    state_delta={"refund_request": "refund-42"},
                    tool_calls=[],
                ),
                WorkflowStep(
                    id="step-policy",
                    name="policy_check",
                    node="policy_check",
                    status="completed",
                    superstep=2,
                    input={"refund_request": "refund-42"},
                    output={"policy_result": "eligible"},
                    state_delta={"policy_result": "eligible"},
                    tool_calls=[
                        {
                            "id": "policy-lookup-1",
                            "name": "policy_lookup",
                            "arguments": {"case_id": "refund-42"},
                        }
                    ],
                ),
                WorkflowStep(
                    id="step-human-review",
                    name="human_review",
                    node="human_review",
                    status="interrupted",
                    superstep=3,
                    input={"policy_result": "eligible"},
                    output={"approval": "pending"},
                    state_delta={"approval": "pending"},
                    tool_calls=[],
                ),
                WorkflowStep(
                    id="step-finalize",
                    name="finalize",
                    node="finalize",
                    status="completed",
                    superstep=4,
                    input={"approval": "approved"},
                    output={"decision": "approved refund"},
                    state_delta={"decision": "approved refund"},
                    tool_calls=[],
                ),
            ],
            workflow_checkpoints=[
                WorkflowCheckpoint(
                    checkpoint_id="checkpoint-policy",
                    thread_id="thread-refund-42",
                    checkpoint_ns="",
                    superstep=2,
                    state={
                        "refund_request": "refund-42",
                        "policy_result": "eligible",
                    },
                    pending_writes=[
                        {"node": "policy_check", "key": "policy_result", "value": "eligible"}
                    ],
                    next_nodes=["human_review"],
                ),
                WorkflowCheckpoint(
                    checkpoint_id="checkpoint-final",
                    thread_id="thread-refund-42",
                    checkpoint_ns="",
                    superstep=4,
                    state={"decision": "approved refund"},
                    pending_writes=[],
                    next_nodes=[],
                ),
            ],
            route_decisions=[
                WorkflowRoute(
                    source="policy_check",
                    target="human_review",
                    condition="needs_review",
                    selected="human_review",
                    reason="refund exceeds auto-approval amount",
                )
            ],
            interrupts=[
                WorkflowInterrupt(
                    id="interrupt-human-review",
                    node="human_review",
                    reason="human approval required",
                    resumable=True,
                    resolved=True,
                )
            ],
            workflow_replay=[
                WorkflowReplay(
                    id="replay-after-approval",
                    from_checkpoint="checkpoint-policy",
                    to_checkpoint="checkpoint-final",
                    skipped_nodes=["intake", "policy_check"],
                    rerun_nodes=["human_review", "finalize"],
                    reason="resume after human approval",
                )
            ],
            pending_writes=[
                {"node": "human_review", "key": "approval", "value": "approved"}
            ],
            state_history=[
                {"checkpoint_id": "checkpoint-policy", "state_keys": ["policy_result"]},
                {"checkpoint_id": "checkpoint-final", "state_keys": ["decision"]},
            ],
            workflow_events=[
                {"type": "start", "node": "intake"},
                {"type": "stop", "node": "finalize"},
            ],
            final_state={
                "decision": "approved refund",
                "approval": "approved",
                "policy_result": "eligible",
            },
        )


def build_manifest() -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_local_adapter(
        name="sdk-framework-adapter-workflow-trace-run",
        framework="langgraph",
        target=TARGET,
        method_candidates=["run", "execute_task"],
        input_mode_candidates=["text", "dict"],
        discovery_max_candidates=6,
        cases=[
            {
                "id": "workflow-refund",
                "input": "Approve the refund with durable graph execution evidence.",
                "expected_contains": ["approved refund"],
                "required_tools": ["policy_lookup"],
                "required_events": [
                    "workflow_step",
                    "workflow_route",
                    "workflow_checkpoint",
                    "workflow_interrupt",
                    "workflow_replay",
                    "workflow_trace",
                ],
                "required_state_keys": ["framework_runtime", "workflow_trace"],
            }
        ],
        auto_evaluation_config=True,
        metadata={"cookbook": "sdk-framework-adapter-workflow-trace"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    manifest = build_manifest()
    simulate.write_manifest_file(manifest, manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
    result["framework_adapter_workflow_trace_manifest"] = manifest

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
        else Path("artifacts") / "sdk-framework-adapter-workflow-trace.json"
    )
    run(destination)
