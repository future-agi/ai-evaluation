from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, optimize


REQUIRED_ENV = "AGENT_LEARNING_SDK_WORKFLOW_TARGET_OPTIMIZATION_KEY"
TARGET_PATH = "simulation.environments.0.data.trace"
SOURCE_FRAMEWORKS = ["crewai", "langgraph", "llamaindex"]


def _weak_workflow_trace() -> dict[str, Any]:
    return {
        "kind": "workflow_trace",
        "framework": "langgraph",
        "source_frameworks": ["langgraph"],
        "workflow_id": "refund-workflow",
        "thread_id": "thread-refund-42",
        "run_id": "run-workflow-weak",
        "nodes": [{"id": "intake", "name": "intake", "type": "start"}],
        "edges": [],
        "steps": [
            {
                "id": "step-intake",
                "name": "intake",
                "node": "intake",
                "status": "completed",
                "tool_calls": [],
            }
        ],
        "checkpoints": [],
        "route_decisions": [],
        "interrupts": [],
        "replay": [],
        "writes": [],
        "state_snapshots": [],
        "final_state": {"decision": "pending"},
    }


def _partial_crewai_flow_trace() -> dict[str, Any]:
    return {
        "kind": "workflow_trace",
        "framework": "crewai",
        "source_frameworks": ["crewai"],
        "workflow_id": "refund-workflow",
        "thread_id": "thread-refund-42",
        "run_id": "run-workflow-crewai-partial",
        "workflow_nodes": [
            {"id": "intake", "name": "intake", "type": "start"},
            {"id": "policy_check", "name": "policy_check", "type": "task"},
            {"id": "finalize", "name": "finalize", "type": "finish"},
        ],
        "workflow_edges": [
            {"source": "intake", "target": "policy_check"},
            {"source": "policy_check", "target": "finalize"},
        ],
        "workflow_steps": [
            {
                "id": "flow-intake",
                "node": "intake",
                "status": "completed",
                "tool_calls": [],
            },
            {
                "id": "flow-policy",
                "node": "policy_check",
                "status": "completed",
                "tool_calls": [
                    {
                        "id": "policy-lookup-crewai",
                        "name": "policy_lookup",
                        "arguments": {"case_id": "refund-42"},
                    }
                ],
            },
            {
                "id": "flow-finalize",
                "node": "finalize",
                "status": "completed",
                "tool_calls": [],
            },
        ],
        "routes": [
            {
                "source": "policy_check",
                "target": "finalize",
                "selected": "finalize",
            }
        ],
        "workflow_checkpoints": [
            {
                "checkpoint_id": "crewai-policy",
                "state": {"policy_result": "eligible"},
            }
        ],
        "workflow_state": {
            "decision": "approved refund",
            "policy_result": "eligible",
        },
        "metadata": {
            "source_export_type": "crewai_flow_route_state",
            "missing": ["interrupt", "replay", "human_review_checkpoint"],
        },
    }


def _strong_workflow_trace() -> dict[str, Any]:
    return {
        "kind": "workflow_trace",
        "framework": "langgraph",
        "source_frameworks": SOURCE_FRAMEWORKS,
        "workflow_id": "refund-workflow",
        "thread_id": "thread-refund-42",
        "run_id": "run-workflow-cross-framework-001",
        "metadata": {
            "source_exports": [
                {
                    "framework": "langgraph",
                    "export_type": "checkpoint_graph",
                    "signals": ["nodes", "edges", "checkpoints", "interrupts"],
                },
                {
                    "framework": "crewai_flow",
                    "export_type": "route_state",
                    "signals": ["routes", "tasks", "state"],
                },
                {
                    "framework": "llamaindex_workflow",
                    "export_type": "event_trace",
                    "signals": ["steps", "events", "tool_calls"],
                },
            ]
        },
        "nodes": [
            {
                "id": "intake",
                "name": "intake",
                "type": "start",
                "input_keys": ["message"],
                "output_keys": ["refund_request"],
            },
            {
                "id": "policy_check",
                "name": "policy_check",
                "type": "tool_step",
                "input_keys": ["refund_request"],
                "output_keys": ["policy_result"],
            },
            {
                "id": "human_review",
                "name": "human_review",
                "type": "interruptible",
                "input_keys": ["policy_result"],
                "output_keys": ["approval"],
            },
            {
                "id": "finalize",
                "name": "finalize",
                "type": "finish",
                "input_keys": ["approval"],
                "output_keys": ["decision"],
            },
        ],
        "edges": [
            {"source": "intake", "target": "policy_check"},
            {
                "source": "policy_check",
                "target": "human_review",
                "condition": "needs_review",
            },
            {
                "source": "human_review",
                "target": "finalize",
                "condition": "approved",
            },
        ],
        "steps": [
            {
                "id": "step-intake",
                "name": "intake",
                "node": "intake",
                "status": "completed",
                "input": {"message": "Approve the refund."},
                "output": {"refund_request": "refund-42"},
                "state_delta": {"refund_request": "refund-42"},
                "tool_calls": [],
            },
            {
                "id": "step-policy",
                "name": "policy_check",
                "node": "policy_check",
                "status": "completed",
                "input": {"refund_request": "refund-42"},
                "output": {"policy_result": "eligible"},
                "state_delta": {"policy_result": "eligible"},
                "tool_calls": [
                    {
                        "id": "policy-lookup-1",
                        "name": "policy_lookup",
                        "arguments": {"case_id": "refund-42"},
                        "source_framework": "llamaindex_workflow",
                    }
                ],
            },
            {
                "id": "step-human-review",
                "name": "human_review",
                "node": "human_review",
                "status": "interrupted",
                "input": {"policy_result": "eligible"},
                "output": {"approval": "pending"},
                "state_delta": {"approval": "pending"},
                "tool_calls": [],
            },
            {
                "id": "step-finalize",
                "name": "finalize",
                "node": "finalize",
                "status": "completed",
                "input": {"approval": "approved"},
                "output": {"decision": "approved refund"},
                "state_delta": {"decision": "approved refund"},
                "tool_calls": [],
            },
        ],
        "checkpoints": [
            {
                "checkpoint_id": "checkpoint-policy",
                "thread_id": "thread-refund-42",
                "checkpoint_ns": "",
                "superstep": 2,
                "state": {"refund_request": "refund-42", "policy_result": "eligible"},
                "pending_writes": [
                    {
                        "node": "policy_check",
                        "key": "policy_result",
                        "value": "eligible",
                    }
                ],
                "next_nodes": ["human_review"],
            },
            {
                "checkpoint_id": "checkpoint-final",
                "thread_id": "thread-refund-42",
                "checkpoint_ns": "",
                "superstep": 4,
                "state": {"decision": "approved refund"},
                "pending_writes": [],
                "next_nodes": [],
            },
        ],
        "route_decisions": [
            {
                "source": "policy_check",
                "target": "human_review",
                "condition": "needs_review",
                "selected": "human_review",
                "reason": "refund exceeds auto-approval amount",
            }
        ],
        "interrupts": [
            {
                "id": "interrupt-human-review",
                "node": "human_review",
                "reason": "human approval required",
                "resumable": True,
                "resolved": True,
            }
        ],
        "replay": [
            {
                "id": "replay-after-approval",
                "from_checkpoint": "checkpoint-policy",
                "to_checkpoint": "checkpoint-final",
                "skipped_nodes": ["intake", "policy_check"],
                "rerun_nodes": ["human_review", "finalize"],
                "reason": "resume after human approval",
            }
        ],
        "writes": [
            {"node": "human_review", "key": "approval", "value": "approved"}
        ],
        "state_snapshots": [
            {"checkpoint_id": "checkpoint-policy", "state_keys": ["policy_result"]},
            {"checkpoint_id": "checkpoint-final", "state_keys": ["decision"]},
        ],
        "final_state": {
            "decision": "approved refund",
            "approval": "approved",
            "policy_result": "eligible",
        },
    }


def _base_config() -> dict[str, Any]:
    return {
        "agent": {
            "type": "scripted",
            "responses": [
                {
                    "content": (
                        "Because the policy lookup finds eligibility, the "
                        "workflow graph routes to human review, resumes from "
                        "checkpoint replay, finalizes the decision, and "
                        "approves the refund."
                    ),
                    "tool_calls": [
                        {
                            "id": "workflow-status",
                            "name": "workflow_trace_status",
                            "arguments": {},
                        }
                    ],
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
                    "type": "workflow_trace",
                    "data": {"trace": _weak_workflow_trace()},
                }
            ],
        },
    }


def _evaluation_config() -> dict[str, Any]:
    return {
        "task_description": (
            "Optimize a deterministic cross-framework refund workflow graph."
        ),
        "expected_result": (
            "Because the policy lookup finds eligibility, the workflow graph "
            "routes to human review, resumes from checkpoint replay, finalizes "
            "the decision, and approves the refund."
        ),
        "required_tools": ["workflow_trace_status"],
        "required_artifact_types": ["trace"],
        "required_events": [
            "workflow_step",
            "workflow_route",
            "workflow_checkpoint",
            "workflow_interrupt",
            "workflow_replay",
            "workflow_trace",
        ],
        "required_workflow_trace": [
            "workflow_trace",
            "trace",
            "graph",
            "node",
            "edge",
            "step",
            "checkpoint",
            "route",
            "interrupt",
            "replay",
            "write",
            "state",
            "tool",
            "tool_call",
            "final_state",
            "topology",
            "framework",
        ],
        "workflow_trace_quality": {
            "framework": "langgraph",
            "required_frameworks": SOURCE_FRAMEWORKS,
            "min_node_count": 4,
            "min_edge_count": 3,
            "min_step_count": 4,
            "min_checkpoint_count": 2,
            "min_route_decision_count": 1,
            "min_interrupt_count": 1,
            "min_replay_count": 1,
            "min_write_count": 1,
            "min_state_snapshot_count": 2,
            "min_tool_call_count": 1,
            "required_tools": ["policy_lookup"],
            "required_final_state_keys": [
                "approval",
                "decision",
                "policy_result",
            ],
            "required_entry_nodes": ["intake"],
            "required_terminal_nodes": ["finalize"],
            "require_replay": True,
            "require_interrupts": True,
            "require_routes": True,
            "require_topology": True,
            "max_error_count": 0,
        },
        "metric_weights": {
            "workflow_trace_coverage": 5.0,
            "workflow_graph_quality": 8.0,
            "tool_selection_accuracy": 2.0,
            "artifact_coverage": 1.0,
            "task_completion": 1.0,
        },
    }


def _target_candidates() -> dict[str, list[dict[str, Any]]]:
    return {
        TARGET_PATH: [
            _weak_workflow_trace(),
            _partial_crewai_flow_trace(),
            _strong_workflow_trace(),
        ]
    }


def _layers() -> list[str]:
    return ["graph", "router", "orchestration", "harness", "evaluator"]


def _scenario() -> dict[str, Any]:
    return {
        "name": "sdk-workflow-target-optimization",
        "dataset": [
            {
                "persona": {"name": "SDK user", "role": "workflow engineer"},
                "situation": (
                    "A deterministic refund workflow must be optimized as "
                    "graph state, not as a prompt."
                ),
                "outcome": (
                    "Because the policy lookup finds eligibility, the workflow "
                    "graph routes to human review, resumes from checkpoint "
                    "replay, finalizes the decision, and approves the refund."
                ),
            }
        ],
    }


def build_manifest() -> dict[str, Any]:
    return optimize.build_target_optimization_manifest(
        name="sdk-workflow-target-optimization",
        required_env=[REQUIRED_ENV],
        base_config=_base_config(),
        evaluation_config=_evaluation_config(),
        scenario=_scenario(),
        target_candidates=_target_candidates(),
        layers=_layers(),
        min_turns=1,
        max_turns=1,
        threshold=0.98,
        target_metadata={
            "cookbook": "sdk-workflow-target-optimization",
            "optimized_surface": "workflow_trace_graph",
        },
    )


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)

    result = optimize.optimize_target(
        name="sdk-workflow-target-optimization",
        required_env=[REQUIRED_ENV],
        base_config=_base_config(),
        evaluation_config=_evaluation_config(),
        scenario=_scenario(),
        target_candidates=_target_candidates(),
        layers=_layers(),
        min_turns=1,
        max_turns=1,
        threshold=0.98,
        target_metadata={
            "cookbook": "sdk-workflow-target-optimization",
            "optimized_surface": "workflow_trace_graph",
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
