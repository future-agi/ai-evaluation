from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, optimize


REQUIRED_ENV = "AGENT_LEARNING_SDK_WORKFLOW_TARGET_PROFILE_MATRIX_KEY"
TARGET_PATH = "simulation.environments.0.data.trace"
PROFILE_FRAMEWORKS = [
    "langgraph",
    "crewai",
    "llamaindex",
    "langchain",
    "pipecat",
    "livekit",
]
SOURCE_EXPORT_TYPES = {
    "langgraph": "langgraph_checkpoint_graph",
    "crewai": "crewai_flow_route_state",
    "llamaindex": "llamaindex_workflow_events",
    "langchain": "langchain_runnable_graph",
    "pipecat": "pipecat_pipeline_workflow_graph",
    "livekit": "livekit_agent_session_workflow_graph",
}
REQUIRED_METRICS = [
    "workflow_trace_coverage",
    "workflow_graph_quality",
    "tool_selection_accuracy",
    "artifact_coverage",
    "task_completion",
]
REQUIRED_COUNTS = {
    "node_count": 4,
    "edge_count": 3,
    "step_count": 4,
    "checkpoint_count": 2,
    "route_decision_count": 1,
    "interrupt_count": 1,
    "replay_count": 1,
    "write_count": 1,
}


def _base_workflow_trace(framework: str, *, strong: bool) -> dict[str, Any]:
    if not strong:
        return {
            "kind": "workflow_trace",
            "framework": framework,
            "workflow_id": f"{framework}-refund-workflow",
            "thread_id": f"thread-{framework}-refund",
            "run_id": f"run-{framework}-weak",
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

    return {
        "kind": "workflow_trace",
        "framework": framework,
        "workflow_id": f"{framework}-refund-workflow",
        "thread_id": f"thread-{framework}-refund",
        "run_id": f"run-{framework}-verified",
        "nodes": [
            {"id": "intake", "name": "intake", "type": "start"},
            {"id": "policy_check", "name": "policy_check", "type": "tool_step"},
            {
                "id": "human_review",
                "name": "human_review",
                "type": "interruptible",
            },
            {"id": "finalize", "name": "finalize", "type": "finish"},
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
                "state_delta": {"refund_request": "refund-42"},
                "tool_calls": [],
            },
            {
                "id": "step-policy",
                "name": "policy_check",
                "node": "policy_check",
                "status": "completed",
                "state_delta": {"policy_result": "eligible"},
                "tool_calls": [
                    {
                        "id": f"policy-lookup-{framework}",
                        "name": "policy_lookup",
                        "arguments": {"case_id": "refund-42"},
                    }
                ],
            },
            {
                "id": "step-human-review",
                "name": "human_review",
                "node": "human_review",
                "status": "interrupted",
                "state_delta": {"approval": "pending"},
                "tool_calls": [],
            },
            {
                "id": "step-finalize",
                "name": "finalize",
                "node": "finalize",
                "status": "completed",
                "state_delta": {"decision": "approved refund"},
                "tool_calls": [],
            },
        ],
        "checkpoints": [
            {
                "checkpoint_id": f"{framework}-policy",
                "thread_id": f"thread-{framework}-refund",
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
                "checkpoint_id": f"{framework}-final",
                "thread_id": f"thread-{framework}-refund",
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
            }
        ],
        "interrupts": [
            {
                "id": f"interrupt-{framework}-human-review",
                "node": "human_review",
                "reason": "human approval required",
                "resumable": True,
                "resolved": True,
            }
        ],
        "replay": [
            {
                "id": f"replay-{framework}-approval",
                "from_checkpoint": f"{framework}-policy",
                "to_checkpoint": f"{framework}-final",
                "rerun_nodes": ["human_review", "finalize"],
            }
        ],
        "writes": [
            {"node": "human_review", "key": "approval", "value": "approved"}
        ],
        "state_snapshots": [
            {"checkpoint_id": f"{framework}-policy", "state_keys": ["policy_result"]},
            {"checkpoint_id": f"{framework}-final", "state_keys": ["decision"]},
        ],
        "final_state": {
            "approval": "approved",
            "decision": "approved refund",
            "policy_result": "eligible",
        },
    }


def _profile_trace(framework: str, *, strong: bool) -> dict[str, Any]:
    trace = _base_workflow_trace(framework, strong=strong)
    if framework == "langgraph":
        trace["metadata"] = {"source_export_type": SOURCE_EXPORT_TYPES[framework]}
        return trace
    if framework == "crewai":
        trace["metadata"] = {"source_export_type": SOURCE_EXPORT_TYPES[framework]}
        return {
            "kind": trace["kind"],
            "framework": trace["framework"],
            "workflow_id": trace["workflow_id"],
            "thread_id": trace["thread_id"],
            "run_id": trace["run_id"],
            "workflow_nodes": trace["nodes"],
            "workflow_edges": trace["edges"],
            "workflow_steps": trace["steps"],
            "workflow_checkpoints": trace["checkpoints"],
            "routes": trace["route_decisions"],
            "workflow_interrupts": trace["interrupts"],
            "workflow_replay": trace["replay"],
            "pending_writes": trace["writes"],
            "state_history": trace["state_snapshots"],
            "workflow_state": trace["final_state"],
            "metadata": trace["metadata"],
        }
    if framework == "llamaindex":
        trace["metadata"] = {"source_export_type": SOURCE_EXPORT_TYPES[framework]}
        return {
            **trace,
            "routes": trace["route_decisions"],
            "pending_writes": trace["writes"],
            "state_history": trace["state_snapshots"],
            "workflow_state": trace["final_state"],
        }
    if framework == "langchain":
        trace["metadata"] = {"source_export_type": SOURCE_EXPORT_TYPES[framework]}
        return trace
    if framework == "pipecat":
        trace["metadata"] = {"source_export_type": SOURCE_EXPORT_TYPES[framework]}
        return trace
    if framework == "livekit":
        trace["metadata"] = {"source_export_type": SOURCE_EXPORT_TYPES[framework]}
        return trace
    raise ValueError(f"Unsupported workflow profile: {framework}")


def _base_config(framework: str) -> dict[str, Any]:
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
                            "id": f"workflow-status-{framework}",
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
                    "data": {"trace": _profile_trace(framework, strong=False)},
                }
            ],
        },
    }


def _evaluation_config(framework: str) -> dict[str, Any]:
    return {
        "task_description": (
            f"Optimize a deterministic {framework} refund workflow graph target."
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
            "framework": framework,
            "min_node_count": REQUIRED_COUNTS["node_count"],
            "min_edge_count": REQUIRED_COUNTS["edge_count"],
            "min_step_count": REQUIRED_COUNTS["step_count"],
            "min_checkpoint_count": REQUIRED_COUNTS["checkpoint_count"],
            "min_route_decision_count": REQUIRED_COUNTS["route_decision_count"],
            "min_interrupt_count": REQUIRED_COUNTS["interrupt_count"],
            "min_replay_count": REQUIRED_COUNTS["replay_count"],
            "min_write_count": REQUIRED_COUNTS["write_count"],
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


def _target_candidates(framework: str) -> dict[str, list[dict[str, Any]]]:
    return {
        TARGET_PATH: [
            _profile_trace(framework, strong=False),
            _profile_trace(framework, strong=True),
        ]
    }


def _scenario(framework: str) -> dict[str, Any]:
    return {
        "name": f"sdk-workflow-target-profile-{framework}",
        "dataset": [
            {
                "persona": {"name": "SDK user", "role": "workflow engineer"},
                "situation": (
                    f"A deterministic {framework} workflow export must be "
                    "optimized as graph state, not as a prompt."
                ),
                "outcome": (
                    "Because the policy lookup finds eligibility, the workflow "
                    "graph routes to human review, resumes from checkpoint "
                    "replay, finalizes the decision, and approves the refund."
                ),
            }
        ],
    }


def build_manifest(framework: str) -> dict[str, Any]:
    if framework not in PROFILE_FRAMEWORKS:
        raise ValueError(f"Unsupported workflow profile: {framework}")
    return optimize.build_target_optimization_manifest(
        name=f"sdk-workflow-target-profile-{framework}",
        required_env=[REQUIRED_ENV],
        base_config=_base_config(framework),
        evaluation_config=_evaluation_config(framework),
        scenario=_scenario(framework),
        target_candidates=_target_candidates(framework),
        layers=["graph", "router", "orchestration", "harness", "evaluator"],
        min_turns=1,
        max_turns=1,
        threshold=0.98,
        target_metadata={
            "cookbook": "sdk-workflow-target-profile-matrix",
            "optimized_surface": "workflow_trace_profile",
            "profile_framework": framework,
        },
    )


def build_manifests() -> dict[str, dict[str, Any]]:
    return {framework: build_manifest(framework) for framework in PROFILE_FRAMEWORKS}


def _profile_summary(framework: str, result: dict[str, Any]) -> dict[str, Any]:
    best_history = max(
        result["optimization"]["history"],
        key=lambda item: item["score"],
    )
    row = best_history["report"]["results"][0]
    workflow = row["metadata"]["environment_state"]["workflow_trace"]
    workflow_metadata = workflow.get("metadata")
    if not isinstance(workflow_metadata, dict):
        workflow_metadata = {}
    topology = workflow["topology"]
    selected_metrics = {
        metric: best_history["metrics"].get(metric)
        for metric in REQUIRED_METRICS
    }
    return {
        "framework": framework,
        "status": result.get("status"),
        "optimization_score": result["summary"].get("optimization_score"),
        "evaluation_score": result["summary"].get("evaluation_score"),
        "best_score": best_history.get("score"),
        "selected_patch_paths": sorted(best_history.get("patch", {})),
        "selected_metrics": selected_metrics,
        "state_keys": sorted(row["metadata"].get("environment_state", {})),
        "workflow_framework": workflow.get("framework"),
        "source_export_type": workflow_metadata.get("source_export_type"),
        "counts": {key: workflow.get(key) for key in REQUIRED_COUNTS},
        "tool_names": list(workflow.get("tool_names") or []),
        "tool_call_names": [
            tool_call.get("name")
            for tool_call in row.get("tool_calls", [])
            if tool_call.get("name")
        ],
        "final_state_keys": list(workflow.get("final_state_keys") or []),
        "entry_nodes": list(topology.get("entry_nodes") or []),
        "terminal_nodes": list(topology.get("terminal_nodes") or []),
        "has_replay": workflow.get("has_replay"),
        "has_interrupts": workflow.get("has_interrupts"),
        "has_routes": workflow.get("has_routes"),
    }


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)

    profile_results: dict[str, dict[str, Any]] = {}
    profile_summaries: list[dict[str, Any]] = []
    for framework in PROFILE_FRAMEWORKS:
        result = optimize.optimize_target(
            name=f"sdk-workflow-target-profile-{framework}",
            required_env=[REQUIRED_ENV],
            base_config=_base_config(framework),
            evaluation_config=_evaluation_config(framework),
            scenario=_scenario(framework),
            target_candidates=_target_candidates(framework),
            layers=["graph", "router", "orchestration", "harness", "evaluator"],
            min_turns=1,
            max_turns=1,
            threshold=0.98,
            target_metadata={
                "cookbook": "sdk-workflow-target-profile-matrix",
                "optimized_surface": "workflow_trace_profile",
                "profile_framework": framework,
            },
            manifest_path=(
                Path(__file__).with_name(
                    f"{Path(__file__).stem}-{framework}.json"
                )
            ),
        )
        profile_results[framework] = result
        profile_summaries.append(_profile_summary(framework, result))

    failed_profiles = [
        summary["framework"]
        for summary in profile_summaries
        if summary["status"] != "passed"
        or summary["selected_patch_paths"] != [TARGET_PATH]
        or summary["workflow_framework"] != summary["framework"]
        or any(
            float(summary["selected_metrics"].get(metric) or 0.0) < 1.0
            for metric in REQUIRED_METRICS
        )
    ]
    payload = {
        "kind": "agent-learning.workflow-target-profile-matrix.v1",
        "schema_version": "agent-learning.cli.v1",
        "status": "passed" if not failed_profiles else "failed",
        "required_env": [REQUIRED_ENV],
        "target_path": TARGET_PATH,
        "frameworks": list(PROFILE_FRAMEWORKS),
        "summary": {
            "profile_count": len(PROFILE_FRAMEWORKS),
            "passed_profile_count": len(PROFILE_FRAMEWORKS) - len(failed_profiles),
            "failed_profiles": failed_profiles,
            "all_patch_paths": sorted(
                {
                    path
                    for summary in profile_summaries
                    for path in summary["selected_patch_paths"]
                }
            ),
        },
        "profiles": profile_summaries,
        "results": profile_results,
    }
    if output_path is not None:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return payload


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    result_payload = run(destination)
    if destination is None:
        print(json.dumps(result_payload, indent=2, sort_keys=True, default=str))
