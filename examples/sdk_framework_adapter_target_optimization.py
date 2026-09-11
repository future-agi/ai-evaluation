from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, optimize


REQUIRED_ENV = "AGENT_LEARNING_SDK_FRAMEWORK_ADAPTER_TARGET_OPTIMIZATION_KEY"
FRAMEWORK = "custom_refund_orchestrator"
TARGET = f"{Path(__file__).resolve().with_name('framework_shims.py')}:build_custom_refund_orchestrator"


def _framework_agent(method: str = "run") -> dict[str, Any]:
    return {
        "type": "framework",
        "framework": FRAMEWORK,
        "target": TARGET,
        "factory": True,
        "trace_runtime": True,
        "method": method,
        "input_mode": "dict",
        "metadata": {
            "cookbook": "multi-framework-simulation",
            "sdk_cookbook": "sdk-framework-adapter-target-optimization",
        },
    }


def _framework_trace_environment() -> dict[str, Any]:
    return {
        "type": "framework_trace",
        "data": {
            "framework": FRAMEWORK,
            "spans": [
                {
                    "id": FRAMEWORK,
                    "name": "CustomRefundOrchestrator.execute_task",
                    "input": "refund workflow",
                    "output": "approved",
                    "tool_calls": [{"name": "framework_trace_status"}],
                    "signals": ["planner", "tool", "policy"],
                }
            ],
            "adapter_required_signals": ["planner", "tool", "policy"],
            "adapter_required_mappings": {"tool": ["tool_name"]},
        },
    }


def _base_config() -> dict[str, Any]:
    return {
        "agent": _framework_agent("run"),
        "simulation": {
            "engine": "local_text",
            "min_turns": 1,
            "max_turns": 1,
            "auto_execute_tools": True,
            "environments": [_framework_trace_environment()],
        },
    }


def _evaluation_config() -> dict[str, Any]:
    return {
        "task_description": (
            "Optimize a local custom framework adapter method through one "
            "explicit generic target path."
        ),
        "expected_result": (
            "The selected adapter runs execute_task with fixed dict input, emits "
            "framework_trace_status tool evidence, and records a local framework "
            "runtime contract for custom_refund_orchestrator."
        ),
        "required_tools": ["framework_trace_status"],
        "available_tools": ["framework_trace_status"],
        "success_criteria": [
            "custom_refund_orchestrator runtime trace is present",
            "execute_task is the invoked adapter method",
            "dict remains the invoked adapter input mode",
            "framework_trace_status tool evidence is emitted",
        ],
        "required_framework_trace": [
            "framework_trace",
            FRAMEWORK,
            "planner",
            "tool",
            "policy",
            "framework_trace_status",
        ],
        "required_framework_runtime": [
            "framework_runtime",
            "method",
            "input",
            "output",
            "tool",
            "metadata",
        ],
        "framework_runtime_contract": {
            "framework": FRAMEWORK,
            "method": "execute_task",
            "input_mode": "dict",
            "required_tools": ["framework_trace_status"],
            "required_signals": ["method", "input", "output", "tool", "metadata"],
            "max_error_count": 0,
            "min_invocation_count": 1,
        },
        "framework_adapter_contract_quality": {
            "kind": "agent-learning.framework-adapter-contract.v1",
            "framework": FRAMEWORK,
            "method": "execute_task",
            "input_mode": "dict",
            "require_trace_runtime": True,
            "require_local_executable_fixture": True,
            "require_no_external_service": True,
            "require_target": True,
            "required_schema_sections": ["input", "output"],
            "required_lifecycle_hooks": ["setup", "invoke", "observe", "teardown"],
            "required_capabilities": [
                "messages",
                "tool_calls",
                "runtime_trace",
                "structured_input",
            ],
            "required_evidence_requirements": [
                "framework_runtime",
                "framework_trace",
                "tool_calls",
                "adapter_conformance",
                "metric_evidence",
            ],
        },
        "metric_weights": {
            "framework_adapter_contract_quality": 8.0,
            "framework_runtime_contract": 10.0,
            "framework_runtime_coverage": 4.0,
            "framework_trace_coverage": 2.0,
            "tool_selection_accuracy": 4.0,
            "task_completion": 1.0,
        },
    }


def _target_candidates() -> dict[str, list[str]]:
    return {"agent.method": ["run", "execute_task"]}


def build_manifest() -> dict[str, Any]:
    return optimize.build_target_optimization_manifest(
        name="sdk-framework-adapter-target-optimization",
        required_env=[REQUIRED_ENV],
        base_config=_base_config(),
        evaluation_config=_evaluation_config(),
        target_candidates=_target_candidates(),
        layers=["framework", "harness", "evaluator"],
        target_metadata={
            "cookbook": "sdk-framework-adapter-target-optimization",
            "optimized_surface": "framework_adapter_method",
            "framework": FRAMEWORK,
        },
    )


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)

    result = optimize.optimize_target(
        name="sdk-framework-adapter-target-optimization",
        required_env=[REQUIRED_ENV],
        base_config=_base_config(),
        evaluation_config=_evaluation_config(),
        target_candidates=_target_candidates(),
        layers=["framework", "harness", "evaluator"],
        target_metadata={
            "cookbook": "sdk-framework-adapter-target-optimization",
            "optimized_surface": "framework_adapter_method",
            "framework": FRAMEWORK,
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
