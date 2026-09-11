from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


TARGET = f"{Path(__file__).resolve()}:LocalRefundOrchestrator"


class LocalRefundOrchestrator:
    """Local framework shim with a weak and a verified adapter method."""

    def run(self, text: str) -> dict[str, Any]:
        assert text
        return {
            "content": "Weak adapter response without tool evidence.",
            "tool_calls": [],
            "metadata": {"framework_conformance": "incomplete"},
        }

    async def execute_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "content": (
                "Adapter promotion approved refund with execute_task runtime "
                "evidence."
            ),
            "tool_calls": [
                {
                    "id": "framework_status",
                    "name": "framework_trace_status",
                    "arguments": {"status": "passed"},
                }
            ],
            "events": [
                {
                    "type": "framework_trace",
                    "name": "execute_task",
                    "payload": {"framework": payload["metadata"]["framework"]},
                }
            ],
        }


def build_probe_optimization() -> dict[str, Any]:
    return optimize.optimize_framework_adapter_probe(
        name="sdk-framework-adapter-probe-promotion",
        framework="custom_refund_orchestrator",
        target=TARGET,
        agent_factory=LocalRefundOrchestrator,
        adapter_candidates=[
            {"method": "run", "input_mode": "text"},
            {"method": "execute_task", "input_mode": "dict"},
        ],
        cases=[
            {
                "id": "refund-status",
                "input": "Approve the refund and emit adapter evidence.",
                "expected_contains": ["approved refund"],
                "required_tools": ["framework_trace_status"],
                "required_events": ["framework_trace"],
                "required_state_keys": ["framework_runtime"],
            }
        ],
        metadata={"cookbook": "sdk-framework-adapter-probe-promotion"},
    )


def evaluation_config() -> dict[str, Any]:
    return {
        "task_description": (
            "Promote the selected custom framework adapter into a runnable "
            "simulation manifest."
        ),
        "expected_result": (
            "The selected execute_task adapter emits framework_trace_status "
            "tool evidence and records custom_refund_orchestrator runtime."
        ),
        "required_tools": ["framework_trace_status"],
        "available_tools": ["framework_trace_status"],
        "success_criteria": [
            "execute_task runtime evidence",
            "framework_trace_status tool evidence",
        ],
        "required_framework_trace": [
            "custom_refund_orchestrator",
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
            "framework": "custom_refund_orchestrator",
            "method": "execute_task",
            "input_mode": "dict",
            "required_tools": ["framework_trace_status"],
            "required_signals": ["method", "input", "output", "tool", "metadata"],
            "max_error_count": 0,
            "min_invocation_count": 1,
        },
        "framework_adapter_call_contract_quality": {
            "kind": "agent-learning.framework-adapter-call-contract.v1",
            "framework": "custom_refund_orchestrator",
            "method": "execute_task",
            "input_mode": "dict",
            "call_style": "positional",
            "require_signature": True,
            "require_signature_inspectable": True,
            "require_signature_bound": True,
            "required_parameter_names": ["payload"],
            "max_error_count": 0,
            "min_contract_count": 1,
        },
        "framework_adapter_observed_io_quality": {
            "kind": "agent-learning.framework-adapter-observed-io-contract.v1",
            "framework": "custom_refund_orchestrator",
            "method": "execute_task",
            "input_mode": "dict",
            "required_call_styles": ["positional"],
            "required_input_types": ["dict"],
            "required_output_types": ["agent_response"],
            "required_output_tool_names": ["framework_trace_status"],
            "required_output_event_types": ["framework_trace"],
            "require_content_observed": True,
            "require_signature_bound": True,
            "max_error_count": 0,
            "min_contract_count": 1,
            "min_invocation_count": 1,
        },
        "framework_adapter_contract_quality": {
            "kind": "agent-learning.framework-adapter-contract.v1",
            "framework": "custom_refund_orchestrator",
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
            "framework_adapter_call_contract_quality": 8.0,
            "framework_adapter_observed_io_quality": 8.0,
            "framework_runtime_contract": 10.0,
            "framework_trace_coverage": 4.0,
            "tool_selection_accuracy": 4.0,
            "task_completion": 1.0,
        },
    }


def build_manifest() -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_probe_optimization(
        build_probe_optimization(),
        name="sdk-framework-adapter-probe-promotion-run",
        evaluation_config=evaluation_config(),
        metadata={"cookbook": "sdk-framework-adapter-probe-promotion"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    simulate.write_manifest_file(build_manifest(), manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
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
        else Path("artifacts") / "sdk-framework-adapter-probe-promotion.json"
    )
    run(destination)
