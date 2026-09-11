from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


TARGET = f"{Path(__file__).resolve()}:LocalRefundOrchestrator"


class LocalRefundOrchestrator:
    """Local framework shim optimized, promoted, evaluated, and run."""

    def run(self, text: str) -> dict[str, Any]:
        assert text
        return {
            "content": "Weak adapter response without tool evidence.",
            "tool_calls": [],
        }

    async def execute_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "content": (
                "Direct one-call adapter run approved refund with execute_task "
                "runtime evidence."
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


async def run_async(output_path: str | Path) -> dict[str, Any]:
    result = await optimize.run_framework_adapter_from_local_adapter(
        name="sdk-framework-adapter-one-call-run",
        framework="custom_refund_orchestrator",
        target=TARGET,
        method_candidates=["run", "execute_task"],
        input_mode_candidates=["text", "dict", "agent_input"],
        discovery_max_candidates=4,
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
        auto_evaluation_config=True,
        metadata={"cookbook": "sdk-framework-adapter-one-call-run"},
    )

    output = Path(output_path).expanduser()
    manifest = result.get("framework_adapter_run_manifest")
    if isinstance(manifest, dict):
        simulate.write_manifest_file(manifest, output.with_suffix(".manifest.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return result


def run(output_path: str | Path) -> dict[str, Any]:
    return asyncio.run(run_async(output_path))


if __name__ == "__main__":
    destination = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts") / "sdk-framework-adapter-one-call-run.json"
    )
    run(destination)
