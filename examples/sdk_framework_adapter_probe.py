from __future__ import annotations

import json
from pathlib import Path

from fi.alk import simulate


class LocalRefundOrchestrator:
    """Tiny local stand-in for any framework object with a callable method."""

    async def execute_task(self, *, payload):
        return {
            "content": "Adapter probe approved refund with runtime evidence.",
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


def run(output_path: str | Path) -> dict:
    result = simulate.run_framework_adapter_probe(
        "custom_refund_orchestrator",
        LocalRefundOrchestrator(),
        target="sdk_framework_adapter_probe.py:LocalRefundOrchestrator",
        method="execute_task",
        input_mode="dict",
        cases=[
            {
                "id": "refund-status",
                "scenario_name": "framework-adapter-cookbook",
                "input": "Approve the refund and emit adapter evidence.",
                "expected_contains": ["approved refund"],
                "required_tools": ["framework_trace_status"],
                "required_events": ["framework_trace"],
                "required_state_keys": ["framework_runtime"],
            }
        ],
        metadata={"cookbook": "sdk-framework-adapter-probe"},
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    run(Path("artifacts") / "sdk-framework-adapter-probe.json")
