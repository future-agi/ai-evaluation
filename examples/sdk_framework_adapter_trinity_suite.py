from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import suite


TARGET = f"{Path(__file__).resolve()}:LocalRefundOrchestrator"


class LocalRefundOrchestrator:
    """Local framework shim covered by simulation, eval, and red-team jobs."""

    def run(self, text: str) -> dict[str, Any]:
        assert text
        return {
            "content": "Weak adapter response without framework evidence.",
            "tool_calls": [],
        }

    async def execute_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "content": (
                "Framework trinity suite approved refund with execute_task "
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


def write_workspace(directory: str | Path) -> dict[str, Any]:
    return suite.write_framework_adapter_trinity_suite_workspace(
        name="sdk-framework-adapter-trinity-suite",
        framework="custom_refund_orchestrator",
        target=TARGET,
        directory=directory,
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
        redteam_attacks=["prompt_injection", "credential_exfiltration"],
        redteam_surfaces=["instruction", "tool"],
        metadata={"cookbook": "sdk-framework-adapter-trinity-suite"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    workspace_dir = output.with_suffix("")
    workspace = write_workspace(workspace_dir)
    result = suite.run_suite_file(workspace["paths"]["suite"])
    result["framework_adapter_trinity_workspace"] = workspace

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
        else Path("artifacts") / "sdk-framework-adapter-trinity-suite.json"
    )
    run(destination)
