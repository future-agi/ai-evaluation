from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


TARGET = f"{Path(__file__).resolve()}:LocalCrewOrchestrator"


class LocalCrewOrchestrator:
    """CrewAI-style local shim whose real entrypoint is keyword-only."""

    def run(self, text: str) -> str:
        assert text
        return "Weak crew response without keyword input or tool evidence."

    async def kickoff(self, *, inputs: dict[str, Any]) -> dict[str, Any]:
        assert inputs["metadata"]["framework"] == "crewai"
        return {
            "content": (
                "Crew keyword adapter approved refund with kickoff inputs "
                "and framework evidence."
            ),
            "tool_calls": [
                {
                    "id": "crew_framework_status",
                    "name": "framework_trace_status",
                    "arguments": {"status": "passed", "input_key": "inputs"},
                }
            ],
            "events": [
                {
                    "type": "framework_trace",
                    "name": "crew_kickoff",
                    "payload": {
                        "framework": inputs["metadata"]["framework"],
                        "input_key": "inputs",
                    },
                }
            ],
            "state": {
                "crew_inputs": {
                    "message_count": len(inputs["messages"]),
                    "input": inputs["input"],
                }
            },
        }


def build_manifest() -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_local_adapter(
        name="sdk-framework-adapter-keyword-inputs-run",
        framework="crewai",
        target=TARGET,
        method_candidates=["run", "kickoff"],
        input_mode_candidates=["text", "dict"],
        discovery_max_candidates=4,
        cases=[
            {
                "id": "crew-refund",
                "input": "Approve the refund and preserve crew inputs.",
                "expected_contains": ["approved refund"],
                "required_tools": ["framework_trace_status"],
                "required_events": ["framework_trace"],
                "required_state_keys": ["framework_runtime", "crew_inputs"],
            }
        ],
        auto_evaluation_config=True,
        metadata={"cookbook": "sdk-framework-adapter-keyword-inputs"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    manifest = build_manifest()
    simulate.write_manifest_file(manifest, manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
    result["framework_adapter_keyword_inputs_manifest"] = manifest

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
        else Path("artifacts") / "sdk-framework-adapter-keyword-inputs.json"
    )
    run(destination)
