from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


TARGET = f"{Path(__file__).resolve()}:LocalPipecatProcessor"


class LocalPipecatProcessor:
    """Pipecat-style frame processor shim with a side-channel direction kwarg."""

    def process_frame(self, *, frame: dict[str, Any], direction: str) -> dict[str, Any]:
        assert frame["metadata"]["framework"] == "pipecat"
        assert direction == "downstream"
        return {
            "content": (
                "Pipecat side-kwarg adapter approved refund with frame direction "
                "and framework evidence."
            ),
            "tool_calls": [
                {
                    "id": "pipecat_framework_status",
                    "name": "framework_trace_status",
                    "arguments": {
                        "status": "passed",
                        "direction": direction,
                        "input_key": "frame",
                    },
                }
            ],
            "events": [
                {
                    "type": "framework_trace",
                    "name": "pipecat_process_frame",
                    "payload": {
                        "framework": frame["metadata"]["framework"],
                        "direction": direction,
                    },
                }
            ],
            "state": {
                "pipecat_frame": {
                    "direction": direction,
                    "input": frame["input"],
                    "message_count": len(frame["messages"]),
                }
            },
        }


def build_manifest() -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_local_adapter(
        name="sdk-framework-adapter-side-kwargs-run",
        framework="pipecat",
        target=TARGET,
        adapter_candidates=[
            {
                "method": "process_frame",
                "input_mode": "dict",
                "input_key": "frame",
            },
            {
                "method": "process_frame",
                "input_mode": "dict",
                "input_key": "frame",
                "input_kwargs": {"direction": "downstream"},
            },
        ],
        cases=[
            {
                "id": "pipecat-refund",
                "input": "Approve the refund and preserve frame direction.",
                "expected_contains": ["approved refund"],
                "required_tools": ["framework_trace_status"],
                "required_events": ["framework_trace"],
                "required_state_keys": ["framework_runtime", "pipecat_frame"],
            }
        ],
        auto_evaluation_config=True,
        metadata={"cookbook": "sdk-framework-adapter-side-kwargs"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    manifest = build_manifest()
    simulate.write_manifest_file(manifest, manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
    result["framework_adapter_side_kwargs_manifest"] = manifest

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
        else Path("artifacts") / "sdk-framework-adapter-side-kwargs.json"
    )
    run(destination)
