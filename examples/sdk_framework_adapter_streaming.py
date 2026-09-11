from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, AsyncIterator

from fi.alk import optimize, simulate


TARGET = f"{Path(__file__).resolve()}:LocalStreamingGraphAgent"


class LocalStreamingGraphAgent:
    """Local LangGraph/AutoGen-style stream shim for adapter discovery."""

    def run(self, text: str) -> dict[str, Any]:
        assert text
        return {
            "content": "Weak non-streaming response without runtime evidence.",
            "tool_calls": [],
        }

    async def astream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        assert payload["metadata"]["framework"] == "custom_streaming_graph"
        yield {
            "id": "stream_start",
            "type": "message_delta",
            "name": "stream_start",
            "content": "Streaming adapter approved ",
            "timestamp_ms": 0,
        }
        yield {
            "id": "stream_tool_delta",
            "type": "tool_delta",
            "name": "framework_trace_status",
            "content": "refund with tool evidence. ",
            "tool_calls": [
                {
                    "id": "framework_status",
                    "name": "framework_trace_status",
                    "arguments": {"status": "passed", "streamed": True},
                }
            ],
            "timestamp_ms": 12,
        }
        yield {
            "id": "stream_final",
            "type": "final",
            "name": "stream_complete",
            "content": "Framework streaming trace complete.",
            "timestamp_ms": 24,
        }


def build_manifest() -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_local_adapter(
        name="sdk-framework-adapter-streaming-run",
        framework="custom_streaming_graph",
        target=TARGET,
        method_candidates=["run", "astream"],
        input_mode_candidates=["text", "dict", "messages"],
        discovery_max_candidates=4,
        cases=[
            {
                "id": "streaming-refund",
                "input": "Approve the refund and stream framework evidence.",
                "expected_contains": ["approved refund"],
                "required_tools": ["framework_trace_status"],
                "required_events": ["tool_delta", "final"],
                "required_state_keys": ["framework_runtime", "streaming_trace"],
            }
        ],
        auto_evaluation_config=True,
        metadata={"cookbook": "sdk-framework-adapter-streaming"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    manifest = build_manifest()
    simulate.write_manifest_file(manifest, manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
    result["framework_adapter_streaming_manifest"] = manifest

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
        else Path("artifacts") / "sdk-framework-adapter-streaming.json"
    )
    run(destination)
