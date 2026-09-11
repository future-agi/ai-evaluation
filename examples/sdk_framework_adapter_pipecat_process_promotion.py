from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


TARGET = f"{Path(__file__).resolve()}:LocalPipecatPipeline"


class LocalPipecatPipeline:
    """Local Pipecat-style frame pipeline promoted through BYO adapter probing."""

    def run(self, input: str) -> dict[str, Any]:
        assert input
        return {
            "content": "Weak run response without frame trace or tool evidence.",
            "tool_calls": [],
            "state": {"pipecat_status": "weak"},
        }

    def process(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {
                "content": "Weak process response without dict frame evidence.",
                "tool_calls": [],
                "state": {"pipecat_status": "weak"},
            }
        metadata = payload.get("metadata")
        framework = metadata.get("framework") if isinstance(metadata, dict) else None
        if framework != "pipecat":
            return {
                "content": "Weak process response without Pipecat metadata.",
                "tool_calls": [],
                "state": {"pipecat_status": "weak"},
            }
        modality = payload.get("modality")
        if modality != "voice":
            return {
                "content": "Weak process response without voice frame modality.",
                "tool_calls": [],
                "state": {"pipecat_status": "weak"},
            }
        content = str(payload.get("content") or payload.get("input") or payload)
        trace = {
            "framework": "pipecat",
            "spans": [
                {
                    "id": "voice-frame",
                    "name": "voice_frame.process",
                    "input": content,
                    "output": "refund_frame_routed",
                    "signals": ["voice", "frame", "pipeline"],
                },
                {
                    "id": "tool-policy",
                    "name": "framework_trace_status",
                    "input": {"case_id": "refund-44"},
                    "output": {"status": "passed"},
                    "tool_calls": [{"name": "framework_trace_status"}],
                    "signals": ["tool", "policy"],
                },
            ],
            "summary": {
                "span_count": 2,
                "tool_span_count": 1,
                "status": "passed",
            },
        }
        return {
            "content": (
                "Pipecat process adapter approved refund with voice frame trace "
                "and framework_trace_status evidence."
            ),
            "tool_calls": [
                {
                    "id": "framework_status",
                    "name": "framework_trace_status",
                    "arguments": {"status": "passed", "framework": "pipecat"},
                }
            ],
            "events": [
                {
                    "type": "framework_trace",
                    "name": "pipecat_process",
                    "payload": {"framework": "pipecat", "status": "passed"},
                }
            ],
            "state": {
                "framework_trace": trace,
                "pipecat_frame": {
                    "direction": "downstream",
                    "message_count": len(payload.get("messages") or []),
                    "modality": modality,
                },
                "pipecat_status": "verified",
            },
        }


def build_manifest() -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_local_adapter(
        name="sdk-framework-adapter-pipecat-process-promotion-run",
        framework="pipecat",
        target=TARGET,
        method_candidates=["run", "process"],
        input_mode_candidates=["text", "dict", "agent_input"],
        discovery_max_candidates=4,
        cases=[
            {
                "id": "pipecat-refund-status",
                "input": "Approve the refund through the Pipecat voice pipeline.",
                "modality": "voice",
                "expected_contains": ["approved refund"],
                "required_tools": ["framework_trace_status"],
                "required_events": ["framework_trace"],
                "required_state_keys": [
                    "framework_runtime",
                    "framework_trace",
                    "pipecat_frame",
                ],
            }
        ],
        auto_evaluation_config=True,
        metadata={"cookbook": "sdk-framework-adapter-pipecat-process-promotion"},
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
        else Path("artifacts")
        / "sdk-framework-adapter-pipecat-process-promotion.json"
    )
    run(destination)
