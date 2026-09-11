from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


TARGET = f"{Path(__file__).resolve()}:LocalLiveKitAgentSession"


class LocalLiveKitAgentSession:
    """Local LiveKit-style session adapter promoted through BYO probing."""

    def respond(self, text: str) -> dict[str, Any]:
        assert text
        return {
            "content": "Weak LiveKit response without room session evidence.",
            "tool_calls": [],
            "state": {"livekit_status": "weak"},
        }

    async def run_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {
                "content": "Weak session response without dict payload evidence.",
                "tool_calls": [],
                "state": {"livekit_status": "weak"},
            }
        metadata = payload.get("metadata")
        framework = metadata.get("framework") if isinstance(metadata, dict) else None
        if framework != "livekit":
            return {
                "content": "Weak session response without LiveKit metadata.",
                "tool_calls": [],
                "state": {"livekit_status": "weak"},
            }
        modality = payload.get("modality")
        if modality != "voice":
            return {
                "content": "Weak session response without voice modality.",
                "tool_calls": [],
                "state": {"livekit_status": "weak"},
            }

        content = str(payload.get("content") or payload.get("input") or payload)
        session_id = "livekit-session-refund-42"
        trace = {
            "framework": "livekit",
            "spans": [
                {
                    "id": "room-session-start",
                    "name": "room.local_session.start",
                    "input": content,
                    "output": session_id,
                    "signals": ["voice", "room", "session"],
                },
                {
                    "id": "participant-turn",
                    "name": "participant.transcript.final",
                    "input": "caller_refund_request",
                    "output": "refund_policy_route",
                    "signals": ["transcript", "voice", "policy"],
                },
                {
                    "id": "tool-policy",
                    "name": "framework_trace_status",
                    "input": {"session_id": session_id, "framework": "livekit"},
                    "output": {"status": "passed"},
                    "tool_calls": [{"name": "framework_trace_status"}],
                    "signals": ["tool", "policy"],
                },
            ],
            "summary": {
                "span_count": 3,
                "tool_span_count": 1,
                "status": "passed",
            },
        }
        return {
            "content": (
                "LiveKit session adapter approved refund with voice room trace "
                "and framework_trace_status evidence."
            ),
            "tool_calls": [
                {
                    "id": "livekit-framework-status",
                    "name": "framework_trace_status",
                    "arguments": {
                        "status": "passed",
                        "framework": "livekit",
                        "session_id": session_id,
                    },
                }
            ],
            "events": [
                {
                    "type": "framework_trace",
                    "name": "livekit_session_trace",
                    "payload": {
                        "framework": "livekit",
                        "session_id": session_id,
                        "status": "passed",
                    },
                },
                {
                    "type": "livekit_session_event",
                    "name": "participant_turn_completed",
                    "payload": {
                        "room": "local-refund-room",
                        "participant": "caller",
                        "modality": modality,
                    },
                },
                {
                    "type": "livekit_transcript",
                    "name": "assistant_final_transcript",
                    "payload": {
                        "session_id": session_id,
                        "role": "assistant",
                        "transcript": "approved refund",
                    },
                },
            ],
            "state": {
                "framework_trace": trace,
                "livekit_session": {
                    "session_id": session_id,
                    "room": "local-refund-room",
                    "room_name": "local-refund-room",
                    "participant_count": 2,
                    "modality": modality,
                    "transport": "in_process",
                    "final_transcript": "approved refund",
                    "closed": True,
                },
                "livekit_status": "verified",
            },
        }


def build_manifest() -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_local_adapter(
        name="sdk-framework-adapter-livekit-run-session-promotion-run",
        framework="livekit",
        target=TARGET,
        method_candidates=["respond", "run_session"],
        input_mode_candidates=["text", "dict", "agent_input"],
        discovery_max_candidates=8,
        cases=[
            {
                "id": "livekit-session-refund",
                "input": "Approve the refund through the LiveKit room session.",
                "modality": "voice",
                "expected_contains": ["approved refund"],
                "required_tools": ["framework_trace_status"],
                "required_events": [
                    "framework_trace",
                    "livekit_session_event",
                    "livekit_transcript",
                ],
                "required_state_keys": [
                    "framework_runtime",
                    "framework_trace",
                    "livekit_session",
                ],
            }
        ],
        auto_evaluation_config=True,
        metadata={"cookbook": "sdk-framework-adapter-livekit-run-session-promotion"},
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
        / "sdk-framework-adapter-livekit-run-session-promotion.json"
    )
    run(destination)
