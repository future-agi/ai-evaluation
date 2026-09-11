import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


TARGET = f"{Path(__file__).resolve()}:LocalRealtimeVoiceStack"


@dataclass(frozen=True)
class PipecatFrame:
    frame_type: str
    category: str
    direction: str
    timestamp_ms: int
    content: str = ""
    transcript: str = ""
    tool_name: str = ""
    arguments: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    sample_rate_hz: int | None = None
    duration_ms: int | None = None


@dataclass(frozen=True)
class LiveKitSessionEvent:
    event: str
    timestamp_ms: int
    name: str = ""
    from_state: str = ""
    to_state: str = ""
    role: str = ""
    transcript: str = ""
    tool_name: str = ""
    arguments: dict[str, Any] | None = None
    result: dict[str, Any] | None = None


@dataclass(frozen=True)
class RealtimeTraceExport:
    frames: list[Any]
    session_events: list[Any]
    stop_reason: str


class LocalRealtimeVoiceStack:
    """Local LiveKit/Pipecat-style voice stack export for adapter discovery."""

    def respond(self, text: str) -> str:
        assert text
        return "Weak realtime response without frame or session event evidence."

    async def run_session(self, payload: dict[str, Any]) -> RealtimeTraceExport:
        assert payload["metadata"]["framework"] == "livekit"
        return RealtimeTraceExport(
            frames=[
                PipecatFrame(
                    frame_type="AudioRawFrame",
                    category="data",
                    direction="inbound",
                    timestamp_ms=0,
                    sample_rate_hz=16000,
                    duration_ms=80,
                ),
                PipecatFrame(
                    frame_type="TranscriptionFrame",
                    category="data",
                    direction="inbound",
                    timestamp_ms=12,
                    transcript="Caller asks whether the refund can be approved.",
                ),
                PipecatFrame(
                    frame_type="FunctionCallFrame",
                    category="data",
                    direction="outbound",
                    timestamp_ms=24,
                    tool_name="lookup_refund_policy",
                    arguments={"order_id": "ord-voice-1"},
                ),
                PipecatFrame(
                    frame_type="FunctionCallResultFrame",
                    category="data",
                    direction="inbound",
                    timestamp_ms=38,
                    tool_name="lookup_refund_policy",
                    result={"eligible": True, "policy": "30_day_return"},
                ),
                PipecatFrame(
                    frame_type="EndFrame",
                    category="control",
                    direction="outbound",
                    timestamp_ms=52,
                ),
            ],
            session_events=[
                LiveKitSessionEvent(
                    event="agent_state_changed",
                    name="agent_listening_to_thinking",
                    timestamp_ms=5,
                    from_state="listening",
                    to_state="thinking",
                ),
                LiveKitSessionEvent(
                    event="tool_execution_started",
                    name="lookup_refund_policy",
                    timestamp_ms=25,
                    tool_name="lookup_refund_policy",
                    arguments={"order_id": "ord-voice-1"},
                ),
                LiveKitSessionEvent(
                    event="tool_execution_completed",
                    name="lookup_refund_policy",
                    timestamp_ms=40,
                    tool_name="lookup_refund_policy",
                    result={"eligible": True, "policy": "30_day_return"},
                ),
                LiveKitSessionEvent(
                    event="transcript_final",
                    name="assistant_final_transcript",
                    timestamp_ms=48,
                    role="assistant",
                    transcript=(
                        "Realtime trace adapter approved refund with policy evidence."
                    ),
                ),
                LiveKitSessionEvent(
                    event="session_closed",
                    name="voice_session_closed",
                    timestamp_ms=54,
                    from_state="speaking",
                    to_state="closed",
                ),
            ],
            stop_reason="completed",
        )


def build_manifest() -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_local_adapter(
        name="sdk-framework-adapter-realtime-trace-run",
        framework="livekit",
        target=TARGET,
        method_candidates=["respond", "run_session"],
        input_mode_candidates=["text", "dict"],
        discovery_max_candidates=8,
        cases=[
            {
                "id": "realtime-refund",
                "input": "Approve the refund through a realtime voice trace.",
                "expected_contains": ["approved refund"],
                "required_tools": ["lookup_refund_policy"],
                "required_events": [
                    "realtime_frame",
                    "realtime_tool_call",
                    "realtime_tool_response",
                    "realtime_transcript",
                    "realtime_lifecycle",
                ],
                "required_state_keys": ["framework_runtime", "realtime_trace"],
            }
        ],
        auto_evaluation_config=True,
        metadata={"cookbook": "sdk-framework-adapter-realtime-trace"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    manifest = build_manifest()
    simulate.write_manifest_file(manifest, manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
    result["framework_adapter_realtime_trace_manifest"] = manifest

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
        else Path("artifacts") / "sdk-framework-adapter-realtime-trace.json"
    )
    run(destination)
