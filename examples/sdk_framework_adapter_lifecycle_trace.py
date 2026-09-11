import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


TARGET = f"{Path(__file__).resolve()}:LocalRealtimeLifecycleAgent"


@dataclass(frozen=True)
class LifecyclePhase:
    id: str
    stage: str
    name: str
    status: str
    session_id: str
    latency_ms: int
    tools: list[str]
    state: dict[str, Any]
    checkpoint: dict[str, Any]
    retry_of: str = ""
    error: dict[str, Any] | None = None
    recovered: bool = False
    state_persisted: bool = False


@dataclass(frozen=True)
class LifecycleTraceExport:
    content: str
    framework: str
    session_id: str
    lifecycle_phases: list[Any]
    lifecycle_state: dict[str, Any]
    lifecycle_metadata: dict[str, Any]
    tool_calls: list[dict[str, Any]]


class LocalRealtimeLifecycleAgent:
    """Local LiveKit/Pipecat-style lifecycle export for adapter promotion."""

    def run(self, text: str) -> str:
        assert text
        return "Weak lifecycle response without retry or cleanup evidence."

    async def execute_task(self, payload: dict[str, Any]) -> LifecycleTraceExport:
        assert payload["metadata"]["framework"] == "livekit"
        session_id = "livekit-session-refund-42"
        phases = [
            LifecyclePhase(
                id="phase-initialize",
                stage="setup",
                name="worker_setup",
                status="completed",
                session_id=session_id,
                latency_ms=12,
                tools=[],
                state={"worker": "ready"},
                checkpoint={},
            ),
            LifecyclePhase(
                id="phase-tool-registration",
                stage="tool_registration",
                name="register_tools",
                status="completed",
                session_id=session_id,
                latency_ms=8,
                tools=["refund_status", "framework_lifecycle_status"],
                state={"registered_tools": 2},
                checkpoint={},
            ),
            LifecyclePhase(
                id="phase-start-session",
                stage="start_session",
                name="agent_session_start",
                status="completed",
                session_id=session_id,
                latency_ms=6,
                tools=[],
                state={"room": "local-fixture"},
                checkpoint={},
            ),
            LifecyclePhase(
                id="phase-invoke-error",
                stage="invoke",
                name="invoke_refund_agent",
                status="failed",
                session_id=session_id,
                latency_ms=31,
                tools=["refund_status"],
                state={"attempt": 1},
                checkpoint={},
                error={"type": "TransientToolTimeout", "message": "local retry"},
            ),
            LifecyclePhase(
                id="phase-retry",
                stage="retry",
                name="retry_refund_agent",
                status="recovered",
                session_id=session_id,
                latency_ms=18,
                tools=["refund_status"],
                state={"attempt": 2, "decision": "approved refund"},
                checkpoint={},
                retry_of="phase-invoke-error",
                recovered=True,
            ),
            LifecyclePhase(
                id="phase-stream",
                stage="stream",
                name="stream_partial_response",
                status="completed",
                session_id=session_id,
                latency_ms=5,
                tools=[],
                state={"stream_chunks": 3},
                checkpoint={},
            ),
            LifecyclePhase(
                id="phase-checkpoint",
                stage="checkpoint",
                name="checkpoint_session_state",
                status="completed",
                session_id=session_id,
                latency_ms=7,
                tools=[],
                state={"decision": "approved refund"},
                checkpoint={"decision": "approved refund", "attempt": 2},
                state_persisted=True,
            ),
            LifecyclePhase(
                id="phase-cancel",
                stage="cancel",
                name="cancel_stale_subtask",
                status="cancelled",
                session_id=session_id,
                latency_ms=4,
                tools=[],
                state={"stale_subtask": "cancelled"},
                checkpoint={},
            ),
            LifecyclePhase(
                id="phase-resume",
                stage="resume",
                name="resume_from_checkpoint",
                status="resumed",
                session_id=session_id,
                latency_ms=9,
                tools=[],
                state={"resumed_from": "phase-checkpoint"},
                checkpoint={"decision": "approved refund"},
                state_persisted=True,
            ),
            LifecyclePhase(
                id="phase-shutdown",
                stage="shutdown",
                name="session_cleanup",
                status="completed",
                session_id=session_id,
                latency_ms=6,
                tools=[],
                state={"cleanup": "complete"},
                checkpoint={},
            ),
        ]
        return LifecycleTraceExport(
            content=(
                "Lifecycle trace adapter approved refund after recovered retry, "
                "checkpoint resume, cancellation, and cleanup."
            ),
            framework="livekit",
            session_id=session_id,
            lifecycle_phases=phases,
            lifecycle_state={
                "decision": "approved refund",
                "attempt": 2,
                "state_persisted": True,
            },
            lifecycle_metadata={
                "runtime": "local",
                "framework_family": "realtime_agent",
            },
            tool_calls=[
                {
                    "id": "lifecycle-status-1",
                    "name": "framework_lifecycle_status",
                    "arguments": {"session_id": session_id},
                }
            ],
        )


def build_manifest() -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_local_adapter(
        name="sdk-framework-adapter-lifecycle-trace-run",
        framework="livekit",
        target=TARGET,
        method_candidates=["run", "execute_task"],
        input_mode_candidates=["text", "dict"],
        discovery_max_candidates=6,
        cases=[
            {
                "id": "lifecycle-refund",
                "input": "Approve the refund with full lifecycle evidence.",
                "expected_contains": ["approved refund"],
                "required_tools": ["framework_lifecycle_status"],
                "required_events": [
                    "framework_lifecycle_phase",
                    "framework_lifecycle_trace",
                ],
                "required_state_keys": [
                    "framework_runtime",
                    "framework_lifecycle_trace",
                ],
            }
        ],
        auto_evaluation_config=True,
        metadata={"cookbook": "sdk-framework-adapter-lifecycle-trace"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    manifest = build_manifest()
    simulate.write_manifest_file(manifest, manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
    result["framework_adapter_lifecycle_trace_manifest"] = manifest

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
        else Path("artifacts") / "sdk-framework-adapter-lifecycle-trace.json"
    )
    run(destination)
