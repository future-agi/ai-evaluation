import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


TARGET = f"{Path(__file__).resolve()}:LocalA2AReviewAgent"


@dataclass(frozen=True)
class A2AProtocolTraceExport:
    content: str
    framework: str
    protocol: str
    agent_card: dict[str, Any]
    a2a_events: list[dict[str, Any]]
    a2a_tasks: list[dict[str, Any]]


class LocalA2AReviewAgent:
    """Local Agent2Agent-style protocol export for adapter promotion."""

    def run(self, text: str) -> str:
        assert text
        return "Weak A2A response without protocol task evidence."

    async def send_message(self, payload: dict[str, Any]) -> A2AProtocolTraceExport:
        assert payload["metadata"]["framework"] == "a2a"
        context_id = "a2a-refund-context-42"
        task_id = "a2a-task-refund-review"
        agent_card = {
            "name": "refund-review-agent",
            "description": "Reviews refund requests through A2A task updates.",
            "url": "https://local.example/a2a/refund-review",
            "version": "1.0.0",
            "protocolVersion": "0.3.0",
            "preferredTransport": "JSONRPC",
            "defaultInputModes": ["text/plain", "application/json"],
            "defaultOutputModes": ["text/plain", "application/json"],
            "capabilities": {
                "streaming": True,
                "pushNotifications": False,
                "stateTransitionHistory": True,
            },
            "skills": [
                {
                    "id": "refund_review",
                    "name": "refund_review",
                    "description": "Evaluate eligibility and return approval state.",
                    "tags": ["refund", "policy", "review"],
                    "examples": ["Approve refund for case refund-42."],
                    "inputModes": ["text/plain", "application/json"],
                    "outputModes": ["text/plain", "application/json"],
                }
            ],
        }
        user_message = {
            "role": "user",
            "messageId": "msg-a2a-user-1",
            "contextId": context_id,
            "parts": [
                {
                    "kind": "text",
                    "text": payload["input"],
                },
                {
                    "kind": "data",
                    "data": {"case_id": "refund-42", "amount": 125},
                },
            ],
        }
        working_message = {
            "role": "agent",
            "messageId": "msg-a2a-agent-working",
            "taskId": task_id,
            "contextId": context_id,
            "parts": [
                {
                    "kind": "text",
                    "text": "Shared task state established; reviewing refund policy.",
                }
            ],
        }
        final_message = {
            "role": "agent",
            "messageId": "msg-a2a-agent-final",
            "taskId": task_id,
            "contextId": context_id,
            "parts": [
                {
                    "kind": "text",
                    "text": "A2A review complete: approved refund.",
                }
            ],
        }
        artifact = {
            "artifactId": "artifact-a2a-refund-decision",
            "name": "refund_decision",
            "description": "Structured A2A refund decision artifact.",
            "parts": [
                {
                    "kind": "data",
                    "data": {
                        "case_id": "refund-42",
                        "decision": "approved refund",
                        "review_agent": "refund-review-agent",
                    },
                }
            ],
        }
        final_task = {
            "id": task_id,
            "contextId": context_id,
            "status": {
                "state": "completed",
                "message": final_message,
            },
            "history": [user_message, working_message, final_message],
            "artifacts": [artifact],
        }
        return A2AProtocolTraceExport(
            content=(
                "A2A protocol adapter approved refund after remote agent card "
                "inspection, task status updates, and decision artifact."
            ),
            framework="a2a",
            protocol="a2a",
            agent_card=agent_card,
            a2a_events=[
                {
                    "jsonrpc": "2.0",
                    "id": "a2a-send-1",
                    "method": "SendMessage",
                    "params": {"message": user_message},
                },
                {
                    "jsonrpc": "2.0",
                    "id": "a2a-send-1",
                    "result": {
                        "id": task_id,
                        "contextId": context_id,
                        "status": {
                            "state": "working",
                            "message": working_message,
                        },
                        "history": [user_message, working_message],
                    },
                },
                {
                    "type": "TaskStatusUpdateEvent",
                    "taskId": task_id,
                    "contextId": context_id,
                    "status": {
                        "state": "working",
                        "message": working_message,
                    },
                    "final": False,
                },
                {
                    "type": "TaskArtifactUpdateEvent",
                    "taskId": task_id,
                    "contextId": context_id,
                    "artifact": artifact,
                    "final": False,
                },
                {
                    "type": "TaskStatusUpdateEvent",
                    "taskId": task_id,
                    "contextId": context_id,
                    "status": {
                        "state": "completed",
                        "message": final_message,
                    },
                    "final": True,
                },
            ],
            a2a_tasks=[final_task],
        )


def build_manifest() -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_local_adapter(
        name="sdk-framework-adapter-a2a-protocol-trace-run",
        framework="a2a",
        target=TARGET,
        method_candidates=["run", "send_message"],
        input_mode_candidates=["text", "dict"],
        discovery_max_candidates=6,
        cases=[
            {
                "id": "a2a-refund-review",
                "input": "Approve refund collaboratively through A2A.",
                "expected_contains": ["approved refund"],
                "required_events": [
                    "a2a_agent_card",
                    "a2a_message_send",
                    "a2a_task_status",
                    "a2a_task_artifact",
                    "a2a_artifact",
                    "a2a_protocol_trace",
                ],
                "required_state_keys": [
                    "framework_runtime",
                    "a2a_protocol_trace",
                ],
            }
        ],
        auto_evaluation_config=True,
        metadata={"cookbook": "sdk-framework-adapter-a2a-protocol-trace"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    manifest = build_manifest()
    simulate.write_manifest_file(manifest, manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
    result["framework_adapter_a2a_protocol_trace_manifest"] = manifest

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
        else Path("artifacts") / "sdk-framework-adapter-a2a-protocol-trace.json"
    )
    run(destination)
