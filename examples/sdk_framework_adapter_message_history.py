import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


TARGET = f"{Path(__file__).resolve()}:LocalAutoGenTeam"


@dataclass(frozen=True)
class TextMessage:
    source: str
    content: str
    type: str = "TextMessage"
    role: str = "assistant"


@dataclass(frozen=True)
class ToolCallRequestEvent:
    source: str
    content: list[dict[str, Any]]
    type: str = "ToolCallRequestEvent"
    role: str = "assistant"


@dataclass(frozen=True)
class ToolCallExecutionEvent:
    source: str
    content: list[dict[str, Any]]
    type: str = "ToolCallExecutionEvent"
    role: str = "tool"


@dataclass(frozen=True)
class TaskResult:
    messages: list[Any]
    stop_reason: str


class LocalAutoGenTeam:
    """AutoGen AgentChat-style team that returns a task transcript."""

    def chat(self, text: str) -> str:
        assert text
        return "Weak team response without transcript or tool evidence."

    async def run(self, *, task: str) -> TaskResult:
        assert task
        return TaskResult(
            messages=[
                TextMessage(
                    source="planner",
                    content="Planner delegates refund evidence to the tool.",
                ),
                ToolCallRequestEvent(
                    source="planner",
                    content=[
                        {
                            "id": "call_framework_status",
                            "name": "framework_trace_status",
                            "arguments": json.dumps(
                                {
                                    "status": "pending",
                                    "task": "refund evidence",
                                }
                            ),
                        }
                    ],
                ),
                ToolCallExecutionEvent(
                    source="tool",
                    content=[
                        {
                            "call_id": "call_framework_status",
                            "name": "framework_trace_status",
                            "content": "framework transcript evidence passed",
                            "is_error": False,
                        }
                    ],
                ),
                TextMessage(
                    source="reviewer",
                    content=(
                        "AutoGen transcript adapter approved refund with "
                        "tool evidence and reviewer closure."
                    ),
                ),
            ],
            stop_reason="completed",
        )


def build_manifest() -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_local_adapter(
        name="sdk-framework-adapter-message-history-run",
        framework="autogen",
        target=TARGET,
        method_candidates=["chat", "run"],
        input_mode_candidates=["text"],
        discovery_max_candidates=4,
        cases=[
            {
                "id": "autogen-transcript-refund",
                "input": "Approve the refund through a tool-backed transcript.",
                "expected_contains": ["approved refund"],
                "required_tools": ["framework_trace_status"],
                "required_events": [
                    "ToolCallRequestEvent",
                    "ToolCallExecutionEvent",
                ],
                "required_state_keys": ["framework_runtime", "message_history"],
            }
        ],
        auto_evaluation_config=True,
        metadata={"cookbook": "sdk-framework-adapter-message-history"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    manifest = build_manifest()
    simulate.write_manifest_file(manifest, manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
    result["framework_adapter_message_history_manifest"] = manifest

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
        else Path("artifacts") / "sdk-framework-adapter-message-history.json"
    )
    run(destination)
