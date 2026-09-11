from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


TARGET = f"{Path(__file__).resolve()}:LocalOpenAICompatibleClient"


class LocalChatCompletions:
    """OpenAI-compatible nested chat completions shim."""

    async def create(self, *, messages: list[dict[str, Any]]) -> dict[str, Any]:
        assert messages
        return {
            "content": (
                "Nested OpenAI-compatible adapter approved refund through "
                "chat completions."
            ),
            "tool_calls": [
                {
                    "id": "nested_framework_status",
                    "name": "framework_trace_status",
                    "arguments": {
                        "status": "passed",
                        "method": "chat.completions.create",
                    },
                }
            ],
            "events": [
                {
                    "type": "framework_trace",
                    "name": "nested_chat_completions",
                    "payload": {
                        "framework": "openai",
                        "message_count": len(messages),
                    },
                }
            ],
            "state": {
                "nested_client": {
                    "method_path": "chat.completions.create",
                    "message_count": len(messages),
                    "last_input": str(messages[-1].get("content") or ""),
                }
            },
        }


class LocalChatNamespace:
    def __init__(self) -> None:
        self.completions = LocalChatCompletions()


class LocalOpenAICompatibleClient:
    """Local provider client whose runnable method is nested below chat."""

    def __init__(self) -> None:
        self.chat = LocalChatNamespace()

    def run(self, text: str) -> str:
        assert text
        return "Weak provider response without nested method or tool evidence."


def build_manifest() -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_local_adapter(
        name="sdk-framework-adapter-nested-method-run",
        framework="openai",
        target=TARGET,
        method_candidates=["run", "chat.completions.create"],
        input_mode_candidates=["text", "messages", "dict"],
        discovery_max_candidates=4,
        cases=[
            {
                "id": "nested-provider-refund",
                "input": "Approve the refund through the nested provider client.",
                "expected_contains": ["approved refund"],
                "required_tools": ["framework_trace_status"],
                "required_events": ["framework_trace"],
                "required_state_keys": ["framework_runtime", "nested_client"],
            }
        ],
        auto_evaluation_config=True,
        metadata={"cookbook": "sdk-framework-adapter-nested-method"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    manifest = build_manifest()
    simulate.write_manifest_file(manifest, manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
    result["framework_adapter_nested_method_manifest"] = manifest

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
        else Path("artifacts") / "sdk-framework-adapter-nested-method.json"
    )
    run(destination)
