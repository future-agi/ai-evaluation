from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


TARGET = f"{Path(__file__).resolve()}:LocalOpenAICompatibleProvider"


class LocalChatCompletions:
    """OpenAI-compatible nested chat completions shim promoted through probing."""

    async def create(self, *, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(messages, list) or not messages:
            return {
                "content": "Weak nested response without message-list evidence.",
                "tool_calls": [],
                "state": {"nested_provider_status": "weak"},
            }
        latest = str(messages[-1].get("content") or "")
        trace = {
            "framework": "openai",
            "spans": [
                {
                    "id": "chat-completions-create",
                    "name": "chat.completions.create",
                    "input": latest,
                    "output": "refund_policy_message",
                    "signals": ["provider", "messages", "nested_method"],
                },
                {
                    "id": "tool-policy",
                    "name": "framework_trace_status",
                    "input": {"method": "chat.completions.create"},
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
                "Nested provider adapter approved refund through "
                "chat.completions.create message routing."
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
                        "method": "chat.completions.create",
                        "message_count": len(messages),
                    },
                }
            ],
            "state": {
                "framework_trace": trace,
                "nested_client": {
                    "method_path": "chat.completions.create",
                    "message_count": len(messages),
                    "last_input": latest,
                    "call_style": "keyword",
                    "input_key": "messages",
                },
            },
        }


class LocalChatNamespace:
    def __init__(self) -> None:
        self.completions = LocalChatCompletions()


class LocalOpenAICompatibleProvider:
    """Local provider client whose runnable method is nested below chat."""

    def __init__(self) -> None:
        self.chat = LocalChatNamespace()

    def run(self, text: str) -> dict[str, Any]:
        assert text
        return {
            "content": "Weak provider response without nested-method evidence.",
            "tool_calls": [],
            "state": {"nested_provider_status": "weak"},
        }


def build_manifest() -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_local_adapter(
        name="sdk-framework-adapter-nested-method-promotion-run",
        framework="openai",
        target=TARGET,
        method_candidates=["run", "chat.completions.create"],
        input_mode_candidates=["text", "messages", "dict", "agent_input"],
        discovery_max_candidates=4,
        cases=[
            {
                "id": "nested-provider-refund",
                "input": "Approve the refund through the nested provider client.",
                "expected_contains": ["approved refund"],
                "required_tools": ["framework_trace_status"],
                "required_events": ["framework_trace"],
                "required_state_keys": [
                    "framework_runtime",
                    "framework_trace",
                    "nested_client",
                ],
            }
        ],
        auto_evaluation_config=True,
        metadata={"cookbook": "sdk-framework-adapter-nested-method-promotion"},
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
        / "sdk-framework-adapter-nested-method-promotion.json"
    )
    run(destination)
