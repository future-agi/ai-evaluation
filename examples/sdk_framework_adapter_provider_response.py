from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


TARGET = f"{Path(__file__).resolve()}:LocalProviderResponseClient"


class LocalChatCompletions:
    """OpenAI-compatible response shim with nested choices and tool calls."""

    async def create(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
    ) -> dict[str, Any]:
        assert messages
        assert model == "local-provider-model"
        return {
            "id": "chatcmpl-provider-response",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": (
                            "Provider response adapter approved refund with "
                            "nested tool-call evidence."
                        ),
                        "tool_calls": [
                            {
                                "id": "provider_framework_status",
                                "type": "function",
                                "function": {
                                    "name": "framework_trace_status",
                                    "arguments": json.dumps(
                                        {
                                            "status": "passed",
                                            "model": model,
                                            "provider_response": True,
                                        }
                                    ),
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 7,
                "total_tokens": 19,
            },
        }


class LocalChatNamespace:
    def __init__(self) -> None:
        self.completions = LocalChatCompletions()


class LocalProviderResponseClient:
    """Local provider client whose valuable evidence is nested in choices."""

    def __init__(self) -> None:
        self.chat = LocalChatNamespace()

    def run(self, text: str) -> str:
        assert text
        return "Weak provider response without nested choice tool evidence."


def build_manifest() -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_local_adapter(
        name="sdk-framework-adapter-provider-response-run",
        framework="openai",
        target=TARGET,
        adapter_candidates=[
            {
                "method": "run",
                "input_mode": "text",
            },
            {
                "method": "chat.completions.create",
                "input_mode": "messages",
                "input_key": "messages",
            },
            {
                "method": "chat.completions.create",
                "input_mode": "messages",
                "input_key": "messages",
                "input_kwargs": {"model": "local-provider-model"},
            },
        ],
        cases=[
            {
                "id": "provider-response-refund",
                "input": "Approve the refund through provider response evidence.",
                "expected_contains": ["approved refund"],
                "required_tools": ["framework_trace_status"],
                "required_events": ["provider_choice", "provider_tool_call"],
                "required_state_keys": ["framework_runtime", "provider_response"],
            }
        ],
        auto_evaluation_config=True,
        metadata={"cookbook": "sdk-framework-adapter-provider-response"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    manifest = build_manifest()
    simulate.write_manifest_file(manifest, manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
    result["framework_adapter_provider_response_manifest"] = manifest

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
        else Path("artifacts") / "sdk-framework-adapter-provider-response.json"
    )
    run(destination)
