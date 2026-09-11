from __future__ import annotations

import json
from pathlib import Path

from fi.alk import simulate


class LocalXAIClient:
    """Local xAI-style client exposing the preset ``chat`` entry.

    Credential-free: no real xai import, no network. Returns a
    synthetic provider-native response (choice + tool_call + usage)
    for the ``provider_response`` IO surface.
    """

    def chat(self, *, message=None, messages=None, model=None, **params):
        return {
            "content": "xAI adapter approved refund with chat runtime evidence.",
            "tool_calls": [
                {
                    "id": "xai_status",
                    "name": "framework_trace_status",
                    "arguments": {"status": "passed"},
                }
            ],
            "events": [
                {
                    "type": "framework_trace",
                    "name": "xai.chat",
                    "payload": {"framework": "xai", "model": model or "grok-2"},
                }
            ],
            "state": {
                "provider_response": {
                    "id": "resp_synthetic",
                    "model": model or "grok-2",
                    "choice_count": 1,
                    "tool_call_count": 1,
                    "finish_reasons": ["tool_calls"],
                    "tool_names": ["framework_trace_status"],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 8,
                        "total_tokens": 20,
                    },
                }
            },
        }


def run(output_path: str | Path) -> dict:
    result = simulate.run_framework_adapter_probe(
        "xai",
        LocalXAIClient(),
        target="sdk_framework_adapter_cert_xai.py:LocalXAIClient",
        method="chat",
        input_mode="dict",
        cases=[
            {
                "id": "xai-refund",
                "scenario_name": "framework-adapter-certification",
                "input": "Approve the refund and emit adapter evidence.",
                "expected_contains": ["approved refund"],
                "required_tools": ["framework_trace_status"],
                "required_events": ["framework_trace"],
                "required_state_keys": ["framework_runtime", "provider_response"],
            }
        ],
        metadata={"certification": "11B", "io_surface": "provider_response"},
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    run(Path("artifacts") / "sdk-framework-adapter-cert-xai.json")
