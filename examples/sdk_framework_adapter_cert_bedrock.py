from __future__ import annotations

import json
from pathlib import Path

from fi.alk import simulate


class LocalBedrockClient:
    """Local AWS Bedrock-style client exposing the preset ``invoke_model``.

    Credential-free: no real boto3/bedrock import, no network. Returns a
    Bedrock-shaped synthetic provider-native response (choice + tool_call +
    usage) for the ``provider_response`` IO surface.
    """

    def invoke_model(self, *, payload=None, **params):
        return {
            "content": (
                "Bedrock adapter approved refund with invoke_model runtime "
                "evidence."
            ),
            "tool_calls": [
                {
                    "id": "bedrock_status",
                    "name": "framework_trace_status",
                    "arguments": {"status": "passed"},
                }
            ],
            "events": [
                {
                    "type": "framework_trace",
                    "name": "bedrock.invoke_model",
                    "payload": {"framework": "bedrock"},
                }
            ],
            "state": {
                "provider_response": {
                    "id": "resp_synthetic",
                    "model": "anthropic.claude-3-haiku",
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
        "bedrock",
        LocalBedrockClient(),
        target="sdk_framework_adapter_cert_bedrock.py:LocalBedrockClient",
        method="invoke_model",
        input_mode="dict",
        cases=[
            {
                "id": "bedrock-refund",
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
    run(Path("artifacts") / "sdk-framework-adapter-cert-bedrock.json")
