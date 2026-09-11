from __future__ import annotations

import json
from pathlib import Path

from fi.alk import simulate


class LocalHuggingFacePipeline:
    """Local Hugging Face-style callable pipeline exposing the preset ``__call__``.

    Credential-free: no real transformers/huggingface_hub import, no network.
    The pipeline result is a nested object the adapter coerces, matching the
    ``nested_method`` IO surface.
    """

    def __call__(self, *, payload=None, **params):
        return {
            "content": (
                "Hugging Face adapter approved refund with pipeline runtime "
                "evidence."
            ),
            "tool_calls": [
                {
                    "id": "huggingface_status",
                    "name": "framework_trace_status",
                    "arguments": {"status": "passed"},
                }
            ],
            "events": [
                {
                    "type": "framework_trace",
                    "name": "huggingface.__call__",
                    "payload": {"framework": "huggingface"},
                }
            ],
            "state": {
                "nested_client": {
                    "method_path": "__call__",
                    "message_count": 1,
                }
            },
        }


def run(output_path: str | Path) -> dict:
    result = simulate.run_framework_adapter_probe(
        "huggingface",
        LocalHuggingFacePipeline(),
        target=(
            "sdk_framework_adapter_cert_huggingface.py:LocalHuggingFacePipeline"
        ),
        method="__call__",
        input_mode="dict",
        cases=[
            {
                "id": "huggingface-refund",
                "scenario_name": "framework-adapter-certification",
                "input": "Approve the refund and emit adapter evidence.",
                "expected_contains": ["approved refund"],
                "required_tools": ["framework_trace_status"],
                "required_events": ["framework_trace"],
                "required_state_keys": ["framework_runtime", "nested_client"],
            }
        ],
        metadata={"certification": "11B", "io_surface": "nested_method"},
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    run(Path("artifacts") / "sdk-framework-adapter-cert-huggingface.json")
