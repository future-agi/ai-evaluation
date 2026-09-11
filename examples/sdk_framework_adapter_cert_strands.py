from __future__ import annotations

import json
from pathlib import Path

from fi.alk import simulate


class LocalStrandsAgent:
    """Local Strands-style callable agent exposing the preset ``__call__`` text entry.

    Credential-free: no real strands import, no network. The text callable
    returns a transcript shape for the ``message_history`` IO surface.
    """

    def __call__(self, text):
        assert text
        return {
            "content": (
                "Strands adapter approved refund with callable runtime evidence."
            ),
            "tool_calls": [
                {
                    "id": "strands_status",
                    "name": "framework_trace_status",
                    "arguments": {"status": "passed"},
                }
            ],
            "events": [
                {
                    "type": "framework_trace",
                    "name": "strands.__call__",
                    "payload": {"framework": "strands"},
                }
            ],
        }


def run(output_path: str | Path) -> dict:
    result = simulate.run_framework_adapter_probe(
        "strands",
        LocalStrandsAgent(),
        target="sdk_framework_adapter_cert_strands.py:LocalStrandsAgent",
        method="__call__",
        input_mode="text",
        cases=[
            {
                "id": "strands-refund",
                "scenario_name": "framework-adapter-certification",
                "input": "Approve the refund and emit adapter evidence.",
                "expected_contains": ["approved refund"],
                "required_tools": ["framework_trace_status"],
                "required_events": ["framework_trace"],
                "required_state_keys": ["framework_runtime"],
            }
        ],
        metadata={"certification": "11B", "io_surface": "message_history"},
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    run(Path("artifacts") / "sdk-framework-adapter-cert-strands.json")
