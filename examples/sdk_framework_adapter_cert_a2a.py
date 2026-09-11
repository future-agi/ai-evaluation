from __future__ import annotations

import json
from pathlib import Path

from fi.alk import simulate


class LocalA2ASession:
    """Local Agent2Agent session shim exposing the preset ``send_message``.

    Credential-free: no real a2a-sdk import, no network. Returns
    contract-shaped synthetic evidence with a message round-trip and a
    session side-kwarg, matching the ``side_kwargs`` IO surface.
    """

    def send_message(self, *, message=None, session=None, **params):
        return {
            "content": (
                "A2A adapter approved refund with send_message runtime evidence."
            ),
            "tool_calls": [
                {
                    "id": "a2a_status",
                    "name": "framework_trace_status",
                    "arguments": {"status": "passed", "session": session or "local"},
                }
            ],
            "events": [
                {
                    "type": "framework_trace",
                    "name": "a2a.send_message",
                    "payload": {"framework": "a2a", "message": message},
                }
            ],
        }


def run(output_path: str | Path) -> dict:
    result = simulate.run_framework_adapter_probe(
        "a2a",
        LocalA2ASession(),
        target="sdk_framework_adapter_cert_a2a.py:LocalA2ASession",
        method="send_message",
        input_mode="dict",
        cases=[
            {
                "id": "a2a-refund",
                "scenario_name": "framework-adapter-certification",
                "input": "Approve the refund and emit adapter evidence.",
                "expected_contains": ["approved refund"],
                "required_tools": ["framework_trace_status"],
                "required_events": ["framework_trace"],
                "required_state_keys": ["framework_runtime"],
            }
        ],
        metadata={
            "certification": "11B",
            "io_surface": "side_kwargs",
            # Cross-links (11B-A11): the redundant cert probe keeps the closed
            # required set homogeneous; the live lane + protocol-trace example
            # remain the deeper A2A surfaces.
            "live_lane": "src/fi/alk/live/a2a_lane.py",
            "protocol_trace": "examples/sdk_framework_adapter_a2a_protocol_trace.py",
        },
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    run(Path("artifacts") / "sdk-framework-adapter-cert-a2a.json")
