from __future__ import annotations

import json
from pathlib import Path

from fi.alk import simulate


class LocalSmolAgentsRunner:
    """Local SmolAgents-style runner exposing the preset ``run`` text entry.

    Credential-free: no real smolagents import, no network. Returns a text-run
    transcript shape for the ``message_history`` IO surface.
    """

    def run(self, text):
        assert text
        return {
            "content": (
                "SmolAgents adapter approved refund with run runtime evidence."
            ),
            "tool_calls": [
                {
                    "id": "smolagents_status",
                    "name": "framework_trace_status",
                    "arguments": {"status": "passed"},
                }
            ],
            "events": [
                {
                    "type": "framework_trace",
                    "name": "smolagents.run",
                    "payload": {"framework": "smolagents"},
                }
            ],
        }


def run(output_path: str | Path) -> dict:
    result = simulate.run_framework_adapter_probe(
        "smolagents",
        LocalSmolAgentsRunner(),
        target="sdk_framework_adapter_cert_smolagents.py:LocalSmolAgentsRunner",
        method="run",
        input_mode="text",
        cases=[
            {
                "id": "smolagents-refund",
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
    run(Path("artifacts") / "sdk-framework-adapter-cert-smolagents.json")
