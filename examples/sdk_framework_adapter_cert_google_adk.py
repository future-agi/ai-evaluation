from __future__ import annotations

import json
from pathlib import Path

from fi.alk import simulate


class LocalGoogleADKRunner:
    """Local Google ADK-style runner exposing the preset ``run`` keyword entry.

    Credential-free: no real google-adk import, no network. Returns
    contract-shaped synthetic evidence via ``run(*, inputs=...)``.
    """

    def run(self, *, inputs=None, **params):
        return {
            "content": (
                "Google ADK adapter approved refund with run runtime evidence."
            ),
            "tool_calls": [
                {
                    "id": "google_adk_status",
                    "name": "framework_trace_status",
                    "arguments": {"status": "passed", "input_key": "inputs"},
                }
            ],
            "events": [
                {
                    "type": "framework_trace",
                    "name": "google_adk.run",
                    "payload": {"framework": "google_adk", "input_key": "inputs"},
                }
            ],
        }


def run(output_path: str | Path) -> dict:
    result = simulate.run_framework_adapter_probe(
        "google_adk",
        LocalGoogleADKRunner(),
        target="sdk_framework_adapter_cert_google_adk.py:LocalGoogleADKRunner",
        method="run",
        input_mode="dict",
        cases=[
            {
                "id": "google-adk-refund",
                "scenario_name": "framework-adapter-certification",
                "input": "Approve the refund and emit adapter evidence.",
                "expected_contains": ["approved refund"],
                "required_tools": ["framework_trace_status"],
                "required_events": ["framework_trace"],
                "required_state_keys": ["framework_runtime"],
            }
        ],
        metadata={"certification": "11B", "io_surface": "keyword_inputs"},
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    run(Path("artifacts") / "sdk-framework-adapter-cert-google_adk.json")
