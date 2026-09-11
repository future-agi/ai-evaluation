from __future__ import annotations

import json
from pathlib import Path

from fi.alk import simulate


class LocalInstructorClient:
    """Local Instructor-style client exposing the preset ``chat`` entry.

    Credential-free: no real instructor/openai import, no network. The whole
    point of instructor is structured output, so the shim returns a
    typed/structured object in ``state.typed_output`` for the ``typed_output``
    IO surface.
    """

    def chat(self, *, message=None, messages=None, model=None, **params):
        return {
            "content": (
                "Instructor adapter approved refund with structured output "
                "evidence."
            ),
            "tool_calls": [
                {
                    "id": "instructor_status",
                    "name": "framework_trace_status",
                    "arguments": {"status": "passed"},
                }
            ],
            "events": [
                {
                    "type": "framework_trace",
                    "name": "instructor.chat",
                    "payload": {"framework": "instructor"},
                }
            ],
            "state": {
                "typed_output": {
                    "schema": "RefundDecision",
                    "decision": {"verdict": "approved"},
                }
            },
        }


def run(output_path: str | Path) -> dict:
    result = simulate.run_framework_adapter_probe(
        "instructor",
        LocalInstructorClient(),
        target="sdk_framework_adapter_cert_instructor.py:LocalInstructorClient",
        method="chat",
        input_mode="dict",
        cases=[
            {
                "id": "instructor-refund",
                "scenario_name": "framework-adapter-certification",
                "input": "Approve the refund and emit a typed decision.",
                "expected_contains": ["approved refund"],
                "required_tools": ["framework_trace_status"],
                "required_events": ["framework_trace"],
                "required_state_keys": ["framework_runtime", "typed_output"],
            }
        ],
        metadata={"certification": "11B", "io_surface": "typed_output"},
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    run(Path("artifacts") / "sdk-framework-adapter-cert-instructor.json")
