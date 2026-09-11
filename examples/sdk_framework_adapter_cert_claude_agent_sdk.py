from __future__ import annotations

import json
from pathlib import Path

from fi.alk import simulate


class LocalClaudeAgentSDKQuery:
    """Local Claude Agent SDK-style shim exposing the preset ``query`` text entry.

    Credential-free: no real claude-agent-sdk import, no network. Returns a
    single-turn transcript shape for the ``message_history`` IO surface.
    """

    def query(self, text):
        assert text
        return {
            "content": (
                "Claude Agent SDK adapter approved refund with query runtime "
                "evidence."
            ),
            "tool_calls": [
                {
                    "id": "claude_agent_status",
                    "name": "framework_trace_status",
                    "arguments": {"status": "passed"},
                }
            ],
            "events": [
                {
                    "type": "framework_trace",
                    "name": "claude_agent_sdk.query",
                    "payload": {"framework": "claude_agent_sdk"},
                }
            ],
        }


def run(output_path: str | Path) -> dict:
    result = simulate.run_framework_adapter_probe(
        "claude_agent_sdk",
        LocalClaudeAgentSDKQuery(),
        target=(
            "sdk_framework_adapter_cert_claude_agent_sdk.py:"
            "LocalClaudeAgentSDKQuery"
        ),
        method="query",
        input_mode="text",
        cases=[
            {
                "id": "claude-agent-sdk-refund",
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
    run(Path("artifacts") / "sdk-framework-adapter-cert-claude_agent_sdk.json")
