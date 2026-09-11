import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


TARGET = f"{Path(__file__).resolve()}:LocalMCPToolSessionAgent"


@dataclass(frozen=True)
class MCPToolSessionExport:
    content: str
    framework: str
    server_name: str
    session_id: str
    mcp_tools: list[dict[str, Any]]
    mcp_resources: list[dict[str, Any]]
    mcp_events: list[dict[str, Any]]


class LocalMCPToolSessionAgent:
    """Local MCP client/server export for adapter promotion."""

    def run(self, text: str) -> str:
        assert text
        return "Weak MCP response without tool protocol evidence."

    async def execute_task(self, payload: dict[str, Any]) -> MCPToolSessionExport:
        assert payload["metadata"]["framework"] == "mcp"
        session_id = "mcp-session-refund-42"
        tools = [
            {
                "name": "refund_policy_lookup",
                "description": "Look up current refund eligibility policy.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "case_id": {"type": "string"},
                        "market": {"type": "string"},
                    },
                    "required": ["case_id"],
                },
            },
            {
                "name": "refund_status",
                "description": "Return the refund approval decision.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "case_id": {"type": "string"},
                        "policy": {"type": "string"},
                    },
                    "required": ["case_id", "policy"],
                },
            },
        ]
        return MCPToolSessionExport(
            content=(
                "MCP tool session adapter approved refund after listing tools, "
                "reading policy context, and calling refund tools."
            ),
            framework="mcp",
            server_name="refund-tools",
            session_id=session_id,
            mcp_tools=tools,
            mcp_resources=[
                {
                    "name": "current_refund_policy",
                    "uri": "refund://policy/current",
                    "mimeType": "application/json",
                }
            ],
            mcp_events=[
                {
                    "jsonrpc": "2.0",
                    "id": "tools-list-1",
                    "result": {"tools": tools},
                    "server_name": "refund-tools",
                    "session_id": session_id,
                },
                {
                    "jsonrpc": "2.0",
                    "id": "policy-call-1",
                    "method": "tools/call",
                    "params": {
                        "name": "refund_policy_lookup",
                        "arguments": {"case_id": "refund-42", "market": "us"},
                    },
                    "server_name": "refund-tools",
                    "session_id": session_id,
                },
                {
                    "jsonrpc": "2.0",
                    "id": "policy-call-1",
                    "result": {
                        "structuredContent": {
                            "policy": "standard_refund",
                            "eligible": True,
                            "reason": "within return window",
                        }
                    },
                    "server_name": "refund-tools",
                    "session_id": session_id,
                },
                {
                    "jsonrpc": "2.0",
                    "id": "status-call-1",
                    "method": "tools/call",
                    "params": {
                        "name": "refund_status",
                        "arguments": {
                            "case_id": "refund-42",
                            "policy": "standard_refund",
                        },
                    },
                    "server_name": "refund-tools",
                    "session_id": session_id,
                },
                {
                    "jsonrpc": "2.0",
                    "id": "status-call-1",
                    "result": {
                        "structuredContent": {
                            "status": "approved refund",
                            "approval_id": "refund-approved-42",
                        }
                    },
                    "server_name": "refund-tools",
                    "session_id": session_id,
                },
            ],
        )


def build_manifest() -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_local_adapter(
        name="sdk-framework-adapter-mcp-tool-session-run",
        framework="mcp",
        target=TARGET,
        method_candidates=["run", "execute_task"],
        input_mode_candidates=["text", "dict"],
        discovery_max_candidates=6,
        cases=[
            {
                "id": "mcp-refund",
                "input": "Approve the refund with MCP tool session evidence.",
                "expected_contains": ["approved refund"],
                "required_tools": ["refund_policy_lookup", "refund_status"],
                "required_events": [
                    "mcp_server",
                    "mcp_tool_schema",
                    "mcp_resource",
                    "mcp_tool_call",
                    "mcp_tool_result",
                    "mcp_tool_session",
                ],
                "required_state_keys": [
                    "framework_runtime",
                    "mcp_tool_session",
                ],
            }
        ],
        auto_evaluation_config=True,
        metadata={"cookbook": "sdk-framework-adapter-mcp-tool-session"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    manifest = build_manifest()
    simulate.write_manifest_file(manifest, manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
    result["framework_adapter_mcp_tool_session_manifest"] = manifest

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
        else Path("artifacts") / "sdk-framework-adapter-mcp-tool-session.json"
    )
    run(destination)
