"""Loopback MCP stdio server (P3-D6) — the shipped credential-free fixture.

A REAL ``FastMCP`` server process with deterministic tools: credential-free
but a genuinely separate process speaking the real protocol over the wire —
that IS the live graduation (R§1 #12). Spawned by ``mcp_worker.py`` via the
MCP SDK's own stdio transport; its stdio IS the MCP wire (it does not speak
the lane JSONL protocol).
"""

from __future__ import annotations

import sys

SERVER_NAME = "agent-learning-loopback"
SERVER_VERSION = "1.0.0"


def main() -> int:
    from mcp.server.fastmcp import FastMCP

    server = FastMCP(SERVER_NAME)

    @server.tool()
    def echo(text: str) -> str:
        """Echo the input text back verbatim."""

        return text

    @server.tool()
    def add(a: float, b: float) -> float:
        """Add two numbers deterministically."""

        return a + b

    @server.tool()
    def sort_unique(items: list[str]) -> list[str]:
        """Return the sorted, de-duplicated items (deterministic)."""

        return sorted(set(items))

    server.run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
