"""MCP servers an environment release may attach to its worlds (`servers.json`).

Only servers named here can be attached, each pinned to what this ALK ships, so a sealed release
always starts the same server. A server is launched over stdio, fresh for every world, by
`world_tools.WorldTools`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

DATABASE_READER = "database_reader"
DIRECTORY = "directory"
# The group role the world's seed grants SELECT on the listed tables to; each world's reader
# login is a member of it.
READER_ROLE = "world_reader"
# Third-party servers are installed from their lockfiles in `mcp-servers/` (the hosted image puts
# them here; a local checkout points this at its own `npm ci`), never downloaded at run time.
INSTALLED = Path(os.environ.get("ALK_MCP_SERVERS_ROOT", "/opt/mcp-servers"))


@dataclass(frozen=True)
class RegisteredServer:
    argv: tuple[str, ...]
    attach: frozenset[str]


REGISTRY = {
    # Our read-only SQL server: one `query` tool over the world's listed tables.
    "sql": RegisteredServer(
        argv=(sys.executable, "-m", "fi.alk.harness.mcp_servers.sql"),
        attach=frozenset({DATABASE_READER}),
    ),
    # @modelcontextprotocol/server-filesystem 2026.1.14 (MIT), confined to the attached directory.
    "filesystem": RegisteredServer(
        argv=(
            "node",
            str(
                INSTALLED
                / "filesystem/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js"
            ),
        ),
        attach=frozenset({DIRECTORY}),
    ),
}
