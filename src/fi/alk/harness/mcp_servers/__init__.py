"""MCP servers an environment release may attach to its worlds (`servers.json`).

Only servers named here can be attached, each pinned to what this ALK ships, so a sealed release
always starts the same server. A server is launched over stdio, fresh for every world, by
`world_tools.WorldTools`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

DATABASE_READER = "database_reader"
# The group role the world's seed grants SELECT on the listed tables to; each world's reader
# login is a member of it.
READER_ROLE = "world_reader"


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
}
