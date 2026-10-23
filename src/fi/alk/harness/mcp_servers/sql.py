"""A read-only SQL MCP server over one world's database.

It connects as the world's reader role, which can SELECT only the tables the release lists, and
runs each query as a single prepared statement in a read-only transaction with a statement
timeout and a row cap. The role is the boundary; the rest keeps one query small and harmless.

    WORLD_READER_URL=postgresql://... python -m fi.alk.harness.mcp_servers.sql
"""

from __future__ import annotations

import os
from typing import Any

import psycopg
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

ROW_CAP = 200
STATEMENT_TIMEOUT_MS = 5_000

# Per-request INFO lines would land in every world's diagnostics.
server = FastMCP("sql", log_level="WARNING")


@server.tool(
    description=(
        "Run one read-only SQL query (PostgreSQL) and get the rows back, at most "
        f"{ROW_CAP}. Only SELECT on the tables you are allowed to read works."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
def query(sql: str) -> dict[str, Any]:
    with psycopg.connect(
        os.environ["WORLD_READER_URL"],
        options=f"-c statement_timeout={STATEMENT_TIMEOUT_MS}",
    ) as connection:
        connection.read_only = True
        # Prepared, so a second statement after a `;` is refused instead of run.
        cursor = connection.execute(sql, prepare=True)
        columns = [column.name for column in cursor.description or []]
        rows = cursor.fetchmany(ROW_CAP + 1)
    return {
        "columns": columns,
        "rows": [dict(zip(columns, row, strict=True)) for row in rows[:ROW_CAP]],
        "truncated": len(rows) > ROW_CAP,
    }


if __name__ == "__main__":
    server.run("stdio")
