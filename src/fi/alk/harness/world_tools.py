"""Every way an agent or a proof reaches one world's tools.

The release tool server (Vapi, Retell, canonical calls), the PostgreSQL proofs and the MCP
endpoint all call `WorldTools.call`, so a proof exercises exactly what a run executes and every
call, whoever made it, leaves the same `_alk_tool_trace` rows. A native call runs the release's
handler in one transaction, committed only when the binding reports `ok`.

The tool server is copied into bundles and run as a script on the ALK interpreter, so this module
only imports what that interpreter has.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from pathlib import Path
from types import CodeType
from typing import Any

import psycopg

from fi.alk.harness.bindings import BoundResult, ToolError, run_bound_call

# A handler is a few statements against one world; anything slower is a broken handler, and
# without a bound one stuck call would hold the world for the rest of the run.
STATEMENT_TIMEOUT_MS = 30_000


def _postgres(statement: str) -> str:
    """SQLite-style ``?`` placeholders to psycopg's, escaping literal ``%`` first."""
    return re.sub(r"\?", "%s", statement.replace("%", "%%"))


class Db:
    """The handler's ``db``: the world runtime's surface over one transaction."""

    def __init__(self, connection: psycopg.Connection) -> None:
        self._connection = connection

    def query(self, sql: str, params: Any = ()) -> list[dict[str, Any]]:
        cursor = self._connection.execute(_postgres(sql), tuple(params))
        columns = [column.name for column in cursor.description or []]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    def one(self, sql: str, params: Any = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params: Any = ()) -> int:
        return max(
            0, int(self._connection.execute(_postgres(sql), tuple(params)).rowcount)
        )

    def records(self, collection: str) -> list[dict[str, Any]]:
        return self.query(f'SELECT * FROM "{collection}"')

    def find(self, collection: str, **fields: Any) -> list[dict[str, Any]]:
        return [
            record
            for record in self.records(collection)
            if all(record.get(field) == value for field, value in fields.items())
        ]


def load_handlers(tools_file: Path, handlers_directory: Path) -> dict[str, CodeType]:
    """One compiled handler per tool in ``tools_file``, from ``<tool>.py``."""
    tools = json.loads(tools_file.read_text(encoding="utf-8"))
    return {
        tool["name"]: compile(
            (handlers_directory / f"{tool['name']}.py").read_text(encoding="utf-8"),
            f"<handler:{tool['name']}>",
            "exec",
        )
        for tool in tools
    }


class WorldTools:
    """One world's tools: the release's handlers on its PostgreSQL database, through a binding
    (`None` is canonical)."""

    def __init__(
        self,
        handlers: dict[str, CodeType],
        environment_tools: Sequence[Any],
        database_url: str,
        *,
        binding: Any = None,
    ) -> None:
        self.handlers = handlers
        self.environment_tools = list(environment_tools)
        self.database_url = database_url
        self.binding = binding

    def call(
        self,
        agent_tool: str,
        arguments: Any,
        *,
        provider_call_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> BoundResult:
        """Run one agent call and record its trace rows; never raises for the world's sake."""
        try:
            with (
                psycopg.connect(
                    self.database_url,
                    options=f"-c statement_timeout={STATEMENT_TIMEOUT_MS}",
                ) as connection,
                connection.transaction() as transaction,
            ):
                outcome = run_bound_call(
                    self.binding,
                    agent_tool,
                    arguments,
                    run_operation=lambda tool, values: self._run_handler(
                        connection, tool, values
                    ),
                    environment_tools=self.environment_tools,
                )
                if outcome.status != "ok":
                    raise psycopg.Rollback(transaction)
        except Exception as exc:  # noqa: BLE001 - the world database itself failed
            outcome = BoundResult(
                status="fault",
                content="",
                result=None,
                error=f"{type(exc).__name__}: {exc}",
                operations=[],
            )
        self._trace(outcome, agent_tool, arguments, provider_call_id, tool_call_id)
        return outcome

    def _run_handler(
        self, connection: psycopg.Connection, tool: str, arguments: dict[str, Any]
    ) -> Any:
        namespace: dict[str, Any] = {"ToolError": ToolError, "json": json}
        exec(self.handlers[tool], namespace)  # noqa: S102 - sealed release handlers only
        return namespace["handle"](dict(arguments), Db(connection))

    def _trace(
        self,
        outcome: BoundResult,
        agent_tool: str,
        arguments: Any,
        provider_call_id: str | None,
        tool_call_id: str | None,
    ) -> None:
        ok = outcome.status == "ok"
        steps = [
            (operation.tool, operation.arguments, operation.result, operation.error)
            for operation in outcome.operations
        ] or [(agent_tool, arguments, outcome.result, outcome.error)]
        try:
            with psycopg.connect(self.database_url, autocommit=True) as connection:
                for name, step_arguments, result, error in steps:
                    connection.execute(
                        "INSERT INTO _alk_tool_trace (name, arguments, result, ok, error, at, "
                        "agent_tool, provider_call_id, tool_call_id, status) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (
                            name,
                            json.dumps(step_arguments, default=str),
                            json.dumps(result if ok else None, default=str),
                            ok,
                            "" if ok else (outcome.error or error or ""),
                            time.time(),
                            agent_tool,
                            provider_call_id,
                            tool_call_id,
                            outcome.status,
                        ),
                    )
        except Exception:  # noqa: BLE001, S110 - evidence must never change the tool's answer
            pass
