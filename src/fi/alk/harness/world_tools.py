"""Every way an agent or a proof reaches one world's tools.

The release tool server (Vapi, Retell, canonical calls), the PostgreSQL proofs and the MCP
endpoint all call `WorldTools.call`, so a proof exercises exactly what a run executes and every
call, whoever made it, leaves the same `_alk_tool_trace` rows. A native call runs the release's
handler in one transaction, committed only when the binding reports `ok`. A call to an attached
MCP server (`servers.json`, tools named `<id>__<tool>`) goes to that server, started fresh for
this world on first use and stopped by `close()`.

The tool server is copied into bundles and run as a script on the ALK interpreter, so this module
only imports what that interpreter has.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import threading
import time
from collections.abc import Sequence
from contextlib import AsyncExitStack
from datetime import timedelta
from pathlib import Path
from types import CodeType
from typing import Any

import psycopg
from psycopg import conninfo, sql

from fi.alk.harness.bindings import BoundResult, ToolError, run_bound_call

# A handler is a few statements against one world; anything slower is a broken handler, and
# without a bound one stuck call would hold the world for the rest of the run.
STATEMENT_TIMEOUT_MS = 30_000
ATTACHED_CALL_SECONDS = 30.0
ATTACHED_START_SECONDS = 60.0
# Passed through to attached servers besides what the MCP client already inherits, so a server
# that is an ALK module runs from the same source as this process.
_SERVER_ENV = ("PYTHONPATH", "LANG", "TZ", "SSL_CERT_FILE", "SSL_CERT_DIR")


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


def ensure_reader_login(admin_url: str, login: str) -> str:
    """A login in the reader role with a fresh password, and the URL it connects with. Roles are
    shared by every database on the server; the grants that decide what it reads are per world."""
    from fi.alk.harness.mcp_servers import READER_ROLE

    password = secrets.token_urlsafe(24)
    with psycopg.connect(admin_url, autocommit=True) as admin:
        exists = admin.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = %s", (login,)
        ).fetchone()
        verb = "ALTER" if exists else "CREATE"
        admin.execute(
            sql.SQL(verb + " ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(login), sql.Literal(password)
            )
        )
        admin.execute(
            sql.SQL("GRANT {} TO {}").format(
                sql.Identifier(READER_ROLE), sql.Identifier(login)
            )
        )
    return conninfo.make_conninfo(admin_url, user=login, password=password)


class AttachedServers:
    """The MCP servers attached to one world, each a stdio process, on one event-loop thread."""

    def __init__(
        self, servers: Sequence[Any], *, reader_url: str | None, cwd: Path | None
    ) -> None:
        self._servers = list(servers)
        self._reader_url = reader_url
        self._cwd = cwd
        self.names = {name for server in self._servers for name in server.tool_names()}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop: asyncio.Event | None = None
        self._sessions: dict[str, tuple[Any, str]] = {}
        self.definitions: list[Any] = []

    def _parameters(self, server: Any) -> Any:
        from mcp.client.stdio import StdioServerParameters

        from fi.alk.harness.mcp_servers import REGISTRY

        registered = REGISTRY[server.server]
        env = {name: os.environ[name] for name in _SERVER_ENV if name in os.environ}
        args, cwd = list(registered.argv[1:]), self._cwd
        if server.attach.database_reader is not None:
            if not self._reader_url:
                raise ToolError(f"{server.id} needs the world's reader role")
            env["WORLD_READER_URL"] = self._reader_url
        if server.attach.directory is not None:
            # Resolved, because the server compares real paths; and its working directory, so a
            # relative path the agent sends lands inside it.
            cwd = (Path(self._cwd or ".") / server.attach.directory).resolve()
            args.append(str(cwd))
        return StdioServerParameters(
            command=registered.argv[0], args=args, env=env, cwd=cwd
        )

    def start(self) -> None:
        if self._loop is not None:
            return
        loop = asyncio.new_event_loop()
        ready: asyncio.Future[None] = loop.create_future()
        self._thread = threading.Thread(target=loop.run_forever, daemon=True)
        self._thread.start()
        self._loop = loop
        asyncio.run_coroutine_threadsafe(self._serve(ready), loop)
        try:
            asyncio.run_coroutine_threadsafe(_wait(ready), loop).result(
                ATTACHED_START_SECONDS
            )
        except BaseException:
            self.close()
            raise

    async def _serve(self, ready: asyncio.Future[None]) -> None:
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        from fi.alk.harness.releases import ToolDefinition

        self._stop = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:
                for server in self._servers:
                    read, write = await stack.enter_async_context(
                        stdio_client(self._parameters(server))
                    )
                    session = await stack.enter_async_context(
                        ClientSession(read, write)
                    )
                    await session.initialize()
                    offered = {
                        tool.name: tool for tool in (await session.list_tools()).tools
                    }
                    if missing := sorted(set(server.tools) - set(offered)):
                        raise ToolError(f"{server.id} offers no {', '.join(missing)}")
                    for tool in server.tools:
                        name = f"{server.id}__{tool}"
                        self._sessions[name] = (session, tool)
                        hints = offered[tool].annotations
                        self.definitions.append(
                            ToolDefinition(
                                name=name,
                                description=offered[tool].description or tool,
                                parameters=offered[tool].inputSchema,
                                # MCP's default is a tool that may write; only a declared
                                # read-only tool is safe to treat as changing nothing.
                                mutates_state=not (
                                    hints is not None and hints.readOnlyHint
                                ),
                            )
                        )
                ready.set_result(None)
                await self._stop.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            else:
                raise

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        self.start()
        assert self._loop is not None
        session, tool = self._sessions[name]
        result = asyncio.run_coroutine_threadsafe(
            session.call_tool(
                tool,
                arguments,
                read_timeout_seconds=timedelta(seconds=ATTACHED_CALL_SECONDS),
            ),
            self._loop,
        ).result(ATTACHED_CALL_SECONDS + 5)
        text = "\n".join(
            part.text for part in result.content if getattr(part, "type", "") == "text"
        )
        if result.isError:
            raise ToolError(text or f"{name} failed")
        if result.structuredContent is not None:
            return result.structuredContent
        try:
            return json.loads(text)
        except ValueError:
            return text

    def close(self) -> None:
        loop, stop, thread = self._loop, self._stop, self._thread
        if loop is None:
            return
        if stop is not None:
            loop.call_soon_threadsafe(stop.set)
        # Give the servers' exit stacks a moment to terminate their processes, then stop.
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0.5), loop).result(5)
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(5)
        if not loop.is_running():
            loop.close()
        self._loop, self._thread = None, None


async def _wait(future: asyncio.Future[None]) -> None:
    await future


def attach_servers(
    servers: Sequence[Any], database_url: str, *, cwd: Path | None = None
) -> AttachedServers | None:
    """The world's attached servers, with a reader login of its own when one reads the
    database. `database_url` is the world's owner; the servers never see it."""
    if not servers:
        return None
    reader_url = None
    if any(server.attach.database_reader is not None for server in servers):
        database = conninfo.conninfo_to_dict(database_url).get("dbname") or "world"
        reader_url = ensure_reader_login(database_url, f"{database}_reader")
    return AttachedServers(servers, reader_url=reader_url, cwd=cwd)


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
        attached: AttachedServers | None = None,
    ) -> None:
        self.handlers = handlers
        self.environment_tools = list(environment_tools)
        self.database_url = database_url
        self.binding = binding
        self.attached = attached

    def tools(self) -> list[Any]:
        """What a canonical agent can call: the release's tools and the attached servers'."""
        if self.attached is None or not self.attached.names:
            return list(self.environment_tools)
        self.attached.start()
        return [*self.environment_tools, *self.attached.definitions]

    def close(self) -> None:
        if self.attached is not None:
            self.attached.close()

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
                    run_operation=lambda tool, values: self._run(
                        connection, tool, values
                    ),
                    environment_tools=self._tools_for(agent_tool),
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

    def _tools_for(self, agent_tool: str) -> list[Any]:
        # Attached servers start only when a call needs them, so a proof attempt or a provider
        # call that never touches them never spawns them.
        if self.attached is not None and agent_tool in self.attached.names:
            return self.tools()
        return self.environment_tools

    def _run(
        self, connection: psycopg.Connection, tool: str, arguments: dict[str, Any]
    ) -> Any:
        if self.attached is not None and tool in self.attached.names:
            return self.attached.call(tool, arguments)
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
