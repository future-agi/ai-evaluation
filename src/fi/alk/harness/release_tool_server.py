"""The tool server a composed release bundle runs once per world (`environment-releases.md` §6.1).

It executes the environment release's own handlers against the world's PostgreSQL database,
through the run's agent binding (`binding.json` beside it; absent means canonical), and answers
the provider shapes that reach it:

- Vapi: ``POST <TOOL_PATH>/...`` with ``{"message": {"type": "tool-calls", "toolCallList": ...}}``,
  always answered 200 ``{"results": [...]}``; any other Vapi message is acknowledged with ``{}``.
- Retell custom functions: ``POST <TOOL_PATH>/...`` with ``{name, call|chat, args}``, answered 200
  with the rendered content or ``{"error": reason}``.
- Args only (``POST <TOOL_PATH>/<tool>``) and canonical calls (``POST /tools/<name>`` with
  ``{id, arguments}``): only while ``REQUEST_BINDING`` is off.
- Provider events: ``POST <EVENT_PATH>/...``, acknowledged.

With ``REQUEST_BINDING=1`` a tool request runs only when it carries the world's clone secret and
the active call's token (§6.2); anything else is recorded in ``_alk_stray_calls`` and never runs.

``execute_agent_call`` is the one execution path: this server and the release proofs (suite
sealing, binding proofs) both call it, so a proof exercises exactly what a run executes. The
module is copied into the bundle and run as a script on the ALK interpreter, hence absolute
imports.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import CodeType
from typing import Any

import psycopg

from fi.alk.harness import request_binding
from fi.alk.harness.bindings import BoundResult, ToolError, run_bound_call

RESULT_LIMIT = 15_000
BINDING_FILE = "binding.json"
_TOO_LONG = "the result exceeded the 15,000 character limit"


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


def _run_handler(
    handlers: dict[str, CodeType],
    connection: psycopg.Connection,
    tool: str,
    arguments: dict[str, Any],
) -> Any:
    namespace: dict[str, Any] = {"ToolError": ToolError, "json": json}
    exec(handlers[tool], namespace)  # noqa: S102 - sealed release handlers only
    return namespace["handle"](dict(arguments), Db(connection))


def _trace(
    database_url: str,
    outcome: Any,
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
        with psycopg.connect(database_url, autocommit=True) as connection:
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
    except Exception:  # noqa: BLE001 - evidence must never change the tool's answer
        pass


def execute_agent_call(
    handlers: dict[str, CodeType],
    database_url: str,
    binding: Any,
    agent_tool: str,
    arguments: Any,
    *,
    environment_tools: Sequence[Any],
    provider_call_id: str | None = None,
    tool_call_id: str | None = None,
) -> Any:
    """Run one agent call through ``bindings.run_bound_call`` (``binding`` None = canonical) in one
    transaction, committed only when the status is ``ok``, then record its ``_alk_tool_trace``
    rows. Returns the ``BoundResult``."""
    try:
        with psycopg.connect(database_url) as connection:
            with connection.transaction() as transaction:
                outcome = run_bound_call(
                    binding,
                    agent_tool,
                    arguments,
                    run_operation=lambda tool, values: _run_handler(
                        handlers, connection, tool, values
                    ),
                    environment_tools=environment_tools,
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
    _trace(database_url, outcome, agent_tool, arguments, provider_call_id, tool_call_id)
    return outcome


def _parsed(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    return raw


def _one_line(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return " ".join(text.splitlines())


class _Refused(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def make_handler(
    handlers: dict[str, CodeType],
    database_url: str,
    tool_path: str,
    event_path: str,
    *,
    binding: Any = None,
    environment_tools: Sequence[Any] = (),
    request_binding_on: bool = False,
) -> type[BaseHTTPRequestHandler]:
    tool_path = "/" + tool_path.strip("/")
    event_path = "/" + event_path.strip("/")
    agent_tools = sorted(
        tool.name for tool in getattr(binding, "agent_tools", None) or []
    ) or sorted(handlers)

    def call(
        name: str,
        arguments: Any,
        call_id: str | None = None,
        tool_id: str | None = None,
    ) -> Any:
        return execute_agent_call(
            handlers,
            database_url,
            binding,
            name,
            arguments,
            environment_tools=environment_tools,
            provider_call_id=call_id,
            tool_call_id=tool_id,
        )

    def check_binding(headers: Any, body: Any, requests: list[tuple[str, Any]]) -> None:
        """Raise ``_Refused`` after recording every request in ``_alk_stray_calls``."""
        token = request_binding.token_from_request(headers, body)
        with psycopg.connect(database_url, autocommit=True) as connection:
            active = connection.execute(
                "SELECT token, secret_sha256 FROM _alk_active_call WHERE slot = 1"
            ).fetchone()
            if active is None:
                reason = "no_active_call"
            elif not request_binding.secret_matches(
                request_binding.presented_secret(headers), active[1]
            ):
                reason = "bad_secret"
            elif token != active[0]:
                reason = "stray"
            else:
                return
            for tool, arguments in requests:
                connection.execute(
                    "INSERT INTO _alk_stray_calls "
                    "(provider_call_id, token, tool, arguments, reason, at) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (
                        request_binding.provider_call_id(body),
                        token,
                        tool,
                        request_binding.stray_arguments(arguments),
                        reason,
                        time.time(),
                    ),
                )
        raise _Refused(reason)

    def remembered(call_id: str | None, tool_id: str | None, answer: Any) -> Any:
        """Run ``answer()`` once per (provider call, tool call) pair; repeats get the stored
        answer. The advisory lock makes a concurrent retry wait instead of executing twice."""
        if not (request_binding_on and call_id and tool_id):
            return answer()
        with psycopg.connect(database_url, autocommit=True) as connection:
            connection.execute(
                "SELECT pg_advisory_lock(hashtext(%s), hashtext(%s))",
                (call_id, tool_id),
            )
            try:
                stored = connection.execute(
                    "SELECT response FROM _alk_tool_responses "
                    "WHERE provider_call_id = %s AND tool_call_id = %s",
                    (call_id, tool_id),
                ).fetchone()
                if stored is not None:
                    return stored[0]
                response = answer()
                connection.execute(
                    "INSERT INTO _alk_tool_responses "
                    "(provider_call_id, tool_call_id, response, at) "
                    "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    (call_id, tool_id, json.dumps(response), time.time()),
                )
                return response
            finally:
                connection.execute(
                    "SELECT pg_advisory_unlock(hashtext(%s), hashtext(%s))",
                    (call_id, tool_id),
                )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def _send(self, status: int, content: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _reply(self, status: int, body: Any) -> None:
            self._send(
                status,
                json.dumps(body, default=str).encode("utf-8"),
                "application/json",
            )

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path.split("?", 1)[0] == "/health":
                self._reply(
                    200,
                    {"ok": True, "tools": sorted(handlers), "agent_tools": agent_tools},
                )
                return
            self._reply(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = self.path.split("?", 1)[0].rstrip("/")
            size = int(self.headers.get("content-length") or 0)
            try:
                body = json.loads(self.rfile.read(size) or b"{}")
            except ValueError:
                self._reply(400, {"error": "body must be JSON"})
                return
            is_tool_path = path == tool_path or path.startswith(tool_path + "/")
            message = body.get("message") if isinstance(body, dict) else None
            if path == event_path or path.startswith(event_path + "/"):
                self._reply(200, {})
            elif is_tool_path and isinstance(message, dict):
                if message.get("type") == "tool-calls":
                    self._vapi(body, message)
                else:
                    self._reply(200, {})
            elif (
                is_tool_path
                and isinstance(body, dict)
                and {"name", "args"} <= set(body)
            ):
                self._retell(body)
            elif request_binding_on:
                self._reply(404, {"error": "not found"})
            elif path.startswith("/tools/"):
                self._canonical(path.removeprefix("/tools/"), body)
            elif is_tool_path:
                self._answer_retell(call(path.rsplit("/", 1)[-1], body))
            else:
                self._reply(404, {"error": "not found"})

        def _vapi(self, body: dict[str, Any], message: dict[str, Any]) -> None:
            entries = [
                entry if isinstance(entry, dict) else {}
                for entry in message.get("toolCallList") or []
            ]
            requests = [
                (
                    str((entry.get("function") or {}).get("name") or ""),
                    _parsed((entry.get("function") or {}).get("arguments", {})),
                )
                for entry in entries
            ]
            if request_binding_on:
                try:
                    check_binding(self.headers, body, requests)
                except _Refused as refused:
                    if refused.reason == "bad_secret":
                        self._reply(401, {"error": "bad_secret"})
                        return
                    self._reply(
                        200,
                        {
                            "results": [
                                {
                                    "toolCallId": entry.get("id"),
                                    "error": f"request refused: {refused.reason}",
                                }
                                for entry in entries
                            ]
                        },
                    )
                    return
            call_id = request_binding.provider_call_id(body)
            results = []
            for entry, (name, arguments) in zip(entries, requests, strict=True):
                tool_id = str(entry.get("id") or "") or None

                def answer(name: str = name, arguments: Any = arguments) -> Any:
                    outcome = call(name, arguments, call_id, tool_id)
                    if outcome.status == "ok":
                        return {"result": _one_line(outcome.content)}
                    return {"error": _one_line(outcome.error or outcome.status)}

                results.append(
                    {
                        "toolCallId": entry.get("id"),
                        **remembered(call_id, tool_id, answer),
                    }
                )
            self._reply(200, {"results": results})

        def _retell(self, body: dict[str, Any]) -> None:
            name, arguments = str(body["name"]), _parsed(body["args"])
            if request_binding_on:
                try:
                    check_binding(self.headers, body, [(name, arguments)])
                except _Refused as refused:
                    if refused.reason == "bad_secret":
                        self._reply(401, {"error": "bad_secret"})
                    else:
                        self._reply(
                            200, {"error": f"request refused: {refused.reason}"}
                        )
                    return
            self._answer_retell(
                call(name, arguments, request_binding.provider_call_id(body))
            )

        def _answer_retell(self, outcome: Any) -> None:
            if outcome.status != "ok":
                self._reply(200, {"error": outcome.error or outcome.status})
            elif len(outcome.content) > RESULT_LIMIT:
                self._reply(200, {"error": _TOO_LONG})
            else:
                is_json = True
                try:
                    json.loads(outcome.content)
                except ValueError:
                    is_json = False
                self._send(
                    200,
                    outcome.content.encode("utf-8"),
                    "application/json" if is_json else "text/plain; charset=utf-8",
                )

        def _canonical(self, name: str, body: Any) -> None:
            body = body if isinstance(body, dict) else {}
            outcome = call(name, _parsed(body.get("arguments", {})))
            self._reply(
                500 if outcome.status == "fault" else 200,
                {
                    "id": body.get("id"),
                    "status": outcome.status,
                    "content": outcome.content
                    if outcome.status == "ok"
                    else outcome.error,
                    "result": outcome.result,
                },
            )

    return Handler


def main() -> None:
    from fi.alk.harness.releases import load_agent_binding, load_tools

    here = Path(__file__).resolve().parent
    binding_file = here / BINDING_FILE
    handler = make_handler(
        load_handlers(here / "tools.json", here / "handlers"),
        os.environ["DATABASE_URL"],
        os.environ.get("TOOL_PATH", "/provider/tools"),
        os.environ.get("EVENT_PATH", "/provider/events"),
        binding=load_agent_binding(binding_file) if binding_file.is_file() else None,
        environment_tools=load_tools(here),
        request_binding_on=os.environ.get("REQUEST_BINDING") == "1",
    )
    ThreadingHTTPServer(("0.0.0.0", int(os.environ["PORT"])), handler).serve_forever()


if __name__ == "__main__":
    main()
