"""Serve source-declared MCP Toolbox SQL tools against an isolated world database."""

from __future__ import annotations

import json
import keyword
import os
import time
import unicodedata
from pathlib import Path
from typing import Any


def _signature(parameters: list[dict[str, Any]]) -> str:
    names: list[str] = []
    reserved = {"_execute", "_spec", "_name", "_database_url", "_trace_path", "locals"}
    python_types = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
    }
    for parameter in parameters:
        name = str(parameter.get("name") or "")
        if (
            not name.isidentifier()
            or keyword.iskeyword(name)
            or name in reserved
            or unicodedata.normalize("NFKC", name) != name
        ):
            raise RuntimeError(f"unsupported Toolbox parameter name: {name!r}")
        reserved.add(name)
        names.append(f"{name}: {python_types.get(str(parameter.get('type')), 'str')}")
    return ", ".join(names)


def _execute(
    tool_name: str,
    spec: dict[str, Any],
    arguments: dict[str, Any],
    *,
    database_url: str,
    trace_path: Path,
) -> str:
    import psycopg

    ordered = [
        arguments[str(parameter["name"])] for parameter in spec.get("parameters") or []
    ]
    started = time.time()
    result: Any = None
    error: str | None = None
    try:
        with psycopg.connect(
            database_url, autocommit=True, cursor_factory=psycopg.RawCursor
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(str(spec["statement"]), ordered)
                if cursor.description:
                    columns = [item.name for item in cursor.description]
                    result = [
                        dict(zip(columns, row, strict=True))
                        for row in cursor.fetchall()
                    ]
                else:
                    result = {"rows_affected": cursor.rowcount}
        return json.dumps(result, default=str, separators=(",", ":"))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as trace:
            trace.write(
                json.dumps(
                    {
                        "name": tool_name,
                        "arguments": arguments,
                        "output": result,
                        "is_error": error is not None,
                        "error": error,
                        "at": started,
                    },
                    default=str,
                    sort_keys=True,
                )
                + "\n"
            )


def _application(definitions: dict[str, Any], database_url: str, trace_path: Path):
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:
        from mcp.server.fastmcp import FastMCP

        server = FastMCP(
            "source-mcp-toolbox",
            streamable_http_path="/mcp/",
            json_response=True,
            stateless_http=True,
            host="127.0.0.1",
        )
        application_factory = server.streamable_http_app
    else:
        server = MCPServer("source-mcp-toolbox")

        def application_factory():
            return server.streamable_http_app(
                streamable_http_path="/mcp/",
                json_response=True,
                stateless_http=True,
                host="127.0.0.1",
            )
    for position, (tool_name, spec) in enumerate(definitions["tools"].items()):
        namespace: dict[str, Any] = {
            "_execute": _execute,
            "_spec": spec,
            "_name": tool_name,
            "_database_url": database_url,
            "_trace_path": trace_path,
        }
        exec(  # noqa: S102 - compiled only from validated declarative parameter names
            f"def generated_{position}({_signature(spec.get('parameters') or [])}):\n"
            "    return _execute(_name, _spec, locals(), "
            "database_url=_database_url, trace_path=_trace_path)\n",
            namespace,
        )
        server.tool(name=tool_name, description=str(spec.get("description") or ""))(
            namespace[f"generated_{position}"]
        )

    application = application_factory()

    async def toolset_compatible(
        scope: dict[str, Any], receive: Any, send: Any
    ) -> None:
        """Toolbox clients append a toolset name to ``/mcp/``; MCP itself does not."""

        if scope.get("type") == "http" and scope.get("path") == "/health":
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": b'{"status":"ok"}'})
            return
        if scope.get("type") == "http" and str(scope.get("path") or "").startswith(
            "/mcp/"
        ):
            scope = {**scope, "path": "/mcp/", "raw_path": b"/mcp/"}
        await application(scope, receive, send)

    return toolset_compatible


if __name__ == "__main__":
    import uvicorn

    definitions = json.loads(Path("toolbox.json").read_text(encoding="utf-8"))
    uvicorn.run(
        _application(
            definitions,
            os.environ["DATABASE_URL"],
            Path(os.environ["HARNESS_TOOL_TRACE"]),
        ),
        host="0.0.0.0",
        port=int(os.environ["PORT"]),
    )
