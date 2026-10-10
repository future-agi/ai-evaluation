"""Serve source-declared MCP Toolbox SQL tools against an isolated world database."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import psycopg
import uvicorn
from mcp.server.mcpserver import MCPServer


DEFINITIONS = json.loads(Path("toolbox.json").read_text(encoding="utf-8"))
DATABASE_URL = os.environ["DATABASE_URL"]
TRACE_PATH = Path(os.environ["HARNESS_TOOL_TRACE"])


def _signature(parameters: list[dict[str, Any]]) -> str:
    names: list[str] = []
    python_types = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
    }
    for parameter in parameters:
        name = str(parameter.get("name") or "").strip()
        if not name.isidentifier():
            raise RuntimeError(f"unsupported Toolbox parameter name: {name!r}")
        names.append(f"{name}: {python_types.get(str(parameter.get('type')), 'str')}")
    return ", ".join(names)


def _execute(tool_name: str, spec: dict[str, Any], arguments: dict[str, Any]) -> str:
    statement = str(spec["statement"])
    ordered: list[Any] = []

    def bind(match: re.Match[str]) -> str:
        index = int(match.group(1)) - 1
        parameters = spec.get("parameters") or []
        if index < 0 or index >= len(parameters):
            raise RuntimeError(f"invalid SQL parameter ${index + 1} in {tool_name}")
        ordered.append(arguments.get(str(parameters[index]["name"])))
        return "%s"

    sql = re.sub(r"\$(\d+)", bind, statement)
    started = time.time()
    result: Any = None
    error: str | None = None
    try:
        with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, ordered)
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
        TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRACE_PATH.open("a", encoding="utf-8") as trace:
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


server = MCPServer("source-mcp-toolbox")
for position, (tool_name, spec) in enumerate(DEFINITIONS["tools"].items()):
    namespace: dict[str, Any] = {
        "_execute": _execute,
        "_spec": spec,
        "_name": tool_name,
    }
    exec(  # noqa: S102 - compiled only from validated declarative parameter names
        f"def generated_{position}({_signature(spec.get('parameters') or [])}):\n"
        f"    return _execute(_name, _spec, locals())\n",
        namespace,
    )
    server.tool(name=tool_name, description=str(spec.get("description") or ""))(
        namespace[f"generated_{position}"]
    )

application = server.streamable_http_app(
    streamable_http_path="/mcp/",
    json_response=True,
    stateless_http=True,
    host="127.0.0.1",
)


async def toolset_compatible(scope: dict[str, Any], receive: Any, send: Any) -> None:
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
    if scope.get("type") == "http" and str(scope.get("path") or "").startswith("/mcp/"):
        scope = {**scope, "path": "/mcp/", "raw_path": b"/mcp/"}
    await application(scope, receive, send)


if __name__ == "__main__":
    uvicorn.run(toolset_compatible, host="0.0.0.0", port=int(os.environ["PORT"]))
