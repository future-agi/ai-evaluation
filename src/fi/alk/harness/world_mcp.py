"""The MCP endpoint a composed release bundle serves once per world (`environment-releases.md` §6.4).

`/mcp` is stateless Streamable HTTP with JSON responses: `tools/list` is the release's tools plus
its attached servers' (`world_tools.WorldTools.tools`), and `tools/call` runs one canonical call
through `WorldTools.call`, so an MCP agent reaches exactly the tools, transactions and trace rows a
webhook agent does. `GET /healthz` answers once the process serves.

It runs from the bundle's tool-server directory (`tools.json`, `servers.json`, `handlers/`) on
the ALK interpreter, and restarts on every lease, so attached servers never carry state into the
next scenario.

    PORT=8090 DATABASE_URL=postgresql://... python -m fi.alk.harness.world_mcp
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import anyio
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from fi.alk.harness.world_tools import WorldTools, attach_servers, load_handlers

MCP_PATH = "/mcp"
HEALTH_PATH = "/healthz"


class _Endpoint:
    def __init__(self, manager: StreamableHTTPSessionManager) -> None:
        self._manager = manager

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self._manager.handle_request(scope, receive, send)


def build_app(world: WorldTools) -> Starlette:
    server: Server[Any, Any] = Server("world")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.parameters,
            )
            for tool in await anyio.to_thread.run_sync(world.tools)
        ]

    # Arguments are not validated here: every call, malformed or not, has to reach WorldTools so
    # the refusal is answered by the environment and recorded like any other call.
    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        outcome = await anyio.to_thread.run_sync(lambda: world.call(name, arguments))
        ok = outcome.status == "ok"
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=outcome.content if ok else (outcome.error or outcome.status),
                )
            ],
            isError=not ok,
        )

    manager = StreamableHTTPSessionManager(
        app=server, stateless=True, json_response=True
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with manager.run():
            try:
                yield
            finally:
                await anyio.to_thread.run_sync(world.close)

    async def health(_request: Request) -> JSONResponse:
        return JSONResponse({"ok": True})

    return Starlette(
        routes=[
            Route(MCP_PATH, endpoint=_Endpoint(manager)),
            Route(HEALTH_PATH, endpoint=health),
        ],
        lifespan=lifespan,
    )


def main() -> None:
    import uvicorn

    from fi.alk.harness.releases import load_servers, load_tools

    here = Path.cwd()
    database_url = os.environ["DATABASE_URL"]
    world = WorldTools(
        load_handlers(here / "tools.json", here / "handlers"),
        load_tools(here),
        database_url,
        attached=attach_servers(load_servers(here), database_url, cwd=here),
    )
    uvicorn.run(
        build_app(world),
        host="0.0.0.0",  # the world's own port, reached from inside the sandbox
        port=int(os.environ["PORT"]),
        log_level="warning",
    )


if __name__ == "__main__":
    main()
