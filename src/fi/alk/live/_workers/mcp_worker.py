"""MCP lane worker (3E) — untrusted subprocess entry (P3-D1).

The client side of the MCP lane: a REAL ``ClientSession`` over the SDK's
stdio transport. It launches the target server (default: the shipped
loopback fixture ``mcp_loopback_server.py``) as ITS OWN subprocess via
``StdioServerParameters`` — the real protocol over the wire between two
separate processes — lists tools (capability hash = sha256 of the sorted
``list_tools()`` JSON, R§1 #11), runs the scenario's tool-call script with
claim-level expectations, and emits the server-behavior snapshot stamp.

IPC with the harness: see livekit_worker.py — same one-boot-line / JSONL
contract on THIS process's stdio (the MCP wire is the server subprocess's
stdio, owned by the SDK).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import traceback
from typing import Any


def _emit(channel: str, type_: str, payload: dict[str, Any]) -> None:
    print(
        json.dumps(
            {"channel": channel, "type": type_, "payload": payload},
            ensure_ascii=False,
            default=str,
        ),
        flush=True,
    )


def _read_boot() -> dict[str, Any]:
    line = sys.stdin.readline()
    if not line.strip():
        raise RuntimeError("missing boot message on stdin")
    boot = json.loads(line)
    if not isinstance(boot, dict) or boot.get("type") != "boot":
        raise RuntimeError("first stdin line must be a boot message")
    return boot


def _result_text(result: Any) -> str:
    texts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            texts.append(str(text))
    return "\n".join(texts)


async def _run(boot: dict[str, Any]) -> None:
    import importlib.metadata

    import mcp as mcp_pkg
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    version = importlib.metadata.version("mcp")
    config = boot.get("config") or {}
    command = [str(part) for part in (config.get("server_command") or [])]
    if not command:
        command = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_loopback_server.py"),
        ]
    env_names = [str(name) for name in (config.get("server_env_names") or [])]
    server_env = {name: os.environ[name] for name in env_names if name in os.environ}
    # The server inherits the kit path so the loopback fixture imports clean.
    if "PYTHONPATH" in os.environ:
        server_env.setdefault("PYTHONPATH", os.environ["PYTHONPATH"])
    if "PATH" in os.environ:
        server_env.setdefault("PATH", os.environ["PATH"])

    params = StdioServerParameters(
        command=command[0], args=command[1:], env=server_env or None
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            tools_result = await session.list_tools()
            tool_summary = sorted(
                (
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                    }
                    for tool in tools_result.tools
                ),
                key=lambda item: item["name"],
            )
            capability_hash = hashlib.sha256(
                json.dumps(tool_summary, sort_keys=True).encode("utf-8")
            ).hexdigest()
            _emit(
                "lane",
                "framework_ready",
                {
                    "framework": "mcp",
                    "framework_version": version,
                    "capability_hash": capability_hash,
                    "package_paths": [os.path.dirname(mcp_pkg.__file__)],
                },
            )
            server_info = getattr(init, "serverInfo", None)
            _emit(
                "lane",
                "server_snapshot",
                {
                    "server_name": getattr(server_info, "name", None),
                    "server_version": getattr(server_info, "version", None),
                    "capability_hash": capability_hash,
                },
            )
            checks: list[bool] = []
            for call in config.get("calls") or []:
                call = call or {}
                name = str(call.get("tool") or "")
                arguments = call.get("arguments") or {}
                _emit("tool", "tool_call", {"name": name, "arguments": arguments})
                result = await session.call_tool(name, arguments)
                text = _result_text(result)
                ok = not bool(getattr(result, "isError", False))
                # Claim-level rubric, tolerant of alternative trajectories
                # (R§1 #11): the claim is about the answer, not the path.
                expect = call.get("expect")
                if isinstance(expect, dict) and isinstance(
                    expect.get("contains"), str
                ):
                    ok = ok and expect["contains"].lower() in text.lower()
                _emit(
                    "tool",
                    "tool_result",
                    {"name": name, "ok": ok, "text": text[:2000]},
                )
                checks.append(ok)
            passed = bool(checks) and all(checks)
            _emit("lane", "verification", {"passed": passed, "checks": checks})


def main() -> int:
    boot = _read_boot()
    try:
        asyncio.run(_run(boot))
    except Exception:
        _emit("lane", "worker_error", {"traceback": traceback.format_exc()})
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
