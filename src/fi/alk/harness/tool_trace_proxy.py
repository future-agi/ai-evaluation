"""Bundle-owned observable HTTP tool proxy used by hosted process environments."""

from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request

import psycopg

PORT = int(os.environ["PORT"])
UPSTREAM = os.environ["UPSTREAM_URL"].rstrip("/")
DATABASE_URL = os.environ["DATABASE_URL"]


def _record(
    name: str, arguments: object, result: object, ok: bool, failure: str = ""
) -> None:
    try:
        with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
            connection.execute(
                "INSERT INTO _alk_tool_trace(name, arguments, result, ok, error, at) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    name,
                    json.dumps(arguments),
                    json.dumps(result),
                    ok,
                    failure,
                    time.time(),
                ),
            )
    except Exception:
        # Evidence persistence must never alter the target tool response.
        pass


# How long a timed-out response holds the agent before it is told the upstream timed out.
FAULT_TIMEOUT_SECONDS = float(os.environ.get("ALK_FAULT_TIMEOUT_SECONDS") or 10)


def _fault_for(name: str) -> dict[str, object] | None:
    """The fault a bound sub-actor applies to this call of this tool, counted from 1, or none."""
    try:
        with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
            faults = connection.execute(
                "SELECT kind, on_call, latency_ms, content, sub_actor FROM _alk_faults WHERE tool = %s",
                (name,),
            ).fetchall()
            if not faults:
                return None
            made = connection.execute(
                "SELECT count(*) FROM _alk_tool_trace WHERE name = %s", (name,)
            ).fetchone()[0]
    except Exception:
        return None
    number = int(made) + 1
    for kind, on_call, latency_ms, content, sub_actor in faults:
        if int(on_call) == number:
            return {
                "kind": kind,
                "latency_ms": int(latency_ms or 0),
                "content": content or "",
                "label": f"fault:{kind}:{sub_actor}",
            }
    return None


def _with_content(result: object, content: str) -> object:
    if isinstance(result, dict):
        return {**result, "note": content}
    if isinstance(result, list):
        return [*result, content]
    return f"{result} {content}".strip() if result is not None else content


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            return
        self._forward()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._forward()

    def _forward(self) -> None:
        size = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(size) if size else b""
        try:
            arguments = json.loads(body) if body else {}
        except ValueError:
            arguments = {"_raw": body.decode("utf-8", errors="replace")}
        outgoing = request.Request(
            UPSTREAM + self.path,
            data=body if self.command != "GET" else None,
            method=self.command,
            headers={
                "content-type": self.headers.get("content-type", "application/json")
            },
        )
        name = self.path.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1] or "unknown"
        fault = _fault_for(name)
        if fault and fault["kind"] in ("timeout", "error", "rate_limit"):
            if fault["kind"] == "timeout":
                time.sleep(FAULT_TIMEOUT_SECONDS)
            status = {"timeout": 504, "error": 500, "rate_limit": 429}[fault["kind"]]
            detail = {"timeout": "upstream timed out", "error": "upstream error", "rate_limit": "too many requests"}
            _record(name, arguments, None, False, str(fault["label"]))
            self._reply(status, json.dumps({"detail": detail[fault["kind"]]}).encode())
            return
        if fault and fault["kind"] == "latency":
            time.sleep(int(fault["latency_ms"]) / 1000)
        try:
            with request.urlopen(outgoing, timeout=30) as response:
                content = response.read()
                status = int(response.status)
                response_type = response.headers.get("content-type", "application/json")
            try:
                result = json.loads(content) if content else None
            except ValueError:
                result = content.decode("utf-8", errors="replace")
            if fault and fault["kind"] == "malformed":
                _record(name, arguments, None, False, str(fault["label"]))
                self._reply(status, content[: max(1, len(content) // 2)] or b"{", response_type)
                return
            if fault and fault["kind"] == "content":
                result = _with_content(result, str(fault["content"]))
                content = json.dumps(result).encode()
                response_type = "application/json"
            _record(name, arguments, result, status < 400, str(fault["label"]) if fault else "")
            self.send_response(status)
            self.send_header("content-type", response_type)
            self.send_header("content-length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except error.HTTPError as exc:
            content = exc.read()
            failure = content.decode("utf-8", errors="replace")[:2000]
            _record(name, arguments, None, False, failure)
            self.send_response(exc.code)
            self.send_header("content-length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as exc:
            _record(name, arguments, None, False, f"{type(exc).__name__}: unavailable")
            content = json.dumps({"detail": "tool_upstream_unavailable"}).encode()
            self.send_response(502)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    def _reply(self, status: int, content: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
