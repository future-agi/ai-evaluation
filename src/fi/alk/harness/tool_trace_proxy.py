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
        try:
            with request.urlopen(outgoing, timeout=30) as response:
                content = response.read()
                status = int(response.status)
                response_type = response.headers.get("content-type", "application/json")
            try:
                result = json.loads(content) if content else None
            except ValueError:
                result = content.decode("utf-8", errors="replace")
            _record(name, arguments, result, status < 400)
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


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
