"""HTTP bridge for submitted one-shot conversational commands.

This is the lowest-common-denominator runtime boundary for a repository that ships a runnable
command but no server or importable ``agent_callback``.  The submitted command remains the code
under test.  Each turn is supplied both as JSON on stdin and through explicit environment
variables, and its stdout is returned as the assistant response.

Repositories can opt into the richer callback or HTTP contracts without using this adapter.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _command() -> list[str]:
    value = json.loads(os.environ["ALK_SUBPROCESS_COMMAND"])
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item and "\x00" not in item for item in value
    ):
        raise RuntimeError("ALK_SUBPROCESS_COMMAND must be a non-empty JSON argv array")
    return value


COMMAND = _command()


def _response_text(stdout: str, stderr: str) -> str:
    cleaned = _ANSI.sub("", stdout).strip()
    if cleaned:
        return cleaned[-20_000:]
    detail = _ANSI.sub("", stderr).strip()
    if detail:
        raise RuntimeError(f"submitted command produced no stdout: {detail[-2000:]}")
    raise RuntimeError("submitted command produced no response")


def _failure_output(output: str, *, limit: int) -> str:
    # Keep whole lines so the HTTP boundary can redact complete credential values.
    remaining = limit
    tail = []
    for line in reversed(_ANSI.sub("", output).splitlines(keepends=True)):
        if len(line) > limit:
            line = "[overlong output line omitted]\n"
        if len(line) > remaining:
            break
        tail.append(line)
        remaining -= len(line)
    return "".join(reversed(tail))


def _invoke(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages")
    messages = messages if isinstance(messages, list) else []
    new_message = payload.get("new_message")
    new_message = new_message if isinstance(new_message, dict) else {}
    utterance = str(new_message.get("content") or "")
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    environment = {
        **os.environ,
        "ALK_AGENT_INPUT_JSON": encoded,
        "ALK_USER_MESSAGE": utterance,
        "ALK_MESSAGES_JSON": json.dumps(messages, separators=(",", ":"), ensure_ascii=False),
    }
    timeout = float(os.environ.get("ALK_SUBPROCESS_TIMEOUT_SECONDS", "110"))
    completed = subprocess.run(  # noqa: S603 - argv is sealed in the bundle manifest
        COMMAND,
        input=encoded,
        text=True,
        capture_output=True,
        env=environment,
        timeout=timeout,
        check=False,
    )
    if completed.returncode:
        streams = [
            (name, output)
            for name, output in (("stdout", completed.stdout), ("stderr", completed.stderr))
            if output
        ]
        detail = "\n".join(
            f"{name}: {_failure_output(output, limit=2000 // len(streams))}"
            for name, output in streams
        ) or "no stdout or stderr"
        raise RuntimeError(
            f"submitted command exited {completed.returncode}: {detail}"
        )
    return {"content": _response_text(completed.stdout, completed.stderr)}


class Handler(BaseHTTPRequestHandler):
    def _respond(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/invoke":
            self._respond(404, {"error": "not_found"})
            return
        started = time.monotonic()
        status = 500
        error_type = "none"
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise TypeError("request body must be a JSON object")
            self._respond(200, _invoke(payload))
            status = 200
        except Exception as exc:  # noqa: BLE001 - target failures cross the HTTP seam
            error_type = type(exc).__name__
            self._respond(500, {"error": f"{type(exc).__name__}: {exc}", "content": ""})
        finally:
            elapsed_ms = round((time.monotonic() - started) * 1000)
            print(
                "subprocess_http_request "
                f"status={status} elapsed_ms={elapsed_ms} error_type={error_type}",
                file=sys.stderr,
                flush=True,
            )

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
