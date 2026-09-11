"""HTTP bridge embedded into hosted bundles for ``fi.simulate`` callbacks.

The bundle producer serializes this module into the process command.  It deliberately uses only
the standard library plus the submitted agent's own ``fi.simulate`` dependency, so a callback-only
repository does not need to be modified or taught about the hosted process runtime.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _load_callback() -> Any:
    target = os.environ["ALK_CALLBACK_ENTRYPOINT"]
    module_name, separator, attribute = target.partition(":")
    if not separator or not module_name or not attribute:
        raise RuntimeError(f"invalid ALK_CALLBACK_ENTRYPOINT: {target!r}")
    callback = getattr(importlib.import_module(module_name), attribute)
    if not callable(callback):
        raise RuntimeError(f"callback entrypoint is not callable: {target!r}")
    return callback


CALLBACK = _load_callback()


def _json_body(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        body = value.model_dump(mode="json", exclude_none=True)
    elif isinstance(value, dict):
        body = value
    else:
        body = {"content": str(value)}
    if not isinstance(body, dict):
        raise TypeError("callback response must serialize to a JSON object")
    body.setdefault("content", "")
    return body


def _invoke(payload: dict[str, Any]) -> dict[str, Any]:
    from fi.simulate import AgentInput

    result = CALLBACK(AgentInput.model_validate(payload))
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    return _json_body(result)


class Handler(BaseHTTPRequestHandler):
    def _respond(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.rstrip("/") == "/health":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
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
        except Exception as exc:  # noqa: BLE001 - target errors cross the HTTP seam
            error_type = type(exc).__name__
            self._respond(
                500,
                {"error": f"{type(exc).__name__}: {exc}", "content": ""},
            )
        finally:
            elapsed_ms = round((time.monotonic() - started) * 1000)
            print(
                "callback_http_request "
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
