from __future__ import annotations

import json
import os
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

from fi.alk import configure, optimize


REQUIRED_ENV = "AGENT_LEARNING_SDK_WORKFLOW_HOOK_KEY"
ENDPOINT_ENV = "AGENT_LEARNING_SDK_WORKFLOW_HOOK_ENDPOINT"


def build_manifest(endpoint: str | None = None) -> dict[str, Any]:
    return optimize.build_workflow_hook_optimization_manifest(
        name="sdk-workflow-hook-optimization",
        endpoint=endpoint
        or os.environ.get(ENDPOINT_ENV)
        or "http://127.0.0.1:8766/workflow/refund",
        required_env=[REQUIRED_ENV],
        api_key_env=REQUIRED_ENV,
        target_metadata={"cookbook": "sdk-workflow-hook-optimization"},
    )


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)

    endpoint = os.environ.get(ENDPOINT_ENV)
    if endpoint:
        result = _run_optimizer(endpoint)
    else:
        with _local_workflow_hook(api_key) as local_endpoint:
            result = _run_optimizer(local_endpoint)

    if output_path is not None:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return result


def _run_optimizer(endpoint: str) -> dict[str, Any]:
    return optimize.optimize_workflow_hooks(
        name="sdk-workflow-hook-optimization",
        endpoint=endpoint,
        required_env=[REQUIRED_ENV],
        api_key_env=REQUIRED_ENV,
        target_metadata={"cookbook": "sdk-workflow-hook-optimization"},
        manifest_path=Path(__file__).with_suffix(".json"),
    )


@contextmanager
def _local_workflow_hook(api_key: str) -> Iterator[str]:
    handler = _handler_for_key(api_key)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/workflow/refund"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _handler_for_key(api_key: str) -> type[BaseHTTPRequestHandler]:
    class WorkflowHookHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/workflow/refund":
                self._write_json(404, {"content": "unknown workflow hook"})
                return
            if self.headers.get("Authorization") != f"Bearer {api_key}":
                self._write_json(
                    401,
                    {
                        "content": "workflow hook authorization missing",
                        "success": False,
                        "error": "missing authorization",
                    },
                )
                return

            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length).decode("utf-8")
            try:
                payload = json.loads(raw or "{}")
            except json.JSONDecodeError:
                self._write_json(
                    400,
                    {
                        "content": "workflow hook received invalid json",
                        "success": False,
                        "error": "invalid json",
                    },
                )
                return

            arguments = payload.get("arguments") or {}
            if arguments.get("action") != "approve_refund":
                self._write_json(
                    422,
                    {
                        "content": "workflow hook rejected unsupported action",
                        "success": False,
                        "error": "unsupported action",
                    },
                )
                return

            self._write_json(
                200,
                {
                    "content": (
                        "Workflow hook completed refund approval with "
                        "approval_id wf_refund_2026. Auth redacted and "
                        "audited."
                    ),
                    "success": True,
                    "result": {
                        "status": "completed",
                        "approval_id": "wf_refund_2026",
                        "auth_redacted": True,
                        "amount": arguments.get("amount"),
                    },
                    "state_updates": {
                        "refund_workflow": {
                            "status": "completed",
                            "approval_id": "wf_refund_2026",
                            "auth_redacted": True,
                            "amount": arguments.get("amount"),
                        }
                    },
                    "metadata": {
                        "workflow": "refund",
                        "audit_log": "redacted-local-audit",
                    },
                },
            )

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _write_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return WorkflowHookHandler


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    payload = run(destination)
    if destination is None:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
