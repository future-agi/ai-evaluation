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


REQUIRED_ENV = "AGENT_LEARNING_SDK_RETRIEVAL_HOOK_KEY"
ENDPOINT_ENV = "AGENT_LEARNING_SDK_RETRIEVAL_HOOK_ENDPOINT"


def build_manifest(endpoint: str | None = None) -> dict[str, Any]:
    return optimize.build_retrieval_hook_optimization_manifest(
        name="sdk-retrieval-hook-optimization",
        endpoint=endpoint
        or os.environ.get(ENDPOINT_ENV)
        or "http://127.0.0.1:8767/retrieval/query",
        required_env=[REQUIRED_ENV],
        api_key_env=REQUIRED_ENV,
        target_metadata={"cookbook": "sdk-retrieval-hook-optimization"},
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
        with _local_retrieval_hook(api_key) as local_endpoint:
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
    return optimize.optimize_retrieval_hooks(
        name="sdk-retrieval-hook-optimization",
        endpoint=endpoint,
        required_env=[REQUIRED_ENV],
        api_key_env=REQUIRED_ENV,
        target_metadata={"cookbook": "sdk-retrieval-hook-optimization"},
        manifest_path=Path(__file__).with_suffix(".json"),
    )


@contextmanager
def _local_retrieval_hook(api_key: str) -> Iterator[str]:
    handler = _handler_for_key(api_key)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/retrieval/query"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _handler_for_key(api_key: str) -> type[BaseHTTPRequestHandler]:
    class RetrievalHookHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/retrieval/query":
                self._write_json(404, {"content": "unknown retrieval hook"})
                return
            if self.headers.get("Authorization") != f"Bearer {api_key}":
                self._write_json(
                    401,
                    {
                        "content": "retrieval hook authorization missing",
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
                        "content": "retrieval hook received invalid json",
                        "success": False,
                        "error": "invalid json",
                    },
                )
                return

            query = str(payload.get("query") or "")
            if "refund" not in query.lower():
                self._write_json(
                    422,
                    {
                        "content": "retrieval hook rejected unsupported query",
                        "success": False,
                        "error": "unsupported query",
                    },
                )
                return

            document = {
                "id": "doc_refund_2026",
                "title": "Current refund policy",
                "content": (
                    "doc_refund_2026 states that the current 2026 refund "
                    "policy authorizes approval when the customer refund "
                    "amount is within support limits and the decision is "
                    "source grounded."
                ),
                "source": "kb://refund-policy/2026",
                "current": True,
                "version": "2026",
                "score": 0.99,
            }
            self._write_json(
                200,
                {
                    "content": (
                        "Retrieved current refund policy doc_refund_2026 with "
                        "citation evidence."
                    ),
                    "answer": (
                        "doc_refund_2026 states that the current 2026 refund "
                        "policy authorizes approval when the customer refund "
                        "amount is within support limits and the decision is "
                        "source grounded."
                    ),
                    "documents": [document],
                    "citations": [
                        {
                            "doc_ids": ["doc_refund_2026"],
                            "claim": (
                                "Refund approval is grounded in the current "
                                "2026 refund policy."
                            ),
                            "freshness_checked": True,
                        }
                    ],
                    "success": True,
                },
            )

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _write_json(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return RetrievalHookHandler


if __name__ == "__main__":
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    payload = run(output)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
