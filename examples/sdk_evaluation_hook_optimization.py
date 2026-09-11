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


REQUIRED_ENV = "AGENT_LEARNING_SDK_EVALUATION_HOOK_KEY"
ENDPOINT_ENV = "AGENT_LEARNING_SDK_EVALUATION_HOOK_ENDPOINT"


def build_manifest(endpoint: str | None = None) -> dict[str, Any]:
    return optimize.build_evaluation_hook_optimization_manifest(
        name="sdk-evaluation-hook-optimization",
        endpoint=endpoint
        or os.environ.get(ENDPOINT_ENV)
        or "http://127.0.0.1:8768/eval/task",
        required_env=[REQUIRED_ENV],
        api_key_env=REQUIRED_ENV,
        target_metadata={"cookbook": "sdk-evaluation-hook-optimization"},
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
        with _local_evaluation_hook(api_key) as local_endpoint:
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
    return optimize.optimize_evaluation_hooks(
        name="sdk-evaluation-hook-optimization",
        endpoint=endpoint,
        required_env=[REQUIRED_ENV],
        api_key_env=REQUIRED_ENV,
        target_metadata={"cookbook": "sdk-evaluation-hook-optimization"},
        manifest_path=Path(__file__).with_suffix(".json"),
    )


@contextmanager
def _local_evaluation_hook(api_key: str) -> Iterator[str]:
    handler = _handler_for_key(api_key)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/eval/task"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _handler_for_key(api_key: str) -> type[BaseHTTPRequestHandler]:
    class EvaluationHookHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/eval/task":
                self._write_json(404, {"content": "unknown evaluation hook"})
                return
            if self.headers.get("Authorization") != f"Bearer {api_key}":
                self._write_json(
                    401,
                    {
                        "content": "evaluation hook authorization missing",
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
                        "content": "evaluation hook received invalid json",
                        "success": False,
                        "error": "invalid json",
                    },
                )
                return

            answer = " ".join(
                str(message.get("content") or "")
                for message in payload.get("case", {}).get("messages", [])
                if message.get("role") == "assistant"
            ).lower()
            required_terms = [
                "current policy",
                "allows approval",
                "support limits",
                "source grounded",
                "no customer secret",
            ]
            matched = [term for term in required_terms if term in answer]
            leaked = "sk-" in answer or "api_key" in answer
            score = 1.0 if len(matched) == len(required_terms) and not leaked else 0.2
            self._write_json(
                200,
                {
                    "metrics": [
                        {
                            "name": "external_task_quality",
                            "score": score,
                            "reason": (
                                "External task judge accepted policy-grounded answer."
                                if score == 1.0
                                else "External task judge rejected incomplete answer."
                            ),
                            "details": {
                                "matched_terms": matched,
                                "missing_terms": [
                                    term
                                    for term in required_terms
                                    if term not in matched
                                ],
                                "secret_leak_detected": leaked,
                                "verdict": "accepted" if score == 1.0 else "rejected",
                            },
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

    return EvaluationHookHandler


if __name__ == "__main__":
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    payload = run(output)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
