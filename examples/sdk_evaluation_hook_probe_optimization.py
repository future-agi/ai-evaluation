from __future__ import annotations

import asyncio
import json
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

from fi.alk import optimize, simulate


def build_probe_optimization(endpoint: str) -> dict[str, Any]:
    return optimize.optimize_evaluation_hook_probe(
        name="sdk-evaluation-hook-probe-optimization",
        endpoint=endpoint,
        metadata={"cookbook": "sdk-evaluation-hook-probe-optimization"},
    )


def build_manifest(endpoint: str) -> dict[str, Any]:
    return optimize.build_evaluation_hook_run_manifest_from_probe_optimization(
        build_probe_optimization(endpoint),
        endpoint=endpoint,
        name="sdk-evaluation-hook-probe-promotion-run",
        metadata={"cookbook": "sdk-evaluation-hook-probe-optimization"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    with _local_evaluation_hook() as endpoint:
        manifest_path = output.with_suffix(".manifest.json")
        simulate.write_manifest_file(build_manifest(endpoint), manifest_path)
        result = asyncio.run(simulate.run_manifest_file(manifest_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return result


@contextmanager
def _local_evaluation_hook() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EvaluationHookHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/eval/task"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


class _EvaluationHookHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/eval/task":
            self._write_json(404, {"content": "unknown evaluation hook"})
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
                            "Local task judge accepted policy-grounded answer."
                            if score == 1.0
                            else "Local task judge rejected incomplete answer."
                        ),
                        "details": {
                            "matched_terms": matched,
                            "missing_terms": [
                                term for term in required_terms if term not in matched
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


if __name__ == "__main__":
    destination = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts") / "sdk-evaluation-hook-probe-optimization.json"
    )
    run(destination)
