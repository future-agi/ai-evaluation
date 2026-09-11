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


REQUIRED_ENV = "AGENT_LEARNING_SDK_EXTERNAL_HTTP_AGENT_KEY"
ENDPOINT_ENV = "AGENT_LEARNING_SDK_EXTERNAL_HTTP_AGENT_ENDPOINT"


def build_manifest(endpoint: str | None = None) -> dict[str, Any]:
    return optimize.build_external_agent_adapter_optimization_manifest(
        name="sdk-external-http-agent-optimization",
        endpoint=endpoint
        or os.environ.get(ENDPOINT_ENV)
        or "http://127.0.0.1:8765/v1/chat/completions",
        required_env=[REQUIRED_ENV],
        api_key_env=REQUIRED_ENV,
        target_metadata={"cookbook": "sdk-external-http-agent-optimization"},
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
        with _local_openai_compatible_agent(api_key) as local_endpoint:
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
    return optimize.optimize_external_agent_adapter(
        name="sdk-external-http-agent-optimization",
        endpoint=endpoint,
        required_env=[REQUIRED_ENV],
        api_key_env=REQUIRED_ENV,
        target_metadata={"cookbook": "sdk-external-http-agent-optimization"},
        manifest_path=Path(__file__).with_suffix(".json"),
    )


@contextmanager
def _local_openai_compatible_agent(api_key: str) -> Iterator[str]:
    handler = _handler_for_key(api_key)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/v1/chat/completions"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _handler_for_key(api_key: str) -> type[BaseHTTPRequestHandler]:
    class ExternalAgentHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.headers.get("Authorization") != f"Bearer {api_key}":
                self._write_json(
                    401,
                    {"error": {"message": "missing or invalid authorization"}},
                )
                return

            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length).decode("utf-8")
            try:
                payload = json.loads(raw or "{}")
            except json.JSONDecodeError:
                self._write_json(400, {"error": {"message": "invalid json"}})
                return

            if self.path.rstrip("/") != "/v1/chat/completions":
                self._write_json(404, {"error": {"message": "unknown path"}})
                return

            if _has_openai_tool_schema(payload, "external_agent_status"):
                self._write_json(
                    200,
                    {
                        "id": "chatcmpl-local-external-agent",
                        "object": "chat.completion",
                        "model": payload.get("model")
                        or "agent-learning-local-http-target",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "tool_calls",
                                "message": {
                                    "role": "assistant",
                                    "content": (
                                        "First, because this is an external "
                                        "HTTP OpenAI-compatible agent, I "
                                        "preserve auth boundaries, collect a "
                                        "redacted trace, and verify tool "
                                        "evidence. Therefore the policy "
                                        "answer is complete: refund approved, "
                                        "no secrets exposed, and "
                                        "external_agent_status verified for "
                                        "the endpoint."
                                    ),
                                    "tool_calls": [
                                        {
                                            "id": "call_external_agent_status",
                                            "type": "function",
                                            "function": {
                                                "name": "external_agent_status",
                                                "arguments": json.dumps(
                                                    {
                                                        "status": "verified",
                                                        "protocol": "openai_chat",
                                                    }
                                                ),
                                            },
                                        }
                                    ],
                                },
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 32,
                            "completion_tokens": 18,
                            "total_tokens": 50,
                        },
                    },
                )
                return

            if isinstance(payload.get("messages"), list):
                self._write_json(
                    200,
                    {
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": (
                                        "Policy answer: refund approved, but "
                                        "tool verification was not requested."
                                    ),
                                },
                            }
                        ]
                    },
                )
                return

            self._write_json(
                200,
                {
                    "content": (
                        "Raw adapter reached endpoint but missed the "
                        "OpenAI-compatible tool-call contract."
                    )
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

    return ExternalAgentHandler


def _has_openai_tool_schema(payload: dict[str, Any], name: str) -> bool:
    for item in payload.get("tools") or []:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if isinstance(function, dict) and function.get("name") == name:
            return True
    return False


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    payload = run(destination)
    if destination is None:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
