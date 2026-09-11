from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

from fi.alk import configure, simulate


REQUIRED_ENV = "AGENT_LEARNING_SDK_FRAMEWORK_HTTP_TRANSPORT_KEY"
ENDPOINT_ENV = "AGENT_LEARNING_SDK_FRAMEWORK_HTTP_TRANSPORT_ENDPOINT"
FRAMEWORK = "langgraph"


def build_manifest(endpoint: str | None = None) -> dict[str, Any]:
    return simulate.build_framework_http_transport_run_manifest(
        name="sdk-framework-adapter-http-transport-run",
        endpoint=endpoint
        or os.environ.get(ENDPOINT_ENV)
        or "http://127.0.0.1:8767/agent-learning/framework",
        framework=FRAMEWORK,
        required_env=[REQUIRED_ENV],
        api_key_env=REQUIRED_ENV,
        metadata={"cookbook": "sdk-framework-adapter-http-transport"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)

    endpoint = os.environ.get(ENDPOINT_ENV)
    if endpoint:
        result = _run_manifest(endpoint, output_path)
    else:
        with _local_framework_http_agent(api_key, framework=FRAMEWORK) as local_endpoint:
            result = _run_manifest(local_endpoint, output_path)
    return result


def _run_manifest(endpoint: str, output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    manifest = build_manifest(endpoint)
    simulate.write_manifest_file(manifest, manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
    result["framework_http_transport_manifest"] = manifest

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return result


@contextmanager
def _local_framework_http_agent(
    api_key: str,
    *,
    framework: str,
) -> Iterator[str]:
    handler = _handler_for_key(api_key, framework=framework)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/agent-learning/framework"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _handler_for_key(api_key: str, *, framework: str) -> type[BaseHTTPRequestHandler]:
    class FrameworkHTTPTransportHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.headers.get("Authorization") != f"Bearer {api_key}":
                self._write_json(
                    401,
                    {"error": {"message": "missing or invalid authorization"}},
                )
                return

            if self.path.rstrip("/") != "/agent-learning/framework":
                self._write_json(404, {"error": {"message": "unknown path"}})
                return

            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length).decode("utf-8")
            try:
                payload = json.loads(raw or "{}")
            except json.JSONDecodeError:
                self._write_json(400, {"error": {"message": "invalid json"}})
                return

            tool_names = [
                str(tool.get("name") or tool.get("tool") or "")
                for tool in payload.get("tools") or []
                if isinstance(tool, dict)
            ]
            if "framework_http_status" not in tool_names:
                self._write_json(
                    400,
                    {"error": {"message": "framework_http_status tool missing"}},
                )
                return

            transport = _framework_http_transport_state(
                payload,
                framework=framework,
                endpoint_host=str(self.headers.get("Host") or "127.0.0.1"),
            )
            trace = _framework_trace(framework)
            runtime = _framework_runtime(payload, framework=framework)
            self._write_json(
                200,
                {
                    "content": (
                        "Framework HTTP transport verified: refund approved, "
                        "no secrets exposed, and framework_http_status verified."
                    ),
                    "tool_calls": [
                        {
                            "id": "call_framework_http_status",
                            "name": "framework_http_status",
                            "arguments": {
                                "framework": framework,
                                "transport": "http",
                                "status": "verified",
                            },
                        }
                    ],
                    "state": {
                        "framework_http_transport": transport,
                        "framework_runtime": runtime,
                        "framework_trace": trace,
                    },
                    "metadata": {
                        "framework": framework,
                        "framework_http_transport": transport,
                    },
                    "artifacts": [
                        {
                            "type": "trace",
                            "role": "agent",
                            "data": trace,
                            "metadata": {
                                "kind": "framework_trace",
                                "framework": framework,
                                "transport": "http",
                            },
                        }
                    ],
                    "events": [
                        {
                            "type": "framework_http_transport",
                            "name": "local_http_framework_request",
                            "payload": transport,
                            "metadata": {
                                "framework": framework,
                                "transport": "http",
                                "signals": ["http", "transport", "latency"],
                            },
                        },
                        {
                            "type": "framework_trace",
                            "name": "framework_trace",
                            "payload": trace,
                            "metadata": {"kind": "framework_trace", "framework": framework},
                        },
                        {
                            "type": "framework_trace_span",
                            "name": "local http framework request",
                            "payload": trace["spans"][0],
                            "metadata": {
                                "framework": framework,
                                "signals": ["http", "transport", "latency"],
                            },
                        },
                    ],
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

    return FrameworkHTTPTransportHandler


def _framework_http_transport_state(
    payload: dict[str, Any],
    *,
    framework: str,
    endpoint_host: str,
) -> dict[str, Any]:
    return {
        "kind": "agent-learning.framework-http-transport.v1",
        "framework": framework,
        "transport": "http",
        "protocol": "fi.alk",
        "endpoint_host": endpoint_host,
        "status_code": 200,
        "success": True,
        "requires_external_service": False,
        "auth": {
            "mode": "bearer",
            "api_key_env": REQUIRED_ENV,
            "redacted": True,
        },
        "request": {
            "message_count": len(payload.get("messages") or []),
            "tool_count": len(payload.get("tools") or []),
        },
        "trace_context": {
            "traceparent": "00-localframeworkhttptransport-0000000000000001-01",
        },
    }


def _framework_runtime(payload: dict[str, Any], *, framework: str) -> dict[str, Any]:
    tool_count = len(payload.get("tools") or [])
    return {
        "kind": "framework_runtime",
        "framework": framework,
        "signals": ["http", "transport", "tool", "state"],
        "summary": {
            "invocation_count": 1,
            "methods": ["http"],
            "input_modes": ["json"],
            "call_styles": ["request_response"],
            "error_count": 0,
        },
        "invocations": [
            {
                "id": "framework_http_transport_call",
                "framework": framework,
                "method": "http",
                "input_mode": "json",
                "call_style": "request_response",
                "signals": ["http", "transport", "tool", "state"],
                "input": {
                    "type": "agent_learning_http",
                    "message_count": len(payload.get("messages") or []),
                    "tool_count": tool_count,
                },
                "output": {
                    "type": "agent_response",
                    "tool_call_count": 1,
                    "tool_names": ["framework_http_status"],
                    "artifact_count": 1,
                    "artifact_types": ["trace"],
                    "event_count": 3,
                    "event_types": [
                        "framework_http_transport",
                        "framework_trace",
                        "framework_trace_span",
                    ],
                    "state_keys": [
                        "framework_http_transport",
                        "framework_runtime",
                        "framework_trace",
                    ],
                    "metadata_keys": [
                        "framework_http_transport",
                        "external_agent_trace",
                    ],
                    "streaming": False,
                },
            }
        ],
    }


def _framework_trace(framework: str) -> dict[str, Any]:
    spans = [
        {
            "id": "local_http_framework_request",
            "name": "local http framework request",
            "type": "transport",
            "latency_ms": 7,
            "signals": ["http", "transport", "latency"],
            "attributes": {
                "http.method": "POST",
                "http.route": "/agent-learning/framework",
                "transport": "http",
            },
        },
        {
            "id": f"{framework}_model_dispatch",
            "name": f"{framework} model dispatch",
            "type": "model",
            "latency_ms": 12,
            "signals": ["model", "latency"],
            "attributes": {"framework": framework, "node": "refund_decision"},
        },
        {
            "id": "tool_call_framework_http_status",
            "name": "tool call framework_http_status",
            "type": "tool",
            "latency_ms": 3,
            "signals": ["tool", "state"],
            "attributes": {
                "tool_name": "framework_http_status",
                "state_key": "framework_http_status",
            },
        },
    ]
    return {
        "kind": "framework_trace",
        "framework": framework,
        "signals": ["http", "transport", "model", "tool", "state", "latency"],
        "spans": spans,
        "summary": {
            "span_count": len(spans),
            "model_span_count": 1,
            "tool_span_count": 1,
            "state_span_count": 1,
            "latency_span_count": 3,
            "tool_count": 1,
            "error_count": 0,
            "signals": ["http", "transport", "model", "tool", "state", "latency"],
            "tool_names": ["framework_http_status"],
            "span_names": [span["name"] for span in spans],
        },
    }


if __name__ == "__main__":
    destination = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts") / "sdk-framework-adapter-http-transport.json"
    )
    run(destination)
