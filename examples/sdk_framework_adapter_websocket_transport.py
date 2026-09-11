from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import socketserver
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from fi.alk import configure, simulate


REQUIRED_ENV = "AGENT_LEARNING_SDK_FRAMEWORK_WEBSOCKET_TRANSPORT_KEY"
ENDPOINT_ENV = "AGENT_LEARNING_SDK_FRAMEWORK_WEBSOCKET_TRANSPORT_ENDPOINT"
FRAMEWORK = "livekit"


def build_manifest(endpoint: str | None = None) -> dict[str, Any]:
    return simulate.build_framework_websocket_transport_run_manifest(
        name="sdk-framework-adapter-websocket-transport-run",
        endpoint=endpoint
        or os.environ.get(ENDPOINT_ENV)
        or "ws://127.0.0.1:8768/agent-learning/framework",
        framework=FRAMEWORK,
        required_env=[REQUIRED_ENV],
        api_key_env=REQUIRED_ENV,
        metadata={"cookbook": "sdk-framework-adapter-websocket-transport"},
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
        with _local_framework_websocket_agent(
            api_key,
            framework=FRAMEWORK,
        ) as local_endpoint:
            result = _run_manifest(local_endpoint, output_path)
    return result


def _run_manifest(endpoint: str, output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    manifest = build_manifest(endpoint)
    simulate.write_manifest_file(manifest, manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
    result["framework_websocket_transport_manifest"] = manifest

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return result


@contextmanager
def _local_framework_websocket_agent(
    api_key: str,
    *,
    framework: str,
) -> Iterator[str]:
    handler = _handler_for_key(api_key, framework=framework)
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"ws://{host}:{port}/agent-learning/framework"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _handler_for_key(
    api_key: str,
    *,
    framework: str,
) -> type[socketserver.BaseRequestHandler]:
    class FrameworkWebSocketTransportHandler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            request = _read_http_headers(self.request)
            headers = request["headers"]
            if request["path"].rstrip("/") != "/agent-learning/framework":
                _write_http_error(self.request, 404, "unknown path")
                return
            if headers.get("authorization") != f"Bearer {api_key}":
                _write_http_error(self.request, 401, "missing or invalid authorization")
                return
            key = headers.get("sec-websocket-key")
            if not key:
                _write_http_error(self.request, 400, "missing websocket key")
                return
            accept = base64.b64encode(
                hashlib.sha1(
                    (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
                ).digest()
            ).decode("ascii")
            response = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n"
                "\r\n"
            )
            self.request.sendall(response.encode("utf-8"))
            _, raw_payload = _read_ws_frame(self.request)
            try:
                payload = json.loads(raw_payload or "{}")
            except json.JSONDecodeError:
                _send_ws_text_frame(
                    self.request,
                    json.dumps({"error": {"message": "invalid json"}}),
                )
                return

            tool_names = [
                str(tool.get("name") or tool.get("tool") or "")
                for tool in payload.get("tools") or []
                if isinstance(tool, dict)
            ]
            if "framework_websocket_status" not in tool_names:
                _send_ws_text_frame(
                    self.request,
                    json.dumps(
                        {
                            "error": {
                                "message": "framework_websocket_status tool missing"
                            }
                        }
                    ),
                )
                return

            transport = _framework_websocket_transport_state(
                payload,
                framework=framework,
                endpoint_host=str(headers.get("host") or "127.0.0.1"),
            )
            trace = _framework_trace(framework)
            runtime = _framework_runtime(payload, framework=framework)
            _send_ws_text_frame(
                self.request,
                json.dumps(
                    {
                        "content": (
                            "Framework WebSocket transport verified: refund "
                            "approved, no secrets exposed, framework runtime "
                            "state preserved, framework trace artifact "
                            "preserved, and framework_websocket_status "
                            "verified."
                        ),
                        "tool_calls": [
                            {
                                "id": "call_framework_websocket_status",
                                "name": "framework_websocket_status",
                                "arguments": {
                                    "framework": framework,
                                    "transport": "websocket",
                                    "status": "verified",
                                },
                            }
                        ],
                        "state": {
                            "framework_websocket_transport": transport,
                            "framework_runtime": runtime,
                            "framework_trace": trace,
                        },
                        "metadata": {
                            "framework": framework,
                            "framework_websocket_transport": transport,
                        },
                        "artifacts": [
                            {
                                "type": "trace",
                                "role": "agent",
                                "data": trace,
                                "metadata": {
                                    "kind": "framework_trace",
                                    "framework": framework,
                                    "transport": "websocket",
                                },
                            }
                        ],
                        "events": [
                            {
                                "type": "framework_websocket_transport",
                                "name": "local_websocket_framework_request",
                                "payload": transport,
                                "metadata": {
                                    "framework": framework,
                                    "transport": "websocket",
                                    "signals": ["websocket", "transport", "latency"],
                                },
                            },
                            {
                                "type": "framework_trace",
                                "name": "framework_trace",
                                "payload": trace,
                                "metadata": {
                                    "kind": "framework_trace",
                                    "framework": framework,
                                },
                            },
                            {
                                "type": "framework_trace_span",
                                "name": "local websocket framework request",
                                "payload": trace["spans"][0],
                                "metadata": {
                                    "framework": framework,
                                    "signals": ["websocket", "transport", "latency"],
                                },
                            },
                        ],
                    },
                    sort_keys=True,
                ),
            )

    return FrameworkWebSocketTransportHandler


def _framework_websocket_transport_state(
    payload: dict[str, Any],
    *,
    framework: str,
    endpoint_host: str,
) -> dict[str, Any]:
    return {
        "kind": "agent-learning.framework-websocket-transport.v1",
        "framework": framework,
        "transport": "websocket",
        "protocol": "fi.alk",
        "endpoint_host": endpoint_host,
        "status_code": 101,
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
        "handshake": {
            "upgrade": "websocket",
            "connection": "Upgrade",
            "accepted": True,
        },
        "frame": {
            "encoding": "json",
            "request_frame_count": 1,
            "response_frame_count": 1,
        },
        "trace_context": {
            "traceparent": "00-localframeworkwebsocket-0000000000000001-01",
        },
    }


def _framework_runtime(payload: dict[str, Any], *, framework: str) -> dict[str, Any]:
    tool_count = len(payload.get("tools") or [])
    return {
        "kind": "framework_runtime",
        "framework": framework,
        "signals": ["websocket", "transport", "tool", "state"],
        "summary": {
            "invocation_count": 1,
            "methods": ["websocket"],
            "input_modes": ["json_frame"],
            "call_styles": ["request_response"],
            "error_count": 0,
        },
        "invocations": [
            {
                "id": "framework_websocket_transport_call",
                "framework": framework,
                "method": "websocket",
                "input_mode": "json_frame",
                "call_style": "request_response",
                "signals": ["websocket", "transport", "tool", "state"],
                "input": {
                    "type": "agent_learning_websocket",
                    "message_count": len(payload.get("messages") or []),
                    "tool_count": tool_count,
                },
                "output": {
                    "status": "verified",
                    "state_keys": [
                        "framework_websocket_transport",
                        "framework_runtime",
                        "framework_trace",
                    ],
                    "artifact_types": ["trace"],
                    "event_types": [
                        "framework_websocket_transport",
                        "framework_trace",
                        "framework_trace_span",
                    ],
                    "metadata_keys": ["framework_websocket_transport"],
                    "tool_names": ["framework_websocket_status"],
                },
            }
        ],
    }


def _framework_trace(framework: str) -> dict[str, Any]:
    spans = [
        {
            "id": "span-websocket-request",
            "name": "local websocket framework request",
            "kind": "client",
            "framework": framework,
            "signals": ["websocket", "transport", "latency"],
            "latency_ms": 11.0,
            "attributes": {
                "network.protocol.name": "websocket",
                "fi.alk.transport": "websocket",
            },
        },
        {
            "id": "span-model-dispatch",
            "name": f"{framework} realtime dispatch",
            "kind": "model",
            "framework": framework,
            "signals": ["model", "latency"],
            "latency_ms": 6.0,
            "attributes": {"gen_ai.operation.name": "chat"},
        },
        {
            "id": "span-tool-status",
            "name": "tool call framework_websocket_status",
            "kind": "tool",
            "framework": framework,
            "signals": ["tool", "state", "latency"],
            "latency_ms": 2.0,
            "attributes": {"tool.name": "framework_websocket_status"},
        },
    ]
    return {
        "kind": "framework_trace",
        "framework": framework,
        "transport": "websocket",
        "spans": spans,
        "tools": [{"name": "framework_websocket_status", "status": "verified"}],
        "summary": {
            "span_count": len(spans),
            "model_span_count": 1,
            "tool_span_count": 1,
            "state_span_count": 1,
            "latency_span_count": 3,
            "tool_count": 1,
            "error_count": 0,
        },
    }


def _read_http_headers(sock: Any) -> dict[str, Any]:
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
    text = data.decode("utf-8", errors="replace")
    lines = text.split("\r\n")
    request_line = lines[0] if lines else ""
    parts = request_line.split()
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            name, value = line.split(":", 1)
            headers[name.strip().lower()] = value.strip()
    return {
        "method": parts[0] if parts else "",
        "path": parts[1] if len(parts) > 1 else "",
        "headers": headers,
    }


def _write_http_error(sock: Any, status: int, message: str) -> None:
    body = json.dumps({"error": {"message": message}}).encode("utf-8")
    reason = {
        400: "Bad Request",
        401: "Unauthorized",
        404: "Not Found",
    }.get(status, "Error")
    sock.sendall(
        (
            f"HTTP/1.1 {status} {reason}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "\r\n"
        ).encode("utf-8")
        + body
    )


def _send_ws_text_frame(sock: Any, text: str) -> None:
    payload = text.encode("utf-8")
    header = bytearray([0x81])
    length = len(payload)
    if length < 126:
        header.append(length)
    elif length <= 0xFFFF:
        header.extend([126, *length.to_bytes(2, "big")])
    else:
        header.extend([127, *length.to_bytes(8, "big")])
    sock.sendall(bytes(header) + payload)


def _read_ws_frame(sock: Any) -> tuple[int, str]:
    first = _read_exact(sock, 2)
    opcode = first[0] & 0x0F
    masked = bool(first[1] & 0x80)
    length = first[1] & 0x7F
    if length == 126:
        length = int.from_bytes(_read_exact(sock, 2), "big")
    elif length == 127:
        length = int.from_bytes(_read_exact(sock, 8), "big")
    mask_key = _read_exact(sock, 4) if masked else b""
    payload = _read_exact(sock, length) if length else b""
    if masked:
        payload = bytes(byte ^ mask_key[index % 4] for index, byte in enumerate(payload))
    return opcode, payload.decode("utf-8")


def _read_exact(sock: Any, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise RuntimeError("WebSocket connection closed unexpectedly")
        data.extend(chunk)
    return bytes(data)


if __name__ == "__main__":
    destination = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts") / "sdk-framework-adapter-websocket-transport.json"
    )
    print(json.dumps(run(destination), indent=2, sort_keys=True, default=str))
