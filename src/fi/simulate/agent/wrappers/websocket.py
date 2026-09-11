from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import socket
import time
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

from fi.simulate.agent.wrapper import (
    AgentInput,
    AgentResponse,
    SimulationArtifact,
    SimulationEvent,
)
from fi.simulate.agent.wrapper import AgentWrapper
from fi.simulate.agent.wrappers.http import (
    _artifact_list,
    _content_text,
    _event_list,
    _optional_mapping,
    _redacted_endpoint,
    _tool_call_list,
    _tool_response_list,
)


class WebSocketAgentWrapper(AgentWrapper):
    """WebSocket target adapter for local framework transport simulation."""

    def __init__(
        self,
        *,
        endpoint: str,
        protocol: str = "fi.alk",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: float = 30.0,
        include_tools: bool = True,
        system_prompt: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint is required")
        self.endpoint = endpoint
        self.protocol = _normalize_protocol(protocol)
        self.model = model
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.headers = {str(k): str(v) for k, v in dict(headers or {}).items()}
        self.timeout = float(timeout)
        self.include_tools = bool(include_tools)
        self.system_prompt = system_prompt
        self.metadata = dict(metadata or {})

    async def call(self, input: AgentInput) -> AgentResponse:
        started = time.time()
        request_payload = self._request_payload(input)
        headers = self._request_headers()
        status_code = 0
        response_payload: dict[str, Any] = {}
        error: Optional[str] = None
        try:
            status_code, response_payload = await asyncio.to_thread(
                self._send_json,
                request_payload,
                headers,
            )
            response = self._agent_response_from_payload(response_payload)
        except Exception as exc:
            error = str(exc)
            response = AgentResponse(content=f"WebSocket target failed: {exc}")

        latency_ms = round((time.time() - started) * 1000, 4)
        parsed = urlparse(self.endpoint)
        trace = {
            "kind": "external_agent_websocket_trace",
            "protocol": self.protocol,
            "endpoint": _redacted_endpoint(self.endpoint),
            "endpoint_host": parsed.netloc,
            "model": self.model,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "request_message_count": len(input.messages),
            "request_tool_count": len(input.tools) if self.include_tools else 0,
            "response_tool_call_count": len(response.tool_calls or []),
            "success": error is None and status_code == 101,
            "request_header_names": sorted(headers),
            "auth": {
                "mode": "bearer" if self._resolved_api_key() else "none",
                "api_key_env": self.api_key_env,
                "redacted": bool(self._resolved_api_key()),
            },
            "error": error,
            **self.metadata,
        }
        response.events.append(
            SimulationEvent(
                type="external_agent",
                name="external_agent_websocket_call",
                payload=trace,
            )
        )
        response.artifacts.append(
            SimulationArtifact(
                type="trace",
                role="agent",
                data=trace,
                metadata={"kind": "external_agent_websocket_trace"},
            )
        )
        state = dict(response.state or {})
        state["external_agent"] = trace
        state["external_agent_trace"] = trace
        response.state = state
        metadata = dict(response.metadata or {})
        metadata["external_agent"] = trace
        metadata["external_agent_trace"] = trace
        response.metadata = metadata
        return response

    def _request_payload(self, input: AgentInput) -> dict[str, Any]:
        messages = list(input.messages)
        if self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}, *messages]
        return {
            "thread_id": input.thread_id,
            "execution_id": input.execution_id,
            "turn_index": input.turn_index,
            "scenario_name": input.scenario_name,
            "persona": input.persona,
            "situation": input.situation,
            "expected_outcome": input.expected_outcome,
            "messages": messages,
            "new_message": input.new_message,
            "tools": list(input.tools) if self.include_tools else [],
            "metadata": input.metadata,
        }

    def _request_headers(self) -> dict[str, str]:
        headers = dict(self.headers)
        api_key = self._resolved_api_key()
        if api_key and not any(key.lower() == "authorization" for key in headers):
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _resolved_api_key(self) -> str:
        if self.api_key not in (None, ""):
            return str(self.api_key)
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return ""

    def _send_json(
        self,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> tuple[int, dict[str, Any]]:
        parsed = urlparse(self.endpoint)
        if parsed.scheme != "ws":
            raise ValueError("WebSocketAgentWrapper currently supports ws:// endpoints")
        host = parsed.hostname or ""
        port = int(parsed.port or 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        with socket.create_connection((host, port), timeout=self.timeout) as sock:
            sock.settimeout(self.timeout)
            status = self._handshake(
                sock,
                host=host,
                port=port,
                path=path,
                key=key,
                headers=headers,
            )
            _send_text_frame(sock, json.dumps(payload, default=str), mask=True)
            opcode, text = _read_frame(sock)
            if opcode == 8:
                raise ValueError("WebSocket target closed before returning JSON")
            try:
                _send_close_frame(sock)
            except OSError:
                pass
        try:
            response = json.loads(text or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"WebSocket target returned non-JSON response: {exc}") from exc
        if not isinstance(response, dict):
            raise ValueError("WebSocket target response must be a JSON object")
        return status, response

    def _handshake(
        self,
        sock: socket.socket,
        *,
        host: str,
        port: int,
        path: str,
        key: str,
        headers: Mapping[str, str],
    ) -> int:
        request_headers = {
            "Host": f"{host}:{port}",
            "Upgrade": "websocket",
            "Connection": "Upgrade",
            "Sec-WebSocket-Key": key,
            "Sec-WebSocket-Version": "13",
            **dict(headers),
        }
        request = "GET " + path + " HTTP/1.1\r\n" + "\r\n".join(
            f"{name}: {value}" for name, value in request_headers.items()
        ) + "\r\n\r\n"
        sock.sendall(request.encode("utf-8"))
        response = _read_until(sock, b"\r\n\r\n").decode("utf-8", errors="replace")
        lines = response.split("\r\n")
        status_line = lines[0] if lines else ""
        parts = status_line.split()
        status = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
        response_headers: dict[str, str] = {}
        for line in lines[1:]:
            if ":" in line:
                name, value = line.split(":", 1)
                response_headers[name.strip().lower()] = value.strip()
        expected_accept = base64.b64encode(
            hashlib.sha1(
                (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
            ).digest()
        ).decode("ascii")
        if status != 101:
            raise ValueError(f"WebSocket target returned status {status}")
        if response_headers.get("sec-websocket-accept") != expected_accept:
            raise ValueError("WebSocket target returned invalid accept key")
        return status

    def _agent_response_from_payload(self, payload: Mapping[str, Any]) -> AgentResponse:
        return AgentResponse(
            content=_content_text(payload.get("content") or payload.get("message")),
            tool_calls=_tool_call_list(payload.get("tool_calls")),
            tool_responses=_tool_response_list(payload.get("tool_responses")),
            artifacts=_artifact_list(payload.get("artifacts")),
            events=_event_list(payload.get("events")),
            memory_updates=_optional_mapping(payload.get("memory_updates")),
            state=_optional_mapping(payload.get("state")),
            metadata=_optional_mapping(payload.get("metadata")),
        )


def _normalize_protocol(value: str) -> str:
    protocol = str(value or "fi.alk").lower().replace("-", "_")
    aliases = {
        "agent_learning_websocket": "fi.alk",
        "websocket": "fi.alk",
        "ws": "fi.alk",
    }
    protocol = aliases.get(protocol, protocol)
    if protocol != "fi.alk":
        raise ValueError("protocol must be fi.alk")
    return protocol


def _read_until(sock: socket.socket, marker: bytes) -> bytes:
    chunks: list[bytes] = []
    data = b""
    while marker not in data:
        chunk = sock.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
        data = b"".join(chunks)
    return data


def _send_text_frame(sock: socket.socket, text: str, *, mask: bool) -> None:
    payload = text.encode("utf-8")
    header = bytearray([0x81])
    length = len(payload)
    mask_bit = 0x80 if mask else 0
    if length < 126:
        header.append(mask_bit | length)
    elif length <= 0xFFFF:
        header.extend([mask_bit | 126, *length.to_bytes(2, "big")])
    else:
        header.extend([mask_bit | 127, *length.to_bytes(8, "big")])
    if mask:
        key = os.urandom(4)
        masked = bytes(byte ^ key[index % 4] for index, byte in enumerate(payload))
        sock.sendall(bytes(header) + key + masked)
    else:
        sock.sendall(bytes(header) + payload)


def _send_close_frame(sock: socket.socket) -> None:
    sock.sendall(b"\x88\x80" + os.urandom(4))


def _read_frame(sock: socket.socket) -> tuple[int, str]:
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


def _read_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ValueError("WebSocket connection closed unexpectedly")
        data.extend(chunk)
    return bytes(data)
