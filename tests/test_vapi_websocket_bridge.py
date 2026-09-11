from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from fi.simulate.agent.definition import VapiTargetConfig
from fi.simulate.simulation.bridge import vapi
from fi.simulate.simulation.bridge.connector import ConnectorConfig
from fi.simulate.simulation.bridge.vapi import VapiWebSocketConnector


class _Response:
    status = 201

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def json(self):
        return {
            "id": "call_web_123",
            "transport": {"websocketCallUrl": "wss://vapi.example/call"},
        }


class _WebSocket:
    closed = False

    def __init__(self) -> None:
        self.sent: list[bytes] = []

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self):
        self._messages = iter(
            [
                SimpleNamespace(type=vapi.aiohttp.WSMsgType.BINARY, data=b"audio"),
                SimpleNamespace(type=vapi.aiohttp.WSMsgType.CLOSE, data=None),
            ]
        )
        return self

    async def __anext__(self):
        try:
            return next(self._messages)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _Session:
    closed = False

    def __init__(self) -> None:
        self.websocket = _WebSocket()
        self.request: dict[str, object] = {}

    def post(self, url, *, headers, json):
        self.request = {"url": url, "headers": headers, "json": json}
        return _Response()

    async def ws_connect(self, url):
        self.request["websocket_url"] = url
        return self.websocket

    async def close(self) -> None:
        self.closed = True


def test_vapi_websocket_connector_creates_and_streams_call(monkeypatch) -> None:
    session = _Session()
    monkeypatch.setattr(vapi.aiohttp, "ClientSession", lambda: session)
    connector = VapiWebSocketConnector(
        ConnectorConfig(
            api_key="test-key",
            assistant_id="assistant_123",
            api_url="https://api.vapi.ai/call",
            first_message_mode="assistant-waits-for-user",
        )
    )

    async def run() -> list[tuple[bytes, int]]:
        await connector.connect()
        await connector.send_audio(b"\x00\x00" * 480, 48000)
        received = [item async for item in connector.recv_audio()]
        await connector.disconnect()
        return received

    received = asyncio.run(run())

    assert connector.call_id == "call_web_123"
    assert session.request["json"] == {
        "assistantId": "assistant_123",
        "transport": {
            "provider": "vapi.websocket",
            "audioFormat": {
                "format": "pcm_s16le",
                "container": "raw",
                "sampleRate": 16000,
            },
        },
        "assistantOverrides": {
            "firstMessageMode": "assistant-waits-for-user",
        },
    }
    assert session.request["websocket_url"] == "wss://vapi.example/call"
    assert len(session.websocket.sent[0]) < 960
    assert received == [(b"audio", 16000)]
    assert session.closed is True


def test_vapi_websocket_connector_uses_explicit_target(monkeypatch) -> None:
    monkeypatch.setenv("HEALTHCARE_VAPI_KEY", "test-key")

    connector = VapiWebSocketConnector.from_target(
        VapiTargetConfig(
            assistant_id="assistant_healthcare",
            api_base_url="https://vapi.healthcare.example",
            api_key_env="HEALTHCARE_VAPI_KEY",
        ),
        first_message_mode="assistant-waits-for-user",
    )

    assert connector._config.assistant_id == "assistant_healthcare"
    assert connector._config.api_key == "test-key"
    assert connector._config.api_url == "https://vapi.healthcare.example/call"
    assert connector._config.first_message_mode == "assistant-waits-for-user"


def test_vapi_websocket_connector_requires_credentials(monkeypatch) -> None:
    monkeypatch.delenv("VAPI_API_KEY", raising=False)
    monkeypatch.delenv("VAPI_ASSISTANT_ID", raising=False)

    with pytest.raises(ValueError, match="VAPI_API_KEY, VAPI_ASSISTANT_ID"):
        VapiWebSocketConnector.from_env()


def test_call_end_text_event_is_recognized() -> None:
    import json

    is_end = VapiWebSocketConnector._is_call_end_event
    assert is_end(json.dumps({"type": "hangup"}))
    assert is_end(json.dumps({"type": "call-ended"}))
    assert is_end(json.dumps({"type": "end-of-call-report"}))
    assert is_end(json.dumps({"type": "status-update", "status": "ended"}))
    assert not is_end(json.dumps({"type": "status-update", "status": "in-progress"}))
    assert not is_end(json.dumps({"type": "transcript", "text": "hi"}))
    assert not is_end("not json")
    assert not is_end(json.dumps(["hangup"]))


def test_recv_audio_stops_on_call_end_text_event() -> None:
    import json

    connector = VapiWebSocketConnector(
        ConnectorConfig(api_key="k", assistant_id="a", api_url="https://api/call")
    )

    class _EndedWS:
        closed = False

        def __aiter__(self):
            self._messages = iter(
                [
                    SimpleNamespace(
                        type=vapi.aiohttp.WSMsgType.BINARY, data=b"audio"
                    ),
                    SimpleNamespace(
                        type=vapi.aiohttp.WSMsgType.TEXT,
                        data=json.dumps({"type": "hangup"}),
                    ),
                    SimpleNamespace(
                        type=vapi.aiohttp.WSMsgType.BINARY, data=b"late"
                    ),
                ]
            )
            return self

        async def __anext__(self):
            try:
                return next(self._messages)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    connector._ws = _EndedWS()
    connector._connected = True

    async def collect():
        return [chunk async for chunk in connector.recv_audio()]

    chunks = asyncio.run(collect())

    assert chunks == [(b"audio", vapi.VAPI_SAMPLE_RATE)]
    assert connector.is_connected is False
