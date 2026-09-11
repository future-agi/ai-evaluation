from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from fi.simulate.agent.definition import RetellTargetConfig
from fi.simulate.simulation.bridge import retell
from fi.simulate.simulation.bridge.connector import ConnectorConfig
from fi.simulate.simulation.bridge.retell import RetellWebCallConnector


class _Response:
    status = 201

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def json(self):
        return {"access_token": "livekit-token", "call_id": "call_retell_123"}


class _Session:
    def __init__(self) -> None:
        self.request: dict[str, object] = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def post(self, url, *, headers, json):
        self.request = {"url": url, "headers": headers, "json": json}
        return _Response()


class _RemoteAudioTrack:
    pass


class _Room:
    def __init__(self) -> None:
        self.handlers = {}
        self.local_participant = SimpleNamespace(
            publish_track=self._publish_track,
        )
        self.connection = None
        self.disconnected = False

    def on(self, event):
        def decorator(callback):
            self.handlers[event] = callback
            return callback

        return decorator

    async def connect(self, url, token):
        self.connection = (url, token)

    async def _publish_track(self, track, options):
        self.published = (track, options)
        self.handlers["track_subscribed"](_RemoteAudioTrack(), None, None)

    async def disconnect(self):
        self.disconnected = True


def test_retell_webcall_connector_creates_and_joins_call(monkeypatch) -> None:
    session = _Session()
    room = _Room()
    monkeypatch.setattr(retell.aiohttp, "ClientSession", lambda: session)
    monkeypatch.setattr(retell.rtc, "Room", lambda: room)
    monkeypatch.setattr(retell.rtc, "RemoteAudioTrack", _RemoteAudioTrack)
    monkeypatch.setattr(retell.rtc, "AudioSource", lambda *_args: SimpleNamespace())
    monkeypatch.setattr(
        retell.rtc.LocalAudioTrack,
        "create_audio_track",
        lambda *_args: "bridge-track",
    )
    monkeypatch.setattr(
        retell.rtc,
        "TrackPublishOptions",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    connector = RetellWebCallConnector(
        ConnectorConfig(
            api_key="test-key",
            assistant_id="agent_123",
            api_url="https://api.retellai.com/v2/create-web-call",
            livekit_url="wss://retell.example.com",
        )
    )

    async def run() -> None:
        await connector.connect()
        await connector.disconnect()

    asyncio.run(run())

    assert connector.call_id == "call_retell_123"
    assert session.request["json"] == {"agent_id": "agent_123"}
    assert room.connection == ("wss://retell.example.com", "livekit-token")
    assert room.disconnected is True


def test_retell_webcall_connector_uses_explicit_target(monkeypatch) -> None:
    monkeypatch.setenv("HEALTHCARE_RETELL_KEY", "test-key")

    connector = RetellWebCallConnector.from_target(
        RetellTargetConfig(
            agent_id="agent_healthcare",
            api_url="https://retell.healthcare.example/v2/create-web-call",
            livekit_url="wss://retell-healthcare.example.com",
            api_key_env="HEALTHCARE_RETELL_KEY",
        )
    )

    assert connector._config.assistant_id == "agent_healthcare"
    assert connector._config.api_key == "test-key"
    assert (
        connector._config.api_url
        == "https://retell.healthcare.example/v2/create-web-call"
    )


def test_retell_webcall_connector_requires_credentials(monkeypatch) -> None:
    monkeypatch.delenv("RETELL_API_KEY", raising=False)
    monkeypatch.delenv("RETELL_AGENT_ID", raising=False)

    with pytest.raises(ValueError, match="RETELL_API_KEY, RETELL_AGENT_ID"):
        RetellWebCallConnector.from_env()


def test_unrelated_participant_disconnect_does_not_end_agent_audio(monkeypatch) -> None:
    session = _Session()
    room = _Room()

    async def publish(track, options):
        room.published = (track, options)
        room.handlers["track_subscribed"](
            _RemoteAudioTrack(),
            None,
            SimpleNamespace(identity="retell-agent"),
        )

    room.local_participant.publish_track = publish
    monkeypatch.setattr(retell.aiohttp, "ClientSession", lambda: session)
    monkeypatch.setattr(retell.rtc, "Room", lambda: room)
    monkeypatch.setattr(retell.rtc, "RemoteAudioTrack", _RemoteAudioTrack)
    monkeypatch.setattr(retell.rtc, "AudioSource", lambda *_args: SimpleNamespace())
    monkeypatch.setattr(
        retell.rtc.LocalAudioTrack,
        "create_audio_track",
        lambda *_args: "bridge-track",
    )
    monkeypatch.setattr(
        retell.rtc,
        "TrackPublishOptions",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    connector = RetellWebCallConnector(
        ConnectorConfig(
            api_key="test-key",
            assistant_id="agent_123",
            api_url="https://api.retellai.com/v2/create-web-call",
            livekit_url="wss://retell.example.com",
        )
    )

    asyncio.run(connector.connect())
    room.handlers["participant_disconnected"](
        SimpleNamespace(identity="unrelated-participant")
    )
    assert connector._agent_disconnected.is_set() is False
    room.handlers["participant_disconnected"](SimpleNamespace(identity="retell-agent"))
    assert connector._agent_disconnected.is_set() is True
    asyncio.run(connector.disconnect())
