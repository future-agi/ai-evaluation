from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator

import aiohttp
from livekit import rtc

from fi.simulate.agent.definition import RetellTargetConfig
from fi.simulate.simulation.bridge.audio import PCMResampler
from fi.simulate.simulation.bridge.connector import ConnectorConfig, ProviderConnector

logger = logging.getLogger(__name__)
RETELL_SAMPLE_RATE = 48000


class RetellWebCallConnector(ProviderConnector):
    def __init__(self, config: ConnectorConfig) -> None:
        self._config = config
        self._room: rtc.Room | None = None
        self._audio_source: rtc.AudioSource | None = None
        self._track_future: asyncio.Future[rtc.RemoteAudioTrack] | None = None
        self._agent_disconnected = asyncio.Event()
        self._agent_identity: str | None = None
        self._agent_ready = asyncio.Event()
        self._call_id: str | None = None
        self._connected = False
        self._resamplers: dict[int, PCMResampler] = {}

    @classmethod
    def from_target(cls, target: RetellTargetConfig) -> "RetellWebCallConnector":
        api_key = os.environ.get(target.api_key_env, "").strip()
        if not api_key:
            raise ValueError("retell_webcall_config_missing: " + target.api_key_env)
        return cls(
            ConnectorConfig(
                api_key=api_key,
                assistant_id=target.agent_id,
                api_url=str(target.api_url),
                livekit_url=str(target.livekit_url),
            )
        )

    @classmethod
    def from_env(cls) -> "RetellWebCallConnector":
        api_key = os.environ.get("RETELL_API_KEY", "").strip()
        agent_id = os.environ.get("RETELL_AGENT_ID", "").strip()
        missing = [
            name
            for name, value in (
                ("RETELL_API_KEY", api_key),
                ("RETELL_AGENT_ID", agent_id),
            )
            if not value
        ]
        if missing:
            raise ValueError("retell_webcall_config_missing: " + ", ".join(missing))
        return cls.from_target(
            RetellTargetConfig(
                agent_id=agent_id,
                api_url=os.environ.get(
                    "RETELL_API_URL",
                    "https://api.retellai.com/v2/create-web-call",
                ),
                livekit_url=os.environ.get(
                    "RETELL_LIVEKIT_URL",
                    "wss://retell-ai-4ihahnq7.livekit.cloud",
                ),
            )
        )

    async def connect(self) -> None:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._config.api_url,
                headers={"Authorization": f"Bearer {self._config.api_key}"},
                json={"agent_id": self._config.assistant_id},
            ) as response:
                if response.status != 201:
                    raise RuntimeError(
                        f"retell_webcall_create_failed:{response.status}"
                    )
                payload = await response.json()
        access_token = (
            payload.get("access_token") if isinstance(payload, dict) else None
        )
        call_id = payload.get("call_id") if isinstance(payload, dict) else None
        if not isinstance(access_token, str) or not access_token.strip():
            raise ValueError("retell_webcall_response_missing_access_token")
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("retell_webcall_response_missing_call_id")
        self._call_id = call_id
        self._room = rtc.Room()
        self._track_future = asyncio.get_running_loop().create_future()

        @self._room.on("track_subscribed")
        def _on_track(track, _publication, participant) -> None:
            if not isinstance(track, rtc.RemoteAudioTrack):
                return
            if self._track_future is None or self._track_future.done():
                return
            identity = getattr(participant, "identity", None)
            if identity is not None:
                self._agent_identity = str(identity)
            self._track_future.set_result(track)
            self._agent_ready.set()

        @self._room.on("participant_disconnected")
        def _on_participant_disconnected(participant) -> None:
            identity = getattr(participant, "identity", None)
            if self._agent_identity is None or str(identity) == self._agent_identity:
                self._agent_disconnected.set()

        @self._room.on("disconnected")
        def _on_disconnected(*_args) -> None:
            self._agent_disconnected.set()

        try:
            await self._room.connect(self._config.livekit_url, access_token)
            self._audio_source = rtc.AudioSource(RETELL_SAMPLE_RATE, 1)
            track = rtc.LocalAudioTrack.create_audio_track(
                "bridge-audio", self._audio_source
            )
            await self._room.local_participant.publish_track(
                track,
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
            )
            self._connected = True
            if self._track_future is None:
                raise RuntimeError("retell_webcall_track_future_missing")
            await asyncio.wait_for(asyncio.shield(self._track_future), timeout=30.0)
            logger.info("retell_webcall_connected", extra={"call_id": call_id})
        except asyncio.TimeoutError as exc:
            await self.disconnect()
            raise RuntimeError("retell_webcall_agent_track_timeout") from exc
        except Exception:
            await self.disconnect()
            raise

    async def send_audio(self, data: bytes, sample_rate: int) -> None:
        if not self._audio_source or not self._connected:
            return
        if sample_rate != RETELL_SAMPLE_RATE:
            resampler = self._resamplers.setdefault(
                sample_rate,
                PCMResampler(from_rate=sample_rate, to_rate=RETELL_SAMPLE_RATE),
            )
            data = resampler.convert(data)
        await self._audio_source.capture_frame(
            rtc.AudioFrame(
                data=data,
                sample_rate=RETELL_SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=len(data) // 2,
            )
        )

    async def recv_audio(self) -> AsyncIterator[tuple[bytes, int]]:
        if self._track_future is None:
            return
        try:
            track = await asyncio.wait_for(self._track_future, timeout=30.0)
        except asyncio.TimeoutError as exc:
            self._connected = False
            raise RuntimeError("retell_webcall_agent_track_timeout") from exc
        async for event in rtc.AudioStream(track):
            frame = event.frame
            yield frame.data.tobytes(), frame.sample_rate
        self._connected = False

    async def disconnect(self) -> None:
        self._connected = False
        if self._room:
            await self._room.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_agent_ready(self) -> bool:
        return self._agent_ready.is_set()

    @property
    def call_id(self) -> str | None:
        return self._call_id
