from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator

import aiohttp

from fi.simulate.agent.definition import VapiTargetConfig
from fi.simulate.simulation.bridge.audio import PCMResampler
from fi.simulate.simulation.bridge.connector import ConnectorConfig, ProviderConnector

logger = logging.getLogger(__name__)
VAPI_SAMPLE_RATE = 16000


class VapiWebSocketConnector(ProviderConnector):
    def __init__(self, config: ConnectorConfig) -> None:
        self._config = config
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._call_id: str | None = None
        self._connected = False
        self._resamplers: dict[int, PCMResampler] = {}

    @classmethod
    def from_target(
        cls,
        target: VapiTargetConfig,
        *,
        first_message_mode: str | None = None,
    ) -> "VapiWebSocketConnector":
        api_key = os.environ.get(target.api_key_env, "").strip()
        if not api_key:
            raise ValueError("vapi_websocket_config_missing: " + target.api_key_env)
        return cls(
            ConnectorConfig(
                api_key=api_key,
                assistant_id=target.assistant_id,
                api_url=f"{str(target.api_base_url).rstrip('/')}/call",
                first_message_mode=first_message_mode,
            )
        )

    @classmethod
    def from_env(cls) -> "VapiWebSocketConnector":
        api_key = os.environ.get("VAPI_API_KEY", "").strip()
        assistant_id = os.environ.get("VAPI_ASSISTANT_ID", "").strip()
        missing = [
            name
            for name, value in (
                ("VAPI_API_KEY", api_key),
                ("VAPI_ASSISTANT_ID", assistant_id),
            )
            if not value
        ]
        if missing:
            raise ValueError("vapi_websocket_config_missing: " + ", ".join(missing))
        return cls.from_target(
            VapiTargetConfig(
                assistant_id=assistant_id,
                api_base_url=os.environ.get("VAPI_API_BASE_URL", "https://api.vapi.ai"),
            )
        )

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession()
        try:
            payload = {
                "assistantId": self._config.assistant_id,
                "transport": {
                    "provider": "vapi.websocket",
                    "audioFormat": {
                        "format": "pcm_s16le",
                        "container": "raw",
                        "sampleRate": VAPI_SAMPLE_RATE,
                    },
                },
            }
            if self._config.first_message_mode:
                payload["assistantOverrides"] = {
                    "firstMessageMode": self._config.first_message_mode,
                }
            async with self._session.post(
                self._config.api_url,
                headers={"Authorization": f"Bearer {self._config.api_key}"},
                json=payload,
            ) as response:
                if response.status != 201:
                    raise RuntimeError(
                        f"vapi_websocket_call_create_failed:{response.status}"
                    )
                payload = await response.json()
            call_id = payload.get("id") if isinstance(payload, dict) else None
            transport = payload.get("transport") if isinstance(payload, dict) else None
            websocket_url = (
                transport.get("websocketCallUrl")
                if isinstance(transport, dict)
                else None
            )
            if not isinstance(call_id, str) or not call_id.strip():
                raise ValueError("vapi_websocket_response_missing_call_id")
            if not isinstance(websocket_url, str) or not websocket_url.startswith(
                ("ws://", "wss://")
            ):
                raise ValueError("vapi_websocket_response_missing_url")
            self._call_id = call_id
            self._ws = await self._session.ws_connect(websocket_url)
            self._connected = True
            logger.info("vapi_websocket_connected", extra={"call_id": call_id})
        except Exception:
            await self.disconnect()
            raise

    async def send_audio(self, data: bytes, sample_rate: int) -> None:
        if not self._ws or self._ws.closed:
            return
        if sample_rate != VAPI_SAMPLE_RATE:
            resampler = self._resamplers.setdefault(
                sample_rate,
                PCMResampler(from_rate=sample_rate, to_rate=VAPI_SAMPLE_RATE),
            )
            data = resampler.convert(data)
        await self._ws.send_bytes(data)

    async def recv_audio(self) -> AsyncIterator[tuple[bytes, int]]:
        if not self._ws or self._ws.closed:
            return
        async for message in self._ws:
            if message.type == aiohttp.WSMsgType.BINARY:
                yield message.data, VAPI_SAMPLE_RATE
            elif message.type == aiohttp.WSMsgType.TEXT:
                if self._is_call_end_event(message.data):
                    logger.info(
                        "vapi_websocket_call_ended",
                        extra={"call_id": self._call_id},
                    )
                    break
            elif message.type in {
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.ERROR,
            }:
                break
        self._connected = False

    @staticmethod
    def _is_call_end_event(payload: str) -> bool:
        try:
            data = json.loads(payload)
        except ValueError:
            return False
        if not isinstance(data, dict):
            return False
        event_type = str(data.get("type") or "").lower()
        if event_type in {"hangup", "call-ended", "end-of-call-report"}:
            return True
        return event_type == "status-update" and str(
            data.get("status") or ""
        ).lower() in {"ended", "ending"}

    async def disconnect(self) -> None:
        self._connected = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def call_id(self) -> str | None:
        return self._call_id
