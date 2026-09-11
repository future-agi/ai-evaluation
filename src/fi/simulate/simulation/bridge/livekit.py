from __future__ import annotations

import asyncio
import logging
import time

from livekit import rtc
from livekit.api import AccessToken, VideoGrants

from fi.simulate.simulation.bridge.audio import PCMResampler
from fi.simulate.simulation.bridge.connector import ProviderConnector

logger = logging.getLogger(__name__)
ROOM_SAMPLE_RATE = 48000
ROOM_CHANNELS = 1
TRACK_TIMEOUT_SECONDS = 30.0
WATCHDOG_TIMEOUT_SECONDS = 60.0
PROVIDER_READY_BUFFER_FRAMES = 3000
PROVIDER_AUDIO_TIMEOUT_SECONDS = 120.0
PROVIDER_QUEUE_MAX_CHUNKS = 200


class LiveKitAudioBridge:
    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        api_secret: str,
        room_name: str,
        identity: str,
        connector: ProviderConnector,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._api_secret = api_secret
        self._room_name = room_name
        self._identity = identity
        self._connector = connector
        self._room = rtc.Room()
        self._track_future: asyncio.Future[rtc.RemoteAudioTrack] | None = None
        self._room_disconnected: asyncio.Future[None] | None = None
        self._audio_source: rtc.AudioSource | None = None
        self._closed = False
        self._close_lock = asyncio.Lock()
        self._last_audio_at = time.monotonic()
        self._last_provider_audio_at = time.monotonic()

    @property
    def call_id(self) -> str | None:
        return self._connector.call_id

    async def connect(self) -> None:
        loop = asyncio.get_running_loop()
        self._track_future = loop.create_future()
        self._room_disconnected = loop.create_future()

        @self._room.on("track_subscribed")
        def _on_track(track, publication, participant) -> None:
            if not isinstance(track, rtc.RemoteAudioTrack):
                return
            if self._track_future is None or self._track_future.done():
                return
            if publication.source != rtc.TrackSource.SOURCE_MICROPHONE:
                return
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
                return
            self._track_future.set_result(track)

        @self._room.on("disconnected")
        def _on_disconnected(*_args) -> None:
            if self._room_disconnected and not self._room_disconnected.done():
                self._room_disconnected.set_result(None)

        token = (
            AccessToken(self._api_key, self._api_secret)
            .with_identity(self._identity)
            .with_name("FutureAGI Web Bridge")
            .with_kind("sip")
            .with_grants(
                VideoGrants(
                    room_join=True,
                    room=self._room_name,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
            .to_jwt()
        )
        try:
            await self._room.connect(self._url, token)
            self._latch_preexisting_track()
            self._audio_source = rtc.AudioSource(ROOM_SAMPLE_RATE, ROOM_CHANNELS)
            track = rtc.LocalAudioTrack.create_audio_track(
                "bridge-audio", self._audio_source
            )
            await self._room.local_participant.publish_track(
                track,
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
            )
            await self._connector.connect()
        except Exception:
            await self.aclose()
            raise

    async def run(self) -> None:
        tasks = {
            asyncio.create_task(self._room_to_provider()),
            asyncio.create_task(self._provider_to_room()),
            asyncio.create_task(self._watchdog()),
            asyncio.create_task(self._wait_for_room_disconnect()),
        }
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                await task
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            try:
                await self._connector.disconnect()
            finally:
                await self._room.disconnect()

    def _latch_preexisting_track(self) -> None:
        if self._track_future is None or self._track_future.done():
            return
        for participant in self._room.remote_participants.values():
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
                continue
            for publication in participant.track_publications.values():
                if (
                    publication.track
                    and isinstance(publication.track, rtc.RemoteAudioTrack)
                    and publication.source == rtc.TrackSource.SOURCE_MICROPHONE
                ):
                    self._track_future.set_result(publication.track)
                    return

    async def _room_to_provider(self) -> None:
        if self._track_future is None:
            raise RuntimeError("bridge_not_connected")
        silence = asyncio.create_task(self._send_silence_until_track())
        try:
            track = await asyncio.wait_for(
                self._track_future, timeout=TRACK_TIMEOUT_SECONDS
            )
        finally:
            silence.cancel()
            await asyncio.gather(silence, return_exceptions=True)
        buffered_frames: list[tuple[bytes, int]] = []
        async for event in rtc.AudioStream(track):
            frame = event.frame
            self._last_audio_at = time.monotonic()
            frame_data = frame.data.tobytes()
            if not self._connector.is_agent_ready:
                if len(buffered_frames) < PROVIDER_READY_BUFFER_FRAMES:
                    buffered_frames.append((frame_data, frame.sample_rate))
                continue
            for buffered_data, buffered_rate in buffered_frames:
                await self._connector.send_audio(buffered_data, buffered_rate)
            buffered_frames.clear()
            await self._connector.send_audio(frame_data, frame.sample_rate)

    async def _provider_to_room(self) -> None:
        if self._audio_source is None:
            raise RuntimeError("bridge_not_connected")
        # The websocket reader must never block on room playback: a stalled
        # ``capture_frame`` would stop close/hangup frames from being seen and
        # keep a dead provider call alive. Live audio, so drop oldest when full.
        queue: asyncio.Queue[tuple[bytes, int] | None] = asyncio.Queue(
            maxsize=PROVIDER_QUEUE_MAX_CHUNKS
        )

        async def _pump() -> None:
            try:
                async for chunk in self._connector.recv_audio():
                    now = time.monotonic()
                    self._last_audio_at = now
                    self._last_provider_audio_at = now
                    if queue.full():
                        queue.get_nowait()
                    queue.put_nowait(chunk)
            finally:
                if queue.full():
                    queue.get_nowait()
                queue.put_nowait(None)

        pump = asyncio.create_task(_pump())
        resamplers: dict[int, PCMResampler] = {}
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                pcm, sample_rate = chunk
                if sample_rate != ROOM_SAMPLE_RATE:
                    resampler = resamplers.setdefault(
                        sample_rate,
                        PCMResampler(
                            from_rate=sample_rate,
                            to_rate=ROOM_SAMPLE_RATE,
                            channels=ROOM_CHANNELS,
                        ),
                    )
                    pcm = resampler.convert(pcm)
                await self._audio_source.capture_frame(
                    rtc.AudioFrame(
                        data=pcm,
                        sample_rate=ROOM_SAMPLE_RATE,
                        num_channels=ROOM_CHANNELS,
                        samples_per_channel=len(pcm) // 2,
                    )
                )
        finally:
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)

    async def _send_silence_until_track(self) -> None:
        frame = b"\x00" * int(16000 * 0.02 * 2)
        while self._track_future is not None and not self._track_future.done():
            await self._connector.send_audio(frame, 16000)
            await asyncio.sleep(0.02)

    async def _watchdog(self) -> None:
        # ``_last_audio_at`` refreshes on simulator silence frames too, so it is
        # blind to a dead provider; track provider-received audio separately.
        while True:
            await asyncio.sleep(5.0)
            now = time.monotonic()
            if now - self._last_audio_at > WATCHDOG_TIMEOUT_SECONDS:
                raise RuntimeError("bridge_audio_watchdog_timeout")
            if (
                now - self._last_provider_audio_at
                > PROVIDER_AUDIO_TIMEOUT_SECONDS
            ):
                raise RuntimeError("provider_audio_timeout")

    async def _wait_for_room_disconnect(self) -> None:
        if self._room_disconnected is None:
            raise RuntimeError("bridge_not_connected")
        await self._room_disconnected
