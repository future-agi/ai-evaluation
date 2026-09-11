from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fi.simulate._logging import redacted_exc_info

try:
    from livekit import rtc
    from livekit.api import AccessToken, VideoGrants
except ImportError:
    rtc = None
    AccessToken = None
    VideoGrants = None

logger = logging.getLogger(__name__)
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class RecordedTrack:
    participant_identity: str
    participant_sid: str
    track_sid: str
    path: Path
    start_offset_frames: int = 0


class RoomRecorder:
    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        api_secret: str,
        room_name: str,
        identity: str = "recorder",
        sample_rate: int = 8000,
        output_dir: str | Path = "recordings",
        join_delay_s: float = 0.2,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._api_secret = api_secret
        self._room_name = room_name
        self._identity = identity
        self._sample_rate = sample_rate
        self._output_dir = Path(output_dir)
        self._join_delay_s = join_delay_s
        self._room: Any | None = None
        self._running = False
        self._tasks: set[asyncio.Task[None]] = set()
        self._track_ids: set[str] = set()
        self._records: list[RecordedTrack] = []
        self._errors: list[BaseException] = []
        self._recording_started_at: float | None = None

    @property
    def records(self) -> tuple[RecordedTrack, ...]:
        return tuple(self._records)

    @property
    def errors(self) -> tuple[BaseException, ...]:
        return tuple(self._errors)

    @property
    def recording_started_at(self) -> float | None:
        """Wall-clock epoch used as t=0 for every recorded track."""
        return self._recording_started_at

    async def start(self) -> None:
        if self._running:
            return
        if rtc is None or AccessToken is None or VideoGrants is None:
            raise ImportError("LiveKit recording requires the 'livekit' extra")
        self._running = True
        await asyncio.sleep(max(0.0, self._join_delay_s))
        token = self._build_token()
        room = rtc.Room()
        try:
            await room.connect(
                self._url,
                token,
                options=rtc.RoomOptions(auto_subscribe=False),
            )
        except BaseException:
            with contextlib.suppress(Exception):
                await room.disconnect()
            raise
        self._room = room
        self._recording_started_at = time.time()
        self._output_dir.mkdir(parents=True, exist_ok=True)

        @room.on("track_published")
        def _on_track_published(publication, participant) -> None:
            self._subscribe_audio(publication)

        @room.on("track_subscribed")
        def _on_track_subscribed(track, publication, participant) -> None:
            self._start_recording(track, publication, participant)

        for participant in tuple(room.remote_participants.values()):
            for publication in tuple(participant.track_publications.values()):
                self._subscribe_audio(publication)

    def _build_token(self) -> str:
        return (
            AccessToken(self._api_key, self._api_secret)
            .with_identity(self._identity)
            .with_grants(
                VideoGrants(
                    room_join=True,
                    room=self._room_name,
                    hidden=True,
                    recorder=True,
                    can_publish=False,
                    can_publish_data=False,
                    can_update_own_metadata=False,
                )
            )
            .to_jwt()
        )

    def _subscribe_audio(self, publication: Any) -> None:
        if getattr(publication, "kind", None) != rtc.TrackKind.KIND_AUDIO:
            return
        publication.set_subscribed(True)

    def paths_for_participant(
        self,
        participant_identity: str,
        *,
        track_sid: str | None = None,
    ) -> list[Path]:
        return [
            record.path
            for record in self._records
            if record.participant_identity == participant_identity
            and (track_sid is None or record.track_sid == track_sid)
        ]

    async def aclose(self) -> None:
        self._running = False
        if self._room is not None:
            try:
                await self._room.disconnect()
            except Exception as exc:
                logger.error(
                    "Recorder room disconnect failed",
                    exc_info=redacted_exc_info(exc),
                    extra={
                        "room_name": self._room_name,
                        "exception_type": type(exc).__name__,
                    },
                )
            self._room = None
        if not self._tasks:
            return
        done, pending = await asyncio.wait(self._tasks, timeout=5)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            self._capture_task_error(task)

    def _start_recording(self, track: Any, publication: Any, participant: Any) -> None:
        if getattr(track, "kind", None) != rtc.TrackKind.KIND_AUDIO:
            return
        track_sid = str(publication.sid)
        if track_sid in self._track_ids:
            return
        self._track_ids.add(track_sid)
        started_at = self._recording_started_at or time.time()
        start_offset_frames = max(
            0,
            round((time.time() - started_at) * self._sample_rate),
        )
        task = asyncio.create_task(
            self._record_track(
                track,
                publication,
                participant,
                start_offset_frames=start_offset_frames,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._recording_done)

    async def _record_track(
        self,
        track: Any,
        publication: Any,
        participant: Any,
        *,
        start_offset_frames: int,
    ) -> None:
        participant_identity = str(participant.identity)
        participant_sid = str(participant.sid)
        track_sid = str(publication.sid)
        path = self._output_dir / (
            f"{_safe_component(participant_identity)}--{_safe_component(track_sid)}.wav"
        )
        record = RecordedTrack(
            participant_identity=participant_identity,
            participant_sid=participant_sid,
            track_sid=track_sid,
            path=path,
            start_offset_frames=start_offset_frames,
        )
        self._records.append(record)
        stream = rtc.AudioStream(track, sample_rate=self._sample_rate, num_channels=1)
        try:
            with wave.open(str(path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self._sample_rate)
                if start_offset_frames:
                    wav_file.writeframes(b"\x00\x00" * start_offset_frames)
                async for event in stream:
                    wav_file.writeframes(event.frame.data)
        finally:
            await stream.aclose()

    def _recording_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        self._capture_task_error(task)

    def _capture_task_error(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is None or error in self._errors:
            return
        self._errors.append(error)
        logger.error(
            "Recorder track failed",
            exc_info=redacted_exc_info(error),
            extra={
                "room_name": self._room_name,
                "exception_type": type(error).__name__,
            },
        )


def _read_mono_int16(paths: list[Path], *, sample_rate: int) -> list[np.ndarray]:
    arrays = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        with wave.open(str(path), "rb") as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                raise ValueError("recording_format_unsupported")
            if wav_file.getframerate() != sample_rate:
                raise ValueError("recording_sample_rate_mismatch")
            arrays.append(
                np.frombuffer(
                    wav_file.readframes(wav_file.getnframes()),
                    dtype=np.int16,
                )
            )
    return arrays


def _sum_int16(arrays: list[np.ndarray], length: int) -> np.ndarray:
    mixed = np.zeros(length, dtype=np.int32)
    for array in arrays:
        mixed[: array.size] += array.astype(np.int32)
    return np.clip(mixed, -32768, 32767).astype(np.int16)


def mix_recordings(
    paths: list[Path],
    destination: Path,
    *,
    sample_rate: int,
) -> Path | None:
    arrays = _read_mono_int16(paths, sample_rate=sample_rate)
    if not arrays:
        return None
    max_length = max(array.size for array in arrays)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(_sum_int16(arrays, max_length).tobytes())
    return destination


def mix_recordings_stereo(
    left_paths: list[Path],
    right_paths: list[Path],
    destination: Path,
    *,
    sample_rate: int,
) -> Path | None:
    left = _read_mono_int16(left_paths, sample_rate=sample_rate)
    right = _read_mono_int16(right_paths, sample_rate=sample_rate)
    if not left and not right:
        return None
    max_length = max(array.size for array in (*left, *right))
    interleaved = np.stack(
        [_sum_int16(left, max_length), _sum_int16(right, max_length)],
        axis=1,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(interleaved.tobytes())
    return destination


def _safe_component(value: str) -> str:
    return _SAFE_COMPONENT.sub("_", value).strip("._") or "unknown"
