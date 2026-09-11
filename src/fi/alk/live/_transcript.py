"""JSONL transcript recorder + reader for live lanes (ARCH Decision 3).

Imports: stdlib only. The file IS the ledger: one JSON object per line,
``{"t": <monotonic-relative s>, "channel": ..., "type": ..., "payload": ...}``
with ``channel`` in {"user", "agent", "tool", "lane"}. Redaction runs at
write time — declared ``required_env`` VALUES never hit disk. The recorder
owns the size cap (``AGENT_LEARNING_LIVE_TRANSCRIPT_MAX_BYTES``, default
64 MiB/scenario): over-cap behavior is retain head+tail, never silently drop.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

TRANSCRIPT_MAX_BYTES_ENV = "AGENT_LEARNING_LIVE_TRANSCRIPT_MAX_BYTES"
DEFAULT_TRANSCRIPT_MAX_BYTES = 64 * 1024 * 1024
TRANSCRIPT_CHANNELS = ("user", "agent", "tool", "lane")

# Bytes reserved out of the tail budget for the truncation marker line.
_TRUNCATION_MARKER_RESERVE = 512


def redact_env_values(text: str, required_env: Sequence[str]) -> str:
    """Replace any occurrence of a declared env var's VALUE with
    '[redacted:<NAME>]'. Extends the existing redacted-auth evidence pattern
    (trinity evaluation-hook gates carry auth.redacted evidence) to lane
    transcripts."""

    for name in required_env:
        value = os.environ.get(name)
        if value:
            text = text.replace(value, f"[redacted:{name}]")
    return text


def transcript_max_bytes() -> int:
    """Resolve the per-scenario transcript size cap (3A owns this env var)."""

    raw = os.environ.get(TRANSCRIPT_MAX_BYTES_ENV)
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return DEFAULT_TRANSCRIPT_MAX_BYTES


class TranscriptRecorder:
    """Append-only JSONL replay transcript (R§3.1, DFAH lineage).

    One line per event. The dual-channel requirement for voice lanes is just
    two channels of this stream. Capture-to-fixture re-reads the file
    verbatim, so the recorder never writes secrets: every serialized event is
    passed through :func:`redact_env_values` before write.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        required_env: Sequence[str],
        max_bytes: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.required_env = tuple(required_env)
        self._max_bytes = int(max_bytes) if max_bytes else transcript_max_bytes()
        self._head_budget = max(self._max_bytes // 2, 1)
        self._tail_budget = max(
            self._max_bytes - self._head_budget - _TRUNCATION_MARKER_RESERVE, 1
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self._t0 = time.monotonic()
        self._bytes_written = 0
        self._original_bytes = 0
        self._original_sha = hashlib.sha256()
        self._event_count = 0
        self._channels: set[str] = set()
        self._buffering_tail = False
        self._tail: deque[tuple[int, str]] = deque()
        self._tail_bytes = 0
        self._dropped_events = 0
        self._closed = False
        self._summary: dict[str, Any] | None = None
        # In-memory (already redacted) copies for attribution/stats. Control
        # events are small; bulk audio lives in side files, never in events.
        self.events: list[dict[str, Any]] = []

    # -- write path ---------------------------------------------------------

    def record(
        self, channel: str, type: str, payload: Mapping[str, Any]
    ) -> None:
        if self._closed:
            raise RuntimeError(f"transcript {self.path} is closed")
        event = {
            "t": round(time.monotonic() - self._t0, 6),
            "channel": str(channel),
            "type": str(type),
            "payload": payload if isinstance(payload, dict) else dict(payload),
        }
        line = json.dumps(event, ensure_ascii=False, default=str)
        line = redact_env_values(line, self.required_env)
        try:
            stored = json.loads(line)
        except ValueError:
            stored = {
                "t": event["t"],
                "channel": event["channel"],
                "type": event["type"],
                "payload": {"unserializable": True},
            }
        self.events.append(stored)
        data = line + "\n"
        nbytes = len(data.encode("utf-8"))
        self._event_count += 1
        self._channels.add(event["channel"])
        self._original_bytes += nbytes
        self._original_sha.update(data.encode("utf-8"))
        if not self._buffering_tail:
            if self._bytes_written + nbytes <= self._head_budget:
                self._fh.write(data)
                self._bytes_written += nbytes
                return
            self._buffering_tail = True
        self._tail.append((nbytes, data))
        self._tail_bytes += nbytes
        while self._tail and self._tail_bytes > self._tail_budget:
            dropped_bytes, _ = self._tail.popleft()
            self._tail_bytes -= dropped_bytes
            self._dropped_events += 1

    # -- close + summary ----------------------------------------------------

    def close(self) -> dict[str, Any]:
        """Flush the tail, close the file, and return the artifact summary:
        ``{path, event_count, channels, duration_s, bytes, sha256, complete}``
        plus a ``truncated`` stanza when the cap dropped events."""

        if self._closed:
            assert self._summary is not None
            return self._summary
        truncated = self._dropped_events > 0
        if self._buffering_tail:
            if truncated:
                marker = json.dumps(
                    {
                        "t": round(time.monotonic() - self._t0, 6),
                        "channel": "lane",
                        "type": "transcript_truncated",
                        "payload": {
                            "retained": "head_and_tail",
                            "dropped_events": self._dropped_events,
                        },
                    },
                    ensure_ascii=False,
                )
                self._fh.write(marker + "\n")
                self._bytes_written += len((marker + "\n").encode("utf-8"))
            for nbytes, data in self._tail:
                self._fh.write(data)
                self._bytes_written += nbytes
            self._tail.clear()
            self._tail_bytes = 0
        self._fh.close()
        self._closed = True
        file_sha = hashlib.sha256()
        try:
            with open(self.path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    file_sha.update(chunk)
            bytes_on_disk = self.path.stat().st_size
        except OSError:
            bytes_on_disk = self._bytes_written
        summary: dict[str, Any] = {
            "path": str(self.path),
            "event_count": self._event_count,
            "channels": sorted(self._channels),
            "duration_s": round(time.monotonic() - self._t0, 6),
            "bytes": bytes_on_disk,
            "sha256": file_sha.hexdigest(),
            "complete": not truncated,
        }
        if truncated:
            summary["truncated"] = {
                "original_bytes": self._original_bytes,
                "original_sha256": self._original_sha.hexdigest(),
                "retained": "head_and_tail",
                "dropped_events": self._dropped_events,
            }
        self._summary = summary
        return summary


def read_transcript(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL transcript back into its event list (the replay source).

    Unparseable lines are surfaced as ``transcript_unreadable_line`` lane
    events rather than silently skipped — attribution treats an unreadable
    transcript as lane_infra evidence.
    """

    events: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                events.append(
                    {
                        "t": None,
                        "channel": "lane",
                        "type": "transcript_unreadable_line",
                        "payload": {"line_number": line_number},
                    }
                )
                continue
            if isinstance(event, dict):
                events.append(event)
            else:
                events.append(
                    {
                        "t": None,
                        "channel": "lane",
                        "type": "transcript_unreadable_line",
                        "payload": {"line_number": line_number},
                    }
                )
    return events
