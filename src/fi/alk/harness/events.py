"""Durable harness progress delivery for both local and hosted execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from typing import Protocol

from fi.simulate.runtime.events import CanonicalEvent


class EventTransport(Protocol):
    """Platform boundary. Implementations must be idempotent by ``event_id``."""

    def send(self, run_id: str, events: list[CanonicalEvent]) -> set[str]: ...


class EventOutbox:
    """Append-only event journal with a durable acknowledgement cursor.

    Every event lands locally before a network attempt. A killed local CLI or hosted sandbox can
    reopen the outbox and retry without losing progress; the platform deduplicates event IDs.
    """

    def __init__(self, root: str | Path, run_id: str) -> None:
        self.root = Path(root).expanduser().resolve() / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "harness-events.jsonl"
        self.acked_path = self.root / "harness-events.acked.json"
        self._lock = RLock()

    def append(self, event: CanonicalEvent) -> None:
        with self._lock, self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(event.model_dump_json(exclude_none=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def pending(self) -> list[CanonicalEvent]:
        acknowledged = self.acknowledged()
        if not self.events_path.exists():
            return []
        events: list[CanonicalEvent] = []
        with self._lock, self.events_path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    event = CanonicalEvent.model_validate_json(line)
                except ValueError as exc:
                    raise ValueError(
                        f"event_outbox_corrupt: line {number}: {exc}"
                    ) from exc
                if event.event_id not in acknowledged:
                    events.append(event)
        return events

    def acknowledged(self) -> set[str]:
        if not self.acked_path.exists():
            return set()
        try:
            value = json.loads(self.acked_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"event_acknowledgements_corrupt: {exc}") from exc
        return {str(item) for item in value.get("event_ids", [])}

    def acknowledge(self, event_ids: set[str]) -> None:
        if not event_ids:
            return
        with self._lock:
            all_ids = self.acknowledged() | event_ids
            temporary = self.acked_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps({"event_ids": sorted(all_ids)}, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.acked_path)


class BufferedEventSink:
    """Write-through local sink with best-effort, retryable platform delivery."""

    def __init__(
        self,
        outbox: EventOutbox,
        transport: EventTransport | None = None,
        *,
        batch_size: int = 100,
    ) -> None:
        self.outbox = outbox
        self.transport = transport
        self.batch_size = max(1, batch_size)

    def write(self, event: CanonicalEvent) -> None:
        self.outbox.append(event)
        if self.transport is not None:
            try:
                self.flush(limit=self.batch_size)
            except Exception:
                # Execution is authoritative; telemetry is retried from the durable outbox.
                pass

    def flush(self, *, limit: int | None = None) -> int:
        if self.transport is None:
            return 0
        pending = self.outbox.pending()
        if limit is not None:
            pending = pending[:limit]
        delivered = 0
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            accepted = self.transport.send(batch[0].run_id, batch)
            expected = {event.event_id for event in batch}
            # A transport may acknowledge a subset. Everything else remains pending.
            acknowledged = expected & set(accepted)
            self.outbox.acknowledge(acknowledged)
            delivered += len(acknowledged)
            if acknowledged != expected:
                break
        return delivered


__all__ = ["BufferedEventSink", "EventOutbox", "EventTransport"]
