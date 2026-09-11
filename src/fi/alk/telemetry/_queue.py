"""Bounded out-of-critical-path emission queue (Phase 8, ARCH §2c).

Imports: stdlib only. The emission contract is the load-bearing safety
property (R§3.5; PRD §4.3): the run thread does an O(1) enqueue and returns;
the ledger append + sync flush run on the drain side, on a single daemon
worker thread that owns the only write handle (the single-writer invariant of
P8-D4 holds within a process too). On overflow the row is dropped in O(1)
with a counter bump and the NEXT successful drain records a gap-marker row —
bounded, recorded, never silent. No queue/handler exception may propagate.
"""

from __future__ import annotations

import atexit
import os
import queue
import threading
import time
from typing import Any, Callable, Mapping

QUEUE_MAX_ENV = "AGENT_LEARNING_LEDGER_QUEUE_MAX"
DEFAULT_QUEUE_MAX = 1024
_FLUSH_TIMEOUT_S = 5.0

# handler(row, dropped_since_last_append) -> None
Handler = Callable[[Mapping[str, Any], int], None]


def _queue_max() -> int:
    raw = os.environ.get(QUEUE_MAX_ENV)
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return DEFAULT_QUEUE_MAX


class TelemetryQueue:
    """Bounded queue + lazy daemon worker; drop-with-gap-marker on overflow."""

    def __init__(self, handler: Handler, maxsize: int | None = None) -> None:
        self._handler = handler
        self._queue: queue.Queue[Mapping[str, Any]] = queue.Queue(
            maxsize=maxsize if maxsize is not None else _queue_max()
        )
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._pending = 0
        self._dropped = 0
        self.dropped_total = 0

    # -- producer side (the run's critical path: O(1), never raises) ---------

    def enqueue(self, row: Mapping[str, Any]) -> bool:
        try:
            with self._lock:
                try:
                    self._queue.put_nowait(row)
                    self._pending += 1
                except queue.Full:
                    self._dropped += 1
                    self.dropped_total += 1
                    return False
            self._ensure_worker()
            return True
        except BaseException:  # noqa: BLE001 — telemetry must never escape
            return False

    # -- worker side -----------------------------------------------------------

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(
                target=self._drain, name="agent-learning-telemetry", daemon=True
            )
            self._worker.start()

    def _drain(self) -> None:
        while True:
            try:
                row = self._queue.get(timeout=0.5)
            except queue.Empty:
                return  # worker parks; a later enqueue restarts it
            with self._lock:
                dropped, self._dropped = self._dropped, 0
            try:
                self._handler(row, dropped)
            except BaseException:  # noqa: BLE001 — never propagate (R§3.5)
                pass
            finally:
                with self._lock:
                    self._pending -= 1

    # -- flush (atexit + tests) -------------------------------------------------

    def flush(self, timeout: float = _FLUSH_TIMEOUT_S) -> bool:
        """Best-effort wait for the queue to drain; True when empty."""

        deadline = time.monotonic() + max(timeout, 0.0)
        while time.monotonic() < deadline:
            with self._lock:
                if self._pending <= 0:
                    return True
            self._ensure_worker()
            time.sleep(0.01)
        with self._lock:
            return self._pending <= 0


_GLOBAL: TelemetryQueue | None = None
_GLOBAL_LOCK = threading.Lock()


def global_queue(handler: Handler) -> TelemetryQueue:
    """The process-wide queue (lazily created; atexit-flushed)."""

    global _GLOBAL
    with _GLOBAL_LOCK:
        if _GLOBAL is None:
            _GLOBAL = TelemetryQueue(handler)
            atexit.register(_GLOBAL.flush)
        return _GLOBAL
