"""Account-integrated telemetry (Phase 8): local run ledger + keyed sync.

Two channels, never a third (P8-D1): (a) the always-on local run ledger —
every kit run appends one content-addressed, hash-chained row to
``${AGENT_LEARNING_HOME:-~/.agent-learning}/ledger/runs.jsonl``; (b) keyed
sync to the USER'S OWN Future AGI account when their keys resolve. There is
no anonymous analytics channel anywhere in the kit — structurally absent and
gate-proven (``telemetry_boundary``, gate #72).

Module scope imports are stdlib + the stdlib-only package internals; the
network-capable sync lane (``_sync``) is imported lazily inside functions
only, after the kill switch and key gates. ``AGENT_LEARNING_TELEMETRY=off``
binds everything, including vendored ``fi/*`` (P8-D6).
"""

from __future__ import annotations

from typing import Any, Mapping

from ._contract import (  # noqa: F401 — package canon re-exports
    AGENT_LEARNING_RUN_KIND,
    EVIDENCE_CLASSES,
    GAP_SCHEMA,
    GENESIS,
    LEDGER_DIR_NAME,
    LEDGER_HOME_ENV,
    LEDGER_PATH_ENV,
    LEDGER_ROW_SCHEMA,
    NON_CANONICAL_FIELDS,
    PHASES,
    RELEASE_ADMISSIBLE_EVIDENCE_CLASSES,
    ROW_FIELDS,
    SYNC_MODE_AUTO,
    SYNC_MODE_ENV,
    SYNC_MODE_LOCAL,
    SYNC_MODES,
    SYNC_STATES,
    TELEMETRY_ENV,
    TELEMETRY_OFF_VALUE,
    TOMBSTONE_FIELDS,
    TOMBSTONE_REASONS,
    TOMBSTONE_SCHEMA,
    UNREADABLE_LINE_SCHEMA,
    VERDICTS,
    kill_switch_on,
    ledger_dir,
    sync_mode,
)
from ._ledger import RunLedger  # noqa: F401
from ._queue import TelemetryQueue, global_queue  # noqa: F401
from ._row import (  # noqa: F401
    build_ledger_row,
    canonical_row_address,
    canonical_row_bytes,
    content_admissible,
    declared_required_env,
)
from ._run import (  # noqa: F401
    RunRecorder,
    RunSummary,
    emit_run,
    run_telemetry,
)
from ._url import build_dashboard_url  # noqa: F401

__all__ = [
    "AGENT_LEARNING_RUN_KIND",
    "EVIDENCE_CLASSES",
    "GAP_SCHEMA",
    "GENESIS",
    "LEDGER_DIR_NAME",
    "LEDGER_HOME_ENV",
    "LEDGER_PATH_ENV",
    "LEDGER_ROW_SCHEMA",
    "NON_CANONICAL_FIELDS",
    "PHASES",
    "RELEASE_ADMISSIBLE_EVIDENCE_CLASSES",
    "ROW_FIELDS",
    "RunLedger",
    "RunRecorder",
    "RunSummary",
    "SYNC_MODE_AUTO",
    "SYNC_MODE_ENV",
    "SYNC_MODE_LOCAL",
    "SYNC_MODES",
    "SYNC_STATES",
    "TELEMETRY_ENV",
    "TELEMETRY_OFF_VALUE",
    "TOMBSTONE_FIELDS",
    "TOMBSTONE_REASONS",
    "TOMBSTONE_SCHEMA",
    "TelemetryQueue",
    "UNREADABLE_LINE_SCHEMA",
    "VERDICTS",
    "build_dashboard_url",
    "build_ledger_row",
    "canonical_row_address",
    "canonical_row_bytes",
    "content_admissible",
    "declared_required_env",
    "emit_run",
    "flush",
    "kill_switch_on",
    "ledger_dir",
    "record_run",
    "run_telemetry",
    "sync_mode",
]


def _handle_row(row: Mapping[str, Any], dropped: int) -> None:
    """Drain-side handler: ledger append (+ gap marker for any drops since
    the last successful append). Runs on the worker thread; every failure is
    swallowed by the queue (R§3.5).

    Sync is EXPLICIT in v1 — ``agent-learn runs sync [<id>|--queued]`` or the
    SDK ``telemetry._sync.sync_run`` — never fired from the emission path.
    Emission-time auto-sync would turn every stray key in the environment
    (test/example dummies included) into a network attempt inside release
    flows; rows queue locally instead and the queued-sync path is idempotent
    by content address, so nothing is ever lost (P8-D3, R§3.5).
    """

    ledger = RunLedger()
    if dropped > 0:
        ledger.append_gap(dropped)
    ledger.append(row)


def record_run(run_payload: Mapping[str, Any]) -> None:
    """The single emission hook target (ARCH Decision 7): called once at the
    run-manifest boundary for every ``agent-learning.run.v1`` payload.

    Out of the critical path: builds the redacted, content-addressed row and
    does an O(1) bounded enqueue. ``AGENT_LEARNING_TELEMETRY=off`` disables
    everything — ledger append and sync alike (P8-D6).
    """

    if kill_switch_on():
        return
    required_env = declared_required_env(run_payload)
    row = build_ledger_row(run_payload, required_env=required_env)
    global_queue(_handle_row).enqueue(row)


def flush(timeout: float = 5.0) -> bool:
    """Best-effort drain of the emission queue (atexit calls this too)."""

    return global_queue(_handle_row).flush(timeout)
