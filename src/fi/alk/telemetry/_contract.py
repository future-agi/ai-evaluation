"""Telemetry-local contract: row schemas, ledger paths, kill switch (Phase 8).

Imports: stdlib only, plus the two reused live/ seams. Evidence classes and
the run kind are IMPORTED from ``live/_contract.py`` and never redeclared —
one vocabulary, one redaction seam (ARCH §2.0; the VS-Code "components exempt
from the off switch" failure mode is structurally impossible with one seam).
The ``telemetry_boundary`` gate scans this package like any release module.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- reused verbatim from live/ (the ONE vocabulary; never redeclared) ------
from ..live._contract import (  # noqa: F401 — re-exported package canon
    AGENT_LEARNING_RUN_KIND,
    EVIDENCE_CLASSES,
    RELEASE_ADMISSIBLE_EVIDENCE_CLASSES,
    VERDICTS,
)

# --- row schema tags (ARCH §3; REVIEW-RULINGS MF3) ---------------------------
LEDGER_ROW_SCHEMA = "agent-learning.ledger-row.v1"
TOMBSTONE_SCHEMA = "agent-learning.ledger-tombstone.v1"
GAP_SCHEMA = "agent-learning.ledger-gap.v1"
UNREADABLE_LINE_SCHEMA = "agent-learning.ledger-unreadable-line.v1"

# --- hash-chain genesis sentinel (ARCH §2b; MF4: self-describing string) ----
GENESIS = "agent-learning.ledger.genesis.v1"

# --- semconv pin (PRD §4.4: OTel GenAI semconv is Development mid-2026) -----
SEMCONV_VERSION_ENV = "OTEL_SEMCONV_STABILITY_OPT_IN"

# --- kill switch (P8-D6: binds EVERYTHING including vendored fi/*) ----------
TELEMETRY_ENV = "AGENT_LEARNING_TELEMETRY"
TELEMETRY_OFF_VALUE = "off"

# --- sync mode (Phase 14, W&B `WANDB_MODE` analogue) -------------------------
# Reconciles the user's "keys present -> dashboard" (W&B-online) intent with the
# P8 doctrine that emission must NOT auto-sync in release/CI flows. ``auto``
# (default) emits when keys resolve; ``local`` queues locally + explicit
# ``runs sync`` only. The kill switch overrides both. The test/gate harness
# pins ``local`` so no internal flow makes a surprise network call.
SYNC_MODE_ENV = "AGENT_LEARNING_SYNC"
SYNC_MODE_AUTO = "auto"
SYNC_MODE_LOCAL = "local"
SYNC_MODES = (SYNC_MODE_AUTO, SYNC_MODE_LOCAL)

# --- ledger disk layout (ARCH §2a / Decision 9; MF5) -------------------------
LEDGER_HOME_ENV = "AGENT_LEARNING_HOME"
LEDGER_PATH_ENV = "AGENT_LEARNING_LEDGER_PATH"  # overrides the DIRECTORY
LEDGER_DIR_NAME = "ledger"
ROWS_FILENAME = "runs.jsonl"
CHAIN_HEAD_FILENAME = "chain.head"
SYNC_CURSOR_FILENAME = "sync.cursor"

# --- canonical row field set (ARCH §2a table; MF3) ---------------------------
ROW_FIELDS = (
    "schema",
    "kind",
    "phase",
    "evidence_class",
    "verdict",
    "scores",
    "gate_outcomes",
    "semconv_version",
    "manifest_address",
    "asset_refs",
    "trace_ids",
    "content_bearing",
    "redaction",
    "created_at",
    "run_id",
    "chain",
)

# fields excluded from the canonical hash preimage — they ARE the hash /
# envelope (ARCH §2a: the addressed core excludes exactly these three):
NON_CANONICAL_FIELDS = ("created_at", "run_id", "chain")

# --- tombstone fields (ARCH §2b; never rewrite a row — append) ---------------
TOMBSTONE_FIELDS = (
    "schema",
    "kind",
    "tombstones",
    "reason",
    "redacted_fields",
    "evidence_class",
    "created_at",
    "run_id",
    "chain",
)
TOMBSTONE_REASONS = ("forget", "rollback", "redaction")

# --- workflow phase enum (ARCH §3) -------------------------------------------
PHASES = ("simulate", "evals", "optimize", "redteam", "suite", "live")

# --- sync attribute namespace (frozen with platform; PLATFORM-GROUNDING §6) --
FI_KIT_RUN_ID_ATTR = "fi.kit.run_id"
FI_KIT_PHASE_ATTR = "fi.kit.phase"
FI_KIT_WORLD_ATTR = "fi.kit.world"

# --- sync states the viewer renders (UI-UX §1.1 SYNCED column) ---------------
SYNC_STATES = ("local", "metadata", "metadata+content", "queued", "off")


def sync_mode() -> str:
    """The W&B-style telemetry mode (Phase 14). ``off`` when the kill switch is
    set (it binds everything, P8-D6); otherwise ``AGENT_LEARNING_SYNC`` —
    ``auto`` (default: emit when keys resolve) or ``local`` (queue only)."""

    if kill_switch_on():
        return TELEMETRY_OFF_VALUE
    value = os.environ.get(SYNC_MODE_ENV, "").strip().lower()
    return value if value in SYNC_MODES else SYNC_MODE_AUTO


def kill_switch_on() -> bool:
    """``AGENT_LEARNING_TELEMETRY=off`` disables ledger + sync, binding every
    component including vendored ``fi/*`` (P8-D6; the gate's check 1 statically
    verifies every emission path routes through this one guard)."""

    return os.environ.get(TELEMETRY_ENV, "").strip().lower() == TELEMETRY_OFF_VALUE


def ledger_dir(root: str | Path | None = None) -> Path:
    """Resolve the user-owned ledger DIRECTORY (ARCH Decision 9).

    Precedence: explicit ``root`` arg > ``AGENT_LEARNING_LEDGER_PATH`` (the
    directory override for tests/CI/per-project layouts) >
    ``${AGENT_LEARNING_HOME:-~/.agent-learning}/ledger/``.
    """

    if root is not None:
        return Path(root)
    override = os.environ.get(LEDGER_PATH_ENV)
    if override:
        return Path(override)
    home = os.environ.get(LEDGER_HOME_ENV) or (Path.home() / ".agent-learning")
    return Path(home) / LEDGER_DIR_NAME
