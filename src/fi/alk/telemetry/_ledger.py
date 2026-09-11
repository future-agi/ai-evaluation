"""Append-only run ledger: hash-chain writer + reader + verifier (Phase 8).

Imports: stdlib only plus the package contract. The ledger is a DIRECTORY
(``${AGENT_LEARNING_HOME:-~/.agent-learning}/ledger/``, ARCH Decision 9)
holding ``runs.jsonl`` (one JSON object per line — the exact
``live/_transcript.py`` "the file IS the ledger" model promoted kit-wide),
``chain.head`` (O(1) append sidecar) and ``sync.cursor`` (resumable-sync
bookmark). Sidecars are conveniences: read-back verification is always the
verifier's job, never the writer's correctness dependency.

Rows are NEVER rewritten (ARCH Decision 5): forget/rollback appends a
tombstone row that is itself chained, so content disappears while the chain
stays verifiable. CRDTs are rejected (P8-D4): single writer, append-only,
union-of-verified-chains across machines — no merge path exists.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ..live._contract import AGENT_LEARNING_RUN_KIND
from ._contract import (
    CHAIN_HEAD_FILENAME,
    GAP_SCHEMA,
    GENESIS,
    ROWS_FILENAME,
    SYNC_CURSOR_FILENAME,
    TOMBSTONE_REASONS,
    TOMBSTONE_SCHEMA,
    UNREADABLE_LINE_SCHEMA,
    ledger_dir,
)
from ._row import canonical_row_address

CHAIN_HEAD_KIND = "agent-learning.ledger-chain-head.v1"
SYNC_CURSOR_KIND = "agent-learning.ledger-sync-cursor.v1"
VERIFY_KIND = "agent-learning.ledger-verify.v1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _chain_link(prev: str, run_id: str) -> str:
    """``chain_i = SHA-256(chain_{i-1} || run_id_i)`` (ARCH §2b)."""

    return hashlib.sha256((prev + run_id).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


class RunLedger:
    """Single-writer append-only ledger over ``runs.jsonl`` (P8-D7)."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.dir = ledger_dir(root)
        self.rows_path = self.dir / ROWS_FILENAME
        self.head_path = self.dir / CHAIN_HEAD_FILENAME
        self.cursor_path = self.dir / SYNC_CURSOR_FILENAME

    # -- chain state ---------------------------------------------------------

    def _last_state(self) -> tuple[str, int]:
        """Current ``(chain digest, row count)`` — O(1) via the ``chain.head``
        sidecar, falling back to a linear scan when missing/stale."""

        head = _read_json(self.head_path)
        if head and isinstance(head.get("chain"), str) and head.get("chain"):
            rows = head.get("rows")
            if isinstance(rows, int) and rows >= 0:
                return head["chain"], rows
        last = GENESIS
        count = 0
        for row in self.iter_rows():
            if row.get("schema") == UNREADABLE_LINE_SCHEMA:
                continue
            count += 1
            chain = row.get("chain")
            if isinstance(chain, str) and chain:
                last = chain
        return last, count

    def _write_chain_head(self, chain: str, rows: int) -> None:
        try:
            self.head_path.write_text(
                json.dumps(
                    {"kind": CHAIN_HEAD_KIND, "chain": chain, "rows": rows},
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError:
            # Best-effort sidecar: the verifier never trusts it anyway.
            pass

    # -- write path ----------------------------------------------------------

    def append(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Append one chained row: compute ``chain_i``, stamp ``created_at``
        (envelope — excluded from the address), write one JSONL line."""

        record = dict(row)
        prev, count = self._last_state()
        record["chain"] = _chain_link(prev, str(record.get("run_id") or ""))
        record["created_at"] = _utc_now_iso()
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.rows_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        self._write_chain_head(record["chain"], count + 1)
        return record

    def append_tombstone(
        self,
        *,
        target_run_id: str,
        reason: str,
        redacted_fields: Sequence[str] = ("*",),
        evidence_class: str,
    ) -> dict[str, Any]:
        """Forget/Rollback (ARCH Decision 5, R§3.3): never rewrite a row —
        append a tombstone referencing the withdrawn content address. The
        chain stays verifiable; the content disappears from resolution."""

        if reason not in TOMBSTONE_REASONS:
            reason = "forget"
        tomb: dict[str, Any] = {
            "schema": TOMBSTONE_SCHEMA,
            "kind": AGENT_LEARNING_RUN_KIND,  # same kind family; chained row
            "tombstones": str(target_run_id),
            "reason": reason,
            "redacted_fields": [str(field) for field in redacted_fields],
            "evidence_class": str(evidence_class),
        }
        tomb["run_id"] = canonical_row_address(tomb)
        return self.append(tomb)  # one writer path: chain + created_at

    def append_gap(self, dropped: int) -> dict[str, Any]:
        """Drop-with-gap-marker (ARCH §2c): a bounded-queue overflow drops
        rows with a counter; the next successful append records the loss as a
        chained gap row — recorded, never silent."""

        gap: dict[str, Any] = {
            "schema": GAP_SCHEMA,
            "kind": AGENT_LEARNING_RUN_KIND,
            "dropped": int(dropped),
        }
        gap["run_id"] = canonical_row_address(gap)
        return self.append(gap)

    # -- read path ------------------------------------------------------------

    def iter_rows(self) -> Iterator[dict[str, Any]]:
        """Tolerant reader (the ``read_transcript`` philosophy): unparseable
        lines surface as unreadable-line markers, never a crash."""

        if not self.rows_path.exists():
            return
        with open(self.rows_path, "r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    yield {
                        "schema": UNREADABLE_LINE_SCHEMA,
                        "line_number": line_number,
                    }
                    continue
                if isinstance(obj, dict):
                    yield obj
                else:
                    yield {
                        "schema": UNREADABLE_LINE_SCHEMA,
                        "line_number": line_number,
                    }

    def rows(self) -> list[dict[str, Any]]:
        return list(self.iter_rows())

    # -- verify ----------------------------------------------------------------

    def verify(self) -> dict[str, Any]:
        """One linear pass: recompute every content address (tamper on the
        body) AND every chain link (tamper on order/insertion). Either
        mismatch is a break. Shared verbatim by the viewer and the gate."""

        prev = GENESIS
        breaks: list[dict[str, Any]] = []
        count = 0
        tombstone_count = 0
        gap_count = 0
        gap_dropped_total = 0
        run_ids: set[str] = set()
        unresolved_tombstones: list[str] = []
        for index, row in enumerate(self.iter_rows()):
            if row.get("schema") == UNREADABLE_LINE_SCHEMA:
                breaks.append(
                    {
                        "index": index,
                        "reason": "unreadable_line",
                        "line_number": row.get("line_number"),
                    }
                )
                continue
            count += 1
            recomputed_id = canonical_row_address(row)
            if row.get("run_id") != recomputed_id:
                breaks.append(
                    {
                        "index": index,
                        "reason": "content_address_mismatch",
                        "run_id": row.get("run_id"),
                        "recomputed": recomputed_id,
                    }
                )
            expected = _chain_link(prev, str(row.get("run_id") or ""))
            if row.get("chain") != expected:
                breaks.append(
                    {
                        "index": index,
                        "reason": "chain_mismatch",
                        "chain": row.get("chain"),
                        "expected": expected,
                    }
                )
            prev = row.get("chain") or prev
            run_id = row.get("run_id")
            if isinstance(run_id, str):
                run_ids.add(run_id)
            if row.get("schema") == TOMBSTONE_SCHEMA:
                tombstone_count += 1
                target = str(row.get("tombstones") or "")
                if target not in run_ids:
                    unresolved_tombstones.append(target)
            elif row.get("schema") == GAP_SCHEMA:
                gap_count += 1
                dropped = row.get("dropped")
                if isinstance(dropped, int):
                    gap_dropped_total += dropped
        return {
            "kind": VERIFY_KIND,
            "genesis": GENESIS,
            "ledger": str(self.rows_path),
            "row_count": count,
            "chain_intact": not breaks,
            "breaks": breaks,
            "tombstone_count": tombstone_count,
            "unresolved_tombstones": unresolved_tombstones,
            "gap_count": gap_count,
            "gap_dropped_total": gap_dropped_total,
        }

    # -- sync cursor (idempotency bookmark — never a correctness dependency) --

    def read_cursor(self) -> dict[str, Any]:
        cursor = _read_json(self.cursor_path)
        if not cursor:
            return {"kind": SYNC_CURSOR_KIND, "cursor": None, "synced": {}}
        synced = cursor.get("synced")
        return {
            "kind": SYNC_CURSOR_KIND,
            "cursor": cursor.get("cursor"),
            "synced": dict(synced) if isinstance(synced, Mapping) else {},
        }

    def write_cursor(self, run_id: str, channel: str) -> dict[str, Any]:
        cursor = self.read_cursor()
        cursor["synced"][str(run_id)] = str(channel)
        cursor["cursor"] = str(run_id)  # high-water: last confirmed address
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            self.cursor_path.write_text(
                json.dumps(cursor, sort_keys=True) + "\n", encoding="utf-8"
            )
        except OSError:
            pass
        return cursor
