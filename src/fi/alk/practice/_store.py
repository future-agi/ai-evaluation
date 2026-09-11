"""Unit 9 (BBG U9 / ARCH §2d) — the consolidation store.

Consolidated records (decks ARE frozen rows, AD-D): record ids use the
optimizer-space frozen-row idiom (``lesson_`` + 16-hex sorted-JSON digest).
Append-only JSONL (AD-G); state transitions are appended full-record snapshots
(latest-wins on read). ``full_deck()`` is the ONLY promotion-row source (13D-D7)
— the FULL union of all P4 frozen rows + every active record's complete deck,
regardless of schedule state. Cap admission REFUSES at active_cap (AD-H), never
evicts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ._contract import (
    LADDER_STATES,
    LESSON_ID_PREFIX,
    LESSON_KINDS,
    PRACTICE_REPLAY_INTERVALS,
    PRACTICE_STORE_ACTIVE_CAP,
    practice_store_path,
)

# fields that do NOT enter the record id (envelope/mutable state).
_NON_ID_FIELDS = {"record_id", "schedule", "history"}


def _sorted_json_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def record_id(body: Mapping[str, Any]) -> str:
    """The frozen-row idiom: lesson_ + 16-hex sorted-JSON digest over all
    non-envelope fields (recipe-agreement with _expected_frozen_row_id)."""
    payload = {k: v for k, v in body.items() if k not in _NON_ID_FIELDS}
    return LESSON_ID_PREFIX + _sorted_json_digest(payload)[:16]


def build_record(
    *,
    lesson: Mapping[str, Any],
    source_justification: Mapping[str, Any],
    deck: List[str],
    cells: List[Any],
    created_round: int,
    seed: int,
    interval_rounds: int = 1,
    due_round: Optional[int] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    ladder_state: str = "episodic",
) -> dict:
    """Construct a consolidated record (ARCH §2d schema verbatim)."""
    if ladder_state not in LADDER_STATES:
        raise ValueError(f"ladder_state {ladder_state!r} not in {LADDER_STATES}")
    if lesson.get("kind") not in LESSON_KINDS:
        raise ValueError(f"lesson.kind {lesson.get('kind')!r} not in {LESSON_KINDS}")
    if interval_rounds not in PRACTICE_REPLAY_INTERVALS:
        raise ValueError(f"interval_rounds {interval_rounds!r} not in {PRACTICE_REPLAY_INTERVALS}")
    body = {
        "kind": "agent-learning.consolidated-lesson.v1",
        "ladder_state": ladder_state,
        "lesson": {
            "kind": lesson["kind"],
            "payload": lesson.get("payload"),
            "applies_to_paths": list(lesson.get("applies_to_paths") or []),
        },
        "source_justification": dict(source_justification),
        "deck": sorted(set(deck)),
        "schedule": {
            "interval_rounds": int(interval_rounds),
            "due_round": int(due_round if due_round is not None else created_round + interval_rounds),
            "consecutive_failures": 0,
            "status": "active",
            "retired_reason": None,
        },
        "cells": list(cells),
        "history": [],
        "created_round": int(created_round),
        "seed": int(seed),
        "provenance": dict(provenance or {}),
    }
    body["record_id"] = record_id(body)
    return body


class ConsolidationStore:
    """Append-only JSONL store under the user-owned home (AD-G)."""

    def __init__(self, path: str | Path | None = None, *, active_cap: int = PRACTICE_STORE_ACTIVE_CAP) -> None:
        self.path = practice_store_path(path)
        self.active_cap = int(active_cap)

    # --- IO (tolerant reader, AD-G) ----------------------------------------
    def _read_snapshots(self) -> List[dict]:
        if not self.path.exists():
            return []
        out: List[dict] = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerant reader (live/_transcript.py philosophy)
        return out

    def _append(self, record: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def latest(self) -> Dict[str, dict]:
        """Latest-wins by record_id over the append-only snapshots."""
        out: Dict[str, dict] = {}
        for snapshot in self._read_snapshots():
            rid = snapshot.get("record_id")
            if rid:
                out[rid] = snapshot
        return out

    def active_records(self) -> List[dict]:
        return [r for r in self.latest().values() if r.get("schedule", {}).get("status") == "active"]

    # --- admission (cap refusal, AD-H) -------------------------------------
    def admit(self, record: Mapping[str, Any]) -> dict:
        """Admit a record, or refuse with cap_deferred at active_cap (refusal
        over eviction — T6/AD-H)."""
        rid = record.get("record_id") or record_id(record)
        existing = self.latest()
        if rid in existing:
            self._append(dict(record))
            return {"admitted": True, "record_id": rid, "reason": "updated"}
        active_count = len([r for r in existing.values()
                            if r.get("schedule", {}).get("status") == "active"])
        if active_count >= self.active_cap:
            return {
                "admitted": False,
                "record_id": rid,
                "status": "cap_deferred",
                "reason": (
                    f"consolidation store at active_cap={self.active_cap}; admission "
                    "REFUSED (a slot frees only via retirement — refusal over eviction)"
                ),
            }
        self._append(dict(record))
        return {"admitted": True, "record_id": rid, "reason": "admitted"}

    def update_record(self, record: Mapping[str, Any]) -> None:
        """Append a full-record snapshot (state transition, never a rewrite)."""
        self._append(dict(record))

    # --- the ONLY promotion-row source (13D-D7) ----------------------------
    def full_deck(self, *, frozen_rows: Optional[List[str]] = None) -> List[str]:
        """The UNION of all P4 frozen rows + every active record's complete deck,
        REGARDLESS of any record's schedule state. This is the 13D-D7 boundary."""
        rows: set[str] = set(frozen_rows or [])
        for record in self.active_records():
            rows.update(record.get("deck") or [])
        return sorted(rows)


def ladder_state(record: Mapping[str, Any]) -> str:
    """Read the stored ladder state (never recomputed from history, AD-H)."""
    return str(record.get("ladder_state") or "episodic")


# --- ladder transitions (up; ARCH §2d promotion ladder) --------------------
_LADDER_ORDER = ("episodic", "instruction", "skill")


def promote_ladder(record: Mapping[str, Any]) -> dict:
    """Move one rung up (episodic→instruction→skill)."""
    out = dict(record)
    current = out.get("ladder_state", "episodic")
    idx = _LADDER_ORDER.index(current)
    if idx < len(_LADDER_ORDER) - 1:
        out["ladder_state"] = _LADDER_ORDER[idx + 1]
    return out


def demote_ladder(record: Mapping[str, Any]) -> dict:
    """Move one rung down (skill→instruction→episodic)."""
    out = dict(record)
    current = out.get("ladder_state", "episodic")
    idx = _LADDER_ORDER.index(current)
    if idx > 0:
        out["ladder_state"] = _LADDER_ORDER[idx - 1]
    return out
