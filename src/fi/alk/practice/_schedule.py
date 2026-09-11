"""Unit 9 (BBG U9 / ARCH §2d) — the spaced-replay schedule state machine.

The T1-T7 transition table as a PURE function. Public surface: ``due_reviews``
ONLY. This module has NO code path into promotion (the update module is the only
promotion invoker, obtaining rows exclusively from the store's deck union) — the
13D-D7 boundary is structural, not disciplinary. There is deliberately NO import
of the promotion invoker here.
"""
from __future__ import annotations

from typing import Any, List, Mapping

from ._contract import MAX_REPLAY_INTERVAL
from ._store import demote_ladder

_LADDER_ORDER = ("episodic", "instruction", "skill")


def _append_history(record: dict, round_no: int, event: str, outcome: str) -> None:
    history = list(record.get("history") or [])
    history.append({"round": int(round_no), "event": str(event), "outcome": str(outcome)})
    record["history"] = history


def transition(record: Mapping[str, Any], event: str, round_no: int) -> dict:
    """The T1-T7 table. ``event`` ∈ {'review_pass', 'review_fail', 'obsolete'}.
    Returns a NEW record snapshot (every transition appends to history)."""
    out = dict(record)
    out["schedule"] = dict(out.get("schedule") or {})
    schedule = out["schedule"]
    ladder = out.get("ladder_state", "episodic")

    if schedule.get("status") == "retired":
        # T7: retired is terminal.
        _append_history(out, round_no, event, "terminal_retired")
        return out

    if event == "obsolete":
        # T5: obsolescence ⇒ retired "obsolete".
        schedule["status"] = "retired"
        schedule["retired_reason"] = "obsolete"
        _append_history(out, round_no, "obsolete", "retired")
        return out

    if event == "review_pass":
        # T1: interval ← min(2·i, 16); due ← round+interval; failures ← 0.
        interval = int(schedule.get("interval_rounds", 1))
        new_interval = min(2 * interval, MAX_REPLAY_INTERVAL)
        schedule["interval_rounds"] = new_interval
        schedule["due_round"] = int(round_no) + new_interval
        schedule["consecutive_failures"] = 0
        _append_history(out, round_no, "review_pass", f"interval->{new_interval}")
        return out

    if event == "review_fail":
        failures = int(schedule.get("consecutive_failures", 0)) + 1
        schedule["consecutive_failures"] = failures
        if failures >= 2:
            # T4: failures ≥ 2 ⇒ retired "repeated_failure".
            schedule["status"] = "retired"
            schedule["retired_reason"] = "repeated_failure"
            _append_history(out, round_no, "review_fail", "retired_repeated_failure")
            return out
        if ladder == "episodic":
            # T3: fail at episodic ⇒ retired "repeated_failure" (no rung below).
            schedule["status"] = "retired"
            schedule["retired_reason"] = "repeated_failure"
            _append_history(out, round_no, "review_fail", "retired_episodic_fail")
            return out
        # T2: fail above episodic ⇒ demote one rung; interval←1; due←round+1.
        demoted = demote_ladder(out)
        out["ladder_state"] = demoted["ladder_state"]
        schedule["interval_rounds"] = 1
        schedule["due_round"] = int(round_no) + 1
        _append_history(out, round_no, "review_fail", f"demote->{out['ladder_state']}")
        return out

    raise ValueError(f"unknown schedule event {event!r}")


def due_reviews(records: List[Mapping[str, Any]], round_no: int) -> List[dict]:
    """Select due active records, sorted by (due_round, record_id) — deterministic
    tie-break by content id. The ONLY public path from this module."""
    due = [
        dict(r)
        for r in records
        if r.get("schedule", {}).get("status") == "active"
        and int(r.get("schedule", {}).get("due_round", 0)) <= int(round_no)
    ]
    due.sort(key=lambda r: (int(r["schedule"]["due_round"]), str(r.get("record_id", ""))))
    return due


def mark_obsolete_if_path_left_space(
    record: Mapping[str, Any], search_paths: List[str], round_no: int
) -> dict:
    """T5 obsolescence trigger helper: a deck row's config path leaves the search
    space ⇒ retire 'obsolete'. (Caller supplies the current search space.)"""
    applies = record.get("lesson", {}).get("applies_to_paths") or []
    if applies and not any(p in set(search_paths) for p in applies):
        return transition(record, "obsolete", round_no)
    return dict(record)
