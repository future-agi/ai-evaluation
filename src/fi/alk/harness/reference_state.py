"""Whether the records a call left match the ones the scenario's reference solution leaves.

A code check settles that a record changed; this settles that it changed to the caller's values. The
reference solution is played at build time and the rows it adds or changes are kept, narrowed to the
fields whose values the solution itself supplied (what the caller said or the instructions fix).
After a call, the rows it added or changed are compared on those fields alone, so generated ids,
timestamps, the order of calls and a mistake corrected before the end do not matter.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from typing import Any

NAME = "records_match_reference_solution"
# Longer values are free text (a description, a note), which the judge reads instead.
_LONGEST_COMPARED = 80
_TIME_TOLERANCE = timedelta(hours=float(os.environ.get("ALK_REFERENCE_TIME_TOLERANCE_HOURS") or 6))
_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y", "%d %B %Y")


def _tables(state: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    return {
        name: [row for row in rows if isinstance(row, dict)]
        for name, rows in (state or {}).items()
        if isinstance(rows, list)
    }


def _key(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, default=str)


def changed_rows(before: dict[str, Any], after: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Rows present after that were not present, exactly, before: added rows and new versions."""
    earlier = _tables(before)
    changed: dict[str, list[dict[str, Any]]] = {}
    for name, rows in _tables(after).items():
        seen = {_key(row) for row in earlier.get(name, [])}
        new = [row for row in rows if _key(row) not in seen]
        if new:
            changed[name] = new
    return changed


def _when(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if not isinstance(value, str) or not re.search(r"\d", value):
        return None
    text = value.strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def normalise(value: Any) -> str:
    """One spelling per value: case, spacing, punctuation, phone digits and number forms."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{float(value):g}"
    text = str(value).strip()
    if re.fullmatch(r"[+\d][\d\s().-]{6,}", text):
        return re.sub(r"\D", "", text)
    try:
        return f"{float(text.replace(',', '')):g}"
    except ValueError:
        pass
    return re.sub(r"[\W_]+", "", text.casefold())


def _same(expected: Any, actual: Any) -> bool:
    when, then = _when(expected), _when(actual)
    if when is not None and then is not None:
        return abs(when - then) <= _TIME_TOLERANCE
    return normalise(expected) == normalise(actual)


def _given(solution: list[Any]) -> set[str]:
    """Every value the reference solution passed, normalised, nested values included; a date or a
    time also by its calendar day, so a stored timestamp still traces to the date it was given."""
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif value not in (None, "") and len(str(value)) <= _LONGEST_COMPARED:
            found.add(normalise(value))
            if (when := _when(value)) is not None:
                found.add(when.date().isoformat())

    for step in solution or []:
        arguments = step.get("arguments") if isinstance(step, dict) else getattr(step, "arguments", None)
        walk(arguments or {})
    return found


def expected_changes(
    before: dict[str, Any], after: dict[str, Any], solution: list[Any]
) -> dict[str, list[dict[str, Any]]]:
    """The rows the solution added or changed, kept only on fields whose value it supplied."""
    given = _given(solution)
    expected: dict[str, list[dict[str, Any]]] = {}
    for name, rows in changed_rows(before, after).items():
        kept = [
            {
                field: value
                for field, value in row.items()
                if value not in (None, "")
                and len(str(value)) <= _LONGEST_COMPARED
                and (
                    normalise(value) in given
                    or ((when := _when(value)) is not None and when.date().isoformat() in given)
                )
            }
            for row in rows
        ]
        kept = [row for row in kept if row]
        if kept:
            expected[name] = kept
    return expected


def compare(
    expected: dict[str, list[dict[str, Any]]], before: dict[str, Any], after: dict[str, Any]
) -> str | None:
    """None when the final records hold every expected row; otherwise, per row, whether it was
    not done at all or done wrong."""
    actual = changed_rows(before, after)
    said = []
    for name in sorted(expected):
        candidates = actual.get(name, [])
        for want in expected[name]:
            if not candidates:
                stated = ", ".join(f"{field} {value}" for field, value in sorted(want.items()))
                said.append(f"not done: no {name} row added or changed; expected one with {stated}")
                continue
            best = max(
                candidates,
                key=lambda row: sum(_same(value, row.get(field)) for field, value in want.items()),
            )
            said.extend(
                f"done wrong: {name}.{field} expected {value}, got {best.get(field)}"
                for field, value in sorted(want.items())
                if not _same(value, best.get(field))
            )
    return "; ".join(said) or None
