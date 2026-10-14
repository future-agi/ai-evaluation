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
# Shorter values (a count, a flag) trace to the solution only under the argument name they came from.
_SHORTEST_TRACED = 3
# A time matches to the minute unless the scenario allows a tolerance for that field.
_TIME_TOLERANCE = timedelta(hours=float(os.environ.get("ALK_REFERENCE_TIME_TOLERANCE_HOURS") or 0))
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
    if re.fullmatch(r"0\d+", text):
        return text
    try:
        return f"{float(text.replace(',', '')):g}"
    except ValueError:
        pass
    return re.sub(r"[\W_]+", "", text.casefold())


def _rule(rules: dict[str, Any] | None, table: str, field: str) -> dict[str, Any]:
    """The scenario's own comparison rule for a field, by `table.field` or by field alone."""
    rules = rules or {}
    found = rules.get(f"{table}.{field}", rules.get(field))
    return found if isinstance(found, dict) else {}


def _same(expected: Any, actual: Any, rule: dict[str, Any] | None = None) -> bool:
    """Whether a stored value is the expected one under the rule the scenario's author set for
    the field (`exact`, `aliases`, `tolerance_minutes`, or `skip` to leave it to the judge), or
    else the same after normalising."""
    rule = rule or {}
    match = str(rule.get("match") or "").lower()
    if match == "skip":
        return True
    if match == "exact":
        return str(expected) == str(actual)
    aliases = [expected, *(rule.get("aliases") or [])]
    when, then = _when(expected), _when(actual)
    if when is not None and then is not None:
        minutes = rule.get("tolerance_minutes")
        tolerance = timedelta(minutes=float(minutes)) if minutes is not None else _TIME_TOLERANCE
        return abs(when - then) <= tolerance + timedelta(seconds=59)
    return any(normalise(one) == normalise(actual) for one in aliases)


def _given(solution: list[Any]) -> tuple[set[str], set[tuple[str, str]]]:
    """Every value the reference solution passed, normalised, nested values included, alone and with
    the argument name it was passed under; a date or a time also by its calendar day, so a stored
    timestamp still traces to the date it was given."""
    found: set[str] = set()
    named: set[tuple[str, str]] = set()

    def walk(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for name, item in value.items():
                walk(item, str(name))
        elif isinstance(value, list):
            for item in value:
                walk(item, key)
        elif value not in (None, "") and len(str(value)) <= _LONGEST_COMPARED:
            forms = {normalise(value)}
            if (when := _when(value)) is not None:
                forms.add(when.date().isoformat())
            found.update(forms)
            named.update((normalise(key), form) for form in forms)

    for step in solution or []:
        arguments = step.get("arguments") if isinstance(step, dict) else getattr(step, "arguments", None)
        walk(arguments or {})
    return found, named


def _traced(field: str, value: Any, given: tuple[set[str], set[tuple[str, str]]]) -> bool:
    """Whether a stored value came from the solution: any long value it passed, or a short one
    (a count, a flag, a code) only under the same name, since short values repeat by chance."""
    found, named = given
    forms = {normalise(value)}
    if (when := _when(value)) is not None:
        forms.add(when.date().isoformat())
    if any(len(form) >= _SHORTEST_TRACED and form in found for form in forms):
        return True
    return any((normalise(field), form) in named for form in forms)


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
                and _traced(field, value, given)
            }
            for row in rows
        ]
        kept = [row for row in kept if row]
        if kept:
            expected[name] = kept
    return expected


def _flat(arguments: Any, prefix: str = "") -> dict[str, Any]:
    """Arguments as `field` or `field.sub` to a comparable value; long free text is the judge's."""
    found: dict[str, Any] = {}
    for key, value in (arguments or {}).items() if isinstance(arguments, dict) else ():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            found.update(_flat(value, f"{name}."))
        elif value not in (None, "") and not isinstance(value, list) and len(str(value)) <= _LONGEST_COMPARED:
            found[name] = value
    return found


def compare_calls(
    solution: list[Any], calls: list[Any], *, rules: dict[str, Any] | None = None
) -> str | None:
    """None when, for every solution step marked as the outcome, the call's last successful call
    to that tool carries the step's arguments; otherwise whether it was not done, done wrong or
    done twice. None too when no step is marked, so older scenarios are not graded this way."""
    said = []
    steps = [
        step for step in solution or ()
        if (step.get("outcome") if isinstance(step, dict) else getattr(step, "outcome", False)) is True
    ]
    for step in steps:
        tool = str(step.get("tool") if isinstance(step, dict) else getattr(step, "tool", ""))
        want = _flat(step.get("arguments") if isinstance(step, dict) else getattr(step, "arguments", {}))
        done = [
            call for call in calls
            if getattr(call, "name", "") == tool and getattr(call, "ok", False) and not getattr(call, "refused", False)
        ]
        if not done:
            stated = ", ".join(f"{field} {value}" for field, value in sorted(want.items()))
            said.append(f"not done: no successful {tool} call" + (f"; expected one with {stated}" if stated else ""))
            continue
        got = _flat(getattr(done[-1], "arguments", {}))
        said.extend(
            f"done wrong: {tool}.{field} expected {value}, got {got.get(field)}"
            for field, value in sorted(want.items())
            if not _same(value, got.get(field), _rule(rules, tool, field))
        )
        same = sum(
            all(_same(v, _flat(getattr(c, "arguments", {})).get(f), _rule(rules, tool, f)) for f, v in want.items())
            for c in done
        )
        wanted = sum(
            1 for one in steps
            if str(one.get("tool") if isinstance(one, dict) else getattr(one, "tool", "")) == tool
        )
        if want and same > wanted:
            said.append(f"done wrong: {same} successful {tool} calls with these values where the reference makes {wanted} (a duplicate)")
    return "; ".join(dict.fromkeys(said)) or None


def compare(
    expected: dict[str, list[dict[str, Any]]],
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    rules: dict[str, Any] | None = None,
    expect_nothing: bool = False,
) -> str | None:
    """None when the final records hold every expected row and nothing more of it; otherwise, per
    row, whether it was not done at all, done wrong or done twice. With `expect_nothing` (the
    reference solution changed no record) any change the call made is reported."""
    actual = changed_rows(before, after)
    said = []
    if expect_nothing and not expected:
        for name in sorted(actual):
            said.append(
                f"done wrong: {len(actual[name])} {name} row(s) added or changed where the "
                "reference solution changes nothing"
            )
        return "; ".join(said) or None
    for name in sorted(expected):
        candidates = actual.get(name, [])

        def matches(row: dict[str, Any], want: dict[str, Any]) -> int:
            return sum(
                _same(value, row.get(field), _rule(rules, name, field))
                for field, value in want.items()
            )

        for want in expected[name]:
            if not candidates:
                stated = ", ".join(f"{field} {value}" for field, value in sorted(want.items()))
                said.append(f"not done: no {name} row added or changed; expected one with {stated}")
                continue
            best = max(candidates, key=lambda row: matches(row, want))
            said.extend(
                f"done wrong: {name}.{field} expected {value}, got {best.get(field)}"
                for field, value in sorted(want.items())
                if not _same(value, best.get(field), _rule(rules, name, field))
            )
        for want in {_key(one): one for one in expected[name]}.values():
            wanted = sum(_key(one) == _key(want) for one in expected[name])
            done = sum(matches(row, want) == len(want) for row in candidates)
            if done > wanted:
                said.append(
                    f"done wrong: {done} {name} rows match where the reference solution leaves "
                    f"{wanted} (a duplicate)"
                )
    return "; ".join(said) or None
