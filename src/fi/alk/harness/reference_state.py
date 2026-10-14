"""Whether the records a call left match the ones the scenario's reference solution leaves.

A code check settles that a record changed; this settles that it changed to the caller's values. The
reference solution is played at build time and the rows it adds or changes are kept, narrowed to the
fields whose values the solution itself supplied (what the caller said or the instructions fix).
After a call, the rows it added or changed are compared on those fields alone, so generated ids,
timestamps, the order of calls and a mistake corrected before the end do not matter.
"""

from __future__ import annotations

import ast
import json
import os
import re
from datetime import datetime, timedelta
from typing import Any

NAME = "records_match_reference_solution"
# Longer values are free text (a description, a note, a reason), which the judge reads instead.
_LONGEST_COMPARED = 80
_MOST_WORDS_COMPARED = 3
# Shorter values (a count, a flag) trace to the solution only under the argument name they came from.
_SHORTEST_TRACED = 3
# A time matches to the minute unless the scenario allows a tolerance for that field.
_TIME_TOLERANCE = timedelta(hours=float(os.environ.get("ALK_REFERENCE_TIME_TOLERANCE_HOURS") or 0))
_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y", "%d %B %Y")


def _comparable(value: Any) -> bool:
    """A value with one right form: short, and not a phrase the agent may word its own way."""
    text = str(value)
    return value not in (None, "") and len(text) <= _LONGEST_COMPARED and len(text.split()) <= _MOST_WORDS_COMPARED


def _rows(value: Any) -> list[dict[str, Any]]:
    """One collection of a world's state as records, whatever it is kept as: a table's rows, a
    mapping (files by path, an agent's own memory by key) or a single value."""
    if isinstance(value, list):
        return [row if isinstance(row, dict) else {"value": row} for row in value]
    if isinstance(value, dict):
        return [
            {"_key": key, **item} if isinstance(item, dict) else {"_key": key, "value": item}
            for key, item in value.items()
        ]
    return [] if value is None else [{"value": value}]


def _tables(state: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    return {str(name): _rows(value) for name, value in (state or {}).items()}


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


_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december")


def given_text(*facts: Any, said: Any = "", world: dict[str, Any] | None = None) -> str:
    """What a scenario fixes as one normalised text: its structured facts (the caller's details,
    the fixture), the identifiers, numbers and dates in what the caller is told, and the values
    that identify one record in the world. A word in a sentence vouches for nothing."""
    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif value not in (None, "") and not isinstance(value, bool):
            found.append(normalise(value))

    for part in facts:
        walk(part)
    text = str(said or "")
    found.extend(normalise(token) for token in re.findall(r"[\w$.,:/+-]*\d[\w.,:/+-]*", text))
    words = re.sub(r"[\W_]+", "", text.casefold())
    found.extend(f"{month}{day}" for month in _MONTHS for day in range(1, 32) if f"{month}{day}" in words)
    counts: dict[str, int] = {}
    for rows in _tables(world).values():
        for row in rows:
            for value in row.values():
                if value not in (None, "") and not isinstance(value, (bool, dict, list)):
                    counts[normalise(value)] = counts.get(normalise(value), 0) + 1
    found.extend(form for form, seen in counts.items() if seen == 1)
    return " " + " ".join(form for form in found if form) + " "


def _from_scenario(value: Any, given: str | None) -> bool:
    """Whether a value is a fact the scenario or the world fixed, rather than something the
    reference's author chose (a wording, a routing, a flag): only those have one right answer."""
    if given is None:
        return True
    if isinstance(value, bool):
        return False
    form = normalise(value)
    if form and f" {form} " in given:
        return True
    when = _when(value)
    if when is not None:
        month = _MONTHS[when.month - 1]
        return f" {normalise(when.date().isoformat())} " in given or f" {month}{when.day} " in given
    return False


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
        elif _comparable(value):
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
                if _comparable(value)
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
        elif not isinstance(value, list) and _comparable(value):
            found[name] = value
    return found


def _effect(call: Any) -> str:
    """What a call did, as one key: the same effect reported twice (a trace that records a call
    from two sources) has the same result, a second effect has its own."""
    result = getattr(call, "result", None)
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError:
            try:
                result = ast.literal_eval(result)
            except (ValueError, SyntaxError):
                pass
    if result in (None, "", {}, []):
        return f"call:{id(call)}"
    return json.dumps(result, sort_keys=True, default=str)


def _effects(calls: list[Any]) -> list[dict[str, Any]]:
    """Each distinct effect once, in order, with the arguments of every report of it merged."""
    merged: dict[str, dict[str, Any]] = {}
    for call in calls:
        merged.setdefault(_effect(call), {}).update(_flat(getattr(call, "arguments", {})))
    return list(merged.values())


def compare_calls(
    solution: list[Any],
    calls: list[Any],
    *,
    rules: dict[str, Any] | None = None,
    given: str | None = None,
) -> str | None:
    """None when every solution step marked as the outcome has a successful call to its tool
    carrying its arguments; otherwise whether it was not done, done wrong or done twice. Each step
    is matched to the call that fits it best, the later one on a tie, so the same tool used twice
    (two addresses) and a value corrected mid-call both read right. None too when no step is
    marked, so older scenarios are not graded this way."""
    said = []

    def field_of(step: Any, name: str) -> Any:
        return step.get(name) if isinstance(step, dict) else getattr(step, name, None)

    steps = [step for step in solution or () if field_of(step, "outcome") is True]
    for step in steps:
        tool = str(field_of(step, "tool") or "")
        want = {
            field: value
            for field, value in _flat(field_of(step, "arguments") or {}).items()
            if _from_scenario(value, given)
        }
        done = [
            call for call in calls
            if getattr(call, "name", "") == tool and getattr(call, "ok", False) and not getattr(call, "refused", False)
        ]
        if not want:
            if not done:
                said.append(f"not done: no successful {tool} call")
            continue
        if not done:
            stated = ", ".join(f"{field} {value}" for field, value in sorted(want.items()))
            said.append(f"not done: no successful {tool} call" + (f"; expected one with {stated}" if stated else ""))
            continue
        effects = _effects(done)

        def fit(effect: dict[str, Any]) -> int:
            return sum(_same(v, effect.get(f), _rule(rules, tool, f)) for f, v in want.items())

        got = max(reversed(effects), key=fit)
        said.extend(
            f"done wrong: {tool}.{field} expected {value}, got {got.get(field)}"
            for field, value in sorted(want.items())
            if not _same(value, got.get(field), _rule(rules, tool, field))
        )
        same = sum(fit(effect) == len(want) for effect in effects)
        wanted = sum(
            1 for one in steps
            if str(field_of(one, "tool") or "") == tool
            and {f: v for f, v in _flat(field_of(one, "arguments") or {}).items() if _from_scenario(v, given)} == want
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
    given: str | None = None,
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
    expected = {
        name: [want for want in (
            {field: value for field, value in row.items() if _from_scenario(value, given)} for row in rows
        ) if want]
        for name, rows in expected.items()
    }
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
