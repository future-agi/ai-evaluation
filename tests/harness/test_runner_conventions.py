"""Pins the runner return conventions in world-handle-interface.md v3.2's "Return conventions"
block: a check returning ``""``/whitespace counts as held, and a bare ``False`` from ``ready()``
is broken rather than a plain not-ready. The unchanged neighbors are pinned alongside them so a
future edit to either runner cannot drift the other conventions without a test noticing.
"""

from __future__ import annotations

import pytest

from fi.alk.harness.checks import run_check
from fi.alk.harness.folder import _run, check_ready
from fi.alk.harness.scenario import Scenario

# --- checks.py: run_check's return convention -------------------------------------------------


@pytest.mark.parametrize("returning", ["None", "True", "''", "'   '"])
def test_check_returning_nothing_or_whitespace_holds(returning: str) -> None:
    outcome = run_check(f"def check(world, calls):\n    return {returning}\n", None, [])
    assert outcome.held and not outcome.broken and outcome.said == ""


def test_check_returning_a_non_empty_string_does_not_hold() -> None:
    outcome = run_check("def check(world, calls):\n    return 'no rows'\n", None, [])
    assert not outcome.held and not outcome.broken and outcome.said == "no rows"


def test_check_returning_bare_false_does_not_hold_with_reason_false() -> None:
    """An agent result, matching checks.py's existing convention — not a broken check."""
    outcome = run_check("def check(world, calls):\n    return False\n", None, [])
    assert not outcome.held and not outcome.broken and outcome.said == "False"


@pytest.mark.parametrize("returning", ["0", "1", "42", "[]", "{}", "0.0", "object()"])
def test_check_returning_any_other_value_is_broken(returning: str) -> None:
    """Anything that is not the held set, a non-empty string, or bare False is our own mistake,
    not a finding about the agent — a check that returns 0 by accident must not be scored as a
    failing sub-goal with an uninterpretable reason."""
    outcome = run_check(f"def check(world, calls):\n    return {returning}\n", None, [])
    assert not outcome.held and outcome.broken


# --- folder.py: ready()'s return convention ---------------------------------------------------


@pytest.mark.parametrize("returning", ["None", "True", "''", "'   '"])
def test_ready_returning_nothing_or_whitespace_is_ready(returning: str) -> None:
    outcome = _run(f"def ready(world):\n    return {returning}\n", "s/ready.py", "ready", None)
    assert outcome.ok and not outcome.broken and outcome.said == ""


def test_ready_returning_a_non_empty_string_is_not_ready() -> None:
    outcome = _run(
        "def ready(world):\n    return 'orders pending'\n", "s/ready.py", "ready", None
    )
    assert not outcome.ok and not outcome.broken and outcome.said == "orders pending"


def test_ready_returning_bare_false_is_broken_not_advisory() -> None:
    """The behavioral change: ready() cannot name what it wants, so a bare False is our own
    mistake, not a precondition failing on the shared sealed baseline."""
    outcome = _run("def ready(world):\n    return False\n", "s/ready.py", "ready", None)
    assert not outcome.ok and outcome.broken


def test_setup_returning_bare_false_stays_advisory_not_broken() -> None:
    """The change is scoped to ready() only; setup()'s own convention is untouched."""
    outcome = _run("def setup(world):\n    return False\n", "s/setup.py", "setup", None)
    assert not outcome.ok and not outcome.broken


@pytest.mark.parametrize("returning", ["0", "1", "[]", "{}", "0.0"])
def test_ready_returning_any_other_value_is_broken(returning: str) -> None:
    """ready() cannot turn a non-string, non-False value into a precondition sentence either, so
    it gets the same broken verdict bare False does."""
    outcome = _run(f"def ready(world):\n    return {returning}\n", "s/ready.py", "ready", None)
    assert not outcome.ok and outcome.broken


@pytest.mark.parametrize("returning", ["0", "1", "[]", "{}", "0.0"])
def test_setup_returning_any_other_value_stays_advisory(returning: str) -> None:
    """The widened rule is keyed on entry == "ready", not on the value alone — setup() keeps its
    own untouched convention for every one of these values too."""
    outcome = _run(f"def setup(world):\n    return {returning}\n", "s/setup.py", "setup", None)
    assert not outcome.broken and not outcome.ok


def test_check_ready_reports_bare_false_as_broken() -> None:
    """Pins the production entry point, not just `_run`: `check_ready` is the only caller that
    can ever produce entry == "ready", so a test that only calls `_run` directly would not catch
    a rename or rewiring that silently reverted this."""
    outcome = check_ready(
        Scenario(name="s", ready_code="def ready(world):\n    return False\n"), None
    )
    assert not outcome.ok and outcome.broken
