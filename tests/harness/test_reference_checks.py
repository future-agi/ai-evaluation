"""A sub-goal's check can be handed the scenario's reference solution and grade correctness on it."""

from __future__ import annotations

import asyncio
from typing import Any

from fi.alk.harness.checks import run_check, takes_reference
from fi.alk.harness.world.runtime import Call

CHECK = '''
def check(world, calls, reference):
    wanted = next((s["arguments"] for s in reference if s["tool"] == "file_claim"), None)
    if wanted is None:
        return None
    made = [c for c in calls if c.name == "file_claim" and c.ok]
    if not made:
        return "not done: no claim was filed"
    rows = [r for r in world.state()["claims"] if r["id"] == made[-1].result.get("claim_id")]
    if not rows:
        return "not done: the claim left no record"
    for field in ("policy", "loss_date"):
        if str(rows[0].get(field)) != str(wanted[field]):
            return f"done wrong: {field} expected {wanted[field]!r}, got {rows[0].get(field)!r}"
    return None
'''
REFERENCE = [{"tool": "file_claim", "arguments": {"policy": "MM-7", "loss_date": "2026-10-11"}}]


class _World:
    def __init__(self, claims: list[dict[str, Any]]) -> None:
        self.claims = claims

    def state(self) -> dict[str, Any]:
        return {"claims": self.claims}


def _filed(**row: Any) -> tuple[_World, list[Call]]:
    call = Call(name="file_claim", arguments=dict(row), result={"claim_id": "c1"}, ok=True)
    return _World([{"id": "c1", **row}]), [call]


def test_a_check_given_the_reference_grades_correctness_from_the_call_and_the_record():
    world, calls = _filed(policy="MM-7", loss_date="2026-10-11")
    assert run_check(CHECK, world, calls, reference=REFERENCE).held
    world, calls = _filed(policy="MM-7", loss_date="2026-10-12")
    said = run_check(CHECK, world, calls, reference=REFERENCE)
    assert not said.held and said.said == "done wrong: loss_date expected '2026-10-11', got '2026-10-12'"
    said = run_check(CHECK, _World([]), [], reference=REFERENCE)
    assert not said.held and said.said == "not done: no claim was filed"


def test_a_two_argument_check_runs_as_before_and_is_never_given_the_reference():
    two = "def check(world, calls):\n    return None if world.state()['claims'] else 'no claim'\n"
    world, calls = _filed(policy="MM-7", loss_date="2026-10-11")
    assert run_check(two, world, calls, reference=REFERENCE).held
    assert not run_check(two, _World([]), []).held
    assert takes_reference(lambda world, calls, reference: None)
    assert not takes_reference(lambda world, calls: None)


