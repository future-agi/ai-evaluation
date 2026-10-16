"""A sub-goal's check can be handed the scenario's reference solution and grade correctness on it."""

from __future__ import annotations

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
    filed = {c.result.get("claim_id") for c in made}
    rows = [r for r in world.state()["claims"] if r["id"] in filed]
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
    assert not takes_reference(type("Deferred", (), {"_alk_source": "def check(:", "_alk_entry": "check"})())


def _call(name: str, ok: bool = True, **arguments: Any) -> Call:
    return Call(name=name, arguments=arguments, result={}, ok=ok)


# Checks written the way the guidance asks, one per row of the comparison table.
EXACT = '''
def check(world, calls, reference):
    want = next(s["arguments"]["account_id"] for s in reference if s["tool"] == "open")
    made = [c for c in calls if c.name == "open" and c.ok]
    if not made:
        return "not done: nothing was opened"
    if not any(str(c.arguments.get("account_id")) == str(want) for c in made):
        return f"done wrong: account_id expected {want!r}"
    return None
'''
FINAL_NO_DUPLICATE = '''
def check(world, calls, reference):
    want = [s["arguments"]["item"] for s in reference if s["tool"] == "book"][-1]
    rows = [r for r in world.state()["bookings"] if r.get("status") != "cancelled"]
    if not rows:
        return "not done: nothing is booked"
    if len(rows) > 1:
        return f"done wrong: {len(rows)} live bookings where one is expected"
    if rows[0]["item"] != want:
        return f"done wrong: item expected {want!r}, got {rows[0]['item']!r}"
    return None
'''
NO_FORBIDDEN_CHANGE = '''
def check(world, calls, reference):
    if any(c.name == "refund" and c.ok for c in calls):
        return "done wrong: a refund was made that the rules forbid"
    return None
'''


def test_case_table_exact_identifier_keeps_leading_zeros():
    ref = [{"tool": "open", "arguments": {"account_id": "0042"}}]
    assert run_check(EXACT, _World([]), [_call("open", account_id="0042")], reference=ref).held
    said = run_check(EXACT, _World([]), [_call("open", account_id="42")], reference=ref)
    assert not said.held and said.said == "done wrong: account_id expected '0042'"


def test_case_table_a_failed_call_is_not_done():
    ref = [{"tool": "open", "arguments": {"account_id": "0042"}}]
    said = run_check(EXACT, _World([]), [_call("open", ok=False, account_id="0042")], reference=ref)
    assert not said.held and said.said.startswith("not done")


class _Bookings:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def state(self) -> dict[str, Any]:
        return {"bookings": self.rows}


def test_case_table_a_correction_counts_by_the_final_value_and_a_leftover_is_a_duplicate():
    ref = [{"tool": "book", "arguments": {"item": "A"}}, {"tool": "book", "arguments": {"item": "B"}}]
    corrected = _Bookings([{"item": "A", "status": "cancelled"}, {"item": "B", "status": "live"}])
    assert run_check(FINAL_NO_DUPLICATE, corrected, [], reference=ref).held
    left = _Bookings([{"item": "A", "status": "live"}, {"item": "B", "status": "live"}])
    said = run_check(FINAL_NO_DUPLICATE, left, [], reference=ref)
    assert not said.held and said.said == "done wrong: 2 live bookings where one is expected"


def test_case_table_a_refusal_checks_the_forbidden_change_is_absent():
    assert run_check(NO_FORBIDDEN_CHANGE, _World([]), [_call("lookup")], reference=[]).held
    assert not run_check(NO_FORBIDDEN_CHANGE, _World([]), [_call("refund")], reference=[]).held


def test_a_check_that_raises_is_an_execution_error_not_an_agent_failure():
    broken = "def check(world, calls, reference):\n    return reference[5]['tool']\n"
    said = run_check(broken, _World([]), [], reference=[])
    assert said.broken and not said.held


def test_a_check_compiled_to_run_in_the_hosted_worker_is_still_given_the_reference():
    from fi.alk.harness.scenario_source import _compile_entry

    deferred = _compile_entry(CHECK, label="t", entry="check", defer_execution=True)
    assert takes_reference(deferred)
    two = _compile_entry("def check(world, calls):\n    return None\n", label="t", entry="check", defer_execution=True)
    assert not takes_reference(two)
    world, calls = _filed(policy="MM-7", loss_date="2026-10-12")
    assert deferred(world, calls, REFERENCE) == "done wrong: loss_date expected '2026-10-11', got '2026-10-12'"


def test_a_reference_check_must_say_where_each_compared_value_comes_from():
    from fi.alk.harness.catalogue import SubGoal, validate_sub_goal

    goal = SubGoal(name="claim_filed", what="the claim is filed for the caller's policy", check=CHECK)
    assert any("where each value it compares comes from" in one for one in validate_sub_goal(goal))
    sourced = goal.model_copy(update={"reference": "policy, loss_date: what the caller said"})
    assert not any("where each value it compares comes from" in one for one in validate_sub_goal(sourced))


def test_a_check_is_given_only_the_arguments_a_step_marks_fixed():
    from fi.alk.harness.checks import reference_of
    from fi.alk.harness.scenario import Step

    marked = Step(tool="hand_over", arguments={"account": "A-1", "queue": "picked"}, fixed={"account": "the caller's record"})
    unmarked = {"tool": "look_up", "arguments": {"account": "A-1"}}
    assert reference_of([marked, unmarked]) == [
        {"tool": "hand_over", "arguments": {"account": "A-1"}},
        {"tool": "look_up", "arguments": {"account": "A-1"}},
    ]



def test_a_fixed_value_must_be_where_its_stated_source_says():
    from fi.alk.harness.scenario import Scenario, Step
    from fi.alk.harness.scenario_tools import _fixed_sources_that_do_not_hold

    rules = "7. Anything needing a specialist is NOT yours to do. Explain briefly and hand it over.\n8. Use the priority line for emergencies."

    class Trial:
        def state(self) -> dict[str, Any]:
            return {"accounts": [{"id": "A-77"}]}

    def problems(source: str, value: str) -> list[str]:
        step = Step(tool="hand_over", arguments={"line": value}, fixed={"line": source})
        return _fixed_sources_that_do_not_hold(Scenario(name="s", instruction="My account is A-12.", solution=[step]), Trial(), rules)

    assert problems("rule 7: specialist line for this", "specialist")
    assert problems('rule: "hand it over"', "specialist")
    assert problems("specialist", "specialist")
    assert problems("caller: they asked for it", "specialist")
    assert not problems('rule: "Use the priority line for emergencies"', "priority")
    assert not problems("caller: says A-12", "A-12")
    assert not problems("record: accounts.id", "A-77")
    assert not problems("record: accounts.id, under the tool's prefix", "account:A-77")
    assert problems("record: accounts.id, under the tool's prefix", "account:A-78")
    assert problems("record: accounts.id, under the tool's prefix", "account:9999")
    assert problems("record: accounts", "id")
    assert problems("record: accounts.id", "7")
    assert not problems("record: accounts.id, in the tool's path form", "accounts/A-77")
    assert not problems("caller: says their account", "my_account")


def test_a_solution_step_must_match_the_agents_contract(tmp_path):
    import json

    from fi.alk.harness.scenario import Scenario, Step
    from fi.alk.harness.scenario_tools import _steps_the_contract_refuses

    tools = [{"name": "hand_over", "args": ["line", "reason"], "arg_values": {"line": ["general", "priority"]}}]
    (tmp_path / "contract.json").write_text(json.dumps({"tools": tools}))

    def problems(*steps: Step) -> list[str]:
        return _steps_the_contract_refuses(Scenario(name="s", instruction="Help.", solution=list(steps)), tmp_path)

    assert not problems(Step(tool="hand_over", arguments={"line": "priority", "reason": "injury"}))
    assert not problems(Step(tool="hand_over", arguments={"line": "$call.look_up.line"}))
    assert problems(Step(tool="look_up", arguments={}))
    assert problems(Step(tool="hand_over", arguments={"queue": "general"}))
    assert problems(Step(tool="hand_over", arguments={"line": "claims"}))


def test_the_proof_varies_every_argument_nothing_fixes_and_keeps_chained_values():
    from fi.alk.harness.prove import _other_choices
    from fi.alk.harness.scenario import Scenario, Step

    scenario = Scenario(
        name="s",
        instruction="i",
        solution=[
            Step(tool="open", arguments={"account": "A-1", "note": "picked"}, fixed={"account": "caller: A-1"}),
            Step(tool="confirm", arguments={"token": "$call.open.token"}, fixed={}),
        ],
    )
    calls = [
        Call(name="open", arguments={"account": "A-1", "note": "picked"}, result={"token": "t1"}, ok=True),
        Call(name="confirm", arguments={"token": "t1"}, result={}, ok=True),
    ]
    varied, changed = _other_choices(scenario, calls)
    assert varied[0].arguments == {"account": "A-1", "note": "another note"}
    assert varied[1].arguments == {"token": "t1"} and changed == {"open.note"}
    strict = "def check(world, calls, reference):\n    return None if any(c.arguments.get('note') == 'picked' for c in calls) else 'done wrong: note'\n"
    assert run_check(strict, _World([]), calls, reference=[]).held
    assert not run_check(strict, _World([]), varied, reference=[]).held
    unfixed = scenario.model_copy(update={"solution": [Step(tool="open", arguments={"note": "x"})]})
    assert _other_choices(unfixed, calls[:1]) == (calls[:1], set())
