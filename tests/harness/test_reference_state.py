from __future__ import annotations

from fi.alk.harness.reference_state import changed_rows, compare, expected_changes, normalise

BEFORE = {"claims": [{"id": "c1", "policy": "MM-1", "loss_date": "2026-09-01"}], "audit": []}
SOLUTION = [{"tool": "file_claim", "arguments": {"policy": "MM-5590318", "loss_date": "2026-10-11"}}]


def _after(**row):
    return {
        "claims": BEFORE["claims"] + [{"id": "generated-9", "filed_at": "2026-10-14T10:00:00", **row}],
        "audit": [{"event": "x"}],
    }


def test_expected_keeps_only_fields_the_solution_supplied():
    expected = expected_changes(BEFORE, _after(policy="MM-5590318", loss_date="2026-10-11"), SOLUTION)
    assert expected == {"claims": [{"policy": "MM-5590318", "loss_date": "2026-10-11"}]}


def test_a_call_leaving_the_same_records_passes_whatever_ids_or_extra_rows():
    expected = {"claims": [{"policy": "MM-5590318", "loss_date": "2026-10-11"}]}
    after = _after(policy="mm 5590318", loss_date="2026-10-11T00:00:00")
    after["claims"][-1]["id"] = "another-id"
    assert compare(expected, BEFORE, after) is None


def test_a_wrong_value_says_where():
    expected = {"claims": [{"policy": "MM-5590318", "loss_date": "2026-10-11"}]}
    said = compare(expected, BEFORE, _after(policy="MM-5590318", loss_date="2026-10-12"))
    assert said == "done wrong: claims.loss_date expected 2026-10-11, got 2026-10-12"


def test_a_missing_row_says_what_was_expected():
    expected = {"claims": [{"policy": "MM-5590318"}]}
    said = compare(expected, BEFORE, {"claims": BEFORE["claims"]})
    assert said == "not done: no claims row added or changed; expected one with policy MM-5590318"


def test_the_final_state_wins_over_a_corrected_mistake():
    expected = {"claims": [{"policy": "MM-5590318", "loss_date": "2026-10-11"}]}
    after = _after(policy="MM-5590318", loss_date="2026-10-11")
    after["claims"].insert(1, {"id": "wrong-1", "policy": "MM-5590318", "loss_date": "2026-01-01", "status": "void"})
    assert compare(expected, BEFORE, after) is None


def test_normalising_case_spacing_punctuation_phones_and_numbers():
    assert normalise("  Interstate  280, Palo Alto. ") == normalise("interstate 280 palo alto")
    assert normalise("+1 (555) 010-2233") == normalise("15550102233")
    assert normalise("120") == normalise(120.0) == normalise("120.00")
    assert normalise("0123") != normalise("123")


def test_times_match_to_the_minute_unless_the_scenario_allows_a_tolerance():
    expected = {"bookings": [{"pickup_at": "2026-10-15T16:00:00"}]}
    before = {"bookings": []}
    assert compare(expected, before, {"bookings": [{"pickup_at": "2026-10-15T16:00:30+00:00"}]}) is None
    assert compare(expected, before, {"bookings": [{"pickup_at": "2026-10-15T16:20:00"}]}) is not None
    rules = {"bookings.pickup_at": {"tolerance_minutes": 30}}
    assert compare(expected, before, {"bookings": [{"pickup_at": "2026-10-15T16:20:00"}]}, rules=rules) is None


def test_exact_ids_keep_leading_zeros_and_an_exact_rule_keeps_case():
    before = {"accounts": []}
    assert compare({"accounts": [{"pin": "0742"}]}, before, {"accounts": [{"pin": "742"}]}) is not None
    rules = {"name": {"match": "exact"}}
    assert compare({"accounts": [{"name": "Reshav"}]}, before, {"accounts": [{"name": "reshav"}]}, rules=rules)


def test_another_form_passes_only_as_a_declared_alias_and_a_skipped_field_is_not_compared():
    before = {"trips": []}
    want = {"trips": [{"pickup": "12 Main Street, Apartment 4", "day": "2026-10-15"}]}
    got = {"trips": [{"pickup": "12 Main St Apt 4", "day": "2026-10-17"}]}
    assert compare(want, before, got) is not None
    rules = {"trips.pickup": {"aliases": ["12 Main St Apt 4"]}, "day": {"match": "skip"}}
    assert compare(want, before, got, rules=rules) is None
    assert compare(want, before, {"trips": [{"pickup": "12 Main St Apt 5", "day": "x"}]}, rules=rules)


def test_names_pass_only_by_spelling_or_a_declared_alias():
    before = {"guests": []}
    want = {"guests": [{"name": "Rishav"}]}
    assert compare(want, before, {"guests": [{"name": "Reshav"}]}) is not None
    rules = {"guests.name": {"aliases": ["Reshav"]}}
    assert compare(want, before, {"guests": [{"name": "Reshav"}]}, rules=rules) is None


def test_a_duplicate_booking_fails_even_when_one_is_right():
    want = {"bookings": [{"pickup": "SFO", "rider": "Ana"}]}
    twice = {"bookings": [{"id": 1, "pickup": "SFO", "rider": "Ana"}, {"id": 2, "pickup": "SFO", "rider": "Ana"}]}
    said = compare(want, {"bookings": []}, twice)
    assert said is not None and "duplicate" in said


def test_a_wrong_value_corrected_before_the_end_passes():
    want = {"bookings": [{"pickup": "SFO"}]}
    corrected = {"bookings": [{"id": 1, "pickup": "SFO"}]}
    assert compare(want, {"bookings": []}, corrected) is None


def test_a_refusal_fails_when_the_call_changed_records_the_reference_leaves_alone():
    before = {"refunds": []}
    assert compare({}, before, {"refunds": []}, expect_nothing=True) is None
    said = compare({}, before, {"refunds": [{"id": 1, "amount": 500}]}, expect_nothing=True)
    assert said is not None and "changes nothing" in said
    assert compare({}, before, {"refunds": [{"id": 1}]}) is None


def test_a_wrong_entity_is_done_wrong_and_a_failed_save_is_not_done():
    want = {"claims": [{"policy": "MM-5590318", "loss_date": "2026-10-11"}]}
    before = {"claims": []}
    wrong = compare(want, before, {"claims": [{"policy": "MM-5590319", "loss_date": "2026-10-11"}]})
    assert wrong == "done wrong: claims.policy expected MM-5590318, got MM-5590319"
    assert compare(want, before, before).startswith("not done:")


def test_long_free_text_is_left_to_the_judge():
    note = "the caller described the hail damage to the bonnet and both side mirrors in detail"
    solution = [{"tool": "file_claim", "arguments": {"policy": "MM-1", "note": note}}]
    expected = expected_changes({"claims": []}, {"claims": [{"policy": "MM-1", "note": note}]}, solution)
    assert expected == {"claims": [{"policy": "MM-1"}]}


def test_changed_rows_ignores_rows_left_as_they_were():
    assert changed_rows(BEFORE, BEFORE) == {}


def test_prove_keeps_what_the_solution_left_and_a_two_argument_check_still_runs(tmp_path):
    from fi.alk.harness.catalogue import Catalogue, SubGoal
    from fi.alk.harness.prove import prove
    from fi.alk.harness.scenario import Scenario, Step
    from fi.alk.harness.world import GeneratedWorld
    from fi.alk.harness.world.snapshot import save

    class W(GeneratedWorld):
        name = "claims"
        tools = [{"name": "file_claim"}]
        handlers = {
            "file_claim": (
                "def handle(args, db):\n"
                "    db.execute('INSERT INTO claims VALUES (?, ?, ?)',"
                " ('gen-' + args['policy'], args['policy'], args['loss_date']))\n"
                "    return {'filed': True}\n"
            )
        }

    world = W(":memory:")
    world.connection.executescript("CREATE TABLE claims(id TEXT, policy TEXT, loss_date TEXT);")
    world.connection.commit()
    save(world, tmp_path, notes="test", sequences=[])
    world.close()

    catalogue = Catalogue(
        sub_goals=[
            SubGoal(
                name="filed",
                what="a claim was filed",
                check=(
                    "def check(world, calls):\n"
                    "    return None if world.state()['claims'] else 'no claim filed'\n"
                ),
            )
        ]
    )
    one = Scenario(
        name="files_a_claim",
        instruction="File it.",
        sub_goals=["filed"],
        solution=[Step(tool="file_claim", arguments={"policy": "MM-7", "loss_date": "2026-10-11"})],
    )
    proof = prove(one, catalogue, tmp_path)
    assert proof.holds is True
    assert proof.expected_changes == {"claims": [{"policy": "MM-7", "loss_date": "2026-10-11"}]}


def test_an_older_scenario_without_expected_changes_compiles_to_none():
    from fi.alk.harness.scenario import Scenario

    assert Scenario(name="old", instruction="x").expected_changes == {}


def test_a_short_value_counts_only_under_its_own_argument_name():
    from fi.alk.harness.reference_state import expected_changes

    before = {"bookings": []}
    after = {"bookings": [{"party_size": 2, "version": 2, "city": "Pune"}]}
    solution = [{"tool": "book", "arguments": {"party_size": 2, "city": "Pune"}}]

    assert expected_changes(before, after, solution) == {"bookings": [{"party_size": 2, "city": "Pune"}]}


class _Call:
    def __init__(self, name, ok=True, refused=False, **arguments):
        self.name, self.ok, self.refused, self.arguments = name, ok, refused, arguments


def test_calls_are_compared_only_on_steps_marked_as_the_outcome():
    from fi.alk.harness.reference_state import compare_calls

    solution = [
        {"tool": "lookup", "arguments": {"phone": "555"}},
        {"tool": "file_claim", "arguments": {"policy": "MM-1", "loss": {"date": "2026-10-11"}}, "outcome": True},
    ]
    right = [_Call("lookup", phone="999"), _Call("file_claim", policy="mm-1", loss={"date": "2026-10-11"})]
    assert compare_calls(solution, right) is None
    assert compare_calls([{**solution[1], "outcome": False}], []) is None


def test_the_last_successful_call_counts_and_a_failed_one_is_not_done():
    from fi.alk.harness.reference_state import compare_calls

    solution = [{"tool": "file_claim", "arguments": {"policy": "MM-1"}, "outcome": True}]
    corrected = [_Call("file_claim", ok=False, policy="MM-1"), _Call("file_claim", policy="MM-2"), _Call("file_claim", policy="MM-1")]
    assert compare_calls(solution, corrected) is None
    assert compare_calls(solution, [_Call("file_claim", ok=False, policy="MM-1")]).startswith("not done:")
    assert compare_calls(solution, [_Call("file_claim", policy="MM-9")]) == "done wrong: file_claim.policy expected MM-1, got MM-9"
    skip = {"file_claim.policy": {"match": "skip"}}
    assert compare_calls(solution, [_Call("file_claim", policy="anything")], rules=skip) is None


def test_a_world_kept_as_a_mapping_or_a_value_is_compared_like_a_table():
    solution = [{"tool": "write_file", "arguments": {"path": "notes/todo.md", "text": "ship it"}}]
    before = {"files": {"README.md": "hi"}, "mode": "draft"}
    after = {"files": {"README.md": "hi", "notes/todo.md": "ship it"}, "mode": "draft"}
    expected = expected_changes(before, after, solution)
    assert expected == {"files": [{"_key": "notes/todo.md", "value": "ship it"}]}
    assert compare(expected, before, after) is None
    wrong = {"files": {"README.md": "hi", "notes/todo.md": "ship it later"}, "mode": "draft"}
    assert compare(expected, before, wrong).startswith("done wrong:")


def test_one_effect_reported_twice_is_one_call_with_its_arguments_merged():
    from fi.alk.harness.reference_state import compare_calls

    solution = [{"tool": "file_claim", "arguments": {"loss_date": "2026-10-13", "confirmed": True}, "outcome": True}]
    full = _Call("file_claim", loss_date="2026-10-13", confirmed=True)
    full.result = {"claim_ref": "CLM-1"}
    partial = _Call("file_claim", confirmed=True)
    partial.result = "{'claim_ref': 'CLM-1'}"
    assert compare_calls(solution, [full, partial]) is None


def test_a_phrase_the_agent_may_word_its_own_way_is_left_to_the_judge():
    from fi.alk.harness.reference_state import compare_calls

    solution = [{"tool": "transfer", "arguments": {"queue": "claims", "reason": "Caller disputes the denied claim"}, "outcome": True}]
    assert compare_calls(solution, [_Call("transfer", queue="claims", reason="denial dispute, wants an adjuster")]) is None
    assert compare_calls(solution, [_Call("transfer", queue="billing", reason="x")]).startswith("done wrong: transfer.queue")


def test_the_same_tool_marked_twice_matches_each_step_to_its_own_call():
    from fi.alk.harness.reference_state import compare_calls

    solution = [
        {"tool": "confirm_address", "arguments": {"role": "pickup", "place": "SFO"}, "outcome": True},
        {"tool": "confirm_address", "arguments": {"role": "dropoff", "place": "OAK"}, "outcome": True},
    ]
    right = [_Call("confirm_address", role="pickup", place="SFO"), _Call("confirm_address", role="dropoff", place="OAK")]
    assert compare_calls(solution, right) is None
    wrong = [_Call("confirm_address", role="pickup", place="SFO"), _Call("confirm_address", role="dropoff", place="SJC")]
    assert compare_calls(solution, wrong) == "done wrong: confirm_address.place expected OAK, got SJC"


def test_only_values_the_scenario_or_world_fixed_are_compared():
    from fi.alk.harness.reference_state import compare_calls, given_text

    given = given_text(
        {"policy_number": "MM-8846099"},
        said="You call about claim CLM-91234 from October 13th and want a human for customer service.",
        world={"queues": [{"name": "general"}, {"name": "general"}], "policies": [{"id": "pol_henrik"}]},
    )
    solution = [{"tool": "transfer", "arguments": {"queue": "customer_service", "claim": "CLM-91234",
                                                   "policy": "pol_henrik", "date": "2026-10-13"}, "outcome": True}]
    right = [_Call("transfer", queue="general", claim="CLM-91234", policy="pol_henrik", date="2026-10-13")]
    assert compare_calls(solution, right, given=given) is None
    wrong = [_Call("transfer", queue="general", claim="CLM-91235", policy="pol_henrik", date="2026-10-14")]
    said = compare_calls(solution, wrong, given=given)
    assert "claim expected CLM-91234" in said and "date expected 2026-10-13" in said and "queue" not in said


def test_one_call_reported_by_a_timed_and_an_untimed_source_is_one_effect():
    from fi.alk.harness.reference_state import compare_calls

    solution = [{"tool": "create_rider", "arguments": {"first_name": "Maya"}, "outcome": True}]
    untimed = _Call("create_rider", phone="+14155550187", first_name="Maya")
    untimed.result, untimed.at = {"rider_id": "r1"}, 0.0
    timed = _Call("create_rider", first_name="Maya")
    timed.result, timed.at = "{'created': True, 'first_name': 'Maya'}", 1792009742.5
    assert compare_calls(solution, [untimed, timed]) is None
    wrong = _Call("create_rider", first_name="Mia")
    wrong.result, wrong.at = {"created": True, "first_name": "Mia"}, 1792009800.0
    assert compare_calls(solution, [wrong]) == "done wrong: create_rider.first_name expected Maya, got Mia"


def test_a_qualified_identifier_is_the_same_identifier():
    from fi.alk.harness.reference_state import compare_calls

    solution = [{"tool": "pay", "arguments": {"payment_method": "pm_eli_amex"}, "outcome": True}]
    assert compare_calls(solution, [_Call("pay", payment_method="saved_card:pm_eli_amex")]) is None
    assert compare_calls(solution, [_Call("pay", payment_method="saved_card:pm_other")]) is not None


def test_a_prohibition_check_is_kept_when_the_reference_grades_whether_anything_was_done(tmp_path):
    from fi.alk.harness.catalogue import Catalogue, SubGoal
    from fi.alk.harness.prove import prove
    from fi.alk.harness.scenario import Scenario, Step
    from fi.alk.harness.world import GeneratedWorld
    from fi.alk.harness.world.snapshot import save

    class W(GeneratedWorld):
        name = "claims"
        tools = [{"name": "file_claim"}]
        handlers = {
            "file_claim": (
                "def handle(args, db):\n"
                "    db.execute('INSERT INTO claims VALUES (?, ?)', ('gen-' + args['policy'], args['policy']))\n"
                "    return {'filed': True}\n"
            )
        }

    world = W(":memory:")
    world.connection.executescript("CREATE TABLE claims(id TEXT, policy TEXT);")
    world.connection.commit()
    save(world, tmp_path, notes="test", sequences=[])
    world.close()

    never_twice = SubGoal(
        name="never_two_claims",
        what="no second claim is filed for one loss",
        check="def check(world, calls):\n    return 'two claims' if len(world.state()['claims']) > 1 else None\n",
    )
    catalogue = Catalogue(sub_goals=[never_twice])
    step = Step(tool="file_claim", arguments={"policy": "MM-7"})
    with_reference = prove(Scenario(name="a", instruction="File it.", sub_goals=["never_two_claims"], solution=[step]), catalogue, tmp_path)
    assert with_reference.expected_changes and with_reference.reference_grades and with_reference.holds
