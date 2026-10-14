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
