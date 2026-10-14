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


def test_times_compare_within_the_tolerance():
    expected = {"bookings": [{"pickup_at": "2026-10-15T16:00:00"}]}
    before = {"bookings": []}
    assert compare(expected, before, {"bookings": [{"pickup_at": "2026-10-15T17:30:00"}]}) is None
    assert compare(expected, before, {"bookings": [{"pickup_at": "2026-10-16T16:00:00"}]}) is not None


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
