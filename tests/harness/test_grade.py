"""What a result reports when the catalogue cannot settle what the scenario named."""

from __future__ import annotations

from fi.alk.harness.catalogue import Catalogue, SubGoal
from fi.alk.harness.run.grade import Result, ungraded_sub_goals
from fi.alk.harness.scenario import Scenario


def _scenario() -> Scenario:
    return Scenario(
        name="refund_after_shipping",
        instruction="Get the fee taken off.",
        sub_goals=["fee_removed", "order_checked"],
    )


def test_a_result_that_checked_nothing_has_not_passed():
    assert Result(scenario="refund_after_shipping").passed is False


def test_a_scenario_still_passes_on_its_checkpoints():
    from fi.alk.harness.run.grade import Checkpoint

    result = Result(
        scenario="refund_after_shipping",
        checkpoints=[Checkpoint(name="fee_removed", kind="code", passed=True)],
    )
    assert result.passed is True


def test_a_sub_goal_missing_from_the_catalogue_is_reported():
    catalogue = Catalogue(
        sub_goals=[SubGoal(name="fee_removed", what="the fee is gone", check="def check(world, calls):\n    return None")]
    )
    assert ungraded_sub_goals(_scenario(), catalogue) == ["order_checked"]


def test_nothing_is_reported_when_the_catalogue_holds_them_all():
    catalogue = Catalogue(
        sub_goals=[
            SubGoal(name="fee_removed", what="the fee is gone", check="def check(world, calls):\n    return None"),
            SubGoal(name="order_checked", what="the order was read", judged="did it read the order"),
        ]
    )
    assert ungraded_sub_goals(_scenario(), catalogue) == []


def test_a_setup_that_only_adjusts_borrowed_records_is_refused():
    """Measured on real suites: refuses 4 of 50 in the reference suite, 30 of 86 in the thin one."""
    from fi.alk.harness.scenario import self_sufficiency_problems

    borrowed = Scenario(
        name="expired_card_on_file",
        instruction="Get the booking paid for.",
        sub_goals=["fee_removed"],
        setup_code=(
            "def setup(world):\n"
            "    world.change('payment_methods', 'pm_existing', {'last4': '6172'}, by='id')\n"
        ),
    )
    assert self_sufficiency_problems(borrowed)

    owns_it = Scenario(
        name="expired_card_on_file",
        instruction="Get the booking paid for.",
        sub_goals=["fee_removed"],
        setup_code=(
            "def setup(world):\n"
            "    world.put('payment_methods', {'id': 'pm_own', 'last4': '6172'})\n"
            "    world.change('payment_methods', 'pm_own', {'expired': 1}, by='id')\n"
        ),
    )
    assert self_sufficiency_problems(owns_it) == []

    # The documented no-seam case: nothing to change, so nothing is claimed.
    assert (
        self_sufficiency_problems(
            Scenario(name="reads_only", instruction="Ask for the status.", sub_goals=["x"])
        )
        == []
    )


def test_a_writer_that_dies_after_proving_still_leaves_its_work(tmp_path):
    """A delegated writer cannot write folders, so the journal is the only record it leaves."""
    from fi.alk.harness.scenario_tools import journal_scenario, journalled

    one = Scenario(
        name="cancel_after_fee_quoted",
        instruction="Get the ride cancelled.",
        sub_goals=["ride_cancelled"],
    )
    journal_scenario(one, tmp_path)
    # A retried slice journals the same name again, and the later line wins rather than duplicating.
    journal_scenario(one.model_copy(update={"instruction": "Cancel it, and ask what it costs."}), tmp_path)
    back = journalled(tmp_path)
    assert [x.name for x in back] == ["cancel_after_fee_quoted"]
    assert back[0].instruction.startswith("Cancel it")
    assert journalled(tmp_path / "nothing-here") == []


def test_a_declared_check_that_never_reached_the_folder_is_not_read_as_judged(tmp_path):
    """Absence of a check file means judged, which is wrong when the catalogue settles it in code."""
    import json

    import pytest

    from fi.alk.harness.scenario_source import ScenarioDocumentInvalid, load_scenarios

    bundle = tmp_path
    (bundle / "sub_goals.json").write_text(
        json.dumps(
            {
                "sub_goals": [
                    {"name": "fee_removed", "what": "the fee is gone",
                     "check": "def check(world, calls):\n    return None\n"},
                    {"name": "explained_kindly", "what": "tone", "judged": "read the transcript"},
                ]
            }
        )
    )
    folder = bundle / "scenarios" / "refund_after_shipping"
    folder.mkdir(parents=True)
    (folder / "scenario.json").write_text(
        json.dumps({"scenario_key": "refund", "scenario_id": "",
                    "sub_goals": ["fee_removed", "explained_kindly"],
                    "name": "refund_after_shipping", "instruction": "Get the fee taken off."})
    )
    (folder / "setup.py").write_text("def setup(world):\n    return None\n")
    (folder / "ready.py").write_text("def ready(world):\n    return None\n")

    with pytest.raises(ScenarioDocumentInvalid) as refused:
        load_scenarios(bundle)
    assert "fee_removed" in str(refused.value)

    # With the check materialised, the judged one beside it is still judged.
    checks = folder / "checks"
    checks.mkdir()
    (checks / "fee_removed.py").write_text("def check(world, calls):\n    return None\n")
    loaded = load_scenarios(bundle)
    assert [one.scenario_key for one in loaded] == ["refund"]
    assert loaded[0].presented["situation"] == "Get the fee taken off."


def test_a_slice_writer_stops_at_the_size_it_was_given(tmp_path):
    """Its turn budget is far larger than its slice, and left alone it keeps writing."""
    import asyncio

    from fi.alk.harness.contract import AgentContract
    from fi.alk.harness.scenario_tools import scenario_tools

    contract = AgentContract(agent="cart", real_use_cases=["add an item"])
    server, kept = scenario_tools(contract, tmp_path, tmp_path, wanted=1, can_save=False)
    submit = next(spec for spec in server.tools if spec.name == "submit_scenario")
    kept.append(Scenario(name="already_here", instruction="one", sub_goals=["x"]))

    said = asyncio.run(submit.handler({"name": "a_second_one", "instruction": "two"}))
    assert said.get("is_error")
    assert "This slice is complete" in said["content"][0]["text"]

    # Replacing one of its own is still allowed, which is how a refused scenario gets fixed. It gets
    # past the cap and into validation, which here has no world to validate against.
    import pytest

    with pytest.raises(FileNotFoundError):
        asyncio.run(submit.handler({"name": "already_here", "instruction": "one, fixed"}))


def test_a_second_fan_out_pass_only_writes_what_is_missing(tmp_path, monkeypatch):
    """Called again with the original number, it wrote a second full suite: 377 against a target of 200."""
    import asyncio

    from fi.alk.harness import scenario_tools as st
    from fi.alk.harness.contract import AgentContract

    contract = AgentContract(agent="cart", real_use_cases=["add an item", "remove an item"])
    # Above FEWEST_WORTH_DELEGATING, or the fan-out tool is not published at all.
    server, _kept = st.scenario_tools(contract, tmp_path, tmp_path, wanted=20)
    suite = next(spec for spec in server.tools if spec.name == "generate_suite")

    asked_for: list[int] = []

    async def _fake_parallel(contract, *, out, wanted, use_cases, slices, at_once):
        asked_for.append(wanted)
        return []

    monkeypatch.setattr("fi.alk.harness.scenarios.write_in_parallel", _fake_parallel)
    # Nothing on disk yet: the full ask goes through.
    asyncio.run(suite.handler({"count": 20}))
    assert asked_for == [20]

    # Fourteen already written, and the model asks for twenty again: six are outstanding.
    monkeypatch.setattr(
        st, "load_scenarios", lambda _d: [Scenario(name=f"s{i}", instruction="i", sub_goals=["x"]) for i in range(14)]
    )
    monkeypatch.setattr(st, "journalled", lambda _d: [])
    asyncio.run(suite.handler({"count": 20}))
    assert asked_for == [20, 6]

    # Once the target is met, it refuses to write more rather than starting another suite. Counted
    # from the folders: a journal that outlives its folders means a retried attempt, where refusing to
    # write is how a run saves 14 of 200 and fails.
    monkeypatch.setattr(
        st, "load_scenarios", lambda _d: [Scenario(name=f"s{i}", instruction="i", sub_goals=["x"]) for i in range(20)]
    )
    said = asyncio.run(suite.handler({"count": 20}))
    assert "already holds the 20" in said["content"][0]["text"]
    assert asked_for == [20, 6]


def test_a_placeholder_code_is_refused_however_it_is_arranged():
    """The hand-kept list caught 111111 and let 000111 through, which reached a 200-scenario suite twice."""
    from fi.alk.harness.scenario import _predictable

    for placeholder in ("000111", "111111", "123456", "987654", "010101", "447744"):
        assert _predictable(placeholder), placeholder
    # Real codes from suites on disk stay allowed, which is what stops this refusing everything.
    for real in ("004928", "592804", "731905", "638204", "112233"):
        assert not _predictable(real), real


def test_the_whole_suite_is_registered_and_only_a_sample_is_run():
    """Sampling before pre-allocation had the platform refuse: expected exactly 30 personas, got 5."""
    import inspect

    from fi.alk.harness.scenario_source import BundleScenarioSource, sampled_for_calling

    source = inspect.getsource(BundleScenarioSource.build)
    registered = source.index("register_with_platform")
    sampled = source.index("sampled_for_calling")
    assert registered < sampled, "the suite must be registered before the sample is taken"

    # And every scenario is called: registering the suite and calling it are the same set.
    assert len(sampled_for_calling(list(range(30)))) == 30


def test_an_empty_save_does_not_take_the_suite_with_it(tmp_path):
    """Dropping a scenario is expressed by saving without it, so an empty save deletes everything."""
    from fi.alk.harness.catalogue import Catalogue
    from fi.alk.harness.scenario_tools import load_scenarios, write_scenarios

    one = Scenario(name="keeps_its_place", instruction="Get it done.", sub_goals=["x"])
    write_scenarios([one], tmp_path, Catalogue())
    assert [x.name for x in load_scenarios(tmp_path)] == ["keeps_its_place"]

    write_scenarios([], tmp_path, Catalogue())
    assert [x.name for x in load_scenarios(tmp_path)] == ["keeps_its_place"]


def test_every_scenario_a_job_asked_for_is_called():
    """The five-call cap was a testing measure for one person's provider credits. It is gone, not
    defaulted off: a setting that would under-deliver a paid run is not worth having."""
    from fi.alk.harness import scenario_source

    assert len(scenario_source.sampled_for_calling(list(range(200)))) == 200
    assert scenario_source.sampled_for_calling([1, 2, 3]) == [1, 2, 3]
    assert not hasattr(scenario_source, "CALLS_AT_MOST")


def test_a_proof_says_which_steps_it_could_not_run():
    """The condition was detected and logged into a void: a scenario whose whole solution was
    recorded without executing saved as "all three gates pass"."""
    from fi.alk.harness.prove import Proof

    proof = Proof(ready=True, solvable=True, vacuous=False)
    assert proof.assumed == []
    proof.assumed = ["book_ride", "verify_otp"]
    # Still holds: on a lane with no endpoints a hard gate would refuse every scenario. What changes
    # is that the caller is told, rather than the fact disappearing into a log line.
    assert proof.holds is True
    assert proof.assumed == ["book_ride", "verify_otp"]


def test_a_world_with_state_still_demands_a_check_in_code(tmp_path):
    """The judged-only path is for a target we cannot see into, not a way around writing a check."""
    from fi.alk.harness.catalogue import Catalogue, SubGoal
    from fi.alk.harness.prove import prove

    root, _contract, catalogue = None, None, None
    from fi.alk.harness.world import GeneratedWorld
    from fi.alk.harness.world.snapshot import save

    class W(GeneratedWorld):
        name = "shop"
        tools = [{"name": "add"}]
        handlers = {"add": "def handle(args, db):\n    return {'ok': 1}\n"}

    world = W(":memory:")
    world.connection.executescript("CREATE TABLE cart(item_id TEXT);")
    world.connection.execute("INSERT INTO cart VALUES ('widget')")
    world.connection.commit()
    save(world, tmp_path, notes="test", sequences=[])
    world.close()

    judged_only = Catalogue(sub_goals=[SubGoal(name="polite", what="tone", judged="read it")])
    one = Scenario(name="asks_politely", instruction="Ask for it.", sub_goals=["polite"])
    proof = prove(one, judged_only, tmp_path)
    assert proof.holds is False
    assert "settle what happened by reading it" in proof.broken[0]
    assert proof.judged_only is False
