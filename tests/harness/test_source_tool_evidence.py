import json
import asyncio

import pytest

from fi.alk.harness.contract import AgentContract, validate_contract
from fi.alk.harness.tools import accept_contract


def payload():
    return {
        "agent": "starter",
        "real_use_cases": ["Answer conversational questions"],
        "tools": [{"name": "lookup_weather", "args": ["location"]}],
        "tool_entrypoints": [
            {
                "tool": "lookup_weather",
                "mode": "construct",
                "module": "agent",
                "callable": "Assistant.lookup_weather",
            }
        ],
    }


@pytest.mark.parametrize("tools", [[], [{"name": "get_status", "args": []}]])
def test_tool_cardinality_is_not_a_validation_error(tools):
    assert (
        validate_contract(AgentContract.model_validate(payload() | {"tools": tools}))
        == []
    )


def test_commented_example_is_returned_to_author_for_correction(tmp_path):
    source = tmp_path / "source"
    (source / "src").mkdir(parents=True)
    (source / "src/agent.py").write_text(
        "class Assistant:\n    def __init__(self): pass\n"
        "    # @function_tool\n    # async def lookup_weather(self, location): pass\n"
    )
    out = tmp_path / "out"
    result = accept_contract(payload(), out, source_root=source)
    assert result["is_error"]
    assert "commented-only-entrypoint" in result["content"][0]["text"]
    assert not (out / "contract.json").exists()
    corrected = payload() | {"tools": [], "tool_entrypoints": []}
    assert not accept_contract(corrected, out, source_root=source).get("is_error")
    assert json.loads((out / "contract.json").read_text())["tools"] == []


@pytest.mark.parametrize(
    "binding",
    [
        "class Assistant:\n    @function_tool\n    async def lookup_weather(self, location): pass\n",
        "from real_tools import lookup_weather\n",
        "lookup_weather = make_tool()\n",
    ],
)
def test_real_or_dynamic_bindings_are_not_rejected(tmp_path, binding):
    # Never execute agent imports to inspect them.
    (tmp_path / "agent.py").write_text(
        "raise RuntimeError('must not import')\n"
        + binding
        + "# def lookup_weather(location): example\n"
    )
    assert not accept_contract(payload(), tmp_path / "out", source_root=tmp_path).get(
        "is_error"
    )


def test_external_modules_and_service_tools_are_not_static_python_claims(tmp_path):
    assert not accept_contract(payload(), tmp_path / "out", source_root=tmp_path).get(
        "is_error"
    )


def test_tool_free_world_saves_without_fabricated_state_checks(tmp_path):
    from fi.alk.harness.catalogue import Catalogue, SubGoal, save_catalogue
    from fi.alk.harness.simulator import save_simulator_prompt
    from fi.alk.harness.world.tools import world_tools

    contract = AgentContract(
        agent="starter",
        tools=[],
        real_use_cases=["Chat"],
        runtime_dependencies=[
            {
                "name": "livekit_cloud",
                "kind": "transport",
                "engine": "livekit",
                "reached": {"dsn_env": "LIVEKIT_URL"},
            }
        ],
    )
    save_catalogue(
        Catalogue(
            sub_goals=[
                SubGoal(
                    name="helpful_answer",
                    judged="Responds accurately and directly to the user's question",
                )
            ]
        ),
        tmp_path,
    )
    server, world = world_tools(contract, tmp_path)
    tools = {one.name: one.handler for one in server.tools}
    assert (
        "not applicable" in asyncio.run(tools["check_world"]({}))["content"][0]["text"]
    )
    check = asyncio.run(
        tools["add_world_check"]({"name": "empty", "check": "assert True"})
    )
    assert "not applicable" in check["content"][0]["text"]
    save = next(one.handler for one in server.tools if one.name == "save_world")
    assert asyncio.run(save({}))["is_error"]
    save_simulator_prompt(
        "You are a caller with persona {{ persona }}. Follow {{ instruction }} and ask natural "
        "follow-up questions. Do not act as the assistant or invent its responses.",
        tmp_path,
    )
    result = asyncio.run(save({}))
    assert not result.get("is_error"), result
    assert "not applicable" in result["content"][0]["text"]
    assert world.state() == {}


def test_tool_free_world_with_baseline_data_does_not_require_tool_sequence(tmp_path):
    """Provider chat agents may have seed variables but no callable tools.

    Requiring a sequence in that shape is impossible to satisfy: empty sequences are invalid and
    every named call would invent a tool the target does not expose.
    """
    from fi.alk.harness.catalogue import Catalogue, SubGoal, save_catalogue
    from fi.alk.harness.simulator import save_simulator_prompt
    from fi.alk.harness.world.tools import world_tools

    contract = AgentContract(
        agent="provider_chat",
        modality="chat",
        conversational=True,
        tools=[],
        real_use_cases=["Discuss an account using provider-supplied context"],
        base_environment={"dynamic_variables": {"customer_name": "Customer 4821"}},
    )
    save_catalogue(
        Catalogue(
            sub_goals=[
                SubGoal(
                    name="uses_customer_context",
                    what="Uses the supplied customer context",
                    check=(
                        "def check(world, calls):\n"
                        "    rows = world.get('dynamic_variables', [])\n"
                        "    return None if rows and rows[0].get('customer_name') else "
                        "'customer context missing'\n"
                    ),
                )
            ]
        ),
        tmp_path,
    )
    save_simulator_prompt(
        "You are a caller with persona {{ persona }}. Follow {{ instruction }} and respond "
        "naturally without inventing the assistant's messages.",
        tmp_path,
    )
    server, _world = world_tools(contract, tmp_path)
    tools = {one.name: one.handler for one in server.tools}
    assert not asyncio.run(
        tools["create_schema"](
            {"sql": "CREATE TABLE dynamic_variables (customer_name TEXT NOT NULL);"}
        )
    ).get("is_error")
    assert not asyncio.run(
        tools["seed"](
            {
                "table": "dynamic_variables",
                "rows": [{"customer_name": "Customer 4821"}],
            }
        )
    ).get("is_error")
    assert not asyncio.run(
        tools["add_world_check"](
            {
                "name": "customer_context_exists",
                "code": (
                    "def check(world):\n"
                    "    rows = world.state().get('dynamic_variables', [])\n"
                    "    return None if rows and rows[0].get('customer_name') else "
                    "'customer context missing'\n"
                ),
            }
        )
    ).get("is_error")

    result = asyncio.run(tools["save_world"]({}))
    assert not result.get("is_error"), result
    assert "0 tools" in result["content"][0]["text"]


def test_generated_world_mutations_do_not_leak_between_agents():
    from fi.alk.harness.world.runtime import GeneratedWorld

    first = GeneratedWorld(":memory:")
    first.tools.append({"name": "lookup_account"})
    first.handlers["lookup_account"] = "def handle(args, db): return args"
    first.state_object = {"accounts": ["one"]}

    second = GeneratedWorld(":memory:")
    assert second.tools == []
    assert second.handlers == {}
    assert second.state_object is None


def test_connection_dependency_is_corrected_before_world_authoring(tmp_path):
    dependency = {
        "name": "livekit_cloud",
        "kind": "service",
        "engine": "livekit",
        "reached": {"dsn_env": "LIVEKIT_URL"},
    }
    body = payload() | {
        "tools": [],
        "tool_entrypoints": [],
        "dependencies": [dependency],
    }
    result = accept_contract(body, tmp_path)
    assert result["is_error"]
    assert "runtime-connection-in-world" in result["content"][0]["text"]
    body.update(
        dependencies=[], runtime_dependencies=[dependency | {"kind": "transport"}]
    )
    assert not accept_contract(body, tmp_path).get("is_error")
    saved = AgentContract.model_validate_json((tmp_path / "contract.json").read_text())
    assert "RUNTIME CONNECTIONS" in saved.brief()
    assert "LIVEKIT_URL" in saved.brief()


def test_business_database_cannot_be_hidden_in_runtime_connections():
    contract = AgentContract(
        agent="x",
        real_use_cases=["Chat"],
        runtime_dependencies=[
            {"name": "orders", "kind": "datastore", "reached": {"database": "orders"}}
        ],
    )
    assert any("business-data-in-runtime" in p for p in validate_contract(contract))


def _goal(name, *, check="", judged="", what="a thing"):
    from fi.alk.harness.catalogue import SubGoal

    return SubGoal(name=name, what=what, check=check, judged=judged)


def test_a_check_that_only_matches_call_names_is_refused():
    """The exact sub-goal a live run reported: 'transfer_to_licensed_agent was not called
    successfully'. It passes an agent that called the right tool with the wrong arguments."""
    from fi.alk.harness.catalogue import validate_sub_goal

    problems = validate_sub_goal(
        _goal(
            "transfers_to_human_agent",
            check=(
                "def check(world, calls):\n"
                '    if not any(c.name == "transfer_to_licensed_agent" and c.ok for c in calls):\n'
                '        return "transfer_to_licensed_agent was not called successfully"\n'
                "    return None\n"
            ),
        )
    )

    assert problems and "only asks whether a tool was called" in problems[0]


def test_a_check_that_asserts_arguments_is_accepted():
    """Reading what the agent passed is the point: an agent that misheard a name calls exactly the
    tool it should have."""
    from fi.alk.harness.catalogue import validate_sub_goal

    assert (
        validate_sub_goal(
            _goal(
                "intake_recorded_for_the_right_person",
                check=(
                    "def check(world, calls):\n"
                    '    done = [c for c in calls if c.name == "record_intake" and c.ok]\n'
                    '    if not done:\n        return "no intake recorded"\n'
                    '    if done[0].arguments.get("name") != "Corwin":\n'
                    "        return f\"recorded {done[0].arguments.get('name')!r}\"\n"
                    "    return None\n"
                ),
            )
        )
        == []
    )


def test_a_check_that_reads_world_state_is_accepted():
    from fi.alk.harness.catalogue import validate_sub_goal

    assert (
        validate_sub_goal(
            _goal(
                "one_lead_written",
                check=(
                    "def check(world, calls):\n"
                    '    rows = world.state()["leads"]\n'
                    '    return None if len(rows) == 1 else f"{len(rows)} rows"\n'
                ),
            )
        )
        == []
    )


def test_mentioning_the_world_in_a_comment_does_not_satisfy_the_gate():
    """The previous gate was a substring test, so a comment naming the world passed it."""
    from fi.alk.harness.catalogue import validate_sub_goal

    problems = validate_sub_goal(
        _goal(
            "looks_right",
            check=(
                "def check(world, calls):\n"
                "    # world is not actually read here\n"
                '    return None if any(c.name == "x" for c in calls) else "no x"\n'
            ),
        )
    )

    assert problems and "only asks whether a tool was called" in problems[0]


def test_a_judged_sub_goal_must_say_why_it_is_judged():
    from fi.alk.harness.catalogue import validate_sub_goal

    thin = validate_sub_goal(_goal("polite", judged="was it polite"))
    assert thin and "does not say what a model has to decide" in thin[0]

    assert (
        validate_sub_goal(
            _goal(
                "refusal_explained",
                judged=(
                    "Whether the agent explained why it refused, which the world records nothing "
                    "about because a refusal leaves no row behind"
                ),
            )
        )
        == []
    )


def test_a_catalogue_that_is_mostly_judged_is_refused():
    """A judge is the fallback, not the method. One run reported six judged sub-goals."""
    from fi.alk.harness.catalogue import catalogue_problems

    judged = _goal(
        "explained",
        judged="Whether the refusal was explained, which nothing in the world records at all",
    )
    coded = _goal(
        "row_written",
        check='def check(world, calls):\n    return None if world.state() else "empty"\n',
    )

    problems = catalogue_problems([judged, judged, coded])
    assert problems and "judged rather than settled by code" in problems[0]
    assert catalogue_problems([judged, coded, coded]) == []


def test_a_target_we_cannot_see_into_may_be_judged_throughout():
    """A conversational agent with no executable tools and no state leaves nothing behind for a
    check to read, so judging is the only thing available and is correct rather than lazy."""
    from fi.alk.harness.catalogue import catalogue_problems

    judged = _goal(
        "answers_accurately",
        judged="Whether the answer was accurate, which no world state records because this agent has none",
    )

    assert catalogue_problems([judged, judged], world_is_observable=False) == []
    assert catalogue_problems([judged, judged], world_is_observable=True)


def test_a_coded_sub_goal_also_carries_its_description():
    """A check says nothing when it holds, so `what` is the only thing a reader has to tell a real
    pass from one nobody wrote a check for. Measured on run 42875830: every passing sub-goal read
    "Held. The check found nothing wrong." because `what` was restored for judged ones only."""
    from dataclasses import replace

    from fi.alk.harness.scenario_source import _CompiledSubGoal, _with_claims

    coded = _CompiledSubGoal(name="schedules_callback", judged="", check=lambda w, c: None)
    judged = _CompiledSubGoal(name="protocol", judged="x", check=lambda w, c: None)

    class Scenario:
        sub_goals = (coded, judged)

    import types

    scenario = types.SimpleNamespace(sub_goals=(coded, judged))
    claims = {
        "schedules_callback": {"what": "a callback was written for the time agreed", "judged": ""},
        "protocol": {"what": "the agent stayed on protocol", "judged": "whether it stayed civil"},
    }

    # _with_claims uses dataclasses.replace on the scenario, so give it a real dataclass field set.
    from fi.alk.harness.scenario_source import _CompiledScenario

    real = _CompiledScenario(
        scenario_key="k",
        scenario_id="",
        sub_goals=(coded, judged),
        requires_tool_evidence=False,
        setup=lambda w: None,
        ready=lambda w: None,
    )
    out = _with_claims(real, claims)

    by_name = {g.name: g for g in out.sub_goals}
    assert by_name["schedules_callback"].what == "a callback was written for the time agreed"
    assert by_name["schedules_callback"].judged == "", "a coded sub-goal must not become judged"
    assert by_name["protocol"].what == "the agent stayed on protocol"
    assert by_name["protocol"].judged == "whether it stayed civil"


def test_a_truthiness_check_is_told_to_compare_against_an_expected_value():
    """Measured on the catalogue a live run authored: five of six coded checks read the arguments
    and tested only that they were present. An agent that mishears a detail and acts confidently
    on the wrong one passes every one of those."""
    from fi.alk.harness.catalogue import compares_to_a_value, weak_check_advisory

    truthiness = _goal(
        "transfers_to_human_agent",
        check=(
            "def check(world, calls):\n"
            '    xfers = [c for c in calls if c.name == "transfer_to_licensed_agent" and c.ok]\n'
            '    if not xfers:\n        return "not called"\n'
            '    reason = xfers[0].arguments.get("reason")\n'
            "    if not reason or not isinstance(reason, str) or not reason.strip():\n"
            '        return "no reason"\n'
            "    return None\n"
        ),
    )
    compares = _goal(
        "intake_recorded_for_the_right_person",
        check=(
            "def check(world, calls):\n"
            '    done = [c for c in calls if c.name == "record_intake" and c.ok]\n'
            '    if done[0].arguments.get("name") != "Corwin":\n        return "wrong name"\n'
            "    return None\n"
        ),
    )

    assert not compares_to_a_value(truthiness.check)
    assert compares_to_a_value(compares.check)
    assert "only tests that they are present" in weak_check_advisory(truthiness)
    assert weak_check_advisory(compares) == ""
    # Advisory, never a refusal: it must not block a catalogue from being accepted.
    from fi.alk.harness.catalogue import validate_sub_goal

    assert validate_sub_goal(truthiness) == []


def test_the_writer_and_the_reader_agree_on_where_the_catalogue_lives(tmp_path):
    """A round trip through the real writer. Run 7b62c314 reported every passing sub-goal as
    "Held. The check found nothing wrong." even though its catalogue carried a description for all
    eight, which is what happens when the reader cannot find sub_goals.json beside the scenarios.
    This pins the two ends together so that gap fails here instead of two systems away."""
    from fi.alk.harness.catalogue import Catalogue, SubGoal, save_catalogue
    from fi.alk.harness.folder import write_folder
    from fi.alk.harness.scenario import Scenario
    from fi.alk.harness.scenario_source import load_scenarios

    coded = SubGoal(
        name="schedules_callback",
        what="a callback was written for the time the caller agreed",
        check=(
            "def check(world, calls):\n"
            '    done = [x for x in calls if x.name == "schedule" and x.ok]\n'
            '    if not done:\n        return "not scheduled"\n'
            '    if done[0].arguments.get("when") != "10:00":\n        return "wrong time"\n'
            "    return None\n"
        ),
    )
    judged = SubGoal(
        name="stayed_within_licence",
        what="no premium figure was given",
        judged=(
            "Whether a premium was implied, which no world row records because speech leaves none"
        ),
    )
    catalogue = Catalogue(sub_goals=[coded, judged])
    scenario = Scenario(
        name="callback", sub_goals=["schedules_callback", "stayed_within_licence"]
    )

    write_folder(scenario, catalogue, tmp_path)
    save_catalogue(catalogue, tmp_path)

    # The catalogue is a sibling of scenarios/, which is the layout the reader has to expect.
    assert (tmp_path / "sub_goals.json").is_file()
    assert (tmp_path / "scenarios" / "callback" / "scenario.json").is_file()

    by_name = {g.name: g for g in load_scenarios(tmp_path)[0].sub_goals}
    assert by_name["schedules_callback"].what == (
        "a callback was written for the time the caller agreed"
    )
    assert by_name["schedules_callback"].judged == ""
    assert by_name["stayed_within_licence"].what == "no premium figure was given"
    assert by_name["stayed_within_licence"].judged.startswith("Whether a premium")


def test_the_bundle_carries_the_catalogue_the_scenarios_reference(tmp_path):
    """The gap that made run 7b62c314 report every passing sub-goal as "the check found nothing
    wrong". bundle_author_v2 copied authoring/scenarios into the bundle and nothing copied
    sub_goals.json beside it, so the reader found the scenarios and not the catalogue they name."""
    from fi.alk.harness.bundle_author_v2 import _copy_scenarios, _copy_sub_goal_catalogue
    from fi.alk.harness.catalogue import CATALOGUE, Catalogue, SubGoal, save_catalogue
    from fi.alk.harness.folder import write_folder
    from fi.alk.harness.scenario import Scenario
    from fi.alk.harness.scenario_source import load_scenarios

    authoring = tmp_path / "authoring"
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    coded = SubGoal(
        name="schedules_callback",
        what="a callback was written for the time the caller agreed",
        check=(
            "def check(world, calls):\n"
            '    done = [x for x in calls if x.name == "schedule" and x.ok]\n'
            '    if not done:\n        return "not scheduled"\n'
            '    if done[0].arguments.get("when") != "10:00":\n        return "wrong time"\n'
            "    return None\n"
        ),
    )
    catalogue = Catalogue(sub_goals=[coded])
    write_folder(Scenario(name="callback", sub_goals=["schedules_callback"]), catalogue, authoring)
    save_catalogue(catalogue, authoring)

    _copy_scenarios(authoring, bundle, count=1)
    adopted = _copy_sub_goal_catalogue(authoring, bundle)

    assert adopted == [CATALOGUE], "the catalogue has to be declared as adopted"
    assert (bundle / CATALOGUE).is_file(), "the bundle must carry the catalogue"

    # And the reader, given only the bundle, gets the description back.
    goal = load_scenarios(bundle)[0].sub_goals[0]
    assert goal.what == "a callback was written for the time the caller agreed"


def test_a_bundle_with_no_catalogue_is_a_warning_not_a_failure(tmp_path):
    """A bundle whose scenarios are all judged has nothing to lose, and failing the run here would
    be worse than the degraded reporting it replaces."""
    from fi.alk.harness.bundle_author_v2 import _copy_sub_goal_catalogue

    authoring = tmp_path / "authoring"
    authoring.mkdir()
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    assert _copy_sub_goal_catalogue(authoring, bundle) == []


def test_the_truthiness_refusal_is_off_by_default():
    """Off because the retry loop is unproven inside a real authoring session: the model writes a
    comparison check in 3 of 3 trials when asked directly, and ignored the equivalent written
    guidance in situ (1 of 6 checks became 1 of 7). Turning it on refuses most of a catalogue."""
    from fi.alk.harness import catalogue as module

    assert module.REFUSE_TRUTHINESS_CHECKS is False

    truthiness = _goal(
        "transfers_to_human_agent",
        check=(
            "def check(world, calls):\n"
            '    xfers = [c for c in calls if c.name == "transfer_to_licensed_agent" and c.ok]\n'
            '    if not xfers:\n        return "not called"\n'
            '    reason = xfers[0].arguments.get("reason")\n'
            "    if not reason or not isinstance(reason, str):\n"
            '        return "no reason"\n'
            "    return None\n"
        ),
    )

    assert module.validate_sub_goal(truthiness) == [], "accepted while the switch is off"
    assert "only tests that they are present" in module.weak_check_advisory(truthiness)


def test_turning_the_refusal_on_rejects_a_truthiness_check(monkeypatch):
    """One line to enable, and this pins what happens when it is: the check is refused with the
    sentence that says what to do instead, and a check comparing a value is still accepted."""
    from fi.alk.harness import catalogue as module

    monkeypatch.setattr(module, "REFUSE_TRUTHINESS_CHECKS", True)

    truthiness = _goal(
        "transfers_to_human_agent",
        check=(
            "def check(world, calls):\n"
            '    xfers = [c for c in calls if c.name == "transfer_to_licensed_agent" and c.ok]\n'
            '    reason = xfers[0].arguments.get("reason")\n'
            "    if not reason:\n        return \"no reason\"\n"
            "    return None\n"
        ),
    )
    compares = _goal(
        "intake_recorded_for_the_right_person",
        check=(
            "def check(world, calls):\n"
            '    done = [c for c in calls if c.name == "record_intake" and c.ok]\n'
            '    if done[0].arguments.get("name") != "Corwin":\n        return "wrong name"\n'
            "    return None\n"
        ),
    )

    problems = module.validate_sub_goal(truthiness)
    assert problems and "only tests that they are present" in problems[0]
    assert "Compare the value against what this scenario expected" in problems[0]
    assert module.validate_sub_goal(compares) == [], "a real comparison must still pass"
