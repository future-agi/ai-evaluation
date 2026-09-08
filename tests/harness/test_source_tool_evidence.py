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
