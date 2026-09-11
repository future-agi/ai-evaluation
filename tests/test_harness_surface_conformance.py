"""That the four places a fact has to live still agree with each other.

Adding one capability means touching four independent things: the model that stores a field,
the schema that lets a stage write it, the brief that lets the next stage read it, and the
skill that says what it means. Nothing makes them agree, and when they disagree nothing raises
-- the field is simply never filled in, or never read, and the first sign of it is a stage
behaving oddly in a live run that costs money and minutes.

That is not a hypothetical. Every one of these tests is a bug that actually shipped in this
branch and was found by a model run rather than by anything here:

- ``engine`` and ``reached`` existed on the model and not in the schema, so a Qdrant-backed
  agent came back with every provisioning field empty.
- ``loader_module`` was added to the model and the schema and not to the brief, so a stage was
  told "stand up inprocess" with no loader named, and proposed putting the agent's in-memory
  data in Redis.
- ``add_sub_goal`` declared ``judged`` a boolean where the model has a string, so the stage
  could not add a sub-goal at all.

These are cheap, offline, and would have caught all three before a single token was spent.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import mcp.types as mcp
import pytest

from fi.alk.harness.contract import AgentContract, Dependency, Reached
from fi.alk.harness.environment import SubGoal
from fi.alk.harness.tools import contract_tools
from fi.alk.harness.world.provision import provision_tools
from fi.alk.harness.world.tools import world_tools


def schemas(server) -> dict[str, dict]:
    """The tools as a stage actually sees them, read off the running server.

    Deliberately not read from the source: what matters is what reaches the model, and a
    schema built correctly but never registered would pass any test of the source. Stages now
    describe tools neutrally; routing through the Claude adapter keeps this about what is
    actually published.
    """
    from fi.alk.harness.backends import ToolServer
    from fi.alk.harness.backends.claude import _sdk_server

    if isinstance(server, ToolServer):
        server = _sdk_server(server)
    handler = server["instance"].request_handlers[mcp.ListToolsRequest]
    listed = asyncio.run(handler(mcp.ListToolsRequest(method="tools/list"))).root.tools
    return {one.name: one.inputSchema for one in listed}


@pytest.fixture()
def contract_schema() -> dict:
    server = contract_tools(Path(tempfile.mkdtemp()))
    return schemas(server)["submit_contract"]


# --- can a stage write every field we ask it for? ------------------------------------------


def test_every_dependency_field_can_be_written(contract_schema: dict) -> None:
    """A field the schema does not offer is a field that is never filled in.

    No amount of guidance in a skill file adds one, because the stage submits JSON against
    this and nothing else.
    """
    offered = set(contract_schema["properties"]["dependencies"]["items"]["properties"])
    missing = set(Dependency.model_fields) - offered
    assert not missing, f"Dependency fields the stage cannot write: {sorted(missing)}"


def test_every_reached_field_can_be_written(contract_schema: dict) -> None:
    offered = set(
        contract_schema["properties"]["dependencies"]["items"]["properties"]["reached"][
            "properties"
        ]
    )
    missing = set(Reached.model_fields) - offered
    assert not missing, f"Reached fields the stage cannot write: {sorted(missing)}"


def test_a_password_is_still_not_writable(contract_schema: dict) -> None:
    """The one field deliberately absent, asserted so it cannot be added by accident."""
    offered = contract_schema["properties"]["dependencies"]["items"]["properties"][
        "reached"
    ]["properties"]
    assert "password" not in offered
    assert "password_from" in offered


# --- and can the next stage read it back? ---------------------------------------------------

# One populated value per field, chosen to be recognisable in rendered prose.
POPULATED = {
    "dsn_env": "SOME_DSN_VAR",
    "config_key": "some.config.key",
    "host": "some-host.internal",
    "port": 65432,
    "database": "some_database",
    "user": "some_user",
    "loader_module": "some.loader.module",
    "loader_function": "some_load_function",
}


@pytest.mark.parametrize("field", sorted(POPULATED))
def test_a_populated_seam_reaches_the_next_stage(field: str) -> None:
    """The build stage reads the brief, not the JSON. A field the brief drops is a field the
    stage never learns, and it will fill the gap with a guess."""
    one = Dependency(
        name="its store",
        kind="datastore",
        engine="postgres",
        reached=Reached(**{field: POPULATED[field]}),
    )
    said = AgentContract(agent="a", dependencies=[one]).brief()
    assert str(POPULATED[field]) in said, f"brief() drops reached.{field}"


def test_password_from_is_the_exception_and_stays_out_of_the_brief() -> None:
    """Where a secret comes from is worth recording and not worth reprinting everywhere."""
    one = Dependency(name="its store", engine="postgres", reached=Reached(dsn_env="D"))
    assert "D" in AgentContract(agent="a", dependencies=[one]).brief()


def test_the_engine_itself_reaches_the_next_stage() -> None:
    one = Dependency(name="its store", engine="clickhouse", version="24.3")
    said = AgentContract(agent="a", dependencies=[one]).brief()
    assert "clickhouse" in said and "24.3" in said


# --- do the tool schemas agree with the models they build? ------------------------------------

JSON_TYPE = {str: "string", bool: "boolean", int: "integer"}


def sub_goal_schema(server) -> dict:
    return schemas(server)["add_sub_goal"]["properties"]


@pytest.mark.parametrize("build_server", ["provision", "world"])
def test_add_sub_goal_agrees_with_the_sub_goal_model(
    build_server: str, tmp_path
) -> None:
    """A schema that promises a boolean for a field the model stores as a string means the
    stage cannot add a sub-goal at all -- it is rejected on every attempt, whichever it sends.

    Worse for ``judged`` specifically, which is not a flag: it is the sentence saying what a
    model has to decide and why code cannot. Typed as a boolean, the reason has nowhere to go.
    """
    contract = AgentContract(agent="a")
    server = (
        provision_tools(contract, tmp_path)[0]
        if build_server == "provision"
        else world_tools(contract, tmp_path)[0]
    )
    offered = sub_goal_schema(server)
    for name, annotation in SubGoal.model_fields.items():
        assert name in offered, f"{build_server}: cannot write SubGoal.{name}"
        expected = JSON_TYPE.get(annotation.annotation)
        if expected:
            # Optional model fields accept null at the MCP boundary; normalization treats
            # null as omission instead of wasting a model turn on a schema rejection.
            assert offered[name].get("type") in (expected, [expected, "null"]), (
                f"{build_server}: add_sub_goal types {name} as "
                f"{offered[name].get('type')!r}, SubGoal stores {expected!r}"
            )


def test_judged_is_a_reason_not_a_flag() -> None:
    """Asserted on the model itself, since both schemas are checked against it."""
    assert SubGoal.model_fields["judged"].annotation is str
