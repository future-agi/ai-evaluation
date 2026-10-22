"""The world-vs-schema gate reads any agent's schema, and never traps the builder in a loop."""

from __future__ import annotations

from types import SimpleNamespace

from fi.alk.harness.world.tools import _still_refusing, _tables_the_source_lacks


def _contract():
    return SimpleNamespace(model_dump=lambda: {})


def test_a_schema_qualified_table_is_recognised(tmp_path):
    (tmp_path / "schema.sql").write_text(
        "CREATE TABLE public.orders (\n  id TEXT PRIMARY KEY,\n  total NUMERIC NOT NULL\n);\n"
    )
    state = {"orders": [{"id": "o1", "total": 10}]}
    assert _tables_the_source_lacks(state, str(tmp_path), _contract()) == []


def test_a_real_mismatch_is_still_reported(tmp_path):
    (tmp_path / "schema.sql").write_text("CREATE TABLE orders (id TEXT PRIMARY KEY);\n")
    state = {"orders": [{"id": "o1", "colour": "red"}]}
    assert _tables_the_source_lacks(state, str(tmp_path), _contract()) == ["orders.{colour}"]


def test_the_same_refusal_a_third_time_becomes_advice():
    counts: dict[str, int] = {}
    assert _still_refusing(counts, "schema", ["orders.{colour}"])
    assert _still_refusing(counts, "schema", ["orders.{colour}"])
    assert not _still_refusing(counts, "schema", ["orders.{colour}"])
    assert _still_refusing(counts, "schema", ["orders.{size}"])


def _array_world(tmp_path):
    """A provider chat world whose users table copies a Postgres array default."""
    import asyncio

    from fi.alk.harness.catalogue import Catalogue, SubGoal, save_catalogue
    from fi.alk.harness.certification import GenericHarnessArtifactStore
    from fi.alk.harness.contract import AgentContract
    from fi.alk.harness.simulator import save_simulator_prompt
    from fi.alk.harness.source_model import (
        LogicalType,
        SourceColumn,
        SourceModel,
        SourceTable,
    )
    from fi.alk.harness.world.tools import world_tools

    contract = AgentContract(
        agent="provider_chat",
        modality="chat",
        conversational=True,
        tools=[],
        real_use_cases=["Book a ride for a rider on file"],
    )
    save_catalogue(
        Catalogue(
            sub_goals=[
                SubGoal(
                    name="rider_on_file",
                    what="The rider is on file",
                    check=(
                        "def check(world, calls):\n"
                        "    return None if world.get('users') else 'no rider'\n"
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
    GenericHarnessArtifactStore(tmp_path / "generic-harness").write_source_model(
        SourceModel.create(
            source_digest="sha256:" + "a" * 64,
            engine="postgres",
            tables=(
                SourceTable(
                    name="users",
                    primary_key=("rider_id",),
                    columns=(
                        SourceColumn(
                            name="rider_id",
                            logical_type=LogicalType.STRING,
                            native_type="text",
                            nullable=False,
                            has_default=False,
                        ),
                        SourceColumn(
                            name="accessibility_needs",
                            logical_type=LogicalType.ARRAY,
                            native_type="text[]",
                            nullable=True,
                            has_default=True,
                            default_expression="'{}'::text[]",
                            element_type=LogicalType.STRING,
                        ),
                    ),
                ),
            ),
        )
    )
    server, _ = world_tools(contract, tmp_path)
    tools = {one.name: one.handler for one in server.tools}

    def call(name, args):
        return asyncio.run(tools[name](args))

    assert not call(
        "create_schema",
        {
            "sql": "CREATE TABLE users (rider_id TEXT PRIMARY KEY, "
            "accessibility_needs TEXT[] DEFAULT '{}');"
        },
    ).get("is_error")
    assert not call(
        "add_world_check",
        {
            "name": "riders_exist",
            "code": (
                "def check(world):\n"
                "    return None if world.state().get('users') else 'no riders'\n"
            ),
        },
    ).get("is_error")
    return call


def test_save_world_writes_an_omitted_array_default_as_json(tmp_path):
    import sqlite3

    from fi.alk.harness.certification import GenericHarnessArtifactStore
    from fi.alk.harness.world_import.sqlite import import_sqlite_world

    call = _array_world(tmp_path)
    assert not call(
        "seed",
        {
            "table": "users",
            "rows": [
                {"rider_id": "rdr_noor"},
                {"rider_id": "rdr_dana", "accessibility_needs": ["wheelchair"]},
            ],
        },
    ).get("is_error")

    saved = call("save_world", {})

    assert not saved.get("is_error"), saved
    source = GenericHarnessArtifactStore(tmp_path / "generic-harness")
    with sqlite3.connect(tmp_path / "world.sqlite") as connection:
        stored = dict(
            connection.execute("SELECT rider_id, accessibility_needs FROM users")
        )
        imported = import_sqlite_world(connection, source.read_source_model())
    assert stored == {"rdr_noor": "[]", "rdr_dana": '["wheelchair"]'}
    values = sorted(
        row.values["accessibility_needs"].value for row in imported.world.tables[0].rows
    )
    assert values == [[], ["wheelchair"]]


def test_save_world_refuses_an_array_cell_the_seed_cannot_read(tmp_path):
    call = _array_world(tmp_path)
    assert not call(
        "seed",
        {
            "table": "users",
            "rows": [{"rider_id": "rdr_noor", "accessibility_needs": "wheelchair"}],
        },
    ).get("is_error")

    refused = call("save_world", {})

    assert refused["is_error"]
    assert "users.accessibility_needs" in refused["content"][0]["text"]
    assert not (tmp_path / "world.sqlite").exists()
