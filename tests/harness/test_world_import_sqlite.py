from __future__ import annotations

import sqlite3

import pytest

from fi.alk.harness.source_model import (
    LogicalType,
    SourceColumn,
    SourceModel,
    SourceTable,
)
from fi.alk.harness.world_import.sqlite import (
    SQLiteNullPolicy,
    SQLiteWorldImportError,
    array_value,
    import_sqlite_world,
    world_holds,
    write_sqlite_world,
)
from fi.alk.harness.world_ir import (
    ValueState,
    WorldIR,
    WorldRow,
    WorldTable,
    WorldValue,
)


def _column(
    name: str,
    logical_type: LogicalType,
    *,
    nullable: bool = False,
    default: str | None = None,
    element_type: LogicalType | None = None,
) -> SourceColumn:
    return SourceColumn(
        name=name,
        logical_type=logical_type,
        native_type="text[]"
        if logical_type is LogicalType.ARRAY
        else logical_type.value,
        nullable=nullable,
        has_default=default is not None,
        default_expression=default,
        element_type=element_type,
    )


def _source(*columns: SourceColumn) -> SourceModel:
    return SourceModel.create(
        source_digest="sha256:" + "a" * 64,
        engine="postgres",
        tables=(SourceTable(name="users", columns=columns, primary_key=("id",)),),
    )


def test_import_uses_source_types_for_legacy_json_strings() -> None:
    source = _source(
        _column("id", LogicalType.STRING),
        _column(
            "accessibility_needs", LogicalType.ARRAY, element_type=LogicalType.STRING
        ),
        _column("profile", LogicalType.JSON),
        _column("active", LogicalType.BOOLEAN),
    )
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE users (id TEXT, accessibility_needs TEXT, profile TEXT, active INTEGER)"
    )
    connection.execute(
        "INSERT INTO users VALUES (?, ?, ?, ?)",
        ("rider-1", '["wheelchair"]', '{"tier":"gold"}', 1),
    )

    result = import_sqlite_world(connection, source)

    row = result.world.tables[0].rows[0]
    assert row.values["accessibility_needs"].value == ["wheelchair"]
    assert row.values["profile"].value == {"tier": "gold"}
    assert row.values["active"].value is True
    assert {decision.code for decision in result.decisions} == {
        "legacy_string_normalized_to_array",
        "legacy_string_normalized_to_json",
    }


def test_import_normalizes_legacy_python_structured_literals() -> None:
    source = _source(
        _column("id", LogicalType.STRING),
        _column("items", LogicalType.ARRAY, element_type=LogicalType.STRING),
        _column("payload", LogicalType.JSON),
    )
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE users (id TEXT, items TEXT, payload TEXT)")
    connection.execute(
        "INSERT INTO users VALUES (?, ?, ?)",
        ("row-1", "['room', 'tax']", "{'currency': 'USD', 'amount': 42}"),
    )

    result = import_sqlite_world(connection, source)

    row = result.world.tables[0].rows[0]
    assert row.values["items"].value == ["room", "tax"]
    assert row.values["payload"].value == {"currency": "USD", "amount": 42}


def test_import_still_rejects_non_literal_json_text() -> None:
    source = _source(
        _column("id", LogicalType.STRING),
        _column("payload", LogicalType.JSON),
    )
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE users (id TEXT, payload TEXT)")
    connection.execute("INSERT INTO users VALUES ('row-1', 'not-json-or-a-literal')")

    with pytest.raises(SQLiteWorldImportError) as raised:
        import_sqlite_world(connection, source)

    assert raised.value.code == "json_value_invalid"


def test_import_keeps_legacy_null_as_absent_but_can_preserve_explicit_null() -> None:
    source = _source(
        _column("id", LogicalType.STRING),
        _column("note", LogicalType.STRING, nullable=True),
        _column("status", LogicalType.STRING, nullable=True, default="'pending'"),
    )
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE users (id TEXT, note TEXT, status TEXT)")
    connection.execute("INSERT INTO users VALUES ('rider-1', NULL, NULL)")

    legacy = import_sqlite_world(connection, source)
    assert legacy.world.tables[0].rows[0].values["note"].state is ValueState.ABSENT
    assert len(legacy.decisions) == 2

    explicit = import_sqlite_world(
        connection, source, null_policy=SQLiteNullPolicy.PRESERVE_NULL
    )
    assert explicit.world.tables[0].rows[0].values["note"].state is ValueState.NULL


def test_import_rejects_unknown_schema_without_exposing_values() -> None:
    source = _source(_column("id", LogicalType.STRING))
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE users (id TEXT, leaked_secret TEXT)")
    connection.execute("INSERT INTO users VALUES ('rider-1', 'do-not-report')")

    with pytest.raises(SQLiteWorldImportError) as raised:
        import_sqlite_world(connection, source)

    assert raised.value.code == "unknown_column"
    assert "do-not-report" not in str(raised.value)


def test_import_rejects_malformed_array_before_postgres() -> None:
    source = _source(
        _column("id", LogicalType.STRING),
        _column("tags", LogicalType.ARRAY, element_type=LogicalType.STRING),
    )
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE users (id TEXT, tags TEXT)")
    connection.execute("INSERT INTO users VALUES ('rider-1', 'not-an-array')")

    with pytest.raises(SQLiteWorldImportError) as raised:
        import_sqlite_world(connection, source)

    assert raised.value.code == "array_value_invalid"
    assert raised.value.logical_type is LogicalType.ARRAY
    assert raised.value.sqlite_type == "str"
    assert "not-an-array" not in str(raised.value)


def test_import_normalizes_legacy_naive_timestamp_to_utc() -> None:
    source = _source(
        _column("id", LogicalType.STRING),
        _column("issued_at", LogicalType.TIMESTAMP),
    )
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE users (id TEXT, issued_at TEXT)")
    connection.execute(
        "INSERT INTO users VALUES (?, ?)",
        ("rider-1", "2026-09-12T12:30:00"),
    )

    result = import_sqlite_world(connection, source)

    row = result.world.tables[0].rows[0]
    assert row.values["issued_at"].value == "2026-09-12T12:30:00+00:00"
    assert [decision.code for decision in result.decisions] == [
        "legacy_naive_timestamp_normalized_to_utc"
    ]


def test_import_reads_a_postgres_array_default_sqlite_stored_as_text() -> None:
    source = _source(
        _column("id", LogicalType.STRING),
        _column(
            "accessibility_needs",
            LogicalType.ARRAY,
            nullable=True,
            default="'{}'::text[]",
            element_type=LogicalType.STRING,
        ),
    )
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE users (id TEXT PRIMARY KEY, accessibility_needs TEXT[] DEFAULT '{}')"
    )
    connection.execute("INSERT INTO users (id) VALUES ('rider-1')")

    result = import_sqlite_world(connection, source)

    row = result.world.tables[0].rows[0]
    assert row.values["accessibility_needs"].value == []


@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        ("{}", []),
        ("{ }", []),
        ('{wheelchair,"service animal",NULL}', ["wheelchair", "service animal", None]),
        ('{"a \\"quoted\\" one"}', ['a "quoted" one']),
    ],
)
def test_array_value_reads_postgres_array_literals(literal, expected) -> None:
    assert array_value(literal, LogicalType.STRING) == expected


@pytest.mark.parametrize(
    "literal",
    ['{"k":1}', '{"k": [1]}', "{a", "{a,}", "{{a},{b}}", '{"open}', "nope", "{a\\,b}"],
)
def test_array_value_refuses_what_is_not_an_array(literal) -> None:
    with pytest.raises(ValueError, match="array_value_invalid"):
        array_value(literal, LogicalType.STRING)


def test_array_value_does_not_guess_element_types_from_a_literal() -> None:
    assert array_value("{}", LogicalType.INTEGER) == []
    assert array_value("[1, 2]", LogicalType.INTEGER) == [1, 2]
    with pytest.raises(ValueError, match="array_value_invalid"):
        array_value("{1,2}", LogicalType.INTEGER)


def test_a_world_written_back_imports_to_the_same_world() -> None:
    source = SourceModel.create(
        source_digest="sha256:" + "a" * 64,
        engine="postgres",
        tables=(
            SourceTable(
                name="users",
                columns=(
                    _column("id", LogicalType.STRING),
                    _column("tags", LogicalType.ARRAY, element_type=LogicalType.STRING),
                    _column("profile", LogicalType.JSON),
                    _column("active", LogicalType.BOOLEAN),
                    _column("balance", LogicalType.NUMBER, nullable=True),
                    _column("note", LogicalType.STRING, nullable=True, default="'x'"),
                ),
                primary_key=("id",),
            ),
            SourceTable(name="trips", columns=(_column("id", LogicalType.INTEGER),)),
        ),
    )
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE users (id TEXT, tags TEXT, profile TEXT, active)")
    connection.execute("INSERT INTO users VALUES ('u1', '{}', '{\"a\": 1}', 'true')")
    world = import_sqlite_world(connection, source).world
    [row] = world.tables[0].rows
    # A repair: a value set, a column the table lacks, a row and a table added.
    edited = WorldIR.create(
        source_model_fingerprint=world.source_model_fingerprint,
        tables=(
            WorldTable(
                source_name="trips",
                rows=(
                    WorldRow(
                        identity="t",
                        values={"id": WorldValue.present(LogicalType.INTEGER, 7)},
                    ),
                ),
            ),
            WorldTable(
                source_name="users",
                rows=tuple(
                    sorted(
                        (
                            WorldRow(
                                identity=row.identity,
                                values={
                                    **row.values,
                                    "balance": WorldValue.present(
                                        LogicalType.NUMBER, 7.25
                                    ),
                                    "profile": WorldValue.present(
                                        LogicalType.JSON, "plain text"
                                    ),
                                },
                            ),
                            WorldRow(
                                identity="u2",
                                values={
                                    "id": WorldValue.present(LogicalType.STRING, "u2"),
                                    "tags": WorldValue.present(
                                        LogicalType.ARRAY, ["a", "b"]
                                    ),
                                    "profile": WorldValue.present(
                                        LogicalType.JSON, None
                                    ),
                                    "active": WorldValue.present(
                                        LogicalType.BOOLEAN, False
                                    ),
                                    "note": WorldValue.absent(),
                                },
                            ),
                        ),
                        key=lambda one: one.identity,
                    )
                ),
            ),
        ),
    )

    write_sqlite_world(connection, edited, source)
    stored = import_sqlite_world(connection, source).world

    def values(ir: WorldIR) -> dict:
        return {
            table.source_name: sorted(
                (
                    {
                        name: value.model_dump(mode="json")
                        for name, value in one.values.items()
                        if value.state is not ValueState.ABSENT
                    }
                    for one in table.rows
                ),
                key=lambda item: str(item.get("id")),
            )
            for table in ir.tables
        }

    assert values(stored) == values(edited)


def test_a_repair_writes_only_what_it_changes_and_lets_defaults_apply() -> None:
    source = SourceModel.create(
        source_digest="sha256:" + "a" * 64,
        engine="postgres",
        tables=(
            SourceTable(
                name="users",
                columns=(
                    _column("id", LogicalType.STRING),
                    _column("active", LogicalType.BOOLEAN),
                    _column("made", LogicalType.STRING, default="'now'"),
                ),
                primary_key=("id",),
            ),
        ),
    )
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE users (id TEXT PRIMARY KEY, active, "
        "made TEXT NOT NULL DEFAULT 'now')"
    )
    connection.execute("INSERT INTO users VALUES ('a', 'true', 'then')")
    connection.execute("INSERT INTO users VALUES ('b', 'true', 'then')")
    world = import_sqlite_world(connection, source).world
    a, b = world.tables[0].rows
    added = WorldRow(
        identity="c",
        values={
            "id": WorldValue.present(LogicalType.STRING, "c"),
            "active": WorldValue.present(LogicalType.BOOLEAN, True),
            "made": WorldValue.absent(),
        },
    )
    unchanged = WorldValue.present(LogicalType.BOOLEAN, True)
    repaired = WorldIR.create(
        source_model_fingerprint=world.source_model_fingerprint,
        tables=(
            WorldTable(
                source_name="users",
                rows=tuple(
                    sorted(
                        (
                            a.model_copy(
                                update={
                                    "values": {
                                        **a.values,
                                        "active": WorldValue.present(
                                            LogicalType.BOOLEAN, False
                                        ),
                                    }
                                }
                            ),
                            b,
                            added,
                        ),
                        key=lambda one: one.identity,
                    )
                ),
            ),
        ),
    )
    assert b.values["active"] == unchanged

    assert write_sqlite_world(connection, repaired, source)
    assert (
        write_sqlite_world(
            connection, import_sqlite_world(connection, source).world, source
        )
        is False
    )

    assert connection.execute("SELECT * FROM users ORDER BY id").fetchall() == [
        ("a", 0, "then"),
        ("b", "true", "then"),
        ("c", 1, "now"),
    ]


def test_a_default_only_sqlite_has_does_not_count_as_the_world() -> None:
    source = _source(
        _column("id", LogicalType.STRING),
        _column("note", LogicalType.STRING, nullable=True),
    )
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE users (id TEXT PRIMARY KEY, note TEXT DEFAULT 'x')"
    )
    world = import_sqlite_world(connection, source).world
    added = WorldIR.create(
        source_model_fingerprint=world.source_model_fingerprint,
        tables=(
            WorldTable(
                source_name="users",
                rows=(
                    WorldRow(
                        identity="u1",
                        values={
                            "id": WorldValue.present(LogicalType.STRING, "u1"),
                            "note": WorldValue.absent(),
                        },
                    ),
                ),
            ),
        ),
    )

    write_sqlite_world(connection, added, source)

    # The source has no default, so a run seeds NULL; SQLite filled 'x'.
    stored = import_sqlite_world(connection, source).world
    assert not world_holds(stored, added, source)
