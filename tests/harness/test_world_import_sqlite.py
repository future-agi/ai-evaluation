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
    import_sqlite_world,
)
from fi.alk.harness.world_ir import ValueState


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
    assert "not-an-array" not in str(raised.value)
