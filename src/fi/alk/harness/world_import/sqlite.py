"""Import a legacy SQLite authored world into the typed World IR.

The importer is deliberately source-model driven.  SQLite affinity is not allowed to decide
PostgreSQL semantics: a JSON string becomes an array only when the discovered source column is
an array, and a JSON string becomes JSON only when the source column is JSON.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import sqlite3
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..source_model import LogicalType, SourceColumn, SourceModel, SourceTable
from ..world_ir import WorldIR, WorldRow, WorldTable, WorldValue, validate_world_ir


class SQLiteNullPolicy(str, Enum):
    """How to interpret NULL from an artifact that cannot represent omission."""

    LEGACY_ABSENT = "legacy_absent"
    PRESERVE_NULL = "preserve_null"


class SQLiteImportDecision(BaseModel):
    """Value-free compatibility decision safe to persist."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    table: str
    row_identity: str
    column: str | None = None


class SQLiteWorldImportResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    world: WorldIR
    decisions: tuple[SQLiteImportDecision, ...] = ()


class SQLiteWorldImportError(ValueError):
    """A value-free import failure; authored values never appear in the exception."""

    def __init__(
        self,
        code: str,
        *,
        table: str,
        column: str | None = None,
        row_identity: str | None = None,
    ) -> None:
        self.code = code
        self.table = table
        self.column = column
        self.row_identity = row_identity
        location = ".".join(
            item for item in (table, row_identity, column) if item is not None
        )
        super().__init__(f"{code}: {location}")


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sqlite_tables(connection: sqlite3.Connection) -> tuple[str, ...]:
    return tuple(
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    )


def _sqlite_columns(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(
        str(row[1])
        for row in connection.execute(f"PRAGMA table_xinfo({_identifier(table)})")
        if not bool(row[6])
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _row_identity(table: SourceTable, row: dict[str, Any]) -> str:
    if table.primary_key and all(
        row.get(column) is not None for column in table.primary_key
    ):
        material = [row[column] for column in table.primary_key]
    else:
        material = {column: row[column] for column in sorted(row)}
    digest = hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()[:24]
    return f"row-{digest}"


def _json_value(value: Any) -> Any:
    # Round-tripping is a compact way to reject tuples, custom objects, NaN and infinity while
    # returning only Pydantic's JSON-compatible value family.
    return json.loads(_canonical_json(value))


def _parse_structured(value: Any, *, expected: type, code: str) -> Any:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(code) from exc
    if not isinstance(parsed, expected):
        raise ValueError(code)
    return parsed


def _convert(column: SourceColumn, value: Any) -> Any:
    logical = column.logical_type
    if logical is LogicalType.BOOLEAN:
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in {
            "true",
            "false",
            "0",
            "1",
        }:
            return value.strip().lower() in {"true", "1"}
        raise ValueError("boolean_value_invalid")
    if logical is LogicalType.INTEGER:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise ValueError("integer_value_invalid")
    if logical is LogicalType.NUMBER:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("number_value_invalid")
            return value
        raise ValueError("number_value_invalid")
    if logical in {
        LogicalType.STRING,
        LogicalType.UUID,
        LogicalType.TIMESTAMP,
        LogicalType.DATE,
        LogicalType.ENUM,
    }:
        if not isinstance(value, str):
            raise ValueError(f"{logical.value}_value_invalid")
        return value
    if logical is LogicalType.ARRAY:
        return _json_value(
            _parse_structured(value, expected=list, code="array_value_invalid")
        )
    if logical is LogicalType.JSON:
        parsed = value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("json_value_invalid") from exc
        return _json_value(parsed)
    if logical is LogicalType.BINARY:
        if isinstance(value, bytes):
            return base64.b64encode(value).decode("ascii")
        if isinstance(value, str):
            base64.b64decode(value, validate=True)
            return value
        raise ValueError("binary_value_invalid")
    raise ValueError("logical_type_unsupported")


def import_sqlite_world(
    connection: sqlite3.Connection,
    source: SourceModel,
    *,
    null_policy: SQLiteNullPolicy = SQLiteNullPolicy.LEGACY_ABSENT,
) -> SQLiteWorldImportResult:
    """Import rows against authoritative source facts and validate the resulting World IR."""

    source_tables = {table.name: table for table in source.tables}
    sqlite_tables = _sqlite_tables(connection)
    unknown_tables = sorted(set(sqlite_tables) - set(source_tables))
    if unknown_tables:
        raise SQLiteWorldImportError("unknown_table", table=unknown_tables[0])

    decisions: list[SQLiteImportDecision] = []
    world_tables: list[WorldTable] = []
    for table_name in sqlite_tables:
        source_table = source_tables[table_name]
        source_columns = {column.name: column for column in source_table.columns}
        authored_columns = _sqlite_columns(connection, table_name)
        unknown_columns = sorted(set(authored_columns) - set(source_columns))
        if unknown_columns:
            raise SQLiteWorldImportError(
                "unknown_column", table=table_name, column=unknown_columns[0]
            )
        selected = ", ".join(_identifier(column) for column in authored_columns)
        ordering = selected
        query = f"SELECT {selected} FROM {_identifier(table_name)}"
        if ordering:
            query += f" ORDER BY {ordering}"
        raw_rows = [
            dict(zip(authored_columns, row, strict=True))
            for row in connection.execute(query).fetchall()
        ]
        identities: dict[str, int] = {}
        rows: list[WorldRow] = []
        for raw_row in raw_rows:
            base_identity = _row_identity(source_table, raw_row)
            collision = identities.get(base_identity, 0)
            identities[base_identity] = collision + 1
            identity = (
                base_identity if collision == 0 else f"{base_identity}-{collision + 1}"
            )
            values: dict[str, WorldValue] = {}
            for column in source_table.columns:
                if column.name not in raw_row:
                    values[column.name] = WorldValue.absent()
                    continue
                raw_value = raw_row[column.name]
                if raw_value is None:
                    if null_policy is SQLiteNullPolicy.LEGACY_ABSENT:
                        values[column.name] = WorldValue.absent()
                        decisions.append(
                            SQLiteImportDecision(
                                code="legacy_null_interpreted_as_absent",
                                table=table_name,
                                row_identity=identity,
                                column=column.name,
                            )
                        )
                    else:
                        values[column.name] = WorldValue.null(column.logical_type)
                    continue
                try:
                    converted = _convert(column, raw_value)
                except (TypeError, ValueError) as exc:
                    code = str(exc) or "value_conversion_failed"
                    raise SQLiteWorldImportError(
                        code,
                        table=table_name,
                        row_identity=identity,
                        column=column.name,
                    ) from None
                values[column.name] = WorldValue.present(column.logical_type, converted)
                if column.logical_type in {
                    LogicalType.ARRAY,
                    LogicalType.JSON,
                } and isinstance(raw_value, str):
                    decisions.append(
                        SQLiteImportDecision(
                            code=f"legacy_string_normalized_to_{column.logical_type.value}",
                            table=table_name,
                            row_identity=identity,
                            column=column.name,
                        )
                    )
            rows.append(WorldRow(identity=identity, values=values))
        world_tables.append(
            WorldTable(
                source_name=table_name,
                rows=tuple(sorted(rows, key=lambda row: row.identity)),
            )
        )

    world = WorldIR.create(
        source_model_fingerprint=source.fingerprint,
        tables=tuple(world_tables),
    )
    validate_world_ir(world, source)
    return SQLiteWorldImportResult(world=world, decisions=tuple(decisions))


__all__ = [
    "SQLiteImportDecision",
    "SQLiteNullPolicy",
    "SQLiteWorldImportError",
    "SQLiteWorldImportResult",
    "import_sqlite_world",
]
