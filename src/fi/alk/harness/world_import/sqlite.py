"""Import a legacy SQLite authored world into the typed World IR.

The importer is deliberately source-model driven.  SQLite affinity is not allowed to decide
PostgreSQL semantics: a JSON string becomes an array only when the discovered source column is
an array, and a JSON string becomes JSON only when the source column is JSON.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..source_model import LogicalType, SourceColumn, SourceModel, SourceTable
from ..world_ir import (
    ValueState,
    WorldIR,
    WorldRow,
    WorldTable,
    WorldValue,
    validate_world_ir,
)


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
        logical_type: LogicalType | None = None,
        sqlite_type: str | None = None,
    ) -> None:
        self.code = code
        self.table = table
        self.column = column
        self.row_identity = row_identity
        self.logical_type = logical_type
        self.sqlite_type = sqlite_type
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


def _identities(table: SourceTable, rows: list[dict[str, Any]]) -> list[str]:
    """Each row's identity; equal rows are told apart by a suffix, in order."""
    seen: dict[str, int] = {}
    identities = []
    for row in rows:
        base = _row_identity(table, row)
        collision = seen.get(base, 0)
        seen[base] = collision + 1
        identities.append(base if collision == 0 else f"{base}-{collision + 1}")
    return identities


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


def _parse_structured(
    value: Any, *, expected: type | tuple[type, ...], code: str
) -> Any:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            # Older/generated SQLite worlds sometimes persist Python's repr for
            # a JSON-compatible list or mapping (single-quoted keys/values),
            # while the authoritative source is PostgreSQL JSON/ARRAY. Parse
            # literals only--never evaluate code--at this legacy boundary.
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, TypeError, ValueError) as exc:
                raise ValueError(code) from exc
    if not isinstance(parsed, expected):
        raise ValueError(code)
    return parsed


def _postgres_array(text: str) -> list[str | None]:
    """A one-dimensional Postgres array literal such as ``{}`` or ``{a,"b c",NULL}``."""
    if not (text.startswith("{") and text.endswith("}")):
        raise ValueError("array_value_invalid")
    body, at = text[1:-1], 0
    items: list[str | None] = []
    if not body.strip():
        return []
    while True:
        while at < len(body) and body[at].isspace():
            at += 1
        if at < len(body) and body[at] == '"':
            at += 1
            chars: list[str] = []
            while at < len(body) and body[at] != '"':
                at += body[at] == "\\"
                if at >= len(body):
                    break
                chars.append(body[at])
                at += 1
            if at >= len(body):
                raise ValueError("array_value_invalid")
            at += 1
            items.append("".join(chars))
        else:
            end = at
            while end < len(body) and body[end] not in ',"{}\\':
                end += 1
            word = body[at:end].strip()
            if not word:
                raise ValueError("array_value_invalid")
            items.append(None if word.upper() == "NULL" else word)
            at = end
        while at < len(body) and body[at].isspace():
            at += 1
        if at == len(body):
            return items
        if body[at] != ",":
            raise ValueError("array_value_invalid")
        at += 1


_TEXT_ELEMENTS = {None, LogicalType.STRING, LogicalType.UUID, LogicalType.ENUM}


def array_value(value: Any, element_type: LogicalType | None = None) -> list[Any]:
    """An array cell as a list: JSON, a legacy Python repr, or a Postgres array literal.

    The literal is what SQLite stores when a column declared ``TEXT[] DEFAULT '{}'`` is left
    out of an insert. A JSON array never starts with ``{``, so reading it here is unambiguous.
    """
    try:
        return _json_value(
            _parse_structured(value, expected=list, code="array_value_invalid")
        )
    except ValueError:
        if not (isinstance(value, str) and value.startswith("{")):
            raise
    items = _postgres_array(value)
    if items and element_type not in _TEXT_ELEMENTS:
        raise ValueError("array_value_invalid")
    return items


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
    if logical is LogicalType.TIMESTAMP:
        if not isinstance(value, str):
            raise ValueError("timestamp_value_invalid")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp_value_invalid") from exc
        # SQLite has no native timestamp-with-time-zone representation. Legacy
        # authoring databases commonly contain ISO wall-clock values even when
        # the source model was discovered from PostgreSQL. The World IR requires
        # an offset for deterministic replay, so use UTC at this compatibility
        # boundary instead of rejecting otherwise valid authored rows.
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.isoformat()
    if logical is LogicalType.DATE:
        if not isinstance(value, str):
            raise ValueError("date_value_invalid")
        try:
            datetime.fromisoformat(value).date()
        except ValueError as exc:
            raise ValueError("date_value_invalid") from exc
        return value
    if logical in {
        LogicalType.STRING,
        LogicalType.UUID,
        LogicalType.ENUM,
    }:
        if not isinstance(value, str):
            raise ValueError(f"{logical.value}_value_invalid")
        return value
    if logical is LogicalType.ARRAY:
        return array_value(value, column.element_type)
    if logical is LogicalType.JSON:
        parsed = _parse_structured(
            value,
            expected=(dict, list, str, int, float, bool, type(None)),
            code="json_value_invalid",
        )
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
    validate: bool = True,
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
        rows: list[WorldRow] = []
        for identity, raw_row in zip(
            _identities(source_table, raw_rows), raw_rows, strict=True
        ):
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
                        logical_type=column.logical_type,
                        sqlite_type=type(raw_value).__name__,
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
                if (
                    column.logical_type is LogicalType.TIMESTAMP
                    and isinstance(raw_value, str)
                    and converted != raw_value
                ):
                    decisions.append(
                        SQLiteImportDecision(
                            code="legacy_naive_timestamp_normalized_to_utc",
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
    if validate:
        validate_world_ir(world, source)
    return SQLiteWorldImportResult(world=world, decisions=tuple(decisions))


def _stored(value: WorldValue | None) -> Any:
    if value is None or value.state is not ValueState.PRESENT:
        return None
    if value.logical_type in {LogicalType.ARRAY, LogicalType.JSON}:
        return _canonical_json(value.value)
    if value.logical_type is LogicalType.BOOLEAN:
        return int(bool(value.value))
    return value.value


def _rowids(
    connection: sqlite3.Connection, table: SourceTable, name: str
) -> dict[str, int]:
    """Each row's identity, as the import gives it, mapped to its SQLite rowid."""
    columns = _sqlite_columns(connection, name)
    selected = ", ".join(_identifier(column) for column in columns)
    fetched = connection.execute(
        f"SELECT rowid, {selected} FROM {_identifier(name)} ORDER BY {selected}"
    ).fetchall()
    rows = [dict(zip(columns, row[1:], strict=True)) for row in fetched]
    return {
        identity: row[0]
        for identity, row in zip(_identities(table, rows), fetched, strict=True)
    }


def write_sqlite_world(
    connection: sqlite3.Connection, world: WorldIR, source: SourceModel
) -> bool:
    """Write the rows where a World IR differs from what a SQLite world imports to.

    Changed values are updated in place and new rows inserted with their present values
    only, so defaults apply and untouched rows stay as stored. Rows the IR lacks are
    kept. Returns whether anything was written; commits nothing.
    """
    tables = {table.name: table for table in source.tables}
    held = {
        table.source_name: {row.identity: row for row in table.rows}
        for table in import_sqlite_world(
            connection, source, validate=False
        ).world.tables
    }
    wrote = False
    for table in world.tables:
        name = table.source_name
        current = held.get(name, {})
        changes = [row for row in table.rows if current.get(row.identity) != row]
        if not changes:
            continue
        wrote = True
        if name not in held:
            connection.execute(
                f"CREATE TABLE {_identifier(name)} ("
                + ", ".join(_identifier(column.name) for column in tables[name].columns)
                + ")"
            )
        # Identities as the import gives them, before any column is added.
        rowids = _rowids(connection, tables[name], name)
        columns = set(_sqlite_columns(connection, name))
        for row in changes:
            for column, value in row.values.items():
                if column not in columns and _stored(value) is not None:
                    connection.execute(
                        f"ALTER TABLE {_identifier(name)} "
                        f"ADD COLUMN {_identifier(column)}"
                    )
                    columns.add(column)
        for row in changes:
            old = current.get(row.identity)
            if old is None:
                # Absent values are the database's to fill; a null is written as one.
                values = {
                    column: _stored(value)
                    for column, value in row.values.items()
                    if value.state is not ValueState.ABSENT and column in columns
                }
                connection.execute(
                    f"INSERT INTO {_identifier(name)} "
                    f"({', '.join(_identifier(column) for column in values)}) "
                    f"VALUES ({', '.join('?' for _ in values)})"
                    if values
                    else f"INSERT INTO {_identifier(name)} DEFAULT VALUES",
                    tuple(values.values()),
                )
                continue
            changed = {
                column: _stored(value)
                for column, value in row.values.items()
                if old.values.get(column) != value and column in columns
            }
            connection.execute(
                f"UPDATE {_identifier(name)} SET "
                + ", ".join(f"{_identifier(column)} = ?" for column in changed)
                + " WHERE rowid = ?",
                (*changed.values(), rowids.get(row.identity)),
            )
    return wrote


def world_holds(stored: WorldIR, world: WorldIR, source: SourceModel) -> bool:
    """Whether a stored world seeds what a World IR does: its rows, with every value
    it sets. A value left absent may be filled only where the source has a default; an
    explicit null there cannot be stored, since SQLite's NULL reads back as absent.
    """
    columns = {
        table.name: {column.name: column for column in table.columns}
        for table in source.tables
    }
    rows = {table.source_name: list(table.rows) for table in stored.tables}
    if {name for name, held in rows.items() if held} != {
        table.source_name for table in world.tables if table.rows
    }:
        return False
    for table in world.tables:
        held = rows.get(table.source_name, [])
        if len(held) != len(table.rows):
            return False
        for row in table.rows:
            wanted: dict[str, Any] = {}
            for name, column in columns[table.source_name].items():
                value = row.values.get(name, WorldValue.absent())
                if value.state is ValueState.PRESENT:
                    try:
                        wanted[name] = _convert(column, _stored(value))
                    except (TypeError, ValueError):
                        return False
                elif value.state is ValueState.NULL and column.has_default:
                    return False
                elif not column.has_default:
                    wanted[name] = None

            def holds(one: WorldRow) -> bool:
                return all(
                    one.values[name].state is not ValueState.PRESENT
                    if value is None
                    else one.values[name].state is ValueState.PRESENT
                    and one.values[name].value == value
                    for name, value in wanted.items()
                )

            # The same row first, so a looser added row cannot take another's place.
            match = next(
                (one for one in held if one.identity == row.identity and holds(one)),
                next((one for one in held if holds(one)), None),
            )
            if match is None:
                return False
            held.remove(match)
    return True


__all__ = [
    "SQLiteImportDecision",
    "SQLiteNullPolicy",
    "SQLiteWorldImportError",
    "SQLiteWorldImportResult",
    "array_value",
    "import_sqlite_world",
    "world_holds",
    "write_sqlite_world",
]
