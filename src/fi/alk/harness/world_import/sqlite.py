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


__all__ = [
    "SQLiteImportDecision",
    "SQLiteNullPolicy",
    "SQLiteWorldImportError",
    "SQLiteWorldImportResult",
    "array_value",
    "import_sqlite_world",
]
