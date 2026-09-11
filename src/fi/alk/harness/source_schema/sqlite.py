"""Loss-minimizing SQLite catalogue inspection."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from typing import Any

from ..source_model import (
    ForeignKey,
    LogicalType,
    SourceColumn,
    SourceModel,
    SourceTable,
    UnsupportedSourceConstruct,
)


def _quoted_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _logical_type(native_type: str) -> tuple[LogicalType, LogicalType | None]:
    normalized = native_type.strip().upper()
    if normalized.endswith("[]"):
        element_native = normalized[:-2].strip()
        element = _logical_type(element_native)[0]
        return LogicalType.ARRAY, element
    if normalized.startswith("ARRAY"):
        element_native = normalized.removeprefix("ARRAY").strip(" <>()")
        element = (
            _logical_type(element_native)[0] if element_native else LogicalType.STRING
        )
        return LogicalType.ARRAY, element
    if "BOOL" in normalized:
        return LogicalType.BOOLEAN, None
    if "UUID" in normalized:
        return LogicalType.UUID, None
    if "TIMESTAMP" in normalized or "DATETIME" in normalized:
        return LogicalType.TIMESTAMP, None
    if normalized == "DATE" or normalized.startswith("DATE("):
        return LogicalType.DATE, None
    if "JSON" in normalized:
        return LogicalType.JSON, None
    if "BLOB" in normalized or "BINARY" in normalized:
        return LogicalType.BINARY, None
    if "INT" in normalized:
        return LogicalType.INTEGER, None
    if any(token in normalized for token in ("REAL", "FLOA", "DOUB", "NUM", "DEC")):
        return LogicalType.NUMBER, None
    return LogicalType.STRING, None


def _rows(connection: sqlite3.Connection, statement: str) -> list[dict[str, Any]]:
    cursor = connection.execute(statement)
    columns = [item[0] for item in cursor.description or ()]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def _columns(connection: sqlite3.Connection, table: str) -> tuple[SourceColumn, ...]:
    rows = _rows(connection, f"PRAGMA table_xinfo({_quoted_identifier(table)})")
    columns: list[SourceColumn] = []
    for row in sorted(rows, key=lambda item: int(item["cid"])):
        native_type = str(row["type"] or "")
        logical_type, element_type = _logical_type(native_type)
        default = row["dflt_value"]
        columns.append(
            SourceColumn(
                name=str(row["name"]),
                logical_type=logical_type,
                native_type=native_type,
                nullable=not bool(row["notnull"]) and not bool(row["pk"]),
                has_default=default is not None,
                default_expression=str(default) if default is not None else None,
                generated=bool(row.get("hidden", 0)),
                element_type=element_type,
            )
        )
    return tuple(columns)


def _primary_key(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
    rows = _rows(connection, f"PRAGMA table_info({_quoted_identifier(table)})")
    keyed = [row for row in rows if int(row["pk"])]
    return tuple(
        str(row["name"]) for row in sorted(keyed, key=lambda item: int(item["pk"]))
    )


def _unique_keys(
    connection: sqlite3.Connection,
    table: str,
    unsupported: list[UnsupportedSourceConstruct],
) -> tuple[tuple[str, ...], ...]:
    keys: set[tuple[str, ...]] = set()
    indexes = _rows(connection, f"PRAGMA index_list({_quoted_identifier(table)})")
    for index in indexes:
        if not bool(index["unique"]) or str(index.get("origin") or "") == "pk":
            continue
        name = str(index["name"])
        columns = _rows(connection, f"PRAGMA index_info({_quoted_identifier(name)})")
        if bool(index.get("partial")) or any(row["name"] is None for row in columns):
            unsupported.append(
                UnsupportedSourceConstruct(
                    code="sqlite_unique_index_not_column_only",
                    component="schema",
                    location=f"{table}.{name}",
                )
            )
            continue
        key = tuple(
            str(row["name"])
            for row in sorted(columns, key=lambda item: int(item["seqno"]))
            if row["name"] is not None
        )
        if key:
            keys.add(key)
    return tuple(sorted(keys))


def _foreign_keys(
    connection: sqlite3.Connection,
    table: str,
    unsupported: list[UnsupportedSourceConstruct],
) -> tuple[ForeignKey, ...]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in _rows(
        connection, f"PRAGMA foreign_key_list({_quoted_identifier(table)})"
    ):
        grouped[int(row["id"])].append(row)
    keys: list[ForeignKey] = []
    for identifier in sorted(grouped):
        parts = sorted(grouped[identifier], key=lambda item: int(item["seq"]))
        if any(row["to"] is None for row in parts):
            unsupported.append(
                UnsupportedSourceConstruct(
                    code="sqlite_implicit_foreign_key_target",
                    component="schema",
                    location=f"{table}.foreign_key.{identifier}",
                )
            )
            continue
        keys.append(
            ForeignKey(
                columns=tuple(str(row["from"]) for row in parts),
                referenced_table=str(parts[0]["table"]),
                referenced_columns=tuple(str(row["to"]) for row in parts),
                on_update=str(parts[0]["on_update"]),
                on_delete=str(parts[0]["on_delete"]),
            )
        )
    return tuple(keys)


def inspect_sqlite(
    connection: sqlite3.Connection,
    *,
    source_digest: str,
    configuration_names: tuple[str, ...] = (),
) -> SourceModel:
    """Inspect a live SQLite database using executable catalogue metadata."""

    tables = [
        str(row["name"])
        for row in _rows(
            connection,
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
        )
    ]
    unsupported: list[UnsupportedSourceConstruct] = []
    discovered = tuple(
        SourceTable(
            name=table,
            columns=_columns(connection, table),
            primary_key=_primary_key(connection, table),
            unique_keys=_unique_keys(connection, table, unsupported),
            foreign_keys=_foreign_keys(connection, table, unsupported),
        )
        for table in tables
    )
    return SourceModel.create(
        source_digest=source_digest,
        engine="sqlite",
        engine_version=sqlite3.sqlite_version,
        tables=discovered,
        configuration_names=configuration_names,
        unsupported=tuple(unsupported),
    )


__all__ = ["inspect_sqlite"]
