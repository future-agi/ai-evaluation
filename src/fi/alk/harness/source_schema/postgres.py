"""PostgreSQL catalogue inspection for the canonical source model."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from ..source_model import (
    CheckConstraint,
    ForeignKey,
    LogicalType,
    SourceColumn,
    SourceModel,
    SourceTable,
    UnsupportedSourceConstruct,
)

_CHECKS_SQL = """
SELECT cls.relname AS table_name,
       con.conname AS constraint_name,
       pg_catalog.pg_get_constraintdef(con.oid, true) AS expression
  FROM pg_catalog.pg_constraint con
  JOIN pg_catalog.pg_class cls ON cls.oid = con.conrelid
  JOIN pg_catalog.pg_namespace namespace ON namespace.oid = cls.relnamespace
 WHERE namespace.nspname = %s
   AND con.contype = 'c'
 ORDER BY cls.relname, con.conname
"""

_COLUMNS_SQL = """
SELECT cls.relname AS table_name,
       attr.attname AS column_name,
       attr.attnum AS ordinal_position,
       pg_catalog.format_type(attr.atttypid, attr.atttypmod) AS native_type,
       NOT attr.attnotnull AS nullable,
       defaults.adbin IS NOT NULL AS has_default,
       pg_catalog.pg_get_expr(defaults.adbin, defaults.adrelid) AS default_expression,
       attr.attgenerated <> '' AS generated,
       typ.typtype AS type_kind,
       typ.typname AS udt_name,
       element.typname AS element_udt_name
  FROM pg_catalog.pg_attribute attr
  JOIN pg_catalog.pg_class cls ON cls.oid = attr.attrelid
  JOIN pg_catalog.pg_namespace namespace ON namespace.oid = cls.relnamespace
  JOIN pg_catalog.pg_type typ ON typ.oid = attr.atttypid
  LEFT JOIN pg_catalog.pg_type element ON element.oid = typ.typelem
  LEFT JOIN pg_catalog.pg_attrdef defaults
    ON defaults.adrelid = attr.attrelid AND defaults.adnum = attr.attnum
 WHERE namespace.nspname = %s
   AND cls.relkind IN ('r', 'p')
   AND attr.attnum > 0
   AND NOT attr.attisdropped
 ORDER BY cls.relname, attr.attnum
"""

_ENUMS_SQL = """
SELECT typ.typname AS enum_name, enum.enumlabel AS enum_value
  FROM pg_catalog.pg_type typ
  JOIN pg_catalog.pg_namespace namespace ON namespace.oid = typ.typnamespace
  JOIN pg_catalog.pg_enum enum ON enum.enumtypid = typ.oid
 WHERE namespace.nspname = %s
 ORDER BY typ.typname, enum.enumsortorder
"""

_INDEXES_SQL = """
SELECT cls.relname AS table_name,
       index_cls.relname AS index_name,
       index.indisprimary AS is_primary,
       index.indisunique AS is_unique,
       index.indexprs IS NOT NULL AS has_expressions,
       index.indpred IS NOT NULL AS is_partial,
       key.ordinality AS key_position,
       attr.attname AS column_name
  FROM pg_catalog.pg_index index
  JOIN pg_catalog.pg_class cls ON cls.oid = index.indrelid
  JOIN pg_catalog.pg_namespace namespace ON namespace.oid = cls.relnamespace
  JOIN pg_catalog.pg_class index_cls ON index_cls.oid = index.indexrelid
  LEFT JOIN LATERAL unnest(index.indkey::smallint[]) WITH ORDINALITY AS key(attnum, ordinality)
    ON key.ordinality <= index.indnkeyatts
  LEFT JOIN pg_catalog.pg_attribute attr
    ON attr.attrelid = cls.oid AND attr.attnum = key.attnum
 WHERE namespace.nspname = %s
   AND index.indisunique
 ORDER BY cls.relname, index_cls.relname, key.ordinality
"""

_FOREIGN_KEYS_SQL = """
SELECT source.relname AS table_name,
       con.oid AS constraint_id,
       con.conname AS constraint_name,
       key.ordinality AS key_position,
       source_attr.attname AS column_name,
       target.relname AS referenced_table,
       target_attr.attname AS referenced_column,
       con.confupdtype AS on_update_code,
       con.confdeltype AS on_delete_code,
       con.condeferrable AS deferrable
  FROM pg_catalog.pg_constraint con
  JOIN pg_catalog.pg_class source ON source.oid = con.conrelid
  JOIN pg_catalog.pg_namespace namespace ON namespace.oid = source.relnamespace
  JOIN pg_catalog.pg_class target ON target.oid = con.confrelid
  JOIN LATERAL unnest(con.conkey, con.confkey) WITH ORDINALITY
       AS key(source_attnum, target_attnum, ordinality) ON TRUE
  JOIN pg_catalog.pg_attribute source_attr
    ON source_attr.attrelid = source.oid AND source_attr.attnum = key.source_attnum
  JOIN pg_catalog.pg_attribute target_attr
    ON target_attr.attrelid = target.oid AND target_attr.attnum = key.target_attnum
 WHERE namespace.nspname = %s
   AND con.contype = 'f'
 ORDER BY source.relname, con.oid, key.ordinality
"""

_ACTION = {
    "a": "NO ACTION",
    "r": "RESTRICT",
    "c": "CASCADE",
    "n": "SET NULL",
    "d": "SET DEFAULT",
}

_STRING_TYPES = {
    "bpchar",
    "char",
    "citext",
    "name",
    "text",
    "varchar",
}
_INTEGER_TYPES = {"int2", "int4", "int8", "oid"}
_NUMBER_TYPES = {"decimal", "float4", "float8", "money", "numeric"}
_TIMESTAMP_TYPES = {"time", "timetz", "timestamp", "timestamptz"}


def _execute_rows(
    connection: Any, statement: str, params: Sequence[Any] = ()
) -> list[dict[str, Any]]:
    cursor = connection.execute(statement, tuple(params))
    columns = [
        str(item.name if hasattr(item, "name") else item[0])
        for item in (cursor.description or ())
    ]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def _logical_scalar(udt_name: str) -> LogicalType | None:
    normalized = udt_name.lower()
    if normalized == "bool":
        return LogicalType.BOOLEAN
    if normalized in _INTEGER_TYPES:
        return LogicalType.INTEGER
    if normalized in _NUMBER_TYPES:
        return LogicalType.NUMBER
    if normalized in _STRING_TYPES:
        return LogicalType.STRING
    if normalized == "uuid":
        return LogicalType.UUID
    if normalized in _TIMESTAMP_TYPES:
        return LogicalType.TIMESTAMP
    if normalized == "date":
        return LogicalType.DATE
    if normalized in {"json", "jsonb"}:
        return LogicalType.JSON
    if normalized in {"bytea", "bit", "varbit"}:
        return LogicalType.BINARY
    return None


def _enum_values(connection: Any, schema: str) -> dict[str, tuple[str, ...]]:
    values: dict[str, list[str]] = defaultdict(list)
    for row in _execute_rows(connection, _ENUMS_SQL, (schema,)):
        values[str(row["enum_name"])].append(str(row["enum_value"]))
    return {name: tuple(items) for name, items in values.items()}


def _columns(
    connection: Any,
    schema: str,
    enums: dict[str, tuple[str, ...]],
    unsupported: list[UnsupportedSourceConstruct],
) -> dict[str, tuple[SourceColumn, ...]]:
    tables: dict[str, list[SourceColumn]] = defaultdict(list)
    for row in _execute_rows(connection, _COLUMNS_SQL, (schema,)):
        table = str(row["table_name"])
        column = str(row["column_name"])
        udt_name = str(row["udt_name"])
        type_kind = str(row["type_kind"])
        element_name = str(row["element_udt_name"] or "")
        enum_values: tuple[str, ...] = ()
        element_type: LogicalType | None = None
        if element_name:
            logical_type = LogicalType.ARRAY
            element_type = _logical_scalar(element_name)
            if element_type is None:
                if element_name in enums:
                    element_type = LogicalType.ENUM
                else:
                    element_type = LogicalType.STRING
                    unsupported.append(
                        UnsupportedSourceConstruct(
                            code="postgres_array_element_type_unsupported",
                            component="schema",
                            location=f"{table}.{column}",
                        )
                    )
        elif type_kind == "e":
            logical_type = LogicalType.ENUM
            enum_values = enums.get(udt_name, ())
        else:
            scalar = _logical_scalar(udt_name)
            if scalar is None:
                logical_type = LogicalType.STRING
                unsupported.append(
                    UnsupportedSourceConstruct(
                        code="postgres_native_type_unsupported",
                        component="schema",
                        location=f"{table}.{column}",
                    )
                )
            else:
                logical_type = scalar
        generated = bool(row["generated"])
        default = None if generated else row["default_expression"]
        tables[table].append(
            SourceColumn(
                name=column,
                logical_type=logical_type,
                native_type=str(row["native_type"]),
                nullable=bool(row["nullable"]),
                has_default=default is not None,
                default_expression=str(default) if default is not None else None,
                generated=generated,
                enum_values=enum_values,
                element_type=element_type,
            )
        )
    return {name: tuple(columns) for name, columns in tables.items()}


def _indexes(
    connection: Any,
    schema: str,
    unsupported: list[UnsupportedSourceConstruct],
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[tuple[str, ...], ...]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _execute_rows(connection, _INDEXES_SQL, (schema,)):
        grouped[(str(row["table_name"]), str(row["index_name"]))].append(row)
    primary: dict[str, tuple[str, ...]] = {}
    unique: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for (table, name), rows in sorted(grouped.items()):
        first = rows[0]
        ordered = sorted(rows, key=lambda item: int(item["key_position"] or 0))
        if (
            bool(first["has_expressions"])
            or bool(first["is_partial"])
            or any(row["column_name"] is None for row in ordered)
        ):
            unsupported.append(
                UnsupportedSourceConstruct(
                    code="postgres_unique_index_not_column_only",
                    component="schema",
                    location=f"{table}.{name}",
                )
            )
            continue
        key = tuple(str(row["column_name"]) for row in ordered)
        if bool(first["is_primary"]):
            primary[table] = key
        elif key:
            unique[table].add(key)
    return primary, {table: tuple(sorted(keys)) for table, keys in unique.items()}


def _foreign_keys(connection: Any, schema: str) -> dict[str, tuple[ForeignKey, ...]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in _execute_rows(connection, _FOREIGN_KEYS_SQL, (schema,)):
        grouped[(str(row["table_name"]), int(row["constraint_id"]))].append(row)
    tables: dict[str, list[ForeignKey]] = defaultdict(list)
    for (table, _identifier), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda item: int(item["key_position"]))
        first = ordered[0]
        tables[table].append(
            ForeignKey(
                columns=tuple(str(row["column_name"]) for row in ordered),
                referenced_table=str(first["referenced_table"]),
                referenced_columns=tuple(
                    str(row["referenced_column"]) for row in ordered
                ),
                on_update=_ACTION.get(str(first["on_update_code"])),
                on_delete=_ACTION.get(str(first["on_delete_code"])),
                deferrable=bool(first["deferrable"]),
            )
        )
    return {table: tuple(keys) for table, keys in tables.items()}


def _check_constraints(
    connection: Any, schema: str
) -> dict[str, tuple[CheckConstraint, ...]]:
    checks: dict[str, list[CheckConstraint]] = defaultdict(list)
    for row in _execute_rows(connection, _CHECKS_SQL, (schema,)):
        checks[str(row["table_name"])].append(
            CheckConstraint(
                name=str(row["constraint_name"]),
                expression=str(row["expression"]),
            )
        )
    return {
        table: tuple(sorted(items, key=lambda item: item.name))
        for table, items in checks.items()
    }


def inspect_postgres(
    connection: Any,
    *,
    source_digest: str,
    schema: str = "public",
    configuration_names: tuple[str, ...] = (),
    engine_version: str | None = None,
) -> SourceModel:
    """Inspect live PostgreSQL catalogues with bound schema parameters."""

    unsupported: list[UnsupportedSourceConstruct] = []
    enums = _enum_values(connection, schema)
    columns = _columns(connection, schema, enums, unsupported)
    primary, unique = _indexes(connection, schema, unsupported)
    foreign = _foreign_keys(connection, schema)
    checks = _check_constraints(connection, schema)
    tables = tuple(
        SourceTable(
            name=table,
            columns=table_columns,
            primary_key=primary.get(table, ()),
            unique_keys=unique.get(table, ()),
            foreign_keys=foreign.get(table, ()),
            check_constraints=checks.get(table, ()),
        )
        for table, table_columns in sorted(columns.items())
    )
    return SourceModel.create(
        source_digest=source_digest,
        engine="postgres",
        engine_version=engine_version,
        tables=tables,
        configuration_names=configuration_names,
        unsupported=tuple(unsupported),
    )


__all__ = ["inspect_postgres"]
