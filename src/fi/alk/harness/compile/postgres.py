"""Pure World IR to parameterized PostgreSQL insert compilation."""

from __future__ import annotations

import base64
import json
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..source_model import LogicalType, SourceColumn, SourceModel
from ..world_ir import ValueState, WorldIR, validate_world_ir

POSTGRES_COMPILER_VERSION = "futureagi.postgres-compiler.v1"


class PostgresCompileError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class CompileDecision(BaseModel):
    """Value-free explanation safe to persist in a certificate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    table: str
    row_identity: str
    column: str | None = None


class PostgresInsert(BaseModel):
    """Runtime-only bound operation; params must never enter diagnostics or certificates."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    table: str
    row_identity: str
    columns: tuple[str, ...]
    statement: str
    params: tuple[Any, ...] = Field(repr=False)


class PostgresCompileResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_schema_hash: str
    world_ir_hash: str
    compiler_version: str = POSTGRES_COMPILER_VERSION
    operations: tuple[PostgresInsert, ...]
    decisions: tuple[CompileDecision, ...]
    warnings: tuple[str, ...] = ()


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _table_order(source: SourceModel, world: WorldIR) -> tuple[str, ...]:
    included = {table.source_name for table in world.tables if table.rows}
    dependencies: dict[str, set[str]] = {table: set() for table in included}
    dependents: dict[str, set[str]] = defaultdict(set)
    for table in source.tables:
        if table.name not in included:
            continue
        for key in table.foreign_keys:
            target = key.referenced_table
            if target not in included:
                continue
            dependencies[table.name].add(target)
            dependents[target].add(table.name)
    ready = sorted(table for table, required in dependencies.items() if not required)
    ordered: list[str] = []
    while ready:
        table = ready.pop(0)
        ordered.append(table)
        for dependent in sorted(dependents.get(table, ())):
            dependencies[dependent].discard(table)
            if (
                not dependencies[dependent]
                and dependent not in ordered
                and dependent not in ready
            ):
                ready.append(dependent)
                ready.sort()
    if len(ordered) != len(included):
        cycle = sorted(included - set(ordered))
        raise PostgresCompileError(
            "seed_order_invalid",
            "foreign-key cycle requires a source-supported deferred strategy: "
            + ", ".join(cycle),
        )
    return tuple(ordered)


def _parameter(column: SourceColumn, value: Any) -> Any:
    if value is None:
        return None
    if column.logical_type is LogicalType.JSON:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
    if column.logical_type is LogicalType.BINARY:
        assert isinstance(value, str)
        return base64.b64decode(value, validate=True)
    return value


def compile_postgres(
    source: SourceModel,
    world: WorldIR,
    *,
    schema: str = "public",
) -> PostgresCompileResult:
    """Compile validated semantic rows into deterministic parameterized inserts."""

    if source.engine != "postgres":
        raise PostgresCompileError(
            "schema_type_mismatch",
            f"expected postgres source model, got {source.engine}",
        )
    validate_world_ir(world, source)
    source_tables = {table.name: table for table in source.tables}
    world_tables = {table.source_name: table for table in world.tables}
    operations: list[PostgresInsert] = []
    decisions: list[CompileDecision] = []
    for table_name in _table_order(source, world):
        source_table = source_tables[table_name]
        world_table = world_tables[table_name]
        for row in world_table.rows:
            columns: list[str] = []
            params: list[Any] = []
            for column in source_table.columns:
                authored = row.values.get(column.name)
                if authored is None or authored.state is ValueState.ABSENT:
                    if column.has_default:
                        decisions.append(
                            CompileDecision(
                                code="source_default_applied",
                                table=table_name,
                                row_identity=row.identity,
                                column=column.name,
                            )
                        )
                    elif column.generated:
                        decisions.append(
                            CompileDecision(
                                code="generated_column_omitted",
                                table=table_name,
                                row_identity=row.identity,
                                column=column.name,
                            )
                        )
                    continue
                columns.append(column.name)
                params.append(_parameter(column, authored.value))
            qualified = f"{_identifier(schema)}.{_identifier(table_name)}"
            if columns:
                names = ", ".join(_identifier(column) for column in columns)
                placeholders = ", ".join("%s" for _ in columns)
                statement = f"INSERT INTO {qualified} ({names}) VALUES ({placeholders})"
            else:
                statement = f"INSERT INTO {qualified} DEFAULT VALUES"
            operations.append(
                PostgresInsert(
                    table=table_name,
                    row_identity=row.identity,
                    columns=tuple(columns),
                    statement=statement,
                    params=tuple(params),
                )
            )
    return PostgresCompileResult(
        source_schema_hash=source.fingerprint,
        world_ir_hash=world.fingerprint,
        operations=tuple(operations),
        decisions=tuple(decisions),
    )


def apply_postgres(connection: Any, compiled: PostgresCompileResult) -> None:
    """Apply a compiled program atomically using driver-bound parameters."""

    with connection.transaction():
        for operation in compiled.operations:
            connection.execute(operation.statement, operation.params or None)


__all__ = [
    "POSTGRES_COMPILER_VERSION",
    "CompileDecision",
    "PostgresCompileError",
    "PostgresCompileResult",
    "PostgresInsert",
    "apply_postgres",
    "compile_postgres",
]
