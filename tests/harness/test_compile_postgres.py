from __future__ import annotations

import os

import pytest

from fi.alk.harness.compile.postgres import apply_postgres, compile_postgres
from fi.alk.harness.source_model import (
    ForeignKey,
    LogicalType,
    SourceColumn,
    SourceModel,
    SourceTable,
)
from fi.alk.harness.world_ir import WorldIR, WorldRow, WorldTable, WorldValue

SOURCE_DIGEST = "sha256:" + ("e" * 64)


def _column(
    name: str,
    logical_type: LogicalType,
    native_type: str,
    *,
    nullable: bool = False,
    default: str | None = None,
    generated: bool = False,
    element_type: LogicalType | None = None,
) -> SourceColumn:
    return SourceColumn(
        name=name,
        logical_type=logical_type,
        native_type=native_type,
        nullable=nullable,
        has_default=default is not None,
        default_expression=default,
        generated=generated,
        element_type=element_type,
    )


def _source() -> SourceModel:
    riders = SourceTable(
        name="riders",
        columns=(
            _column("id", LogicalType.INTEGER, "bigint"),
            _column(
                "accessibility_needs",
                LogicalType.ARRAY,
                "text[]",
                element_type=LogicalType.STRING,
            ),
            _column("metadata", LogicalType.JSON, "jsonb", nullable=True),
            _column(
                "created_at", LogicalType.TIMESTAMP, "timestamptz", default="now()"
            ),
            _column("label", LogicalType.STRING, "text", generated=True),
        ),
        primary_key=("id",),
    )
    rides = SourceTable(
        name="rides",
        columns=(
            _column("id", LogicalType.INTEGER, "bigint"),
            _column("rider_id", LogicalType.INTEGER, "bigint"),
        ),
        primary_key=("id",),
        foreign_keys=(
            ForeignKey(
                columns=("rider_id",),
                referenced_table="riders",
                referenced_columns=("id",),
            ),
        ),
    )
    return SourceModel.create(
        source_digest=SOURCE_DIGEST, engine="postgres", tables=(rides, riders)
    )


def _world(source: SourceModel) -> WorldIR:
    return WorldIR.create(
        source_model_fingerprint=source.fingerprint,
        tables=(
            WorldTable(
                source_name="rides",
                rows=(
                    WorldRow(
                        identity="ride-1",
                        values={
                            "id": WorldValue.present(LogicalType.INTEGER, 5),
                            "rider_id": WorldValue.present(LogicalType.INTEGER, 1),
                        },
                    ),
                ),
            ),
            WorldTable(
                source_name="riders",
                rows=(
                    WorldRow(
                        identity="rider-1",
                        values={
                            "id": WorldValue.present(LogicalType.INTEGER, 1),
                            "accessibility_needs": WorldValue.present(
                                LogicalType.ARRAY, ["wheelchair"]
                            ),
                            "metadata": WorldValue.present(
                                LogicalType.JSON, {"tier": "gold"}
                            ),
                            "created_at": WorldValue.absent(),
                            "label": WorldValue.absent(),
                        },
                    ),
                ),
            ),
        ),
    )


def test_compile_uses_bound_native_values_and_source_defaults() -> None:
    source = _source()
    compiled = compile_postgres(source, _world(source))

    assert [operation.table for operation in compiled.operations] == ["riders", "rides"]
    rider = compiled.operations[0]
    assert rider.columns == ("id", "accessibility_needs", "metadata")
    assert rider.params == (1, ["wheelchair"], '{"tier":"gold"}')
    assert "created_at" not in rider.statement
    assert "label" not in rider.statement
    assert "wheelchair" not in rider.statement
    assert rider.statement.count("%s") == 3
    assert [(decision.code, decision.column) for decision in compiled.decisions] == [
        ("source_default_applied", "created_at"),
        ("generated_column_omitted", "label"),
    ]


def test_compile_is_deterministic() -> None:
    source = _source()
    world = _world(source)

    assert compile_postgres(source, world) == compile_postgres(source, world)


def test_compile_quotes_source_identifiers() -> None:
    source = SourceModel.create(
        source_digest=SOURCE_DIGEST,
        engine="postgres",
        tables=(
            SourceTable(
                name='odd"table',
                columns=(_column('odd"column', LogicalType.STRING, "text"),),
            ),
        ),
    )
    world = WorldIR.create(
        source_model_fingerprint=source.fingerprint,
        tables=(
            WorldTable(
                source_name='odd"table',
                rows=(
                    WorldRow(
                        identity="row-1",
                        values={
                            'odd"column': WorldValue.present(LogicalType.STRING, "safe")
                        },
                    ),
                ),
            ),
        ),
    )

    operation = compile_postgres(source, world, schema='custom"schema').operations[0]
    assert operation.statement == (
        'INSERT INTO "custom""schema"."odd""table" ("odd""column") VALUES (%s)'
    )
    assert operation.params == ("safe",)


def test_compiled_operations_apply_to_real_postgres() -> None:
    dsn = os.environ.get("ALK_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set ALK_TEST_POSTGRES_DSN to run the real compiler test")
    psycopg = pytest.importorskip("psycopg")
    source = _source()
    compiled = compile_postgres(source, _world(source), schema="compiler_test")
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute("DROP SCHEMA IF EXISTS compiler_test CASCADE")
        connection.execute("CREATE SCHEMA compiler_test")
        connection.execute(
            """
            CREATE TABLE compiler_test.riders (
                id bigint PRIMARY KEY,
                accessibility_needs text[] NOT NULL,
                metadata jsonb,
                created_at timestamptz NOT NULL DEFAULT now(),
                label text GENERATED ALWAYS AS (id::text) STORED
            );
            CREATE TABLE compiler_test.rides (
                id bigint PRIMARY KEY,
                rider_id bigint NOT NULL REFERENCES compiler_test.riders(id)
            );
            """
        )
        apply_postgres(connection, compiled)
        rider = connection.execute(
            "SELECT id, accessibility_needs, metadata, created_at IS NOT NULL, label "
            "FROM compiler_test.riders"
        ).fetchone()
        ride = connection.execute(
            "SELECT id, rider_id FROM compiler_test.rides"
        ).fetchone()

    assert rider == (1, ["wheelchair"], {"tier": "gold"}, True, "1")
    assert ride == (5, 1)
