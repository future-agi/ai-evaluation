from __future__ import annotations

import os
from base64 import b64encode

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
    enum_values: tuple[str, ...] = (),
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
        enum_values=enum_values,
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


def test_compile_can_reconcile_rows_installed_by_source_seed() -> None:
    source = _source()

    compiled = compile_postgres(source, _world(source), reconcile_existing=True)

    rider, ride = compiled.operations
    assert rider.statement.endswith(
        'ON CONFLICT ("id") DO UPDATE SET '
        '"accessibility_needs" = EXCLUDED."accessibility_needs", '
        '"metadata" = EXCLUDED."metadata"'
    )
    assert ride.statement.endswith(
        'ON CONFLICT ("id") DO UPDATE SET "rider_id" = EXCLUDED."rider_id"'
    )
    assert [decision.code for decision in compiled.decisions].count(
        "existing_primary_key_reconciled"
    ) == 2


def test_reconcile_is_not_added_when_authored_row_omits_primary_key() -> None:
    source = SourceModel.create(
        source_digest=SOURCE_DIGEST,
        engine="postgres",
        tables=(
            SourceTable(
                name="riders",
                columns=(
                    _column(
                        "id",
                        LogicalType.INTEGER,
                        "bigint",
                        default="nextval('riders_id_seq')",
                    ),
                    _column(
                        "accessibility_needs",
                        LogicalType.ARRAY,
                        "text[]",
                        element_type=LogicalType.STRING,
                    ),
                ),
                primary_key=("id",),
            ),
        ),
    )
    world = WorldIR.create(
        source_model_fingerprint=source.fingerprint,
        tables=(
            WorldTable(
                source_name="riders",
                rows=(
                    WorldRow(
                        identity="generated-rider",
                        values={
                            "accessibility_needs": WorldValue.present(
                                LogicalType.ARRAY, []
                            )
                        },
                    ),
                ),
            ),
        ),
    )

    operation = compile_postgres(source, world, reconcile_existing=True).operations[0]

    assert "ON CONFLICT" not in operation.statement


def test_compile_reconciles_serial_sequence_after_explicit_seed_ids() -> None:
    source = SourceModel.create(
        source_digest=SOURCE_DIGEST,
        engine="postgres",
        tables=(
            SourceTable(
                name="tickets",
                columns=(
                    _column(
                        "ticket_id",
                        LogicalType.INTEGER,
                        "integer",
                        default="nextval('tickets_ticket_id_seq'::regclass)",
                    ),
                    _column("title", LogicalType.STRING, "text"),
                ),
                primary_key=("ticket_id",),
            ),
        ),
    )
    world = WorldIR.create(
        source_model_fingerprint=source.fingerprint,
        tables=(
            WorldTable(
                source_name="tickets",
                rows=(
                    WorldRow(
                        identity="ticket-10",
                        values={
                            "ticket_id": WorldValue.present(LogicalType.INTEGER, 10),
                            "title": WorldValue.present(LogicalType.STRING, "Seeded"),
                        },
                    ),
                ),
            ),
        ),
    )

    compiled = compile_postgres(source, world, reconcile_existing=True)

    assert [reset.model_dump() for reset in compiled.sequence_resets] == [
        {"schema_name": "public", "table": "tickets", "column": "ticket_id"}
    ]


def test_serial_sequence_advances_past_seeded_ids_in_real_postgres() -> None:
    dsn = os.environ.get("ALK_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set ALK_TEST_POSTGRES_DSN to run the real compiler test")
    psycopg = pytest.importorskip("psycopg")
    source = SourceModel.create(
        source_digest=SOURCE_DIGEST,
        engine="postgres",
        tables=(
            SourceTable(
                name="tickets",
                columns=(
                    _column(
                        "ticket_id",
                        LogicalType.INTEGER,
                        "integer",
                        default="nextval('compiler_sequence.tickets_ticket_id_seq'::regclass)",
                    ),
                    _column("title", LogicalType.STRING, "text"),
                ),
                primary_key=("ticket_id",),
            ),
        ),
    )
    world = WorldIR.create(
        source_model_fingerprint=source.fingerprint,
        tables=(
            WorldTable(
                source_name="tickets",
                rows=(
                    WorldRow(
                        identity="ticket-10",
                        values={
                            "ticket_id": WorldValue.present(LogicalType.INTEGER, 10),
                            "title": WorldValue.present(LogicalType.STRING, "Seeded"),
                        },
                    ),
                ),
            ),
        ),
    )
    compiled = compile_postgres(source, world, schema="compiler_sequence")

    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute("DROP SCHEMA IF EXISTS compiler_sequence CASCADE")
        connection.execute("CREATE SCHEMA compiler_sequence")
        connection.execute(
            "CREATE TABLE compiler_sequence.tickets ("
            "ticket_id serial PRIMARY KEY, title text NOT NULL)"
        )
        apply_postgres(connection, compiled)
        generated_id = connection.execute(
            "INSERT INTO compiler_sequence.tickets (title) VALUES ('Generated') "
            "RETURNING ticket_id"
        ).fetchone()[0]

    assert generated_id == 11


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


def test_all_logical_types_round_trip_through_real_postgres() -> None:
    """Exercise every portable World IR scalar plus nested arrays on the real backend."""

    dsn = os.environ.get("ALK_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set ALK_TEST_POSTGRES_DSN to run the real compiler test")
    psycopg = pytest.importorskip("psycopg")
    columns = (
        _column("id", LogicalType.INTEGER, "bigint"),
        _column("flag", LogicalType.BOOLEAN, "boolean"),
        _column("ratio", LogicalType.NUMBER, "double precision"),
        _column("label", LogicalType.STRING, "text"),
        _column("uid", LogicalType.UUID, "uuid"),
        _column("seen_at", LogicalType.TIMESTAMP, "timestamptz"),
        _column("day", LogicalType.DATE, "date"),
        _column(
            "status",
            LogicalType.ENUM,
            "compiler_matrix.status_enum",
            enum_values=("active", "disabled"),
        ),
        _column(
            "matrix",
            LogicalType.ARRAY,
            "integer[]",
            element_type=LogicalType.INTEGER,
        ),
        _column("payload", LogicalType.JSON, "jsonb"),
        _column("payload_null", LogicalType.JSON, "jsonb"),
        _column("raw", LogicalType.BINARY, "bytea"),
        _column("optional", LogicalType.STRING, "text", nullable=True),
        _column(
            "defaulted",
            LogicalType.STRING,
            "text",
            default="'source-default'::text",
        ),
        _column("generated", LogicalType.STRING, "text", generated=True),
    )
    source = SourceModel.create(
        source_digest=SOURCE_DIGEST,
        engine="postgres",
        tables=(SourceTable(name="type_matrix", columns=columns, primary_key=("id",)),),
    )
    binary = b"\x00portable-world\xff"
    values = {
        "id": WorldValue.present(LogicalType.INTEGER, 7),
        "flag": WorldValue.present(LogicalType.BOOLEAN, True),
        "ratio": WorldValue.present(LogicalType.NUMBER, 1.25),
        "label": WorldValue.present(LogicalType.STRING, "hello 世界 🌍"),
        "uid": WorldValue.present(
            LogicalType.UUID, "12345678-1234-5678-1234-567812345678"
        ),
        "seen_at": WorldValue.present(
            LogicalType.TIMESTAMP, "2026-09-14T10:30:00+00:00"
        ),
        "day": WorldValue.present(LogicalType.DATE, "2026-09-14"),
        "status": WorldValue.present(LogicalType.ENUM, "active"),
        "matrix": WorldValue.present(LogicalType.ARRAY, [[1, 2], [3, 4]]),
        "payload": WorldValue.present(LogicalType.JSON, {"nested": [True, None, "é"]}),
        "payload_null": WorldValue.present(LogicalType.JSON, None),
        "raw": WorldValue.present(
            LogicalType.BINARY, b64encode(binary).decode("ascii")
        ),
        "optional": WorldValue.null(LogicalType.STRING),
        "defaulted": WorldValue.absent(),
        "generated": WorldValue.absent(),
    }
    world = WorldIR.create(
        source_model_fingerprint=source.fingerprint,
        tables=(
            WorldTable(
                source_name="type_matrix",
                rows=(WorldRow(identity="matrix-1", values=values),),
            ),
        ),
    )
    compiled = compile_postgres(source, world, schema="compiler_matrix")
    assert compiled == compile_postgres(source, world, schema="compiler_matrix")

    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute("DROP SCHEMA IF EXISTS compiler_matrix CASCADE")
        connection.execute("CREATE SCHEMA compiler_matrix")
        connection.execute(
            """
            CREATE TYPE compiler_matrix.status_enum AS ENUM ('active', 'disabled');
            CREATE TABLE compiler_matrix.type_matrix (
                id bigint PRIMARY KEY,
                flag boolean NOT NULL,
                ratio double precision NOT NULL,
                label text NOT NULL,
                uid uuid NOT NULL,
                seen_at timestamptz NOT NULL,
                day date NOT NULL,
                status compiler_matrix.status_enum NOT NULL,
                matrix integer[] NOT NULL,
                payload jsonb NOT NULL,
                payload_null jsonb NOT NULL,
                raw bytea NOT NULL,
                optional text,
                defaulted text NOT NULL DEFAULT 'source-default',
                generated text GENERATED ALWAYS AS (id::text) STORED
            )
            """
        )
        apply_postgres(connection, compiled)
        row = connection.execute(
            """
            SELECT id, flag, ratio, label, uid::text, seen_at::text, day::text,
                   status::text, matrix, payload, payload_null IS NULL,
                   jsonb_typeof(payload_null), raw, optional, defaulted, generated
            FROM compiler_matrix.type_matrix
            """
        ).fetchone()
        connection.execute("DROP SCHEMA compiler_matrix CASCADE")

    assert row == (
        7,
        True,
        1.25,
        "hello 世界 🌍",
        "12345678-1234-5678-1234-567812345678",
        "2026-09-14 10:30:00+00",
        "2026-09-14",
        "active",
        [[1, 2], [3, 4]],
        {"nested": [True, None, "é"]},
        False,
        "null",
        binary,
        None,
        "source-default",
        "7",
    )
