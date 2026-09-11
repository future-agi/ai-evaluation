from __future__ import annotations

import os
from typing import Any

import pytest

from fi.alk.harness.source_model import LogicalType
from fi.alk.harness.source_schema.postgres import inspect_postgres

SOURCE_DIGEST = "sha256:" + ("c" * 64)


class Cursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.description = [(name,) for name in (rows[0] if rows else ())]
        self._rows = [tuple(row.values()) for row in rows]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


class CatalogueConnection:
    def execute(self, statement: str, params: tuple[Any, ...]) -> Cursor:
        assert params == ("public",)
        if "con.contype = 'c'" in statement:
            return Cursor(
                [
                    {
                        "table_name": "rides",
                        "constraint_name": "rides_status_check",
                        "expression": "CHECK ((status <> 'accepted'))",
                    }
                ]
            )
        if "pg_catalog.pg_enum" in statement:
            return Cursor(
                [
                    {"enum_name": "ride_status", "enum_value": "requested"},
                    {"enum_name": "ride_status", "enum_value": "accepted"},
                ]
            )
        if "pg_catalog.pg_index index" in statement:
            return Cursor(
                [
                    {
                        "table_name": "rides",
                        "index_name": "rides_pkey",
                        "is_primary": True,
                        "is_unique": True,
                        "has_expressions": False,
                        "is_partial": False,
                        "key_position": 1,
                        "column_name": "id",
                    },
                    {
                        "table_name": "rides",
                        "index_name": "rides_status_created_key",
                        "is_primary": False,
                        "is_unique": True,
                        "has_expressions": False,
                        "is_partial": False,
                        "key_position": 1,
                        "column_name": "status",
                    },
                    {
                        "table_name": "rides",
                        "index_name": "rides_status_created_key",
                        "is_primary": False,
                        "is_unique": True,
                        "has_expressions": False,
                        "is_partial": False,
                        "key_position": 2,
                        "column_name": "created_at",
                    },
                ]
            )
        if "pg_catalog.pg_attribute attr" in statement:
            base = {
                "nullable": False,
                "has_default": False,
                "default_expression": None,
                "generated": False,
                "type_kind": "b",
                "element_udt_name": None,
            }
            return Cursor(
                [
                    {
                        **base,
                        "table_name": "rides",
                        "column_name": "id",
                        "ordinal_position": 1,
                        "native_type": "uuid",
                        "udt_name": "uuid",
                    },
                    {
                        **base,
                        "table_name": "rides",
                        "column_name": "accessibility_needs",
                        "ordinal_position": 2,
                        "native_type": "text[]",
                        "udt_name": "_text",
                        "element_udt_name": "text",
                    },
                    {
                        **base,
                        "table_name": "rides",
                        "column_name": "status",
                        "ordinal_position": 3,
                        "native_type": "ride_status",
                        "type_kind": "e",
                        "udt_name": "ride_status",
                    },
                    {
                        **base,
                        "table_name": "rides",
                        "column_name": "metadata",
                        "ordinal_position": 4,
                        "native_type": "jsonb",
                        "udt_name": "jsonb",
                        "nullable": True,
                    },
                    {
                        **base,
                        "table_name": "rides",
                        "column_name": "created_at",
                        "ordinal_position": 5,
                        "native_type": "timestamp with time zone",
                        "udt_name": "timestamptz",
                        "has_default": True,
                        "default_expression": "now()",
                    },
                    {
                        **base,
                        "table_name": "events",
                        "column_name": "ride_id",
                        "ordinal_position": 1,
                        "native_type": "uuid",
                        "udt_name": "uuid",
                    },
                ]
            )
        if "pg_catalog.pg_constraint con" in statement:
            return Cursor(
                [
                    {
                        "table_name": "events",
                        "constraint_id": 42,
                        "constraint_name": "events_ride_id_fkey",
                        "key_position": 1,
                        "column_name": "ride_id",
                        "referenced_table": "rides",
                        "referenced_column": "id",
                        "on_update_code": "a",
                        "on_delete_code": "c",
                        "deferrable": True,
                    }
                ]
            )
        raise AssertionError("unexpected catalogue query")


def test_postgres_inspection_preserves_native_types_and_relations() -> None:
    model = inspect_postgres(
        CatalogueConnection(),
        source_digest=SOURCE_DIGEST,
        configuration_names=("DATABASE_URL",),
        engine_version="16.4",
    )

    assert [table.name for table in model.tables] == ["events", "rides"]
    events, rides = model.tables
    columns = {column.name: column for column in rides.columns}
    assert rides.primary_key == ("id",)
    assert rides.unique_keys == (("status", "created_at"),)
    assert columns["id"].logical_type is LogicalType.UUID
    assert columns["accessibility_needs"].logical_type is LogicalType.ARRAY
    assert columns["accessibility_needs"].element_type is LogicalType.STRING
    assert columns["accessibility_needs"].native_type == "text[]"
    assert columns["status"].logical_type is LogicalType.ENUM
    assert columns["status"].enum_values == ("requested", "accepted")
    assert columns["metadata"].logical_type is LogicalType.JSON
    assert columns["created_at"].logical_type is LogicalType.TIMESTAMP
    assert columns["created_at"].default_expression == "now()"
    assert events.foreign_keys[0].referenced_table == "rides"
    assert events.foreign_keys[0].on_delete == "CASCADE"
    assert events.foreign_keys[0].deferrable is True
    assert rides.check_constraints[0].name == "rides_status_check"
    assert "status" in rides.check_constraints[0].expression
    assert model.unsupported == ()


def test_postgres_inspection_is_stable() -> None:
    first = inspect_postgres(CatalogueConnection(), source_digest=SOURCE_DIGEST)
    second = inspect_postgres(CatalogueConnection(), source_digest=SOURCE_DIGEST)

    assert first == second


def test_postgres_inspection_against_real_catalogues() -> None:
    dsn = os.environ.get("ALK_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set ALK_TEST_POSTGRES_DSN to run the real-catalogue test")
    psycopg = pytest.importorskip("psycopg")
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute("DROP TABLE IF EXISTS rides, riders CASCADE")
        connection.execute("DROP TYPE IF EXISTS ride_status CASCADE")
        connection.execute(
            """
            CREATE TYPE ride_status AS ENUM ('requested', 'accepted');
            CREATE TABLE riders (
                tenant_id uuid NOT NULL,
                rider_id bigint NOT NULL,
                accessibility_needs text[] NOT NULL DEFAULT '{}',
                status ride_status NOT NULL DEFAULT 'requested',
                metadata jsonb,
                created_at timestamptz NOT NULL DEFAULT now(),
                label text GENERATED ALWAYS AS (tenant_id::text || ':' || rider_id) STORED,
                PRIMARY KEY (tenant_id, rider_id),
                UNIQUE (rider_id, status)
            );
            CREATE TABLE rides (
                id uuid PRIMARY KEY,
                tenant_id uuid NOT NULL,
                rider_id bigint NOT NULL,
                CHECK (rider_id > 0),
                FOREIGN KEY (tenant_id, rider_id)
                    REFERENCES riders (tenant_id, rider_id)
                    ON UPDATE CASCADE ON DELETE RESTRICT DEFERRABLE
            );
            """
        )
        model = inspect_postgres(connection, source_digest=SOURCE_DIGEST)

    riders = next(table for table in model.tables if table.name == "riders")
    columns = {column.name: column for column in riders.columns}
    assert riders.primary_key == ("tenant_id", "rider_id")
    assert ("rider_id", "status") in riders.unique_keys
    assert columns["accessibility_needs"].native_type == "text[]"
    assert columns["accessibility_needs"].logical_type is LogicalType.ARRAY
    assert columns["status"].enum_values == ("requested", "accepted")
    assert columns["metadata"].logical_type is LogicalType.JSON
    assert columns["created_at"].default_expression == "now()"
    assert columns["label"].generated is True
    rides = next(table for table in model.tables if table.name == "rides")
    assert rides.foreign_keys[0].columns == ("tenant_id", "rider_id")
    assert rides.foreign_keys[0].on_update == "CASCADE"
    assert rides.foreign_keys[0].on_delete == "RESTRICT"
    assert len(rides.check_constraints) == 1
    assert "rider_id" in rides.check_constraints[0].expression
