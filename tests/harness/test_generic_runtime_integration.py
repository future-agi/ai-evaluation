from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from fi.alk.harness.process_runtime import (
    EngineCredentials,
    GenericWorldSeedError,
    apply_postgres_sqlite_world,
)


def test_real_postgres_catalogue_drives_legacy_world_import(tmp_path: Path) -> None:
    dsn = os.environ.get("ALK_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set ALK_TEST_POSTGRES_DSN to run the generic runtime test")
    psycopg = pytest.importorskip("psycopg")
    parameters = psycopg.conninfo.conninfo_to_dict(dsn)
    world = tmp_path / "world.sqlite"
    with sqlite3.connect(world) as connection:
        connection.execute(
            "CREATE TABLE generic_users ("
            "id TEXT, accessibility_needs TEXT, profile TEXT, active INTEGER, created_at TEXT)"
        )
        connection.execute(
            "INSERT INTO generic_users VALUES (?, ?, ?, ?, ?)",
            (
                "user-1",
                '["wheelchair"]',
                '{"tier":"gold"}',
                1,
                None,
            ),
        )

    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute("DROP TABLE IF EXISTS generic_users")
        connection.execute(
            "CREATE TABLE generic_users ("
            "id text PRIMARY KEY, "
            "accessibility_needs text[] NOT NULL, "
            "profile jsonb NOT NULL, "
            "active boolean NOT NULL, "
            "created_at timestamptz NOT NULL DEFAULT now())"
        )

    apply_postgres_sqlite_world(
        world,
        port=int(parameters.get("port", 5432)),
        dbname=str(parameters["dbname"]),
        credentials=EngineCredentials(
            username=str(parameters["user"]), password=str(parameters["password"])
        ),
        source_digest="sha256:" + "a" * 64,
    )

    with psycopg.connect(dsn, autocommit=True) as connection:
        row = connection.execute(
            "SELECT id, accessibility_needs, profile, active, created_at IS NOT NULL "
            "FROM generic_users"
        ).fetchone()
        connection.execute("DROP TABLE generic_users")

    assert row == ("user-1", ["wheelchair"], {"tier": "gold"}, True, True)


def test_real_postgres_constraint_failure_is_typed_and_value_free(
    tmp_path: Path,
) -> None:
    dsn = os.environ.get("ALK_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set ALK_TEST_POSTGRES_DSN to run the generic runtime test")
    psycopg = pytest.importorskip("psycopg")
    parameters = psycopg.conninfo.conninfo_to_dict(dsn)
    world = tmp_path / "world.sqlite"
    with sqlite3.connect(world) as connection:
        connection.execute("CREATE TABLE constrained_users (id TEXT, status TEXT)")
        connection.execute(
            "INSERT INTO constrained_users VALUES (?, ?)",
            ("user-1", "private-invalid-status"),
        )

    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute("DROP TABLE IF EXISTS constrained_users")
        connection.execute(
            "CREATE TABLE constrained_users ("
            "id text PRIMARY KEY, "
            "status text NOT NULL CHECK (status IN ('active', 'suspended')))"
        )

    with pytest.raises(GenericWorldSeedError) as raised:
        apply_postgres_sqlite_world(
            world,
            port=int(parameters.get("port", 5432)),
            dbname=str(parameters["dbname"]),
            credentials=EngineCredentials(
                username=str(parameters["user"]), password=str(parameters["password"])
            ),
            source_digest="sha256:" + "a" * 64,
        )

    diagnostic = raised.value.diagnostics[0]
    assert diagnostic.code == "constraint_value_invalid"
    assert diagnostic.location is not None
    assert diagnostic.location.table == "constrained_users"
    assert diagnostic.location.constraint == "constrained_users_status_check"
    assert "private-invalid-status" not in diagnostic.model_dump_json()
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute("DROP TABLE constrained_users")
