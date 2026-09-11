from __future__ import annotations

import sqlite3

import pytest
from pydantic import ValidationError

from fi.alk.harness.source_model import (
    LogicalType,
    SourceEvidence,
    SourceModel,
)
from fi.alk.harness.source_schema.sqlite import inspect_sqlite

SOURCE_DIGEST = "sha256:" + ("a" * 64)


def _database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE accounts (
            tenant_id UUID NOT NULL,
            account_id INTEGER NOT NULL,
            active BOOLEAN NOT NULL DEFAULT 1,
            profile JSON,
            balance NUMERIC(12, 2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            external_key TEXT UNIQUE,
            label TEXT GENERATED ALWAYS AS (tenant_id || '-' || account_id) STORED,
            PRIMARY KEY (tenant_id, account_id)
        );
        CREATE TABLE invoices (
            id INTEGER PRIMARY KEY,
            tenant_id UUID NOT NULL,
            account_id INTEGER NOT NULL,
            issued_on DATE,
            payload BLOB,
            UNIQUE (tenant_id, id),
            FOREIGN KEY (tenant_id, account_id)
                REFERENCES accounts (tenant_id, account_id)
                ON UPDATE CASCADE ON DELETE RESTRICT
        );
        CREATE TABLE untyped (value, tags TEXT[], opaque ARRAY);
        """
    )
    return connection


def test_sqlite_inspection_preserves_schema_semantics() -> None:
    model = inspect_sqlite(
        _database(),
        source_digest=SOURCE_DIGEST,
        configuration_names=("DATABASE_URL", "DATABASE_URL", "FEATURE_FLAG"),
    )

    assert [table.name for table in model.tables] == [
        "accounts",
        "invoices",
        "untyped",
    ]
    accounts = model.tables[0]
    columns = {column.name: column for column in accounts.columns}
    assert accounts.primary_key == ("tenant_id", "account_id")
    assert ("external_key",) in accounts.unique_keys
    assert columns["tenant_id"].logical_type is LogicalType.UUID
    assert columns["active"].logical_type is LogicalType.BOOLEAN
    assert columns["active"].default_expression == "1"
    assert columns["profile"].logical_type is LogicalType.JSON
    assert columns["balance"].logical_type is LogicalType.NUMBER
    assert columns["created_at"].logical_type is LogicalType.TIMESTAMP
    assert columns["label"].generated is True
    assert model.configuration_names == ("DATABASE_URL", "FEATURE_FLAG")

    invoices = model.tables[1]
    assert invoices.unique_keys == (("tenant_id", "id"),)
    assert invoices.foreign_keys[0].columns == ("tenant_id", "account_id")
    assert invoices.foreign_keys[0].referenced_table == "accounts"
    assert invoices.foreign_keys[0].referenced_columns == ("tenant_id", "account_id")
    assert invoices.foreign_keys[0].on_update == "CASCADE"
    assert invoices.foreign_keys[0].on_delete == "RESTRICT"
    assert model.tables[2].columns[0].native_type == ""
    assert model.tables[2].columns[0].logical_type is LogicalType.STRING
    assert model.tables[2].columns[1].logical_type is LogicalType.ARRAY
    assert model.tables[2].columns[1].element_type is LogicalType.STRING
    assert model.tables[2].columns[2].logical_type is LogicalType.ARRAY
    assert model.tables[2].columns[2].element_type is LogicalType.STRING


def test_repeated_sqlite_inspection_has_a_stable_fingerprint() -> None:
    first = inspect_sqlite(_database(), source_digest=SOURCE_DIGEST)
    second = inspect_sqlite(_database(), source_digest=SOURCE_DIGEST)

    assert first == second
    assert first.fingerprint.startswith("sha256:")


def test_source_model_fingerprint_covers_source_identity() -> None:
    first = inspect_sqlite(_database(), source_digest=SOURCE_DIGEST)
    second = inspect_sqlite(_database(), source_digest="sha256:" + ("b" * 64))

    assert first.fingerprint != second.fingerprint


def test_source_model_rejects_tampered_fingerprint() -> None:
    valid = inspect_sqlite(_database(), source_digest=SOURCE_DIGEST)
    payload = valid.model_dump()
    payload["fingerprint"] = "sha256:" + ("0" * 64)

    with pytest.raises(ValidationError, match="source_model_fingerprint_mismatch"):
        SourceModel.model_validate(payload)


def test_source_evidence_is_digest_only_and_portable() -> None:
    evidence = SourceEvidence(
        path="migrations/001.sql", digest=SOURCE_DIGEST, kind="migration"
    )
    assert set(evidence.model_dump()) == {"path", "digest", "kind"}

    with pytest.raises(ValidationError, match="source_evidence_path_must_be_relative"):
        SourceEvidence(path="/tmp/001.sql", digest=SOURCE_DIGEST, kind="migration")


def test_sqlite_expression_index_is_explicitly_unsupported() -> None:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE users (email TEXT);
        CREATE UNIQUE INDEX users_email_ci ON users (lower(email));
        """
    )

    model = inspect_sqlite(connection, source_digest=SOURCE_DIGEST)

    assert model.tables[0].unique_keys == ()
    assert [item.code for item in model.unsupported] == [
        "sqlite_unique_index_not_column_only"
    ]
