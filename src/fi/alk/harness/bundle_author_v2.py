"""Deterministic producer for hosted ``EnvironmentBundleV2`` directories.

This module is deliberately a compiler, not a second execution engine.  It converts the
packaging already present in a submitted repository into the process vocabulary consumed by the
Daytona guest, adds the harness-owned world database, adopts frozen scenario artifacts, seals the
result, and runs the guest's exact preflight before publishing it.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import re
import shlex
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from .bundle import CapabilityProtocol
from .catalogue import CATALOGUE
from .bundle_v2 import (
    BUNDLE_V2_MANIFEST,
    BUNDLE_V2_SCHEMA_VERSION,
    BaselineStrategy,
    BundleFileV2,
    BundleProvenanceV2,
    BundleRuntimeV2,
    CapabilityV2,
    EnvironmentBundleV2,
    EvidenceSeam,
    ManagedEngine,
    ManagedProcess,
    ProcessUser,
    ReadinessProbeV2,
    RuntimeKindV2,
    SecretPurpose,
    Seed,
    Sentinel,
    SourceProcess,
    StartedCheck,
    StoreBaseline,
    StoreEntry,
    compute_inputs_digest,
    load_bundle_v2,
    seal_bundle_v2,
)
from .contract import ToolEntry
from .credentials import discover_credentials
from .job import HarnessJob
from .job import ProviderExecutionMode, SourceKind
from .process_preflight import preflight_bundle
from .provision import source_fingerprint
from .provider_lifecycle import ProviderRepositoryManifest, load_provider_manifest
from .provider_import import ProviderImportSpec
from .world.tools import _binding


class BundleAuthorError(RuntimeError):
    """A source cannot be compiled into an honest hosted process bundle."""


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnvironmentPlanV2:
    packaging: str
    control_service: str | None
    processes: tuple[ManagedProcess | SourceProcess, ...]
    capabilities: dict[str, CapabilityV2]
    readiness: tuple[ReadinessProbeV2, ...]

    def __post_init__(self) -> None:
        names = [process.name for process in self.processes]
        if len(names) != len(set(names)):
            raise BundleAuthorError("environment_plan_process_names_not_unique")
        if self.control_service is not None and self.control_service not in names:
            raise BundleAuthorError("environment_plan_control_service_missing")
        known = set(names)
        for process in self.processes:
            missing = sorted(set(process.depends_on) - known)
            if missing:
                raise BundleAuthorError(
                    f"environment_plan_dependency_missing: {process.name}: {', '.join(missing)}"
                )
        for slug, capability in self.capabilities.items():
            if capability.service not in known:
                raise BundleAuthorError(
                    f"environment_plan_capability_service_missing: {slug}: {capability.service}"
                )
        missing_probes = sorted(
            {
                probe.capability
                for probe in self.readiness
                if probe.capability not in self.capabilities
            }
        )
        if missing_probes:
            raise BundleAuthorError(
                "environment_plan_readiness_capability_missing: "
                + ", ".join(missing_probes)
            )


_COMPOSE_NAMES = (
    "compose.yml",
    "compose.yaml",
    "docker-compose.yml",
    "docker-compose.yaml",
)
_IGNORED_ARTIFACT_PARTS = {".git", ".venv", "__pycache__", "node_modules"}
_CERTIFIED_BUNDLE_DIRECTORY = "certified-bundle"


def _reuse_certified_bundle(
    *, source: Path, job: HarnessJob, authoring: Path, output: Path
) -> EnvironmentBundleV2 | None:
    """Publish the exact generic bundle that passed runtime validation.

    The authoring and execution sandboxes are separate.  Recompiling between them creates a
    time-of-check/time-of-use gap and can produce a different digest even when the submitted
    source fingerprint is unchanged.  A frozen bundle is accepted only when its certificate,
    source, contract, scenarios and canonical world still match the current job inputs.
    """

    artifact_root = authoring / "generic-harness"
    frozen = artifact_root / _CERTIFIED_BUNDLE_DIRECTORY
    certificate_path = authoring / "runtime-validation.json"
    if not frozen.is_dir() or not certificate_path.is_file():
        return None

    from .authoring_runtime_validation import _artifact_digest
    from .certification import (
        CheckStatus,
        GenericHarnessArtifactStore,
        verify_runtime_certification,
    )

    manifest = load_bundle_v2(frozen)
    source_digest = source_fingerprint(source)
    certificate = verify_runtime_certification(
        certificate_path,
        bundle_digest=manifest.digest,
        source_digest=source_digest,
    )
    store = GenericHarnessArtifactStore(artifact_root)
    scenarios = authoring / "scenarios"
    if not scenarios.is_dir():
        scenarios = authoring / "scenario"
    expected = {
        "contract": _artifact_digest(authoring / "contract.json"),
        "scenarios": _artifact_digest(scenarios),
    }
    actual = {
        "contract": certificate.authoring.contract_hash,
        "scenarios": certificate.authoring.scenario_set_hash,
    }
    # Hosted/black-box agents deliberately have no harness-owned source schema or world to
    # compare. Runtime validation records those checks as not applicable and explains why in the
    # certificate. Keep the tamper checks for every artifact that does exist, without comparing
    # model-authored placeholder state to an intentionally synthetic external-state fingerprint.
    if certificate.checks.source_invariants is not CheckStatus.NOT_APPLICABLE:
        expected["world"] = store.read_world_ir().fingerprint
        actual["world"] = certificate.authoring.world_ir_hash
    if certificate.checks.schema_and_seed is not CheckStatus.NOT_APPLICABLE:
        expected["source_schema"] = store.read_source_model().fingerprint
        actual["source_schema"] = certificate.source.schema_hash
    mismatched = sorted(name for name in expected if expected[name] != actual[name])
    if mismatched:
        raise BundleAuthorError(
            "certified_bundle_inputs_changed: " + ", ".join(mismatched)
        )
    scenario_total = sum(
        1 for path in (frozen / "scenarios").iterdir() if path.is_dir()
    )
    if scenario_total != job.scenario_count:
        raise BundleAuthorError(
            "certified_bundle_scenario_count_mismatch: "
            f"expected {job.scenario_count}, found {scenario_total}"
        )
    preflight_bundle(
        frozen,
        manifest,
        parallelism=job.runtime.parallelism,
        secret_refs={
            alias: reference.purpose
            for alias, reference in job.agent.secret_refs.items()
        },
    )

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    shutil.rmtree(temporary)
    shutil.copytree(frozen, temporary)
    if output.exists():
        backup = output.with_name(output.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        output.rename(backup)
        temporary.rename(output)
        shutil.rmtree(backup)
    else:
        temporary.rename(output)
    shutil.copy2(certificate_path, output.parent / "runtime-validation.json")
    return load_bundle_v2(output)


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True)
    return "'" + str(value).replace("'", "''") + "'"


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _json_type(values: list[Any]) -> str:
    present = [value for value in values if value is not None]
    if present and all(isinstance(value, bool) for value in present):
        return "boolean"
    if present and all(
        isinstance(value, int) and not isinstance(value, bool) for value in present
    ):
        return "bigint"
    if present and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in present
    ):
        return "double precision"
    if present and all(isinstance(value, (dict, list)) for value in present):
        return "jsonb"
    return "text"


def _constraint_checked_seed_sql(statements: list[str]) -> str:
    """Load generated rows in dependency order without disabling source constraints.

    Retry only foreign-key failures after other rows have been inserted. Each failed
    insert rolls back in its PL/pgSQL subtransaction. A pass with no progress rejects
    missing references/cycles instead of silently producing an invalid world.
    """
    if not statements:
        return ""
    commands = ",\n".join(_sql_literal(statement) for statement in statements)
    body = (
        "DECLARE\n"
        f" pending text[] := ARRAY[{commands}];\n"
        " remaining text[]; command text; progress boolean; failure_detail text;\n"
        "BEGIN\n"
        " WHILE cardinality(pending) > 0 LOOP\n"
        "  remaining := ARRAY[]::text[]; progress := false;\n"
        "  FOREACH command IN ARRAY pending LOOP\n"
        "   BEGIN\n"
        "    EXECUTE command; progress := true;\n"
        "   EXCEPTION WHEN foreign_key_violation THEN\n"
        "    GET STACKED DIAGNOSTICS failure_detail = MESSAGE_TEXT;\n"
        "    remaining := array_append(remaining, command);\n"
        "   END;\n"
        "  END LOOP;\n"
        "  IF cardinality(remaining) > 0 AND NOT progress THEN\n"
        "   RAISE EXCEPTION 'seed_dependency_unresolved: % statements; %', "
        "cardinality(remaining), failure_detail USING ERRCODE = '23503';\n"
        "  END IF;\n"
        "  pending := remaining;\n"
        " END LOOP;\n"
        "END"
    )
    return "DO " + _sql_literal(body) + ";\n"


def _collections_sql(path: Path, *, include_schema: bool = True) -> str:
    body = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise BundleAuthorError("collections_invalid: expected an object")
    statements: list[str] = []
    for table, raw_rows in body.items():
        rows = raw_rows if isinstance(raw_rows, list) else []
        records = [row for row in rows if isinstance(row, dict)]
        columns = sorted({str(column) for row in records for column in row})
        if not columns:
            columns = ["id"]
        definitions = [
            f"{_identifier(column)} {_json_type([row.get(column) for row in records])}"
            for column in columns
        ]
        if include_schema:
            statements.append(
                f"CREATE TABLE IF NOT EXISTS {_identifier(str(table))} "
                f"({', '.join(definitions)});"
            )
        for row in records:
            values = ", ".join(_sql_literal(row.get(column)) for column in columns)
            names = ", ".join(_identifier(column) for column in columns)
            statements.append(
                f"INSERT INTO {_identifier(str(table))} ({names}) VALUES ({values});"
            )
    if not include_schema:
        return _constraint_checked_seed_sql(statements)
    return "\n".join(statements) + "\n"


def _sqlite_type(declared: str) -> str:
    normalized = declared.upper()
    if "BOOL" in normalized:
        return "boolean"
    if "INT" in normalized:
        return "bigint"
    if any(mark in normalized for mark in ("REAL", "FLOA", "DOUB")):
        return "double precision"
    if any(mark in normalized for mark in ("NUMERIC", "DECIMAL")):
        return "numeric"
    if "BLOB" in normalized:
        return "bytea"
    return "text"


def _sqlite_json_type(values: list[Any], sql_type: str) -> str:
    """Preserve structured SQLite TEXT values when moving a world to Postgres.

    SQLite has no native JSON/array storage class, so generated worlds store lists and
    objects as JSON text.  Treating those columns as Postgres ``text`` changes the tool
    contract (``[]`` becomes the string ``"[]"``).  Only promote a column when every
    non-null value is a JSON object or array; ordinary strings remain text.
    """
    if sql_type != "text":
        return sql_type
    present = [value for value in values if value is not None]
    if not present or not all(isinstance(value, str) for value in present):
        return sql_type
    try:
        decoded = [json.loads(value) for value in present]
    except (TypeError, ValueError, json.JSONDecodeError):
        return sql_type
    return (
        "jsonb"
        if all(isinstance(value, (dict, list)) for value in decoded)
        else sql_type
    )


def _postgres_text_array_literal(value: Any) -> str:
    decoded = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, list) or any(
        isinstance(item, (dict, list)) for item in decoded
    ):
        raise BundleAuthorError("sqlite_text_array_invalid: expected scalar JSON array")
    escaped = [
        '"' + str(item).replace("\\", "\\\\").replace('"', '\\"') + '"'
        for item in decoded
    ]
    return "{" + ",".join(escaped) + "}"


def _sqlite_value(value: Any, sql_type: str) -> Any:
    if value is not None and sql_type == "boolean":
        return bool(value)
    if value is not None and sql_type == "jsonb" and isinstance(value, str):
        return json.loads(value)
    if value is not None and sql_type == "text[]":
        return _postgres_text_array_literal(value)
    return value


def _contract_column_declarations(
    contract: dict[str, Any],
) -> dict[tuple[str, str], str]:
    """Return authored SQL declarations keyed by table and column.

    SQLite affinity erases semantic types (notably BOOLEAN -> INTEGER and
    TIMESTAMPTZ -> TEXT). The contract is the authoritative schema description,
    so retain its safe type/default hints while still deriving keys and indexes
    from the executable SQLite world.
    """

    schema = contract.get("data_schema")
    if not isinstance(schema, dict):
        return {}
    return {
        (str(table), str(column)): str(declaration).strip()
        for table, raw_columns in schema.items()
        if isinstance(raw_columns, dict)
        for column, declaration in raw_columns.items()
        if str(declaration).strip()
    }


def _contract_sql_type(declaration: str) -> str | None:
    normalized = declaration.strip().upper()
    tokens = set(re.findall(r"[A-Z][A-Z0-9_]*", normalized))
    language_type = "|" in normalized or normalized.startswith(
        ("UNION[", "OPTIONAL[", "LIST[", "DICT[", "MAPPING[", "TUPLE[", "SET[")
    )
    if language_type:
        # Preserve the outer value shape before considering words that describe
        # members.  Human-readable contracts commonly say things such as
        # ``list of dicts with qty (int), cost_basis (float)``; classifying that
        # declaration as a number destroys the authored JSON value.
        if tokens & {"DICT", "MAPPING", "OBJECT", "JSON", "ANY"}:
            return "jsonb"
        if tokens & {"LIST", "ARRAY", "TUPLE", "SET"}:
            return "jsonb"
        if tokens & {"FLOAT", "NUMBER", "DECIMAL", "DOUBLE"}:
            return "double precision"
        if tokens & {"INT", "INTEGER"}:
            return "bigint"
        if tokens & {"BOOL", "BOOLEAN"}:
            return "boolean"
        if tokens & {"STR", "STRING"}:
            return "text"
    patterns = (
        (r"^(?:BOOL|BOOLEAN)\b", "boolean"),
        (r"^(?:BIGINT|INTEGER|INT|SMALLINT)\b", "bigint"),
        (r"^(?:DOUBLE PRECISION|REAL|FLOAT)\b", "double precision"),
        (
            r"^(?:NUMERIC|DECIMAL)(?:\s*\(\s*\d+\s*(?:,\s*\d+\s*)?\))?\b",
            "numeric",
        ),
        (r"^TIMESTAMPTZ\b", "timestamptz"),
        (r"^TIMESTAMP\b", "timestamp"),
        (r"^JSONB?\b", "jsonb"),
        (r"^TEXT\s*\[\s*\]", "text[]"),
        (r"^(?:TEXT|VARCHAR|CHAR)\b", "text"),
    )
    for pattern, sql_type in patterns:
        if re.match(pattern, normalized):
            return sql_type

    # Application contracts often use language-level unions instead of SQL
    # declarations. Interpret the whole declaration as a type set, with the wider
    # compatible representation winning independently of token order.
    if re.match(r"^(?:DICT|MAPPING|OBJECT|JSON|ANY)\b", normalized):
        return "jsonb"
    if re.match(r"^(?:LIST|ARRAY|TUPLE|SET)\b", normalized):
        # Without a proven homogeneous leaf type, JSONB preserves the value shape.
        return "jsonb"
    if tokens & {"FLOAT", "NUMBER", "DECIMAL", "DOUBLE"}:
        return "double precision"
    if tokens & {"INT", "INTEGER"}:
        return "bigint"
    if tokens & {"BOOL", "BOOLEAN"}:
        return "boolean"
    if tokens & {"STR", "STRING"}:
        return "text"
    return None


def _safe_sql_default(raw: Any, *, sql_type: str) -> str | None:
    """Translate a small, non-executable default grammar to PostgreSQL."""

    if raw is None:
        return None
    value = str(raw).strip()
    while len(value) >= 2 and value[0] == "(" and value[-1] == ")":
        value = value[1:-1].strip()
    upper = value.upper()
    if sql_type == "text[]" and re.fullmatch(r"'(?:[^']|'')*'", value):
        inner = value[1:-1].replace("''", "'")
        if inner.startswith("["):
            return _sql_literal(_postgres_text_array_literal(inner))
    if sql_type == "boolean" and upper in {"TRUE", "FALSE", "1", "0"}:
        return "TRUE" if upper in {"TRUE", "1"} else "FALSE"
    if upper in {"CURRENT_TIMESTAMP", "NOW()"}:
        return "now()"
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value):
        return value
    if re.fullmatch(r"'(?:[^']|'')*'", value):
        return value
    return None


def _contract_default(declaration: str, *, sql_type: str) -> str | None:
    match = re.search(
        r"\bDEFAULT\s+(NOW\(\)|CURRENT_TIMESTAMP|TRUE|FALSE|"
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)|'(?:[^']|'')*')",
        declaration,
        flags=re.IGNORECASE,
    )
    return _safe_sql_default(match.group(1), sql_type=sql_type) if match else None


def _sqlite_sql(
    path: Path,
    *,
    contract_declarations: dict[tuple[str, str], str] | None = None,
    include_schema: bool = True,
    include_rows: bool = True,
) -> str:
    statements: list[str] = []
    contract_declarations = contract_declarations or {}
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        for table in tables:
            info = list(connection.execute(f"PRAGMA table_info({_identifier(table)})"))
            selected = connection.execute(
                f"SELECT * FROM {_identifier(table)}"
            ).fetchall()
            definitions: list[str] = []
            columns: list[str] = []
            column_types: list[str] = []
            primary_key_columns = [
                str(row[1])
                for row in sorted(info, key=lambda item: int(item[5] or 0))
                if int(row[5] or 0)
            ]
            for row in info:
                name = str(row[1])
                declaration = contract_declarations.get((table, name), "")
                sql_type = _contract_sql_type(declaration) or _sqlite_json_type(
                    [record[name] for record in selected],
                    _sqlite_type(str(row[2] or "")),
                )
                # SQLite reports the ordinal of every column in a composite key.  Marking each
                # such column as an inline PostgreSQL primary key creates multiple conflicting
                # constraints.  Only a single-column key is emitted inline; composite keys are
                # emitted once as a table constraint below.
                suffix = (
                    " PRIMARY KEY"
                    if int(row[5] or 0) and len(primary_key_columns) == 1
                    else ""
                )
                if int(row[3] or 0) and not int(row[5] or 0):
                    suffix += " NOT NULL"
                default = _safe_sql_default(row[4], sql_type=sql_type)
                if default is None and declaration:
                    default = _contract_default(declaration, sql_type=sql_type)
                if default is not None:
                    suffix += f" DEFAULT {default}"
                definitions.append(f"{_identifier(name)} {sql_type}{suffix}")
                columns.append(name)
                column_types.append(sql_type)
            if len(primary_key_columns) > 1:
                definitions.append(
                    "PRIMARY KEY ("
                    + ", ".join(_identifier(column) for column in primary_key_columns)
                    + ")"
                )
            # ``PRAGMA table_info`` exposes primary keys but not UNIQUE constraints.  Dropping
            # those constraints during the SQLite -> Postgres compilation changes executable
            # tool semantics: a source statement such as ``ON CONFLICT (phone)`` becomes invalid
            # even though it worked against the authored world.  Preserve every concrete,
            # non-partial unique index except the primary-key index already represented above.
            for index in connection.execute(f"PRAGMA index_list({_identifier(table)})"):
                unique = bool(index[2])
                origin = str(index[3] or "")
                partial = bool(index[4])
                if not unique or origin == "pk" or partial:
                    continue
                index_name = str(index[1])
                index_columns = [
                    str(column[2])
                    for column in connection.execute(
                        f"PRAGMA index_info({_identifier(index_name)})"
                    )
                    if column[2] is not None
                ]
                if index_columns:
                    definitions.append(
                        "UNIQUE ("
                        + ", ".join(_identifier(column) for column in index_columns)
                        + ")"
                    )
            if include_schema:
                statements.append(
                    f"CREATE TABLE IF NOT EXISTS {_identifier(table)} "
                    f"({', '.join(definitions)});"
                )
            for record in selected if include_rows else ():
                # An authored SQLite world cannot retain the distinction between an
                # omitted source column and an explicitly stored NULL: every row is
                # read back with every column present.  When the real source schema is
                # adopted below, sending those NULLs explicitly suppresses PostgreSQL
                # defaults and can violate source NOT NULL constraints.  Treat NULL in
                # the generated world as "unspecified" and omit it from this row.  On
                # PostgreSQL that produces exactly the source-schema behaviour: its
                # default is applied when one exists, otherwise the value remains NULL.
                populated = [
                    (column, sql_type)
                    for column, sql_type in zip(columns, column_types, strict=True)
                    if record[column] is not None
                ]
                if not populated:
                    statements.append(
                        f"INSERT INTO {_identifier(table)} DEFAULT VALUES;"
                    )
                    continue
                names = ", ".join(
                    _identifier(column) for column, _sql_type in populated
                )
                values = ", ".join(
                    _sql_literal(_sqlite_value(record[column], sql_type))
                    for column, sql_type in populated
                )
                statements.append(
                    f"INSERT INTO {_identifier(table)} ({names}) VALUES ({values});"
                )
    finally:
        connection.close()
    if not include_schema:
        return _constraint_checked_seed_sql(statements)
    return "\n".join(statements) + "\n"


def _store_json_seed_sql(path: Path) -> str:
    """Restore rows exported by the existing ALK world store into adopted PostgreSQL tables.

    ``schema.sql`` is deliberately schema-only in several established authoring outputs.  The
    matching ``store.json`` carries the frozen rows under ``rows``.  PostgreSQL's
    ``jsonb_populate_recordset`` performs the type-aware conversion (including arrays, numerics,
    timestamps and JSON) against the adopted table definition instead of guessing SQL types.
    """
    body = json.loads(path.read_text(encoding="utf-8"))
    rows = body.get("rows") if isinstance(body, dict) else None
    if not isinstance(rows, dict):
        raise BundleAuthorError("store_invalid: expected an object with a rows object")
    statements: list[str] = []
    for table in sorted(rows):
        raw_rows = rows[table]
        if not isinstance(raw_rows, list):
            raise BundleAuthorError(f"store_invalid: rows.{table} must be an array")
        records = [row for row in raw_rows if isinstance(row, dict)]
        if len(records) != len(raw_rows):
            raise BundleAuthorError(
                f"store_invalid: rows.{table} contains a non-object row"
            )
        if not records:
            continue
        payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
        statements.append(
            f"INSERT INTO public.{_identifier(str(table))} "
            f"SELECT * FROM jsonb_populate_recordset(NULL::public.{_identifier(str(table))}, "
            f"{_sql_literal(payload)}::jsonb);"
        )
    if not statements:
        return ""
    return _constraint_checked_seed_sql(statements)


def _contained_source_path(source: Path, raw_path: str) -> Path | None:
    """Resolve a submitted path without ever following it outside the checkout."""

    try:
        candidate = (source / raw_path).resolve()
        root = source.resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if not candidate.is_relative_to(root) or not candidate.exists():
        return None
    return candidate


def _schema_like(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() == ".sql" and any(
        marker in name for marker in ("schema", "migration", "migrate", "ddl")
    )


def _compose_source_schema_paths(source: Path) -> list[Path]:
    """Discover repository-owned DDL mounted into a database init directory.

    Compose is only evidence here; it is never executed by the hosted guest. Restricting this
    to schema/migration-named SQL files avoids adopting fixture/seed data, which must come from
    the freshly authored scenario world instead.
    """

    compose_path = _compose_path(source)
    if compose_path is None:
        return []
    compose = _load_compose(compose_path)
    discovered: list[Path] = []
    for service in compose["services"].values():
        if not isinstance(service, dict):
            continue
        for volume in service.get("volumes") or []:
            raw_source = ""
            target = ""
            if isinstance(volume, str):
                pieces = volume.split(":")
                if len(pieces) >= 2:
                    raw_source, target = pieces[0], pieces[1]
            elif isinstance(volume, dict):
                raw_source = str(volume.get("source") or "")
                target = str(volume.get("target") or "")
            if "docker-entrypoint-initdb.d" not in target or not raw_source:
                continue
            path = _contained_source_path(source, raw_source)
            if path is None:
                continue
            if path.is_file() and _schema_like(path):
                discovered.append(path)
            elif path.is_dir():
                discovered.extend(
                    candidate
                    for candidate in sorted(path.rglob("*.sql"))
                    if candidate.is_file() and _schema_like(candidate)
                )
    return discovered


def _source_schema_paths(
    source: Path, *, contract: dict[str, Any] | None = None
) -> list[Path]:
    """Return deterministic source-owned schema artifacts in precedence order.

    Executable repository evidence is authoritative. The generated contract may point at that
    evidence, but it cannot replace or truncate it. This is intentionally independent of model
    output so two fresh authoring runs compile the same source schema.
    """

    candidates = _compose_source_schema_paths(source)
    for conventional in ("db/schema.sql", "schema.sql"):
        path = _contained_source_path(source, conventional)
        if path is not None and path.is_file():
            candidates.append(path)

    store = (contract or {}).get("data_store")
    declared = (
        str(store.get("schema_from") or "").strip() if isinstance(store, dict) else ""
    )
    if declared:
        path = _contained_source_path(source, declared)
        if path is not None:
            if path.is_file() and path.suffix.lower() == ".sql":
                candidates.append(path)
            elif path.is_dir():
                candidates.extend(
                    candidate
                    for candidate in sorted(path.rglob("*.sql"))
                    if candidate.is_file() and _schema_like(candidate)
                )
        elif declared.lower().endswith(".sql") and not candidates:
            raise BundleAuthorError(f"source_schema_missing: {declared}")

    unique: dict[str, Path] = {}
    for path in candidates:
        relative = path.relative_to(source.resolve()).as_posix()
        unique.setdefault(relative, path)
    return [unique[key] for key in sorted(unique)]


def _adopted_seed_sql(
    authoring: Path,
    *,
    source: Path | None = None,
    contract: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    source_schemas = (
        _source_schema_paths(source, contract=contract) if source is not None else []
    )
    if source_schemas:
        schema_sql = "\n".join(
            path.read_text(encoding="utf-8") for path in source_schemas
        )
        adopted = [
            f"source/{path.relative_to(source.resolve()).as_posix()}"
            for path in source_schemas
        ]
        store = authoring / "store.json"
        if store.is_file():
            return (
                schema_sql + "\n" + _store_json_seed_sql(store),
                adopted + ["store.json"],
            )
        sqlite = authoring / "world.sqlite"
        if sqlite.is_file():
            rows = _sqlite_sql(
                sqlite,
                contract_declarations=_contract_column_declarations(contract or {}),
                include_schema=False,
            )
            return schema_sql + "\n" + rows, adopted + ["world.sqlite"]
        collections = authoring / "collections.json"
        if collections.is_file():
            rows = _collections_sql(collections, include_schema=False)
            return schema_sql + "\n" + rows, adopted + ["collections.json"]
        return schema_sql, adopted

    schema = authoring / "schema.sql"
    if schema.is_file():
        sql = schema.read_text(encoding="utf-8")
        adopted = ["schema.sql"]
        store = authoring / "store.json"
        if store.is_file():
            sql += "\n" + _store_json_seed_sql(store)
            adopted.append("store.json")
        return sql, adopted
    sqlite = authoring / "world.sqlite"
    if sqlite.is_file():
        return _sqlite_sql(
            sqlite,
            contract_declarations=_contract_column_declarations(contract or {}),
        ), ["world.sqlite"]
    collections = authoring / "collections.json"
    if collections.is_file():
        return _collections_sql(collections), ["collections.json"]
    return "", []


def _generic_postgres_seed_artifacts(
    authoring: Path,
    staging: Path,
    *,
    source: Path,
    contract: dict[str, Any],
    prefix: str,
    allow_harness_owned_schema: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    """Package source schema and semantic rows separately for runtime catalogue inspection."""

    source_schemas = _source_schema_paths(source, contract=contract)
    canonical_world = authoring / "generic-harness" / "world-ir.json"
    legacy_world = authoring / "world.sqlite"
    data_store = contract.get("data_store")
    data_store = data_store if isinstance(data_store, dict) else {}
    store_kind = str(data_store.get("kind") or "").strip().lower()
    normalized_store_kind = store_kind.replace("-", "_").replace(" ", "_")
    embedded_store = any(
        marker in normalized_store_kind
        for marker in (
            "none",
            "in_process",
            "in_memory",
            "memory",
            "sqlite",
            "filesystem",
            "file_store",
            "local_state",
        )
    )
    declared_tools = contract.get("tools")
    explicitly_tool_free = isinstance(declared_tools, list) and not declared_tools
    if (
        not source_schemas
        and not embedded_store
        and not explicitly_tool_free
        and not allow_harness_owned_schema
    ):
        raise BundleAuthorError(
            "generic_pipeline_source_schema_required: no source-owned PostgreSQL schema found"
        )
    if not canonical_world.is_file() and not legacy_world.is_file():
        raise BundleAuthorError(
            "generic_pipeline_world_ir_required: expected generic-harness/world-ir.json "
            "or compatibility world.sqlite"
        )
    seed = staging / "seed"
    schema_path = seed / "source-schema.sql"
    if source_schemas:
        schema_sql = "\n".join(
            path.read_text(encoding="utf-8") for path in source_schemas
        )
    else:
        # A data-free/in-process source has no repository-owned database schema to adopt.
        # During the compatibility window SQLite may carry that harness-owned schema. Canonical
        # World IR deliberately contains logical values only and cannot invent native DDL.
        if not legacy_world.is_file():
            raise BundleAuthorError(
                "generic_pipeline_source_schema_required: canonical World IR requires "
                "source-owned schema metadata"
            )
        schema_sql = _sqlite_sql(
            legacy_world,
            contract_declarations=_contract_column_declarations(contract),
            include_rows=False,
        )
    schema_path.write_text(prefix + schema_sql, encoding="utf-8")
    if canonical_world.is_file():
        world_path = seed / "world-ir.json"
        shutil.copy2(canonical_world, world_path)
        adopted_world = "generic-harness/world-ir.json"
    else:
        world_path = seed / "world.sqlite"
        shutil.copy2(legacy_world, world_path)
        adopted_world = "world.sqlite"
    adopted_contracts: list[str] = []
    # Runtime seed compilation must see the same code/action facts as validation.
    # Keep them inside the hashed bundle: the execution sandbox has no authoring
    # artifact-store dependency and PostgreSQL inspection only recovers state facts.
    source_model = authoring / "generic-harness" / "source-model.json"
    if source_model.is_file():
        shutil.copy2(source_model, seed / "source-model.json")
        adopted_contracts.append("generic-harness/source-model.json")
    contracts = staging / "contracts"
    for name in ("source-model.schema.json", "world-ir.schema.json"):
        schema = authoring / "generic-harness" / name
        if schema.is_file():
            contracts.mkdir(exist_ok=True)
            shutil.copy2(schema, contracts / name)
            adopted_contracts.append(f"generic-harness/{name}")
    adopted = [
        f"source/{path.relative_to(source.resolve()).as_posix()}"
        for path in source_schemas
    ] + [adopted_world, *adopted_contracts]
    return ["seed/source-schema.sql"], [f"seed/{world_path.name}"], adopted


def _compose_path(source: Path) -> Path | None:
    matches = [source / name for name in _COMPOSE_NAMES if (source / name).is_file()]
    if len(matches) > 1:
        raise BundleAuthorError(
            "compose_ambiguous: " + ", ".join(path.name for path in matches)
        )
    return matches[0] if matches else None


def _load_compose(path: Path) -> dict[str, Any]:
    try:
        body = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise BundleAuthorError(f"compose_invalid: {exc}") from exc
    if not isinstance(body, dict) or not isinstance(body.get("services"), dict):
        raise BundleAuthorError("compose_invalid: services must be an object")
    return body


_RUNTIME_ENVIRONMENT_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")
_SECRET_ENVIRONMENT_NAME = re.compile(
    r"(?:API_?KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL|PRIVATE_?KEY)", re.IGNORECASE
)


def _declared_runtime_environment(source: Path) -> dict[str, str]:
    """Load public, non-secret process defaults declared by the repository.

    Bundle V2 processes do not execute a Docker image and therefore cannot inherit image-level
    ``ENV`` values. Repositories that need deterministic runtime knobs can declare them in
    ``alk.yaml`` under ``runtime.environment``. Values are sealed into the bundle manifest, so
    credential-shaped names and shell-style interpolation are rejected; secrets must continue
    to travel through purpose-scoped refs.
    """
    path = source / "alk.yaml"
    if not path.is_file():
        return {}
    try:
        body = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise BundleAuthorError(f"runtime_manifest_invalid: {exc}") from exc
    if not isinstance(body, dict):
        raise BundleAuthorError("runtime_manifest_invalid: root must be an object")
    runtime = body.get("runtime") or {}
    if not isinstance(runtime, dict):
        raise BundleAuthorError("runtime_manifest_invalid: runtime must be an object")
    raw_environment = runtime.get("environment") or {}
    if not isinstance(raw_environment, dict):
        raise BundleAuthorError(
            "runtime_manifest_invalid: runtime.environment must be an object"
        )
    environment: dict[str, str] = {}
    for raw_name, raw_value in raw_environment.items():
        name = str(raw_name)
        if not _RUNTIME_ENVIRONMENT_NAME.fullmatch(name):
            raise BundleAuthorError(f"runtime_environment_name_invalid: {name}")
        if _SECRET_ENVIRONMENT_NAME.search(name):
            raise BundleAuthorError(f"runtime_environment_secret_forbidden: {name}")
        if not isinstance(raw_value, (str, int, float, bool)):
            raise BundleAuthorError(f"runtime_environment_value_invalid: {name}")
        value = str(raw_value)
        if "${" in value or "{{" in value:
            raise BundleAuthorError(
                f"runtime_environment_interpolation_forbidden: {name}"
            )
        environment[name] = value
    return environment


def _python_process(
    *,
    name: str,
    working_directory: str,
    entry: str,
    control: bool,
    needs_secrets: bool,
    port: int | None = None,
    environment: dict[str, str] | None = None,
    depends_on: list[str] | None = None,
) -> SourceProcess:
    # The build tree is writable; the submitted source remains read-only.  ``uv sync`` creates a
    # project-local venv for pyproject repositories, while requirements/stdlib sources get the
    # same explicit venv boundary.  No dependency is installed into the immutable snapshot.
    build: list[list[str]]
    run: list[str]
    relative_root = Path(working_directory)
    # Discovery happens at the caller's source root; these placeholders are resolved below by
    # `_plan_python`, which replaces this conservative default where necessary.
    build = [["python3.12", "-m", "venv", ".venv"]]
    run = [".venv/bin/python", entry]
    del relative_root
    return SourceProcess(
        name=name,
        working_directory=working_directory,
        build_commands=build,
        run_command=run,
        # Match the established Compose harness lane: submitted processes may adapt
        # deterministic test-only provider seams without receiving an extra credential or
        # control-plane capability.
        environment={"HARNESS_MODE": "1", **(environment or {})},
        fixed_port=port,
        started_check=StartedCheck(port=True, timeout_seconds=180) if port else None,
        secret_purposes=[SecretPurpose.TARGET_PROVIDER] if needs_secrets else [],
        user=ProcessUser.SVC_AGENT if control else ProcessUser.SVC_TOOLS,
        depends_on=depends_on or [],
    )


def _plan_python(
    source: Path,
    *,
    name: str,
    root: Path,
    entry: str,
    control: bool,
    needs_secrets: bool,
    port: int | None = None,
    environment: dict[str, str] | None = None,
    depends_on: list[str] | None = None,
    livekit_download: bool = False,
    run_override: list[str] | None = None,
) -> SourceProcess:
    relative = root.relative_to(source).as_posix() or "."
    process = _python_process(
        name=name,
        working_directory=relative,
        entry=entry,
        control=control,
        needs_secrets=needs_secrets,
        port=port,
        environment=environment,
        depends_on=depends_on,
    )
    python = _docker_python(root)
    if (root / "pyproject.toml").is_file():
        commands = [["uv", "sync", "--no-cache", "--python", python]]
        if (root / "uv.lock").is_file():
            commands[0].append("--locked")
        if livekit_download:
            commands.append(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    "-m",
                    "livekit.agents",
                    "download-files",
                ]
            )
        run = ["uv", "run", "--no-sync", "python", entry]
    elif (root / "requirements.txt").is_file():
        commands = [
            [python, "-m", "venv", ".venv"],
            [
                ".venv/bin/python",
                "-m",
                "pip",
                "install",
                "--requirement",
                "requirements.txt",
            ],
        ]
        run = [".venv/bin/python", entry]
    else:
        commands = []
        run = [python, entry]
    return process.model_copy(
        update={"build_commands": commands, "run_command": run_override or run}
    )


def _docker_python(root: Path) -> str:
    dockerfile = root / "Dockerfile"
    if not dockerfile.is_file():
        return "python3.12"
    text = dockerfile.read_text(encoding="utf-8", errors="replace")
    argument = re.search(r"(?mi)^ARG\s+PYTHON_VERSION\s*=\s*([0-9]+\.[0-9]+)\s*$", text)
    if argument:
        return f"python{argument.group(1)}"
    direct = re.search(r"(?mi)^FROM\s+(?:[^/\s]+/)*python:([0-9]+\.[0-9]+)", text)
    return f"python{direct.group(1)}" if direct else "python3.12"


def _dockerfile_run(root: Path) -> list[str] | None:
    dockerfile = root / "Dockerfile"
    if not dockerfile.is_file():
        return None
    commands = []
    for line in dockerfile.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("CMD "):
            commands.append(stripped[4:].strip())
    if not commands:
        return None
    raw = commands[-1]
    if not raw.startswith("["):
        raise BundleAuthorError(
            f"dockerfile_command_unsupported: {dockerfile} uses shell-form CMD"
        )
    try:
        argv = json.loads(raw)
    except ValueError as exc:
        raise BundleAuthorError(f"dockerfile_command_invalid: {dockerfile}") from exc
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(item, str) for item in argv)
    ):
        raise BundleAuthorError(f"dockerfile_command_invalid: {dockerfile}")
    if argv[0] == "python":
        argv[0] = ".venv/bin/python" if (root / "requirements.txt").is_file() else "uv"
        if argv[0] == "uv":
            argv[1:1] = ["run", "--no-sync", "python"]
    elif (
        argv[0] in {"uvicorn", "gunicorn", "flask"}
        and (root / "requirements.txt").is_file()
    ):
        argv[0] = f".venv/bin/{argv[0]}"
    return argv


# LiveKit's CLI needs a subcommand: `agent.py` alone prints usage and exits without registering.
_LIVEKIT_WORKER_SUBCOMMANDS = frozenset({"start", "dev", "connect", "console"})


def _livekit_cli_fixed_health_port(
    command: list[str], environment: dict[str, str], *roots: Path
) -> int | None:
    """Describe LiveKit's production CLI port so the runtime can avoid contention.

    ``start`` binds its health server to 8081 and does not expose a ``--port``
    CLI option. Declaring that fixed port makes the existing port planner
    safely degrade a multi-world sandbox to one world. ``dev`` already asks the
    OS for an ephemeral port and needs no declaration, and neither does a worker
    the isolation shim moves off that port.
    """
    if "start" not in command:
        return None
    if "FI_WORKER_HEALTH_PORT" in environment and _livekit_isolatable(*roots):
        return None
    return 8081


def _livekit_isolatable(*roots: Path) -> bool:
    """Whether the pinned livekit-agents has the ``AgentServer`` the isolation shim drives (1.3+)."""
    for root in roots:
        for name in ("uv.lock", "poetry.lock", "requirements.txt"):
            path = root / name
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            found = re.search(
                r'name = "livekit-agents"\s*\nversion = "(\d+)\.(\d+)', text
            ) or re.search(r"livekit-agents(?:\[[^\]]*\])?\s*==\s*(\d+)\.(\d+)", text)
            if found:
                return (int(found[1]), int(found[2])) >= (1, 3)
    return False


def _hands_off_to_livekit_cli(root: Path, entry: str) -> bool:
    """Whether the entry delegates to LiveKit's CLI. An agent that runs its own worker must not."""
    path = root / entry
    if not path.is_file():
        return False
    return "cli.run_app" in path.read_text(encoding="utf-8", errors="replace")


def _discover_callback_entrypoint(root: Path) -> str | None:
    """Return the repository's unique module-level ``agent_callback``, if present.

    Callback support is a source property, not an LLM-authored contract property.  Contract
    authoring can legitimately omit ``runtime.interface`` even when the repository exports the
    canonical callback.  Treating that omission as authoritative used to compile such agents as
    ``python agent.py`` HTTP services, which can never pass the generated port readiness probe.
    """
    candidates: list[str] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if any(part in _IGNORED_ARTIFACT_PARTS for part in relative.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "agent_callback"
            for node in tree.body
        ):
            module = ".".join(relative.with_suffix("").parts)
            candidates.append(f"{module}:agent_callback")
    if not candidates:
        return None
    if len(candidates) != 1:
        raise BundleAuthorError(
            "callback_entrypoint_ambiguous: " + ", ".join(candidates)
        )
    return candidates[0]


def _callback_entrypoint(root: Path) -> str:
    """Find the callback promised by an explicitly callable runtime contract."""
    candidate = _discover_callback_entrypoint(root)
    if candidate is None:
        raise BundleAuthorError(
            "callback_entrypoint_missing: callable runtime requires one module-level "
            "agent_callback"
        )
    return candidate


def _langgraph_entrypoint(root: Path) -> str | None:
    """Read a source-declared LangGraph graph without guessing an agent script."""

    path = root / "langgraph.json"
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BundleAuthorError(f"langgraph_config_invalid: {exc}") from exc
    graphs = document.get("graphs") if isinstance(document, dict) else None
    if not isinstance(graphs, dict) or len(graphs) != 1:
        raise BundleAuthorError(
            "langgraph_graph_ambiguous: expected exactly one declared graph"
        )
    declaration = next(iter(graphs.values()))
    if not isinstance(declaration, str) or ":" not in declaration:
        raise BundleAuthorError("langgraph_graph_invalid: expected path:attribute")
    file_name, attribute = declaration.rsplit(":", 1)
    graph_path = (root / file_name).resolve()
    if (
        not graph_path.is_relative_to(root)
        or not graph_path.is_file()
        or not graph_path.suffix == ".py"
    ):
        raise BundleAuthorError(
            "langgraph_graph_invalid: graph source must be a Python file in the repository"
        )
    if not attribute.isidentifier():
        raise BundleAuthorError("langgraph_graph_invalid: graph attribute is invalid")
    return f"{graph_path.relative_to(root).as_posix()}:{attribute}"


def _callback_adapter_source() -> str:
    return (
        Path(__file__).with_name("callback_http_adapter.py").read_text(encoding="utf-8")
    )


def _langgraph_adapter_source() -> str:
    return (
        Path(__file__)
        .with_name("langgraph_http_adapter.py")
        .read_text(encoding="utf-8")
    )


def _subprocess_adapter_source() -> str:
    """Return the generic command-to-HTTP bridge embedded in a Bundle V2 process."""

    return (
        Path(__file__)
        .with_name("subprocess_http_adapter.py")
        .read_text(encoding="utf-8")
    )


def _runtime_component(root: Path, configured_workdir: str) -> Path:
    """Resolve a source-declared workdir to its nearest installable project root."""

    configured = configured_workdir.strip()
    component = (
        (root / configured).resolve() if configured not in {"", ".", "/"} else root
    )
    if not component.is_relative_to(root) or not component.is_dir():
        raise BundleAuthorError(f"runtime_workdir_invalid: {configured_workdir or '.'}")
    for candidate in (component, *component.parents):
        if not candidate.is_relative_to(root):
            break
        if any(
            (candidate / name).is_file()
            for name in ("pyproject.toml", "requirements.txt", "setup.py")
        ):
            return candidate
    return component


def _submitted_command(process: SourceProcess, command: list[str]) -> list[str]:
    """Run a contract argv inside the dependency environment selected for the source."""

    if not command:
        return list(process.run_command)
    normalized = [str(item) for item in command]
    if normalized[0] in {"python", "python3", "python3.11", "python3.12", "python3.13"}:
        if process.run_command[:3] == ["uv", "run", "--no-sync"]:
            return [*process.run_command[:4], *normalized[1:]]
        return [process.run_command[0], *normalized[1:]]
    if process.run_command[:3] == ["uv", "run", "--no-sync"] and normalized[0] != "uv":
        return ["uv", "run", "--no-sync", *normalized]
    return normalized


def _managed_world_db() -> ManagedProcess:
    return ManagedProcess(
        name="world-db",
        engine=ManagedEngine.POSTGRES,
        version="16",
        user=ProcessUser.SVC_DATA,
    )


def _tool_proxy_process() -> SourceProcess:
    return SourceProcess(
        name="tool-proxy",
        working_directory="generated/tool-proxy",
        source_origin="bundle",
        # The ALK interpreter carries the proxy's dependencies in both local and
        # hosted runtimes; its path is not necessarily /opt/alk-venv.
        run_command=[sys.executable, "proxy.py"],
        environment={
            "PORT": "{{PORT_tool-proxy}}",
            "UPSTREAM_URL": "{{TOOLS_UPSTREAM_URL}}",
            "DATABASE_URL": "{{WORLD_DATABASE_URL}}",
        },
        started_check=StartedCheck(port=True, timeout_seconds=180),
        user=ProcessUser.SVC_TOOLS,
        depends_on=["tools-api", "world-db"],
    )


# --- C1 (world-port-model v1.3) authoring seams ----------------------------------------------

# The env var the rewritten tools-api command reads its per-world port from (C1 §1, checklist 2).
_FI_TOOLS_PORT = "FI_TOOLS_PORT"

# C1 §4 worker knob — the port livekit-agents 1.7.1 exposes ONLY as a WorkerOptions constructor
# arg (no env/CLI override), so the harness delivers it as an env var and consumes it itself (see
# `_worker_knob_env` below) rather than requiring the agent under test to read it.


def _shell_port_command(argv: list[str], port: str, env_key: str) -> str | None:
    """Render ``argv`` as a single ``sh -c`` string with the literal ``--port <port>`` replaced by
    an unquoted ``$env_key`` reference, or ``None`` if the command declares no such port.

    C1 §1: ``run_command`` is exec'd verbatim and never template-rendered, so a ``{{PORT_<self>}}``
    token in argv would reach the process as literal bytes. The ONLY valid consumability wiring is
    a ``{{PORT_<self>}}``-bearing environment value referenced by ``$KEY`` inside ``sh -c``. Every
    other token is shell-quoted; only the ``$KEY`` reference is left unquoted so the shell expands
    it. Returning ``None`` (no ``--port`` literal to rewrite) means the process cannot be wired
    honestly and MUST NOT be flagged consumable (that is the ``fixed_port_consumable_unwired`` lie).
    """
    ref = f"${env_key}"
    rendered: list[str] = []
    replaced = False
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--port" and index + 1 < len(argv) and argv[index + 1] == port:
            rendered.append(shlex.quote(token))
            rendered.append(ref)
            index += 2
            replaced = True
            continue
        if token == f"--port={port}":
            rendered.append(f"--port={ref}")
            index += 1
            replaced = True
            continue
        rendered.append(shlex.quote(token))
        index += 1
    if not replaced:
        return None
    return " ".join(rendered)


def _consumable_source_process(
    process: SourceProcess, env_key: str, port_token: str
) -> SourceProcess:
    """Opt a fixed-port source process into env-consumable parallelism, or leave it code-fixed.

    Rewrites ``run_command`` to the ``sh -c`` + ``$KEY`` form, adds ``env_key: port_token`` to the
    environment, and sets ``fixed_port_consumable=True`` — but ONLY when the command carries a
    rewritable ``--port <fixed_port>`` literal. When it does not, the process is returned unchanged
    (code-fixed, honestly degrading at W>1) rather than flagged with a wiring that does not exist.
    """
    if process.fixed_port is None:
        return process
    shelled = _shell_port_command(
        list(process.run_command), str(process.fixed_port), env_key
    )
    if shelled is None:
        return process
    environment = dict(process.environment)
    environment[env_key] = port_token
    return process.model_copy(
        update={
            "run_command": ["sh", "-c", shelled],
            "environment": environment,
            "fixed_port_consumable": True,
        }
    )


def _worker_knob_env(process_name: str) -> dict[str, str]:
    """C1 §4: the FI_* worker knob for one LiveKit-worker process, fed its OWN token.

    ``FI_WORKER_HEALTH_PORT``'s presence IS the knob-bearing mark (both authoring and runtime key
    on it). It is not read by the agent under test: the harness's own ``sitecustomize`` shim
    (``livekit_tool_trace_bootstrap.py``) consumes it at worker start, flipping the worker into
    livekit-agents' own side-by-side mode -- the agent under test is never modified. Absent, the
    shim is a no-op and the worker stays on library defaults.

    NOTE: ``FI_HOSTED_DISPATCH_ACK`` (D32 / C3 §4.5) is deliberately NOT authored here. The
    dispatch-ack ladder in ``engines/livekit.py`` runs in the GUEST MAIN PROCESS (under
    ``hosted_entrypoint`` -> ``call_runner``), not in this spawned agent-under-test child, so the
    engine reads the flag from the guest main process's own ``os.environ`` -- ``hosted_entrypoint``
    arms it there. Putting it on this worker env would leave the ladder dormant (wrong process).
    """
    return {
        "FI_WORKER_HEALTH_PORT": f"{{{{PORT_{process_name}}}}}",
    }


def _rewrite_managed_dependency_environment(
    environment: dict[str, str],
    *,
    managed_services: set[str],
    capabilities: dict[str, CapabilityV2],
) -> dict[str, str]:
    """Translate Compose service URLs to runtime-owned capability addresses.

    Source containers normally address dependencies through Compose DNS names. Hosted Bundle V2
    processes share a sandbox host instead, with ports allocated per world. The translation is
    derived only from declared service/capability metadata; it does not know an agent, framework,
    environment-variable name, or repository layout.
    """

    by_service = {
        service: [
            (slug, capability)
            for slug, capability in capabilities.items()
            if capability.service == ("world-db" if service == "postgres" else service)
            and capability.configuration_name
        ]
        for service in managed_services
    }
    rewritten: dict[str, str] = {}
    for name, value in environment.items():
        parsed = urlsplit(value)
        service = parsed.hostname or ""
        candidates = by_service.get(service, [])
        if parsed.scheme and candidates:
            if len(candidates) != 1:
                raise BundleAuthorError(
                    f"managed_dependency_capability_ambiguous: {service}: "
                    + ", ".join(slug for slug, _ in candidates)
                )
            _, capability = candidates[0]
            replacement = f"{{{{{capability.configuration_name}}}}}"
            if capability.protocol is CapabilityProtocol.REDIS:
                # Redis DB selectors and query options are source semantics and remain valid on
                # the harness-owned endpoint. PostgreSQL database names do not: the runtime must
                # select its isolated wN database, already encoded in the capability address.
                if parsed.path and parsed.path != "/":
                    replacement += parsed.path
                if parsed.query:
                    replacement += "?" + parsed.query
            rewritten[name] = replacement
            continue
        if value in managed_services:
            rewritten[name] = f"{{{{HOST_{value}}}}}"
            continue
        rewritten[name] = value
    return rewritten


def resolve_environment_plan(
    source: str | Path,
    job: HarnessJob,
    *,
    contract_modality: str | None = None,
    contract_interface_kind: str | None = None,
    contract_runtime: dict[str, Any] | None = None,
) -> EnvironmentPlanV2:
    """Resolve packaging once.  Authoring and provisioning consume this same immutable plan."""
    root = Path(source).resolve()
    if not root.is_dir():
        raise BundleAuthorError(f"source_unavailable: {root}")
    connector = job.agent.connector.lower()
    if job.agent.mode is ProviderExecutionMode.ENVIRONMENT_BACKED:
        declaration = load_provider_manifest(
            root, str(job.agent.config.get("lifecycle_manifest") or "alk.yaml")
        )
        if declaration.provider.type.value != connector:
            raise BundleAuthorError(
                "provider_lifecycle_connector_mismatch: "
                f"job={connector}, manifest={declaration.provider.type.value}"
            )
    # Hosted repository submissions normally arrive as ``connector=auto``.  In the unified
    # Daytona lane the contract is authored *after* dispatch, so the control plane cannot rewrite
    # that field before this compiler runs.  The frozen contract is therefore the authoritative
    # late-bound modality signal.  Voice is routed through LiveKit because that is the hosted
    # repository voice connector implemented by the guest; explicit vapi/retell values never
    # enter this path.
    is_livekit = connector == "livekit" or (
        connector == "auto" and (contract_modality or "").strip().lower() == "voice"
    )
    needs_target_secrets = any(
        reference.purpose == SecretPurpose.TARGET_PROVIDER.value
        for reference in job.agent.secret_refs.values()
    )
    compose = _compose_path(root)
    processes: list[ManagedProcess | SourceProcess] = [_managed_world_db()]
    capabilities: dict[str, CapabilityV2] = {
        "world_db": CapabilityV2(
            protocol=CapabilityProtocol.POSTGRES,
            service="world-db",
            container_port=5432,
            configuration_name="WORLD_DATABASE_URL",
        )
    }
    readiness = [ReadinessProbeV2(capability="world_db", timeout_seconds=180)]
    declared_runtime_environment = _declared_runtime_environment(root)
    contract_runtime = contract_runtime if isinstance(contract_runtime, dict) else {}
    runtime_command = [str(item) for item in (contract_runtime.get("command") or [])]
    runtime_workdir = str(contract_runtime.get("workdir") or "")
    interface = contract_runtime.get("interface")
    interface = interface if isinstance(interface, dict) else {}
    # A generated contract may correctly identify the HTTP seam but omit its start command.
    # The repository's exec-form Dockerfile CMD is an explicit source-owned declaration; use it
    # before falling back to a guessed agent.py entrypoint. This works for any HTTP framework.
    if not runtime_command and interface.get("kind") == "http" and not compose:
        runtime_command = _dockerfile_run(root) or []
    interface_port = interface.get("port")
    interface_health_path = str(interface.get("health_path") or "/health")

    # A connect-only provider target is hosted by Vapi/Retell and is addressed by the
    # provider ID in the job.  When no repository was submitted there is deliberately no
    # customer process to discover or launch; the local runtime only owns the isolated world.
    # Keep repository-backed connect-only jobs on the normal path so uploaded tool/backend
    # implementations are still compiled and exercised.
    if (
        job.agent.mode is ProviderExecutionMode.CONNECT_ONLY
        and job.source.kind is SourceKind.PROVIDER
    ):
        return EnvironmentPlanV2(
            packaging="provider_connect_only",
            control_service=None,
            processes=tuple(processes),
            capabilities=capabilities,
            readiness=tuple(readiness),
        )

    if compose is not None:
        body = _load_compose(compose)
        services = body["services"]
        # Compile the submitted topology.  Supported managed services become snapshot engines;
        # source services remain source processes.  Unknown image-only dependencies are rejected
        # explicitly instead of being silently emulated.
        managed_names: set[str] = set()
        for service_name, raw in services.items():
            service = raw if isinstance(raw, dict) else {}
            image = str(service.get("image") or "")
            if image.startswith("postgres:"):
                if service_name != "postgres":
                    raise BundleAuthorError(
                        f"managed_name_unsupported: postgres service must be named postgres, got {service_name}"
                    )
                managed_names.add(service_name)
                continue
            if image and "redis" in image:
                processes.append(
                    ManagedProcess(
                        name=service_name,
                        engine=ManagedEngine.REDIS,
                        version=image.split(":", 1)[1].split("-", 1)[0]
                        if ":" in image
                        else "7",
                        user=ProcessUser.SVC_DATA,
                    )
                )
                managed_names.add(service_name)
                capabilities[f"{service_name}_redis"] = CapabilityV2(
                    protocol=CapabilityProtocol.REDIS,
                    service=service_name,
                    container_port=6379,
                    configuration_name=f"{service_name.upper().replace('-', '_')}_URL",
                )
                readiness.append(ReadinessProbeV2(capability=f"{service_name}_redis"))
                continue
            if image and not service.get("build"):
                raise BundleAuthorError(
                    f"engine_unsupported: image-only service {service_name!r} ({image!r}) is not in the snapshot catalog"
                )

        source_services = [name for name in services if name not in managed_names]
        control_name = (
            "agent"
            if "agent" in source_services
            else (
                "api"
                if "api" in source_services
                else source_services[-1]
                if source_services
                else ""
            )
        )
        if not control_name:
            raise BundleAuthorError(
                "control_service_missing: compose has no source-built service"
            )
        for service_name in source_services:
            service = services[service_name]
            build = service.get("build", ".")
            if isinstance(build, dict):
                context = str(build.get("context") or ".")
            else:
                context = str(build)
            service_root = (root / context).resolve()
            if not service_root.is_relative_to(root):
                raise BundleAuthorError(f"build_context_escape: {service_name}")
            environment: dict[str, str] = {}
            raw_env = service.get("environment") or {}
            if isinstance(raw_env, dict):
                environment = {
                    str(k): str(v) for k, v in raw_env.items() if v is not None
                }
            depends = (
                list((service.get("depends_on") or {}).keys())
                if isinstance(service.get("depends_on"), dict)
                else list(service.get("depends_on") or [])
            )
            environment = _rewrite_managed_dependency_environment(
                environment,
                managed_services=managed_names,
                capabilities=capabilities,
            )
            depends = ["world-db" if item == "postgres" else item for item in depends]
            if service_name == "tools-api" and "postgres" in managed_names:
                # The target DB is intentionally a distinct per-world logical DB on the same
                # harness-owned Postgres engine. This preserves reset/isolation without another
                # daemon per call.
                environment.setdefault("DATABASE_URL", "{{WORLD_DATABASE_URL}}")
            if service_name == control_name and "tools-api" in source_services:
                environment["TOOLS_API_URL"] = "{{TOOLS_API_URL}}"
            if is_livekit and service_name == control_name:
                environment = {**declared_runtime_environment, **environment}
                environment.setdefault(
                    "LIVEKIT_AGENT_NAME",
                    "voice-booking-{{JOB_ID}}-w{{WORLD_INDEX}}",
                )
                environment.setdefault(
                    "HARNESS_TOOL_TRACE",
                    "{{WORLD_DIR}}/agent-tool-calls.jsonl",
                )
                # C1 §4: author the worker knob UNCONDITIONALLY into the LiveKit worker,
                # fed its own `{{PORT_<name>}}`. This IS what marks it knob-bearing.
                environment.update(_worker_knob_env(service_name))
            entry = (
                "agent/agent.py"
                if (service_root / "agent" / "agent.py").is_file()
                else "agent.py"
            )
            port = (
                8080
                if service_name in {"api", "tools-api"}
                or (service_name == control_name and not is_livekit)
                else None
            )
            process = _plan_python(
                root,
                name=service_name,
                root=service_root,
                entry=entry,
                control=service_name == control_name,
                needs_secrets=needs_target_secrets and service_name == control_name,
                port=port,
                environment=environment,
                depends_on=[item for item in depends if item != "postgres"],
                livekit_download=is_livekit and service_name == control_name,
                run_override=_dockerfile_run(service_root),
            )
            if (
                port
                and service_name in {"api", "tools-api"}
                and not (is_livekit and service_name == control_name)
            ):
                # C1 §1 / checklist 2: the tools-api/api server pins its port in a Dockerfile CMD
                # copied verbatim into run_command. Rewrite it to consume its per-world allocated
                # port through the one valid wiring ($FI_TOOLS_PORT in `sh -c`) so it parallelizes
                # at W>1 instead of forcing a degrade to W=1.
                #
                # Track A′ D37 LIMITATION: a single process that is BOTH the knob-bearing LiveKit
                # control worker AND a consumable HTTP server runs W=1 only. It is excluded here so
                # it does NOT receive FI_TOOLS_PORT alongside FI_WORKER_HEALTH_PORT — both would
                # carry the SAME `{{PORT_<name>}}` token, colliding the worker health server and the
                # HTTP server on one port at any W. It stays a plain fixed_port (non-consumable) and
                # degrades to W=1 honestly (at W=1 the default health port does not collide). The
                # normal topology (control=agent + a separate tools-api) is unaffected.
                process = _consumable_source_process(
                    process, _FI_TOOLS_PORT, f"{{{{PORT_{service_name}}}}}"
                )
            # Compose commonly publishes a fixed host port for developer convenience while the
            # application itself already accepts its listen port through an environment value.
            # Keeping that published port as ``fixed_port`` unnecessarily collapses a generic
            # runtime to one world and prevents the isolation canary from being exercised.  When
            # the repository has an explicit, unambiguous port seam, bind each world to the
            # provisioner's allocated port instead.  This is framework- and modality-neutral:
            # services that truly hard-code their port retain the safe single-world fallback.
            if port and not (is_livekit and service_name == control_name):
                configurable_port_names = (
                    "PORT",
                    "HTTP_PORT",
                    "SERVER_PORT",
                    "UVICORN_PORT",
                )
                configured_port = next(
                    (
                        name
                        for name in configurable_port_names
                        if process.environment.get(name) == str(port)
                    ),
                    None,
                )
                if configured_port is not None:
                    process = process.model_copy(
                        update={
                            "environment": {
                                **process.environment,
                                configured_port: f"{{{{PORT_{service_name}}}}}",
                            },
                            "fixed_port": None,
                        }
                    )
            if is_livekit and service_name == control_name:
                # The LiveKit worker opens its HTTP health port before it has registered with
                # the dispatch service.  Treating the port as readiness creates a race where a
                # named dispatch is submitted in that gap; self-hosted LiveKit leaves that
                # dispatch unassigned even after the worker subsequently registers.  The worker
                # log is the first observable signal that it can actually accept the call.
                process = process.model_copy(
                    update={
                        "fixed_port": _livekit_cli_fixed_health_port(
                            process.run_command, process.environment, service_root, root
                        )
                        or process.fixed_port,
                        "started_check": StartedCheck(
                            log_marker="registered worker", timeout_seconds=180
                        ),
                    }
                )
            processes.append(process)
            if port:
                slug = "target_http" if service_name == control_name else "tools_api"
                config = (
                    "TARGET_HTTP_URL"
                    if service_name == control_name
                    else "TOOLS_UPSTREAM_URL"
                )
                capabilities[slug] = CapabilityV2(
                    protocol=CapabilityProtocol.HTTP,
                    service=service_name,
                    container_port=port,
                    configuration_name=config,
                )
                readiness.append(
                    ReadinessProbeV2(
                        capability=slug, path="/health", timeout_seconds=180
                    )
                )
        if "tools-api" in source_services:
            processes.append(_tool_proxy_process())
            # The target must not become eligible to start until the evidence proxy is ready.
            # Depending only on the upstream tools process leaves a race where the agent starts
            # with TOOLS_API_URL pointing at a port that has not been bound yet.
            rewritten: list[ManagedProcess | SourceProcess] = []
            for process in processes:
                if isinstance(process, SourceProcess) and process.name == control_name:
                    dependencies = [
                        "tool-proxy" if item == "tools-api" else item
                        for item in process.depends_on
                    ]
                    if "tool-proxy" not in dependencies:
                        dependencies.append("tool-proxy")
                    process = process.model_copy(update={"depends_on": dependencies})
                rewritten.append(process)
            processes = rewritten
            capabilities["tool_proxy"] = CapabilityV2(
                protocol=CapabilityProtocol.HTTP,
                service="tool-proxy",
                container_port=8080,
                configuration_name="TOOLS_API_URL",
            )
            readiness.append(
                ReadinessProbeV2(
                    capability="tool_proxy", path="/health", timeout_seconds=180
                )
            )
        packaging = "compose"
    else:
        contract_is_callback = (contract_interface_kind or "").strip().lower().replace(
            "-", "_"
        ) == "callable"
        discovered_callback = (
            None if is_livekit else _discover_callback_entrypoint(root)
        )
        graph_entrypoint = (
            _langgraph_entrypoint(root)
            if not is_livekit and contract_modality == "chat" and not runtime_command
            else None
        )
        is_graph = graph_entrypoint is not None and not discovered_callback
        is_callback = (
            not is_livekit
            and not is_graph
            and (contract_is_callback or discovered_callback is not None)
        )
        is_command_adapter = bool(
            not is_livekit
            and not is_callback
            and contract_modality == "chat"
            and runtime_command
            and (contract_interface_kind or "") in {"", "command"}
        )
        is_http_runtime = bool(
            not is_livekit
            and not is_callback
            and contract_modality == "chat"
            and runtime_command
            and contract_interface_kind == "http"
        )
        is_declared_runtime = is_command_adapter or is_http_runtime
        entry = "agent.py"
        if is_declared_runtime or is_graph:
            component = _runtime_component(root, runtime_workdir)
            entry = next(
                (
                    item
                    for item in runtime_command
                    if item.endswith(".py") and (component / item).is_file()
                ),
                "agent.py",
            )
        elif not is_callback and not (root / entry).is_file():
            # A submitted checkout may contain a developer's virtualenv or dependency tree.
            # Those files are not agent entrypoints and must not make source discovery
            # ambiguous.  Apply the same generated-artifact exclusions used by staging.
            candidates = sorted(
                path
                for path in root.glob("**/agent.py")
                if not any(
                    part in _IGNORED_ARTIFACT_PARTS
                    for part in path.relative_to(root).parts[:-1]
                )
            )
            if len(candidates) != 1:
                raise BundleAuthorError(
                    "component_ambiguous: expected exactly one agent.py"
                )
            component = candidates[0].parent
            # An entrypoint directory is not necessarily its Python project root. Preserve
            # the nearest enclosing manifest and its sibling packages instead of flattening
            # src/ and silently running without the repository's dependencies.
            for parent in (component, *component.parents):
                if not parent.is_relative_to(root):
                    break
                if any(
                    (parent / name).is_file()
                    for name in ("pyproject.toml", "requirements.txt")
                ):
                    component = parent
                    break
            entry = candidates[0].relative_to(component).as_posix()
        else:
            component = root
        control_name = "agent"
        port = None if is_livekit else int(interface_port or 8080)
        environment = (
            {
                **declared_runtime_environment,
                "LIVEKIT_AGENT_NAME": (
                    root.name.replace("_", "-") + "-{{JOB_ID}}-w{{WORLD_INDEX}}"
                ),
                "HARNESS_TOOL_TRACE": "{{WORLD_DIR}}/agent-tool-calls.jsonl",
                # C1 §4: the single LiveKit worker carries the worker knob, fed its own token.
                **_worker_knob_env(control_name),
            }
            if is_livekit
            else dict(declared_runtime_environment)
        )
        callback_entrypoint = (
            discovered_callback or _callback_entrypoint(root) if is_callback else None
        )
        if is_graph:
            environment.update(
                {
                    "PORT": "{{PORT_agent}}",
                    "ALK_LANGGRAPH_ENTRYPOINT": graph_entrypoint,
                }
            )
        if callback_entrypoint:
            environment.update(
                {
                    "PORT": "{{PORT_agent}}",
                    "ALK_CALLBACK_ENTRYPOINT": callback_entrypoint,
                }
            )
        process = _plan_python(
            root,
            name=control_name,
            root=root if is_callback else component,
            entry=entry,
            control=True,
            needs_secrets=needs_target_secrets,
            port=port,
            environment=environment,
            livekit_download=is_livekit,
            run_override=(
                None
                if is_callback or is_command_adapter or is_graph
                else runtime_command or _dockerfile_run(component)
            ),
        )
        if is_graph:
            python_command = process.run_command[:-1]
            process = process.model_copy(
                update={
                    "run_command": python_command + ["-c", _langgraph_adapter_source()],
                    "started_check": StartedCheck(port=True, timeout_seconds=180),
                }
            )
        if is_callback:
            python_command = process.run_command[:-1]
            process = process.model_copy(
                update={
                    "run_command": python_command + ["-c", _callback_adapter_source()],
                    "started_check": StartedCheck(port=True, timeout_seconds=180),
                }
            )
        if is_command_adapter:
            command = _submitted_command(process, runtime_command)
            python_command = process.run_command[:-1]
            process = process.model_copy(
                update={
                    "run_command": python_command
                    + ["-c", _subprocess_adapter_source()],
                    "environment": {
                        **process.environment,
                        "PORT": "{{PORT_agent}}",
                        "ALK_SUBPROCESS_COMMAND": json.dumps(command),
                    },
                    "started_check": StartedCheck(port=True, timeout_seconds=180),
                }
            )
        if is_livekit:
            update: dict[str, Any] = {
                "started_check": StartedCheck(
                    log_marker="registered worker", timeout_seconds=180
                )
            }
            # Only a Dockerfile CMD carries the subcommand today, so a repository without one
            # starts `agent.py` bare and never reaches the registration this check waits for.
            if _hands_off_to_livekit_cli(component, entry) and not (
                set(process.run_command) & _LIVEKIT_WORKER_SUBCOMMANDS
            ):
                update["run_command"] = [*process.run_command, "start"]
            final_run_command = update.get("run_command", process.run_command)
            update["fixed_port"] = _livekit_cli_fixed_health_port(
                final_run_command, process.environment, component, root
            )
            process = process.model_copy(update=update)
        processes.append(process)
        if port:
            capabilities["target_http"] = CapabilityV2(
                protocol=CapabilityProtocol.HTTP,
                service=control_name,
                container_port=port,
                configuration_name="TARGET_HTTP_URL",
            )
            readiness.append(
                ReadinessProbeV2(
                    capability="target_http",
                    path=(
                        "/health"
                        if is_command_adapter or is_callback
                        else interface_health_path
                    ),
                    timeout_seconds=180,
                )
            )
        packaging = (
            "dockerfile" if (root / "Dockerfile").is_file() else "generated_python"
        )

    return EnvironmentPlanV2(
        packaging=packaging,
        control_service=control_name,
        processes=tuple(processes),
        capabilities=capabilities,
        readiness=tuple(readiness),
    )


def _copy_scenarios(authoring: Path, staging: Path, *, count: int) -> None:
    source = authoring / "scenarios"
    if not source.is_dir():
        raise BundleAuthorError(f"scenario_artifacts_missing: {source}")
    target = staging / "scenarios"
    target.mkdir()
    folders = sorted(path for path in source.iterdir() if path.is_dir())
    if len(folders) < count:
        raise BundleAuthorError(
            f"scenario_artifacts_insufficient: requested {count}, found {len(folders)}"
        )
    for source_folder in folders[:count]:
        shutil.copytree(source_folder, target / source_folder.name)
    for folder in sorted(path for path in target.iterdir() if path.is_dir()):
        document = folder / "scenario.json"
        if not document.is_file():
            continue
        body = json.loads(document.read_text(encoding="utf-8"))
        body["scenario_key"] = str(
            body.get("scenario_key") or body.get("name") or folder.name
        )
        body["scenario_id"] = str(body.get("scenario_id") or "")
        document.write_text(
            json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _copy_sub_goal_catalogue(authoring: Path, staging: Path) -> list[str]:
    """Put the sub-goal catalogue beside the scenarios that name its entries.

    Scenarios reference sub-goals by name only, so without the catalogue a description, a judged
    sub-goal's claim and `_deterministic_names` all come back empty, each silently. A warning
    rather than an error, since failing the run is worse than the degraded reporting.
    """
    catalogue = authoring / CATALOGUE
    if not catalogue.is_file():
        logger.warning(
            "no %s in %s: sub-goals will reach the platform without their descriptions or claims",
            CATALOGUE,
            authoring,
        )
        return []
    shutil.copy2(catalogue, staging / CATALOGUE)
    return [CATALOGUE]


def _copy_chat_authoring(authoring: Path, staging: Path) -> list[str]:
    """Adopt the frozen target/tool contract needed by response-carried HTTP tools.

    These are authoring outputs, not repository inference performed by the hosted consumer. The
    producer validates and seals them exactly like scenario code. Voice bundles legitimately have
    none; HTTP chat bundles require a contract at pre-dial time and fail there with a typed error.
    """
    adopted: list[str] = []
    contract = authoring / "contract.json"
    if contract.is_file():
        shutil.copy2(contract, staging / "contract.json")
        adopted.append("contract.json")
    handlers = authoring / "handlers"
    if handlers.is_dir():
        shutil.copytree(handlers, staging / "handlers")
        adopted.append("handlers/")
    prompt = authoring / "simulator_prompt.md"
    if prompt.is_file():
        shutil.copy2(prompt, staging / "simulator_prompt.md")
        adopted.append("simulator_prompt.md")
    return adopted


def _compile_source_tool_handlers(contract: dict[str, Any], staging: Path) -> list[str]:
    """Seal bindings for caller-executed tools that live in the submitted source.

    HTTP/chat agents can return a tool request for the harness caller to execute.  Contract
    discovery already records the repository's real import/construct entrypoint; hosted bundle
    authoring must carry that binding into the guest just as local world authoring does.  This
    compiles only recorded source entrypoints and never supplies a replacement implementation.
    Explicit authoring handlers win, which preserves bindings that needed custom invocation code.
    """
    raw_entries = contract.get("tool_entrypoints")
    if not isinstance(raw_entries, list):
        return []
    handlers = staging / "handlers"
    written: list[str] = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            continue
        try:
            entry = ToolEntry.model_validate(raw)
        except ValueError as exc:
            raise BundleAuthorError(f"contract_tool_entry_invalid: {exc}") from exc
        if entry.mode not in {"import", "construct"}:
            continue
        if not entry.module or not entry.callable:
            raise BundleAuthorError(
                f"contract_tool_entry_incomplete: {entry.tool}: "
                f"{entry.mode} requires module and callable"
            )
        # Provider/framework tool names are display names, not Python identifiers (CrewAI,
        # for example, permits spaces). The handler is loaded by exact filename, not imported
        # as a module. Reject path/control characters while preserving that exact display name.
        if not re.fullmatch(
            r"[A-Za-z0-9_][A-Za-z0-9_. -]{0,127}", entry.tool
        ) or entry.tool in {".", ".."}:
            raise BundleAuthorError(f"contract_tool_name_unsafe: {entry.tool!r}")
        handlers.mkdir(parents=True, exist_ok=True)
        destination = handlers / f"{entry.tool}.py"
        if destination.exists():
            continue
        destination.write_text(
            _binding(
                module=entry.module,
                called=entry.callable,
                style="method" if entry.mode == "construct" else "function",
                first_arg=entry.first_arg,
                factory=entry.factory,
            ),
            encoding="utf-8",
        )
        written.append(f"handlers/{entry.tool}.py")
    return written


def _files(root: Path) -> list[BundleFileV2]:
    records: list[BundleFileV2] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() or path.name == BUNDLE_V2_MANIFEST:
            continue
        if path.is_symlink():
            raise BundleAuthorError(
                f"bundle_symlink_forbidden: {path.relative_to(root)}"
            )
        relative = path.relative_to(root)
        if any(part in _IGNORED_ARTIFACT_PARTS for part in relative.parts):
            continue
        content = path.read_bytes()
        records.append(
            BundleFileV2(
                path=relative.as_posix(),
                sha256=hashlib.sha256(content).hexdigest(),
                size=len(content),
            )
        )
    return records


def author_bundle_v2(
    *,
    source: str | Path,
    job: HarnessJob,
    authoring: str | Path,
    output: str | Path,
) -> EnvironmentBundleV2:
    source_root = Path(source).resolve()
    authoring_root = Path(authoring).resolve()
    output_root = Path(output).resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if job.metadata.get("generic_harness_v1") is True:
        certified = _reuse_certified_bundle(
            source=source_root,
            job=job,
            authoring=authoring_root,
            output=output_root,
        )
        if certified is not None:
            return certified
    contract_modality: str | None = None
    contract_interface_kind: str | None = None
    contract_body: dict[str, Any] = {}
    runtime: dict[str, Any] = {}
    command_adapter_contract = False
    contract_path = authoring_root / "contract.json"
    if contract_path.is_file():
        try:
            contract_body = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BundleAuthorError(
                f"contract_invalid: cannot read {contract_path}: {exc}"
            ) from exc
        if not isinstance(contract_body, dict):
            raise BundleAuthorError("contract_invalid: contract.json must be an object")
        contract_modality = str(contract_body.get("modality") or "").strip().lower()
        runtime = contract_body.get("runtime")
        runtime = runtime if isinstance(runtime, dict) else {}
        interface = runtime.get("interface") if isinstance(runtime, dict) else None
        if isinstance(interface, dict):
            contract_interface_kind = str(interface.get("kind") or "").strip().lower()
        elif (
            contract_modality == "chat"
            and _discover_callback_entrypoint(source_root) is not None
        ):
            # The callback is a deterministic source property.  Do not let a stochastic
            # authoring omission make the compiled adapter unreachable at call time: the
            # environment plan already discovers and exposes this same callback, so seal the
            # matching interface into the bundle's contract as part of compilation.
            runtime = dict(runtime) if isinstance(runtime, dict) else {}
            runtime["interface"] = {
                "kind": "callable",
                "protocol": "fi.alk",
                "path": "",
                "health_path": "",
                "include_tools": True,
            }
            contract_body = {**contract_body, "runtime": runtime}
            contract_interface_kind = "callable"
        elif (
            contract_modality == "chat"
            and not isinstance(interface, dict)
            and not runtime.get("command")
            and _langgraph_entrypoint(source_root) is not None
        ):
            runtime = dict(runtime)
            runtime["interface"] = {
                "kind": "callable",
                "protocol": "fi.alk",
                "path": "",
                "health_path": "",
                "include_tools": False,
            }
            contract_body = {**contract_body, "runtime": runtime}
            contract_interface_kind = "callable"
        elif (
            contract_modality == "chat"
            and not isinstance(interface, dict)
            and runtime.get("command")
        ):
            # A runnable one-shot command is a real source-owned execution boundary even when it
            # is not a server. Compile the generic stdin/environment subprocess bridge below and
            # expose that bridge to the chat runner as the standard callable protocol.
            runtime = dict(runtime)
            runtime["interface"] = {
                "kind": "command",
                "protocol": "fi.alk",
                "path": "",
                "health_path": "",
                "include_tools": False,
            }
            contract_body = {**contract_body, "runtime": runtime}
            contract_interface_kind = "command"
            command_adapter_contract = True
        if (
            contract_modality == "chat"
            and contract_interface_kind == "callable"
            and runtime.get("command")
            and _discover_callback_entrypoint(source_root) is None
        ):
            # Authoring models sometimes infer ``callable`` from an in-process agent object even
            # though the repository exports no harness callback.  The executable command is the
            # stronger, source-verifiable boundary in that case.  Compile it through the generic
            # command bridge instead of starting a one-shot script and waiting for an HTTP port it
            # can never open.  This rule is framework-neutral and does not modify submitted code.
            runtime = dict(runtime)
            runtime["interface"] = {
                "kind": "command",
                "protocol": "fi.alk",
                "path": "",
                "health_path": "",
                "include_tools": False,
            }
            contract_body = {**contract_body, "runtime": runtime}
            contract_interface_kind = "command"
            command_adapter_contract = True
    plan = resolve_environment_plan(
        source_root,
        job,
        contract_modality=contract_modality,
        contract_interface_kind=contract_interface_kind,
        contract_runtime=(runtime if isinstance(runtime, dict) else None),
    )
    # In-process framework tools do not cross the HTTP tool proxy. Trace only source-declared
    # callable boundaries inside the target Python process; the trace is observational and
    # never replays the tool. The same mechanism works for any Python framework with an
    # importable callable recorded during source understanding.
    trace_bindings = [
        {"name": entry.tool, "module": entry.module, "callable": entry.callable}
        for raw in contract_body.get("tool_entrypoints", [])
        if isinstance(raw, dict)
        for entry in [ToolEntry.model_validate(raw)]
        if entry.mode in {"import", "construct"} and entry.module and entry.callable
    ]
    if contract_modality == "chat" and trace_bindings:
        plan = replace(
            plan,
            processes=tuple(
                process.model_copy(
                    update={
                        "environment": {
                            **process.environment,
                            "HARNESS_TOOL_TRACE": "{{WORLD_DIR}}/agent-tool-calls.jsonl",
                            "ALK_TOOL_TRACE_BINDINGS": json.dumps(trace_bindings),
                        }
                    }
                )
                if isinstance(process, SourceProcess)
                and process.name == plan.control_service
                else process
                for process in plan.processes
            ),
        )
    if command_adapter_contract:
        sealed_runtime = dict(contract_body["runtime"])
        sealed_runtime["interface"] = {
            "kind": "callable",
            "protocol": "fi.alk",
            "path": "",
            "health_path": "",
            "include_tools": False,
        }
        contract_body = {**contract_body, "runtime": sealed_runtime}
    provided_environment = {
        str(name).upper()
        for name in (job.metadata.get("environment_value_names", []) or [])
    }
    provided_environment.update(_declared_runtime_environment(source_root))
    for process in plan.processes:
        provided_environment.update(
            str(name).upper() for name in (getattr(process, "environment", None) or {})
        )
    credential_manifest = discover_credentials(
        source_root,
        secret_refs=job.agent.secret_refs,
        provided_environment=provided_environment,
        scan_paths={
            str(getattr(process, "working_directory", ".") or ".")
            for process in plan.processes
            if isinstance(process, SourceProcess)
        },
        # Match preflight: a template documents possible integrations, not every
        # credential needed by the selected generic-runtime path.
        template_secrets_required=job.metadata.get("generic_harness_v1") is not True,
    )
    if not credential_manifest.ready:
        missing = sorted(
            item.environment_name for item in credential_manifest.missing_required
        )
        unsatisfied = sorted(
            choice.id
            for choice in credential_manifest.credential_choices
            if not choice.satisfied
        )
        details = [*(f"environment:{name}" for name in missing)]
        details.extend(f"credential_choice:{name}" for name in unsatisfied)
        raise BundleAuthorError(
            "target_runtime_configuration_missing: " + ", ".join(details)
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        _copy_scenarios(authoring_root, temporary, count=job.scenario_count)
        adopted_catalogue = _copy_sub_goal_catalogue(authoring_root, temporary)
        adopted_chat_files = _copy_chat_authoring(authoring_root, temporary)
        if "contract.json" in adopted_chat_files and contract_body:
            (temporary / "contract.json").write_text(
                json.dumps(contract_body, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        adopted_chat_files.extend(
            _compile_source_tool_handlers(contract_body, temporary)
        )
        if any(process.name == "tool-proxy" for process in plan.processes):
            generated = temporary / "generated" / "tool-proxy"
            generated.mkdir(parents=True)
            shutil.copy2(
                Path(__file__).with_name("tool_trace_proxy.py"),
                generated / "proxy.py",
            )
        seed_dir = temporary / "seed"
        seed_dir.mkdir()
        prefix = (
            "CREATE TABLE IF NOT EXISTS harness_seed_sentinel (id text PRIMARY KEY);\n"
            "INSERT INTO harness_seed_sentinel(id) VALUES ('ready') ON CONFLICT DO NOTHING;\n"
            "CREATE TABLE IF NOT EXISTS _alk_tool_trace ("
            "id bigserial PRIMARY KEY, name text NOT NULL, arguments jsonb NOT NULL, "
            "result jsonb, ok boolean NOT NULL, error text, at double precision NOT NULL);\n"
        )
        generic_pipeline = job.metadata.get("generic_harness_v1") is True
        if generic_pipeline:
            migrations, seed_files, adopted_seed = _generic_postgres_seed_artifacts(
                authoring_root,
                temporary,
                source=source_root,
                contract=contract_body,
                prefix=prefix,
                allow_harness_owned_schema=(
                    command_adapter_contract
                    or job.agent.mode
                    in {
                        ProviderExecutionMode.CONNECT_ONLY,
                        ProviderExecutionMode.ENVIRONMENT_BACKED,
                        ProviderExecutionMode.PROVIDER_IMPORT,
                    }
                ),
            )
        else:
            seed_path = seed_dir / "world.sql"
            schema, adopted_seed = _adopted_seed_sql(
                authoring_root,
                source=source_root,
                contract=contract_body,
            )
            seed_path.write_text(prefix + schema, encoding="utf-8")
            migrations = ["seed/world.sql"]
            seed_files = []
        store = StoreEntry(
            capability="world_db",
            migrations=migrations,
            seed_files=seed_files,
            baseline=StoreBaseline(
                strategy=BaselineStrategy.TEMPLATE_DATABASE,
                inputs_digest=compute_inputs_digest(
                    temporary,
                    migrations,
                    seed_files,
                    engine=ManagedEngine.POSTGRES,
                    version="16",
                ),
            ),
            sentinel=Sentinel(
                query="SELECT id FROM harness_seed_sentinel WHERE id='ready'",
                expected="ready",
            ),
        )
        provider_manifest: ProviderRepositoryManifest | None = None
        provider_import: ProviderImportSpec | None = None
        if job.agent.mode is ProviderExecutionMode.ENVIRONMENT_BACKED:
            provider_manifest = load_provider_manifest(
                source_root,
                str(job.agent.config.get("lifecycle_manifest") or "alk.yaml"),
            )
            declared = set(provider_manifest.provider.required_secrets)
            supplied = set(job.agent.secret_refs)
            missing = sorted(declared - supplied)
            if missing:
                raise BundleAuthorError(
                    "provider_lifecycle_secrets_missing: " + ", ".join(missing)
                )
        elif job.agent.mode is ProviderExecutionMode.PROVIDER_IMPORT:
            connector = job.agent.connector.strip().lower()
            provider = "retell" if connector == "retell_chat" else connector
            secret_name = "VAPI_API_KEY" if provider == "vapi" else "RETELL_API_KEY"
            if secret_name not in job.agent.secret_refs:
                raise BundleAuthorError(
                    f"provider_import_secret_missing: {secret_name}"
                )
            configured_capability = str(
                job.agent.config.get("public_capability") or ""
            ).strip()
            http_capabilities = sorted(
                name
                for name, capability in plan.capabilities.items()
                if capability.protocol.value == "http"
            )
            if configured_capability:
                if configured_capability not in http_capabilities:
                    raise BundleAuthorError(
                        "provider_import_public_capability_invalid: "
                        f"{configured_capability!r} is not an HTTP capability"
                    )
                public_capability = configured_capability
            elif len(http_capabilities) == 1:
                public_capability = http_capabilities[0]
            else:
                raise BundleAuthorError(
                    "provider_import_public_capability_ambiguous: configure public_capability; "
                    f"found {http_capabilities}"
                )
            target_key = "assistant_id" if provider == "vapi" else "agent_id"
            provider_import = ProviderImportSpec(
                type=provider,
                source_target_id=str(job.agent.config[target_key]),
                public_capability=public_capability,
                environment_tools=sorted(
                    {
                        str(tool.get("name") or "").strip()
                        for tool in contract_body.get("tools", [])
                        if isinstance(tool, dict)
                        and str(tool.get("name") or "").strip()
                    }
                ),
                event_path=str(
                    job.agent.config.get("event_path") or "/provider/events"
                ),
                tool_path=str(job.agent.config.get("tool_path") or "/provider/tools"),
                api_base_url=str(job.agent.config.get("provider_api_base_url") or "")
                or None,
                target_modality="chat" if connector == "retell_chat" else "voice",
            )

        manifest = EnvironmentBundleV2(
            schema_version=BUNDLE_V2_SCHEMA_VERSION,
            digest="sha256:" + "0" * 64,
            name=str(job.metadata.get("name") or source_root.name),
            runtime=BundleRuntimeV2(
                kind=RuntimeKindV2.PROCESS,
                control_service=plan.control_service,
                evidence_seam=EvidenceSeam.TOOL_TRACE,
            ),
            processes=list(plan.processes),
            seed=Seed(stores=[store]),
            capabilities=plan.capabilities,
            readiness=list(plan.readiness),
            files=_files(temporary),
            provenance=BundleProvenanceV2(
                source_kind=job.source.kind.value,
                repository=job.source.repository,
                commit=job.source.commit_sha,
                source_digest=source_fingerprint(source_root),
                generator="fi.alk.harness.bundle_author_v2",
                generator_version="2",
                adopted_files=["scenarios/"]
                + adopted_catalogue
                + adopted_seed
                + adopted_chat_files,
                generated_files=["manifest.json"]
                + (
                    [*migrations, *seed_files]
                    if generic_pipeline
                    else ["seed/world.sql"]
                ),
            ),
            metadata={
                "packaging": plan.packaging,
                "environment_plan_version": "2",
                **({"generic_harness": "v1"} if generic_pipeline else {}),
                **(
                    {
                        "provider_connect_only": {
                            "connector": job.agent.connector.strip().lower()
                        }
                    }
                    if job.agent.mode is ProviderExecutionMode.CONNECT_ONLY
                    and job.source.kind is SourceKind.PROVIDER
                    else {}
                ),
                **(
                    {
                        "provider_lifecycle": provider_manifest.provider.model_dump(
                            mode="json"
                        )
                    }
                    if provider_manifest is not None
                    else {}
                ),
                **(
                    {"provider_import": provider_import.model_dump(mode="json")}
                    if provider_import is not None
                    else {}
                ),
                "environment_plan_hash": hashlib.sha256(
                    json.dumps(
                        {
                            "packaging": plan.packaging,
                            "control_service": plan.control_service,
                            "processes": [
                                item.model_dump(mode="json") for item in plan.processes
                            ],
                            "capabilities": {
                                key: value.model_dump(mode="json")
                                for key, value in plan.capabilities.items()
                            },
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            },
        )
        manifest = manifest.model_copy(update={"digest": seal_bundle_v2(manifest)})
        (temporary / BUNDLE_V2_MANIFEST).write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        loaded = load_bundle_v2(temporary)
        preflight_bundle(
            temporary,
            loaded,
            parallelism=job.runtime.parallelism,
            secret_refs={
                alias: reference.purpose
                for alias, reference in job.agent.secret_refs.items()
            },
        )
        if output_root.exists():
            backup = output_root.with_name(output_root.name + ".previous")
            if backup.exists():
                shutil.rmtree(backup)
            output_root.rename(backup)
            temporary.rename(output_root)
            shutil.rmtree(backup)
        else:
            temporary.rename(output_root)
        # The certificate binds to the sealed bundle digest, so it must remain a control-plane
        # sidecar rather than becoming a manifest-listed file (which would create a digest cycle).
        # Production compilation receives this file in the frozen authoring archive and places it
        # beside /work/bundle for the hosted entrypoint's mandatory pre-call gate.
        certificate = authoring_root / "runtime-validation.json"
        if generic_pipeline and certificate.is_file() and not certificate.is_symlink():
            shutil.copy2(certificate, output_root.parent / "runtime-validation.json")
        return loaded
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_job(path: Path) -> HarnessJob:
    return HarnessJob.model_validate_json(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alk-bundle-author-v2")
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--authoring", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    author_bundle_v2(
        source=args.source,
        job=_load_job(args.job),
        authoring=args.authoring,
        output=args.output,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


def _schema_column_types(paths: list[Path]) -> dict[tuple[str, str], str]:
    """Column declarations read off the agent's own schema, keyed by table and column."""
    declared: dict[tuple[str, str], str] = {}
    for path in paths:
        sql = re.sub(r"--[^\n]*", "", path.read_text(encoding="utf-8"))
        for found in re.finditer(
            r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"?([A-Za-z_]\w*)"?\s*\(',
            sql,
            re.IGNORECASE,
        ):
            depth, at = 1, found.end()
            while at < len(sql) and depth:
                depth += (sql[at] == "(") - (sql[at] == ")")
                at += 1
            for part in re.split(r",(?![^()]*\))", sql[found.end() : at - 1]):
                words = part.strip().split()
                if len(words) < 2:
                    continue
                name = words[0].strip('"')
                if name.upper() in (
                    "PRIMARY",
                    "FOREIGN",
                    "UNIQUE",
                    "CHECK",
                    "CONSTRAINT",
                ):
                    continue
                kind = words[1].upper()
                rest = " ".join(words[1:]).upper()
                if kind.startswith("BOOL"):
                    declared[(found.group(1).lower(), name.lower())] = "boolean"
                elif "[]" in rest or kind.startswith("ARRAY"):
                    declared[(found.group(1).lower(), name.lower())] = "text[]"
                elif kind.startswith("JSONB"):
                    declared[(found.group(1).lower(), name.lower())] = "jsonb"
    return declared
