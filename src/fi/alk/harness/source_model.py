"""Canonical, credential-free facts discovered from submitted source.

The source model is an immutable boundary between discovery and authoring.  Adapters preserve
backend-native metadata while exposing enough logical structure for deterministic compilation.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, model_validator

SOURCE_MODEL_SCHEMA_VERSION = "futureagi.source-model.v2"
_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


class LogicalType(str, Enum):
    BOOLEAN = "boolean"
    INTEGER = "integer"
    NUMBER = "number"
    STRING = "string"
    UUID = "uuid"
    TIMESTAMP = "timestamp"
    DATE = "date"
    ENUM = "enum"
    ARRAY = "array"
    JSON = "json"
    BINARY = "binary"


class ForeignKey(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    columns: tuple[str, ...]
    referenced_table: str
    referenced_columns: tuple[str, ...]
    on_update: str | None = None
    on_delete: str | None = None
    deferrable: bool = False

    @model_validator(mode="after")
    def _complete_key(self) -> "ForeignKey":
        if not self.columns or len(self.columns) != len(self.referenced_columns):
            raise ValueError("source_foreign_key_columns_invalid")
        return self


class CheckConstraint(BaseModel):
    """A source-owned predicate retained for provenance and database enforcement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    expression: str = Field(min_length=1)


class SourceColumn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    logical_type: LogicalType
    # Empty is a real SQLite declaration (a column may be declared without a type), so it must
    # remain distinct from TEXT rather than being filled with a convenient default.
    native_type: str
    nullable: bool
    has_default: bool
    default_expression: str | None = None
    generated: bool = False
    identity: bool = False
    enum_values: tuple[str, ...] = ()
    element_type: LogicalType | None = None

    @model_validator(mode="after")
    def _type_metadata_is_consistent(self) -> "SourceColumn":
        if self.has_default != (self.default_expression is not None):
            raise ValueError("source_column_default_metadata_inconsistent")
        if self.logical_type is LogicalType.ENUM and not self.enum_values:
            raise ValueError("source_enum_values_missing")
        if self.logical_type is not LogicalType.ENUM and self.enum_values:
            raise ValueError("source_enum_values_not_allowed")
        if self.logical_type is LogicalType.ARRAY and self.element_type is None:
            raise ValueError("source_array_element_type_missing")
        if (
            self.logical_type is LogicalType.ARRAY
            and self.element_type is LogicalType.ARRAY
        ):
            raise ValueError("source_array_element_type_must_be_scalar")
        if self.logical_type is not LogicalType.ARRAY and self.element_type is not None:
            raise ValueError("source_array_element_type_not_allowed")
        return self


class SourceTable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    columns: tuple[SourceColumn, ...]
    primary_key: tuple[str, ...] = ()
    unique_keys: tuple[tuple[str, ...], ...] = ()
    foreign_keys: tuple[ForeignKey, ...] = ()
    check_constraints: tuple[CheckConstraint, ...] = ()

    @model_validator(mode="after")
    def _keys_reference_columns(self) -> "SourceTable":
        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("source_table_column_names_not_unique")
        known = set(names)
        referenced = [*self.primary_key]
        referenced.extend(column for key in self.unique_keys for column in key)
        referenced.extend(column for key in self.foreign_keys for column in key.columns)
        unknown = sorted(set(referenced) - known)
        if unknown:
            raise ValueError("source_table_key_column_unknown: " + ", ".join(unknown))
        if any(not key for key in self.unique_keys):
            raise ValueError("source_unique_key_empty")
        if self.check_constraints != tuple(
            sorted(self.check_constraints, key=lambda item: item.name)
        ):
            raise ValueError("source_check_constraints_not_canonical")
        check_names = [item.name for item in self.check_constraints]
        if len(check_names) != len(set(check_names)):
            raise ValueError("source_check_constraint_name_duplicate")
        return self


class SourceEvidence(BaseModel):
    """Digest-only provenance for a source-owned file; never its contents."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    digest: str
    kind: str = Field(min_length=1)

    @model_validator(mode="after")
    def _portable_and_hashed(self) -> "SourceEvidence":
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("source_evidence_path_must_be_relative")
        if not _DIGEST_PATTERN.fullmatch(self.digest):
            raise ValueError("source_evidence_digest_invalid")
        return self


class UnsupportedSourceConstruct(BaseModel):
    """An explicit discovery gap; unsupported facts are never silently discarded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    component: str = Field(min_length=1)
    location: str | None = None


class SourceProcess(BaseModel):
    """One runnable source-owned component, independent of language or agent framework."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    working_directory: str = "."
    entrypoint: str | None = None
    command: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    configuration_names: tuple[str, ...] = ()
    ports: tuple[int, ...] = ()
    evidence: tuple[SourceEvidence, ...] = ()

    @model_validator(mode="after")
    def _canonical(self) -> "SourceProcess":
        if self.kind != self.kind.strip().lower():
            raise ValueError("source_process_kind_not_canonical")
        if self.entrypoint is None and not self.command:
            raise ValueError("source_process_launch_missing")
        for value, code in (
            (self.working_directory, "source_process_workdir_invalid"),
        ):
            path = PurePosixPath(value)
            if ".." in path.parts:
                raise ValueError(code)
        if self.entrypoint is not None:
            path = PurePosixPath(self.entrypoint)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("source_process_entrypoint_invalid")
        if any(not item.strip() for item in self.command):
            raise ValueError("source_process_command_invalid")
        if self.dependencies != tuple(sorted(set(self.dependencies))):
            raise ValueError("source_process_dependencies_not_canonical")
        if self.configuration_names != tuple(sorted(set(self.configuration_names))):
            raise ValueError("source_process_configuration_names_not_canonical")
        if self.ports != tuple(sorted(set(self.ports))) or any(
            port < 1 or port > 65535 for port in self.ports
        ):
            raise ValueError("source_process_ports_not_canonical")
        if self.evidence != tuple(sorted(self.evidence, key=lambda item: item.path)):
            raise ValueError("source_process_evidence_not_canonical")
        return self


class SourceInterface(BaseModel):
    """A callable/transport/UI boundary offered by submitted code.

    ``kind`` and ``protocol`` are intentionally extensible strings: HTTP, Python callables,
    queues, browser sessions, desktop sessions and future transports share this model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    protocol: str = Field(min_length=1)
    process: str | None = None
    endpoint: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    input_schema: dict[str, object] | None = None
    output_schema: dict[str, object] | None = None
    configuration_names: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _canonical(self) -> "SourceInterface":
        if self.kind != self.kind.strip().lower():
            raise ValueError("source_interface_kind_not_canonical")
        if self.protocol != self.protocol.strip().lower():
            raise ValueError("source_interface_protocol_not_canonical")
        if self.configuration_names != tuple(sorted(set(self.configuration_names))):
            raise ValueError("source_interface_configuration_names_not_canonical")
        if self.endpoint is not None:
            path = PurePosixPath(self.endpoint)
            if ".." in path.parts:
                raise ValueError("source_interface_endpoint_invalid")
        try:
            if self.input_schema is not None:
                json.dumps(self.input_schema, allow_nan=False, sort_keys=True)
            if self.output_schema is not None:
                json.dumps(self.output_schema, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("source_interface_schema_not_json") from exc
        return self


class SourceAction(BaseModel):
    """A source-owned action/tool and the evidence needed to reach it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    input_schema: dict[str, object] = Field(default_factory=dict)
    output_schema: dict[str, object] | None = None
    implementation_kind: str = Field(min_length=1)
    implementation_ref: str = Field(min_length=1)
    interface: str | None = None
    effect: str = "unknown"
    evidence: tuple[SourceEvidence, ...] = ()

    @model_validator(mode="after")
    def _canonical(self) -> "SourceAction":
        if self.implementation_kind != self.implementation_kind.strip().lower():
            raise ValueError("source_action_implementation_kind_not_canonical")
        if self.effect not in {"read_only", "mutating", "external", "unknown"}:
            raise ValueError("source_action_effect_invalid")
        if self.evidence != tuple(sorted(self.evidence, key=lambda item: item.path)):
            raise ValueError("source_action_evidence_not_canonical")
        try:
            json.dumps(self.input_schema, allow_nan=False, sort_keys=True)
            if self.output_schema is not None:
                json.dumps(self.output_schema, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("source_action_schema_not_json") from exc
        return self


class SourceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SOURCE_MODEL_SCHEMA_VERSION
    source_digest: str
    engine: str = Field(min_length=1)
    engine_version: str | None = None
    tables: tuple[SourceTable, ...] = ()
    processes: tuple[SourceProcess, ...] = ()
    interfaces: tuple[SourceInterface, ...] = ()
    actions: tuple[SourceAction, ...] = ()
    configuration_names: tuple[str, ...] = ()
    migrations: tuple[SourceEvidence, ...] = ()
    seeds: tuple[SourceEvidence, ...] = ()
    evidence: tuple[SourceEvidence, ...] = ()
    unsupported: tuple[UnsupportedSourceConstruct, ...] = ()
    fingerprint: str

    @classmethod
    def create(
        cls,
        *,
        source_digest: str,
        engine: str,
        engine_version: str | None = None,
        tables: tuple[SourceTable, ...] = (),
        processes: tuple[SourceProcess, ...] = (),
        interfaces: tuple[SourceInterface, ...] = (),
        actions: tuple[SourceAction, ...] = (),
        configuration_names: tuple[str, ...] = (),
        migrations: tuple[SourceEvidence, ...] = (),
        seeds: tuple[SourceEvidence, ...] = (),
        evidence: tuple[SourceEvidence, ...] = (),
        unsupported: tuple[UnsupportedSourceConstruct, ...] = (),
    ) -> "SourceModel":
        raw = {
            "schema_version": SOURCE_MODEL_SCHEMA_VERSION,
            "source_digest": source_digest,
            "engine": engine.strip().lower(),
            "engine_version": engine_version,
            "tables": tuple(sorted(tables, key=lambda item: item.name)),
            "processes": tuple(sorted(processes, key=lambda item: item.name)),
            "interfaces": tuple(sorted(interfaces, key=lambda item: item.name)),
            "actions": tuple(sorted(actions, key=lambda item: item.name)),
            "configuration_names": tuple(sorted(set(configuration_names))),
            "migrations": tuple(sorted(migrations, key=lambda item: item.path)),
            "seeds": tuple(sorted(seeds, key=lambda item: item.path)),
            "evidence": tuple(sorted(evidence, key=lambda item: item.path)),
            "unsupported": tuple(
                sorted(
                    unsupported,
                    key=lambda item: (item.code, item.component, item.location or ""),
                )
            ),
        }
        raw["fingerprint"] = _source_model_fingerprint(raw)
        return cls.model_validate(raw)

    @model_validator(mode="after")
    def _canonical_and_hashed(self) -> "SourceModel":
        if self.schema_version != SOURCE_MODEL_SCHEMA_VERSION:
            raise ValueError("source_model_schema_version_unsupported")
        if not _DIGEST_PATTERN.fullmatch(self.source_digest):
            raise ValueError("source_model_source_digest_invalid")
        if self.tables != tuple(sorted(self.tables, key=lambda item: item.name)):
            raise ValueError("source_model_tables_not_canonical")
        for collection, code in (
            (self.processes, "source_model_processes_not_canonical"),
            (self.interfaces, "source_model_interfaces_not_canonical"),
            (self.actions, "source_model_actions_not_canonical"),
        ):
            if collection != tuple(sorted(collection, key=lambda item: item.name)):
                raise ValueError(code)
            names = [item.name for item in collection]
            if len(names) != len(set(names)):
                raise ValueError(code.replace("not_canonical", "name_duplicate"))
        process_names = {process.name for process in self.processes}
        for process in self.processes:
            unknown = sorted(set(process.dependencies) - process_names)
            if unknown:
                raise ValueError(
                    "source_process_dependency_unknown: " + ", ".join(unknown)
                )
        visiting: set[str] = set()
        visited: set[str] = set()
        dependencies = {
            process.name: set(process.dependencies) for process in self.processes
        }

        def visit(name: str) -> None:
            if name in visiting:
                raise ValueError(f"source_process_dependency_cycle: {name}")
            if name in visited:
                return
            visiting.add(name)
            for dependency in sorted(dependencies[name]):
                visit(dependency)
            visiting.remove(name)
            visited.add(name)

        for name in sorted(process_names):
            visit(name)
        for interface in self.interfaces:
            if interface.process is not None and interface.process not in process_names:
                raise ValueError(
                    f"source_interface_process_unknown: {interface.process}"
                )
        interface_names = {interface.name for interface in self.interfaces}
        for action in self.actions:
            if action.interface is not None and action.interface not in interface_names:
                raise ValueError(f"source_action_interface_unknown: {action.interface}")
        if self.configuration_names != tuple(sorted(set(self.configuration_names))):
            raise ValueError("source_model_configuration_names_not_canonical")
        for collection, code in (
            (self.migrations, "source_model_migrations_not_canonical"),
            (self.seeds, "source_model_seeds_not_canonical"),
            (self.evidence, "source_model_evidence_not_canonical"),
        ):
            if collection != tuple(sorted(collection, key=lambda item: item.path)):
                raise ValueError(code)
            paths = [item.path for item in collection]
            if len(paths) != len(set(paths)):
                raise ValueError("source_model_evidence_path_duplicate")
        canonical_unsupported = tuple(
            sorted(
                self.unsupported,
                key=lambda item: (item.code, item.component, item.location or ""),
            )
        )
        if self.unsupported != canonical_unsupported:
            raise ValueError("source_model_unsupported_not_canonical")
        raw = self.model_dump(mode="python", exclude={"fingerprint"})
        if self.fingerprint != _source_model_fingerprint(raw):
            raise ValueError("source_model_fingerprint_mismatch")
        return self


def _source_model_fingerprint(raw: dict[str, object]) -> str:
    def jsonable(value: object) -> object:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, tuple):
            return [jsonable(item) for item in value]
        if isinstance(value, dict):
            return {str(key): jsonable(item) for key, item in value.items()}
        return value

    encoded = json.dumps(
        jsonable(raw),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CheckConstraint",
    "SOURCE_MODEL_SCHEMA_VERSION",
    "ForeignKey",
    "LogicalType",
    "SourceAction",
    "SourceColumn",
    "SourceEvidence",
    "SourceInterface",
    "SourceModel",
    "SourceProcess",
    "SourceTable",
    "UnsupportedSourceConstruct",
]
