"""Typed semantic world data, independent of any storage backend."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .source_model import LogicalType, SourceColumn, SourceModel

WORLD_IR_SCHEMA_VERSION = "futureagi.world-ir.v1"


class ValueState(str, Enum):
    ABSENT = "absent"
    NULL = "null"
    PRESENT = "present"


class WorldValue(BaseModel):
    """One authored value with omission kept distinct from database null."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: ValueState
    logical_type: LogicalType | None = None
    value: JsonValue | None = None

    @classmethod
    def absent(cls) -> "WorldValue":
        return cls(state=ValueState.ABSENT)

    @classmethod
    def null(cls, logical_type: LogicalType | None = None) -> "WorldValue":
        return cls(state=ValueState.NULL, logical_type=logical_type)

    @classmethod
    def present(cls, logical_type: LogicalType, value: JsonValue) -> "WorldValue":
        return cls(state=ValueState.PRESENT, logical_type=logical_type, value=value)

    @model_validator(mode="after")
    def _state_is_unambiguous(self) -> "WorldValue":
        if self.state is ValueState.ABSENT:
            if self.logical_type is not None or self.value is not None:
                raise ValueError("world_value_absent_has_payload")
        elif self.state is ValueState.NULL:
            if self.value is not None:
                raise ValueError("world_value_null_has_payload")
        elif self.logical_type is None or self.value is None:
            raise ValueError("world_value_present_payload_missing")
        return self


class WorldRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    identity: str = Field(min_length=1)
    values: dict[str, WorldValue]


class WorldTable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_name: str = Field(min_length=1)
    rows: tuple[WorldRow, ...]

    @model_validator(mode="after")
    def _row_identities_are_unique(self) -> "WorldTable":
        identities = [row.identity for row in self.rows]
        if len(identities) != len(set(identities)):
            raise ValueError("world_table_row_identity_duplicate")
        if identities != sorted(identities):
            raise ValueError("world_table_rows_not_canonical")
        return self


class WorldIR(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = WORLD_IR_SCHEMA_VERSION
    source_model_fingerprint: str
    tables: tuple[WorldTable, ...]
    fingerprint: str

    @classmethod
    def create(
        cls, *, source_model_fingerprint: str, tables: tuple[WorldTable, ...]
    ) -> "WorldIR":
        canonical_tables = tuple(sorted(tables, key=lambda item: item.source_name))
        raw: dict[str, Any] = {
            "schema_version": WORLD_IR_SCHEMA_VERSION,
            "source_model_fingerprint": source_model_fingerprint,
            "tables": canonical_tables,
        }
        raw["fingerprint"] = _fingerprint(raw)
        return cls.model_validate(raw)

    @model_validator(mode="after")
    def _canonical_and_hashed(self) -> "WorldIR":
        if self.schema_version != WORLD_IR_SCHEMA_VERSION:
            raise ValueError("world_ir_schema_version_unsupported")
        names = [table.source_name for table in self.tables]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("world_ir_tables_not_canonical")
        expected = _fingerprint(self.model_dump(mode="python", exclude={"fingerprint"}))
        if self.fingerprint != expected:
            raise ValueError("world_ir_fingerprint_mismatch")
        return self


class WorldIRIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    table: str
    row_identity: str | None = None
    column: str | None = None
    message: str


class WorldIRValidationError(ValueError):
    def __init__(self, issues: tuple[WorldIRIssue, ...]) -> None:
        self.issues = issues
        codes = ", ".join(issue.code for issue in issues)
        super().__init__(f"world_ir_invalid: {codes}")


def _fingerprint(raw: dict[str, Any]) -> str:
    def jsonable(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, tuple):
            return [jsonable(item) for item in value]
        if isinstance(value, dict):
            return {key: jsonable(value[key]) for key in sorted(value)}
        return value

    encoded = json.dumps(
        jsonable(raw),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _value_matches(logical_type: LogicalType, value: JsonValue) -> bool:
    if logical_type is LogicalType.BOOLEAN:
        return isinstance(value, bool)
    if logical_type is LogicalType.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if logical_type is LogicalType.NUMBER:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if logical_type in {LogicalType.STRING, LogicalType.ENUM}:
        return isinstance(value, str)
    if logical_type is LogicalType.UUID:
        if not isinstance(value, str):
            return False
        try:
            UUID(value)
        except ValueError:
            return False
        return True
    if logical_type is LogicalType.TIMESTAMP:
        if not isinstance(value, str):
            return False
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return False
        return parsed.tzinfo is not None
    if logical_type is LogicalType.DATE:
        if not isinstance(value, str):
            return False
        try:
            date.fromisoformat(value)
        except ValueError:
            return False
        return True
    if logical_type is LogicalType.ARRAY:
        return isinstance(value, list)
    if logical_type is LogicalType.JSON:
        return True
    if logical_type is LogicalType.BINARY:
        if not isinstance(value, str):
            return False
        try:
            base64.b64decode(value, validate=True)
        except ValueError:
            return False
        return True
    return False


def _issue(
    code: str,
    table: str,
    message: str,
    *,
    row: WorldRow | None = None,
    column: str | None = None,
) -> WorldIRIssue:
    return WorldIRIssue(
        code=code,
        table=table,
        row_identity=row.identity if row else None,
        column=column,
        message=message,
    )


def _validate_value(
    table: str,
    row: WorldRow,
    column: SourceColumn,
    authored: WorldValue,
) -> list[WorldIRIssue]:
    issues: list[WorldIRIssue] = []
    if column.generated and authored.state is not ValueState.ABSENT:
        return [
            _issue(
                "generated_column_authored",
                table,
                "generated columns must be absent",
                row=row,
                column=column.name,
            )
        ]
    if authored.state is ValueState.ABSENT:
        if not column.nullable and not column.has_default and not column.generated:
            issues.append(
                _issue(
                    "required_value_missing",
                    table,
                    "required column has no authored value or source default",
                    row=row,
                    column=column.name,
                )
            )
        return issues
    if authored.state is ValueState.NULL:
        if (
            authored.logical_type is not None
            and authored.logical_type is not column.logical_type
        ):
            issues.append(
                _issue(
                    "schema_type_mismatch",
                    table,
                    f"expected {column.logical_type.value}, got {authored.logical_type.value}",
                    row=row,
                    column=column.name,
                )
            )
        if not column.nullable:
            issues.append(
                _issue(
                    "explicit_null_not_allowed",
                    table,
                    "explicit null is not valid for a non-null column",
                    row=row,
                    column=column.name,
                )
            )
        return issues
    if authored.logical_type is not column.logical_type:
        return [
            _issue(
                "schema_type_mismatch",
                table,
                f"expected {column.logical_type.value}, got {authored.logical_type.value}",
                row=row,
                column=column.name,
            )
        ]
    assert authored.value is not None
    if not _value_matches(column.logical_type, authored.value):
        issues.append(
            _issue(
                "value_shape_mismatch",
                table,
                f"value does not match logical type {column.logical_type.value}",
                row=row,
                column=column.name,
            )
        )
        return issues
    if (
        column.logical_type is LogicalType.ENUM
        and authored.value not in column.enum_values
    ):
        issues.append(
            _issue(
                "enum_value_invalid",
                table,
                "value is not a member of the source enum",
                row=row,
                column=column.name,
            )
        )
    if column.logical_type is LogicalType.ARRAY and column.element_type is not None:
        assert isinstance(authored.value, list)
        if any(
            not _value_matches(column.element_type, item) for item in authored.value
        ):
            issues.append(
                _issue(
                    "array_shape_mismatch",
                    table,
                    f"array element does not match {column.element_type.value}",
                    row=row,
                    column=column.name,
                )
            )
    return issues


def validate_world_ir(world: WorldIR, source: SourceModel) -> None:
    """Reject semantic mismatches before any backend compiler or process starts."""

    issues: list[WorldIRIssue] = []
    if world.source_model_fingerprint != source.fingerprint:
        issues.append(
            _issue(
                "source_model_fingerprint_mismatch",
                "<world>",
                "world IR was authored for a different source model",
            )
        )
    source_tables = {table.name: table for table in source.tables}
    world_tables = {table.source_name: table for table in world.tables}
    for table_name, table in world_tables.items():
        source_table = source_tables.get(table_name)
        if source_table is None:
            issues.append(
                _issue("unknown_table", table_name, "table is absent from source model")
            )
            continue
        columns = {column.name: column for column in source_table.columns}
        for row in table.rows:
            for column_name in sorted(set(row.values) - set(columns)):
                issues.append(
                    _issue(
                        "unknown_column",
                        table_name,
                        "column is absent from source model",
                        row=row,
                        column=column_name,
                    )
                )
            for column in source_table.columns:
                authored = row.values.get(column.name, WorldValue.absent())
                issues.extend(_validate_value(table_name, row, column, authored))

        for key in (source_table.primary_key, *source_table.unique_keys):
            if not key:
                continue
            seen: set[str] = set()
            for row in table.rows:
                values = [row.values.get(column, WorldValue.absent()) for column in key]
                if any(value.state is not ValueState.PRESENT for value in values):
                    continue
                encoded = json.dumps(
                    [value.value for value in values],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if encoded in seen:
                    issues.append(
                        _issue(
                            "unique_key_duplicate",
                            table_name,
                            "authored rows duplicate a source key",
                            row=row,
                            column=",".join(key),
                        )
                    )
                seen.add(encoded)

    for source_table in source.tables:
        authored_table = world_tables.get(source_table.name)
        if authored_table is None:
            continue
        for foreign_key in source_table.foreign_keys:
            target = world_tables.get(foreign_key.referenced_table)
            target_values = (
                {
                    json.dumps(
                        [
                            row.values.get(column, WorldValue.absent()).value
                            for column in foreign_key.referenced_columns
                        ],
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    for row in target.rows
                    if all(
                        row.values.get(column, WorldValue.absent()).state
                        is ValueState.PRESENT
                        for column in foreign_key.referenced_columns
                    )
                }
                if target is not None
                else set()
            )
            for row in authored_table.rows:
                values = [
                    row.values.get(column, WorldValue.absent())
                    for column in foreign_key.columns
                ]
                if any(value.state is ValueState.NULL for value in values):
                    continue
                if not all(value.state is ValueState.PRESENT for value in values):
                    continue
                encoded = json.dumps(
                    [value.value for value in values],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if encoded not in target_values:
                    issues.append(
                        _issue(
                            "foreign_key_missing",
                            source_table.name,
                            "referenced authored row does not exist",
                            row=row,
                            column=",".join(foreign_key.columns),
                        )
                    )
    if issues:
        raise WorldIRValidationError(tuple(issues))


__all__ = [
    "WORLD_IR_SCHEMA_VERSION",
    "ValueState",
    "WorldIR",
    "WorldIRIssue",
    "WorldIRValidationError",
    "WorldRow",
    "WorldTable",
    "WorldValue",
    "validate_world_ir",
]
