"""Constrained, evidence-backed patches for typed World IR."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .source_model import SourceModel
from .world_ir import WorldIR, WorldRow, WorldTable, WorldValue, validate_world_ir

REPAIR_PATCH_SCHEMA_VERSION = "futureagi.world-ir-repair-patch.v1"


class RepairPatchOp(str, Enum):
    ADD_ROW = "add_row"
    SET_VALUE = "set_value"
    MARK_ABSENT = "mark_absent"


class RepairPatchOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    op: RepairPatchOp
    table: str = Field(min_length=1)
    row_identity: str = Field(min_length=1)
    column: str | None = None
    value: WorldValue | None = None
    row: WorldRow | None = None
    reason: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _operation_shape(self) -> "RepairPatchOperation":
        if self.op is RepairPatchOp.ADD_ROW:
            if self.row is None or self.row.identity != self.row_identity:
                raise ValueError("repair_patch_add_row_payload_invalid")
            if self.column is not None or self.value is not None:
                raise ValueError("repair_patch_add_row_fields_conflict")
        elif self.op is RepairPatchOp.SET_VALUE:
            if self.column is None or self.value is None or self.row is not None:
                raise ValueError("repair_patch_set_value_payload_invalid")
        elif self.column is None or self.value is not None or self.row is not None:
            raise ValueError("repair_patch_mark_absent_payload_invalid")
        introduces_semantics = self.op is RepairPatchOp.ADD_ROW or (
            self.op is RepairPatchOp.SET_VALUE and self.value is not None
        )
        if introduces_semantics and not self.evidence_refs:
            raise ValueError("repair_patch_source_evidence_required")
        return self


class WorldIRRepairPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = REPAIR_PATCH_SCHEMA_VERSION
    base_world_ir_hash: str
    operations: tuple[RepairPatchOperation, ...]
    fingerprint: str

    @classmethod
    def create(
        cls,
        *,
        base_world_ir_hash: str,
        operations: tuple[RepairPatchOperation, ...],
    ) -> "WorldIRRepairPatch":
        raw: dict[str, Any] = {
            "schema_version": REPAIR_PATCH_SCHEMA_VERSION,
            "base_world_ir_hash": base_world_ir_hash,
            "operations": operations,
        }
        raw["fingerprint"] = _fingerprint(raw)
        return cls.model_validate(raw)

    @model_validator(mode="after")
    def _canonical_hash(self) -> "WorldIRRepairPatch":
        if self.schema_version != REPAIR_PATCH_SCHEMA_VERSION:
            raise ValueError("repair_patch_schema_version_unsupported")
        expected = _fingerprint(self.model_dump(mode="python", exclude={"fingerprint"}))
        if self.fingerprint != expected:
            raise ValueError("repair_patch_fingerprint_mismatch")
        return self


class RepairPatchRejected(ValueError):
    def __init__(self, code: str, operation_index: int | None = None) -> None:
        self.code = code
        self.operation_index = operation_index
        location = (
            f" at operation {operation_index}" if operation_index is not None else ""
        )
        super().__init__(f"{code}{location}")


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


def apply_world_ir_repair_patch(
    world: WorldIR,
    source: SourceModel,
    patch: WorldIRRepairPatch,
    *,
    allowed_reason_codes: set[str],
) -> WorldIR:
    """Apply an authorized patch atomically and revalidate against source facts."""

    if patch.base_world_ir_hash != world.fingerprint:
        raise RepairPatchRejected("repair_patch_base_hash_mismatch")
    source_tables = {table.name: table for table in source.tables}
    tables = {
        table.source_name: {row.identity: row for row in table.rows}
        for table in world.tables
    }
    for index, operation in enumerate(patch.operations):
        if operation.reason not in allowed_reason_codes:
            raise RepairPatchRejected("repair_patch_reason_not_authorized", index)
        source_table = source_tables.get(operation.table)
        if source_table is None:
            raise RepairPatchRejected("repair_patch_table_unknown", index)
        rows = tables.setdefault(operation.table, {})
        if operation.op is RepairPatchOp.ADD_ROW:
            if operation.row_identity in rows:
                raise RepairPatchRejected("repair_patch_row_already_exists", index)
            assert operation.row is not None
            rows[operation.row_identity] = operation.row
            continue
        row = rows.get(operation.row_identity)
        if row is None:
            raise RepairPatchRejected("repair_patch_row_unknown", index)
        assert operation.column is not None
        if operation.column not in {column.name for column in source_table.columns}:
            raise RepairPatchRejected("repair_patch_column_unknown", index)
        values = dict(row.values)
        values[operation.column] = (
            WorldValue.absent()
            if operation.op is RepairPatchOp.MARK_ABSENT
            else operation.value
        )
        rows[operation.row_identity] = WorldRow(identity=row.identity, values=values)
    candidate = WorldIR.create(
        source_model_fingerprint=world.source_model_fingerprint,
        tables=tuple(
            WorldTable(
                source_name=table,
                rows=tuple(sorted(rows.values(), key=lambda item: item.identity)),
            )
            for table, rows in sorted(tables.items())
        ),
    )
    validate_world_ir(candidate, source)
    return candidate


__all__ = [
    "REPAIR_PATCH_SCHEMA_VERSION",
    "RepairPatchOp",
    "RepairPatchOperation",
    "RepairPatchRejected",
    "WorldIRRepairPatch",
    "apply_world_ir_repair_patch",
]
