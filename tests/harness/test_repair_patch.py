from __future__ import annotations

import pytest
from pydantic import ValidationError

from fi.alk.harness.repair_patch import (
    RepairPatchOp,
    RepairPatchOperation,
    RepairPatchRejected,
    WorldIRRepairPatch,
    apply_world_ir_repair_patch,
)
from fi.alk.harness.source_model import (
    ForeignKey,
    LogicalType,
    SourceColumn,
    SourceModel,
    SourceTable,
)
from fi.alk.harness.world_ir import (
    WorldIR,
    WorldRow,
    WorldTable,
    WorldValue,
    validate_world_ir,
)

SOURCE_DIGEST = "sha256:" + ("f" * 64)


def _column(name: str) -> SourceColumn:
    return SourceColumn(
        name=name,
        logical_type=LogicalType.INTEGER,
        native_type="bigint",
        nullable=False,
        has_default=False,
    )


def _source() -> SourceModel:
    return SourceModel.create(
        source_digest=SOURCE_DIGEST,
        engine="postgres",
        tables=(
            SourceTable(name="accounts", columns=(_column("id"),), primary_key=("id",)),
            SourceTable(
                name="payments",
                columns=(_column("id"), _column("account_id")),
                primary_key=("id",),
                foreign_keys=(
                    ForeignKey(
                        columns=("account_id",),
                        referenced_table="accounts",
                        referenced_columns=("id",),
                    ),
                ),
            ),
        ),
    )


def _invalid_world(source: SourceModel) -> WorldIR:
    return WorldIR.create(
        source_model_fingerprint=source.fingerprint,
        tables=(
            WorldTable(
                source_name="payments",
                rows=(
                    WorldRow(
                        identity="payment-1",
                        values={
                            "id": WorldValue.present(LogicalType.INTEGER, 10),
                            "account_id": WorldValue.present(LogicalType.INTEGER, 1),
                        },
                    ),
                ),
            ),
        ),
    )


def test_evidence_backed_add_row_repairs_foreign_key() -> None:
    source = _source()
    world = _invalid_world(source)
    operation = RepairPatchOperation(
        op=RepairPatchOp.ADD_ROW,
        table="accounts",
        row_identity="account-1",
        row=WorldRow(
            identity="account-1",
            values={"id": WorldValue.present(LogicalType.INTEGER, 1)},
        ),
        reason="foreign_key_missing",
        evidence_refs=("src/models.py:Account",),
    )
    patch = WorldIRRepairPatch.create(
        base_world_ir_hash=world.fingerprint, operations=(operation,)
    )

    repaired = apply_world_ir_repair_patch(
        world, source, patch, allowed_reason_codes={"foreign_key_missing"}
    )

    validate_world_ir(repaired, source)
    assert [table.source_name for table in repaired.tables] == ["accounts", "payments"]
    assert repaired.fingerprint != world.fingerprint


def test_patch_cannot_introduce_semantics_without_evidence() -> None:
    with pytest.raises(ValidationError, match="repair_patch_source_evidence_required"):
        RepairPatchOperation(
            op=RepairPatchOp.ADD_ROW,
            table="accounts",
            row_identity="account-1",
            row=WorldRow(
                identity="account-1",
                values={"id": WorldValue.present(LogicalType.INTEGER, 1)},
            ),
            reason="foreign_key_missing",
        )


def test_patch_rejects_unrelated_reason_and_stale_base() -> None:
    source = _source()
    world = _invalid_world(source)
    operation = RepairPatchOperation(
        op=RepairPatchOp.MARK_ABSENT,
        table="payments",
        row_identity="payment-1",
        column="account_id",
        reason="source_default_suppressed",
    )
    stale = WorldIRRepairPatch.create(
        base_world_ir_hash="sha256:" + ("0" * 64), operations=(operation,)
    )
    with pytest.raises(RepairPatchRejected, match="base_hash_mismatch"):
        apply_world_ir_repair_patch(
            world,
            source,
            stale,
            allowed_reason_codes={"source_default_suppressed"},
        )

    patch = WorldIRRepairPatch.create(
        base_world_ir_hash=world.fingerprint, operations=(operation,)
    )
    with pytest.raises(RepairPatchRejected, match="reason_not_authorized"):
        apply_world_ir_repair_patch(
            world, source, patch, allowed_reason_codes={"foreign_key_missing"}
        )


def test_patch_surface_has_no_delete_or_source_file_operation() -> None:
    assert set(RepairPatchOp) == {
        RepairPatchOp.ADD_ROW,
        RepairPatchOp.SET_VALUE,
        RepairPatchOp.MARK_ABSENT,
    }
