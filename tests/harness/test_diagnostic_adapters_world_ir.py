from __future__ import annotations

from fi.alk.harness.diagnostic_adapters.world_ir import diagnose_world_ir_error
from fi.alk.harness.diagnostics import RepairOwner
from fi.alk.harness.source_model import (
    LogicalType,
    SourceColumn,
    SourceModel,
    SourceTable,
)
from fi.alk.harness.world_ir import (
    WorldIR,
    WorldIRValidationError,
    WorldRow,
    WorldTable,
    WorldValue,
    validate_world_ir,
)


def test_adapter_reports_complete_value_free_failure_set() -> None:
    source = SourceModel.create(
        source_digest="sha256:" + "a" * 64,
        engine="postgres",
        tables=(
            SourceTable(
                name="users",
                columns=(
                    SourceColumn(
                        name="id",
                        logical_type=LogicalType.STRING,
                        native_type="text",
                        nullable=False,
                        has_default=False,
                    ),
                    SourceColumn(
                        name="computed",
                        logical_type=LogicalType.STRING,
                        native_type="text",
                        nullable=False,
                        has_default=False,
                        generated=True,
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
                source_name="users",
                rows=(
                    WorldRow(
                        identity="row-1",
                        values={
                            "id": WorldValue.present(
                                LogicalType.STRING, "customer-secret"
                            ),
                            "computed": WorldValue.present(
                                LogicalType.STRING, "another-secret"
                            ),
                            "invented": WorldValue.present(
                                LogicalType.STRING, "third-secret"
                            ),
                        },
                    ),
                ),
            ),
        ),
    )
    try:
        validate_world_ir(world, source)
    except WorldIRValidationError as error:
        diagnostics = diagnose_world_ir_error(
            error,
            secret_values=("customer-secret", "another-secret", "third-secret"),
        )
    else:  # pragma: no cover - the fixture intentionally violates two source facts.
        raise AssertionError("expected World IR validation failure")

    assert [item.code for item in diagnostics] == [
        "generated_column_authored",
        "unknown_column",
    ]
    assert {item.owner for item in diagnostics} == {
        RepairOwner.COMPILER,
        RepairOwner.AUTHORING,
    }
    serialized = "".join(item.model_dump_json() for item in diagnostics)
    assert "customer-secret" not in serialized
    assert "another-secret" not in serialized
    assert "third-secret" not in serialized
    assert all(item.evidence_refs for item in diagnostics)
