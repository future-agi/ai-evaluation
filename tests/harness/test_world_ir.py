from __future__ import annotations

import pytest

from fi.alk.harness.source_model import (
    ForeignKey,
    LogicalType,
    SourceColumn,
    SourceModel,
    SourceTable,
)
from fi.alk.harness.world_ir import (
    ValueState,
    WorldIR,
    WorldIRValidationError,
    WorldRow,
    WorldTable,
    WorldValue,
    validate_world_ir,
)

SOURCE_DIGEST = "sha256:" + ("d" * 64)


def _column(
    name: str,
    logical_type: LogicalType,
    *,
    nullable: bool = False,
    default: str | None = None,
    generated: bool = False,
    enum_values: tuple[str, ...] = (),
    element_type: LogicalType | None = None,
) -> SourceColumn:
    return SourceColumn(
        name=name,
        logical_type=logical_type,
        native_type=logical_type.value,
        nullable=nullable,
        has_default=default is not None,
        default_expression=default,
        generated=generated,
        enum_values=enum_values,
        element_type=element_type,
    )


def _source() -> SourceModel:
    users = SourceTable(
        name="users",
        columns=(
            _column("id", LogicalType.INTEGER),
            _column("name", LogicalType.STRING),
            _column("nickname", LogicalType.STRING, nullable=True),
            _column("created_at", LogicalType.TIMESTAMP, default="now()"),
            _column("display", LogicalType.STRING, generated=True),
            _column("roles", LogicalType.ARRAY, element_type=LogicalType.STRING),
            _column(
                "status",
                LogicalType.ENUM,
                enum_values=("active", "disabled"),
            ),
        ),
        primary_key=("id",),
    )
    orders = SourceTable(
        name="orders",
        columns=(
            _column("id", LogicalType.INTEGER),
            _column("user_id", LogicalType.INTEGER),
        ),
        primary_key=("id",),
        foreign_keys=(
            ForeignKey(
                columns=("user_id",),
                referenced_table="users",
                referenced_columns=("id",),
            ),
        ),
    )
    return SourceModel.create(
        source_digest=SOURCE_DIGEST,
        engine="postgres",
        tables=(orders, users),
    )


def _valid_world(source: SourceModel) -> WorldIR:
    return WorldIR.create(
        source_model_fingerprint=source.fingerprint,
        tables=(
            WorldTable(
                source_name="orders",
                rows=(
                    WorldRow(
                        identity="order-1",
                        values={
                            "id": WorldValue.present(LogicalType.INTEGER, 10),
                            "user_id": WorldValue.present(LogicalType.INTEGER, 1),
                        },
                    ),
                ),
            ),
            WorldTable(
                source_name="users",
                rows=(
                    WorldRow(
                        identity="user-1",
                        values={
                            "id": WorldValue.present(LogicalType.INTEGER, 1),
                            "name": WorldValue.present(LogicalType.STRING, "Ada"),
                            "nickname": WorldValue.null(LogicalType.STRING),
                            "created_at": WorldValue.absent(),
                            "display": WorldValue.absent(),
                            "roles": WorldValue.present(
                                LogicalType.ARRAY, ["admin", "rider"]
                            ),
                            "status": WorldValue.present(LogicalType.ENUM, "active"),
                        },
                    ),
                ),
            ),
        ),
    )


def test_absent_null_and_present_serialize_distinctly() -> None:
    values = [
        WorldValue.absent(),
        WorldValue.null(LogicalType.STRING),
        WorldValue.present(LogicalType.STRING, ""),
    ]

    assert [value.state for value in values] == [
        ValueState.ABSENT,
        ValueState.NULL,
        ValueState.PRESENT,
    ]
    assert len({value.model_dump_json() for value in values}) == 3


def test_valid_world_matches_source_model() -> None:
    source = _source()
    world = _valid_world(source)

    validate_world_ir(world, source)
    assert world.fingerprint.startswith("sha256:")


def test_world_hash_is_stable_across_table_input_order() -> None:
    source = _source()
    first = _valid_world(source)
    second = WorldIR.create(
        source_model_fingerprint=source.fingerprint,
        tables=tuple(reversed(first.tables)),
    )

    assert first == second


def test_validation_reports_schema_and_semantic_issues_together() -> None:
    source = _source()
    world = WorldIR.create(
        source_model_fingerprint=source.fingerprint,
        tables=(
            WorldTable(
                source_name="users",
                rows=(
                    WorldRow(
                        identity="user-1",
                        values={
                            "id": WorldValue.present(LogicalType.STRING, "wrong"),
                            "name": WorldValue.absent(),
                            "nickname": WorldValue.null(),
                            "created_at": WorldValue.null(),
                            "display": WorldValue.present(
                                LogicalType.STRING, "generated"
                            ),
                            "roles": WorldValue.present(
                                LogicalType.ARRAY, "not-an-array"
                            ),
                            "status": WorldValue.present(LogicalType.ENUM, "unknown"),
                            "invented": WorldValue.present(LogicalType.STRING, "value"),
                        },
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(WorldIRValidationError) as caught:
        validate_world_ir(world, source)

    codes = {issue.code for issue in caught.value.issues}
    assert codes == {
        "unknown_column",
        "schema_type_mismatch",
        "required_value_missing",
        "explicit_null_not_allowed",
        "generated_column_authored",
        "value_shape_mismatch",
        "enum_value_invalid",
    }


def test_validation_detects_duplicate_keys_and_missing_foreign_keys() -> None:
    source = _source()
    world = WorldIR.create(
        source_model_fingerprint=source.fingerprint,
        tables=(
            WorldTable(
                source_name="orders",
                rows=(
                    WorldRow(
                        identity="order-1",
                        values={
                            "id": WorldValue.present(LogicalType.INTEGER, 10),
                            "user_id": WorldValue.present(LogicalType.INTEGER, 999),
                        },
                    ),
                ),
            ),
            WorldTable(
                source_name="users",
                rows=(
                    WorldRow(
                        identity="user-1",
                        values={
                            "id": WorldValue.present(LogicalType.INTEGER, 1),
                            "name": WorldValue.present(LogicalType.STRING, "Ada"),
                            "roles": WorldValue.present(LogicalType.ARRAY, []),
                            "status": WorldValue.present(LogicalType.ENUM, "active"),
                        },
                    ),
                    WorldRow(
                        identity="user-2",
                        values={
                            "id": WorldValue.present(LogicalType.INTEGER, 1),
                            "name": WorldValue.present(LogicalType.STRING, "Grace"),
                            "roles": WorldValue.present(LogicalType.ARRAY, []),
                            "status": WorldValue.present(LogicalType.ENUM, "active"),
                        },
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(WorldIRValidationError) as caught:
        validate_world_ir(world, source)

    codes = [issue.code for issue in caught.value.issues]
    assert "unique_key_duplicate" in codes
    assert "foreign_key_missing" in codes


def test_validation_rejects_foreign_key_when_target_table_is_omitted() -> None:
    source = _source()
    world = WorldIR.create(
        source_model_fingerprint=source.fingerprint,
        tables=(
            WorldTable(
                source_name="orders",
                rows=(
                    WorldRow(
                        identity="order-1",
                        values={
                            "id": WorldValue.present(LogicalType.INTEGER, 10),
                            "user_id": WorldValue.present(LogicalType.INTEGER, 1),
                        },
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(WorldIRValidationError) as caught:
        validate_world_ir(world, source)

    assert [issue.code for issue in caught.value.issues] == ["foreign_key_missing"]
