from __future__ import annotations

from dataclasses import dataclass

from fi.alk.harness.diagnostic_adapters.postgres import diagnose_postgres_error
from fi.alk.harness.diagnostics import RepairOwner
from fi.alk.harness.job import HarnessStage


@dataclass
class Diag:
    message_primary: str
    table_name: str | None = None
    column_name: str | None = None
    constraint_name: str | None = None
    datatype_name: str | None = None


class DatabaseError(Exception):
    def __init__(self, sqlstate: str | None, diag: Diag) -> None:
        self.sqlstate = sqlstate
        self.diag = diag
        super().__init__(diag.message_primary)


def test_not_null_violation_uses_structured_location_and_redaction() -> None:
    secret = "database-password"
    diagnostic = diagnose_postgres_error(
        DatabaseError(
            "23502",
            Diag(
                message_primary=f"null value violates constraint via {secret}",
                table_name="rides",
                column_name="created_at",
            ),
        ),
        stage=HarnessStage.VALIDATING_ENVIRONMENT,
        secret_values=(secret,),
    )

    assert diagnostic.code == "explicit_null_not_allowed"
    assert diagnostic.owner is RepairOwner.AUTHORING
    assert diagnostic.location is not None
    assert diagnostic.location.table == "rides"
    assert diagnostic.location.column == "created_at"
    assert secret not in diagnostic.redacted_message


def test_array_failure_is_identified_from_datatype_not_message() -> None:
    diagnostic = diagnose_postgres_error(
        DatabaseError(
            "22P02",
            Diag(
                message_primary="localized or changing driver wording",
                table_name="riders",
                column_name="accessibility_needs",
                datatype_name="_text",
            ),
        ),
        stage=HarnessStage.VALIDATING_ENVIRONMENT,
    )

    assert diagnostic.code == "array_shape_mismatch"
    assert diagnostic.owner is RepairOwner.COMPILER


def test_constraint_codes_have_stable_ownership() -> None:
    cases = {
        "23503": "foreign_key_missing",
        "23505": "unique_key_duplicate",
        "23514": "constraint_value_invalid",
        "42601": "generated_setup_invalid",
    }
    for sqlstate, code in cases.items():
        diagnostic = diagnose_postgres_error(
            DatabaseError(sqlstate, Diag(message_primary="failure")),
            stage=HarnessStage.VALIDATING_ENVIRONMENT,
        )
        assert diagnostic.code == code


def test_unknown_sqlstate_is_an_actionable_unsupported_diagnostic() -> None:
    diagnostic = diagnose_postgres_error(
        DatabaseError("XX999", Diag(message_primary="new backend condition")),
        stage=HarnessStage.VALIDATING_ENVIRONMENT,
    )

    assert diagnostic.code == "unsupported_source_construct"
    assert diagnostic.owner is RepairOwner.UNSUPPORTED
