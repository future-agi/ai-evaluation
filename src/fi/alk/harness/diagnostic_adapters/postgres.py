"""Decode PostgreSQL failures from SQLSTATE and structured diagnostic fields."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..diagnostics import DiagnosticLocation, HarnessDiagnostic
from ..job import HarnessStage

_SQLSTATE_CODES = {
    "23502": "explicit_null_not_allowed",
    "23503": "foreign_key_missing",
    "23505": "unique_key_duplicate",
    "23514": "constraint_value_invalid",
    "22P02": "schema_type_mismatch",
    "42804": "schema_type_mismatch",
    "42P01": "generated_setup_invalid",
    "42703": "generated_setup_invalid",
    "42601": "generated_setup_invalid",
}


def _text(value: Any) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def diagnose_postgres_error(
    error: BaseException,
    *,
    stage: HarnessStage,
    component: str = "postgres",
    secret_values: Iterable[str] = (),
    evidence_refs: Iterable[str] = (),
) -> HarnessDiagnostic:
    """Produce a persistable diagnosis without retaining the raw exception."""

    sqlstate = _text(getattr(error, "sqlstate", None))
    diag = getattr(error, "diag", None)
    datatype = _text(getattr(diag, "datatype_name", None)) if diag else None
    code = _SQLSTATE_CODES.get(sqlstate or "", "unsupported_source_construct")
    if (
        sqlstate == "22P02"
        and datatype
        and (datatype.startswith("_") or datatype.endswith("[]"))
    ):
        code = "array_shape_mismatch"
    primary = _text(getattr(diag, "message_primary", None)) if diag else None
    message = primary or _text(error) or "PostgreSQL operation failed"
    return HarnessDiagnostic.create(
        stage=stage,
        component=component,
        code=code,
        message=f"PostgreSQL {sqlstate or 'unknown SQLSTATE'}: {message}",
        location=DiagnosticLocation(
            table=_text(getattr(diag, "table_name", None)) if diag else None,
            column=_text(getattr(diag, "column_name", None)) if diag else None,
            constraint=_text(getattr(diag, "constraint_name", None)) if diag else None,
            datatype=datatype,
        ),
        secret_values=secret_values,
        evidence_refs=evidence_refs,
    )


__all__ = ["diagnose_postgres_error"]
