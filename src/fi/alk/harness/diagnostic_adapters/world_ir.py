"""Convert semantic World IR validation issues into stable harness diagnostics."""

from __future__ import annotations

from collections.abc import Iterable

from ..diagnostics import DiagnosticLocation, HarnessDiagnostic
from ..job import HarnessStage
from ..world_ir import WorldIRValidationError


def diagnose_world_ir_error(
    error: WorldIRValidationError,
    *,
    secret_values: Iterable[str] = (),
) -> tuple[HarnessDiagnostic, ...]:
    """Return the complete typed failure set rather than repairing the first bad row."""

    diagnostics = [
        HarnessDiagnostic.create(
            stage=HarnessStage.VALIDATING_ENVIRONMENT,
            component="world_ir",
            code=issue.code,
            message=issue.message,
            location=DiagnosticLocation(
                table=None if issue.table == "<world>" else issue.table,
                column=issue.column,
            ),
            evidence_refs=(
                "artifact://world-ir"
                + (f"/{issue.table}" if issue.table != "<world>" else "")
                + (f"/{issue.row_identity}" if issue.row_identity else "")
                + (f"/{issue.column}" if issue.column else ""),
            ),
            secret_values=secret_values,
        )
        for issue in error.issues
    ]
    return tuple(
        sorted(
            diagnostics,
            key=lambda item: (
                item.code,
                item.location.table if item.location else "",
                item.location.column if item.location else "",
                item.fingerprint,
            ),
        )
    )


__all__ = ["diagnose_world_ir_error"]
