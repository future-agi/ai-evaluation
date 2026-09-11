from __future__ import annotations

import pytest
from pydantic import ValidationError

from fi.alk.harness.diagnostics import (
    DiagnosticLocation,
    HarnessDiagnostic,
    RepairOwner,
    diagnostic_policy,
)
from fi.alk.harness.job import FailureDomain, HarnessStage


def test_create_classifies_redacts_and_fingerprints_diagnostic() -> None:
    secret = "top-secret-token"
    diagnostic = HarnessDiagnostic.create(
        stage=HarnessStage.VALIDATING_ENVIRONMENT,
        component="postgres_seed",
        code="array_shape_mismatch",
        message=(
            f"token={secret}; connection=postgresql://harness:password@db.example/test"
        ),
        location=DiagnosticLocation(table="riders", column="accessibility_needs"),
        evidence_refs=(f"artifact://diagnostics/{secret}", "artifact://schema/source"),
        secret_values=(secret,),
    )

    assert diagnostic.domain is FailureDomain.ENVIRONMENT
    assert diagnostic.owner is RepairOwner.COMPILER
    assert diagnostic.repair_strategy == "normalize_logical_array"
    assert secret not in diagnostic.redacted_message
    assert "password" not in diagnostic.redacted_message
    assert all(secret not in reference for reference in diagnostic.evidence_refs)
    assert diagnostic.fingerprint.startswith("sha256:")


def test_fingerprint_ignores_message_secrets_and_evidence_order() -> None:
    common = {
        "stage": HarnessStage.VALIDATING_ENVIRONMENT,
        "component": "postgres_seed",
        "code": "schema_type_mismatch",
        "location": DiagnosticLocation(table="rides", column="created_at"),
    }
    first = HarnessDiagnostic.create(
        **common,
        message="driver said first-secret",
        evidence_refs=("artifact://b", "artifact://a"),
        secret_values=("first-secret",),
    )
    second = HarnessDiagnostic.create(
        **common,
        message="different driver wording second-secret",
        evidence_refs=("artifact://a", "artifact://b"),
        secret_values=("second-secret",),
    )

    assert first.fingerprint == second.fingerprint
    assert first.evidence_refs == second.evidence_refs


def test_fingerprint_changes_with_structural_location() -> None:
    first = HarnessDiagnostic.create(
        stage=HarnessStage.VALIDATING_ENVIRONMENT,
        component="postgres_seed",
        code="schema_type_mismatch",
        message="bad value",
        location=DiagnosticLocation(table="rides", column="created_at"),
    )
    second = HarnessDiagnostic.create(
        stage=HarnessStage.VALIDATING_ENVIRONMENT,
        component="postgres_seed",
        code="schema_type_mismatch",
        message="bad value",
        location=DiagnosticLocation(table="rides", column="updated_at"),
    )

    assert first.fingerprint != second.fingerprint


def test_unknown_code_requires_explicit_taxonomy_change() -> None:
    with pytest.raises(ValueError, match="unknown_harness_diagnostic_code"):
        diagnostic_policy("new_unclassified_failure")


def test_direct_construction_cannot_override_policy_or_fingerprint() -> None:
    valid = HarnessDiagnostic.create(
        stage=HarnessStage.BUILDING_ENVIRONMENT,
        component="dependency",
        code="process_dependency_timeout",
        message="dependency timed out",
    )
    payload = valid.model_dump()
    payload["owner"] = RepairOwner.AGENT

    with pytest.raises(ValidationError, match="harness_diagnostic_policy_mismatch"):
        HarnessDiagnostic.model_validate(payload)

    payload = valid.model_dump()
    payload["fingerprint"] = "sha256:" + ("0" * 64)
    with pytest.raises(
        ValidationError, match="harness_diagnostic_fingerprint_mismatch"
    ):
        HarnessDiagnostic.model_validate(payload)


def test_diagnostic_source_path_must_be_portable() -> None:
    with pytest.raises(
        ValidationError, match="diagnostic_source_path_must_be_relative"
    ):
        DiagnosticLocation(source_path="/tmp/private/agent.py", line=2)
