"""Typed, secret-safe diagnostics for harness validation and repair.

Diagnostics are deliberately separate from exception text.  A stable code and fingerprint let
the repair controller recognize repeated failures, while the redacted message and evidence
references are safe to persist or send to the platform.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from types import MappingProxyType
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .job import FailureDomain, HarnessStage
from .outbound import redact_outbound_text

HARNESS_DIAGNOSTIC_SCHEMA_VERSION = "futureagi.harness-diagnostic.v1"


class RepairOwner(str, Enum):
    """Subsystem allowed to act on a diagnostic."""

    COMPILER = "compiler"
    AUTHORING = "authoring"
    INFRASTRUCTURE = "infrastructure"
    SOURCE = "source"
    AGENT = "agent"
    UNSUPPORTED = "unsupported"


class DiagnosticLocation(BaseModel):
    """Non-sensitive structural location extracted from a failing subsystem."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_path: str | None = None
    line: int | None = Field(default=None, ge=1)
    table: str | None = None
    column: str | None = None
    constraint: str | None = None
    datatype: str | None = None
    process: str | None = None
    tool: str | None = None

    @model_validator(mode="after")
    def _source_path_is_portable(self) -> "DiagnosticLocation":
        if self.source_path and (
            self.source_path.startswith("/") or ".." in self.source_path.split("/")
        ):
            raise ValueError("diagnostic_source_path_must_be_relative")
        return self


class DiagnosticPolicy(BaseModel):
    """Deterministic default disposition for one diagnostic code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    domain: FailureDomain
    owner: RepairOwner
    retryable: bool = False
    repair_strategy: str | None = None


_POLICIES = MappingProxyType(
    {
        "schema_type_mismatch": DiagnosticPolicy(
            domain=FailureDomain.ENVIRONMENT,
            owner=RepairOwner.COMPILER,
            repair_strategy="recompile_with_discovered_type",
        ),
        "source_model_fingerprint_mismatch": DiagnosticPolicy(
            domain=FailureDomain.ENVIRONMENT,
            owner=RepairOwner.COMPILER,
            repair_strategy="regenerate_world_from_current_source_model",
        ),
        "generated_column_authored": DiagnosticPolicy(
            domain=FailureDomain.ENVIRONMENT,
            owner=RepairOwner.COMPILER,
            repair_strategy="mark_generated_column_absent",
        ),
        "value_shape_mismatch": DiagnosticPolicy(
            domain=FailureDomain.ENVIRONMENT,
            owner=RepairOwner.COMPILER,
            repair_strategy="normalize_value_to_discovered_type",
        ),
        "unknown_table": DiagnosticPolicy(
            domain=FailureDomain.ENVIRONMENT,
            owner=RepairOwner.AUTHORING,
            repair_strategy="remove_unverified_table",
        ),
        "unknown_column": DiagnosticPolicy(
            domain=FailureDomain.ENVIRONMENT,
            owner=RepairOwner.AUTHORING,
            repair_strategy="remove_unverified_column",
        ),
        "required_value_missing": DiagnosticPolicy(
            domain=FailureDomain.ENVIRONMENT,
            owner=RepairOwner.AUTHORING,
            repair_strategy="request_targeted_data_patch",
        ),
        "explicit_null_not_allowed": DiagnosticPolicy(
            domain=FailureDomain.ENVIRONMENT,
            owner=RepairOwner.AUTHORING,
            repair_strategy="mark_value_absent_or_author_non_null_value",
        ),
        "source_default_suppressed": DiagnosticPolicy(
            domain=FailureDomain.ENVIRONMENT,
            owner=RepairOwner.COMPILER,
            repair_strategy="mark_value_absent",
        ),
        "array_shape_mismatch": DiagnosticPolicy(
            domain=FailureDomain.ENVIRONMENT,
            owner=RepairOwner.COMPILER,
            repair_strategy="normalize_logical_array",
        ),
        "enum_value_invalid": DiagnosticPolicy(
            domain=FailureDomain.ENVIRONMENT,
            owner=RepairOwner.AUTHORING,
            repair_strategy="select_source_valid_value",
        ),
        "foreign_key_missing": DiagnosticPolicy(
            domain=FailureDomain.ENVIRONMENT,
            owner=RepairOwner.AUTHORING,
            repair_strategy="add_source_consistent_related_record",
        ),
        "unique_key_duplicate": DiagnosticPolicy(
            domain=FailureDomain.ENVIRONMENT,
            owner=RepairOwner.AUTHORING,
            repair_strategy="author_unique_key_value",
        ),
        "constraint_value_invalid": DiagnosticPolicy(
            domain=FailureDomain.ENVIRONMENT,
            owner=RepairOwner.AUTHORING,
            repair_strategy="author_constraint_valid_value",
        ),
        "seed_order_invalid": DiagnosticPolicy(
            domain=FailureDomain.ENVIRONMENT,
            owner=RepairOwner.COMPILER,
            repair_strategy="reorder_inserts",
        ),
        "generated_setup_invalid": DiagnosticPolicy(
            domain=FailureDomain.ENVIRONMENT,
            owner=RepairOwner.AUTHORING,
            repair_strategy="regenerate_setup_artifact",
        ),
        "ready_condition_invalid": DiagnosticPolicy(
            domain=FailureDomain.ENVIRONMENT,
            owner=RepairOwner.AUTHORING,
            repair_strategy="repair_ready_condition",
        ),
        "process_dependency_timeout": DiagnosticPolicy(
            domain=FailureDomain.INFRASTRUCTURE,
            owner=RepairOwner.INFRASTRUCTURE,
            retryable=True,
            repair_strategy="retry_dependency_startup",
        ),
        "credential_missing": DiagnosticPolicy(
            domain=FailureDomain.AGENT,
            owner=RepairOwner.SOURCE,
        ),
        "egress_blocked": DiagnosticPolicy(
            domain=FailureDomain.INFRASTRUCTURE,
            owner=RepairOwner.INFRASTRUCTURE,
        ),
        "tool_endpoint_unreachable": DiagnosticPolicy(
            domain=FailureDomain.CONNECTIVITY,
            owner=RepairOwner.INFRASTRUCTURE,
            retryable=True,
            repair_strategy="repair_tool_wiring",
        ),
        "tool_schema_mismatch": DiagnosticPolicy(
            domain=FailureDomain.ENVIRONMENT,
            owner=RepairOwner.SOURCE,
            repair_strategy="reconcile_discovered_contract",
        ),
        "agent_tool_argument_invalid": DiagnosticPolicy(
            domain=FailureDomain.AGENT,
            owner=RepairOwner.AGENT,
        ),
        "unsupported_source_construct": DiagnosticPolicy(
            domain=FailureDomain.ENVIRONMENT,
            owner=RepairOwner.UNSUPPORTED,
        ),
    }
)


def diagnostic_policy(code: str) -> DiagnosticPolicy:
    """Return the closed policy for ``code``; unknown codes must be classified explicitly."""

    try:
        return _POLICIES[code]
    except KeyError as exc:
        raise ValueError(f"unknown_harness_diagnostic_code: {code}") from exc


def _fingerprint_payload(
    *,
    stage: HarnessStage,
    component: str,
    code: str,
    policy: DiagnosticPolicy,
    location: DiagnosticLocation | None,
) -> dict[str, object]:
    return {
        "schema_version": HARNESS_DIAGNOSTIC_SCHEMA_VERSION,
        "stage": stage.value,
        "domain": policy.domain.value,
        "component": component,
        "code": code,
        "owner": policy.owner.value,
        "retryable": policy.retryable,
        "repair_strategy": policy.repair_strategy,
        "location": location.model_dump(mode="json", exclude_none=True)
        if location
        else None,
    }


def _fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class HarnessDiagnostic(BaseModel):
    """Persistable diagnosis with deterministic ownership and repair semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = HARNESS_DIAGNOSTIC_SCHEMA_VERSION
    stage: HarnessStage
    domain: FailureDomain
    component: str = Field(min_length=1)
    code: str = Field(min_length=1)
    owner: RepairOwner
    retryable: bool
    repair_strategy: str | None = None
    location: DiagnosticLocation | None = None
    redacted_message: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = ()
    fingerprint: str

    @classmethod
    def create(
        cls,
        *,
        stage: HarnessStage,
        component: str,
        code: str,
        message: str,
        location: DiagnosticLocation | None = None,
        evidence_refs: Iterable[str] = (),
        secret_values: Iterable[str] = (),
    ) -> "HarnessDiagnostic":
        """Classify, redact, and fingerprint a raw subsystem failure."""

        policy = diagnostic_policy(code)
        secrets = tuple(secret_values)
        redacted_refs = tuple(
            sorted(
                {
                    redact_outbound_text(reference, secrets)
                    for reference in evidence_refs
                }
            )
        )
        payload = _fingerprint_payload(
            stage=stage,
            component=component,
            code=code,
            policy=policy,
            location=location,
        )
        return cls(
            stage=stage,
            domain=policy.domain,
            component=component,
            code=code,
            owner=policy.owner,
            retryable=policy.retryable,
            repair_strategy=policy.repair_strategy,
            location=location,
            redacted_message=redact_outbound_text(message, secrets),
            evidence_refs=redacted_refs,
            fingerprint=_fingerprint(payload),
        )

    @model_validator(mode="after")
    def _policy_and_fingerprint_are_canonical(self) -> "HarnessDiagnostic":
        if self.schema_version != HARNESS_DIAGNOSTIC_SCHEMA_VERSION:
            raise ValueError("harness_diagnostic_schema_version_unsupported")
        policy = diagnostic_policy(self.code)
        if (
            self.domain,
            self.owner,
            self.retryable,
            self.repair_strategy,
        ) != (
            policy.domain,
            policy.owner,
            policy.retryable,
            policy.repair_strategy,
        ):
            raise ValueError("harness_diagnostic_policy_mismatch")
        expected = _fingerprint(
            _fingerprint_payload(
                stage=self.stage,
                component=self.component,
                code=self.code,
                policy=policy,
                location=self.location,
            )
        )
        if self.fingerprint != expected:
            raise ValueError("harness_diagnostic_fingerprint_mismatch")
        return self
