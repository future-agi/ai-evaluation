"""Canonical, immutable description of an ALK environment.

The bundle is the portable payload.  The environment plan is the stable decision record inside
that payload: which source revision was admitted, which runtime document will be used, which
services/capabilities exist, and how readiness is proven.  It deliberately excludes ephemeral
project names, allocated host ports, and resolved secret values.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pydantic import BaseModel, Field, JsonValue, model_validator

from .bundle import (
    BundleRuntime,
    Capability,
    EnvironmentBundle,
    ReadinessProbe,
)

ENVIRONMENT_PLAN_SCHEMA_VERSION = "futureagi.environment-plan.v1"
ENVIRONMENT_PLAN_FILE = "environment-plan.json"


class EnvironmentPlanError(RuntimeError):
    """The frozen plan is absent, corrupt, or inconsistent with its bundle."""


class PlanSource(BaseModel):
    kind: str
    repository: str | None = None
    commit_sha: str | None = None
    source_digest: str


class EnvironmentPlan(BaseModel):
    schema_version: str = ENVIRONMENT_PLAN_SCHEMA_VERSION
    digest: str
    source: PlanSource
    runtime: BundleRuntime
    services: list[str] = Field(default_factory=list)
    capabilities: dict[str, Capability] = Field(default_factory=dict)
    readiness: list[ReadinessProbe] = Field(default_factory=list)
    configuration_names: list[str] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid_plan(self) -> "EnvironmentPlan":
        if self.schema_version != ENVIRONMENT_PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"environment_plan_schema_unsupported: {self.schema_version}"
            )
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.digest):
            raise ValueError("environment_plan_digest_invalid")
        missing = [
            probe.capability
            for probe in self.readiness
            if probe.capability not in self.capabilities
        ]
        if missing:
            raise ValueError(
                "environment_plan_readiness_capability_missing: "
                + ", ".join(sorted(set(missing)))
            )
        if self.configuration_names != sorted(set(self.configuration_names)):
            raise ValueError("environment_plan_configuration_names_not_canonical")
        return self


def _digest(raw: dict) -> str:
    canonical = dict(raw)
    canonical.pop("digest", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def create_environment_plan(
    *,
    source: PlanSource,
    runtime: BundleRuntime,
    services: list[str],
    capabilities: dict[str, Capability],
    readiness: list[ReadinessProbe],
    metadata: dict[str, JsonValue] | None = None,
) -> EnvironmentPlan:
    """Create the same plan digest for the same admitted environment decision."""
    names = sorted(
        {
            capability.configuration_name
            for capability in capabilities.values()
            if capability.configuration_name
        }
    )
    raw = {
        "schema_version": ENVIRONMENT_PLAN_SCHEMA_VERSION,
        "digest": "sha256:" + "0" * 64,
        "source": source.model_dump(mode="json"),
        "runtime": runtime.model_dump(mode="json"),
        "services": sorted(set(services)),
        "capabilities": {
            key: capabilities[key].model_dump(mode="json")
            for key in sorted(capabilities)
        },
        "readiness": [
            probe.model_dump(mode="json") for probe in readiness
        ],
        "configuration_names": names,
        "metadata": metadata or {},
    }
    raw["digest"] = _digest(raw)
    return EnvironmentPlan.model_validate(raw)


def write_environment_plan(root: str | Path, plan: EnvironmentPlan) -> Path:
    target = Path(root) / ENVIRONMENT_PLAN_FILE
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def load_environment_plan(
    root: str | Path, *, bundle: EnvironmentBundle | None = None
) -> EnvironmentPlan:
    target = Path(root) / ENVIRONMENT_PLAN_FILE
    if not target.is_file():
        raise EnvironmentPlanError(f"environment_plan_missing: {target}")
    try:
        plan = EnvironmentPlan.model_validate_json(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EnvironmentPlanError(f"environment_plan_invalid: {exc}") from exc
    if _digest(plan.model_dump(mode="json")) != plan.digest:
        raise EnvironmentPlanError("environment_plan_digest_mismatch")
    if bundle is not None:
        expected = create_environment_plan(
            source=PlanSource(
                kind=bundle.provenance.source_kind,
                repository=bundle.provenance.repository,
                commit_sha=bundle.provenance.commit,
                source_digest=bundle.provenance.source_digest,
            ),
            runtime=bundle.runtime,
            services=bundle.services,
            capabilities=bundle.capabilities,
            readiness=bundle.readiness,
            metadata=plan.metadata,
        )
        if expected.digest != plan.digest:
            raise EnvironmentPlanError("environment_plan_bundle_mismatch")
    return plan


__all__ = [
    "ENVIRONMENT_PLAN_FILE",
    "ENVIRONMENT_PLAN_SCHEMA_VERSION",
    "EnvironmentPlan",
    "EnvironmentPlanError",
    "PlanSource",
    "create_environment_plan",
    "load_environment_plan",
    "write_environment_plan",
]
