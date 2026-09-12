"""Versioned, integrity-checked certification artifacts for generic harness runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .diagnostics import HarnessDiagnostic
from .repair_controller import RepairHistory
from .source_model import SourceModel
from .world_ir import WorldIR

HARNESS_CERTIFICATION_SCHEMA_VERSION = "futureagi.harness-certification.v1"


class CertificationStatus(str, Enum):
    CERTIFIED = "certified"
    REJECTED = "rejected"


class CheckStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"


class CertificationSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repository: str | None = None
    commit: str | None = None
    digest: str
    schema_hash: str


class CertificationAuthoring(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_hash: str
    world_ir_hash: str
    scenario_set_hash: str


class CertificationCompiler(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1)
    bundle_digest: str


class CertificationRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot: str | None = None
    validation_attempts: int = Field(ge=1)


class CertificationChecks(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    static: CheckStatus = CheckStatus.NOT_RUN
    schema_and_seed: CheckStatus = CheckStatus.NOT_RUN
    processes: CheckStatus = CheckStatus.NOT_RUN
    source_invariants: CheckStatus = CheckStatus.NOT_RUN
    scenario_setup_ready: str = "0/0"
    tool_contract: str = "0/0"
    reset_equivalence: CheckStatus = CheckStatus.NOT_RUN
    world_isolation: CheckStatus = CheckStatus.NOT_RUN


class RuntimeValidationEvidence(BaseModel):
    """Facts captured by the disposable runtime before it is torn down.

    This is deliberately separate from the final certificate: the runtime can prove what it
    exercised, while the repair controller owns the terminal decision and complete history.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_digest: str
    source_schema_hash: str
    world_ir_hash: str
    compiler_version: str = Field(min_length=1)
    bundle_digest: str
    contract_hash: str
    scenario_set_hash: str
    checks: CertificationChecks
    limitations: tuple[str, ...] = ()


class HarnessCertification(BaseModel):
    """Secret-free proof describing exactly what was and was not validated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = HARNESS_CERTIFICATION_SCHEMA_VERSION
    status: CertificationStatus
    source: CertificationSource
    authoring: CertificationAuthoring
    compiler: CertificationCompiler
    runtime: CertificationRuntime
    checks: CertificationChecks
    diagnostics: tuple[HarnessDiagnostic, ...] = ()
    repairs: RepairHistory
    limitations: tuple[str, ...] = ()
    fingerprint: str

    @classmethod
    def create(
        cls,
        *,
        status: CertificationStatus,
        source: CertificationSource,
        authoring: CertificationAuthoring,
        compiler: CertificationCompiler,
        runtime: CertificationRuntime,
        checks: CertificationChecks,
        repairs: RepairHistory,
        diagnostics: tuple[HarnessDiagnostic, ...] = (),
        limitations: tuple[str, ...] = (),
    ) -> "HarnessCertification":
        raw: dict[str, Any] = {
            "schema_version": HARNESS_CERTIFICATION_SCHEMA_VERSION,
            "status": status,
            "source": source,
            "authoring": authoring,
            "compiler": compiler,
            "runtime": runtime,
            "checks": checks,
            "diagnostics": tuple(
                sorted(diagnostics, key=lambda item: item.fingerprint)
            ),
            "repairs": repairs,
            "limitations": tuple(sorted(set(limitations))),
        }
        raw["fingerprint"] = _fingerprint(raw)
        return cls.model_validate(raw)

    @model_validator(mode="after")
    def _canonical_and_consistent(self) -> "HarnessCertification":
        if self.schema_version != HARNESS_CERTIFICATION_SCHEMA_VERSION:
            raise ValueError("harness_certification_schema_version_unsupported")
        if self.diagnostics != tuple(
            sorted(self.diagnostics, key=lambda item: item.fingerprint)
        ):
            raise ValueError("harness_certification_diagnostics_not_canonical")
        if self.limitations != tuple(sorted(set(self.limitations))):
            raise ValueError("harness_certification_limitations_not_canonical")
        if self.status is CertificationStatus.CERTIFIED and self.diagnostics:
            raise ValueError("certified_harness_has_blocking_diagnostics")
        expected = _fingerprint(self.model_dump(mode="python", exclude={"fingerprint"}))
        if self.fingerprint != expected:
            raise ValueError("harness_certification_fingerprint_mismatch")
        return self


def _fingerprint(raw: dict[str, Any]) -> str:
    def jsonable(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, tuple):
            return [jsonable(item) for item in value]
        if isinstance(value, dict):
            return {str(key): jsonable(value[key]) for key in sorted(value)}
        return value

    encoded = json.dumps(
        jsonable(raw),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


Artifact = TypeVar("Artifact", bound=BaseModel)


class GenericHarnessArtifactStore:
    """Atomic readers/writers for resumable generic-harness state."""

    SOURCE_MODEL = "source-model.json"
    WORLD_IR = "world-ir.json"
    REPAIR_HISTORY = "repair-history.json"
    RUNTIME_EVIDENCE = "runtime-evidence.json"
    CERTIFICATION = "certification.json"

    def __init__(self, root: Path) -> None:
        self.root = root

    def _write(self, name: str, value: BaseModel) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{name}.", dir=self.root)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(value.model_dump_json(indent=2))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.root / name)
        finally:
            temporary.unlink(missing_ok=True)
        return self.root / name

    def _read(self, name: str, model: type[Artifact]) -> Artifact:
        path = self.root / name
        if path.is_symlink():
            raise ValueError("generic_harness_artifact_symlink_forbidden")
        return model.model_validate_json(path.read_text(encoding="utf-8"))

    def write_source_model(self, value: SourceModel) -> Path:
        return self._write(self.SOURCE_MODEL, value)

    def read_source_model(self) -> SourceModel:
        return self._read(self.SOURCE_MODEL, SourceModel)

    def write_world_ir(self, value: WorldIR) -> Path:
        return self._write(self.WORLD_IR, value)

    def read_world_ir(self) -> WorldIR:
        return self._read(self.WORLD_IR, WorldIR)

    def write_repair_history(self, value: RepairHistory) -> Path:
        return self._write(self.REPAIR_HISTORY, value)

    def read_repair_history(self) -> RepairHistory:
        return self._read(self.REPAIR_HISTORY, RepairHistory)

    def write_runtime_evidence(self, value: RuntimeValidationEvidence) -> Path:
        return self._write(self.RUNTIME_EVIDENCE, value)

    def read_runtime_evidence(self) -> RuntimeValidationEvidence:
        return self._read(self.RUNTIME_EVIDENCE, RuntimeValidationEvidence)

    def write_certification(self, value: HarnessCertification) -> Path:
        return self._write(self.CERTIFICATION, value)

    def read_certification(self) -> HarnessCertification:
        return self._read(self.CERTIFICATION, HarnessCertification)


__all__ = [
    "HARNESS_CERTIFICATION_SCHEMA_VERSION",
    "CertificationAuthoring",
    "CertificationChecks",
    "CertificationCompiler",
    "CertificationRuntime",
    "CertificationSource",
    "CertificationStatus",
    "CheckStatus",
    "GenericHarnessArtifactStore",
    "HarnessCertification",
    "RuntimeValidationEvidence",
]
