"""Deterministic core of the generic, bounded harness pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .certification import GenericHarnessArtifactStore
from .compile.postgres import (
    POSTGRES_COMPILER_VERSION,
    PostgresCompileError,
    PostgresCompileResult,
    compile_postgres,
)
from .diagnostic_adapters.world_ir import diagnose_world_ir_error
from .diagnostics import HarnessDiagnostic
from .job import HarnessStage
from .repair_controller import (
    CandidateObservation,
    RepairController,
    RepairDecision,
    RepairOutcome,
    RepairPhase,
)
from .source_model import SourceModel
from .world_ir import WorldIR, WorldIRValidationError, validate_world_ir


class GenericCandidate(BaseModel):
    """An immutable source/world/compiler identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_hash: str
    source_model_hash: str
    world_ir_hash: str
    compiler_version: str

    @classmethod
    def create(cls, source: SourceModel, world: WorldIR) -> "GenericCandidate":
        payload = {
            "source_model_hash": source.fingerprint,
            "world_ir_hash": world.fingerprint,
            "compiler_version": POSTGRES_COMPILER_VERSION,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return cls(
            candidate_hash="sha256:" + hashlib.sha256(encoded).hexdigest(),
            **payload,
        )


class CandidateEvaluation(BaseModel):
    """One complete validation result; bound values stay only in compiled operations."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    candidate: GenericCandidate
    diagnostics: tuple[HarnessDiagnostic, ...]
    decision: RepairDecision
    compiled: PostgresCompileResult | None = None


def _compile_diagnostic(error: PostgresCompileError) -> HarnessDiagnostic:
    return HarnessDiagnostic.create(
        stage=HarnessStage.VALIDATING_ENVIRONMENT,
        component="postgres_compiler",
        code=error.code,
        message=error.message,
        evidence_refs=("artifact://source-model", "artifact://world-ir"),
    )


class GenericHarnessPipeline:
    """Evaluate immutable candidates and persist every bounded policy decision."""

    def __init__(
        self,
        artifact_root: Path,
        *,
        controller: RepairController | None = None,
    ) -> None:
        self.artifacts = GenericHarnessArtifactStore(artifact_root)
        self.controller = controller or RepairController()

    def evaluate(
        self,
        source: SourceModel,
        world: WorldIR,
        *,
        phase: RepairPhase = RepairPhase.ENVIRONMENT,
    ) -> CandidateEvaluation:
        candidate = GenericCandidate.create(source, world)
        diagnostics: tuple[HarnessDiagnostic, ...] = ()
        compiled: PostgresCompileResult | None = None
        try:
            validate_world_ir(world, source)
        except WorldIRValidationError as error:
            diagnostics = diagnose_world_ir_error(error)
        else:
            try:
                compiled = compile_postgres(source, world)
            except PostgresCompileError as error:
                diagnostics = (_compile_diagnostic(error),)

        observation = CandidateObservation(
            candidate_hash=candidate.candidate_hash,
            phase=phase,
            diagnostics=diagnostics,
        )
        decision = self.controller.decide(observation)
        self.artifacts.write_source_model(source)
        self.artifacts.write_world_ir(world)
        self.artifacts.write_repair_history(self.controller.history)
        return CandidateEvaluation(
            candidate=candidate,
            diagnostics=diagnostics,
            decision=decision,
            compiled=compiled,
        )

    def record_result(
        self,
        decision_sequence: int,
        *,
        after_candidate_hash: str | None,
        outcome: RepairOutcome,
    ) -> None:
        self.controller.record_result(
            decision_sequence,
            after_candidate_hash=after_candidate_hash,
            outcome=outcome,
        )
        self.artifacts.write_repair_history(self.controller.history)


__all__ = [
    "CandidateEvaluation",
    "GenericCandidate",
    "GenericHarnessPipeline",
]
