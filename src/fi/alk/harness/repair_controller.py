"""Bounded, ownership-aware policy engine for harness repair attempts."""

from __future__ import annotations

import random
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from .diagnostics import HarnessDiagnostic, RepairOwner


class RepairAction(str, Enum):
    CERTIFY = "certify"
    RETRY_INFRASTRUCTURE = "retry_infrastructure"
    RECOMPILE = "recompile"
    PATCH_ENVIRONMENT = "patch_environment"
    PATCH_SCENARIOS = "patch_scenarios"
    REJECT = "reject"


class RepairPhase(str, Enum):
    ENVIRONMENT = "environment"
    SCENARIOS = "scenarios"


class RepairBudgets(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    compiler_candidates: int = Field(default=3, ge=0, le=10)
    environment_patches: int = Field(default=2, ge=0, le=10)
    scenario_patches: int = Field(default=2, ge=0, le=10)
    infrastructure_retries_per_candidate: int = Field(default=2, ge=0, le=5)
    infrastructure_initial_backoff_seconds: float = Field(default=1.0, ge=0, le=60)
    infrastructure_max_backoff_seconds: float = Field(default=30.0, ge=0, le=300)
    infrastructure_jitter_ratio: float = Field(default=0.2, ge=0, le=1)


class CandidateObservation(BaseModel):
    """Value-free result of validating one immutable candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_hash: str
    phase: RepairPhase
    diagnostics: tuple[HarnessDiagnostic, ...] = ()


class RepairDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int
    candidate_hash: str
    action: RepairAction
    reason: str
    diagnostic_fingerprints: tuple[str, ...]
    repair_strategies: tuple[str, ...] = ()
    retry_after_seconds: float | None = Field(default=None, ge=0)


class RepairOutcome(str, Enum):
    APPLIED = "applied"
    FAILED = "failed"
    NO_MATERIAL_CHANGE = "no_material_change"


class RepairAttemptResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_sequence: int
    action: RepairAction
    before_candidate_hash: str
    after_candidate_hash: str | None = None
    outcome: RepairOutcome
    diagnostic_fingerprints: tuple[str, ...]


class RepairHistory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decisions: tuple[RepairDecision, ...]
    results: tuple[RepairAttemptResult, ...]
    compiler_candidates_used: int
    environment_patches_used: int
    scenario_patches_used: int


class RepairController:
    """Select exactly one bounded action for each complete diagnostic set."""

    def __init__(
        self,
        budgets: RepairBudgets | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.budgets = budgets or RepairBudgets()
        self._rng = rng or random.Random()
        self._compiler_candidates = 0
        self._environment_patches = 0
        self._scenario_patches = 0
        self._infrastructure_retries: dict[str, int] = {}
        self._repair_signatures: set[tuple[str, tuple[str, ...]]] = set()
        self._history: list[RepairDecision] = []
        self._results: list[RepairAttemptResult] = []

    @property
    def history(self) -> RepairHistory:
        return RepairHistory(
            decisions=tuple(self._history),
            results=tuple(self._results),
            compiler_candidates_used=self._compiler_candidates,
            environment_patches_used=self._environment_patches,
            scenario_patches_used=self._scenario_patches,
        )

    def _decision(
        self,
        observation: CandidateObservation,
        action: RepairAction,
        reason: str,
        diagnostics: tuple[HarnessDiagnostic, ...],
        retry_after_seconds: float | None = None,
    ) -> RepairDecision:
        decision = RepairDecision(
            sequence=len(self._history) + 1,
            candidate_hash=observation.candidate_hash,
            action=action,
            reason=reason,
            diagnostic_fingerprints=tuple(
                sorted({diagnostic.fingerprint for diagnostic in diagnostics})
            ),
            repair_strategies=tuple(
                sorted(
                    {
                        diagnostic.repair_strategy
                        for diagnostic in diagnostics
                        if diagnostic.repair_strategy
                    }
                )
            ),
            retry_after_seconds=retry_after_seconds,
        )
        self._history.append(decision)
        return decision

    def record_result(
        self,
        decision_sequence: int,
        *,
        after_candidate_hash: str | None,
        outcome: RepairOutcome,
    ) -> RepairAttemptResult:
        """Attach the immutable result of executing one selected repair action."""

        try:
            decision = self._history[decision_sequence - 1]
        except IndexError as exc:
            raise ValueError("repair_decision_sequence_unknown") from exc
        if decision.sequence != decision_sequence:
            raise ValueError("repair_decision_sequence_unknown")
        if any(item.decision_sequence == decision_sequence for item in self._results):
            raise ValueError("repair_decision_result_already_recorded")
        if decision.action in {RepairAction.CERTIFY, RepairAction.REJECT}:
            raise ValueError("terminal_repair_decision_has_no_execution_result")
        if (
            outcome is RepairOutcome.APPLIED
            and after_candidate_hash == decision.candidate_hash
            and decision.action
            in {
                RepairAction.RECOMPILE,
                RepairAction.PATCH_ENVIRONMENT,
                RepairAction.PATCH_SCENARIOS,
            }
        ):
            outcome = RepairOutcome.NO_MATERIAL_CHANGE
        result = RepairAttemptResult(
            decision_sequence=decision.sequence,
            action=decision.action,
            before_candidate_hash=decision.candidate_hash,
            after_candidate_hash=after_candidate_hash,
            outcome=outcome,
            diagnostic_fingerprints=decision.diagnostic_fingerprints,
        )
        self._results.append(result)
        return result

    def decide(self, observation: CandidateObservation) -> RepairDecision:
        diagnostics = tuple(
            sorted(
                observation.diagnostics,
                key=lambda item: (item.owner.value, item.code, item.fingerprint),
            )
        )
        if not diagnostics:
            return self._decision(
                observation,
                RepairAction.CERTIFY,
                "candidate passed validation",
                diagnostics,
            )

        blocking = tuple(
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.owner
            in {RepairOwner.AGENT, RepairOwner.SOURCE, RepairOwner.UNSUPPORTED}
        )
        if blocking:
            return self._decision(
                observation,
                RepairAction.REJECT,
                "diagnostic ownership forbids harness repair",
                blocking,
            )

        infrastructure = tuple(
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.owner is RepairOwner.INFRASTRUCTURE
        )
        if infrastructure:
            if any(not diagnostic.retryable for diagnostic in infrastructure):
                return self._decision(
                    observation,
                    RepairAction.REJECT,
                    "non-retryable infrastructure failure",
                    infrastructure,
                )
            used = self._infrastructure_retries.get(observation.candidate_hash, 0)
            if used >= self.budgets.infrastructure_retries_per_candidate:
                return self._decision(
                    observation,
                    RepairAction.REJECT,
                    "infrastructure retry budget exhausted for candidate",
                    infrastructure,
                )
            self._infrastructure_retries[observation.candidate_hash] = used + 1
            base = min(
                self.budgets.infrastructure_max_backoff_seconds,
                self.budgets.infrastructure_initial_backoff_seconds * (2**used),
            )
            jitter = self.budgets.infrastructure_jitter_ratio
            retry_after = base * self._rng.uniform(1 - jitter, 1 + jitter)
            return self._decision(
                observation,
                RepairAction.RETRY_INFRASTRUCTURE,
                "retry transient infrastructure failure with fresh runtime",
                infrastructure,
                retry_after_seconds=retry_after,
            )

        fingerprints = tuple(sorted({item.fingerprint for item in diagnostics}))
        signature = (observation.candidate_hash, fingerprints)
        if signature in self._repair_signatures:
            return self._decision(
                observation,
                RepairAction.REJECT,
                "repeated diagnostics without a materially changed candidate",
                diagnostics,
            )

        compiler = tuple(
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.owner is RepairOwner.COMPILER
        )
        if compiler:
            if self._compiler_candidates >= self.budgets.compiler_candidates:
                return self._decision(
                    observation,
                    RepairAction.REJECT,
                    "compiler candidate budget exhausted",
                    compiler,
                )
            self._repair_signatures.add(signature)
            self._compiler_candidates += 1
            return self._decision(
                observation,
                RepairAction.RECOMPILE,
                "apply deterministic compiler strategies in one fresh candidate",
                compiler,
            )

        authoring = tuple(
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.owner is RepairOwner.AUTHORING
        )
        if authoring:
            if observation.phase is RepairPhase.SCENARIOS:
                if self._scenario_patches >= self.budgets.scenario_patches:
                    return self._decision(
                        observation,
                        RepairAction.REJECT,
                        "scenario patch budget exhausted",
                        authoring,
                    )
                self._scenario_patches += 1
                action = RepairAction.PATCH_SCENARIOS
            else:
                if self._environment_patches >= self.budgets.environment_patches:
                    return self._decision(
                        observation,
                        RepairAction.REJECT,
                        "environment patch budget exhausted",
                        authoring,
                    )
                self._environment_patches += 1
                action = RepairAction.PATCH_ENVIRONMENT
            self._repair_signatures.add(signature)
            return self._decision(
                observation,
                action,
                "request one constrained evidence-backed authoring patch",
                authoring,
            )

        return self._decision(
            observation,
            RepairAction.REJECT,
            "diagnostic set has no permitted repair policy",
            diagnostics,
        )


__all__ = [
    "CandidateObservation",
    "RepairAction",
    "RepairAttemptResult",
    "RepairBudgets",
    "RepairController",
    "RepairDecision",
    "RepairHistory",
    "RepairOutcome",
    "RepairPhase",
]
