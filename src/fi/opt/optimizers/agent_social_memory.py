from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence

from ..base.base_optimizer import BaseOptimizer
from ..components import ComponentDiagnosis, relevant_search_paths
from ..observability import AgentObservabilityRecord, AgentObservabilityWindow
from ..targets import AgentCandidate, CandidateEvaluation, OptimizationTarget
from ..types import EvaluationResult, IterationHistory, OptimizationResult
from .agent import (
    _dedupe_diagnoses,
    _diagnose_candidate_evaluation,
    _dump_model,
    _history_from_candidate,
    _normalize_candidate_evaluation,
    _normalize_diagnoses,
)

logger = logging.getLogger(__name__)


CandidateScorer = Callable[
    [AgentCandidate],
    CandidateEvaluation | EvaluationResult | float,
]


@dataclass
class _PatchCredit:
    path: str
    value: Any
    observations: int = 0
    total_delta: float = 0.0
    total_score: float = 0.0
    best_score: float = float("-inf")
    passed: int = 0
    failed: int = 0
    sources: set[str] = field(default_factory=set)

    @property
    def mean_delta(self) -> float:
        if self.observations == 0:
            return 0.0
        return self.total_delta / self.observations

    @property
    def mean_score(self) -> float:
        if self.observations == 0:
            return 0.0
        return self.total_score / self.observations


@dataclass(frozen=True)
class _MemoryProposal:
    patch: dict[str, Any]
    role: str
    reason: str
    parent_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


class AgentSocialMemoryOptimizer(BaseOptimizer):
    """
    Multi-round agent optimizer with metric-bound social memory.

    Each evaluated patch updates a deterministic credit ledger. Later rounds
    combine high-credit choices, critique promising candidates with one more
    change, and remove weak changes. Role and archetype labels are metadata
    only; the evaluator's numeric scores decide every candidate.
    """

    def __init__(
        self,
        target: Optional[OptimizationTarget] = None,
        *,
        evaluate_candidate: Optional[CandidateScorer] = None,
        simulation_evaluator: Any = None,
        experiment_history: Optional[AgentObservabilityWindow] = None,
        diagnoses: Optional[Iterable[ComponentDiagnosis | dict[str, Any]]] = None,
        max_rounds: int = 4,
        beam_width: int = 4,
        max_proposals_per_round: int = 16,
        target_score: float = 1.0,
        include_seed: bool = True,
        auto_diagnose: bool = True,
        diagnostic_score_threshold: float = 0.85,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1.")
        if beam_width < 1:
            raise ValueError("beam_width must be at least 1.")
        if max_proposals_per_round < 1:
            raise ValueError("max_proposals_per_round must be at least 1.")

        self.target = target
        self.evaluate_candidate = evaluate_candidate
        self.simulation_evaluator = simulation_evaluator
        self.experiment_history = experiment_history
        self.diagnoses = _normalize_diagnoses(diagnoses)
        self.max_rounds = max_rounds
        self.beam_width = beam_width
        self.max_proposals_per_round = max_proposals_per_round
        self.target_score = target_score
        self.include_seed = include_seed
        self.auto_diagnose = auto_diagnose
        self.diagnostic_score_threshold = diagnostic_score_threshold
        super().__init__()

    def optimize(
        self,
        evaluator: Any = None,
        data_mapper: Any = None,
        dataset: Optional[List[dict[str, Any]]] = None,
        metric: Optional[Callable] = None,
        *,
        target: Optional[OptimizationTarget] = None,
        evaluate_candidate: Optional[CandidateScorer] = None,
        simulation_evaluator: Any = None,
        experiment_history: Optional[AgentObservabilityWindow] = None,
        diagnoses: Optional[Iterable[ComponentDiagnosis | dict[str, Any]]] = None,
        max_rounds: Optional[int] = None,
        beam_width: Optional[int] = None,
        max_proposals_per_round: Optional[int] = None,
        target_score: Optional[float] = None,
        include_seed: Optional[bool] = None,
        auto_diagnose: Optional[bool] = None,
        diagnostic_score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        active_target = target or self.target
        if active_target is None:
            raise ValueError("AgentSocialMemoryOptimizer requires a target.")

        active_evaluator = (
            evaluate_candidate
            or self.evaluate_candidate
            or getattr(simulation_evaluator, "evaluate_candidate", None)
            or getattr(self.simulation_evaluator, "evaluate_candidate", None)
            or evaluator
        )
        if active_evaluator is None:
            raise ValueError(
                "AgentSocialMemoryOptimizer requires evaluate_candidate or simulation_evaluator."
            )

        active_max_rounds = self.max_rounds if max_rounds is None else max_rounds
        active_beam_width = self.beam_width if beam_width is None else beam_width
        active_max_proposals = (
            self.max_proposals_per_round
            if max_proposals_per_round is None
            else max_proposals_per_round
        )
        if active_max_rounds < 1:
            raise ValueError("max_rounds must be at least 1.")
        if active_beam_width < 1:
            raise ValueError("beam_width must be at least 1.")
        if active_max_proposals < 1:
            raise ValueError("max_proposals_per_round must be at least 1.")

        active_target_score = (
            self.target_score if target_score is None else target_score
        )
        use_include_seed = self.include_seed if include_seed is None else include_seed
        use_auto_diagnose = self.auto_diagnose if auto_diagnose is None else auto_diagnose
        active_diagnostic_threshold = (
            self.diagnostic_score_threshold
            if diagnostic_score_threshold is None
            else diagnostic_score_threshold
        )
        active_history = experiment_history or self.experiment_history
        active_diagnoses = _normalize_diagnoses(diagnoses)
        if diagnoses is None:
            active_diagnoses = list(self.diagnoses)

        seed_candidate = active_target.seed_candidate()
        evaluated: dict[str, CandidateEvaluation] = {}
        history: List[IterationHistory] = []
        role_counts: dict[str, int] = {}
        round_summaries: list[dict[str, Any]] = []
        proposal_audit: list[dict[str, Any]] = []
        ledger: dict[str, _PatchCredit] = {}
        best: CandidateEvaluation | None = None

        if use_include_seed:
            seed_evaluation = self._evaluate(
                seed_candidate,
                active_evaluator,
                evaluated,
                history,
                role_counts,
                role="seed",
                round_number=0,
                reason="evaluate_deployed_seed",
                metadata={},
            )
            best = seed_evaluation
            if use_auto_diagnose and not active_diagnoses:
                active_diagnoses = _diagnose_candidate_evaluation(
                    seed_evaluation,
                    failing_threshold=active_diagnostic_threshold,
                )

        search_paths = _ordered_search_paths(active_target, active_diagnoses)
        search_paths = _merge_history_search_paths(
            search_paths,
            active_history,
            target=active_target,
            seed_candidate=seed_candidate,
        )
        if not search_paths:
            raise ValueError(
                "AgentSocialMemoryOptimizer target search space cannot be empty."
            )

        baseline_score = (
            best.score
            if best is not None
            else _history_baseline_score(active_history)
        )
        historical_prior_count = _seed_credit_from_history(
            active_history,
            target=active_target,
            seed_candidate=seed_candidate,
            search_paths=search_paths,
            baseline_score=baseline_score,
            ledger=ledger,
        )
        prior_proposals = _prior_proposals_from_history(
            active_history,
            target=active_target,
            seed_candidate=seed_candidate,
            search_paths=search_paths,
        )

        for round_number in range(1, active_max_rounds + 1):
            proposals = _build_social_memory_proposals(
                seed_candidate=seed_candidate,
                evaluations=list(evaluated.values()),
                search_space=active_target.search_space,
                search_paths=search_paths,
                diagnoses=active_diagnoses,
                ledger=ledger,
                prior_proposals=prior_proposals if round_number == 1 else (),
                beam_width=active_beam_width,
                max_proposals=active_max_proposals,
                round_number=round_number,
            )
            proposals = [
                proposal
                for proposal in proposals
                if _candidate_id_for_patch(seed_candidate, proposal.patch)
                not in evaluated
            ]
            logger.info(
                "Social memory round %s evaluating %s proposal(s)",
                round_number,
                len(proposals),
            )

            round_best = best
            round_evaluated = 0
            for proposal in proposals:
                candidate = seed_candidate.with_patch(
                    proposal.patch,
                    metadata={
                        "kind": "social_memory_proposal",
                        "optimizer": "AgentSocialMemoryOptimizer",
                        "proposal_role": proposal.role,
                        "proposal_reason": proposal.reason,
                        "proposal_round": round_number,
                        "proposal_parent_ids": list(proposal.parent_ids),
                        "proposal_metadata": dict(proposal.metadata),
                    },
                )
                evaluation = self._evaluate(
                    candidate,
                    active_evaluator,
                    evaluated,
                    history,
                    role_counts,
                    role=proposal.role,
                    round_number=round_number,
                    reason=proposal.reason,
                    metadata=dict(proposal.metadata),
                )
                _record_credit_from_evaluation(
                    evaluation,
                    baseline_score=baseline_score,
                    ledger=ledger,
                    source=proposal.role,
                )
                proposal_audit.append(
                    {
                        "round": round_number,
                        "role": proposal.role,
                        "candidate_id": candidate.id,
                        "patch": copy.deepcopy(candidate.patch),
                        "score": evaluation.score,
                        "reason": proposal.reason,
                    }
                )
                round_evaluated += 1
                if round_best is None or evaluation.score > round_best.score:
                    round_best = evaluation
                if best is None or evaluation.score > best.score:
                    best = evaluation
                    logger.info(
                        "New best social-memory candidate %s score=%.4f",
                        candidate.id,
                        evaluation.score,
                    )
                if best.score >= active_target_score:
                    break

            if round_best is not None and use_auto_diagnose:
                round_diagnoses = _diagnose_candidate_evaluation(
                    round_best,
                    failing_threshold=active_diagnostic_threshold,
                )
                if round_diagnoses:
                    active_diagnoses = _dedupe_diagnoses(
                        [*active_diagnoses, *round_diagnoses]
                    )
                    search_paths = _ordered_search_paths(active_target, active_diagnoses)
                    search_paths = _merge_history_search_paths(
                        search_paths,
                        active_history,
                        target=active_target,
                        seed_candidate=seed_candidate,
                    )

            round_summaries.append(
                {
                    "round": round_number,
                    "proposals": len(proposals),
                    "evaluated": round_evaluated,
                    "best_score": best.score if best is not None else None,
                    "search_paths": list(search_paths),
                    "ledger_size": len(ledger),
                }
            )
            if best is not None and best.score >= active_target_score:
                break

        if best is None:
            raise ValueError("AgentSocialMemoryOptimizer did not evaluate any candidates.")

        metadata = {
            "optimizer": "AgentSocialMemoryOptimizer",
            "strategy": "futureagi_social_memory",
            "strategy_inspiration": (
                "social credit assignment, working memory, critique, synthesis, "
                "and stewardship; names are metadata only"
            ),
            "roles": ["smriti", "arjuna", "vidura", "sangha", "dharma_steward"],
            "target_name": best.candidate.target_name,
            "best_candidate_id": best.candidate.id,
            "search_paths": list(search_paths),
            "rounds": round_summaries,
            "beam_width": active_beam_width,
            "max_proposals_per_round": active_max_proposals,
            "role_evaluations": role_counts,
            "historical_prior_count": historical_prior_count,
            "history_source": active_history.source if active_history else None,
            "history_record_count": len(active_history.records) if active_history else 0,
            "credit_ledger": _ledger_summary(ledger),
            "proposal_audit": proposal_audit,
        }
        if active_diagnoses:
            metadata["diagnostics"] = [_dump_model(item) for item in active_diagnoses]
            metadata["auto_diagnosed"] = use_auto_diagnose

        return OptimizationResult(
            best_generator=best.candidate,
            best_candidate=best.candidate,
            history=history,
            final_score=best.score,
            total_iterations=len(history),
            total_evaluations=len(history),
            metadata=metadata,
        )

    def _evaluate(
        self,
        candidate: AgentCandidate,
        evaluator: CandidateScorer,
        evaluated: dict[str, CandidateEvaluation],
        history: List[IterationHistory],
        role_counts: dict[str, int],
        *,
        role: str,
        round_number: int,
        reason: str,
        metadata: Mapping[str, Any],
    ) -> CandidateEvaluation:
        if candidate.id in evaluated:
            return evaluated[candidate.id]

        value = evaluator(candidate)
        evaluation = _normalize_candidate_evaluation(value, candidate)
        evaluation.metadata = {
            **candidate.metadata,
            **evaluation.metadata,
            "optimizer": "AgentSocialMemoryOptimizer",
            "proposal_role": role,
            "proposal_round": round_number,
            "proposal_reason": reason,
            "proposal_metadata": dict(metadata),
        }
        evaluated[candidate.id] = evaluation
        history.append(_history_from_candidate(evaluation))
        role_counts[role] = role_counts.get(role, 0) + 1
        return evaluation


def _ordered_search_paths(
    target: OptimizationTarget,
    diagnoses: Sequence[ComponentDiagnosis],
) -> List[str]:
    allowed_paths = relevant_search_paths(target.search_space, diagnoses)
    return [path for path in target.search_space if path in allowed_paths]


def _merge_history_search_paths(
    search_paths: Sequence[str],
    history: Optional[AgentObservabilityWindow],
    *,
    target: OptimizationTarget,
    seed_candidate: AgentCandidate,
) -> list[str]:
    merged = list(search_paths)
    if history is None:
        return merged
    all_paths = list(target.search_space)
    for record in history.records:
        patch = _patch_from_record(
            record,
            target=target,
            seed_candidate=seed_candidate,
            search_paths=all_paths,
        )
        for path in all_paths:
            if path in patch and path not in merged:
                merged.append(path)
    return merged


def _build_social_memory_proposals(
    *,
    seed_candidate: AgentCandidate,
    evaluations: Sequence[CandidateEvaluation],
    search_space: Mapping[str, List[Any]],
    search_paths: Sequence[str],
    diagnoses: Sequence[ComponentDiagnosis],
    ledger: Mapping[str, _PatchCredit],
    prior_proposals: Sequence[_MemoryProposal],
    beam_width: int,
    max_proposals: int,
    round_number: int,
) -> list[_MemoryProposal]:
    proposals: list[_MemoryProposal] = []
    seen: set[str] = set()
    ranked = sorted(
        evaluations,
        key=lambda item: (item.score, -len(item.candidate.patch), item.candidate.id),
        reverse=True,
    )
    changed_ranked = [item for item in ranked if item.candidate.patch]

    for proposal in prior_proposals:
        _append_proposal(proposals, seen, proposal, max_proposals)

    if round_number > 1:
        for proposal in _ledger_synthesis_proposals(
            ledger,
            search_paths=search_paths,
            evaluations=changed_ranked,
        ):
            _append_proposal(proposals, seen, proposal, max_proposals)

        for evaluation in changed_ranked[:beam_width]:
            for proposal in _critic_proposals(
                evaluation,
                search_space=search_space,
                search_paths=search_paths,
                ledger=ledger,
            ):
                _append_proposal(proposals, seen, proposal, max_proposals)

        for evaluation in changed_ranked[:beam_width]:
            for proposal in _steward_proposals(evaluation, ledger=ledger):
                _append_proposal(proposals, seen, proposal, max_proposals)

    for proposal in _specialist_proposals(
        seed_candidate,
        search_space=search_space,
        search_paths=search_paths,
        diagnoses=diagnoses,
    ):
        _append_proposal(proposals, seen, proposal, max_proposals)

    for proposal in _explorer_proposals(
        seed_candidate,
        search_space=search_space,
        search_paths=search_paths,
        ledger=ledger,
    ):
        _append_proposal(proposals, seen, proposal, max_proposals)

    if round_number > 1:
        for proposal in _adversary_proposals(
            seed_candidate,
            search_space=search_space,
            search_paths=search_paths,
        ):
            _append_proposal(proposals, seen, proposal, max_proposals)

    return proposals


def _specialist_proposals(
    seed_candidate: AgentCandidate,
    *,
    search_space: Mapping[str, List[Any]],
    search_paths: Sequence[str],
    diagnoses: Sequence[ComponentDiagnosis],
) -> Iterable[_MemoryProposal]:
    for group_key, paths in _path_groups(search_paths, diagnoses).items():
        patch: dict[str, Any] = {}
        for path in paths:
            value = _first_non_seed_value(seed_candidate, search_space, path)
            if value is not _NO_VALUE:
                patch[path] = value
        if patch:
            yield _MemoryProposal(
                patch=patch,
                role="smriti",
                parent_ids=(seed_candidate.id,),
                reason=f"apply_diagnosed_memory_bundle:{group_key}",
                metadata={
                    "role_archetype": "working_memory",
                    "role_kind": "specialist",
                },
            )


def _explorer_proposals(
    seed_candidate: AgentCandidate,
    *,
    search_space: Mapping[str, List[Any]],
    search_paths: Sequence[str],
    ledger: Mapping[str, _PatchCredit],
) -> Iterable[_MemoryProposal]:
    tested = {_credit_key(credit.path, credit.value) for credit in ledger.values()}
    for path in search_paths:
        for value in search_space.get(path, []):
            if seed_candidate.get_path(path) == value:
                continue
            yield _MemoryProposal(
                patch={path: value},
                role="arjuna",
                parent_ids=(seed_candidate.id,),
                reason="isolate_single_path_effect",
                metadata={
                    "role_archetype": "focused_action",
                    "role_kind": "explorer",
                    "previously_tested": _credit_key(path, value) in tested,
                },
            )


def _critic_proposals(
    evaluation: CandidateEvaluation,
    *,
    search_space: Mapping[str, List[Any]],
    search_paths: Sequence[str],
    ledger: Mapping[str, _PatchCredit],
) -> Iterable[_MemoryProposal]:
    source_patch = dict(evaluation.candidate.patch)
    for path in search_paths:
        if path in source_patch:
            continue
        value = _best_credit_value(path, ledger)
        if value is _NO_VALUE:
            value = _first_non_candidate_value(evaluation.candidate, search_space, path)
        if value is _NO_VALUE:
            continue
        yield _MemoryProposal(
            patch={**source_patch, path: value},
            role="vidura",
            parent_ids=(evaluation.candidate.id,),
            reason="critique_best_candidate_with_next_memory",
            metadata={
                "role_archetype": "prudent_critic",
                "role_kind": "critic",
            },
        )


def _ledger_synthesis_proposals(
    ledger: Mapping[str, _PatchCredit],
    *,
    search_paths: Sequence[str],
    evaluations: Sequence[CandidateEvaluation],
) -> Iterable[_MemoryProposal]:
    patch = _top_credit_patch(ledger, search_paths=search_paths)
    parent_ids: list[str] = []
    for evaluation in evaluations:
        if set(evaluation.candidate.patch) & set(patch):
            parent_ids.append(evaluation.candidate.id)
    if patch:
        yield _MemoryProposal(
            patch=patch,
            role="sangha",
            parent_ids=tuple(dict.fromkeys(parent_ids)),
            reason="combine_high_credit_path_memories",
            metadata={
                "role_archetype": "collective_synthesis",
                "role_kind": "synthesizer",
            },
        )


def _steward_proposals(
    evaluation: CandidateEvaluation,
    *,
    ledger: Mapping[str, _PatchCredit],
) -> Iterable[_MemoryProposal]:
    source_patch = dict(evaluation.candidate.patch)
    if len(source_patch) < 2:
        return
    ranked_paths = sorted(
        source_patch,
        key=lambda path: (
            _credit_for_value(path, source_patch[path], ledger).mean_delta
            if _credit_for_value(path, source_patch[path], ledger)
            else 0.0,
            path,
        ),
    )
    for path in ranked_paths:
        patch = {
            key: value
            for key, value in source_patch.items()
            if key != path
        }
        yield _MemoryProposal(
            patch=patch,
            role="dharma_steward",
            parent_ids=(evaluation.candidate.id,),
            reason="remove_low_credit_change_to_check_minimality",
            metadata={
                "role_archetype": "minimal_process_guardian",
                "role_kind": "steward",
                "removed_path": path,
            },
        )


def _adversary_proposals(
    seed_candidate: AgentCandidate,
    *,
    search_space: Mapping[str, List[Any]],
    search_paths: Sequence[str],
) -> Iterable[_MemoryProposal]:
    patch: dict[str, Any] = {}
    for path in search_paths:
        value = _last_non_seed_value(seed_candidate, search_space, path)
        if value is not _NO_VALUE:
            patch[path] = value
        if len(patch) >= 3:
            break
    if patch:
        yield _MemoryProposal(
            patch=patch,
            role="vidura",
            parent_ids=(seed_candidate.id,),
            reason="stress_boundary_combination",
            metadata={
                "role_archetype": "prudent_critic",
                "role_kind": "adversary",
            },
        )


def _seed_credit_from_history(
    history: Optional[AgentObservabilityWindow],
    *,
    target: OptimizationTarget,
    seed_candidate: AgentCandidate,
    search_paths: Sequence[str],
    baseline_score: float,
    ledger: dict[str, _PatchCredit],
) -> int:
    if history is None:
        return 0
    count = 0
    for record in history.records:
        patch = _patch_from_record(
            record,
            target=target,
            seed_candidate=seed_candidate,
            search_paths=search_paths,
        )
        if not patch:
            continue
        count += 1
        delta = record.score - baseline_score
        for path, value in patch.items():
            _record_credit(
                ledger,
                path=path,
                value=value,
                score=record.score,
                delta=delta,
                passed=record.passed,
                source="futureagi_history",
            )
    return count


def _prior_proposals_from_history(
    history: Optional[AgentObservabilityWindow],
    *,
    target: OptimizationTarget,
    seed_candidate: AgentCandidate,
    search_paths: Sequence[str],
) -> list[_MemoryProposal]:
    if history is None:
        return []
    proposals: list[_MemoryProposal] = []
    seen: set[str] = set()
    ranked_records = sorted(
        history.records,
        key=lambda item: (item.passed, item.score, item.run_id or "", item.index),
        reverse=True,
    )
    for record in ranked_records:
        patch = _patch_from_record(
            record,
            target=target,
            seed_candidate=seed_candidate,
            search_paths=search_paths,
        )
        if not patch:
            continue
        proposal = _MemoryProposal(
            patch=patch,
            role="smriti",
            parent_ids=tuple(filter(None, [record.candidate_id, record.run_id])),
            reason="replay_high_signal_futureagi_history_patch",
            metadata={
                "role_archetype": "working_memory",
                "role_kind": "futureagi_prior",
                "futureagi_record_index": record.index,
                "futureagi_run_id": record.run_id,
                "futureagi_record_score": record.score,
                "futureagi_record_passed": record.passed,
            },
        )
        _append_proposal(proposals, seen, proposal, max_proposals=64)
    return proposals


def _patch_from_record(
    record: AgentObservabilityRecord,
    *,
    target: OptimizationTarget,
    seed_candidate: AgentCandidate,
    search_paths: Sequence[str],
) -> dict[str, Any]:
    allowed = set(search_paths)
    for payload in _record_payloads(record):
        patch = _explicit_patch_from_payload(payload, target=target, allowed=allowed)
        if patch:
            return patch
        config = _candidate_config_from_payload(payload)
        if config:
            return _patch_from_config(
                config,
                target=target,
                seed_candidate=seed_candidate,
                allowed=allowed,
            )
    return {}


def _record_payloads(record: AgentObservabilityRecord) -> Iterable[Mapping[str, Any]]:
    yield record.metadata
    yield record.raw
    for payload in (record.metadata, record.raw):
        for key in (
            "metadata",
            "raw_variant",
            "row_values",
            "raw_row",
            "candidate",
            "variant",
            "outputs",
        ):
            value = payload.get(key)
            if isinstance(value, Mapping):
                yield value


def _explicit_patch_from_payload(
    payload: Mapping[str, Any],
    *,
    target: OptimizationTarget,
    allowed: set[str],
) -> dict[str, Any]:
    for key in (
        "candidate_patch",
        "config_patch",
        "patch",
        "agent_patch",
        "optimized_patch",
    ):
        value = payload.get(key)
        if isinstance(value, Mapping):
            patch = {
                str(path): copy.deepcopy(patch_value)
                for path, patch_value in value.items()
                if str(path) in allowed
                and _value_allowed(str(path), patch_value, target.search_space)
            }
            if patch:
                return patch
    return {}


def _candidate_config_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    for key in (
        "candidate_config",
        "config",
        "agent_config",
        "optimized_config",
        "workflow_config",
    ):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return copy.deepcopy(dict(value))
    return {}


def _patch_from_config(
    config: Mapping[str, Any],
    *,
    target: OptimizationTarget,
    seed_candidate: AgentCandidate,
    allowed: set[str],
) -> dict[str, Any]:
    candidate = AgentCandidate.from_config(dict(config), target_name=target.name)
    patch: dict[str, Any] = {}
    for path in target.search_space:
        if path not in allowed:
            continue
        value = candidate.get_path(path, _NO_VALUE)
        if value is _NO_VALUE:
            continue
        if value == seed_candidate.get_path(path):
            continue
        if not _value_allowed(path, value, target.search_space):
            continue
        patch[path] = copy.deepcopy(value)
    return patch


def _record_credit_from_evaluation(
    evaluation: CandidateEvaluation,
    *,
    baseline_score: float,
    ledger: dict[str, _PatchCredit],
    source: str,
) -> None:
    if not evaluation.candidate.patch:
        return
    delta = evaluation.score - baseline_score
    passed = evaluation.score >= baseline_score
    for path, value in evaluation.candidate.patch.items():
        _record_credit(
            ledger,
            path=path,
            value=value,
            score=evaluation.score,
            delta=delta,
            passed=passed,
            source=source,
        )


def _record_credit(
    ledger: dict[str, _PatchCredit],
    *,
    path: str,
    value: Any,
    score: float,
    delta: float,
    passed: bool,
    source: str,
) -> None:
    key = _credit_key(path, value)
    credit = ledger.get(key)
    if credit is None:
        credit = _PatchCredit(path=path, value=copy.deepcopy(value))
        ledger[key] = credit
    credit.observations += 1
    credit.total_delta += delta
    credit.total_score += score
    credit.best_score = max(credit.best_score, score)
    if passed:
        credit.passed += 1
    else:
        credit.failed += 1
    credit.sources.add(source)


def _top_credit_patch(
    ledger: Mapping[str, _PatchCredit],
    *,
    search_paths: Sequence[str],
) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    for path in search_paths:
        credit = _best_credit(path, ledger)
        if credit is None:
            continue
        if credit.mean_delta <= 0.0 and credit.passed <= credit.failed:
            continue
        patch[path] = copy.deepcopy(credit.value)
    return patch


def _best_credit_value(path: str, ledger: Mapping[str, _PatchCredit]) -> Any:
    credit = _best_credit(path, ledger)
    if credit is None:
        return _NO_VALUE
    return copy.deepcopy(credit.value)


def _best_credit(
    path: str,
    ledger: Mapping[str, _PatchCredit],
) -> Optional[_PatchCredit]:
    candidates = [credit for credit in ledger.values() if credit.path == path]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda credit: (
            credit.mean_delta,
            credit.best_score,
            credit.mean_score,
            credit.passed,
            -credit.failed,
            _canonical_value(credit.value),
        ),
    )


def _credit_for_value(
    path: str,
    value: Any,
    ledger: Mapping[str, _PatchCredit],
) -> Optional[_PatchCredit]:
    return ledger.get(_credit_key(path, value))


def _path_groups(
    search_paths: Sequence[str],
    diagnoses: Sequence[ComponentDiagnosis],
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for path in search_paths:
        group_key = _diagnostic_group_key(path, diagnoses) or path.split(".", 1)[0]
        groups.setdefault(group_key, []).append(path)
    return groups


def _diagnostic_group_key(
    path: str,
    diagnoses: Sequence[ComponentDiagnosis],
) -> Optional[str]:
    for diagnosis in diagnoses:
        for suggested_path in diagnosis.suggested_paths:
            if path == suggested_path or path.startswith(f"{suggested_path}."):
                return f"{diagnosis.component}:{suggested_path}"
        if path == diagnosis.component or path.startswith(f"{diagnosis.component}."):
            return diagnosis.component
    return None


def _first_non_seed_value(
    seed_candidate: AgentCandidate,
    search_space: Mapping[str, List[Any]],
    path: str,
) -> Any:
    current = seed_candidate.get_path(path)
    for value in search_space.get(path, []):
        if value != current:
            return copy.deepcopy(value)
    return _NO_VALUE


def _last_non_seed_value(
    seed_candidate: AgentCandidate,
    search_space: Mapping[str, List[Any]],
    path: str,
) -> Any:
    current = seed_candidate.get_path(path)
    for value in reversed(search_space.get(path, [])):
        if value != current:
            return copy.deepcopy(value)
    return _NO_VALUE


def _first_non_candidate_value(
    candidate: AgentCandidate,
    search_space: Mapping[str, List[Any]],
    path: str,
) -> Any:
    current = candidate.get_path(path)
    for value in search_space.get(path, []):
        if value != current:
            return copy.deepcopy(value)
    return _NO_VALUE


def _append_proposal(
    proposals: list[_MemoryProposal],
    seen: set[str],
    proposal: _MemoryProposal,
    max_proposals: int,
) -> None:
    if len(proposals) >= max_proposals or not proposal.patch:
        return
    key = _canonical_patch(proposal.patch)
    if key in seen:
        return
    seen.add(key)
    proposals.append(proposal)


def _candidate_id_for_patch(
    seed_candidate: AgentCandidate,
    patch: dict[str, Any],
) -> str:
    return seed_candidate.with_patch(patch).id


def _value_allowed(
    path: str,
    value: Any,
    search_space: Mapping[str, List[Any]],
) -> bool:
    return any(candidate_value == value for candidate_value in search_space.get(path, []))


def _history_baseline_score(
    history: Optional[AgentObservabilityWindow],
) -> float:
    if history is None or history.average_score is None:
        return 0.0
    return history.average_score


def _ledger_summary(ledger: Mapping[str, _PatchCredit]) -> list[dict[str, Any]]:
    return [
        {
            "path": credit.path,
            "value": copy.deepcopy(credit.value),
            "observations": credit.observations,
            "mean_delta": credit.mean_delta,
            "mean_score": credit.mean_score,
            "best_score": credit.best_score,
            "passed": credit.passed,
            "failed": credit.failed,
            "sources": sorted(credit.sources),
        }
        for credit in sorted(
            ledger.values(),
            key=lambda item: (
                item.mean_delta,
                item.best_score,
                item.path,
                _canonical_value(item.value),
            ),
            reverse=True,
        )
    ]


def _credit_key(path: str, value: Any) -> str:
    return f"{path}:{_canonical_value(value)}"


def _canonical_patch(patch: Mapping[str, Any]) -> str:
    return json.dumps(patch, sort_keys=True, default=str)


def _canonical_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


class _NoValue:
    pass


_NO_VALUE = _NoValue()
