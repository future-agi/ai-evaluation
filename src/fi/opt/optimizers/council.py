from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence

from ..base.base_optimizer import BaseOptimizer
from ..components import ComponentDiagnosis, relevant_search_paths
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


DEFAULT_COUNCIL_ROLES = ("explorer", "critic", "synthesizer", "steward")
SOCIETY_ROLES = (
    "explorer",
    "critic",
    "synthesizer",
    "steward",
    "specialist",
    "adversary",
)

# Phase 4 society vocabulary (scholarly design devices used as deterministic
# engineering metadata — psychometric/philological grounding only, zero
# doctrinal claims).

GUNA_AXES = ("rajas", "sattva", "tamas")

GUNA_ARCHETYPE_DEFAULTS: dict[str, tuple[float, float, float]] = {
    # (rajas, sattva, tamas) — dominant axis per the Phase-4 architecture
    # archetype-default table (canon home; values stay byte-identical).
    "focused_action": (0.8, 0.4, 0.2),          # arjuna — explorer
    "prudent_critic": (0.7, 0.5, 0.4),          # vidura — adversary
    "orchestrator": (0.5, 0.6, 0.4),            # sutradhara
    "working_memory": (0.4, 0.6, 0.5),          # smriti
    "bridge_builder": (0.6, 0.5, 0.3),          # hanuman
    "charioteer_counsel": (0.3, 0.8, 0.4),      # krishna — critic
    "collective_synthesis": (0.2, 0.9, 0.3),    # sangha — synthesizer
    "minimal_process_guardian": (0.1, 0.5, 0.9),# dharma_steward — steward
    "": (0.5, 0.5, 0.5),
}

CHAMBER_TOKENS = ("samiti", "sabha")

# Chambers are ORTHOGONAL to phases/stages: within every phase samiti roles
# generate widely and sabha roles deliberate/promote — chamber derives from
# role kind, never from phase.
SAMITI_PROPOSAL_KINDS = frozenset({"specialist", "explorer", "adversary"})
SABHA_PROPOSAL_KINDS = frozenset(
    {"critic", "synthesizer", "coverage_synthesis", "steward"}
)

PANCA_AVAYAVA_MEMBERS = (
    # Five-member (panca-avayava) proposal justification — Nyaya-Sutra syllogism
    # structure used as an auditable record schema (Pramana arXiv:2604.04937
    # operationalization precedent). Scholarly design device, not a doctrinal
    # claim.
    "pratijna",   # claim: what this patch asserts will improve
    "hetu",       # reason: the diagnosis/metric evidence relied on
    "udaharana",  # rule + example: the prior candidate/row exhibiting the rule
    "upanaya",    # application: why the rule covers THIS candidate
    "nigamana",   # conclusion: the expected admissible evidence delta
)

HETVABHASA_REJECTION_CLASSES = (
    "savyabhichara",   # inconclusive reason: evidence does not discriminate candidates
    "viruddha",        # contradictory reason: evidence contradicts the claim
    "satpratipaksha",  # counterbalanced: an equal counter-justification exists
    "asiddha",         # unestablished reason: cited evidence/row not found in lineage
    "badhita",         # defeated: claim contradicted by a stronger admissible check
)

CRITIQUE_OPERATOR_CLASSES = (
    "vada",     # truth-seeking review — critic
    "jalpa",    # adversarial stress — adversary; findings admissible only via evidence
    "vitanda",  # refutation-only veto pass — may reject, never proposes
)


@dataclass(frozen=True)
class AgentSearchProposal:
    """One deterministic candidate patch proposed by an agent-search strategy."""

    patch: dict[str, Any]
    role: str
    parent_ids: tuple[str, ...]
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentSearchState:
    """Read-only state passed to a pluggable agent-search strategy."""

    seed_candidate: AgentCandidate
    evaluations: Sequence[CandidateEvaluation]
    search_space: Mapping[str, List[Any]]
    search_paths: Sequence[str]
    diagnoses: Sequence[ComponentDiagnosis]
    beam_width: int
    max_proposals: int
    round_number: int
    # Phase 4: round-scoped pooled-diagnosis society ledger (GEA experience
    # pooling). None = legacy behavior.
    ledger: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class AgentSocietyRole:
    """One role node in a deterministic society-search proposal graph."""

    name: str
    proposal_kind: str
    phase: int = 1
    depends_on: tuple[str, ...] = ()
    path_prefixes: tuple[str, ...] = ()
    archetype: str = ""
    description: str = ""
    # Phase 4 (placed after the legacy fields so positional construction stays
    # byte-compatible): ONE nested optional guna mapping — {"rajas", "sattva",
    # "tamas"} each in [0, 1]; None/absent = derive from archetype defaults —
    # and an optional chamber ("samiti" | "sabha"); None = derive from role
    # kind. No sentinel values.
    guna: Optional[Mapping[str, float]] = None
    chamber: Optional[str] = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "proposal_kind": self.proposal_kind,
            "phase": self.phase,
            "depends_on": list(self.depends_on),
            "path_prefixes": list(self.path_prefixes),
            "archetype": self.archetype,
            "description": self.description,
            "guna": dict(self.guna) if self.guna is not None else None,
            "chamber": self.chamber,
        }


ROLE_GRAPH_PROPOSAL_KINDS = {
    "adversary",
    "coverage_synthesis",
    "critic",
    "explorer",
    "specialist",
    "steward",
    "synthesizer",
}


DEFAULT_SOCIETY_ROLE_GRAPH = (
    AgentSocietyRole(
        name="sutradhara",
        proposal_kind="specialist",
        phase=1,
        path_prefixes=("multi_agent", "orchestration", "router", "graph"),
        archetype="orchestrator",
        description="Bundle coordination, routing, handoff, and graph repairs.",
    ),
    AgentSocietyRole(
        name="smriti",
        proposal_kind="specialist",
        phase=1,
        path_prefixes=("memory", "retrieval", "retriever"),
        archetype="working_memory",
        description="Bundle memory, retrieval, and retained context repairs.",
    ),
    AgentSocietyRole(
        name="arjuna",
        proposal_kind="explorer",
        phase=1,
        archetype="focused_action",
        description="Probe one controllable path at a time under metric feedback.",
    ),
    AgentSocietyRole(
        name="hanuman",
        proposal_kind="specialist",
        phase=1,
        path_prefixes=("tools", "framework", "voice", "browser", "cua", "implementation"),
        archetype="bridge_builder",
        description="Bundle tool, framework, world-interface, and runtime repairs.",
    ),
    AgentSocietyRole(
        name="vidura",
        proposal_kind="adversary",
        phase=1,
        path_prefixes=("security", "policy", "trust", "environment"),
        archetype="prudent_critic",
        description="Stress policy, security, trust-boundary, and environment choices.",
    ),
    AgentSocietyRole(
        name="krishna",
        proposal_kind="critic",
        phase=2,
        depends_on=("arjuna", "sutradhara", "smriti"),
        archetype="charioteer_counsel",
        description="Test one more change against current strong partial candidates.",
    ),
    AgentSocietyRole(
        name="sangha",
        proposal_kind="coverage_synthesis",
        phase=2,
        depends_on=("sutradhara", "smriti", "hanuman", "vidura", "arjuna"),
        archetype="collective_synthesis",
        description="Combine best path representatives across role evidence.",
    ),
    AgentSocietyRole(
        name="dharma_steward",
        proposal_kind="steward",
        phase=3,
        depends_on=("sangha", "krishna"),
        archetype="minimal_process_guardian",
        description="Remove one change at a time to keep only metric-proven repairs.",
    ),
)


class AgentSearchStrategy:
    """Proposal-generation strategy for framework-neutral agent optimization."""

    name = "agent_search_strategy"
    roles: Sequence[str] = ()

    def propose(self, state: AgentSearchState) -> List[AgentSearchProposal]:
        raise NotImplementedError


class DeterministicCouncilStrategy(AgentSearchStrategy):
    """Current council search: explore, critique, synthesize, and steward."""

    name = "deterministic_council_search"
    roles = DEFAULT_COUNCIL_ROLES

    def propose(self, state: AgentSearchState) -> List[AgentSearchProposal]:
        return _build_round_proposals(
            seed_candidate=state.seed_candidate,
            evaluations=state.evaluations,
            search_space=dict(state.search_space),
            search_paths=state.search_paths,
            beam_width=state.beam_width,
            max_proposals=state.max_proposals,
            round_number=state.round_number,
        )


class SocietySearchStrategy(AgentSearchStrategy):
    """
    Deterministic role-diverse search for multi-interaction agent systems.

    The strategy keeps metric-bound candidate evaluation, but allocates proposal
    slots across social roles so search can test isolated mutations, component
    bundles, stress combinations, synthesis, critique, and simplification.
    """

    name = "deterministic_society_search"
    roles = SOCIETY_ROLES

    def propose(self, state: AgentSearchState) -> List[AgentSearchProposal]:
        return _build_society_proposals(
            seed_candidate=state.seed_candidate,
            evaluations=state.evaluations,
            search_space=dict(state.search_space),
            search_paths=state.search_paths,
            diagnoses=state.diagnoses,
            beam_width=state.beam_width,
            max_proposals=state.max_proposals,
            round_number=state.round_number,
            ledger=state.ledger,
        )


class SocietyRoleGraphSearchStrategy(AgentSearchStrategy):
    """
    Deterministic society search with explicit role graph metadata.

    Role names and archetypes are inspiration labels only. Candidate acceptance
    still depends entirely on the provided metric/evaluator contract.
    """

    name = "deterministic_role_graph_society_search"

    def __init__(
        self,
        role_graph: Optional[Sequence[AgentSocietyRole | Mapping[str, Any]]] = None,
        *,
        max_paths_per_proposal: int = 1,
        staged_conditioning: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if max_paths_per_proposal < 1:
            raise ValueError("max_paths_per_proposal must be at least 1.")
        self.role_graph = _normalize_society_role_graph(role_graph)
        self.roles = tuple(role.name for role in self.role_graph)
        # Guna patch-radius base: explorer/adversary streams propose
        # max(1, round(rajas * max_paths_per_proposal)) paths per patch. The
        # default of 1 reproduces the legacy single-path radius for every
        # default-archetype triple.
        self.max_paths_per_proposal = max_paths_per_proposal
        # 4C staged conditioning declaration (stage -> phase -> path-class);
        # the strategy EXECUTES stages through role-graph phases — this is the
        # declared map the optimizer trace proves the order from.
        self.staged_conditioning = (
            dict(staged_conditioning) if staged_conditioning is not None else None
        )

    def propose(self, state: AgentSearchState) -> List[AgentSearchProposal]:
        return _build_role_graph_society_proposals(
            seed_candidate=state.seed_candidate,
            evaluations=state.evaluations,
            search_space=dict(state.search_space),
            search_paths=state.search_paths,
            diagnoses=state.diagnoses,
            beam_width=state.beam_width,
            max_proposals=state.max_proposals,
            round_number=state.round_number,
            role_graph=self.role_graph,
            ledger=state.ledger,
            max_paths_per_proposal=self.max_paths_per_proposal,
        )

    def to_metadata(self) -> dict[str, Any]:
        metadata = {
            "role_graph": [role.to_metadata() for role in self.role_graph],
            "role_graph_inspiration": (
                "human social coordination, metacognition, and Hindu mythic "
                "archetypes used only as deterministic proposal metadata"
            ),
            "guna_mix": _guna_mix(self.role_graph),
            "chambers": {
                chamber: [
                    role.name
                    for role in self.role_graph
                    if (role.chamber or _chamber_for_proposal_kind(role.proposal_kind))
                    == chamber
                ]
                for chamber in CHAMBER_TOKENS
            },
            "max_paths_per_proposal": self.max_paths_per_proposal,
        }
        if self.staged_conditioning is not None:
            metadata["staged_conditioning"] = dict(self.staged_conditioning)
        return metadata


class CouncilAgentOptimizer(BaseOptimizer):
    """
    Optimizes agent configs with deterministic multi-round social search.

    `AgentOptimizer` is best when exhaustive candidate enumeration is acceptable.
    This optimizer is intended for multi-interaction agents where useful fixes
    are often combinations of partial changes: one role explores isolated
    mutations, one critiques the current best candidate, one synthesizes strong
    partial candidates, and one steward tests whether combined patches can be
    simplified without losing score.
    """

    def __init__(
        self,
        target: Optional[OptimizationTarget] = None,
        *,
        evaluate_candidate: Optional[
            Callable[[AgentCandidate], CandidateEvaluation | EvaluationResult | float]
        ] = None,
        simulation_evaluator: Any = None,
        diagnoses: Optional[Iterable[ComponentDiagnosis | dict[str, Any]]] = None,
        max_rounds: int = 3,
        beam_width: int = 4,
        max_proposals_per_round: int = 16,
        target_score: float = 1.0,
        include_seed: bool = True,
        auto_diagnose: bool = True,
        diagnostic_score_threshold: float = 0.85,
        search_strategy: Optional[AgentSearchStrategy | str] = None,
        samiti_budget: Optional[int] = None,
        sabha_budget: Optional[int] = None,
        society_ledger: bool = False,
        social_memory: Optional[Any] = None,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1.")
        if beam_width < 1:
            raise ValueError("beam_width must be at least 1.")
        if max_proposals_per_round < 1:
            raise ValueError("max_proposals_per_round must be at least 1.")
        _validate_chamber_budgets(samiti_budget, sabha_budget)

        self.target = target
        self.evaluate_candidate = evaluate_candidate
        self.simulation_evaluator = simulation_evaluator
        self.diagnoses = _normalize_diagnoses(diagnoses)
        self.max_rounds = max_rounds
        self.beam_width = beam_width
        self.max_proposals_per_round = max_proposals_per_round
        self.target_score = target_score
        self.include_seed = include_seed
        self.auto_diagnose = auto_diagnose
        self.diagnostic_score_threshold = diagnostic_score_threshold
        self.search_strategy = _resolve_search_strategy(search_strategy)
        self.samiti_budget = samiti_budget
        self.sabha_budget = sabha_budget
        self.society_ledger = society_ledger
        self.social_memory = social_memory
        super().__init__()

    def optimize(
        self,
        evaluator: Any = None,
        data_mapper: Any = None,
        dataset: Optional[List[dict[str, Any]]] = None,
        metric: Optional[Callable] = None,
        *,
        target: Optional[OptimizationTarget] = None,
        evaluate_candidate: Optional[
            Callable[[AgentCandidate], CandidateEvaluation | EvaluationResult | float]
        ] = None,
        simulation_evaluator: Any = None,
        diagnoses: Optional[Iterable[ComponentDiagnosis | dict[str, Any]]] = None,
        max_rounds: Optional[int] = None,
        beam_width: Optional[int] = None,
        max_proposals_per_round: Optional[int] = None,
        target_score: Optional[float] = None,
        include_seed: Optional[bool] = None,
        auto_diagnose: Optional[bool] = None,
        diagnostic_score_threshold: Optional[float] = None,
        search_strategy: Optional[AgentSearchStrategy | str] = None,
        samiti_budget: Optional[int] = None,
        sabha_budget: Optional[int] = None,
        society_ledger: Optional[bool] = None,
        social_memory: Optional[Any] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        active_target = target or self.target
        if active_target is None:
            raise ValueError("CouncilAgentOptimizer requires a target.")

        active_evaluator = (
            evaluate_candidate
            or self.evaluate_candidate
            or getattr(simulation_evaluator, "evaluate_candidate", None)
            or getattr(self.simulation_evaluator, "evaluate_candidate", None)
        )
        if active_evaluator is None:
            raise ValueError(
                "CouncilAgentOptimizer requires evaluate_candidate or simulation_evaluator."
            )

        active_diagnoses = _normalize_diagnoses(diagnoses)
        if diagnoses is None:
            active_diagnoses = list(self.diagnoses)
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
        active_search_strategy = (
            self.search_strategy
            if search_strategy is None
            else _resolve_search_strategy(search_strategy)
        )
        active_samiti_budget = (
            self.samiti_budget if samiti_budget is None else samiti_budget
        )
        active_sabha_budget = (
            self.sabha_budget if sabha_budget is None else sabha_budget
        )
        _validate_chamber_budgets(active_samiti_budget, active_sabha_budget)
        use_society_ledger = (
            self.society_ledger if society_ledger is None else bool(society_ledger)
        )
        active_social_memory = (
            self.social_memory if social_memory is None else social_memory
        )

        seed_candidate = active_target.seed_candidate()
        evaluated: dict[str, CandidateEvaluation] = {}
        history: List[IterationHistory] = []
        role_counts: dict[str, int] = {}
        round_summaries: List[dict[str, Any]] = []
        best: CandidateEvaluation | None = None

        role_chambers = _strategy_role_chambers(active_search_strategy)
        chamber_budgets = {
            "samiti": active_samiti_budget,
            "sabha": active_sabha_budget,
        }
        chamber_used = {"samiti": 0, "sabha": 0}
        chamber_skipped = {"samiti": 0, "sabha": 0}
        rejections: List[dict[str, Any]] = []
        ledger_rounds: List[dict[str, Any]] = []
        current_ledger: Optional[dict[str, Any]] = None
        persisted_via: Optional[str] = None
        if use_society_ledger and active_social_memory is not None:
            persisted_via = active_social_memory.__class__.__name__
            prior_ledgers = list(
                getattr(active_social_memory, "society_ledgers", None) or []
            )
            prior_diagnoses = [
                dict(item)
                for entry in prior_ledgers
                if isinstance(entry, Mapping)
                for item in entry.get("diagnoses", []) or []
                if isinstance(item, Mapping)
            ]
            if prior_diagnoses:
                # Cross-campaign preload: previously persisted ledgers seed
                # round 1 of this campaign (GEA experience pooling).
                current_ledger = {
                    "round": 0,
                    "diagnoses": prior_diagnoses,
                    "pooled_from_candidates": sum(
                        int(entry.get("pooled_from_candidates") or 0)
                        for entry in prior_ledgers
                        if isinstance(entry, Mapping)
                    ),
                    "preloaded": True,
                }

        if use_include_seed:
            seed_evaluation = self._evaluate(
                seed_candidate,
                active_evaluator,
                evaluated,
                history,
                role_counts,
                role="seed",
                round_number=0,
            )
            best = seed_evaluation
            if use_auto_diagnose and not active_diagnoses:
                active_diagnoses = _diagnose_candidate_evaluation(
                    seed_evaluation,
                    failing_threshold=active_diagnostic_threshold,
                )

        search_paths = _ordered_search_paths(active_target, active_diagnoses)
        if not search_paths:
            raise ValueError("CouncilAgentOptimizer target search space cannot be empty.")

        for round_number in range(1, active_max_rounds + 1):
            proposals = active_search_strategy.propose(
                AgentSearchState(
                    seed_candidate=seed_candidate,
                    evaluations=list(evaluated.values()),
                    search_space=active_target.search_space,
                    search_paths=search_paths,
                    diagnoses=active_diagnoses,
                    beam_width=active_beam_width,
                    max_proposals=active_max_proposals,
                    round_number=round_number,
                    ledger=current_ledger,
                )
            )
            admitted_proposals: List[AgentSearchProposal] = []
            for proposal in proposals:
                duplicate_id = _candidate_id_for_patch(seed_candidate, proposal.patch)
                if duplicate_id in evaluated:
                    rejections.append(
                        {
                            "round": round_number,
                            "role": proposal.role,
                            "candidate_id": duplicate_id,
                            "rejected": True,
                            "hetvabhasa_class": "savyabhichara",
                            "detail": (
                                "duplicate patch: evidence does not discriminate "
                                "from an already-evaluated candidate"
                            ),
                        }
                    )
                    continue
                admitted_proposals.append(proposal)
            proposals = admitted_proposals
            logger.info(
                "Council round %s evaluating %s proposal(s)",
                round_number,
                len(proposals),
            )

            round_best = best
            round_evaluated = 0
            round_evaluations: List[CandidateEvaluation] = []
            for proposal in proposals:
                allowed_paths = set(search_paths)
                if proposal.patch and not (set(proposal.patch) & allowed_paths):
                    # Locality breach is recorded (asiddha) but not enforced
                    # here — promotion-time enforcement is the replay veto's
                    # job; in-round evidence must stay visible.
                    rejections.append(
                        {
                            "round": round_number,
                            "role": proposal.role,
                            "candidate_id": _candidate_id_for_patch(
                                seed_candidate, proposal.patch
                            ),
                            "rejected": False,
                            "hetvabhasa_class": "asiddha",
                            "detail": (
                                "patch touches no path inside the diagnosed "
                                "search locality"
                            ),
                        }
                    )
                chamber = _proposal_chamber(proposal, role_chambers)
                candidate = seed_candidate.with_patch(
                    proposal.patch,
                    metadata={
                        "kind": "council_proposal",
                        "optimizer": self.__class__.__name__,
                        "proposal_role": proposal.role,
                        "proposal_reason": proposal.reason,
                        "proposal_round": round_number,
                        "proposal_parent_ids": list(proposal.parent_ids),
                        "proposal_metadata": dict(proposal.metadata),
                    },
                )
                is_new_candidate = candidate.id not in evaluated
                declared_chamber_budget = chamber_budgets.get(chamber)
                if (
                    is_new_candidate
                    and declared_chamber_budget is not None
                    and chamber_used[chamber] >= declared_chamber_budget
                ):
                    chamber_skipped[chamber] += 1
                    continue
                evaluation = self._evaluate(
                    candidate,
                    active_evaluator,
                    evaluated,
                    history,
                    role_counts,
                    role=proposal.role,
                    round_number=round_number,
                )
                if is_new_candidate:
                    chamber_used[chamber] += 1
                    round_evaluations.append(evaluation)
                parent_scores = [
                    evaluated[parent_id].score
                    for parent_id in proposal.parent_ids
                    if parent_id in evaluated and parent_id != candidate.id
                ]
                role_kind = str(
                    proposal.metadata.get("role_kind") or proposal.role
                )
                if parent_scores and evaluation.score < max(parent_scores):
                    rejections.append(
                        {
                            "round": round_number,
                            "role": proposal.role,
                            "candidate_id": candidate.id,
                            "rejected": True,
                            "hetvabhasa_class": "viruddha",
                            "detail": (
                                f"score {evaluation.score:.4f} regresses parent "
                                f"best {max(parent_scores):.4f}"
                            ),
                        }
                    )
                elif (
                    role_kind == "steward"
                    and parent_scores
                    and evaluation.score == max(parent_scores)
                ):
                    rejections.append(
                        {
                            "round": round_number,
                            "role": proposal.role,
                            "candidate_id": proposal.parent_ids[0]
                            if proposal.parent_ids
                            else candidate.id,
                            "rejected": True,
                            "hetvabhasa_class": "satpratipaksha",
                            "detail": (
                                "steward removal kept the score unchanged: the "
                                "removed change carries an equal counter-"
                                "justification"
                            ),
                        }
                    )
                if round_best is None or evaluation.score > round_best.score:
                    round_best = evaluation
                round_evaluated += 1
                if best is None or evaluation.score > best.score:
                    best = evaluation
                    logger.info(
                        "New best council candidate %s score=%.4f",
                        candidate.id,
                        evaluation.score,
                    )
                if best.score >= active_target_score:
                    break

            if use_society_ledger:
                pooled_diagnoses: List[ComponentDiagnosis] = []
                contributing = 0
                for evaluation in round_evaluations:
                    candidate_diagnoses = _diagnose_candidate_evaluation(
                        evaluation,
                        failing_threshold=active_diagnostic_threshold,
                    )
                    if candidate_diagnoses:
                        contributing += 1
                        pooled_diagnoses.extend(candidate_diagnoses)
                round_ledger = {
                    "round": round_number,
                    "diagnoses": [
                        _dump_model(item)
                        for item in _dedupe_diagnoses(pooled_diagnoses)
                    ],
                    "pooled_from_candidates": len(round_evaluations),
                    "contributing_candidates": contributing,
                }
                ledger_rounds.append(
                    {
                        "round": round_number,
                        "diagnoses_pooled": len(round_ledger["diagnoses"]),
                        "pooled_from_candidates": round_ledger[
                            "pooled_from_candidates"
                        ],
                        "persisted_via": persisted_via,
                    }
                )
                current_ledger = round_ledger
                if active_social_memory is not None:
                    society_ledgers = getattr(
                        active_social_memory, "society_ledgers", None
                    )
                    if society_ledgers is None:
                        society_ledgers = []
                        active_social_memory.society_ledgers = society_ledgers
                    society_ledgers.append(dict(round_ledger))

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

            round_summaries.append(
                {
                    "round": round_number,
                    "proposals": len(proposals),
                    "evaluated": round_evaluated,
                    "best_score": best.score if best is not None else None,
                    "search_paths": list(search_paths),
                }
            )
            if best is not None and best.score >= active_target_score:
                break

        if best is None:
            raise ValueError("CouncilAgentOptimizer did not evaluate any candidates.")

        strategy_name = getattr(
            active_search_strategy,
            "name",
            active_search_strategy.__class__.__name__,
        )
        strategy_roles = list(getattr(active_search_strategy, "roles", ()))
        metadata = {
            "optimizer": self.__class__.__name__,
            "strategy": strategy_name,
            "roles": strategy_roles,
            "target_name": best.candidate.target_name,
            "best_candidate_id": best.candidate.id,
            "search_paths": list(search_paths),
            "rounds": round_summaries,
            "beam_width": active_beam_width,
            "max_proposals_per_round": active_max_proposals,
            "role_evaluations": role_counts,
        }
        strategy_metadata = _strategy_metadata(active_search_strategy)
        if strategy_metadata:
            metadata["strategy_metadata"] = strategy_metadata
            if "role_graph" in strategy_metadata:
                metadata["role_graph"] = strategy_metadata["role_graph"]
            if "guna_mix" in strategy_metadata:
                metadata["guna_mix"] = strategy_metadata["guna_mix"]
        if active_diagnoses:
            metadata["diagnostics"] = [_dump_model(item) for item in active_diagnoses]
            metadata["auto_diagnosed"] = use_auto_diagnose

        # Phase 4 society/governance surfaces (additive; ranking comes only
        # from the evaluation suite — external-verification rule).
        metadata["ranking_source"] = "evaluation_suite"
        metadata["chambers"] = {
            chamber: {
                "roles": sorted(
                    name
                    for name, value in role_chambers.items()
                    if value == chamber
                    and name in set(strategy_roles) | set(role_counts)
                ),
                "declared_budget": chamber_budgets[chamber],
                "evaluations_used": chamber_used[chamber],
                "skipped_proposals": chamber_skipped[chamber],
            }
            for chamber in CHAMBER_TOKENS
        }
        if rejections:
            metadata["rejections"] = rejections
        if ledger_rounds:
            metadata["ledger_rounds"] = ledger_rounds
            metadata["society_ledger"] = True
        metadata["nirnaya"] = [
            {
                "round": round_summaries[-1]["round"] if round_summaries else 1,
                "decision": "promote",
                "selected_candidate_id": best.candidate.id,
                "justification": _justification(
                    pratijna=(
                        f"candidate {best.candidate.id} is the promotable winner"
                    ),
                    hetu=(
                        f"top admissible evaluation score {best.score:.4f} from "
                        "the evaluation suite"
                    ),
                    udaharana=(
                        "rule: selection is single-lineage — the steward promotes "
                        "the top-ranked evidence-backed candidate, never an average"
                    ),
                    upanaya=(
                        f"candidate {best.candidate.id} holds the top rank in this "
                        "run's lineage"
                    ),
                    nigamana=(
                        "promotion is expected to re-close every frozen evidence "
                        "row on replay"
                    ),
                ),
                "rejected_alternatives": [
                    {
                        "candidate_id": rejection.get("candidate_id"),
                        "hetvabhasa_class": rejection.get("hetvabhasa_class"),
                    }
                    for rejection in rejections
                    if rejection.get("rejected")
                ],
                "replay_verdict": None,
                "admissible_evidence_refs": [best.candidate.id],
                "frozen_rows_closed": None,
            }
        ]

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
        evaluator: Callable[[AgentCandidate], CandidateEvaluation | EvaluationResult | float],
        evaluated: dict[str, CandidateEvaluation],
        history: List[IterationHistory],
        role_counts: dict[str, int],
        *,
        role: str,
        round_number: int,
    ) -> CandidateEvaluation:
        if candidate.id in evaluated:
            return evaluated[candidate.id]

        value = evaluator(candidate)
        evaluation = _normalize_candidate_evaluation(value, candidate)
        evaluation.metadata = {
            **candidate.metadata,
            **evaluation.metadata,
            "proposal_role": role,
            "proposal_round": round_number,
        }
        evaluated[candidate.id] = evaluation
        history.append(_history_from_candidate(evaluation))
        role_counts[role] = role_counts.get(role, 0) + 1
        return evaluation


class SocietyAgentOptimizer(CouncilAgentOptimizer):
    """
    Council optimizer preset using role-diverse society search.

    It is deterministic by default and uses the same `OptimizationTarget` and
    evaluator contracts as `AgentOptimizer`/`CouncilAgentOptimizer`.
    """

    def __init__(
        self,
        *args: Any,
        search_strategy: Optional[AgentSearchStrategy | str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            search_strategy=search_strategy or SocietySearchStrategy(),
            **kwargs,
        )


def _ordered_search_paths(
    target: OptimizationTarget,
    diagnoses: Sequence[ComponentDiagnosis],
) -> List[str]:
    allowed_paths = relevant_search_paths(target.search_space, diagnoses)
    return [path for path in target.search_space if path in allowed_paths]


def _resolve_search_strategy(
    strategy: Optional[AgentSearchStrategy | str | Mapping[str, Any]],
) -> AgentSearchStrategy:
    if strategy is None or strategy == "council":
        return DeterministicCouncilStrategy()
    if strategy == "society":
        return SocietySearchStrategy()
    if isinstance(strategy, str) and strategy in {"role_graph", "society_role_graph"}:
        return SocietyRoleGraphSearchStrategy()
    if isinstance(strategy, AgentSearchStrategy):
        return strategy
    if isinstance(strategy, Mapping):
        # Phase 4 (extend-only): JSON-declarable strategy — lets optimization
        # manifests declare a staged role-graph society search without
        # constructing strategy objects.
        token = str(
            strategy.get("strategy")
            or strategy.get("name")
            or strategy.get("type")
            or "role_graph"
        )
        if token not in {"role_graph", "society_role_graph"}:
            return _resolve_search_strategy(token)
        return SocietyRoleGraphSearchStrategy(
            strategy.get("role_graph"),
            max_paths_per_proposal=int(strategy.get("max_paths_per_proposal", 1)),
            staged_conditioning=strategy.get("staged_conditioning"),
        )
    if hasattr(strategy, "propose"):
        return strategy  # type: ignore[return-value]
    raise ValueError(
        "search_strategy must be 'council', 'society', 'role_graph', "
        "'society_role_graph', a strategy mapping, or an AgentSearchStrategy."
    )


def _strategy_metadata(strategy: AgentSearchStrategy) -> dict[str, Any]:
    to_metadata = getattr(strategy, "to_metadata", None)
    if callable(to_metadata):
        metadata = to_metadata()
        if isinstance(metadata, Mapping):
            return dict(metadata)
    return {}


def _validate_chamber_budgets(
    samiti_budget: Optional[int],
    sabha_budget: Optional[int],
) -> None:
    if samiti_budget is not None and samiti_budget < 1:
        raise ValueError("samiti_budget must be at least 1 when declared.")
    if sabha_budget is not None and sabha_budget < 1:
        raise ValueError("sabha_budget must be at least 1 when declared.")


def _strategy_role_chambers(strategy: AgentSearchStrategy) -> dict[str, str]:
    """Role name/kind -> chamber map for evaluation attribution."""

    chambers: dict[str, str] = {}
    role_graph = getattr(strategy, "role_graph", None) or ()
    for role in role_graph:
        if isinstance(role, AgentSocietyRole):
            chambers[role.name] = role.chamber or _chamber_for_proposal_kind(
                role.proposal_kind
            )
    for kind in ROLE_GRAPH_PROPOSAL_KINDS:
        chambers.setdefault(kind, _chamber_for_proposal_kind(kind))
    return chambers


def _proposal_chamber(
    proposal: AgentSearchProposal,
    role_chambers: Mapping[str, str],
) -> str:
    explicit = proposal.metadata.get("role_chamber")
    if explicit in CHAMBER_TOKENS:
        return str(explicit)
    if proposal.role in role_chambers:
        return role_chambers[proposal.role]
    role_kind = str(proposal.metadata.get("role_kind") or proposal.role)
    return _chamber_for_proposal_kind(role_kind)


def _normalize_society_role_graph(
    role_graph: Optional[Sequence[AgentSocietyRole | Mapping[str, Any]]],
) -> tuple[AgentSocietyRole, ...]:
    roles: List[AgentSocietyRole] = []
    for item in role_graph or DEFAULT_SOCIETY_ROLE_GRAPH:
        if isinstance(item, AgentSocietyRole):
            role = item
        elif isinstance(item, Mapping):
            role = AgentSocietyRole(
                name=str(item["name"]),
                proposal_kind=str(item["proposal_kind"]),
                phase=int(item.get("phase", 1)),
                depends_on=tuple(str(value) for value in item.get("depends_on", ())),
                path_prefixes=tuple(
                    str(value) for value in item.get("path_prefixes", ())
                ),
                archetype=str(item.get("archetype", "")),
                description=str(item.get("description", "")),
                guna=item.get("guna"),
                chamber=item.get("chamber"),
            )
        else:
            raise TypeError("role_graph entries must be AgentSocietyRole or mappings")
        if role.proposal_kind not in ROLE_GRAPH_PROPOSAL_KINDS:
            raise ValueError(
                f"Unsupported society role proposal_kind '{role.proposal_kind}'."
            )
        if role.phase < 1:
            raise ValueError("society role phase must be at least 1.")
        # Phase 4: resolve absent guna through the archetype-default table,
        # validate explicit triples, derive absent chamber from role kind.
        guna = _normalized_guna(role)
        chamber = role.chamber or _chamber_for_proposal_kind(role.proposal_kind)
        if chamber not in CHAMBER_TOKENS:
            raise ValueError(
                f"society role chamber must be one of {CHAMBER_TOKENS}, "
                f"got {chamber!r}."
            )
        roles.append(replace(role, guna=guna, chamber=chamber))

    names = [role.name for role in roles]
    if len(names) != len(set(names)):
        raise ValueError("society role names must be unique.")
    return tuple(roles)


def _normalized_guna(role: AgentSocietyRole) -> dict[str, float]:
    if role.guna is None:
        rajas, sattva, tamas = GUNA_ARCHETYPE_DEFAULTS.get(
            role.archetype, GUNA_ARCHETYPE_DEFAULTS[""]
        )
        return {"rajas": rajas, "sattva": sattva, "tamas": tamas}
    guna = dict(role.guna)
    if set(guna) != set(GUNA_AXES):
        raise ValueError(
            f"society role guna must declare exactly the axes {GUNA_AXES}, "
            f"got {sorted(guna)}."
        )
    for axis in GUNA_AXES:
        value = guna[axis]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"society role guna {axis} must be a number in [0, 1].")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"society role guna {axis} must be in [0, 1], got {value!r}."
            )
    return {axis: float(guna[axis]) for axis in GUNA_AXES}


def _chamber_for_proposal_kind(proposal_kind: str) -> str:
    return "samiti" if proposal_kind in SAMITI_PROPOSAL_KINDS else "sabha"


def _guna_mix(role_graph: Sequence[AgentSocietyRole]) -> dict[str, float]:
    """Society mean guna triple — the declared, tunable meta-parameter."""

    resolved = [_normalized_guna(role) for role in role_graph]
    if not resolved:
        return {axis: 0.0 for axis in GUNA_AXES}
    return {
        axis: round(sum(item[axis] for item in resolved) / len(resolved), 4)
        for axis in GUNA_AXES
    }


def _guna_radius(rajas: float, max_paths_per_proposal: int) -> int:
    """Mechanical rajas mapping: patch-radius units for generative streams."""

    return max(1, round(rajas * max_paths_per_proposal))


def _validate_justification(justification: Mapping[str, Any]) -> dict[str, str]:
    """Reject panca-avayava mappings missing any member or carrying empties."""

    if not isinstance(justification, Mapping):
        raise ValueError("proposal justification must be a mapping.")
    record: dict[str, str] = {}
    for member in PANCA_AVAYAVA_MEMBERS:
        value = str(justification.get(member) or "").strip()
        if not value:
            raise ValueError(
                f"proposal justification is missing a non-empty '{member}' member."
            )
        record[member] = value
    return record


def _justification(
    *,
    pratijna: str,
    hetu: str,
    udaharana: str,
    upanaya: str,
    nigamana: str,
) -> dict[str, str]:
    return {
        "pratijna": pratijna,
        "hetu": hetu,
        "udaharana": udaharana,
        "upanaya": upanaya,
        "nigamana": nigamana,
    }


def _ledger_diagnoses(
    ledger: Optional[Mapping[str, Any]],
) -> List[ComponentDiagnosis]:
    if not ledger:
        return []
    return _normalize_diagnoses(
        item
        for item in ledger.get("diagnoses", []) or []
        if isinstance(item, (Mapping, ComponentDiagnosis))
    )


def _build_round_proposals(
    *,
    seed_candidate: AgentCandidate,
    evaluations: Sequence[CandidateEvaluation],
    search_space: dict[str, List[Any]],
    search_paths: Sequence[str],
    beam_width: int,
    max_proposals: int,
    round_number: int,
) -> List[AgentSearchProposal]:
    proposals: List[AgentSearchProposal] = []
    seen: set[str] = set()
    ranked = sorted(
        evaluations,
        key=lambda item: (item.score, -len(item.candidate.patch), item.candidate.id),
        reverse=True,
    )
    changed_ranked = [item for item in ranked if item.candidate.patch]
    beam = ranked[:beam_width] or [
        CandidateEvaluation(candidate=seed_candidate, score=0.0)
    ]

    if round_number > 1:
        for proposal in _synthesis_proposals(changed_ranked[:beam_width], search_paths):
            _append_proposal(proposals, seen, proposal, max_proposals)

    if round_number > 1:
        for evaluation in beam:
            if not evaluation.candidate.patch:
                continue
            for proposal in _critic_proposals(
                evaluation.candidate,
                search_space,
                search_paths,
            ):
                _append_proposal(proposals, seen, proposal, max_proposals)

    for proposal in _explorer_proposals(seed_candidate, search_space, search_paths):
        _append_proposal(proposals, seen, proposal, max_proposals)

    if round_number > 1:
        for evaluation in changed_ranked[:beam_width]:
            for proposal in _steward_proposals(evaluation.candidate):
                _append_proposal(proposals, seen, proposal, max_proposals)

    return proposals


def _build_society_proposals(
    *,
    seed_candidate: AgentCandidate,
    evaluations: Sequence[CandidateEvaluation],
    search_space: dict[str, List[Any]],
    search_paths: Sequence[str],
    diagnoses: Sequence[ComponentDiagnosis],
    beam_width: int,
    max_proposals: int,
    round_number: int,
    ledger: Optional[Mapping[str, Any]] = None,
) -> List[AgentSearchProposal]:
    ranked = sorted(
        evaluations,
        key=lambda item: (item.score, -len(item.candidate.patch), item.candidate.id),
        reverse=True,
    )
    changed_ranked = [item for item in ranked if item.candidate.patch]
    beam = ranked[:beam_width] or [
        CandidateEvaluation(candidate=seed_candidate, score=0.0)
    ]
    pooled_diagnoses = list(diagnoses)
    ledger_diagnoses = _ledger_diagnoses(ledger)
    if ledger_diagnoses:
        # GEA experience pooling: no role reasons only from its own
        # candidate's diagnoses — the round-scoped society ledger joins in.
        pooled_diagnoses = _dedupe_diagnoses([*pooled_diagnoses, *ledger_diagnoses])

    streams: List[Iterable[AgentSearchProposal]] = []
    if round_number > 1:
        streams.append(_coverage_synthesis_proposals(changed_ranked, search_paths))
        streams.append(_synthesis_proposals(changed_ranked[:beam_width], search_paths))
        streams.append(
            proposal
            for evaluation in beam
            if evaluation.candidate.patch
            for proposal in _critic_proposals(
                evaluation.candidate,
                search_space,
                search_paths,
            )
        )

    streams.extend(
        [
            _specialist_proposals(
                seed_candidate,
                search_space,
                search_paths,
                pooled_diagnoses,
            ),
            _explorer_proposals(seed_candidate, search_space, search_paths),
            _adversary_proposals(
                seed_candidate,
                ranked[:beam_width],
                search_space,
                search_paths,
            ),
        ]
    )

    if round_number > 1:
        streams.append(
            proposal
            for evaluation in changed_ranked[:beam_width]
            for proposal in _steward_proposals(evaluation.candidate)
        )

    return _interleave_proposal_streams(streams, max_proposals)


def _build_role_graph_society_proposals(
    *,
    seed_candidate: AgentCandidate,
    evaluations: Sequence[CandidateEvaluation],
    search_space: dict[str, List[Any]],
    search_paths: Sequence[str],
    diagnoses: Sequence[ComponentDiagnosis],
    beam_width: int,
    max_proposals: int,
    round_number: int,
    role_graph: Sequence[AgentSocietyRole],
    ledger: Optional[Mapping[str, Any]] = None,
    max_paths_per_proposal: int = 1,
) -> List[AgentSearchProposal]:
    ranked = sorted(
        evaluations,
        key=lambda item: (item.score, -len(item.candidate.patch), item.candidate.id),
        reverse=True,
    )
    changed_ranked = [item for item in ranked if item.candidate.patch]
    beam = ranked[:beam_width] or [
        CandidateEvaluation(candidate=seed_candidate, score=0.0)
    ]
    evaluated_roles = {
        str(evaluation.metadata.get("proposal_role"))
        for evaluation in evaluations
        if evaluation.metadata.get("proposal_role")
    }
    pooled_diagnoses = list(diagnoses)
    ledger_diagnoses = _ledger_diagnoses(ledger)
    if ledger_diagnoses:
        pooled_diagnoses = _dedupe_diagnoses([*pooled_diagnoses, *ledger_diagnoses])

    streams: List[Iterable[AgentSearchProposal]] = []
    for role in _ordered_role_graph_roles(role_graph, round_number):
        if not _society_role_is_active(role, evaluated_roles, round_number):
            continue
        role_paths = _role_search_paths(role, search_paths)
        if role.proposal_kind != "steward" and not role_paths:
            continue
        stream = _role_graph_stream(
            role,
            seed_candidate=seed_candidate,
            ranked=ranked,
            changed_ranked=changed_ranked,
            beam=beam,
            search_space=search_space,
            search_paths=role_paths,
            diagnoses=pooled_diagnoses,
            beam_width=beam_width,
            round_number=round_number,
            max_paths_per_proposal=max_paths_per_proposal,
        )
        streams.append(stream)

    return _interleave_proposal_streams(streams, max_proposals)


def _ordered_role_graph_roles(
    role_graph: Sequence[AgentSocietyRole],
    round_number: int,
) -> List[AgentSocietyRole]:
    if round_number <= 1:
        return list(role_graph)
    priority = {
        "coverage_synthesis": 0,
        "synthesizer": 0,
        "critic": 1,
        "adversary": 2,
        "specialist": 2,
        "explorer": 2,
        "steward": 3,
    }
    return [
        role
        for _, role in sorted(
            enumerate(role_graph),
            key=lambda item: (priority.get(item[1].proposal_kind, 2), item[0]),
        )
    ]


def _society_role_is_active(
    role: AgentSocietyRole,
    evaluated_roles: set[str],
    round_number: int,
) -> bool:
    if role.phase > round_number:
        return False
    if not role.depends_on:
        return True
    return bool(set(role.depends_on) & evaluated_roles)


def _role_graph_stream(
    role: AgentSocietyRole,
    *,
    seed_candidate: AgentCandidate,
    ranked: Sequence[CandidateEvaluation],
    changed_ranked: Sequence[CandidateEvaluation],
    beam: Sequence[CandidateEvaluation],
    search_space: dict[str, List[Any]],
    search_paths: Sequence[str],
    diagnoses: Sequence[ComponentDiagnosis],
    beam_width: int,
    round_number: int,
    max_paths_per_proposal: int = 1,
) -> Iterable[AgentSearchProposal]:
    # Deterministic guna behavioral mappings (pure functions of the resolved
    # triple): rajas scales generative patch radius, sattva scales synthesis
    # breadth / reconciliation, tamas scales steward removal aggressiveness.
    guna = _normalized_guna(role)
    radius_units = _guna_radius(guna["rajas"], max_paths_per_proposal)
    if role.proposal_kind == "specialist":
        proposals = _specialist_proposals(
            seed_candidate,
            search_space,
            search_paths,
            diagnoses,
        )
    elif role.proposal_kind == "explorer":
        proposals = _explorer_proposals(
            seed_candidate,
            search_space,
            search_paths,
            max_paths=radius_units,
        )
    elif role.proposal_kind == "adversary":
        proposals = _adversary_proposals(
            seed_candidate,
            ranked[:beam_width],
            search_space,
            search_paths,
            max_boundary_paths=3 * radius_units,
        )
    elif role.proposal_kind == "critic" and round_number > 1:
        proposals = (
            proposal
            for evaluation in beam
            if evaluation.candidate.patch
            for proposal in _critic_proposals(
                evaluation.candidate,
                search_space,
                search_paths,
            )
        )
    elif role.proposal_kind == "coverage_synthesis" and round_number > 1:
        proposals = _coverage_synthesis_proposals(
            changed_ranked,
            search_paths,
            reconcile=guna["sattva"] >= 0.5,
        )
    elif role.proposal_kind == "synthesizer" and round_number > 1:
        breadth = max(1, round(guna["sattva"] * beam_width))
        proposals = _synthesis_proposals(changed_ranked[:breadth], search_paths)
    elif role.proposal_kind == "steward" and round_number > 1:
        allowed = set(search_paths)
        proposals = (
            proposal
            for evaluation in changed_ranked[:beam_width]
            for proposal in _steward_proposals(evaluation.candidate, tamas=guna["tamas"])
            if not allowed or allowed & set(proposal.patch)
        )
    else:
        proposals = ()

    return _annotate_role_graph_proposals(role, proposals)


def _annotate_role_graph_proposals(
    role: AgentSocietyRole,
    proposals: Iterable[AgentSearchProposal],
) -> Iterable[AgentSearchProposal]:
    role_guna = _normalized_guna(role)
    role_chamber = role.chamber or _chamber_for_proposal_kind(role.proposal_kind)
    for proposal in proposals:
        metadata = {
            **dict(proposal.metadata),
            "role_kind": role.proposal_kind,
            "role_phase": role.phase,
            "role_archetype": role.archetype,
            "role_description": role.description,
            "role_path_prefixes": list(role.path_prefixes),
            "role_depends_on": list(role.depends_on),
            "role_guna": dict(role_guna),
            "role_chamber": role_chamber,
        }
        yield AgentSearchProposal(
            patch=proposal.patch,
            role=role.name,
            parent_ids=proposal.parent_ids,
            reason=f"{role.proposal_kind}:{proposal.reason}",
            metadata=metadata,
        )


def _role_search_paths(
    role: AgentSocietyRole,
    search_paths: Sequence[str],
) -> List[str]:
    if not role.path_prefixes:
        return list(search_paths)
    return [
        path
        for path in search_paths
        if any(path == prefix or path.startswith(f"{prefix}.") for prefix in role.path_prefixes)
    ]


def _interleave_proposal_streams(
    streams: Sequence[Iterable[AgentSearchProposal]],
    max_proposals: int,
) -> List[AgentSearchProposal]:
    proposals: List[AgentSearchProposal] = []
    seen: set[str] = set()
    iterators = [iter(stream) for stream in streams]
    active = [True for _ in iterators]

    while len(proposals) < max_proposals and any(active):
        for index, iterator in enumerate(iterators):
            if not active[index]:
                continue
            while True:
                try:
                    proposal = next(iterator)
                except StopIteration:
                    active[index] = False
                    break
                before = len(proposals)
                _append_proposal(proposals, seen, proposal, max_proposals)
                if len(proposals) > before:
                    break
                if len(proposals) >= max_proposals:
                    break
            if len(proposals) >= max_proposals:
                break
    return proposals


def _specialist_proposals(
    seed_candidate: AgentCandidate,
    search_space: dict[str, List[Any]],
    search_paths: Sequence[str],
    diagnoses: Sequence[ComponentDiagnosis],
) -> Iterable[AgentSearchProposal]:
    target_name = seed_candidate.target_name or "the optimization target"
    for group_key, paths in _path_groups(search_paths, diagnoses).items():
        patch: dict[str, Any] = {}
        for path in paths:
            value = _first_non_seed_value(seed_candidate, search_space, path)
            if value is not _NO_VALUE:
                patch[path] = value
        if not patch:
            continue
        evidence = "; ".join(
            diagnosis.evidence
            for diagnosis in diagnoses
            if diagnosis.evidence
            and _diagnostic_group_key(sorted(patch)[0], [diagnosis])
        ) or f"component grouping over declared search paths {sorted(patch)}"
        yield AgentSearchProposal(
            patch=patch,
            role="specialist",
            parent_ids=(seed_candidate.id,),
            reason=f"apply_component_bundle:{group_key}",
            metadata={
                "justification": _justification(
                    pratijna=(
                        f"bundling component '{group_key}' repairs improves {target_name}"
                    ),
                    hetu=evidence,
                    udaharana=(
                        f"seed candidate {seed_candidate.id} exhibits the diagnosed "
                        f"component state; rule: diagnosed components are repaired as one bundle"
                    ),
                    upanaya=(
                        f"this candidate patches exactly the '{group_key}' paths "
                        f"{sorted(patch)}"
                    ),
                    nigamana=(
                        "expect the diagnosed-component metrics to close on the "
                        "next admissible evaluation"
                    ),
                )
            },
        )


def _adversary_proposals(
    seed_candidate: AgentCandidate,
    ranked: Sequence[CandidateEvaluation],
    search_space: dict[str, List[Any]],
    search_paths: Sequence[str],
    max_boundary_paths: int = 3,
) -> Iterable[AgentSearchProposal]:
    target_name = seed_candidate.target_name or "the optimization target"
    boundary_patch: dict[str, Any] = {}
    for path in search_paths:
        value = _last_non_seed_value(seed_candidate, search_space, path)
        if value is not _NO_VALUE:
            boundary_patch[path] = value
        if len(boundary_patch) >= max_boundary_paths:
            break
    if boundary_patch:
        yield AgentSearchProposal(
            patch=boundary_patch,
            role="adversary",
            parent_ids=(seed_candidate.id,),
            reason="stress_boundary_combination",
            metadata={
                "justification": _justification(
                    pratijna=(
                        f"a boundary-value combination stresses {target_name} "
                        "into revealing brittle settings"
                    ),
                    hetu=(
                        "search space declares boundary values on paths "
                        f"{sorted(boundary_patch)}"
                    ),
                    udaharana=(
                        f"seed candidate {seed_candidate.id} holds interior values; "
                        "rule: adversarial probes test the declared extremes"
                    ),
                    upanaya=(
                        "this candidate combines the last non-seed value of each "
                        "boundary path in one patch"
                    ),
                    nigamana=(
                        "expect either a robustness confirmation or an admissible "
                        "failure signal at the boundary"
                    ),
                )
            },
        )

    for evaluation in ranked:
        source_patch = dict(evaluation.candidate.patch)
        if not source_patch:
            continue
        for path in search_paths:
            value = _last_non_seed_value(evaluation.candidate, search_space, path)
            if value is _NO_VALUE:
                continue
            patch = {**source_patch, path: value}
            yield AgentSearchProposal(
                patch=patch,
                role="adversary",
                parent_ids=(evaluation.candidate.id,),
                reason="stress_candidate_with_boundary_change",
                metadata={
                    "justification": _justification(
                        pratijna=(
                            f"candidate {evaluation.candidate.id} should survive a "
                            f"boundary change on {path}"
                        ),
                        hetu=(
                            f"candidate {evaluation.candidate.id} scored "
                            f"{evaluation.score:.4f} with patch {sorted(source_patch)}"
                        ),
                        udaharana=(
                            "rule: strong candidates are stress-tested with one "
                            "additional boundary value before promotion"
                        ),
                        upanaya=(
                            f"this candidate keeps the parent patch and sets {path} "
                            "to its boundary value"
                        ),
                        nigamana=(
                            "expect a measurable score delta isolating the boundary "
                            "sensitivity of the parent"
                        ),
                    )
                },
            )


def _path_groups(
    search_paths: Sequence[str],
    diagnoses: Sequence[ComponentDiagnosis],
) -> dict[str, List[str]]:
    groups: dict[str, List[str]] = {}
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


class _NoValue:
    pass


_NO_VALUE = _NoValue()


def _first_non_seed_value(
    seed_candidate: AgentCandidate,
    search_space: dict[str, List[Any]],
    path: str,
) -> Any:
    current = seed_candidate.get_path(path)
    for value in search_space.get(path, []):
        if value != current:
            return value
    return _NO_VALUE


def _last_non_seed_value(
    candidate: AgentCandidate,
    search_space: dict[str, List[Any]],
    path: str,
) -> Any:
    current = candidate.get_path(path)
    for value in reversed(search_space.get(path, [])):
        if value != current:
            return value
    return _NO_VALUE


def _synthesis_proposals(
    evaluations: Sequence[CandidateEvaluation],
    search_paths: Sequence[str],
) -> Iterable[AgentSearchProposal]:
    if len(evaluations) < 2:
        return

    allowed = set(search_paths)
    all_sources = tuple(evaluations)
    yield AgentSearchProposal(
        patch=_merge_ranked_patches(all_sources, allowed),
        role="synthesizer",
        parent_ids=tuple(item.candidate.id for item in all_sources),
        reason="combine_best_partial_candidates",
        metadata={
            "justification": _justification(
                pratijna="merging the strongest partial candidates compounds their gains",
                hetu=(
                    "evaluated parents "
                    f"{[item.candidate.id for item in all_sources]} each improved "
                    "disjoint or compatible paths"
                ),
                udaharana=(
                    "rule: compatible partial repairs are merged rank-first so the "
                    "strongest parent wins conflicting paths"
                ),
                upanaya="this candidate is the rank-first merge of every parent patch",
                nigamana=(
                    "expect a combined score at or above the best parent on the "
                    "next admissible evaluation"
                ),
            )
        },
    )
    for left, right in combinations(evaluations, 2):
        yield AgentSearchProposal(
            patch=_merge_ranked_patches((left, right), allowed),
            role="synthesizer",
            parent_ids=(left.candidate.id, right.candidate.id),
            reason="combine_pairwise_partial_candidates",
            metadata={
                "justification": _justification(
                    pratijna=(
                        f"the pair {left.candidate.id} + {right.candidate.id} "
                        "combines compatible repairs"
                    ),
                    hetu=(
                        f"parents scored {left.score:.4f} and {right.score:.4f} on "
                        "admissible evaluations"
                    ),
                    udaharana=(
                        "rule: pairwise merges isolate which parent combination "
                        "carries the gain"
                    ),
                    upanaya="this candidate merges exactly the two parent patches",
                    nigamana=(
                        "expect the pairwise merge to attribute the combined gain "
                        "on the next admissible evaluation"
                    ),
                )
            },
        )


def _coverage_synthesis_proposals(
    evaluations: Sequence[CandidateEvaluation],
    search_paths: Sequence[str],
    *,
    reconcile: bool = True,
) -> Iterable[AgentSearchProposal]:
    if not evaluations:
        return

    allowed = set(search_paths)
    patch: dict[str, Any] = {}
    parent_ids: List[str] = []
    for path in search_paths:
        path_evaluations = [
            evaluation
            for evaluation in evaluations
            if path in evaluation.candidate.patch and path in allowed
        ]
        if not path_evaluations:
            continue
        if not reconcile:
            # Low-sattva synthesis skips conflicting paths instead of
            # reconciling them (deterministic guna mapping; default-archetype
            # sattva >= 0.5 keeps the legacy reconciliation).
            distinct_values = {
                json.dumps(
                    evaluation.candidate.patch[path], sort_keys=True, default=str
                )
                for evaluation in path_evaluations
            }
            if len(distinct_values) > 1:
                continue
        selected = max(
            path_evaluations,
            key=lambda item: (
                item.score,
                -len(item.candidate.patch),
                item.candidate.id,
            ),
        )
        patch[path] = selected.candidate.patch[path]
        parent_ids.append(selected.candidate.id)

    if patch:
        yield AgentSearchProposal(
            patch=patch,
            role="synthesizer",
            parent_ids=tuple(dict.fromkeys(parent_ids)),
            reason="combine_best_path_representatives",
            metadata={
                "justification": _justification(
                    pratijna=(
                        "selecting the best representative per path covers the "
                        "whole repaired surface"
                    ),
                    hetu=(
                        f"per-path winners {sorted(set(parent_ids))} carry the "
                        "highest admissible score for their path"
                    ),
                    udaharana=(
                        "rule: coverage synthesis promotes each path's best "
                        "evidence-backed value"
                    ),
                    upanaya=(
                        f"this candidate sets {sorted(patch)} to their per-path "
                        "winning values"
                    ),
                    nigamana=(
                        "expect coverage of every repaired path without losing "
                        "any single-path gain"
                    ),
                )
            },
        )


def _critic_proposals(
    source_candidate: AgentCandidate,
    search_space: dict[str, List[Any]],
    search_paths: Sequence[str],
) -> Iterable[AgentSearchProposal]:
    source_patch = dict(source_candidate.patch)
    for path in search_paths:
        for value in search_space.get(path, []):
            if source_candidate.get_path(path) == value:
                continue
            patch = {**source_patch, path: value}
            yield AgentSearchProposal(
                patch=patch,
                role="critic",
                parent_ids=(source_candidate.id,),
                reason="test_next_change_against_current_candidate",
                metadata={
                    "justification": _justification(
                        pratijna=(
                            f"candidate {source_candidate.id} improves further with "
                            f"{path} changed"
                        ),
                        hetu=(
                            f"parent patch {sorted(source_patch)} passed an "
                            "admissible evaluation and the search space declares "
                            f"another value for {path}"
                        ),
                        udaharana=(
                            "rule: critics test exactly one more change against the "
                            "current strong candidate"
                        ),
                        upanaya=(
                            f"this candidate keeps the parent patch and sets {path} "
                            "to the next declared value"
                        ),
                        nigamana=(
                            f"expect the evaluation to confirm or refute {path} as "
                            "the next improving change"
                        ),
                    )
                },
            )


def _explorer_proposals(
    seed_candidate: AgentCandidate,
    search_space: dict[str, List[Any]],
    search_paths: Sequence[str],
    max_paths: int = 1,
) -> Iterable[AgentSearchProposal]:
    target_name = seed_candidate.target_name or "the optimization target"
    for path in search_paths:
        for value in search_space.get(path, []):
            if seed_candidate.get_path(path) == value:
                continue
            yield AgentSearchProposal(
                patch={path: value},
                role="explorer",
                parent_ids=(seed_candidate.id,),
                reason="isolate_single_path_effect",
                metadata={
                    "justification": _justification(
                        pratijna=f"setting {path} improves {target_name}",
                        hetu=(
                            "the declared search space lists an untested value "
                            f"for {path}"
                        ),
                        udaharana=(
                            f"seed candidate {seed_candidate.id} holds "
                            f"{seed_candidate.get_path(path)!r} on {path}; rule: "
                            "isolated single-path probes attribute metric deltas"
                        ),
                        upanaya=(
                            f"this candidate patches only {path}, so any score "
                            "delta is attributable to it"
                        ),
                        nigamana=(
                            f"expect an admissible evaluation-score delta for {path}"
                        ),
                    )
                },
            )
    if max_paths > 1:
        # Rajas-widened exploration: deterministic sliding windows of adjacent
        # admissible paths, each set to its first non-seed value. max_paths == 1
        # (the default radius for every default-archetype triple) skips this
        # block entirely, preserving legacy proposals byte-for-byte.
        paths = list(search_paths)
        for start in range(len(paths)):
            window = paths[start : start + max_paths]
            if len(window) < 2:
                continue
            patch: dict[str, Any] = {}
            for path in window:
                value = _first_non_seed_value(seed_candidate, search_space, path)
                if value is not _NO_VALUE:
                    patch[path] = value
            if len(patch) < 2:
                continue
            yield AgentSearchProposal(
                patch=patch,
                role="explorer",
                parent_ids=(seed_candidate.id,),
                reason="explore_adjacent_path_window",
                metadata={
                    "justification": _justification(
                        pratijna=(
                            f"jointly setting {sorted(patch)} improves {target_name}"
                        ),
                        hetu=(
                            "high-rajas exploration widens the mutation radius over "
                            "adjacent admissible paths"
                        ),
                        udaharana=(
                            f"seed candidate {seed_candidate.id} holds the seed "
                            "values; rule: widened probes test interacting paths "
                            "together"
                        ),
                        upanaya=(
                            f"this candidate patches the adjacent window {sorted(patch)}"
                        ),
                        nigamana=(
                            "expect an admissible evaluation delta attributable to "
                            "the window"
                        ),
                    )
                },
            )


def _steward_proposals(
    source_candidate: AgentCandidate,
    *,
    tamas: Optional[float] = None,
) -> Iterable[AgentSearchProposal]:
    if len(source_candidate.patch) < 2:
        return
    patch_paths = list(source_candidate.patch)
    if tamas is None:
        removal_limit = len(patch_paths)
    else:
        # Tamas mapping: removal attempts per round scale with the steward's
        # tamas (ceil keeps every default-archetype triple at full coverage
        # for the patch sizes the deterministic fixtures use).
        removal_limit = max(1, math.ceil(float(tamas) * len(patch_paths)))
    for path in patch_paths[:removal_limit]:
        patch = {
            key: value
            for key, value in source_candidate.patch.items()
            if key != path
        }
        yield AgentSearchProposal(
            patch=patch,
            role="steward",
            parent_ids=(source_candidate.id,),
            reason="remove_one_change_to_check_minimality",
            metadata={
                "justification": _justification(
                    pratijna=(
                        f"candidate {source_candidate.id} keeps its score without "
                        f"the change on {path}"
                    ),
                    hetu=(
                        f"parent patch {sorted(source_candidate.patch)} passed an "
                        "admissible evaluation with multiple combined changes"
                    ),
                    udaharana=(
                        "rule: stewards remove one change at a time so only "
                        "metric-proven repairs survive"
                    ),
                    upanaya=(
                        f"this candidate is the parent patch minus {path} and "
                        "nothing else"
                    ),
                    nigamana=(
                        f"expect an equal score if {path} was unnecessary, or a "
                        "regression proving it was load-bearing"
                    ),
                )
            },
        )


def _merge_ranked_patches(
    evaluations: Sequence[CandidateEvaluation],
    allowed_paths: set[str],
) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    for evaluation in evaluations:
        for path, value in evaluation.candidate.patch.items():
            if path in allowed_paths and path not in patch:
                patch[path] = value
    return patch


def _append_proposal(
    proposals: List[AgentSearchProposal],
    seen: set[str],
    proposal: AgentSearchProposal,
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


def _canonical_patch(patch: dict[str, Any]) -> str:
    return json.dumps(patch, sort_keys=True, default=str)
