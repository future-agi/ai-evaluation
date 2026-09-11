from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

from .types import OptimizationResult

# Canon vocabularies live in council.py (single home); imported here for
# governance-record validation. Safe: council does not import this module.
from .optimizers.council import (
    CHAMBER_TOKENS,
    GUNA_AXES,
    HETVABHASA_REJECTION_CLASSES,
    PANCA_AVAYAVA_MEMBERS,
    _validate_justification,
)

OPTIMIZER_TRAJECTORY_PROFILE_KIND = "agent-learning.optimizer-trajectory-profile.v1"

_CRITIQUE_KIND_BY_ROLE_KIND = {
    "critic": "vada",
    "adversary": "jalpa",
}


def optimizer_trajectory_profile(result: OptimizationResult) -> dict[str, Any]:
    """Trajectory fitness profile (ACL-Findings 2026, arXiv:2604.19440):
    trajectory shape, not endpoint score, as backend-routing evidence.

    Computed post-hoc from ``OptimizationResult.history`` — no backend loop
    changes, every backend gets it for free.
    """

    history = list(result.history or [])
    metadata = dict(result.metadata or {})

    running_best: Optional[float] = None
    improvements = 0
    locality_terms: list[float] = []
    regression_count = 0
    scores_by_candidate: Dict[str, float] = {}
    previous_score: Optional[float] = None
    candidate_keys: list[str] = []

    for index, item in enumerate(history):
        score = float(getattr(item, "average_score", 0.0) or 0.0)
        item_metadata = dict(getattr(item, "metadata", {}) or {})
        candidate_id = str(
            getattr(item, "candidate_id", None)
            or item_metadata.get("candidate_id")
            or f"iteration-{index}"
        )
        candidate_keys.append(candidate_id)

        improved = running_best is None or score > running_best
        if improved and index > 0:
            improvements += 1
        if improved:
            running_best = score
            patch = item_metadata.get("patch") or item_metadata.get(
                "candidate_patch"
            )
            paths_touched = len(patch) if isinstance(patch, Mapping) else 1
            locality_terms.append(1.0 / max(1, paths_touched))

        parent_ids = [
            str(parent)
            for parent in _as_list(
                item_metadata.get("proposal_parent_ids")
                or item_metadata.get("evolution_parent_ids")
            )
            if str(parent)
        ]
        parent_scores = [
            scores_by_candidate[parent]
            for parent in parent_ids
            if parent in scores_by_candidate
        ]
        if parent_scores:
            if score < max(parent_scores):
                regression_count += 1
        elif previous_score is not None and score < previous_score:
            regression_count += 1

        scores_by_candidate.setdefault(candidate_id, score)
        previous_score = score

    iteration_count = len(history)
    comparable = max(1, iteration_count - 1)
    return {
        # Embedded payload, not a top-level artifact kind.
        "kind": OPTIMIZER_TRAJECTORY_PROFILE_KIND,
        "improvement_frequency": round(improvements / comparable, 4)
        if iteration_count > 1
        else (1.0 if iteration_count == 1 else 0.0),
        "semantic_locality": round(
            sum(locality_terms) / len(locality_terms), 4
        )
        if locality_terms
        else 0.0,
        "dedupe_rate": round(
            1.0 - (len(set(candidate_keys)) / iteration_count), 4
        )
        if iteration_count
        else 0.0,
        "regression_count": regression_count,
        "iterations": iteration_count,
        "evaluations": int(result.total_evaluations or 0),
        "early_stopped": bool(result.early_stopped),
        "selection": metadata.get("selection"),
        "eval_budget": metadata.get("eval_budget"),
    }


def build_optimizer_society_trace(
    result: OptimizationResult,
    *,
    name: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Export a council/society optimization run as portable trace evidence."""

    result_metadata = dict(result.metadata or {})
    roles = _role_records(result_metadata)
    proposals = [_proposal_record(item) for item in result.history]
    proposals = [item for item in proposals if item]
    for proposal in proposals:
        justification = dict(proposal.get("metadata") or {}).get("justification")
        if justification is not None:
            _validate_justification(justification)
    search_paths = sorted(str(path) for path in result_metadata.get("search_paths", []) if str(path))
    diagnostics = [
        dict(item)
        for item in _as_list(result_metadata.get("diagnostics"))
        if isinstance(item, Mapping)
    ]
    role_credit = _role_credit(proposals)
    best_candidate_id = str(result_metadata.get("best_candidate_id") or getattr(result.best_candidate, "id", "") or "")
    final_score = float(result.final_score)
    governance = _governance_records(
        result_metadata=result_metadata,
        roles=roles,
        proposals=proposals,
        diagnostics=diagnostics,
        search_paths=search_paths,
        role_credit=role_credit,
        best_candidate_id=best_candidate_id,
        final_score=final_score,
    )
    signals = _signals(
        roles=roles,
        proposals=proposals,
        diagnostics=diagnostics,
        search_paths=search_paths,
        role_credit=role_credit,
        best_candidate_id=best_candidate_id,
        governance=governance,
    )
    summary = _summary(
        roles=roles,
        proposals=proposals,
        diagnostics=diagnostics,
        search_paths=search_paths,
        role_credit=role_credit,
        best_candidate_id=best_candidate_id,
        final_score=final_score,
        rounds=_as_list(result_metadata.get("rounds")),
        governance=governance,
    )
    ledger_records = [
        dict(item)
        for item in _as_list(result_metadata.get("ledger_rounds"))
        if isinstance(item, Mapping)
    ]
    return {
        "kind": "optimizer_society_trace",
        "name": name or str(result_metadata.get("target_name") or "optimizer-society-trace"),
        "optimizer": str(result_metadata.get("optimizer") or "agent-opt"),
        "strategy": result_metadata.get("strategy"),
        "roles": roles,
        "proposals": proposals,
        "rounds": [dict(item) for item in _as_list(result_metadata.get("rounds")) if isinstance(item, Mapping)],
        "diagnostics": diagnostics,
        "search_paths": search_paths,
        "role_credit": role_credit,
        "governance": governance,
        "ledger": ledger_records,
        "best_candidate_id": best_candidate_id or None,
        "final_score": final_score,
        "signals": sorted(signals),
        "summary": summary,
        "metadata": {
            "source": "agent-opt",
            **{k: v for k, v in result_metadata.items() if k not in {"rounds", "diagnostics"}},
            **dict(metadata or {}),
        },
    }


def _role_records(metadata: Mapping[str, Any]) -> list[Dict[str, Any]]:
    role_graph = [
        dict(item)
        for item in _as_list(metadata.get("role_graph"))
        if isinstance(item, Mapping)
    ]
    if role_graph:
        return role_graph
    return [{"name": str(role)} for role in _as_list(metadata.get("roles")) if str(role)]


def _proposal_record(history: Any) -> Dict[str, Any]:
    item_metadata = dict(getattr(history, "metadata", {}) or {})
    proposal_metadata = dict(item_metadata.get("proposal_metadata") or {})
    role = str(item_metadata.get("proposal_role") or "unknown")
    patch = dict(item_metadata.get("patch") or {})
    return {
        "id": str(getattr(history, "candidate_id", None) or item_metadata.get("candidate_id") or ""),
        "candidate_id": str(getattr(history, "candidate_id", None) or item_metadata.get("candidate_id") or ""),
        "role": role,
        "round": item_metadata.get("proposal_round"),
        "score": float(getattr(history, "average_score", 0.0)),
        "reason": str(item_metadata.get("proposal_reason") or item_metadata.get("reason") or ""),
        "parent_ids": [
            str(parent)
            for parent in _as_list(item_metadata.get("proposal_parent_ids"))
            if str(parent)
        ],
        "patch": patch,
        "search_paths": sorted(str(path) for path in patch.keys()),
        "role_kind": str(item_metadata.get("role_kind") or proposal_metadata.get("role_kind") or ""),
        "role_archetype": str(item_metadata.get("role_archetype") or proposal_metadata.get("role_archetype") or ""),
        "metadata": proposal_metadata,
    }


def _role_credit(proposals: Iterable[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    credit: Dict[str, Dict[str, Any]] = {}
    for proposal in proposals:
        role = str(proposal.get("role") or "unknown")
        key = _normalize(role) or "unknown"
        score = _optional_float(proposal.get("score"))
        entry = credit.setdefault(
            key,
            {
                "role": role,
                "proposal_count": 0,
                "evaluated_count": 0,
                "best_score": None,
                "best_candidate_id": None,
                "search_paths": set(),
            },
        )
        entry["proposal_count"] += 1
        entry["search_paths"].update(str(path) for path in _as_list(proposal.get("search_paths")) if str(path))
        if score is None:
            continue
        entry["evaluated_count"] += 1
        if entry["best_score"] is None or score > float(entry["best_score"]):
            entry["best_score"] = score
            entry["best_candidate_id"] = proposal.get("candidate_id")
    return [
        {
            **entry,
            "search_paths": sorted(entry["search_paths"]),
        }
        for entry in sorted(credit.values(), key=lambda item: str(item["role"]))
    ]


def _signals(
    *,
    roles: list[Mapping[str, Any]],
    proposals: list[Mapping[str, Any]],
    diagnostics: list[Mapping[str, Any]],
    search_paths: list[str],
    role_credit: list[Mapping[str, Any]],
    best_candidate_id: str,
    governance: Mapping[str, Any],
) -> set[str]:
    signals = {"optimizer", "society_trace"}
    if roles:
        signals.add("role")
    if any(role.get("proposal_kind") for role in roles) or any(proposal.get("role_kind") for proposal in proposals):
        signals.add("role_graph")
    if any(role.get("archetype") for role in roles) or any(proposal.get("role_archetype") for proposal in proposals):
        signals.add("archetype")
    if proposals:
        signals.update({"proposal", "candidate", "evaluation", "score", "stop"})
    if diagnostics:
        signals.add("diagnostic")
    if search_paths or any(proposal.get("search_paths") for proposal in proposals):
        signals.add("search_path")
    if role_credit:
        signals.add("credit")
    governance_signals = {
        _normalize(signal)
        for signal in _as_list(governance.get("signals"))
        if _normalize(signal)
    }
    if governance_signals or _as_list(governance.get("checks")):
        signals.update({"governance", *governance_signals})
    if best_candidate_id:
        signals.add("best_candidate")
    role_tokens = {
        _normalize(proposal.get("role"))
        for proposal in proposals
    } | {
        _normalize(proposal.get("role_kind"))
        for proposal in proposals
    }
    if role_tokens & {"critic", "adversary", "vidura", "krishna"}:
        signals.add("critique")
    if role_tokens & {"synthesizer", "coverage_synthesis", "sangha"}:
        signals.add("synthesis")
    if role_tokens & {"steward", "dharma_steward"}:
        signals.add("steward")
    return signals


def _summary(
    *,
    roles: list[Mapping[str, Any]],
    proposals: list[Mapping[str, Any]],
    diagnostics: list[Mapping[str, Any]],
    search_paths: list[str],
    role_credit: list[Mapping[str, Any]],
    best_candidate_id: str,
    final_score: float,
    rounds: list[Any],
    governance: Mapping[str, Any],
) -> Dict[str, Any]:
    candidate_ids = [str(proposal.get("candidate_id") or "") for proposal in proposals if proposal.get("candidate_id")]
    role_tokens = {
        _normalize(proposal.get("role"))
        for proposal in proposals
    } | {
        _normalize(proposal.get("role_kind"))
        for proposal in proposals
    }
    governance_summary = dict(governance.get("summary") or {})
    summary = {
        "role_count": len(roles),
        "proposal_count": len(proposals),
        "evaluation_count": len(proposals),
        "round_count": len(rounds) or len({proposal.get("round") for proposal in proposals if proposal.get("round") is not None}),
        "diagnostic_count": len(diagnostics),
        "search_path_count": len(search_paths),
        "role_credit_count": len(role_credit),
        "duplicate_candidate_count": max(0, len(candidate_ids) - len(set(candidate_ids))),
        "best_candidate_id": best_candidate_id or None,
        "final_score": final_score,
        "has_role_graph": any(role.get("proposal_kind") for role in roles),
        "has_critique": bool(role_tokens & {"critic", "adversary", "vidura", "krishna"}),
        "has_synthesis": bool(role_tokens & {"synthesizer", "coverage_synthesis", "sangha"}),
        "has_steward": bool(role_tokens & {"steward", "dharma_steward"}),
        "terminal_status": "completed",
    }
    for key in (
        "governance_check_count",
        "governance_passed_count",
        "governance_pass_rate",
        "has_governance",
        "has_role_diversity",
        "has_mediator",
        "has_contract_gate",
        "has_rollback",
        "has_locality",
        "has_dependency_audit",
        # Phase 4 society/contract flags (additive)
        "has_guna_axes",
        "has_two_chamber",
        "has_nyaya_justifications",
        "has_hetvabhasa_rejections",
        "has_nirnaya",
        "has_staged_conditioning",
        "has_layer_locality",
        "has_declared_budget",
        "has_external_ranking",
    ):
        if key in governance_summary:
            summary[key] = governance_summary[key]
    return summary


def _governance_records(
    *,
    result_metadata: Mapping[str, Any],
    roles: list[Mapping[str, Any]],
    proposals: list[Mapping[str, Any]],
    diagnostics: list[Mapping[str, Any]],
    search_paths: list[str],
    role_credit: list[Mapping[str, Any]],
    best_candidate_id: str,
    final_score: float,
) -> Dict[str, Any]:
    explicit = result_metadata.get("governance") or result_metadata.get("optimizer_governance")
    explicit_checks = []
    explicit_signals = []
    if isinstance(explicit, Mapping):
        explicit_checks = _as_list(explicit.get("checks"))
        explicit_signals = _as_list(explicit.get("signals"))
    elif explicit:
        explicit_checks = _as_list(explicit)
    explicit_checks.extend(_as_list(result_metadata.get("governance_checks")))

    role_names = {
        _normalize(role.get("name") or role.get("role"))
        for role in roles
    }
    role_kinds = {
        _normalize(role.get("proposal_kind"))
        for role in roles
    } | {
        _normalize(proposal.get("role_kind"))
        for proposal in proposals
    }
    proposal_roles = {
        _normalize(proposal.get("role"))
        for proposal in proposals
    }
    path_set = {str(path) for path in search_paths if str(path)}
    patched_paths = {
        str(path)
        for proposal in proposals
        for path in _as_list(proposal.get("search_paths"))
        if str(path)
    }
    patched_paths.update(
        str(path)
        for proposal in proposals
        for path in dict(proposal.get("patch") or {}).keys()
        if str(path)
    )
    non_seed_roles = {role for role in proposal_roles if role and role not in {"seed", "unknown"}}
    has_critique = bool((proposal_roles | role_kinds | role_names) & {"critic", "adversary", "vidura", "krishna"})
    has_synthesis = bool((proposal_roles | role_kinds | role_names) & {"synthesizer", "coverage_synthesis", "sangha"})
    has_steward = bool((proposal_roles | role_kinds | role_names) & {"steward", "dharma_steward"})
    has_contract_path = any(
        any(token in path for token in ("contract", "policy", "security", "safety", "guardrail"))
        for path in path_set | patched_paths
    )
    has_dependency_audit = bool(
        result_metadata.get("leave_one_backend_dependency")
        or result_metadata.get("leave_one_backend_out")
        or result_metadata.get("backend_lineage")
    )
    checks = [
        _governance_check(
            "role_diversity",
            len(non_seed_roles) >= 3 or len(role_names) >= 3,
            evidence={"roles": sorted(role_names | non_seed_roles)},
            reason="multiple independent proposal roles reduce single-strategy collapse",
        ),
        _governance_check(
            "topology_adaptation",
            bool(role_kinds) or bool(result_metadata.get("role_graph")),
            evidence={"role_kinds": sorted(kind for kind in role_kinds if kind)},
            reason="role graph or role-kind metadata records the optimizer topology",
        ),
        _governance_check(
            "adversarial_review",
            has_critique,
            evidence={"roles": sorted(role_names | proposal_roles | role_kinds)},
            reason="critic or adversary role challenges candidate changes",
        ),
        _governance_check(
            "mediator_review",
            has_synthesis,
            evidence={"roles": sorted(role_names | proposal_roles | role_kinds)},
            reason="synthesis role combines compatible local repairs",
        ),
        _governance_check(
            "steward_review",
            has_steward,
            evidence={"roles": sorted(role_names | proposal_roles | role_kinds)},
            reason="steward role tests minimality and process safety",
        ),
        _governance_check(
            "credit_assignment",
            bool(role_credit),
            evidence={"credit_roles": [str(item.get("role")) for item in role_credit]},
            reason="role credit ledger connects outcomes to proposal sources",
        ),
        _governance_check(
            "search_locality",
            bool(path_set) and patched_paths.issubset(path_set),
            evidence={"search_paths": sorted(path_set), "patched_paths": sorted(patched_paths)},
            reason="candidate patches stay inside diagnosed search paths",
        ),
        _governance_check(
            "contract_gate",
            has_contract_path,
            evidence={"search_paths": sorted(path_set), "diagnostics": diagnostics},
            reason="policy/security/contract paths are tied to diagnosed failures",
        ),
        _governance_check(
            "rollback_check",
            has_steward or any(_as_list(proposal.get("parent_ids")) for proposal in proposals),
            evidence={"has_steward": has_steward},
            reason="steward or parent lineage supports rollback/minimality audit",
        ),
        _governance_check(
            "terminal_selection",
            bool(best_candidate_id) and final_score is not None,
            evidence={"best_candidate_id": best_candidate_id, "final_score": final_score},
            reason="trace names the selected candidate and final score",
        ),
        _governance_check(
            "dependency_audit",
            has_dependency_audit,
            evidence={
                "leave_one_backend_dependency": result_metadata.get("leave_one_backend_dependency"),
                "backend_lineage": result_metadata.get("backend_lineage"),
            },
            reason="multi-backend runs should expose dependency or backend-lineage evidence",
        ),
    ]
    # ---- Phase 4 additive governance records (conditional: emitted only when
    # the producing metadata is present, so legacy traces keep their exact
    # pre-Phase-4 check census). ----
    rejection_records = [
        dict(item)
        for item in _as_list(result_metadata.get("rejections"))
        if isinstance(item, Mapping)
    ]
    nirnaya_records = [
        dict(item)
        for item in _as_list(result_metadata.get("nirnaya"))
        if isinstance(item, Mapping)
    ]
    selected_by_round: Dict[Any, set[str]] = {}
    for record in nirnaya_records:
        selected = record.get("selected_candidate_id")
        if selected:
            seen_round = selected_by_round.setdefault(record.get("round"), set())
            seen_round.add(str(selected))
            if len(seen_round) > 1:
                raise ValueError(
                    "nirnaya records more than one selected candidate for round "
                    f"{record.get('round')!r}: selection is single-lineage and "
                    "proposals are never averaged."
                )
        justification = record.get("justification")
        if justification is not None:
            _validate_justification(justification)
    chambers_meta = result_metadata.get("chambers")
    chambers_meta = dict(chambers_meta) if isinstance(chambers_meta, Mapping) else None
    ledger_round_records = [
        dict(item)
        for item in _as_list(result_metadata.get("ledger_rounds"))
        if isinstance(item, Mapping)
    ]
    strategy_metadata = result_metadata.get("strategy_metadata")
    strategy_metadata = (
        dict(strategy_metadata) if isinstance(strategy_metadata, Mapping) else {}
    )

    proposal_author_counts: Dict[str, int] = {}
    for proposal in proposals:
        author = _normalize(proposal.get("role"))
        if author and author not in {"seed", "unknown"}:
            proposal_author_counts[author] = proposal_author_counts.get(author, 0) + 1
    critique_operators: list[Dict[str, Any]] = []
    for role in roles:
        role_kind = _normalize(role.get("proposal_kind"))
        critique_kind = _normalize(role.get("critique_kind")) or (
            _CRITIQUE_KIND_BY_ROLE_KIND.get(role_kind, "")
        )
        if not critique_kind:
            continue
        role_name = _normalize(role.get("name") or role.get("role"))
        authored = proposal_author_counts.get(role_name, 0)
        operator: Dict[str, Any] = {
            "role": str(role.get("name") or role.get("role") or ""),
            "critique_kind": critique_kind,
            "proposals_authored": authored,
        }
        if critique_kind == "vitanda" and authored:
            # Refutation-only operators may reject; they must never appear as
            # a proposal author.
            operator["error"] = "vitanda_operator_authored_proposal"
        critique_operators.append(operator)

    role_prefixes = {
        _normalize(role.get("name") or role.get("role")): [
            str(prefix) for prefix in _as_list(role.get("path_prefixes")) if str(prefix)
        ]
        for role in roles
    }
    role_kinds_by_name = {
        _normalize(role.get("name") or role.get("role")): _normalize(
            role.get("proposal_kind")
        )
        for role in roles
    }
    authority_weights: list[Dict[str, Any]] = []
    for proposal in proposals:
        role_name = _normalize(proposal.get("role"))
        prefixes = role_prefixes.get(role_name) or [
            str(prefix)
            for prefix in _as_list(
                dict(proposal.get("metadata") or {}).get("role_path_prefixes")
            )
            if str(prefix)
        ]
        role_kind = role_kinds_by_name.get(role_name) or _normalize(
            proposal.get("role_kind")
        )
        if role_kind != "specialist" and not prefixes:
            continue
        patch_paths = [str(path) for path in _as_list(proposal.get("search_paths"))]
        in_scope = bool(prefixes) and all(
            any(path == prefix or path.startswith(f"{prefix}.") for prefix in prefixes)
            for path in patch_paths
        )
        authority_weights.append(
            {
                "candidate_id": proposal.get("candidate_id"),
                "role": proposal.get("role"),
                "weight": 1.0 if in_scope else 0.5,
                "in_scope": in_scope,
            }
        )

    if chambers_meta is not None and any(
        isinstance(entry, Mapping) and entry.get("declared_budget") is not None
        for entry in chambers_meta.values()
    ):
        checks.append(
            _governance_check(
                "chamber_budgets_declared",
                all(
                    isinstance(entry, Mapping)
                    and entry.get("declared_budget") is not None
                    for entry in chambers_meta.values()
                ),
                evidence={"chambers": chambers_meta},
                reason="every chamber declares its evaluation budget per round",
            )
        )
    if rejection_records:
        checks.append(
            _governance_check(
                "rejections_classed",
                all(
                    str(record.get("hetvabhasa_class"))
                    in HETVABHASA_REJECTION_CLASSES
                    for record in rejection_records
                ),
                evidence={
                    "rejection_count": len(rejection_records),
                    "classes": sorted(
                        {
                            str(record.get("hetvabhasa_class"))
                            for record in rejection_records
                        }
                    ),
                },
                reason="every recorded rejection carries a closed-vocabulary class",
            )
        )
    if nirnaya_records:
        checks.append(
            _governance_check(
                "nirnaya_recorded",
                all(
                    record.get("selected_candidate_id")
                    and isinstance(record.get("justification"), Mapping)
                    and all(
                        str(record["justification"].get(member) or "").strip()
                        for member in PANCA_AVAYAVA_MEMBERS
                    )
                    for record in nirnaya_records
                ),
                evidence={"nirnaya_count": len(nirnaya_records)},
                reason="the steward decision is recorded with a complete justification",
            )
        )
        checks.append(
            _governance_check(
                "proposals_never_averaged",
                all(
                    isinstance(record.get("selected_candidate_id"), str)
                    and record.get("selected_candidate_id")
                    for record in nirnaya_records
                )
                and all(len(selected) == 1 for selected in selected_by_round.values()),
                evidence={
                    "selected_candidates": sorted(
                        str(record.get("selected_candidate_id"))
                        for record in nirnaya_records
                    )
                },
                reason="selection is single-lineage: one decided candidate, never an average",
            )
        )
    if authority_weights:
        checks.append(
            _governance_check(
                "specialist_authority_respected",
                all(
                    record["weight"] == (1.0 if record["in_scope"] else 0.5)
                    for record in authority_weights
                ),
                evidence={"authority_weight_count": len(authority_weights)},
                reason=(
                    "specialist proposals inside their path prefixes carry full "
                    "authority; out-of-scope counter-proposals carry half"
                ),
            )
        )
    if ledger_round_records:
        checks.append(
            _governance_check(
                "society_ledger_pooled_across_candidates",
                any(
                    int(record.get("pooled_from_candidates") or 0) > 1
                    for record in ledger_round_records
                ),
                evidence={"ledger_rounds": ledger_round_records},
                reason=(
                    "the round ledger pools diagnoses across all evaluated "
                    "candidates, not just the round winner"
                ),
            )
        )

    checks.extend(_normalize_governance_check(item) for item in explicit_checks)
    checks = [check for check in checks if check]
    seen: Dict[str, int] = {}
    deduped_checks: list[Dict[str, Any]] = []
    for check in checks:
        name = _normalize(check.get("name"))
        if not name:
            continue
        if name in seen:
            existing_index = seen[name]
            if check.get("passed") and not deduped_checks[existing_index].get("passed"):
                deduped_checks[existing_index] = check
            continue
        seen[name] = len(deduped_checks)
        deduped_checks.append(check)
    signals = {
        "governance",
        *(_normalize(check.get("name")) for check in deduped_checks if check.get("passed")),
        *(_normalize(signal) for signal in explicit_signals if _normalize(signal)),
    }
    passed_count = sum(1 for check in deduped_checks if check.get("passed"))
    check_count = len(deduped_checks)

    def _valid_guna(value: Any) -> bool:
        if not isinstance(value, Mapping):
            return False
        if set(str(key) for key in value) != set(GUNA_AXES):
            return False
        try:
            return all(0.0 <= float(value[axis]) <= 1.0 for axis in GUNA_AXES)
        except (TypeError, ValueError):
            return False

    non_seed_proposals = [
        proposal
        for proposal in proposals
        if _normalize(proposal.get("role")) not in {"seed", "unknown", ""}
    ]
    guna_mix = result_metadata.get("guna_mix") or strategy_metadata.get("guna_mix")
    staged_conditioning = result_metadata.get(
        "staged_conditioning"
    ) or strategy_metadata.get("staged_conditioning")
    layer_locality = result_metadata.get("layer_locality")
    declared_chamber_budget = chambers_meta is not None and any(
        isinstance(entry, Mapping) and entry.get("declared_budget") is not None
        for entry in chambers_meta.values()
    )
    summary = {
        "governance_check_count": check_count,
        "governance_passed_count": passed_count,
        "governance_pass_rate": round(passed_count / check_count, 4) if check_count else 0.0,
        "has_governance": check_count > 0,
        "has_role_diversity": _governance_passed(deduped_checks, "role_diversity"),
        "has_mediator": _governance_passed(deduped_checks, "mediator_review"),
        "has_contract_gate": _governance_passed(deduped_checks, "contract_gate"),
        "has_rollback": _governance_passed(deduped_checks, "rollback_check"),
        "has_locality": _governance_passed(deduped_checks, "search_locality"),
        "has_dependency_audit": _governance_passed(deduped_checks, "dependency_audit"),
        # ---- Phase 4 society/contract flags (additive) ----
        "has_guna_axes": bool(roles)
        and all(_valid_guna(role.get("guna")) for role in roles)
        and isinstance(guna_mix, Mapping),
        "has_two_chamber": bool(roles)
        and all(str(role.get("chamber")) in CHAMBER_TOKENS for role in roles)
        and chambers_meta is not None,
        "has_nyaya_justifications": bool(non_seed_proposals)
        and all(
            isinstance(
                dict(proposal.get("metadata") or {}).get("justification"), Mapping
            )
            and all(
                str(
                    dict(proposal.get("metadata") or {})["justification"].get(member)
                    or ""
                ).strip()
                for member in PANCA_AVAYAVA_MEMBERS
            )
            for proposal in non_seed_proposals
        ),
        "has_hetvabhasa_rejections": bool(rejection_records)
        and _governance_passed(deduped_checks, "rejections_classed"),
        "has_nirnaya": bool(nirnaya_records)
        and _governance_passed(deduped_checks, "nirnaya_recorded"),
        "has_staged_conditioning": isinstance(staged_conditioning, Mapping)
        and bool(staged_conditioning),
        "has_layer_locality": bool(layer_locality)
        or any(
            str(diagnostic.get("harness_layer") or "")
            for diagnostic in diagnostics
        ),
        "has_declared_budget": result_metadata.get("eval_budget") is not None
        or declared_chamber_budget,
        "has_external_ranking": str(result_metadata.get("ranking_source") or "")
        in {"evaluation_suite", "evaluator"},
    }
    return {
        "checks": deduped_checks,
        "signals": sorted(signal for signal in signals if signal),
        "summary": summary,
        "nirnaya": nirnaya_records,
        "critique_operators": critique_operators,
        "authority_weights": authority_weights,
        "rejections": rejection_records,
    }


def _governance_check(
    name: str,
    passed: bool,
    *,
    evidence: Optional[Mapping[str, Any]] = None,
    reason: str = "",
) -> Dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "reason": reason,
        "evidence": dict(evidence or {}),
    }


def _normalize_governance_check(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        check = dict(value)
        name = _normalize(check.get("name") or check.get("check") or check.get("signal"))
        if not name:
            return {}
        return {
            "name": name,
            "passed": bool(check.get("passed", check.get("match", True))),
            "reason": str(check.get("reason") or ""),
            "evidence": dict(check.get("evidence") or {}),
        }
    name = _normalize(value)
    if not name:
        return {}
    return {"name": name, "passed": True, "reason": "", "evidence": {}}


def _governance_passed(checks: Iterable[Mapping[str, Any]], name: str) -> bool:
    normalized = _normalize(name)
    return any(_normalize(check.get("name")) == normalized and bool(check.get("passed")) for check in checks)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_").replace(".", "_")
