"""Phase 4 (optimizer expansion) focused tests — engine units 1.x + facade
units 2.x. Gates (trinity.py) land separately; nothing here touches them."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from fi.alk import cli, optimize
from fi.opt.components import (
    COMPONENT_SPECS,
    HARNESS_LAYER_PATH_PREFIXES,
    HARNESS_LAYERS,
    ComponentDiagnosis,
    relevant_search_paths,
)
from fi.opt.optimizer_trace import (
    build_optimizer_society_trace,
    optimizer_trajectory_profile,
)
from fi.opt.optimizers.agent_evolution import AgentEvolutionOptimizer
from fi.opt.optimizers.council import (
    CHAMBER_TOKENS,
    GUNA_ARCHETYPE_DEFAULTS,
    HETVABHASA_REJECTION_CLASSES,
    PANCA_AVAYAVA_MEMBERS,
    SocietyAgentOptimizer,
    SocietyRoleGraphSearchStrategy,
    _guna_mix,
    _normalize_society_role_graph,
)
from fi.opt.targets import OptimizationTarget
from fi.opt.types import EvaluationResult, IterationHistory, OptimizationResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SEARCH_SPACE = {
    "memory.depth": [1, 2, 3],
    "tools.retries": [0, 1],
    "policy.mode": ["lax", "strict"],
}


def _target() -> OptimizationTarget:
    return OptimizationTarget(
        name="phase4-fixture",
        layers=[],
        base_config={
            "memory": {"depth": 1},
            "tools": {"retries": 0},
            "policy": {"mode": "lax"},
        },
        search_space=copy.deepcopy(_SEARCH_SPACE),
    )


def _evaluate(candidate: Any) -> float:
    score = 0.4
    if candidate.get_path("memory.depth") == 3:
        score += 0.3
    if candidate.get_path("tools.retries") == 1:
        score += 0.2
    if candidate.get_path("policy.mode") == "strict":
        score += 0.1
    return score


def _society_result(**kwargs: Any) -> OptimizationResult:
    optimizer = SocietyAgentOptimizer(
        target=_target(),
        evaluate_candidate=_evaluate,
        max_rounds=3,
        search_strategy=SocietyRoleGraphSearchStrategy(),
        **kwargs,
    )
    return optimizer.optimize()


# ---------------------------------------------------------------------------
# Unit 1.1: harness-layer diagnosis locality
# ---------------------------------------------------------------------------


def test_harness_layers_canon_and_prefixes_exist_in_component_specs() -> None:
    assert HARNESS_LAYERS == (
        "execution",
        "tool_interface",
        "context_memory",
        "lifecycle",
        "observability",
        "verification",
        "governance",
    )
    all_config_paths = {
        path for spec in COMPONENT_SPECS.values() for path in spec.config_paths
    }
    for layer, prefixes in HARNESS_LAYER_PATH_PREFIXES.items():
        assert layer in HARNESS_LAYERS
        for prefix in prefixes:
            assert any(
                config_path == prefix or config_path.startswith(f"{prefix}.")
                for config_path in all_config_paths
            ), f"layer prefix {prefix!r} ({layer}) not grounded in COMPONENT_SPECS"
    # Validator: non-member layer strings are rejected; members pass.
    with pytest.raises(ValueError):
        ComponentDiagnosis(
            component="memory",
            failure_mode="memory_retrieval_failure",
            harness_layer="not_a_layer",
        )
    diagnosis = ComponentDiagnosis(
        component="memory",
        failure_mode="memory_retrieval_failure",
        harness_layer="context_memory",
    )
    assert diagnosis.harness_layer == "context_memory"


def test_relevant_search_paths_untagged_reproduces_legacy_behavior() -> None:
    search_space = {
        "memory.depth": [1, 2],
        "tools.retries": [0, 1],
        "unrelated.knob": [True, False],
    }
    diagnosis = ComponentDiagnosis(
        component="memory",
        failure_mode="memory_retrieval_failure",
        suggested_paths=["memory"],
    )
    # Legacy semantics: component+suggested prefixes admit memory.*; the
    # memory spec also admits tool/session-adjacent paths through its
    # config_paths — recompute the legacy expectation explicitly.
    prefixes = {
        *diagnosis.suggested_paths,
        *COMPONENT_SPECS["memory"].config_paths,
        "memory",
    }
    expected = {
        path
        for path in search_space
        if any(path == prefix or path.startswith(f"{prefix}.") for prefix in prefixes)
    } or set(search_space)
    assert relevant_search_paths(search_space, [diagnosis]) == expected
    assert relevant_search_paths(search_space, []) == set(search_space)


def test_relevant_search_paths_layer_scoping_narrows_and_falls_back() -> None:
    search_space = {
        "memory.depth": [1, 2],
        "tools.retries": [0, 1],
    }
    tagged = ComponentDiagnosis(
        component="memory",
        failure_mode="memory_retrieval_failure",
        suggested_paths=["memory", "tools"],
        harness_layer="context_memory",
    )
    scoped = relevant_search_paths(search_space, [tagged])
    assert scoped == {"memory.depth"}  # tools.* rejected by the layer

    # Empty layer intersection degrades to the component-scoped set, never to
    # the whole space.
    governance_tagged = ComponentDiagnosis(
        component="memory",
        failure_mode="memory_retrieval_failure",
        suggested_paths=["memory", "tools"],
        harness_layer="governance",
    )
    fallback = relevant_search_paths(search_space, [governance_tagged])
    untagged = ComponentDiagnosis(
        component="memory",
        failure_mode="memory_retrieval_failure",
        suggested_paths=["memory", "tools"],
    )
    assert fallback == relevant_search_paths(search_space, [untagged])


# ---------------------------------------------------------------------------
# Unit 1.2: Elo tournament selection mode
# ---------------------------------------------------------------------------


def test_evolution_tournament_default_is_deterministic_and_annotated() -> None:
    runs = []
    for _ in range(2):
        result = AgentEvolutionOptimizer(
            target=_target(),
            evaluate_candidate=_evaluate,
            population_size=4,
            generations=2,
            seed=7,
        ).optimize()
        runs.append(result)
    assert runs[0].best_candidate.id == runs[1].best_candidate.id
    assert [item.candidate_id for item in runs[0].history] == [
        item.candidate_id for item in runs[1].history
    ]
    assert runs[0].metadata["selection"] == "tournament"
    assert runs[0].metadata["eval_budget"] is None
    assert runs[0].metadata["evaluations_used"] == len(runs[0].history)
    assert "elo_ratings" not in runs[0].metadata


def test_evolution_elo_mode_is_deterministic_and_records_ratings() -> None:
    runs = []
    for _ in range(2):
        result = AgentEvolutionOptimizer(
            target=_target(),
            evaluate_candidate=_evaluate,
            population_size=4,
            generations=2,
            seed=7,
            selection="elo",
        ).optimize()
        runs.append(result)
    assert runs[0].best_candidate.id == runs[1].best_candidate.id
    assert runs[0].metadata["elo_ratings"] == runs[1].metadata["elo_ratings"]
    assert runs[0].metadata["selection"] == "elo"
    assert runs[0].metadata["elo_initial_rating"] == 1500.0
    ratings = runs[0].metadata["elo_ratings"]
    assert ratings and runs[0].best_candidate.id in ratings
    # Round-robin Elo with fixed K keeps ratings centred on the initial 1500.
    mean_rating = sum(ratings.values()) / len(ratings)
    assert mean_rating == pytest.approx(1500.0, abs=1.0)


def test_evolution_eval_budget_exhaustion_stops_early() -> None:
    result = AgentEvolutionOptimizer(
        target=_target(),
        evaluate_candidate=_evaluate,
        population_size=6,
        generations=4,
        seed=11,
        eval_budget=5,
    ).optimize()
    assert result.total_evaluations <= 5
    assert result.metadata["evaluations_used"] <= 5
    assert result.early_stopped is True
    assert result.stop_reason == "eval_budget_exhausted"
    # Invalid selection mode / budget declarations raise.
    with pytest.raises(ValueError):
        AgentEvolutionOptimizer(target=_target(), evaluate_candidate=_evaluate, selection="ladder")
    with pytest.raises(ValueError):
        AgentEvolutionOptimizer(target=_target(), evaluate_candidate=_evaluate, eval_budget=0)


# ---------------------------------------------------------------------------
# Unit 1.3: guna mapping + two-chamber roles
# ---------------------------------------------------------------------------


def test_guna_archetype_defaults_table_chambers_and_mix() -> None:
    assert GUNA_ARCHETYPE_DEFAULTS == {
        "focused_action": (0.8, 0.4, 0.2),
        "prudent_critic": (0.7, 0.5, 0.4),
        "orchestrator": (0.5, 0.6, 0.4),
        "working_memory": (0.4, 0.6, 0.5),
        "bridge_builder": (0.6, 0.5, 0.3),
        "charioteer_counsel": (0.3, 0.8, 0.4),
        "collective_synthesis": (0.2, 0.9, 0.3),
        "minimal_process_guardian": (0.1, 0.5, 0.9),
        "": (0.5, 0.5, 0.5),
    }
    roles = _normalize_society_role_graph(None)
    by_name = {role.name: role for role in roles}
    assert by_name["arjuna"].guna == {"rajas": 0.8, "sattva": 0.4, "tamas": 0.2}
    assert by_name["dharma_steward"].guna == {"rajas": 0.1, "sattva": 0.5, "tamas": 0.9}
    # Chamber derives from role kind: generative -> samiti, deliberative -> sabha.
    assert by_name["arjuna"].chamber == "samiti"
    assert by_name["vidura"].chamber == "samiti"
    assert by_name["krishna"].chamber == "sabha"
    assert by_name["sangha"].chamber == "sabha"
    assert by_name["dharma_steward"].chamber == "sabha"
    # Default-graph society mean (ARCH §2e): rajas 0.45 / sattva 0.60 / tamas 0.425.
    assert _guna_mix(roles) == {"rajas": 0.45, "sattva": 0.6, "tamas": 0.425}
    metadata = by_name["arjuna"].to_metadata()
    assert metadata["guna"] == by_name["arjuna"].guna
    assert metadata["chamber"] == "samiti"
    # Validation: out-of-range / incomplete triples and bad chambers raise;
    # explicit values override the derivations.
    base = {"name": "r1", "proposal_kind": "explorer"}
    with pytest.raises(ValueError):
        _normalize_society_role_graph([{**base, "guna": {"rajas": 1.2, "sattva": 0.5, "tamas": 0.5}}])
    with pytest.raises(ValueError):
        _normalize_society_role_graph([{**base, "guna": {"rajas": 0.5, "sattva": 0.5}}])
    with pytest.raises(ValueError):
        _normalize_society_role_graph([{**base, "chamber": "senate"}])
    explicit = _normalize_society_role_graph(
        [{**base, "guna": {"rajas": 0.9, "sattva": 0.1, "tamas": 0.0}, "chamber": "sabha"}]
    )[0]
    assert explicit.guna == {"rajas": 0.9, "sattva": 0.1, "tamas": 0.0}
    assert explicit.chamber == "sabha"  # explicit override beats kind derivation


def test_chamber_budget_exhaustion_skips_only_that_chamber() -> None:
    result = _society_result(samiti_budget=3, sabha_budget=24)
    chambers = result.metadata["chambers"]
    assert set(chambers) == set(CHAMBER_TOKENS)
    assert chambers["samiti"]["declared_budget"] == 3
    assert chambers["samiti"]["evaluations_used"] <= 3
    assert chambers["samiti"]["skipped_proposals"] > 0
    assert chambers["sabha"]["evaluations_used"] > 0
    assert chambers["sabha"]["skipped_proposals"] == 0


def test_rajas_widens_explorer_patches_with_radius_base() -> None:
    narrow_graph = [
        {
            "name": "probe",
            "proposal_kind": "explorer",
            "guna": {"rajas": 0.1, "sattva": 0.5, "tamas": 0.5},
        }
    ]
    wide_graph = [
        {
            "name": "probe",
            "proposal_kind": "explorer",
            "guna": {"rajas": 1.0, "sattva": 0.5, "tamas": 0.5},
        }
    ]

    def _max_patch_size(graph: list[dict[str, Any]]) -> int:
        strategy = SocietyRoleGraphSearchStrategy(graph, max_paths_per_proposal=3)
        result = SocietyAgentOptimizer(
            target=_target(),
            evaluate_candidate=_evaluate,
            max_rounds=1,
            search_strategy=strategy,
            auto_diagnose=False,
        ).optimize()
        return max(
            (len(item.metadata.get("patch") or {}) for item in result.history),
            default=0,
        )

    assert _max_patch_size(wide_graph) > _max_patch_size(narrow_graph)
    assert _max_patch_size(narrow_graph) == 1


# ---------------------------------------------------------------------------
# Units 1.4 + 1.6: justifications, rejections, nirnaya, society ledger
# ---------------------------------------------------------------------------


def test_society_round_emits_guna_chambers_justifications_ledger_nirnaya() -> None:
    """The deterministic society-round proof: guna + chambers + panca-avayava
    justifications + classed rejections + nirnaya + pooled ledger all present
    in the governance trace."""

    result = _society_result(samiti_budget=20, sabha_budget=12, society_ledger=True)
    trace = build_optimizer_society_trace(result)
    summary = trace["summary"]
    for flag in (
        "has_guna_axes",
        "has_two_chamber",
        "has_nyaya_justifications",
        "has_hetvabhasa_rejections",
        "has_nirnaya",
        "has_declared_budget",
        "has_external_ranking",
    ):
        assert summary[flag] is True, flag
    # Every non-seed proposal carries a complete five-member justification.
    for proposal in trace["proposals"]:
        if proposal["role"] in {"seed", "unknown"}:
            continue
        justification = proposal["metadata"]["justification"]
        for member in PANCA_AVAYAVA_MEMBERS:
            assert str(justification[member]).strip(), (proposal["role"], member)
    # Every rejection carries a closed-vocabulary class.
    rejections = trace["governance"]["rejections"]
    assert rejections
    assert all(
        record["hetvabhasa_class"] in HETVABHASA_REJECTION_CLASSES
        for record in rejections
    )
    # Single nirnaya, single-lineage selection.
    nirnaya = trace["governance"]["nirnaya"]
    assert len(nirnaya) == 1
    assert nirnaya[0]["selected_candidate_id"] == trace["best_candidate_id"]
    assert nirnaya[0]["decision"] == "promote"
    # Round-scoped ledger pooled across more than one candidate.
    assert trace["ledger"]
    assert any(entry["pooled_from_candidates"] > 1 for entry in trace["ledger"])
    # Governance checks for the new families are present and passing.
    checks = {check["name"]: check["passed"] for check in trace["governance"]["checks"]}
    for name in (
        "chamber_budgets_declared",
        "rejections_classed",
        "nirnaya_recorded",
        "proposals_never_averaged",
        "society_ledger_pooled_across_candidates",
    ):
        assert checks.get(name) is True, name


def test_society_ledger_persists_through_social_memory_store() -> None:
    class _Store:
        pass

    store = _Store()
    _society_result(society_ledger=True, social_memory=store)
    assert getattr(store, "society_ledgers", None)
    first_campaign_rounds = len(store.society_ledgers)
    # Second campaign preloads the persisted ledgers and keeps appending.
    result = _society_result(society_ledger=True, social_memory=store)
    assert len(store.society_ledgers) > first_campaign_rounds
    assert all(
        entry["persisted_via"] == "_Store" for entry in result.metadata["ledger_rounds"]
    )


def test_trace_raises_on_two_selected_candidates_per_round() -> None:
    result = _society_result()
    nirnaya = dict(result.metadata["nirnaya"][0])
    rival = {**nirnaya, "selected_candidate_id": "candidate_other"}
    result.metadata["nirnaya"] = [nirnaya, rival]
    with pytest.raises(ValueError, match="single-lineage"):
        build_optimizer_society_trace(result)


def test_vitanda_operator_with_authored_proposal_gets_error_record() -> None:
    result = _society_result()
    role_graph = [dict(role) for role in result.metadata["role_graph"]]
    for role in role_graph:
        if role["name"] == "vidura":
            role["critique_kind"] = "vitanda"
    result.metadata["role_graph"] = role_graph
    trace = build_optimizer_society_trace(result)
    operators = {
        record["role"]: record for record in trace["governance"]["critique_operators"]
    }
    assert operators["krishna"]["critique_kind"] == "vada"
    vidura = operators["vidura"]
    assert vidura["critique_kind"] == "vitanda"
    if vidura["proposals_authored"]:
        assert vidura["error"] == "vitanda_operator_authored_proposal"


# ---------------------------------------------------------------------------
# Unit 1.5: trajectory fitness profile
# ---------------------------------------------------------------------------


def test_trajectory_profile_fields_on_hand_built_history() -> None:
    def _iteration(candidate_id: str, score: float, patch: dict[str, Any]) -> IterationHistory:
        return IterationHistory(
            prompt="fixture",
            average_score=score,
            individual_results=[EvaluationResult(score=score)],
            candidate_id=candidate_id,
            metadata={"patch": patch},
        )

    history = [
        _iteration("c1", 0.5, {}),
        _iteration("c2", 0.7, {"a": 1}),          # improvement, 1 path
        _iteration("c3", 0.6, {"a": 1, "b": 2}),  # regression vs previous
        _iteration("c4", 0.9, {"a": 1, "b": 2}),  # improvement, 2 paths
        _iteration("c4", 0.9, {"a": 1, "b": 2}),  # duplicate candidate id
    ]
    result = OptimizationResult(
        best_generator="x",
        history=history,
        final_score=0.9,
        total_evaluations=5,
        metadata={"selection": "tournament", "eval_budget": 8},
    )
    profile = optimizer_trajectory_profile(result)
    assert profile["kind"] == "agent-learning.optimizer-trajectory-profile.v1"
    assert profile["iterations"] == 5
    assert profile["evaluations"] == 5
    assert profile["improvement_frequency"] == pytest.approx(2 / 4)
    # Accepted patches: c1 (seed, counts 1 path), c2 (1 path), c4 (2 paths).
    assert profile["semantic_locality"] == pytest.approx(
        round((1.0 + 1.0 + 0.5) / 3, 4)
    )
    assert profile["dedupe_rate"] == pytest.approx(1 - 4 / 5)
    assert profile["regression_count"] == 1
    assert profile["selection"] == "tournament"
    assert profile["eval_budget"] == 8


# ---------------------------------------------------------------------------
# Unit 2.1: frozen capability profiles
# ---------------------------------------------------------------------------

_FROZEN_SETTING = {
    "engine": "local_text",
    "driver": "deterministic",
    "eval_budget": 8,
    "target_kind": "prompt",
}


def _frozen_fixture() -> dict[str, Any]:
    profiles = {
        "kind": "agent-learning.framework-adapter-capability-profiles.v1",
        "profiles": [
            {
                "framework": "langgraph",
                "kind": "agent-learning.framework-adapter-capability-profile.v1",
                "capabilities": [
                    {"name": "task_completion"},
                    {"name": "tool_selection_accuracy"},
                ],
            }
        ],
    }
    return optimize.freeze_capability_profile(
        profiles,
        setting=_FROZEN_SETTING,
        metric_floors={"task_completion": 0.9, "tool_selection_accuracy": 0.8},
        security_rows=[
            {
                "framework": "langgraph",
                "metric": "stored_injection_block_rate",
                "floor": 1.0,
            }
        ],
        source_manifest_ref="fixture://manifest",
        frozen_at="fixture",
    )


def _candidate_payload(metrics: dict[str, float], **extra: Any) -> dict[str, Any]:
    payload = {
        "summary": {"metric_averages": dict(metrics)},
        "setting": dict(_FROZEN_SETTING),
        "patch": {"memory.retrieval.depth": 2},
        "optimization": {"history": []},
    }
    payload.update(extra)
    return payload


def test_freeze_capability_profile_rows_are_content_addressed() -> None:
    frozen = _frozen_fixture()
    assert frozen["kind"] == optimize.AGENT_LEARNING_FROZEN_CAPABILITY_PROFILE_KIND
    assert len(frozen["rows"]) == 3
    for row in frozen["rows"]:
        assert set(row) == set(optimize.FROZEN_CAPABILITY_PROFILE_ROW_FIELDS)
    security_rows = [row for row in frozen["rows"] if row["security"]]
    assert len(security_rows) == 1
    assert security_rows[0]["source"] == "redteam.stored_injection_readiness"
    # Tampering with a row field breaks its content address and is detected.
    tampered = copy.deepcopy(frozen)
    tampered["rows"][0]["floor"] = 0.01
    verdict = optimize.replay_frozen_profile(
        _candidate_payload(
            {
                "task_completion": 1.0,
                "tool_selection_accuracy": 1.0,
                "stored_injection_block_rate": 1.0,
            }
        ),
        tampered,
    )
    assert verdict["veto"] is True
    assert any(not row["integrity_ok"] for row in verdict["rows"])
    assert any(row.get("hetvabhasa_class") == "asiddha" for row in verdict["vetoed_rows"])


def test_replay_vetoes_improving_candidate_with_broken_row() -> None:
    frozen = _frozen_fixture()
    # The candidate improves its searched metric (task_completion 0.95 > 0.9)
    # but breaks the tool_selection_accuracy frozen row — veto wins anyway.
    verdict = optimize.replay_frozen_profile(
        _candidate_payload(
            {
                "task_completion": 0.95,
                "tool_selection_accuracy": 0.5,
                "stored_injection_block_rate": 1.0,
            }
        ),
        frozen,
    )
    assert verdict["veto"] is True
    assert verdict["hetvabhasa_class"] == "badhita"
    assert verdict["vetoed_rows"]
    # A fully compliant candidate promotes.
    verdict_ok = optimize.replay_frozen_profile(
        _candidate_payload(
            {
                "task_completion": 0.95,
                "tool_selection_accuracy": 0.85,
                "stored_injection_block_rate": 1.0,
            }
        ),
        frozen,
    )
    assert verdict_ok["veto"] is False
    assert verdict_ok["closed_row_count"] == verdict_ok["row_count"]


def test_replay_rejects_out_of_setting_wins_and_security_trades() -> None:
    frozen = _frozen_fixture()
    # Different setting digest: rows are non-admissible — the win does not count.
    out_of_setting = optimize.replay_frozen_profile(
        _candidate_payload(
            {
                "task_completion": 1.0,
                "tool_selection_accuracy": 1.0,
                "stored_injection_block_rate": 1.0,
            },
            setting={**_FROZEN_SETTING, "eval_budget": 999},
        ),
        frozen,
    )
    assert out_of_setting["non_admissible_wins"]
    # Security rows are non-tradable: a memory-path candidate that fails the
    # stored-injection row is vetoed regardless of its searched-metric score.
    security_trade = optimize.replay_frozen_profile(
        _candidate_payload(
            {
                "task_completion": 1.0,
                "tool_selection_accuracy": 1.0,
                "stored_injection_block_rate": 0.2,
            }
        ),
        frozen,
    )
    assert security_trade["security_veto"] is True
    assert security_trade["veto"] is True
    assert security_trade["touches_context_memory_paths"] is True
    # Attachment rides the documented key.
    attached = optimize.attach_frozen_profile({"summary": {}}, frozen)
    assert (
        attached[optimize.FROZEN_CAPABILITY_PROFILE_ATTACHMENT_KEY]["contract_digest"]
        == frozen["contract_digest"]
    )


# ---------------------------------------------------------------------------
# Unit 2.2: optimizer profile matrix
# ---------------------------------------------------------------------------


def test_matrix_declares_exactly_40_cells_with_required_coverage() -> None:
    cells = optimize.OPTIMIZER_PROFILE_MATRIX_CELLS
    assert len(cells) == 40  # was 33; +7 Phase-9D modality cells
    assert len(set(cells)) == 40
    assert len(optimize.OPTIMIZER_PROFILE_MATRIX_INHERITED_CELLS) == 6
    new_cells = [
        cell
        for cell in cells
        if cell not in set(optimize.OPTIMIZER_PROFILE_MATRIX_INHERITED_CELLS)
    ]
    assert len(new_cells) == 34  # was 27; +7 Phase-9D modality cells
    # Coverage rules (ARCH §6): every target kind >= 2 backends; every backend
    # >= 2 cells; every framework profile >= 1 Phase-4 (new) cell.
    by_kind: dict[str, set[str]] = {}
    by_backend: dict[str, int] = {}
    new_frameworks = set()
    for framework, target_kind, backend in cells:
        assert framework in optimize.OPTIMIZER_PROFILE_MATRIX_FRAMEWORKS
        assert target_kind in optimize.OPTIMIZER_PROFILE_MATRIX_TARGET_KINDS
        assert backend in optimize.OPTIMIZER_PROFILE_MATRIX_BACKENDS
        by_kind.setdefault(target_kind, set()).add(backend)
        by_backend[backend] = by_backend.get(backend, 0) + 1
    for framework, _, _ in new_cells:
        new_frameworks.add(framework)
    assert set(by_kind) == set(optimize.OPTIMIZER_PROFILE_MATRIX_TARGET_KINDS)
    assert all(len(backends) >= 2 for backends in by_kind.values())
    assert all(count >= 2 for count in by_backend.values())
    assert new_frameworks == set(optimize.OPTIMIZER_PROFILE_MATRIX_FRAMEWORKS)


def test_matrix_manifests_encode_cell_design_rules() -> None:
    manifests = optimize.build_optimizer_profile_matrix_manifests(
        cells=[
            ("langgraph", "memory_ops", "bandit"),
            ("crewai", "multi_agent_roster", "society"),
            ("langgraph", "whole_agent", "tpe"),
        ],
        eval_budget=8,
    )
    memory_manifest = manifests["langgraph/memory_ops/bandit"]
    memory_meta = memory_manifest["optimization"]["target"]["metadata"]
    assert memory_meta["gain_density_prior"] == "retrieval"
    assert memory_meta["slices"] == list(
        optimize.OPTIMIZER_PROFILE_MATRIX_MEMORY_REQUIRED_SLICES
    )
    assert memory_meta["security_row_refs"]
    memory_paths = list(memory_manifest["optimization"]["target"]["search_space"])
    retrieval_index = min(
        index for index, path in enumerate(memory_paths) if "retrieval" in path
    )
    write_index = min(
        index for index, path in enumerate(memory_paths) if ".write." in path
    )
    assert retrieval_index < write_index  # retrieval-side paths come first

    roster_manifest = manifests["crewai/multi_agent_roster/society"]
    roster_paths = list(roster_manifest["optimization"]["target"]["search_space"])
    assert any(
        path.split(".", 1)[0] in optimize.OPTIMIZER_PROFILE_MATRIX_TOPOLOGY_PREFIXES
        for path in roster_paths
    )

    whole_agent_manifest = manifests["langgraph/whole_agent/tpe"]
    assert whole_agent_manifest["whole_agent"]["eval_budget"] == 8
    assert whole_agent_manifest["optimization"]["ranking_source"] == "evaluation_suite"
    for manifest in manifests.values():
        cell = manifest["metadata"]["optimizer_profile_matrix_cell"]
        assert cell["eval_budget"] <= optimize.OPTIMIZER_PROFILE_MATRIX_CELL_EVAL_BUDGET
        assert cell["setting"]["engine"] == "local_text"


def test_matrix_single_cell_runs_end_to_end_with_per_cell_winner() -> None:
    manifests = optimize.build_optimizer_profile_matrix_manifests(
        cells=[("llamaindex", "prompt", "bandit")],
        eval_budget=8,
    )
    payload = optimize.run_optimizer_profile_matrix(manifests)
    assert payload["kind"] == optimize.AGENT_LEARNING_OPTIMIZER_PROFILE_MATRIX_KIND
    assert payload["status"] == "passed"
    cell = payload["cells"][0]
    assert cell["native_proof_closed"] is True
    assert cell["winner"]
    assert cell["trajectory_profile"]["iterations"] > 0
    assert cell["evidence_class"] == "local_gate"
    # Winners are per-cell only — no global aggregate key may appear.
    for key in optimize.OPTIMIZER_PROFILE_MATRIX_FORBIDDEN_AGGREGATE_KEYS:
        assert key not in payload
        assert key not in payload["summary"]
    # The routing table is regenerated from the same-run cells, byte-stably.
    table = payload["routing_table"]
    assert table["kind"] == optimize.AGENT_LEARNING_OPTIMIZER_ROUTING_TABLE_KIND
    assert table["rows"][0]["recommended_backend"] == "bandit"
    regenerated = optimize.build_optimizer_routing_table(payload["cells"])
    assert optimize.render_optimizer_routing_table_json(
        table
    ) == optimize.render_optimizer_routing_table_json(regenerated)
    assert payload["report_card"]["section"] == "optimizer_profile_matrix"
    assert payload["report_card"]["rows"]


# ---------------------------------------------------------------------------
# Unit 2.3: whole-agent contract + apply plan
# ---------------------------------------------------------------------------


def _whole_agent_manifest(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        name="whole-agent-test",
        base_agent={
            "type": "scripted",
            "provider": "elevenlabs",
            "agent_ref": "AGENT_LEARNING_FIXTURE_AGENT_ID",
            "model": "base-model",
            "voice": "base-voice",
            "first_message": "Hello.",
            "instructions": "Answer briefly.",
            "responses": [{"content": "weak"}],
        },
        search_space={
            "model": ["base-model", "tuned-model"],
            "voice": ["base-voice", "warm-voice"],
            "first_message": ["Hello.", "Hi there!"],
            "instructions": ["Answer briefly.", "Answer with a resolution."],
        },
        evaluation_config={"task_description": "t", "expected_result": "strong"},
        eval_budget=12,
    )
    kwargs.update(overrides)
    return optimize.build_whole_agent_optimization_manifest(**kwargs)


def test_whole_agent_contract_orders_stages_and_pins_properties() -> None:
    manifest = _whole_agent_manifest()
    contract = manifest["whole_agent"]
    staged = contract["staged_conditioning"]["stages"]
    # Stage tokens map one-to-one to role-graph phases 1/2/3 (canon order).
    assert list(staged) == list(optimize.WHOLE_AGENT_CONTRACT_STAGES)
    assert [staged[stage]["phase"] for stage in optimize.WHOLE_AGENT_CONTRACT_STAGES] == [1, 2, 3]
    # Text-class paths condition phase 1; structural paths phase 2; all phase 3.
    assert "agent.first_message" in staged["component_text"]["paths"]
    assert "agent.instructions" in staged["component_text"]["paths"]
    assert "agent.model" in staged["structural_config"]["paths"]
    assert "agent.voice" in staged["structural_config"]["paths"]
    assert set(staged["global_repolish"]["paths"]) == set(contract["search_paths"])
    # Declared budget + external-verification-only ranking are gate-pinned.
    assert manifest["optimization"]["eval_budget"] == 12
    assert manifest["optimization"]["ranking_source"] == "evaluation_suite"
    assert contract["ranking_source"] == "evaluation_suite"
    # The default optimizer executes staging inside the role-graph strategy.
    optimizer = manifest["optimization"]["optimizer"]
    assert optimizer["algorithm"] == "council"
    assert optimizer["search_strategy"]["strategy"] == "role_graph"
    assert optimizer["samiti_budget"] + optimizer["sabha_budget"] == 12
    role_names = [role["name"] for role in optimizer["search_strategy"]["role_graph"]]
    assert "component_text_samiti_explorer" in role_names
    assert "global_repolish_sabha_steward" in role_names
    # Required budget + finiteness validations.
    with pytest.raises(ValueError, match="eval_budget"):
        _whole_agent_manifest(eval_budget=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="FINITE"):
        _whole_agent_manifest(search_space={"model": "open-text"})
    with pytest.raises(ValueError, match="selection"):
        _whole_agent_manifest(selection="auto")
    # Elo stays an explicit opt-in mode on the evolution backend.
    elo_manifest = _whole_agent_manifest(selection="elo")
    assert elo_manifest["optimization"]["optimizer"]["algorithm"] == "evolution"
    assert elo_manifest["optimization"]["optimizer"]["selection"] == "elo"


def test_build_apply_plan_emits_diff_and_readback_checks() -> None:
    base_agent = {
        "provider": "elevenlabs",
        "agent_ref": "AGENT_ENV_NAME",
        "model": "base-model",
        "voice": "base-voice",
        "first_message": "Hello.",
    }
    payload = {
        "optimization": {
            "best_candidate_id": "candidate_w",
            "best_config": {
                "agent": {
                    **base_agent,
                    "model": "tuned-model",
                    "first_message": "Hi there!",
                }
            },
            "source_manifest": {
                "whole_agent": {
                    "provider": "elevenlabs",
                    "agent_ref": "AGENT_ENV_NAME",
                    "base_agent": base_agent,
                    "search_paths": [
                        "agent.first_message",
                        "agent.model",
                        "agent.voice",
                    ],
                    "staged_conditioning": {
                        "stages": {
                            "component_text": {"phase": 1, "paths": ["agent.first_message"]},
                            "structural_config": {"phase": 2, "paths": ["agent.model", "agent.voice"]},
                            "global_repolish": {"phase": 3, "paths": []},
                        }
                    },
                    "frozen_profile_ref": "digest123",
                }
            },
            "history": [],
            "optimizer_trace": {
                "governance": {
                    "nirnaya": [{"selected_candidate_id": "candidate_w"}]
                }
            },
        }
    }
    plan = optimize.build_apply_plan(payload)
    assert plan["kind"] == optimize.AGENT_LEARNING_APPLY_PLAN_KIND
    assert sorted(plan) == sorted(["kind", *optimize.WHOLE_AGENT_APPLY_PLAN_FIELDS])
    # Ordered field-level ops, stage order first (text before structural).
    assert plan["apply_fields"] == [
        {"path": "first_message", "from": "Hello.", "to": "Hi there!"},
        {"path": "model", "from": "base-model", "to": "tuned-model"},
    ]
    assert plan["read_back_checks"] == [
        {"path": "first_message", "expected": "Hi there!"},
        {"path": "model", "expected": "tuned-model"},
    ]
    assert plan["mismatch_policy"] == "abort"
    assert plan["frozen_profile_ref"] == "digest123"
    assert plan["nirnaya_ref"] == "candidate_w"
    assert plan["agent_ref"] == "AGENT_ENV_NAME"  # opaque ref, never a credential


# ---------------------------------------------------------------------------
# Unit 2.4 + 2.5: routing table, default picker, CLI --backend flag
# ---------------------------------------------------------------------------

_ROUTING_ARTIFACTS = [
    {
        "target_kind": "prompt",
        "framework_profile": "llamaindex",
        "backend": "bandit",
        "score": 0.98,
        "trajectory_profile": {"improvement_frequency": 0.4},
        "evidence_class": "local_gate",
        "cell_ref": "llamaindex/prompt/bandit",
    },
    {
        "target_kind": "prompt",
        "framework_profile": "llamaindex",
        "backend": "tpe",
        "score": 0.91,
        "trajectory_profile": {"improvement_frequency": 0.6},
        "evidence_class": "local_gate",
        "cell_ref": "llamaindex/prompt/tpe",
    },
    {
        "target_kind": "prompt",
        "framework_profile": "llamaindex",
        "backend": "gepa",
        "score": 0.99,
        "trajectory_profile": {"improvement_frequency": 0.9},
        "evidence_class": "live_lane",
        "cell_ref": "llamaindex/prompt/gepa",
    },
]


def _routing_base_config() -> dict[str, Any]:
    return {
        "agent": {"type": "scripted", "responses": [{"content": "a"}]},
        "simulation": {"engine": "local_text", "environments": []},
    }


def test_routing_table_cites_evidence_and_excludes_live_lane(tmp_path: Path) -> None:
    table = optimize.build_optimizer_routing_table(_ROUTING_ARTIFACTS)
    assert table["kind"] == optimize.AGENT_LEARNING_OPTIMIZER_ROUTING_TABLE_KIND
    row = table["rows"][0]
    # Live-lane gepa scored highest but is excluded from the recommendation.
    assert row["recommended_backend"] == "bandit"
    assert all(
        entry["evidence_class"]
        in optimize.OPTIMIZER_ROUTING_ADMISSIBLE_EVIDENCE_CLASSES
        for entry in row["evidence"]
    )
    assert [entry["cell_ref"] for entry in row["live_lane_evidence"]] == [
        "llamaindex/prompt/gepa"
    ]
    # Every recommendation cites >= 1 matching-axes evidence entry whose
    # winner equals the recommendation.
    assert any(
        entry["backend"] == row["recommended_backend"] for entry in row["evidence"]
    )
    # Deterministic, byte-stable regeneration.
    assert optimize.render_optimizer_routing_table_json(
        table
    ) == optimize.render_optimizer_routing_table_json(
        optimize.build_optimizer_routing_table(_ROUTING_ARTIFACTS)
    )
    # Byte-compare support against a committed copy.
    committed = tmp_path / "optimizer_routing_table.json"
    committed.write_text(
        optimize.render_optimizer_routing_table_json(table), encoding="utf-8"
    )
    assert optimize.routing_table_matches_committed(table, committed) is True
    drifted = copy.deepcopy(table)
    drifted["rows"][0]["recommended_backend"] = "tpe"
    assert optimize.routing_table_matches_committed(drifted, committed) is False


def test_routing_default_picker_engages_overrides_and_cold_starts() -> None:
    table = optimize.build_optimizer_routing_table(_ROUTING_ARTIFACTS)
    common = dict(
        base_config=_routing_base_config(),
        target_candidates={"agent.responses.0.content": ["a", "b"]},
        evaluation_config={"task_description": "t", "expected_result": "b"},
        routing_table=table,
    )
    # Default engagement: omitted optimizer consults the table.
    picked = optimize.build_target_optimization_manifest(
        name="routing-pick",
        target_metadata={"task_kind": "prompt", "framework_profile": "llamaindex"},
        **common,
    )
    evidence = picked["optimization"]["optimizer_routing_evidence"]
    assert evidence["selected_by"] == "routing_table"
    assert evidence["recommended_backend"] == "bandit"
    assert evidence["citations"]
    assert picked["optimization"]["optimizer"]["algorithm"] == "bandit"
    # Cold start: no row -> static default, warning finding, never an error.
    cold = optimize.build_target_optimization_manifest(
        name="routing-cold",
        target_metadata={"task_kind": "framework_method", "framework_profile": "livekit"},
        **common,
    )
    cold_evidence = cold["optimization"]["optimizer_routing_evidence"]
    assert cold_evidence["selected_by"] == "cold_start"
    assert cold_evidence["citations"] == []
    assert cold_evidence["warning"]
    assert cold["optimization"]["optimizer"]["algorithm"] == "agent"
    # Explicit optimizer always overrides; the spurned recommendation stays
    # visible.
    override = optimize.build_target_optimization_manifest(
        name="routing-override",
        optimizer={"algorithm": "agent", "max_candidates": 3},
        target_metadata={"task_kind": "prompt", "framework_profile": "llamaindex"},
        **common,
    )
    override_evidence = override["optimization"]["optimizer_routing_evidence"]
    assert override_evidence["selected_by"] == "override"
    assert override_evidence["routing_table_recommendation"] == "bandit"
    assert override["optimization"]["optimizer"] == {
        "algorithm": "agent",
        "max_candidates": 3,
    }


def test_cli_optimize_backend_flag_overrides_routing(tmp_path: Path) -> None:
    manifest = optimize.build_target_optimization_manifest(
        name="cli-backend-override",
        base_config=_routing_base_config(),
        target_candidates={"agent.responses.0.content": ["a", "b"]},
        evaluation_config={"task_description": "t", "expected_result": "b"},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, default=str), encoding="utf-8")
    output_path = tmp_path / "result.json"
    exit_code = cli._optimize(
        [
            str(manifest_path),
            "--backend",
            "bandit",
            "--dry-run",
            "-o",
            str(output_path),
            "--quiet",
        ]
    )
    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    routing = payload["optimizer_routing"]
    assert routing["selected_by"] == "override"
    assert routing["override_flag"] == "--backend bandit"
    assert routing["backend"] == "bandit"
    assert "routing_table_recommendation" in routing
    # Unknown backend tokens fail loudly (exit 1), not silently.
    bad_exit = cli._optimize(
        [str(manifest_path), "--backend", "warpdrive", "--dry-run", "--quiet"]
    )
    assert bad_exit == 1
