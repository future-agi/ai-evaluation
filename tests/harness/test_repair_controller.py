from __future__ import annotations

import random

import pytest

from fi.alk.harness.diagnostics import HarnessDiagnostic
from fi.alk.harness.job import HarnessStage
from fi.alk.harness.repair_controller import (
    CandidateObservation,
    RepairAction,
    RepairBudgets,
    RepairController,
    RepairOutcome,
    RepairPhase,
)


def _diagnostic(code: str) -> HarnessDiagnostic:
    return HarnessDiagnostic.create(
        stage=HarnessStage.VALIDATING_ENVIRONMENT,
        component="test",
        code=code,
        message=f"failure {code}",
    )


def _observation(
    candidate: str,
    *codes: str,
    phase: RepairPhase = RepairPhase.ENVIRONMENT,
) -> CandidateObservation:
    return CandidateObservation(
        candidate_hash=candidate,
        phase=phase,
        diagnostics=tuple(_diagnostic(code) for code in codes),
    )


def test_passing_candidate_is_certified() -> None:
    decision = RepairController().decide(_observation("candidate-a"))

    assert decision.action is RepairAction.CERTIFY


def test_agent_source_and_unsupported_failures_stop_immediately() -> None:
    for code in (
        "agent_tool_argument_invalid",
        "credential_missing",
        "unsupported_source_construct",
    ):
        decision = RepairController().decide(_observation("candidate-a", code))
        assert decision.action is RepairAction.REJECT
        assert "forbids harness repair" in decision.reason


def test_infrastructure_retry_budget_is_per_candidate() -> None:
    controller = RepairController(
        RepairBudgets(
            infrastructure_retries_per_candidate=2,
            infrastructure_initial_backoff_seconds=2,
            infrastructure_jitter_ratio=0,
        ),
        rng=random.Random(7),
    )

    actions = [
        controller.decide(
            _observation("candidate-a", "process_dependency_timeout")
        ).action
        for _ in range(3)
    ]
    other = controller.decide(_observation("candidate-b", "process_dependency_timeout"))

    assert actions == [
        RepairAction.RETRY_INFRASTRUCTURE,
        RepairAction.RETRY_INFRASTRUCTURE,
        RepairAction.REJECT,
    ]
    assert other.action is RepairAction.RETRY_INFRASTRUCTURE
    assert [item.retry_after_seconds for item in controller.history.decisions[:3]] == [
        2.0,
        4.0,
        None,
    ]


def test_repeated_compiler_failure_without_new_candidate_is_rejected() -> None:
    controller = RepairController()
    observation = _observation("candidate-a", "array_shape_mismatch")

    assert controller.decide(observation).action is RepairAction.RECOMPILE
    repeated = controller.decide(observation)

    assert repeated.action is RepairAction.REJECT
    assert "without a materially changed candidate" in repeated.reason


def test_compiler_is_applied_before_authoring_for_mixed_failures() -> None:
    decision = RepairController().decide(
        _observation("candidate-a", "schema_type_mismatch", "foreign_key_missing")
    )

    assert decision.action is RepairAction.RECOMPILE
    assert len(decision.diagnostic_fingerprints) == 1


def test_environment_and_scenario_patch_budgets_are_independent() -> None:
    controller = RepairController(
        RepairBudgets(environment_patches=1, scenario_patches=1)
    )

    environment = controller.decide(_observation("env-a", "foreign_key_missing"))
    scenarios = controller.decide(
        _observation(
            "scenarios-a",
            "ready_condition_invalid",
            phase=RepairPhase.SCENARIOS,
        )
    )
    environment_exhausted = controller.decide(
        _observation("env-b", "foreign_key_missing")
    )
    scenarios_exhausted = controller.decide(
        _observation(
            "scenarios-b",
            "ready_condition_invalid",
            phase=RepairPhase.SCENARIOS,
        )
    )

    assert environment.action is RepairAction.PATCH_ENVIRONMENT
    assert scenarios.action is RepairAction.PATCH_SCENARIOS
    assert environment_exhausted.action is RepairAction.REJECT
    assert scenarios_exhausted.action is RepairAction.REJECT


def test_history_contains_value_free_audit_data() -> None:
    controller = RepairController()
    controller.decide(_observation("candidate-a", "array_shape_mismatch"))
    controller.decide(_observation("candidate-b"))

    history = controller.history
    assert [decision.sequence for decision in history.decisions] == [1, 2]
    assert history.compiler_candidates_used == 1
    assert history.environment_patches_used == 0
    assert "failure array_shape_mismatch" not in history.model_dump_json()


def test_repair_result_records_before_after_and_no_op() -> None:
    controller = RepairController()
    decision = controller.decide(_observation("candidate-a", "array_shape_mismatch"))

    result = controller.record_result(
        decision.sequence,
        after_candidate_hash="candidate-a",
        outcome=RepairOutcome.APPLIED,
    )

    assert result.before_candidate_hash == "candidate-a"
    assert result.after_candidate_hash == "candidate-a"
    assert result.outcome is RepairOutcome.NO_MATERIAL_CHANGE
    assert controller.history.results == (result,)

    with pytest.raises(ValueError, match="already_recorded"):
        controller.record_result(
            decision.sequence,
            after_candidate_hash="candidate-b",
            outcome=RepairOutcome.APPLIED,
        )
