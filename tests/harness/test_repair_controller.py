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
        for _ in range(4)
    ]
    other = controller.decide(_observation("candidate-b", "process_dependency_timeout"))

    assert actions == [
        RepairAction.RETRY_INFRASTRUCTURE,
        RepairAction.RETRY_INFRASTRUCTURE,
        RepairAction.PATCH_ENVIRONMENT,
        RepairAction.REJECT,
    ]
    assert other.action is RepairAction.RETRY_INFRASTRUCTURE
    assert [item.retry_after_seconds for item in controller.history.decisions[:3]] == [
        2.0,
        4.0,
        None,
    ]


def test_external_service_outage_does_not_trigger_runtime_reauthoring() -> None:
    controller = RepairController(RepairBudgets(infrastructure_retries_per_candidate=0))

    decision = controller.decide(
        _observation("candidate-a", "external_service_unavailable")
    )

    assert decision.action is RepairAction.REJECT


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


@pytest.mark.parametrize("changed", [True, False])
@pytest.mark.parametrize("code", ["chat_ingress_failed", "chat_tool_evidence_unverified"])
def test_chat_setup_hypotheses_are_bounded_and_keep_source_ownership(changed, code):
    from fi.alk.harness.diagnostics import RepairOwner

    diagnostic = HarnessDiagnostic.create(
        stage=HarnessStage.CONNECTING_AGENT,
        component="chat_ingress",
        code=code,
        message="Application error",
    )
    controller = RepairController(RepairBudgets(environment_patches=2))
    expected_actions = (
        [
            RepairAction.PATCH_ENVIRONMENT,
            RepairAction.PATCH_ENVIRONMENT,
            RepairAction.REJECT,
        ]
        if changed
        else [RepairAction.PATCH_ENVIRONMENT, RepairAction.REJECT, RepairAction.REJECT]
    )
    for index, expected in enumerate(expected_actions):
        decision = controller.decide(
            CandidateObservation(
                candidate_hash=f"candidate-{index if changed else 0}",
                phase=RepairPhase.ENVIRONMENT,
                diagnostics=(diagnostic,),
            )
        )
        assert decision.action is expected
    assert diagnostic.owner is RepairOwner.SOURCE
    assert controller.history.environment_patches_used == (2 if changed else 1)


@pytest.mark.parametrize(
    "budget,phase,extra,component",
    [
        (0, RepairPhase.ENVIRONMENT, (), "chat_ingress"),
        (4, RepairPhase.SCENARIOS, (), "chat_ingress"),
        (4, RepairPhase.ENVIRONMENT, ("credential_missing",), "chat_ingress"),
        (
            4,
            RepairPhase.ENVIRONMENT,
            ("generated_runtime_plan_invalid",),
            "chat_ingress",
        ),
        (4, RepairPhase.ENVIRONMENT, (), "other"),
    ],
)
def test_chat_dependency_hypothesis_cannot_expand_repair_permission(
    budget, phase, extra, component
):
    diagnostic = HarnessDiagnostic.create(
        stage=HarnessStage.CONNECTING_AGENT,
        component=component,
        code="chat_ingress_failed",
        message="Application error",
    )
    controller = RepairController(RepairBudgets(environment_patches=budget))
    assert (
        controller.decide(
            CandidateObservation(
                candidate_hash="candidate",
                phase=phase,
                diagnostics=(diagnostic,) + tuple(_diagnostic(code) for code in extra),
            )
        ).action
        is RepairAction.REJECT
    )


def test_chat_investigation_cannot_exceed_already_consumed_environment_budget():
    controller = RepairController(RepairBudgets(environment_patches=1))
    assert controller.decide(_observation("first", "generated_runtime_plan_invalid")).action is RepairAction.PATCH_ENVIRONMENT
    diagnostic = HarnessDiagnostic.create(stage=HarnessStage.CONNECTING_AGENT, component="chat_ingress", code="chat_ingress_failed", message="application error")
    assert controller.decide(CandidateObservation(candidate_hash="second", phase=RepairPhase.ENVIRONMENT, diagnostics=(diagnostic,))).action is RepairAction.REJECT


@pytest.mark.parametrize("code", ["external_service_unavailable", "chat_ingress_timeout"])
def test_ingress_failure_retry_budget_is_stable_and_does_not_reauthor(code):
    controller = RepairController(RepairBudgets(infrastructure_retries_per_candidate=2))
    actions = [controller.decide(_observation("same-candidate", code)).action for _ in range(4)]
    assert actions == [RepairAction.RETRY_INFRASTRUCTURE, RepairAction.RETRY_INFRASTRUCTURE,
                       RepairAction.REJECT, RepairAction.REJECT]
    assert controller.decide(_observation("changed-candidate", code)).action is RepairAction.RETRY_INFRASTRUCTURE


@pytest.mark.parametrize("code", ["ready_condition_invalid", "process_dependency_timeout"])
def test_runtime_budget_is_bounded_and_independent_of_chat_repairs(code):
    controller = RepairController(RepairBudgets(
        runtime_patches=1, environment_patches=1,
        infrastructure_retries_per_candidate=0,
    ))
    runtime = _observation("runtime-a", code, phase=RepairPhase.RUNTIME)
    assert controller.decide(runtime).action is RepairAction.PATCH_ENVIRONMENT
    assert controller.decide(runtime).action is RepairAction.REJECT
    assert controller.decide(_observation(
        "runtime-b", code, phase=RepairPhase.RUNTIME,
    )).action is RepairAction.REJECT
    assert controller.decide(_observation("chat-a", code)).action is RepairAction.PATCH_ENVIRONMENT
    assert controller.decide(_observation("chat-b", code)).action is RepairAction.REJECT
    assert controller.history.runtime_patches_used == 1
    assert controller.history.environment_patches_used == 1


def test_default_runtime_budget_allows_eight_changed_candidates_but_not_repeats():
    controller = RepairController()
    for attempt in range(8):
        observation = _observation(
            f"candidate-{attempt}", "ready_condition_invalid", phase=RepairPhase.RUNTIME,
        )
        assert controller.decide(observation).action is RepairAction.PATCH_ENVIRONMENT
        assert controller.decide(observation).action is RepairAction.REJECT
    assert controller.history.runtime_patches_used == 8
    assert controller.decide(_observation(
        "candidate-8", "ready_condition_invalid", phase=RepairPhase.RUNTIME,
    )).action is RepairAction.REJECT
