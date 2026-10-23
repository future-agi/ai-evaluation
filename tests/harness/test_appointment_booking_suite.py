"""The clinic base suite against the real SQLite world: handoffs must be for this caller, and
scored agent rules must be graded by a scenario in the suite."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from fi.alk.environments.appointment_booking import build, suite
from fi.alk.harness.catalogue import save_catalogue
from fi.alk.harness.checks import run_check
from fi.alk.harness.folder import write_folder
from fi.alk.harness.prove import _run
from fi.alk.harness.releases import (
    WORLD,
    PolicyProfile,
    ReleaseError,
    apply_policy_profile,
    load_policy,
    seal_suite_release,
)
from fi.alk.harness.scenario import Step


@pytest.fixture(scope="module")
def environment(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("clinic") / "environment"
    build.build_environment(root)
    return root


@pytest.fixture
def world_root(environment, tmp_path) -> Path:
    root = tmp_path / WORLD
    shutil.copytree(environment / WORLD, root)
    apply_policy_profile(root, load_policy(environment), PolicyProfile())
    return root


def _outcome(world_root: Path, scenario_name: str, steps: list[Step], sub_goal: str):
    scenario = next(one for one in suite.scenarios() if one.name == scenario_name)
    played = scenario.model_copy(update={"solution": steps})
    world, calls, refused, _ = _run(played, world_root, with_solution=True)
    try:
        assert not refused, refused
        return run_check(suite.CHECKS[sub_goal][1], world, calls, name=sub_goal)
    finally:
        world.close()


GRACE = Step(
    tool="verify_patient", arguments={"full_name": "Grace Okafor", "dob": "1968-09-30"}
)


def test_late_cancel_handoff_for_another_patient_fails(world_root):
    outcome = _outcome(
        world_root,
        "late_cancel_to_front_desk",
        [
            GRACE,
            Step(
                tool="escalate_to_staff",
                arguments={
                    "team": "front_desk",
                    "reason": "Late cancellation request for this afternoon",
                    "patient_id": "P-1001",
                },
            ),
        ],
        "grace_late_cancel_escalated",
    )
    assert not outcome.held


def test_unverified_handoff_filed_under_another_patient_fails(world_root):
    outcome = _outcome(
        world_root,
        "unverified_caller_to_front_desk",
        [
            Step(
                tool="escalate_to_staff",
                arguments={
                    "team": "front_desk",
                    "reason": "Unrelated request",
                    "patient_id": "P-1001",
                },
            )
        ],
        "unverified_cancel_escalated",
    )
    assert not outcome.held


def test_unverified_handoff_needs_no_identity_or_failed_cancel(world_root):
    outcome = _outcome(
        world_root,
        "unverified_caller_to_front_desk",
        [
            Step(
                tool="escalate_to_staff",
                arguments={
                    "team": "front_desk",
                    "reason": "Caller could not be verified and wants an appointment cancelled",
                },
            )
        ],
        "unverified_cancel_escalated",
    )
    assert outcome.held, outcome.said


def _suite_tree(root: Path) -> Path:
    catalogue = suite.catalogue()
    for scenario in suite.scenarios():
        write_folder(scenario, catalogue, root)
    save_catalogue(catalogue, root)
    (root / "simulator_prompt.md").write_text(build.SIMULATOR_PROMPT, encoding="utf-8")
    return root


@pytest.mark.parametrize(
    "scored",
    [
        # A rule graded by a sub-goal that does not exist.
        {"no_medical_advice": ["gave_no_medical_advice"]},
        # A rule listed as scored with nothing that grades it.
        {"red_flags_to_emergency": []},
    ],
    ids=["unknown-sub-goal", "nothing-grades-it"],
)
def test_a_rule_no_scenario_grades_cannot_be_sealed_as_scored(
    environment, tmp_path, scored
):
    with pytest.raises(ReleaseError) as raised:
        seal_suite_release(
            _suite_tree(tmp_path / "suite"),
            name="clinic_rules",
            environment_root=environment,
            policy_profile={"scored_agent_rules": scored},
        )
    assert raised.value.code == "suite_policy_profile_invalid"
