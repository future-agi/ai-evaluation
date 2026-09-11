"""Task + TaskDataset schema tests.

The Task is a thin composition over existing typed/content-addressed models; the
Goodhart guard is reused VERBATIM from loss.compile_objective. These tests pin:
content addressing + determinism, the guard rejecting a guardless objective, the
deterministic-anchor requirement, world-kind resolution, execution_class honesty
(no overclaim), and dataset coverage/splits/dup-id validation.
"""

from __future__ import annotations

import pytest

from fi.alk import tasks


# --- helpers ---------------------------------------------------------------
def _objective(*, anchored: bool = True, guards: bool = True) -> dict:
    evals = [
        {"eval": "task_success", "weight": 1.0, "direction": "maximize",
         "anchor": True} if anchored else
        {"eval": "instruction_adherence", "weight": 1.0, "direction": "maximize"},
        {"eval": "instruction_adherence", "weight": 0.4, "direction": "maximize"},
    ]
    payload: dict = {"source": "declared", "evals": evals}
    if guards:
        payload["guards"] = {
            "sentinel_rows": [{"id": "answerable_without_tool"}],
            "canary_evals": [{"eval": "refusal_canary"}],
            "min_guard_count": 1,
        }
    else:
        payload["guards"] = {}
    return payload


def _task(**overrides) -> dict:
    base = {
        "id": "refund-policy-lookup",
        "title": "Find and cite the refund policy",
        "world": {"kind": "conversation"},
        "difficulty": "easy",
        "tags": ["support", "grounding"],
        "scenario": {
            "name": "refund-policy-lookup",
            "kind": "task",
            "dataset": [
                {
                    "persona": {"name": "Dana"},
                    "situation": "Dana asks where the refund policy is.",
                    "outcome": "Agent cites the refund policy location.",
                }
            ],
        },
        "objective": _objective(),
        "verification": {"checks": [{"type": "contains", "value": "policy"}], "threshold": 0.7},
    }
    base.update(overrides)
    return base


# --- Task happy path + content addressing ----------------------------------
def test_task_compiles_and_content_addresses() -> None:
    task = tasks.compile_task(_task())
    assert task["kind"] == tasks.AGENT_LEARNING_TASK_KIND
    assert task["version"].startswith("sha256:")
    assert task["execution_class"] == "executable"  # conversation executes in v1
    assert task["anchor_terms"] == ["task_success"]


def test_task_version_is_deterministic() -> None:
    a = tasks.compile_task(_task())
    b = tasks.compile_task(_task())
    assert a["version"] == b["version"]


def test_task_version_changes_with_content() -> None:
    a = tasks.compile_task(_task())
    b = tasks.compile_task(_task(title="A different title"))
    assert a["version"] != b["version"]


# --- guard discipline (reused from loss.compile_objective) ------------------
def test_task_rejects_guardless_objective() -> None:
    with pytest.raises(tasks.TaskError):
        tasks.compile_task(_task(objective=_objective(guards=False)))


def test_task_error_is_valueerror() -> None:
    assert issubclass(tasks.TaskError, ValueError)
    assert issubclass(tasks.TaskDatasetError, tasks.TaskError)


# --- deterministic-anchor requirement --------------------------------------
def test_task_requires_deterministic_anchor() -> None:
    with pytest.raises(tasks.TaskError):
        tasks.compile_task(_task(objective=_objective(anchored=False)))


# --- world-kind resolution --------------------------------------------------
def test_task_rejects_unresolved_world_kind() -> None:
    with pytest.raises(tasks.TaskError):
        tasks.compile_task(_task(world={"kind": "telepathy"}))


def test_task_accepts_typed_only_world_kind_as_typed_only() -> None:
    task = tasks.compile_task(_task(world={"kind": "browser"}))
    assert task["execution_class"] == "typed_only"


# --- execution_class honesty (no overclaim) --------------------------------
def test_browser_task_cannot_be_executable() -> None:
    with pytest.raises(tasks.TaskError):
        tasks.compile_task(_task(world={"kind": "browser"}, execution_class="executable"))


def test_fixture_only_task_is_fixture_class() -> None:
    task = tasks.compile_task(_task(fixture_only=True))
    assert task["execution_class"] == "fixture"


def test_execution_class_underclaim_allowed() -> None:
    # claiming a LOWER class than the substrate supports is fine (honest)
    task = tasks.compile_task(_task(world={"kind": "conversation"}, execution_class="typed_only"))
    assert task["execution_class"] == "executable"  # derived wins; underclaim not error


def test_unknown_execution_class_rejected() -> None:
    with pytest.raises(tasks.TaskError):
        tasks.compile_task(_task(execution_class="live_lane"))


# --- scenario / difficulty / id validation ---------------------------------
def test_task_requires_scenario_kind_task() -> None:
    bad = _task()
    bad["scenario"]["kind"] = "adversarial"
    with pytest.raises(tasks.TaskError):
        tasks.compile_task(bad)


def test_task_requires_exactly_one_row() -> None:
    bad = _task()
    bad["scenario"]["dataset"] = bad["scenario"]["dataset"] * 2
    with pytest.raises(tasks.TaskError):
        tasks.compile_task(bad)


def test_task_rejects_bad_difficulty() -> None:
    with pytest.raises(tasks.TaskError):
        tasks.compile_task(_task(difficulty="trivial"))


def test_task_requires_id_and_title() -> None:
    with pytest.raises(tasks.TaskError):
        tasks.compile_task(_task(id=""))
    with pytest.raises(tasks.TaskError):
        tasks.compile_task(_task(title=""))


# --- TaskDataset ------------------------------------------------------------
def _dataset(**overrides) -> dict:
    base = {
        "name": "support-mini",
        "license": "internal",
        "tasks": [
            _task(),
            _task(id="escalation-path", title="Escalate angry customer",
                  difficulty="medium", world={"kind": "tool_api"}),
        ],
    }
    base.update(overrides)
    return base


def test_dataset_compiles_with_coverage() -> None:
    ds = tasks.compile_task_dataset(_dataset())
    assert ds["kind"] == tasks.AGENT_LEARNING_TASK_DATASET_KIND
    assert ds["coverage"]["count"] == 2
    assert ds["coverage"]["by_world_kind"] == {"conversation": 1, "tool_api": 1}
    assert ds["coverage"]["by_difficulty"] == {"easy": 1, "medium": 1}
    assert ds["version"].startswith("sha256:")
    assert set(tasks.task_world_kinds(ds)) == {"conversation", "tool_api"}


def test_dataset_rejects_duplicate_ids() -> None:
    with pytest.raises(tasks.TaskDatasetError):
        tasks.compile_task_dataset(_dataset(tasks=[_task(), _task()]))


def test_dataset_rejects_empty() -> None:
    with pytest.raises(tasks.TaskDatasetError):
        tasks.compile_task_dataset(_dataset(tasks=[]))


def test_dataset_splits_must_reference_existing_ids() -> None:
    with pytest.raises(tasks.TaskDatasetError):
        tasks.compile_task_dataset(_dataset(splits={"test": ["nonexistent-id"]}))


def test_dataset_valid_splits_kept() -> None:
    ds = tasks.compile_task_dataset(
        _dataset(splits={"test": ["refund-policy-lookup"], "train": ["escalation-path"]})
    )
    assert ds["splits"]["test"] == ["refund-policy-lookup"]


def test_dataset_propagates_task_errors() -> None:
    with pytest.raises(tasks.TaskDatasetError):
        tasks.compile_task_dataset(_dataset(tasks=[_task(objective=_objective(guards=False))]))


def test_dataset_version_deterministic() -> None:
    a = tasks.compile_task_dataset(_dataset())
    b = tasks.compile_task_dataset(_dataset())
    assert a["version"] == b["version"]
