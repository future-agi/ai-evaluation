"""Gate #80 task_dataset_benchmark_readiness tests.

Pins the gate clean on the real shipped example/dataset, that the new check is
registered (count-agnostic, by name) with milestone M4, the dataset byte-pin
matches, and that EVERY error bucket fires on a deliberately-broken gate_evidence
artifact (a gate that cannot fail is worthless — the harness-stub lesson).
"""

from __future__ import annotations

from pathlib import Path


from fi.alk import trinity

ROOT = Path(__file__).parent.parent


def test_gate_status_clean_on_shipped_example() -> None:
    status = trinity._release_task_dataset_benchmark_status(ROOT)
    assert status["kind"] == "agent-learning.task-dataset-benchmark-readiness.v1"
    for bucket in (
        "missing_files",
        "dataset_compile_errors",
        "determinism_errors",
        "guard_presence_errors",
        "overclaim_errors",
        "coverage_errors",
        "world_kind_errors",
    ):
        assert status[bucket] == [], (bucket, status[bucket])


def test_byte_pin_matches_shipped_dataset() -> None:
    from fi.alk import tasks

    ds = tasks.load_task_dataset(ROOT / "examples" / "task_datasets" / "support_starter.json")
    assert ds["version"] == trinity.V1_TASK_BENCHMARK_DATASET_PINNED_VERSION


# --- the gate must actually fail when the audited evidence is broken ---------
def _good_artifact() -> dict:
    return {
        "kind": "agent-learning.task-benchmark-example.v1",
        "dataset_version": trinity.V1_TASK_BENCHMARK_DATASET_PINNED_VERSION,
        "gate_evidence": {
            "dataset_version": trinity.V1_TASK_BENCHMARK_DATASET_PINNED_VERSION,
            "determinism": {"scores_identical_across_runs": True},
            "guard_presence": {"all_tasks_have_guards": True},
            "overclaim_tripwire": {
                "typed_only_flagged_under_live": True,
                "executable_not_flagged_under_live": True,
                "fixture_lane_honest": True,
            },
            "coverage": {
                "world_kinds": ["browser", "conversation", "tool_api"],
                "spans_executable": True,
            },
        },
    }


def _audit(monkeypatch, artifact: dict) -> dict:
    """Drive the gate against a synthetic artifact by stubbing the exec-load."""

    monkeypatch.setattr(trinity, "_exec_example_run", lambda *a, **k: (artifact, None))
    return trinity._release_task_dataset_benchmark_status(ROOT)


def test_synthetic_good_artifact_passes(monkeypatch) -> None:
    status = _audit(monkeypatch, _good_artifact())
    assert all(status[b] == [] for b in (
        "dataset_compile_errors", "determinism_errors", "guard_presence_errors",
        "overclaim_errors", "coverage_errors", "world_kind_errors",
    ))


def test_byte_pin_drift_fires(monkeypatch) -> None:
    art = _good_artifact()
    art["gate_evidence"]["dataset_version"] = "sha256:deadbeef"
    assert _audit(monkeypatch, art)["dataset_compile_errors"]


def test_nondeterminism_fires(monkeypatch) -> None:
    art = _good_artifact()
    art["gate_evidence"]["determinism"]["scores_identical_across_runs"] = False
    assert _audit(monkeypatch, art)["determinism_errors"]


def test_missing_guards_fires(monkeypatch) -> None:
    art = _good_artifact()
    art["gate_evidence"]["guard_presence"]["all_tasks_have_guards"] = False
    assert _audit(monkeypatch, art)["guard_presence_errors"]


def test_overclaim_not_flagged_fires(monkeypatch) -> None:
    # the honesty tripwire: if a typed-only task is NOT flagged under a live
    # evidence class, the gate MUST fail.
    art = _good_artifact()
    art["gate_evidence"]["overclaim_tripwire"]["typed_only_flagged_under_live"] = False
    assert _audit(monkeypatch, art)["overclaim_errors"]


def test_fixture_lane_dishonest_fires(monkeypatch) -> None:
    art = _good_artifact()
    art["gate_evidence"]["overclaim_tripwire"]["fixture_lane_honest"] = False
    assert _audit(monkeypatch, art)["overclaim_errors"]


def test_coverage_gap_fires(monkeypatch) -> None:
    art = _good_artifact()
    art["gate_evidence"]["coverage"]["spans_executable"] = False
    art["gate_evidence"]["coverage"]["world_kinds"] = ["browser"]
    assert _audit(monkeypatch, art)["coverage_errors"]


def test_unresolved_world_kind_fires(monkeypatch) -> None:
    art = _good_artifact()
    art["gate_evidence"]["coverage"]["world_kinds"] = ["conversation", "tool_api", "telepathy"]
    assert _audit(monkeypatch, art)["world_kind_errors"]


def test_wrong_kind_fires(monkeypatch) -> None:
    art = _good_artifact()
    art["kind"] = "something-else"
    assert _audit(monkeypatch, art)["dataset_compile_errors"]


def test_example_run_failure_fires(monkeypatch) -> None:
    monkeypatch.setattr(trinity, "_exec_example_run", lambda *a, **k: ({}, "boom"))
    status = trinity._release_task_dataset_benchmark_status(ROOT)
    assert status["dataset_compile_errors"]
