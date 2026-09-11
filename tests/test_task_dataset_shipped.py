"""The shipped out-of-the-box task dataset + example runner.

Pins that the shipped ``support_starter`` dataset compiles (every task carries a
deterministic anchor + Goodhart guards), spans the executable worlds plus a
typed-only browser task, and that the example benchmark runs deterministically
and honestly on the credential-free fixture lane.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from fi.alk import tasks

DATASET_PATH = Path(__file__).parent.parent / "examples" / "task_datasets" / "support_starter.json"
EXAMPLE_PATH = Path(__file__).parent.parent / "examples" / "sdk_task_benchmark.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("sdk_task_benchmark_under_test", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shipped_dataset_exists() -> None:
    assert DATASET_PATH.exists(), DATASET_PATH


def test_shipped_dataset_compiles() -> None:
    ds = tasks.load_task_dataset(DATASET_PATH)
    assert ds["kind"] == tasks.AGENT_LEARNING_TASK_DATASET_KIND
    assert ds["coverage"]["count"] == 5
    assert ds["version"].startswith("sha256:")


def test_shipped_dataset_spans_executable_and_typed_only() -> None:
    ds = tasks.load_task_dataset(DATASET_PATH)
    kinds = set(tasks.task_world_kinds(ds))
    assert {"conversation", "tool_api"} <= kinds  # executable worlds present
    assert "browser" in kinds  # a typed-only world present too


def test_every_shipped_task_has_anchor_and_guards() -> None:
    ds = tasks.load_task_dataset(DATASET_PATH)
    for task in ds["tasks"]:
        # compile_task already enforced these; assert the evidence is on the row
        assert task["anchor_terms"], task["id"]
        guards = task["objective"]["guards"]
        assert guards["min_guard_count"] >= 1, task["id"]
        assert guards["sentinel_rows"] or guards["canary_evals"], task["id"]


def test_shipped_dataset_splits_resolve() -> None:
    ds = tasks.load_task_dataset(DATASET_PATH)
    ids = {t["id"] for t in ds["tasks"]}
    for split_ids in ds["splits"].values():
        assert set(split_ids) <= ids


@pytest.mark.integration
def test_example_runs_honestly_and_deterministically(tmp_path) -> None:
    module = _load_example()
    out1 = tmp_path / "run1.json"
    out2 = tmp_path / "run2.json"
    payload1 = module.run(out1)
    payload2 = module.run(out2)

    agg = payload1["aggregate"]
    assert agg["count"] == 5
    # honest: fixture lane, no live, no overclaim
    assert agg["honesty"]["any_live"] is False
    assert agg["honesty"]["any_overclaim"] is False
    # the browser task is stamped typed_only; the rest executable
    by_id = {t["task_id"]: t for t in payload1["per_task"]}
    assert by_id["browser-find-docs"]["execution_class"] == "typed_only"
    assert by_id["refund-policy-lookup"]["execution_class"] == "executable"
    # the scripted agent name-drops the anchors -> real, non-zero scores
    assert agg["mean_score"] > 0.0
    # deterministic: identical scores across two runs (fixture lane)
    scores1 = {t["task_id"]: t["score"] for t in payload1["per_task"]}
    scores2 = {t["task_id"]: t["score"] for t in payload2["per_task"]}
    assert scores1 == scores2
