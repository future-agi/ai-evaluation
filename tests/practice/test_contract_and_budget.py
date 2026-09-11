"""Unit 8 (BBG U8) — practice canon constants + the single budget meter."""
from __future__ import annotations

import pytest

from fi.alk.practice import _contract
from fi.alk.practice._budget import BudgetExhausted, BudgetMeter


def test_constants_byte_table():
    assert _contract.PRACTICE_PHASES == ("assess", "diagnose", "drill", "update", "consolidate", "calibrate")
    assert _contract.SCAFFOLD_TYPES == ("world_simplification", "hint_tool", "worked_example", "relaxed_success")
    assert _contract.LADDER_STATES == ("episodic", "instruction", "skill")
    assert _contract.PRACTICE_REPLAY_INTERVALS == (1, 2, 4, 8, 16)
    assert _contract.ZPD_BAND == (0.2, 0.7)
    assert _contract.REVIEW_RATIO == 0.25
    assert _contract.BUDGET_PLAN == (0.25, 0.35, 0.25, 0.15)
    assert _contract.PRACTICE_STORE_ACTIVE_CAP == 64
    assert _contract.SCAFFOLD_FADE_DEFAULT == (1.0, 0.5, 0.0)
    assert len(_contract.PRACTICE_ARTIFACT_KINDS) == 8
    # imported, never redeclared
    assert _contract.DEFAULT_REPEATS == 8
    assert _contract.UNSTABLE_ICC_FLOOR == 0.5


def test_meter_conservation():
    m = BudgetMeter(100)
    m.charge("assess", 10)
    m.charge("drill", 20)
    led = m.ledger()
    assert led["consumed"] == 30
    assert sum(led["by_phase"].values()) == 30
    assert led["remaining"] == 70


def test_meter_exhaustion_stop():
    m = BudgetMeter(5)
    m.charge("assess", 5)
    with pytest.raises(BudgetExhausted):
        m.charge("drill", 1)


def test_slice_integerization():
    m = BudgetMeter(100)
    assert m.slice("drill", 0.35) == 35
    assert isinstance(m.slice("drill", 0.35), int)
    # bounded by remaining
    m.charge("assess", 90)
    assert m.slice("drill", 0.35) == 10


def test_meter_rejects_bad_total():
    with pytest.raises(ValueError):
        BudgetMeter(0)
    with pytest.raises(ValueError):
        BudgetMeter(True)  # bool is not a valid int total


def test_store_path_env_override(monkeypatch, tmp_path):
    target = tmp_path / "p" / "records.jsonl"
    monkeypatch.setenv("AGENT_LEARNING_PRACTICE_STORE_PATH", str(target))
    assert _contract.practice_store_path() == target
    # explicit arg wins
    other = tmp_path / "other.jsonl"
    assert _contract.practice_store_path(other) == other


def test_lazy_facade():
    from fi.alk import practice
    assert practice.PRACTICE_PHASES == _contract.PRACTICE_PHASES
    assert practice.BudgetMeter is BudgetMeter
