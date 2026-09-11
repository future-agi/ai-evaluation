"""Unit 22 (BBG U22 / RU-7) — capstone harness MECHANICS (never asserts outcomes).

The harness must run offline-deterministically and enforce the equal-budget
headline rule. Running the capstone experiment + the paper is a separate task.
"""
from __future__ import annotations

import json
from pathlib import Path


from fi.alk import cli
from fi.alk.practice import _capstone

CAPSTONE_DIR = Path(__file__).resolve().parents[1].parent / "examples" / "practice_capstone"


def test_arms_are_real_backend_tokens():
    # RU-7: real backend tokens only; "greedy" = bandit; canon stays closed.
    assert _capstone.CAPSTONE_ARMS == ("practice_loop", "gepa", "tpe", "society", "bandit")
    assert "bandit_greedy" not in _capstone.CAPSTONE_ARMS
    assert "evolution_elo" not in _capstone.CAPSTONE_ARMS  # not an arm
    assert "regression_replay" not in _capstone.CAPSTONE_ARMS  # the deck machinery


def test_ablations_present():
    assert _capstone.CAPSTONE_ABLATIONS == (
        "a1_no_zpd", "a2_no_spacing", "a3_no_consolidation", "a4_no_calibration")


def test_equal_budget_headline():
    result = _capstone.run_ab(CAPSTONE_DIR)
    assert result["budget_match"] is True
    assert result["headline"] is not None
    # arms in fixed order; best_found printed per arm.
    assert [a["arm"] for a in result["arms"]] == list(_capstone.CAPSTONE_ARMS)
    for arm in result["arms"]:
        assert "best_found" in arm
        assert "retention_after_interference" in arm


def test_budget_mismatch_nulls_headline(tmp_path):
    config = json.loads((CAPSTONE_DIR / "capstone.json").read_text())
    config["arm_budgets"]["gepa"] = 999  # mismatch
    d = tmp_path / "cap"
    d.mkdir()
    (d / "capstone.json").write_text(json.dumps(config))
    result = _capstone.run_ab(d)
    assert result["budget_match"] is False
    assert result["headline"] is None
    assert any(f["type"] == "ab_budget_mismatch" for f in result["findings"])


def test_cli_ab_dispatch(capsys):
    rc = cli.main(["practice", "ab", str(CAPSTONE_DIR)])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["status"] == "ran"
    assert "ab_harness" in out


def test_harness_never_asserts_outcomes():
    """The harness reports best_found/retention as placeholders (the experiment
    is a later task) — it never fabricates a winner."""
    result = _capstone.run_ab(CAPSTONE_DIR)
    for arm in result["arms"]:
        assert arm["retention_after_interference"] is None
