"""Units 18 & 20 (BBG U18/U20) — the planted-failure tripwire tests.

These exercise the gate ASSERTIONS on doctored fixtures (not just presence) —
the two highest-risk correctness properties (D7 promotion veto + non-forgetting)
must actually flip their arrays on a tampered fixture.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path


from fi.alk import trinity

REPO = Path(__file__).resolve().parents[1].parent
SIM_FIX = REPO / "examples" / "simulation_contract_fixtures"
PRAC_FIX = REPO / "examples" / "practice_loop_fixture"


def _clone_repo(tmp_path: Path) -> Path:
    """Copy the committed fixtures into a tmp project root so doctoring is
    isolated from the real tree."""
    root = tmp_path / "proj"
    (root / "examples").mkdir(parents=True)
    shutil.copytree(SIM_FIX, root / "examples" / "simulation_contract_fixtures")
    shutil.copytree(PRAC_FIX, root / "examples" / "practice_loop_fixture")
    return root


# --- simulation gate tripwires ---------------------------------------------
def test_clean_sim_gate_passes(tmp_path):
    root = _clone_repo(tmp_path)
    s = trinity._release_simulation_contract_status(root)
    assert all(not v for k, v in s.items() if k.endswith("_errors"))


def test_drifted_hash_flips_canonicalization(tmp_path):
    root = _clone_repo(tmp_path)
    hashes_path = root / "examples" / "simulation_contract_fixtures" / "hashes.json"
    data = json.loads(hashes_path.read_text())
    data["_drifted_row"]["stored_hash"] = "sha256:DRIFTED_WRONG"
    hashes_path.write_text(json.dumps(data))
    s = trinity._release_simulation_contract_status(root)
    assert s["canonicalization_errors"], "drifted hash must flip canonicalization_errors"


def test_unguarded_objective_fixture_flips_objective_schema(tmp_path):
    root = _clone_repo(tmp_path)
    p = root / "examples" / "simulation_contract_fixtures" / "objective" / "declared_unguarded_input.json"
    # doctor it to actually carry guards (so the gate notices the fixture is wrong)
    p.write_text(json.dumps({"evals": [{"eval": "x"}], "source": "declared",
                             "guards": {"sentinel_rows": ["row_g"], "min_guard_count": 1}}))
    s = trinity._release_simulation_contract_status(root)
    assert s["objective_schema_errors"]


def test_doctored_roundtrip_digest_flips_roundtrip(tmp_path):
    root = _clone_repo(tmp_path)
    census_path = root / "examples" / "simulation_contract_fixtures" / "roundtrip" / "census.json"
    data = json.loads(census_path.read_text())
    first = next(iter(data))
    data[first]["rederived_digest"] = "sha256:TAMPERED"
    data[first]["equal"] = False
    census_path.write_text(json.dumps(data))
    s = trinity._release_simulation_contract_status(root)
    assert s["roundtrip_errors"]


def test_casting_together_is_passing_state(tmp_path):
    """A casting:together fixture without the multiparty gate flips NOTHING —
    the typed refusal IS the passing state."""
    root = _clone_repo(tmp_path)
    s = trinity._release_simulation_contract_status(root)
    # cast_dynamics fixture includes casting_together; gate stays green.
    assert not s["cast_role_errors"]


# --- practice gate tripwires (D7 + non-forgetting are highest-risk) ---------
def test_clean_practice_gate_passes(tmp_path):
    root = _clone_repo(tmp_path)
    s = trinity._release_practice_loop_status(root)
    assert all(not v for k, v in s.items() if k.endswith("_errors"))


def test_tampered_schedule_flips_schedule_errors(tmp_path):
    root = _clone_repo(tmp_path)
    p = root / "examples" / "practice_loop_fixture" / "schedule_histories" / "expected.json"
    data = json.loads(p.read_text())
    data["cases"][0]["observed"] = 999  # corrupt a transition outcome
    p.write_text(json.dumps(data))
    s = trinity._release_practice_loop_status(root)
    assert s["schedule_errors"]


def test_tampered_detection_flag_flips_schedule_errors(tmp_path):
    root = _clone_repo(tmp_path)
    p = root / "examples" / "practice_loop_fixture" / "schedule_histories" / "expected.json"
    data = json.loads(p.read_text())
    data["tampered_detected"] = False
    p.write_text(json.dumps(data))
    s = trinity._release_practice_loop_status(root)
    assert s["schedule_errors"]


def test_doctored_sweep_flips_promotion_veto(tmp_path):
    """D7 HIGHEST-RISK: omit one deck row at the zero-due promotion ⇒ the veto
    array MUST flip (the sweep no longer replays the full union)."""
    root = _clone_repo(tmp_path)
    p = root / "examples" / "practice_loop_fixture" / "promotion_zero_due" / "sweep.json"
    data = json.loads(p.read_text())
    data["rows_replayed"] = data["rows_replayed"][:-1]  # drop one row
    data["all_rows_replayed"] = False
    p.write_text(json.dumps(data))
    s = trinity._release_practice_loop_status(root)
    assert s["promotion_veto_errors"], "a schedule-filtered promotion MUST flip promotion_veto_errors"


def test_schedule_filtered_promotion_flips_veto(tmp_path):
    root = _clone_repo(tmp_path)
    p = root / "examples" / "practice_loop_fixture" / "promotion_zero_due" / "sweep.json"
    data = json.loads(p.read_text())
    data["schedule_filtered"] = True
    p.write_text(json.dumps(data))
    s = trinity._release_practice_loop_status(root)
    assert s["promotion_veto_errors"]


def test_interference_beyond_bound_flips_interference(tmp_path):
    """Non-forgetting HIGHEST-RISK: a regression detected outside the latency
    bound MUST flip interference_errors."""
    root = _clone_repo(tmp_path)
    p = root / "examples" / "practice_loop_fixture" / "interference" / "non_forgetting.json"
    data = json.loads(p.read_text())
    data["detected_within_bound"] = False
    p.write_text(json.dumps(data))
    s = trinity._release_practice_loop_status(root)
    assert s["interference_errors"]


def test_frozen_row_not_closed_flips_interference(tmp_path):
    root = _clone_repo(tmp_path)
    p = root / "examples" / "practice_loop_fixture" / "interference" / "non_forgetting.json"
    data = json.loads(p.read_text())
    data["all_frozen_rows_closed_every_promotion"] = False
    p.write_text(json.dumps(data))
    s = trinity._release_practice_loop_status(root)
    assert s["interference_errors"]


def test_no_budget_not_rejected_flips_budget(tmp_path):
    root = _clone_repo(tmp_path)
    p = root / "examples" / "practice_loop_fixture" / "budget" / "conservation.json"
    data = json.loads(p.read_text())
    data["no_budget_rejected_at_build"] = False
    p.write_text(json.dumps(data))
    s = trinity._release_practice_loop_status(root)
    assert s["budget_errors"]


def test_broken_conservation_flips_budget(tmp_path):
    root = _clone_repo(tmp_path)
    p = root / "examples" / "practice_loop_fixture" / "budget" / "conservation.json"
    data = json.loads(p.read_text())
    led = data.get("ledger") or {}
    led["consumed"] = led.get("total", 100) + 5  # consumed > total
    led.setdefault("by_phase", {})["assess"] = led["consumed"]
    data["ledger"] = led
    p.write_text(json.dumps(data))
    s = trinity._release_practice_loop_status(root)
    assert s["budget_errors"]
