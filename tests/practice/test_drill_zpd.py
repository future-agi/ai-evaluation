"""Unit 11 (BBG U11) — deficit-targeted drills at measured ZPD."""
from __future__ import annotations

import pytest

from fi.alk.practice import _drill


SIM = {"name": "drill", "world": {"kind": "conversation", "spec": {}}, "scenarios": [{"cast": []}],
       "verification": {"threshold": 0.7}}
DEFICIT = {"cell": {"intent": "a", "persona": "sha256:p"}, "harness_layer": "execution"}


def test_scaffold_transforms_pure_and_hash_distinct():
    a = _drill.apply_scaffold(SIM, "hint_tool", {})
    b = _drill.apply_scaffold(SIM, "relaxed_success", {"threshold": 0.3})
    assert _drill._hash(a) != _drill._hash(SIM)  # scaffolded ⇒ different simulation
    assert _drill._hash(a) != _drill._hash(b)
    assert SIM["world"].get("tools") is None  # original untouched (pure)


def test_all_four_scaffolds():
    for stype in ("world_simplification", "hint_tool", "worked_example", "relaxed_success"):
        out = _drill.apply_scaffold(SIM, stype, {})
        assert _drill._hash(out) != _drill._hash(SIM)


def test_fade_must_end_0():
    with pytest.raises(ValueError, match="end at 0.0"):
        _drill.drill(DEFICIT, SIM, seed=1, round_no=0,
                     repeat_scorer=lambda s, seed: 1.0, fade_intensities=(1.0, 0.5))


def test_zpd_in_band():
    # unscaffolded rate 0.5 ∈ band (0.2, 0.7) and ICC stable
    rec = _drill.drill(DEFICIT, SIM, seed=1, round_no=0,
                       repeat_scorer=lambda s, seed: 1.0 if seed % 2 == 0 else 0.0, k=8)
    assert rec["zpd_measurement"]["verdict"] in ("in_band", "unstable")


def test_zpd_above_band():
    rec = _drill.drill(DEFICIT, SIM, seed=1, round_no=0,
                       repeat_scorer=lambda s, seed: 1.0, k=8)
    # all-pass: rate 1.0 > high; zero-variance ICC := 1.0 ⇒ above_band
    assert rec["zpd_measurement"]["verdict"] == "above_band"


def test_zpd_below_band_no_scaffold_help():
    rec = _drill.drill(DEFICIT, SIM, seed=1, round_no=0,
                       repeat_scorer=lambda s, seed: 0.0, k=8)
    # all-fail unscaffolded, no scaffold passes ⇒ below_band
    assert rec["zpd_measurement"]["verdict"] == "below_band"


def test_zpd_vygotsky_form():
    # fails unscaffolded, passes under a scaffold
    def scorer(sim, seed):
        return 1.0 if sim.get("metadata", {}).get("_scaffold_worked_example") else 0.0
    rec = _drill.drill(DEFICIT, SIM, seed=1, round_no=0, repeat_scorer=scorer, k=8,
                       scaffolds=[{"type": "worked_example", "params": {}}])
    assert rec["zpd_measurement"]["verdict"] == "vygotsky_form"
    assert rec["zpd_measurement"]["scaffolded_pass_rates"]["worked_example"] == 1.0


def test_admission_refusal_runs_nothing():
    rec = _drill.drill(DEFICIT, SIM, seed=1, round_no=0,
                       repeat_scorer=lambda s, seed: pytest.fail("should not run"),
                       admission={"admissible": False, "reason": "solvability"})
    assert rec["unscaffolded_exit"] is False
    assert rec["zpd_measurement"]["k"] == 0


def test_repeat_seeds_deterministic():
    s1 = [_drill.child_seed(42, "drill", "ck", i) for i in range(4)]
    s2 = [_drill.child_seed(42, "drill", "ck", i) for i in range(4)]
    assert s1 == s2
    assert len(set(s1)) == 4  # distinct per index
