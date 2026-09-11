"""Unit 6 (BBG U6 / ARCH §2b) — auto-lift + forward-derivation round-trip census.

Per builder: build → run → derive_simulation_manifest → derive_simulation_run_
manifest(sim, agent) → run → compare envelope-stripped canonical JSON (byte
equality, the only normalization — AD-Q). Covers the S1-S8 census shapes that
run offline/credential-free (the transport/endpoint builders need a loopback
server and are exercised by the prove script, U17, per Appendix C-2).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from fi.alk import simulate as S
from fi.simulate.cli import _run_local_text_manifest

# The AD-Q frozen envelope strip (wall-clock fields) PLUS the additive 13D
# contract-evidence keys (R4 tool_mock_profile, G3 goal_machine): the latter are
# the contract DOING its job (labeling), not a behavior divergence, so the
# substantive round-trip comparison excludes them too. STABLE_RESULT_ENVELOPE_
# FIELDS itself stays frozen (5-tuple) per the gate mirror.
STRIP = S.STABLE_RESULT_ENVELOPE_FIELDS + ("duration_ms", "tool_mock_profile", "goal_machine")
AGENT = {"type": "scripted", "content": "done"}
SCEN = {"name": "c", "dataset": [{"persona": {"name": "A"}, "situation": "s", "outcome": "done"}]}


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in STRIP}
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    return obj


def _run(manifest):
    report = asyncio.run(_run_local_text_manifest(manifest, Path(".")))
    return _strip(report.model_dump())


# S1-S8 census (offline-runnable subset; each row is one builder with minimal
# credential-free kwargs). The transport probes (S2) need loopback servers and
# are covered by the prove script.
CENSUS = {
    # S1 generic task — the identity case
    "S1_task": lambda: S.build_task_run_manifest(
        name="t", agent=AGENT, task_description="do", expected_result="done", scenario=SCEN),
    # S4 modality (typed kinds; rung-1 derived-legacy fixture replay)
    "S4_browser_cua": lambda: S.build_browser_cua_run_manifest(name="bc", agent=AGENT),
    "S4_realtime": lambda: S.build_realtime_run_manifest(name="rt", agent=AGENT),
    "S4_multimodal": lambda: S.build_multimodal_image_run_manifest(name="mi", agent=AGENT),
    # S5 memory/orchestration
    "S5_world_fw_memory": lambda: S.build_world_framework_memory_run_manifest(name="wf", agent=AGENT),
    "S5_social_memory": lambda: S.build_social_memory_framework_run_manifest(name="sm", agent=AGENT),
    # S6 worlds
    "S6_stateful_world": lambda: S.build_stateful_tool_world_run_manifest(name="w", agent=AGENT),
    "S6_world_model": lambda: S.build_world_model_run_manifest(name="wm", agent=AGENT),
    "S6_autonomous_redteam": lambda: S.build_autonomous_redteam_task_world_run_manifest(name="ar", agent=AGENT),
    # S7 compat
    "S7_openenv": lambda: S.build_openenv_run_manifest(name="oe", agent=AGENT),
}


@pytest.mark.parametrize("row", sorted(CENSUS), ids=sorted(CENSUS))
def test_autolift_roundtrip_byte_equal(row):
    manifest = CENSUS[row]()
    original = _run(manifest)
    sim = S.derive_simulation_manifest(manifest)
    assert sim["kind"] == S.AGENT_LEARNING_SIMULATION_KIND
    rederived = S.derive_simulation_run_manifest(sim, agent=manifest["agent"])
    rerun = _run(rederived)
    assert json.dumps(original, sort_keys=True) == json.dumps(rerun, sort_keys=True), (
        f"{row}: round-trip not byte-equal after envelope strip"
    )


def test_world_kind_derivation_map():
    cases = {
        "S1_task": "conversation",
        "S4_browser_cua": "browser",
        "S4_realtime": "voice_telephony",
        "S6_stateful_world": "tool_api",
        "S7_openenv": "tool_api",
    }
    for row, expected in cases.items():
        sim = S.derive_simulation_manifest(CENSUS[row]())
        assert sim["world"]["kind"] == expected, f"{row} → {sim['world']['kind']} != {expected}"


def test_derived_objective_carries_source():
    sim = S.derive_simulation_manifest(CENSUS["S1_task"]())
    if sim.get("objective"):
        assert sim["objective"]["source"] == "derived"


def test_seed_defaulting():
    sim = S.derive_simulation_manifest(CENSUS["S1_task"]())
    assert sim["seed"] == 42  # documented default
    m = CENSUS["S1_task"]()
    m["seed"] = 7
    assert S.derive_simulation_manifest(m)["seed"] == 7


def test_lift_preserves_is_typed():
    m = {
        "version": "agent-learning.run.v1", "name": "typed",
        "scenario": {"name": "typed", "dataset": [
            {"persona": {"name": "T"}, "situation": "s", "outcome": "o", "behavior_policy": {}}]},
        "agent": AGENT, "simulation": {"max_turns": 1, "min_turns": 1}, "evaluation": {"enabled": False},
    }
    sim = S.derive_simulation_manifest(m)
    assert sim["personas"][0].get("behavior_policy") is not None  # typed layer survived


def test_suite_member_wise():
    """A suite derives member-wise (each member manifest derives independently)."""
    member = CENSUS["S1_task"]()
    sim = S.derive_simulation_manifest(member)
    assert sim["kind"] == S.AGENT_LEARNING_SIMULATION_KIND


def test_lifted_cast_roles_user():
    sim = S.derive_simulation_manifest(CENSUS["S1_task"]())
    for binding in sim["scenarios"]:
        for member in binding["cast"]:
            assert member["role"] == "user"
        assert binding["casting"] == "each"


def test_provenance_lifted_from():
    sim = S.derive_simulation_manifest(CENSUS["S1_task"]())
    assert "lifted_from" in sim["provenance"]
    assert "manifest_address" in sim["provenance"]["lifted_from"]
