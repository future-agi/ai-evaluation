"""Unit 7 (BBG U7 / ARCH §2a) — contract-native rung-1 execution + refusals."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from fi.alk import simulate as S
from fi.simulate.cli import _run_local_text_manifest, load_manifest
from fi.simulate.manifest import ManifestError

AGENT = {"type": "scripted", "content": "done"}
STRIP = S.STABLE_RESULT_ENVELOPE_FIELDS + ("duration_ms", "tool_mock_profile", "goal_machine")


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in STRIP}
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    return obj


def _run(manifest):
    return asyncio.run(_run_local_text_manifest(manifest, Path(".")))


def _build_sim(world_kind, tools=None, **over):
    sim = S.build_simulation_manifest(
        name="cn",
        personas=[_persona_dump("A")],
        scenarios=[{"cast": [{"persona": _persona_hash("A"), "role": "user"}], "casting": "each"}],
        world={"kind": world_kind, "tools": tools or []},
        **over,
    )
    return sim


def _persona_dump(name):
    from fi.simulate.simulation.models import Persona
    return Persona(persona={"name": name}, situation="s", outcome="done", behavior_policy={}).model_dump(exclude_none=True)


def _persona_hash(name):
    from fi.simulate.simulation.models import Persona
    return Persona(persona={"name": name}, situation="s", outcome="done", behavior_policy={}).version


def test_conversation_contract_native_runs():
    sim = _build_sim("conversation")
    run = S.derive_simulation_run_manifest(sim, agent=AGENT)
    report = _run(run)
    assert report.results  # ran end-to-end


def test_tool_api_contract_native_runs():
    tb = [{"name": "lookup", "mock": {"level": "static_fixture"}}]
    sim = _build_sim("tool_api", tools=tb)
    run = S.derive_simulation_run_manifest(sim, agent=AGENT)
    report = _run(run)
    assert report.results
    # mock profile recorded
    meta = report.results[0].metadata
    assert meta.get("tool_mock_profile", {}).get("lookup", {}).get("level") == "static_fixture"


def test_legacy_manifest_without_block_byte_identical():
    path = Path("examples/run_manifest.json")
    report = _run(load_manifest(path))
    observed = _strip(report.model_dump())
    baseline = json.loads(
        (Path(__file__).parent / "fixtures" / "g4_baseline_result.json").read_text()
    )
    assert json.dumps(observed, sort_keys=True) == json.dumps(baseline, sort_keys=True)


def test_browser_derived_legacy_runs():
    """browser runs derived-legacy rung-1 (no contract-native feature requested)."""
    m = S.build_browser_cua_run_manifest(name="bc", agent=AGENT)
    sim = S.derive_simulation_manifest(m)
    assert sim["world"]["kind"] == "browser"
    run = S.derive_simulation_run_manifest(sim, agent=AGENT)
    report = _run(run)  # derived-legacy execution allowed
    assert report.results


def test_computer_use_refuses_at_validation():
    sim = _build_sim("computer_use")
    run = S.derive_simulation_run_manifest(sim, agent=AGENT)
    with pytest.raises(ManifestError, match="world_kind_refusal"):
        _run(run)


def test_code_exec_refuses():
    sim = _build_sim("code_exec")
    run = S.derive_simulation_run_manifest(sim, agent=AGENT)
    with pytest.raises(ManifestError, match="world_kind_refusal"):
        _run(run)


def test_live_unkeyed_preflight_refusal():
    tb = [{"name": "pay", "mock": {"level": "live"}, "required_env": ["PAY_API_KEY"]}]
    sim = _build_sim("tool_api", tools=tb)
    run = S.derive_simulation_run_manifest(sim, agent=AGENT)
    with pytest.raises(ManifestError, match="tool_mock_live_unkeyed"):
        _run(run)


def test_episodes_gt_1_refusal():
    sim = _build_sim("conversation", episodes={"count": 2, "persistence": "fresh"})
    run = S.derive_simulation_run_manifest(sim, agent=AGENT)
    with pytest.raises(ManifestError, match="world_kind_refusal"):
        _run(run)


def test_dynamics_refusal():
    sim = _build_sim("conversation", dynamics=[{"at": {"turn": 1}, "event": "env_state_patch", "payload": {"x": 1}}])
    run = S.derive_simulation_run_manifest(sim, agent=AGENT)
    with pytest.raises(ManifestError, match="world_kind_refusal"):
        _run(run)
