"""Unit 3 (BBG U3 / ARCH §2a) — the Simulation contract models."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from fi.simulate.simulation import contract as C
from fi.simulate.simulation.contract import (
    CastMember,
    ScenarioBinding,
    Simulation,
    ToolBinding,
    WorldSpec,
)
from fi.simulate.simulation.models import Persona


def _persona(name="A"):
    p = Persona(persona={"name": name}, situation="s", outcome="o", behavior_policy={})
    return p


def _sim(**over):
    p = _persona()
    base = dict(
        name="sim",
        personas=[p],
        scenarios=[ScenarioBinding(cast=[CastMember(persona=p.version, role="user")])],
        world=WorldSpec(kind="conversation"),
    )
    base.update(over)
    return Simulation(**base)


def test_construct_minimal():
    sim = _sim()
    assert sim.kind == C.SIMULATION_KIND
    assert sim.version and sim.version.startswith("sha256:")


def test_hash_determinism():
    sim1 = _sim()
    sim2 = _sim()
    assert sim1.content_hash() == sim2.content_hash()
    # twice on the same object
    assert sim1.content_hash() == sim1.content_hash()


def test_mock_level_flip_changes_hash():
    p = _persona()
    tb_a = ToolBinding(name="t", mock={"level": "static_fixture"})
    tb_b = ToolBinding(name="t", mock={"level": "recorded_replay",
                                       "source": "cap://x",
                                       "provenance": {"capture": "sha256:abc"},
                                       "recorded_replay": {"miss_policy": "fail"}})
    sim_a = Simulation(name="s", personas=[p],
                       scenarios=[ScenarioBinding(cast=[CastMember(persona=p.version)])],
                       world=WorldSpec(kind="tool_api", tools=[tb_a]))
    sim_b = Simulation(name="s", personas=[p],
                       scenarios=[ScenarioBinding(cast=[CastMember(persona=p.version)])],
                       world=WorldSpec(kind="tool_api", tools=[tb_b]))
    assert sim_a.content_hash() != sim_b.content_hash()


def test_duplicate_persona_rejected():
    p = _persona()
    with pytest.raises(ValidationError, match="duplicate persona"):
        Simulation(name="s", personas=[p, p],
                   scenarios=[ScenarioBinding(cast=[CastMember(persona=p.version)])],
                   world=WorldSpec(kind="conversation"))


def test_cast_ref_closure():
    p = _persona()
    with pytest.raises(ValidationError, match="does not resolve"):
        Simulation(name="s", personas=[p],
                   scenarios=[ScenarioBinding(cast=[CastMember(persona="sha256:nope")])],
                   world=WorldSpec(kind="conversation"))


def test_cast_role_unknown():
    with pytest.raises(ValidationError, match="cast_role_unknown"):
        CastMember(persona="sha256:x", role="banana")


def test_r2_litmus_dynamics_turn_holder_rejected():
    p = _persona()
    with pytest.raises(ValidationError, match="counterpart_misclassified"):
        Simulation(
            name="s", personas=[p],
            scenarios=[ScenarioBinding(cast=[CastMember(persona=p.version)])],
            world=WorldSpec(kind="conversation"),
            dynamics=[{"at": {"turn": 1}, "event": "counterpart_message",
                       "payload": {"responds_to": "user", "text": "hi"}}],
        )


def test_casting_together_structural_pass():
    p1 = _persona("A")
    p2 = _persona("B")
    sim = Simulation(
        name="s", personas=[p1, p2],
        scenarios=[ScenarioBinding(
            cast=[CastMember(persona=p1.version), CastMember(persona=p2.version)],
            casting="together")],
        world=WorldSpec(kind="conversation"),
    )
    assert sim.scenarios[0].casting == "together"


def test_tool_mock_level_undeclared():
    with pytest.raises(ValidationError, match="tool_mock_level_undeclared"):
        ToolBinding(name="t", mock={})


def test_tool_mock_replay_missing():
    with pytest.raises(ValidationError, match="tool_mock_replay_missing"):
        ToolBinding(name="t", mock={"level": "recorded_replay"})


def test_tool_mock_live_unkeyed():
    with pytest.raises(ValidationError, match="tool_mock_live_unkeyed"):
        ToolBinding(name="t", mock={"level": "live"})


def test_world_kind_unsupported():
    with pytest.raises(ValidationError, match="world_kind_unsupported"):
        WorldSpec(kind="quantum")


def test_clock_simulated_requires_step_s():
    with pytest.raises(ValidationError, match="step_s"):
        from fi.simulate.simulation.contract import ClockSpec
        ClockSpec(model="simulated")


def test_studio_lazy_import():
    from fi.alk.studio import Simulation as S
    assert S is Simulation


def test_canon_tuples_frozen():
    assert C.TOOL_MOCK_LEVELS == ("static_fixture", "recorded_replay", "emulated", "live")
    assert C.SIMULATION_CAST_ROLES == ("user", "opponent", "coworker", "counterpart")
    assert len(C.SIMULATION_WORLD_KINDS) == 6
