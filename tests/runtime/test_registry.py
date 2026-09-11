from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from fi.simulate.registry import (
    AdapterNotFound,
    AdapterRegistry,
    endpoint_registry,
    environment_registry,
    register_environment,
)
from fi.simulate.runtime import (
    AgentEndpointSpec,
    EnvironmentSpec,
    RunStatus,
    SimulationSpec,
    SimulatorPolicySpec,
)
from fi.simulate.runtime.runner import SimulationRunner
from fi.simulate.simulation.models import (
    Persona,
    Scenario,
    TestCaseResult as _TestCaseResult,
    TestReport as _TestReport,
)


# --------------------------------------------------------------------------- #
# Registry mechanics
# --------------------------------------------------------------------------- #
def test_register_and_get_roundtrip() -> None:
    reg: AdapterRegistry = AdapterRegistry("thing")

    @reg.register("a")
    def factory_a() -> str:
        return "A"

    assert reg.get("a") is factory_a
    assert reg.create("a") == "A"
    assert reg.has("a")
    assert reg.names() == ["a"]


def test_duplicate_registration_raises_unless_override() -> None:
    reg: AdapterRegistry = AdapterRegistry("thing")
    reg.register("a", lambda: 1)

    with pytest.raises(ValueError, match="thing_adapter_already_registered"):
        reg.register("a", lambda: 2)

    reg.register("a", lambda: 3, override=True)
    assert reg.create("a") == 3


def test_reregistering_same_object_is_idempotent() -> None:
    reg: AdapterRegistry = AdapterRegistry("thing")

    def f() -> int:
        return 1

    reg.register("a", f)
    reg.register("a", f)  # same object, must not raise
    assert reg.create("a") == 1


def test_missing_name_raises_adapter_not_found() -> None:
    reg: AdapterRegistry = AdapterRegistry("thing")
    reg.register("known", lambda: 1)

    with pytest.raises(AdapterNotFound) as exc:
        reg.get("nope")
    assert exc.value.name == "nope"
    assert exc.value.available == ["known"]
    assert reg.get_or_none("nope") is None


def test_entry_point_discovery_is_lazy_and_cached(monkeypatch) -> None:
    """The "pip-install a plugin and it just works" path."""

    def _plugin_factory():
        return "PLUGIN"

    fake_ep = SimpleNamespace(name="external", load=lambda: _plugin_factory)

    calls = {"n": 0}

    def fake_entry_points(*, group: str):
        calls["n"] += 1
        assert group == "fi.simulate.things"
        return [fake_ep]

    monkeypatch.setattr("fi.simulate.registry.metadata.entry_points", fake_entry_points)

    reg: AdapterRegistry = AdapterRegistry("thing", "fi.simulate.things")
    # nothing registered in-process -> discovery kicks in on first lookup
    assert reg.create("external") == "PLUGIN"
    # cached: a second lookup does not re-scan entry points
    assert reg.has("external")
    assert calls["n"] == 1


def test_in_process_registration_wins_over_entry_points(monkeypatch) -> None:
    fake_ep = SimpleNamespace(name="dup", load=lambda: (lambda: "FROM_EP"))
    monkeypatch.setattr(
        "fi.simulate.registry.metadata.entry_points",
        lambda *, group: [fake_ep],
    )
    reg: AdapterRegistry = AdapterRegistry("thing", "fi.simulate.things")
    reg.register("dup", lambda: "FROM_CODE")
    assert reg.create("dup") == "FROM_CODE"


# --------------------------------------------------------------------------- #
# Builtins are wired
# --------------------------------------------------------------------------- #
def test_chat_environment_is_registered() -> None:
    assert environment_registry.has("chat")


@pytest.mark.parametrize(
    "name,expected",
    [
        ("callable", ["text", "tool_events", "transcript_events"]),
        ("http", ["text", "tool_events", "transcript_events"]),
        ("websocket", ["streaming", "text", "tool_events", "transcript_events"]),
        (
            "livekit",
            [
                "audio",
                "interruption",
                "recording",
                "streaming",
                "transcript_events",
                "web_rtc",
            ],
        ),
    ],
)
def test_endpoint_capability_manifests_match_legacy_table(name, expected) -> None:
    profile = endpoint_registry.get(name)
    assert sorted(profile.manifest.capabilities.supported()) == sorted(expected)


# --------------------------------------------------------------------------- #
# Plan §3 contract: the core assumes no chat/voice-specific fields
# --------------------------------------------------------------------------- #
def test_runner_is_world_agnostic_over_a_third_party_environment() -> None:
    """Register a dummy world under a name the core has never heard of and prove
    SimulationRunner drives it end-to-end. If the runner assumed chat/voice
    fields, an unknown adapter could not complete."""

    scenario = Scenario(
        name="probe",
        dataset=[Persona(persona={"name": "Probe"}, situation="s", outcome="o")],
    )

    @register_environment("contract_probe_world")
    class _ProbeEnvironment:
        manifest = SimpleNamespace(name="contract_probe_world")
        seen: dict = {}

        async def run(self, spec, *, target, **kwargs):
            # record that the core handed us the spec verbatim
            _ProbeEnvironment.seen["adapter"] = spec.environment.adapter
            _ProbeEnvironment.seen["target"] = target
            return _TestReport(
                results=[
                    _TestCaseResult(
                        persona=scenario.dataset[0],
                        transcript="Probe: hello\nAgent: done",
                        messages=[{"role": "assistant", "content": "done"}],
                        metadata={"engine": "probe", "scenario_name": scenario.name},
                    )
                ]
            )

    spec = SimulationSpec(
        run_id="run_contract_probe",
        environment=EnvironmentSpec(adapter="contract_probe_world", world_kind="probe"),
        target=AgentEndpointSpec(adapter="callable"),
        simulator=SimulatorPolicySpec(adapter="synthetic_user"),
        scenario=scenario,
    )

    report = asyncio.run(SimulationRunner().run(spec, target="SENTINEL_TARGET"))

    assert report.status == RunStatus.COMPLETED
    assert report.test_cases[0].result.transcript
    assert _ProbeEnvironment.seen == {
        "adapter": "contract_probe_world",
        "target": "SENTINEL_TARGET",
    }


def test_unknown_environment_yields_failed_report_not_crash() -> None:
    scenario = Scenario(
        name="x",
        dataset=[Persona(persona={"name": "N"}, situation="s", outcome="o")],
    )
    spec = SimulationSpec(
        run_id="run_unknown_env",
        environment=EnvironmentSpec(adapter="no_such_world", world_kind="x"),
        target=AgentEndpointSpec(adapter="callable"),
        simulator=SimulatorPolicySpec(adapter="synthetic_user"),
        scenario=scenario,
    )
    report = asyncio.run(SimulationRunner().run(spec, target=lambda *_: "hi"))
    assert report.status == RunStatus.FAILED
