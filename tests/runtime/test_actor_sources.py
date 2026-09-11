from __future__ import annotations

import asyncio

import pytest

from fi.simulate.agent.wrapper import AgentWrapper
from fi.simulate.endpoints import actor_sources
from fi.simulate.endpoints.actor_sources import ActorSourceError
from fi.simulate.endpoints.profiles import get_profile
from fi.simulate.hosted.targets import resolve_chat_target
from fi.simulate.runtime import (
    AgentEndpointSpec,
    EnvironmentSpec,
    RunStatus,
    SimulationSpec,
    SimulatorPolicySpec,
)
from fi.simulate.runtime.runner import SimulationRunner
from fi.simulate.simulation.models import Persona, Scenario


async def _echo(_input) -> str:
    return "The status is complete."


class _EchoObject:
    async def call(self, _input) -> str:
        return "The status is complete."


class _EchoFactory:
    def __init__(self, reply: str = "The status is complete.") -> None:
        self._reply = reply

    async def call(self, _input) -> str:
        return self._reply


# --------------------------------------------------------------------------- #
# per-kind resolution
# --------------------------------------------------------------------------- #
def test_python_callable_returns_raw_callable(monkeypatch) -> None:
    monkeypatch.setattr(actor_sources, "_load_attr", lambda ref: _echo)
    resolved = get_profile("python_callable").resolve_target({"target": "m:f"})
    assert resolved is _echo


def test_import_object_is_wrapped(monkeypatch) -> None:
    monkeypatch.setattr(actor_sources, "_load_attr", lambda ref: _EchoObject())
    resolved = get_profile("import_object").resolve_target({"target": "m:Obj"})
    assert isinstance(resolved, AgentWrapper)


def test_factory_instantiates_then_wraps(monkeypatch) -> None:
    monkeypatch.setattr(actor_sources, "_load_attr", lambda ref: _EchoFactory)
    resolved = get_profile("factory").resolve_target(
        {"target": "m:C", "args": ["hi there"]}
    )
    assert isinstance(resolved, AgentWrapper)


def test_framework_loads_and_wraps(monkeypatch) -> None:
    monkeypatch.setattr(actor_sources, "_load_attr", lambda ref: _EchoObject())
    resolved = get_profile("framework").resolve_target(
        {"target": "m:agent", "framework": "langgraph", "method": "call"}
    )
    assert isinstance(resolved, AgentWrapper)


def test_malformed_target_raises() -> None:
    with pytest.raises(ActorSourceError):
        get_profile("import_object").resolve_target({"target": "no_colon"})


def test_system_prompt_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ActorSourceError, match="OPENAI_API_KEY"):
        get_profile("system_prompt").resolve_target({"system_prompt": "be nice"})


def test_voice_profile_is_not_turn_based() -> None:
    assert get_profile("vapi_websocket").is_turn_based_target is False
    with pytest.raises(ValueError, match="not_turn_based"):
        get_profile("vapi_websocket").resolve_target({})


# --------------------------------------------------------------------------- #
# hosted security posture: caller-code kinds are denied in-process (deny-by-default)
# --------------------------------------------------------------------------- #
def _chat_spec(adapter: str, config: dict, secret_refs: dict | None = None) -> SimulationSpec:
    target = AgentEndpointSpec(adapter=adapter, config=config)
    if secret_refs:
        target = AgentEndpointSpec(adapter=adapter, config=config, secret_refs=secret_refs)
    return SimulationSpec(
        run_id="run_actor",
        environment=EnvironmentSpec(
            adapter="chat", world_kind="conversation",
            config={"max_turns": 1, "min_turns": 1},
        ),
        target=target,
        simulator=SimulatorPolicySpec(adapter="synthetic_user"),
        scenario=Scenario(
            name="actor",
            dataset=[Persona(persona={"name": "M"}, situation="s", outcome="status complete")],
        ),
    )


def test_profile_runs_caller_code_flags() -> None:
    for kind in ("callable", "python_callable", "import_object", "factory", "framework"):
        assert get_profile(kind).runs_caller_code is True
    for kind in ("http", "system_prompt"):
        assert get_profile(kind).runs_caller_code is False


@pytest.mark.parametrize("kind", ["python_callable", "import_object", "factory", "framework"])
def test_code_actor_denied_in_hosted(monkeypatch, kind) -> None:
    monkeypatch.delenv("ALK_UNSAFE_INPROCESS_CODE_ACTORS", raising=False)
    spec = _chat_spec(kind, {"target": "builtins:exec"})
    with pytest.raises(ActorSourceError, match="code_actor_denied_in_hosted"):
        resolve_chat_target(spec)


def test_scary_escape_permits_trusted_default(monkeypatch) -> None:
    monkeypatch.setenv("ALK_UNSAFE_INPROCESS_CODE_ACTORS", "true")
    monkeypatch.setattr(actor_sources, "_load_attr", lambda ref: _echo)
    spec = _chat_spec("python_callable", {"target": "trusted:default"})
    assert resolve_chat_target(spec) is _echo


def test_http_target_allowed_in_hosted() -> None:
    spec = _chat_spec("http", {"url": "https://agent.example/chat"})
    target = resolve_chat_target(spec)
    assert isinstance(target, AgentWrapper)


def test_hosted_env_read_restricted_to_provisioned_secrets(monkeypatch) -> None:
    monkeypatch.setenv("LIVEKIT_API_SECRET", "another-tenants-secret")
    # api_key_env names an env var the job never provisioned -> refused
    with pytest.raises(ActorSourceError, match="env_not_provisioned"):
        actor_sources.resolve_system_prompt(
            {"system_prompt": "hi", "api_key_env": "LIVEKIT_API_SECRET"},
            {},
            hosted=True,
        )


# --------------------------------------------------------------------------- #
# local (developer) resolution — in-process is expected; run E2E through the runner
# --------------------------------------------------------------------------- #
def test_local_python_callable_end_to_end(monkeypatch) -> None:
    monkeypatch.setattr(actor_sources, "_load_attr", lambda ref: _echo)
    spec = _chat_spec("python_callable", {"target": "m:f"})
    target = get_profile("python_callable").resolve_target(
        spec.target.config, hosted=False
    )
    report = asyncio.run(SimulationRunner().run(spec, target=target))
    assert report.status == RunStatus.COMPLETED
    assert report.test_cases[0].result.transcript


def test_local_import_object_end_to_end(monkeypatch) -> None:
    monkeypatch.setattr(actor_sources, "_load_attr", lambda ref: _EchoObject())
    spec = _chat_spec("import_object", {"target": "m:Obj"})
    target = get_profile("import_object").resolve_target(
        spec.target.config, hosted=False
    )
    report = asyncio.run(SimulationRunner().run(spec, target=target))
    assert report.status == RunStatus.COMPLETED
