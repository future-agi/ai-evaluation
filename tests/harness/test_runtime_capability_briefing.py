import asyncio
import importlib

import pytest

from fi.alk.harness import runtime_repair
from fi.alk.harness.contract import AgentContract, Runtime
from fi.alk.harness.diagnostics import HarnessDiagnostic
from fi.alk.harness.job import HarnessStage
from fi.alk.harness.sources import ProviderSource, RepoSource


@pytest.mark.parametrize("socket_variable", [
    None, "ALK_COMPOSE_BROKER_SOCKET", "ALK_TRUSTED_COMPOSE_SOCKET",
])
@pytest.mark.parametrize("stage_name", ["understand", "repair"])
def test_authoring_receives_actual_execution_capabilities(
    tmp_path, monkeypatch, socket_variable, stage_name,
):
    monkeypatch.delenv("ALK_COMPOSE_BROKER_SOCKET", raising=False)
    monkeypatch.delenv("ALK_TRUSTED_COMPOSE_SOCKET", raising=False)
    if socket_variable:
        monkeypatch.setenv(socket_variable, "/private/control/runtime.sock")
    understand = importlib.import_module("fi.alk.harness.understand")
    prompts = []

    class CapturingStage:
        def __init__(self, spec, **kwargs):
            prompts.append(spec.system_prompt)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def say(self, *args, **kwargs):
            pass

    if stage_name == "understand":
        monkeypatch.setattr(understand, "Stage", CapturingStage)
        understand.open_stage(
            RepoSource("agent", tmp_path), out=tmp_path / "output", require_setup_contract=True,
        )
    else:
        monkeypatch.setattr(runtime_repair, "Stage", CapturingStage)
        diagnostic = HarnessDiagnostic.create(
            stage=HarnessStage.CONNECTING_AGENT,
            component="runtime",
            code="generated_runtime_plan_invalid",
            message="source startup needs correction",
        )
        with pytest.raises(RuntimeError, match="typed_runtime_plan_patch_not_submitted"):
            asyncio.run(runtime_repair.request_runtime_plan_patch(
                tmp_path, AgentContract(runtime=Runtime(language="python")), (diagnostic,),
            ))
    assert prompts
    for prompt in prompts:
        assert "Available execution capabilities" in prompt
        assert "supports Python or source-declared pinned Bun" in prompt
        assert "not arbitrary native Node.js startup" in prompt
        assert "Do not translate a Node.js project to Bun without source evidence" in prompt
        assert "/private/control/runtime.sock" not in prompt
        if socket_variable:
            assert "Managed source-owned Compose deployment is available for chat agents with an HTTP ingress" in prompt
            assert "runtime.execution=compose" in prompt
            assert "runtime.compose_file" in prompt and "runtime.service" in prompt
            assert "migration steps that the repository already assembles" in prompt
            assert "forwarding provider credentials alone is not configuration" in prompt
            assert "secret_aliases" in prompt
            assert "source-declared minimal" in prompt
            assert "nested or ambiguous deployment files require an explicit selection" in prompt
            assert "Explicit native execution does not auto-select Compose" in prompt
            assert "Source validation, deployment policy, and real chat readiness" in prompt
            assert "This capability does not add standalone Dockerfile execution" in prompt
        else:
            assert "Managed source-owned Compose deployment is not enabled" in prompt
            assert "do not assume container execution or select published Compose images" in prompt
            assert "runtime.execution=compose" not in prompt


@pytest.mark.parametrize("provider", [False, True])
def test_other_understanding_flows_keep_their_runtime_profile(tmp_path, monkeypatch, provider):
    understand = importlib.import_module("fi.alk.harness.understand")
    monkeypatch.setenv("ALK_COMPOSE_BROKER_SOCKET", "/private/control/runtime.sock")
    source = ProviderSource("agent", {}, tmp_path) if provider else RepoSource("agent", tmp_path)
    stage, _ = understand.open_stage(
        source, out=tmp_path / "output", require_setup_contract=provider,
    )
    assert "Available execution capabilities" not in stage._spec.system_prompt
    assert "not arbitrary native Node.js startup" not in stage._spec.system_prompt
