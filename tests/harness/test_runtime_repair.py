from __future__ import annotations

import json

import pytest

from fi.alk.harness.contract import AgentContract, Runtime, RuntimeInterface
from fi.alk.harness.runtime_repair import (
    RuntimePlanPatch,
    _contract_hash,
    apply_runtime_plan_patch,
    validate_runtime_plan_patch,
)


def _patch(contract: AgentContract, **changes) -> RuntimePlanPatch:
    runtime = Runtime(
        language="python",
        workdir="service",
        command=["python", "app.py"],
        interface=RuntimeInterface(kind="http", port=8080, path="/chat"),
        **changes,
    )
    return RuntimePlanPatch(
        base_contract_hash=_contract_hash(contract),
        runtime=runtime,
        evidence_paths=("service/pyproject.toml", "service/app.py"),
        diagnostic_codes=("generated_runtime_plan_invalid",),
        summary="Run the HTTP service declared by the submitted component.",
    )


def test_runtime_patch_changes_only_runtime_contract_metadata(tmp_path) -> None:
    source = tmp_path / "source"
    service = source / "service"
    service.mkdir(parents=True)
    (service / "pyproject.toml").write_text("[project]\nname='agent'\nversion='1'\n")
    (service / "app.py").write_text("# entrypoint\n")
    contract = AgentContract(
        agent="support-agent",
        hard_constraints=["Never disclose secrets"],
        runtime=Runtime(language="python"),
    )
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(contract.model_dump_json(indent=2) + "\n")

    repaired = apply_runtime_plan_patch(
        source,
        contract_path,
        _patch(contract),
        allowed_diagnostic_codes={"generated_runtime_plan_invalid"},
    )

    assert repaired.agent == contract.agent
    assert repaired.hard_constraints == contract.hard_constraints
    assert repaired.runtime.workdir == "service"
    assert repaired.runtime.command == ["python", "app.py"]
    assert json.loads(contract_path.read_text())["runtime"]["workdir"] == "service"
    assert (service / "app.py").read_text() == "# entrypoint\n"


def test_runtime_patch_rejects_missing_or_outside_evidence(tmp_path) -> None:
    source = tmp_path / "source"
    (source / "service").mkdir(parents=True)
    contract = AgentContract(agent="agent", runtime=Runtime(language="python"))
    patch = RuntimePlanPatch(
        base_contract_hash=_contract_hash(contract),
        runtime=Runtime(language="python", workdir="service"),
        evidence_paths=("service/missing.py",),
        diagnostic_codes=("generated_runtime_plan_invalid",),
        summary="Use service.",
    )

    with pytest.raises(ValueError, match="evidence_missing"):
        validate_runtime_plan_patch(
            source,
            contract,
            patch,
            allowed_diagnostic_codes={"generated_runtime_plan_invalid"},
        )


def test_runtime_patch_rejects_stale_contract(tmp_path) -> None:
    source = tmp_path / "source"
    service = source / "service"
    service.mkdir(parents=True)
    (service / "pyproject.toml").write_text("[project]\n")
    (service / "app.py").write_text("# entrypoint\n")
    contract = AgentContract(agent="agent", runtime=Runtime(language="python"))
    patch = _patch(contract).model_copy(
        update={"base_contract_hash": "sha256:" + "0" * 64}
    )

    with pytest.raises(ValueError, match="base_contract_hash_mismatch"):
        validate_runtime_plan_patch(
            source,
            contract,
            patch,
            allowed_diagnostic_codes={"generated_runtime_plan_invalid"},
        )


def test_runtime_patch_rejects_model_authored_shell_commands(tmp_path) -> None:
    source = tmp_path / "source"
    service = source / "service"
    service.mkdir(parents=True)
    (service / "pyproject.toml").write_text("[project]\n")
    (service / "app.py").write_text("# entrypoint\n")
    contract = AgentContract(agent="agent", runtime=Runtime(language="python"))
    patch = RuntimePlanPatch(
        base_contract_hash=_contract_hash(contract),
        runtime=Runtime(
            language="python",
            workdir="service",
            command=["bash", "-c", "python app.py"],
        ),
        evidence_paths=("service/pyproject.toml", "service/app.py"),
        diagnostic_codes=("generated_runtime_plan_invalid",),
        summary="Shell launch.",
    )

    with pytest.raises(ValueError, match="shell_command_forbidden"):
        validate_runtime_plan_patch(
            source,
            contract,
            patch,
            allowed_diagnostic_codes={"generated_runtime_plan_invalid"},
        )


def test_compose_repair_path_is_relative_to_repository_not_workdir(tmp_path):
    (tmp_path / "service").mkdir()
    (tmp_path / "deployment").mkdir()
    (tmp_path / "service/app.py").write_text("print('source')")
    (tmp_path / "service/pyproject.toml").write_text(
        '[project]\nname="fixture"\nversion="1"\n'
    )
    (tmp_path / "deployment/custom.yaml").write_text(
        "services:\n  dialogue:\n    build: ../service\n"
    )
    contract = AgentContract(agent="fixture")
    patch = _patch(
        contract,
        execution="compose",
        compose_file="deployment/custom.yaml",
        service="dialogue",
    )
    repaired = validate_runtime_plan_patch(
        tmp_path,
        contract,
        patch,
        allowed_diagnostic_codes={"generated_runtime_plan_invalid"},
    )
    assert repaired.runtime.compose_file == "deployment/custom.yaml"
    assert repaired.runtime.service == "dialogue"


@pytest.fixture
def dependency_patch(tmp_path):
    (tmp_path / "app.py").write_text("print('source')")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="fixture"\nversion="1"\n'
        '[project.optional-dependencies]\nexisting=[]\ncustom_transport=["some-library"]\n'
    )
    contract = AgentContract(
        agent="fixture",
        runtime=Runtime(
            language="python",
            execution="native",
            command=["python", "app.py"],
            extras=["existing"],
            interface=RuntimeInterface(kind="http", port=8080, path="/chat"),
        ),
    )
    patch = RuntimePlanPatch(
        base_contract_hash=_contract_hash(contract),
        runtime=contract.runtime.model_copy(
            update={"extras": ["existing", "custom_transport"]}
        ),
        evidence_paths=("pyproject.toml",),
        diagnostic_codes=("chat_ingress_failed",),
        summary="The relevant branch imports a dependency in custom_transport.",
    )
    return contract, patch


def test_readiness_patch_accepts_only_source_declared_additive_extras(
    tmp_path, dependency_patch
):
    contract, patch = dependency_patch
    repaired = validate_runtime_plan_patch(
        tmp_path, contract, patch, allowed_diagnostic_codes={"chat_ingress_failed"}
    )
    assert repaired.runtime.extras == ["existing", "custom_transport"]
    assert repaired.runtime.model_dump(
        exclude={"extras"}
    ) == contract.runtime.model_dump(exclude={"extras"})


@pytest.mark.parametrize(
    "extras",
    [["existing"], ["custom_transport"], ["existing", "unknown"], ["EXISTING"]],
)
def test_readiness_patch_rejects_noop_removal_and_undeclared_group(
    tmp_path, dependency_patch, extras
):
    contract, patch = dependency_patch
    patch = patch.model_copy(
        update={"runtime": patch.runtime.model_copy(update={"extras": extras})}
    )
    with pytest.raises(ValueError):
        validate_runtime_plan_patch(
            tmp_path, contract, patch, allowed_diagnostic_codes={"chat_ingress_failed"}
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"command": ["python", "other.py"]},
        {"workdir": "other"},
        {"install": "pip install arbitrary"},
        {"execution": "compose"},
        {"interface": RuntimeInterface(kind="http", port=8080, path="/error")},
        {"language": "javascript"},
        {"required_environment": {"agent": ["NEW"]}},
        {"context_excludes": ["tests"]},
        {"platform": "linux/amd64"},
    ],
)
def test_readiness_patch_locks_runtime_fields(tmp_path, dependency_patch, changes):
    contract, patch = dependency_patch
    patch = patch.model_copy(
        update={"runtime": patch.runtime.model_copy(update=changes)}
    )
    with pytest.raises(ValueError, match="only_extras_allowed"):
        validate_runtime_plan_patch(
            tmp_path, contract, patch, allowed_diagnostic_codes={"chat_ingress_failed"}
        )


@pytest.mark.parametrize("layout", ["ambiguous", "excluded", "escaped"])
def test_readiness_patch_rejects_unselectable_workspace_groups(
    tmp_path, dependency_patch, layout
):
    contract, patch = dependency_patch
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="fixture"\nversion="1"\n[project.optional-dependencies]\nexisting=[]\n'
        '[tool.uv.workspace]\nmembers=["modules/*"]\n'
        + ('exclude=["modules/*"]\n' if layout == "excluded" else "")
    )
    for name in ["a", "b"] if layout == "ambiguous" else ["a"]:
        member = tmp_path / "modules" / name
        member.mkdir(parents=True)
        (member / "pyproject.toml").write_text(
            '[project]\nname="member"\nversion="1"\n[project.optional-dependencies]\ncustom_transport=[]\n'
        )
    if layout == "escaped":
        outside = tmp_path.parent / "outside-member"
        outside.mkdir()
        (tmp_path / "modules/link").symlink_to(outside)
    with pytest.raises(ValueError):
        validate_runtime_plan_patch(
            tmp_path, contract, patch, allowed_diagnostic_codes={"chat_ingress_failed"}
        )


def test_dependency_investigation_may_decline_without_nudge(
    tmp_path, dependency_patch, monkeypatch
):
    import asyncio
    from fi.alk.harness import runtime_repair
    from fi.alk.harness.diagnostics import HarnessDiagnostic
    from fi.alk.harness.job import HarnessStage

    contract, _patch_value = dependency_patch
    calls = []

    class DecliningStage:
        def __init__(self, spec, **kwargs):
            assert spec.max_turns == 12

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def say(self, briefing, *, on_event=None):
            calls.append(briefing)

    monkeypatch.setattr(runtime_repair, "Stage", DecliningStage)
    diagnostic = HarnessDiagnostic.create(
        stage=HarnessStage.CONNECTING_AGENT,
        component="chat_ingress",
        code="chat_ingress_failed",
        message="unproven",
    )
    assert (
        asyncio.run(
            runtime_repair.request_runtime_plan_patch(tmp_path, contract, (diagnostic,))
        )
        is None
    )
    assert len(calls) == 1


def test_dependency_investigation_shows_sibling_optional_extras_without_manifest_values(
    tmp_path, monkeypatch
):
    import asyncio
    from fi.alk.harness import runtime_repair
    from fi.alk.harness.diagnostics import HarnessDiagnostic
    from fi.alk.harness.job import HarnessStage

    (tmp_path / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers=["agent-server", "agent-sdk"]\n'
    )
    server = tmp_path / "agent-server"
    server.mkdir()
    (server / "pyproject.toml").write_text(
        '[project]\nname="agent-server"\nversion="1"\ndependencies=["agent-sdk"]\n'
    )
    sdk = tmp_path / "agent-sdk"
    sdk.mkdir()
    (sdk / "pyproject.toml").write_text(
        '[project]\nname="agent-sdk"\nversion="1"\n'
        "[project.optional-dependencies]\n"
        'vertex=["google-cloud-aiplatform @ https://example.invalid/pkg?token=secret-value"]\n'
    )
    contract = AgentContract(
        agent="test",
        runtime=Runtime(
            language="python",
            execution="native",
            workdir="agent-server",
            command=["python", "-m", "agent_server"],
        ),
    )
    briefings = []

    class CapturingStage:
        def __init__(self, spec, **kwargs):
            schema = spec.servers["runtime_repair"].tools[0].input_schema
            assert "runtime_dependencies" not in schema["properties"]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def say(self, briefing, *, on_event=None):
            briefings.append(briefing)

    monkeypatch.setattr(runtime_repair, "Stage", CapturingStage)
    diagnostic = HarnessDiagnostic.create(
        stage=HarnessStage.CONNECTING_AGENT,
        component="chat_ingress",
        code="chat_ingress_failed",
        message="HTTP 500",
    )
    guidance = (
        "<untrusted_current_attempt_process_logs>\n"
        "build.log\ninstallation failed\n"
        "worlds-w0-agent.log\nConversation run failed\n"
        "Vertex models require an optional SDK. Install with:\n"
        '"agent-sdk[vertex]"\n'
        "</untrusted_current_attempt_process_logs>"
    )
    assert (
        asyncio.run(
            runtime_repair.request_runtime_plan_patch(
                tmp_path, contract, (diagnostic,), guidance=guidance
            )
        )
        is None
    )
    assert "agent-sdk/pyproject.toml: vertex -> google-cloud-aiplatform" in briefings[0]
    assert "secret-value" not in briefings[0]
    clues = (
        briefings[0]
        .split("<untrusted_error_clues>", 1)[1]
        .split("</untrusted_error_clues>", 1)[0]
    )
    assert '"agent-sdk[vertex]"' in clues
    assert "installation failed" not in clues


def test_optional_extra_inventory_ignores_malformed_source_metadata(tmp_path):
    from fi.alk.harness.runtime_repair import _optional_extra_inventory

    (tmp_path / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers=["agent"]\n')
    member = tmp_path / "agent"
    member.mkdir()
    (member / "pyproject.toml").write_text('project="malformed"\n')
    runtime = Runtime(
        language="python",
        execution="native",
        workdir="agent",
        command=["python", "-m", "agent"],
    )
    assert _optional_extra_inventory(tmp_path, runtime) == ""
    (tmp_path / "pyproject.toml").write_text('[tool]\nuv="malformed"\n')
    assert _optional_extra_inventory(tmp_path, runtime) == ""


@pytest.mark.parametrize("location", ["root", "member"])
def test_optional_extra_inventory_never_reads_external_manifest_symlinks(
    tmp_path, monkeypatch, location
):
    from fi.alk.harness import generated_runtime
    from fi.alk.harness.runtime_repair import _optional_extra_inventory

    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.toml"
    outside.write_text(
        '[project]\nname="outside"\n[project.optional-dependencies]\n'
        'leaked=["private-package"]\n'
    )
    member = source / "agent"
    member.mkdir()
    if location == "root":
        (source / "pyproject.toml").symlink_to(outside)
        (member / "pyproject.toml").write_text('[project]\nname="agent"\n')
    else:
        (source / "pyproject.toml").write_text(
            '[tool.uv.workspace]\nmembers=["agent"]\n'
        )
        (member / "pyproject.toml").symlink_to(outside)
    original = generated_runtime._pyproject_document

    def checked(path):
        assert path.resolve().is_relative_to(source)
        return original(path)

    monkeypatch.setattr(generated_runtime, "_pyproject_document", checked)
    runtime = Runtime(
        language="python",
        execution="native",
        workdir="agent",
        command=["python", "-m", "agent"],
    )
    assert "leaked" not in _optional_extra_inventory(source, runtime)


def test_runtime_repair_progress_reports_tools_and_rejection_codes_without_payload(
    capsys,
):
    from fi.alk.harness.runtime_repair import _repair_progress
    from fi.alk.harness.session import Event

    _repair_progress(Event(kind="tool", tool="Read", detail={"target": "secret-file"}))
    _repair_progress(
        Event(
            kind="result",
            text="patch rejected: ValueError: runtime_repair_extra_unknown secret-value",
            detail={"is_error": True},
        )
    )
    _repair_progress(
        Event(kind="done", detail={"outcome": "failed", "error": "secret-value"})
    )
    output = capsys.readouterr().out
    assert "runtime repair: Read" in output
    assert "runtime repair: rejected runtime_repair_extra_unknown" in output
    assert "runtime repair: failed" in output
    assert "secret-file" not in output
    assert "secret-value" not in output


def test_runtime_repair_hides_loader_field_without_missing_loader(
    tmp_path, monkeypatch
):
    import asyncio
    from fi.alk.harness import runtime_repair
    from fi.alk.harness.diagnostics import HarnessDiagnostic
    from fi.alk.harness.job import HarnessStage

    contract = AgentContract(agent="test", runtime=Runtime(language="python"))

    class InspectingStage:
        def __init__(self, spec, **kwargs):
            schema = spec.servers["runtime_repair"].tools[0].input_schema
            assert "runtime_dependencies" not in schema["properties"]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def say(self, briefing, *, on_event=None):
            pass

    monkeypatch.setattr(runtime_repair, "Stage", InspectingStage)
    diagnostic = HarnessDiagnostic.create(
        stage=HarnessStage.BUILDING_ENVIRONMENT,
        component="runtime",
        code="generated_runtime_plan_invalid",
        message="bad runtime",
    )
    with pytest.raises(RuntimeError, match="typed_runtime_plan_patch_not_submitted"):
        asyncio.run(
            runtime_repair.request_runtime_plan_patch(tmp_path, contract, (diagnostic,))
        )


def test_explicit_install_hint_selects_only_validated_workspace_extra(
    tmp_path, monkeypatch
):
    import asyncio
    from fi.alk.harness import runtime_repair
    from fi.alk.harness.diagnostics import HarnessDiagnostic
    from fi.alk.harness.job import HarnessStage
    from fi.alk.harness.runtime_repair import _explicit_dependency_candidate

    (tmp_path / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers=["server", "sdk"]\n'
    )
    server = tmp_path / "server"
    server.mkdir()
    (server / "pyproject.toml").write_text(
        '[project]\nname="server"\nversion="1"\ndependencies=["agent-sdk"]\n'
    )
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    (sdk / "pyproject.toml").write_text(
        '[project]\nname="agent-sdk"\nversion="1"\n'
        '[project.optional-dependencies]\nvertex=["google-cloud-aiplatform"]\n'
    )
    contract = AgentContract(
        agent="test",
        runtime=Runtime(
            language="python",
            execution="native",
            workdir="server",
            command=["python", "-m", "server"],
        ),
    )
    guidance = (
        "<untrusted_current_target_process_logs>\nagent.log\n"
        'Vertex SDK. Install with: pip\ninstall\n"agent-sdk[vertex]"\n'
        "</untrusted_current_target_process_logs>"
    )
    patch = _explicit_dependency_candidate(tmp_path, contract, guidance)
    assert patch is not None
    assert patch.runtime.extras == ["vertex"]
    assert patch.evidence_paths == ("sdk/pyproject.toml",)
    assert patch.runtime_dependencies is None
    monkeypatch.setattr(
        runtime_repair,
        "Stage",
        lambda *_args, **_kwargs: pytest.fail(
            "explicit candidate should bypass model repair"
        ),
    )
    diagnostic = HarnessDiagnostic.create(
        stage=HarnessStage.CONNECTING_AGENT,
        component="chat_ingress",
        code="chat_ingress_failed",
        message="HTTP 500",
    )
    assert (
        asyncio.run(
            runtime_repair.request_runtime_plan_patch(
                tmp_path, contract, (diagnostic,), guidance=guidance
            )
        )
        == patch
    )
    assert (
        _explicit_dependency_candidate(tmp_path, contract, "install agent-sdk[vertex]")
        is None
    )
    assert _explicit_dependency_candidate(
        tmp_path, contract,
        guidance.replace("current_target_process", "current_attempt_process"),
    ) is None
    assert _explicit_dependency_candidate(
        tmp_path, contract,
        "<untrusted_current_attempt_process_logs>\n"
        "build.log\ninstall agent-sdk[vertex]\n"
        "sidecar.log\ninstall agent-sdk[vertex]\n"
        "</untrusted_current_attempt_process_logs>\n"
        "<untrusted_current_target_process_logs>\n"
        "agent.log\nHTTP 500\n</untrusted_current_target_process_logs>",
    ) is None
    assert (
        _explicit_dependency_candidate(
            tmp_path, contract, guidance.replace("agent-sdk[vertex]", "unknown[vertex]")
        )
        is None
    )
    assert (
        _explicit_dependency_candidate(
            tmp_path,
            contract,
            guidance.replace(
                "</untrusted_current_target_process_logs>",
                "install server[other]\n</untrusted_current_target_process_logs>",
            ),
        )
        is None
    )


def test_explicit_install_hint_rejects_excluded_or_external_manifest(tmp_path):
    from fi.alk.harness.runtime_repair import _explicit_dependency_candidate

    source = tmp_path / "source"
    source.mkdir()
    sdk = source / "sdk"
    sdk.mkdir()
    manifest = sdk / "pyproject.toml"
    manifest.write_text(
        '[project]\nname="agent-sdk"\nversion="1"\n'
        "[project.optional-dependencies]\nvertex=[]\n"
    )
    root_manifest = source / "pyproject.toml"
    root_manifest.write_text('[tool.uv.workspace]\nmembers=["sdk"]\nexclude=["sdk"]\n')
    contract = AgentContract(agent="test", runtime=Runtime(language="python"))
    guidance = (
        "<untrusted_current_target_process_logs>\n"
        'install "agent-sdk[vertex]"\n</untrusted_current_target_process_logs>'
    )
    assert _explicit_dependency_candidate(source, contract, guidance) is None
    root_manifest.write_text('[tool.uv.workspace]\nmembers=["sdk"]\n')
    manifest.unlink()
    outside = tmp_path / "outside.toml"
    outside.write_text(
        '[project]\nname="agent-sdk"\n[project.optional-dependencies]\nvertex=[]\n'
    )
    manifest.symlink_to(outside)
    assert _explicit_dependency_candidate(source, contract, guidance) is None


@pytest.mark.parametrize(
    "code,install",
    [
        ("chat_ingress_failed", ""),
        ("generated_runtime_plan_invalid", "pip install -e modules/a[transport]"),
    ],
)
def test_runtime_patch_cannot_select_context_excluded_extra_owner(
    tmp_path, code, install
):
    (tmp_path / "app.py").write_text("print('agent')")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="root"\nversion="1"\n[tool.uv.workspace]\nmembers=["modules/*"]\n'
    )
    member = tmp_path / "modules/a"
    member.mkdir(parents=True)
    (member / "pyproject.toml").write_text(
        '[project]\nname="member"\nversion="1"\n[project.optional-dependencies]\ntransport=["some-library"]\n'
    )
    contract = AgentContract(
        agent="fixture",
        runtime=Runtime(
            language="python",
            execution="native",
            command=["python", "app.py"],
            context_excludes=["modules/a"],
            install=install,
        ),
    )
    patch = RuntimePlanPatch(
        base_contract_hash=_contract_hash(contract),
        runtime=contract.runtime.model_copy(update={"extras": ["transport"]}),
        evidence_paths=("modules/a/pyproject.toml",),
        diagnostic_codes=(code,),
        summary="Select the declared transport",
    )
    with pytest.raises(ValueError, match="extra_unknown"):
        validate_runtime_plan_patch(
            tmp_path, contract, patch, allowed_diagnostic_codes={code}
        )


def test_readiness_patch_adds_source_requirements_without_changing_extras(
    tmp_path, dependency_patch
):
    contract, patch = dependency_patch
    (tmp_path / "backend.txt").write_text("client>=1\n")
    patch = patch.model_copy(
        update={
            "runtime": contract.runtime.model_copy(
                update={"requirements_files": ["backend.txt"]}
            ),
            "evidence_paths": ("backend.txt",),
        }
    )
    repaired = validate_runtime_plan_patch(
        tmp_path, contract, patch, allowed_diagnostic_codes={"chat_ingress_failed"}
    )
    assert repaired.runtime.requirements_files == ["backend.txt"]
    assert repaired.runtime.extras == ["existing"]
    (tmp_path / "backend.txt").write_text("-r missing.txt\n")
    with pytest.raises(ValueError, match="requirements_path_invalid"):
        validate_runtime_plan_patch(
            tmp_path, contract, patch, allowed_diagnostic_codes={"chat_ingress_failed"}
        )


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("public_image", [False, True])
def test_compose_repair_uses_same_native_selection_policy(
    tmp_path, monkeypatch, native, public_image
):
    monkeypatch.delenv("ALK_COMPOSE_BROKER_SOCKET", raising=False)
    monkeypatch.delenv("ALK_TRUSTED_COMPOSE_SOCKET", raising=False)
    if native:
        monkeypatch.setenv("ALK_TRUSTED_COMPOSE_SOCKET", "/private/alk/docker.sock")
    (tmp_path / "service").mkdir()
    (tmp_path / "service/app.py").write_text("print('source')")
    (tmp_path / "service/pyproject.toml").write_text(
        '[project]\nname="fixture"\nversion="1"\n'
    )
    image = "${IMAGE}" if public_image else "example/app:1"
    (tmp_path / "compose.yaml").write_text(
        f"services:\n  app:\n    build: service\n    image: {image}\n"
        '    environment: ["OPTIONAL_URL=http://optional:8080"]\n'
        "  optional:\n    image: example/optional:1\n"
    )
    contract = AgentContract(agent="fixture")
    patch = _patch(
        contract,
        execution="compose",
        compose_file="compose.yaml",
        service="app",
        config_files={
            ".env": {"format": "dotenv", "values": {"IMAGE": "example/app:1"}}
        },
    )
    if native:
        repaired = validate_runtime_plan_patch(
            tmp_path,
            contract,
            patch,
            allowed_diagnostic_codes={"generated_runtime_plan_invalid"},
        )
        from fi.alk.harness.runtime_repair import _dependency_root

        assert _dependency_root(tmp_path, repaired.runtime) == tmp_path / "service"
    else:
        with pytest.raises(ValueError, match="compose_dependency_undeclared"):
            validate_runtime_plan_patch(
                tmp_path,
                contract,
                patch,
                allowed_diagnostic_codes={"generated_runtime_plan_invalid"},
            )


@pytest.mark.parametrize("entry", ["server.ts", "server.tsx"])
def test_bun_repair_validates_child_workdir_entrypoint(tmp_path, entry):
    source = tmp_path / "source"
    child = source / "packages" / "chat"
    child.mkdir(parents=True)
    (source / "package.json").write_text('{"name":"chat"}')
    (child / entry).write_text("export const ready = true")
    contract = AgentContract(agent="chat", runtime=Runtime(language="bun"))
    patch = RuntimePlanPatch(
        base_contract_hash=_contract_hash(contract),
        runtime=Runtime(
            language="bun", command=["bun", "run", "--cwd", "packages/chat", entry]
        ),
        evidence_paths=("package.json",),
        diagnostic_codes=("generated_runtime_plan_invalid",),
        summary="Run the native TypeScript source.",
    )
    validate_runtime_plan_patch(
        source,
        contract,
        patch,
        allowed_diagnostic_codes={"generated_runtime_plan_invalid"},
    )
    (child / entry).unlink()
    with pytest.raises(ValueError, match="entrypoint_missing"):
        validate_runtime_plan_patch(
            source,
            contract,
            patch,
            allowed_diagnostic_codes={"generated_runtime_plan_invalid"},
        )


@pytest.mark.parametrize(
    "arguments",
    [["--cwd", "../escape", "app.ts"], ["--cwd=../escape", "app.ts"], ["--cwd"]],
)
def test_bun_repair_rejects_unsafe_child_workdir(tmp_path, arguments):
    (tmp_path / "package.json").write_text('{"packageManager":"bun@1.3.14"}')
    contract = AgentContract(agent="chat", runtime=Runtime(language="bun"))
    patch = RuntimePlanPatch(
        base_contract_hash=_contract_hash(contract),
        runtime=Runtime(language="bun", command=["bun", "run", *arguments]),
        evidence_paths=("package.json",),
        diagnostic_codes=("generated_runtime_plan_invalid",),
        summary="Run the native TypeScript source.",
    )
    with pytest.raises(ValueError, match="workdir"):
        validate_runtime_plan_patch(
            tmp_path,
            contract,
            patch,
            allowed_diagnostic_codes={"generated_runtime_plan_invalid"},
        )


def test_readiness_patch_resolves_extra_in_sibling_workspace_package(tmp_path):
    (tmp_path / "server").mkdir()
    (tmp_path / "library").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers=["server", "library"]\n'
    )
    (tmp_path / "server/pyproject.toml").write_text(
        '[project]\nname="server"\nversion="1"\n'
    )
    (tmp_path / "server/app.py").write_text("print('source')")
    (tmp_path / "library/pyproject.toml").write_text(
        '[project]\nname="library"\nversion="1"\n[project.optional-dependencies]\ntransport=[]\n'
    )
    contract = AgentContract(
        agent="fixture",
        runtime=Runtime(
            language="python",
            execution="native",
            workdir="server",
            command=["python", "app.py"],
            interface=RuntimeInterface(kind="http", port=8080, path="/chat"),
        ),
    )
    patch = RuntimePlanPatch(
        base_contract_hash=_contract_hash(contract),
        runtime=contract.runtime.model_copy(update={"extras": ["transport"]}),
        evidence_paths=("library/pyproject.toml",),
        diagnostic_codes=("chat_ingress_failed",),
        summary="Source-declared transport dependency is in the sibling package.",
    )
    repaired = validate_runtime_plan_patch(
        tmp_path, contract, patch, allowed_diagnostic_codes={"chat_ingress_failed"}
    )
    assert repaired.runtime.extras == ["transport"]
    assert repaired.runtime.workdir == "server"
