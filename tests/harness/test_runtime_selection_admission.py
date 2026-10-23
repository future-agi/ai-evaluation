import pytest

from fi.alk.harness.bundle_author_v2 import BundleAuthorError, resolve_environment_plan
from fi.alk.harness.contract import Runtime
from fi.alk.harness.job import HarnessJob


def job():
    return HarnessJob.model_validate(
        {
            "job_id": "selection-test",
            "run_id": "selection-test",
            "execution": "hosted",
            "source": {"kind": "archive", "archive_artifact_id": "fixture"},
            "agent": {"connector": "http"},
            "scenario_count": 1,
            "runtime": {
                "isolation": "dedicated_vm",
                "cpu_units": 2,
                "memory_mb": 4096,
                "parallelism": 1,
            },
        }
    )


def native():
    return {
        "execution": "native",
        "command": ["python", "server.py"],
        "interface": {"kind": "http", "port": 8111, "path": "/chat"},
    }


def test_explicit_native_ignores_unrelated_compose(tmp_path):
    (tmp_path / "server.py").write_text("print('source')")
    (tmp_path / "compose.yaml").write_text(
        "services:\n  unrelated:\n    image: arbitrary:1\n"
    )
    plan = resolve_environment_plan(
        tmp_path,
        job(),
        contract_modality="chat",
        contract_interface_kind="http",
        contract_runtime=native(),
    )
    assert plan.control_service == "agent"
    assert plan.processes[-1].run_command == ["python", "server.py"]


def test_selected_compose_path_and_service_preserve_dependencies(tmp_path):
    (tmp_path / "deployment").mkdir()
    (tmp_path / "component").mkdir()
    (tmp_path / "component/server.py").write_text("print('source')")
    (tmp_path / "deployment/custom.yaml").write_text("""services:
  dialogue:
    build: ../component
    command: [python, server.py]
    depends_on: [cache]
  cache:
    image: redis:7
  unrelated:
    image: arbitrary:1
""")
    runtime = {
        **native(),
        "execution": "compose",
        "compose_file": "deployment/custom.yaml",
        "service": "dialogue",
        "workdir": "component",
    }
    plan = resolve_environment_plan(
        tmp_path,
        job(),
        contract_modality="chat",
        contract_interface_kind="http",
        contract_runtime=runtime,
    )
    assert {p.name for p in plan.processes} == {"world-db", "dialogue", "cache"}
    process = next(p for p in plan.processes if p.name == "dialogue")
    assert process.working_directory == "component"
    assert process.depends_on == ["cache"]


def test_selected_image_reports_capability(tmp_path):
    (tmp_path / "compose.yaml").write_text(
        "services:\n  dialogue:\n    image: arbitrary:1\n"
    )
    with pytest.raises(BundleAuthorError, match="engine_unsupported"):
        resolve_environment_plan(
            tmp_path,
            job(),
            contract_modality="chat",
            contract_interface_kind="http",
            contract_runtime={
                **native(),
                "execution": "compose",
                "compose_file": "compose.yaml",
                "service": "dialogue",
            },
        )


def test_node_is_explicitly_unsupported_in_process_lane(tmp_path):
    (tmp_path / "package.json").write_text("{}")
    (tmp_path / "server.js").write_text("console.log(1)")
    with pytest.raises(BundleAuthorError, match="runtime_language_unsupported"):
        resolve_environment_plan(
            tmp_path,
            job(),
            contract_modality="chat",
            contract_interface_kind="http",
            contract_runtime={
                **native(),
                "language": "javascript",
                "command": ["node", "server.js"],
            },
        )


def test_runtime_rejects_conflicting_selection():
    with pytest.raises(ValueError):
        Runtime(execution="native", compose_file="compose.yaml")


def test_unused_static_credentials_stay_inventory_only(tmp_path):
    from fi.alk.harness.credentials import discover_credentials

    (tmp_path / "server.py").write_text("print('source')")
    (tmp_path / "optional.py").write_text(
        "import os\ndef unused(): return os.environ['UNUSED_TOKEN']\n"
    )
    (tmp_path / "client.js").write_text(
        "const path = process.env.OPTIONAL_PATH ?? 'default';"
    )
    manifest = discover_credentials(tmp_path)
    assert {"UNUSED_TOKEN", "OPTIONAL_PATH"} <= {
        r.environment_name for r in manifest.requirements
    }
    assert not manifest.ready
    plan = resolve_environment_plan(
        tmp_path,
        job(),
        contract_modality="chat",
        contract_interface_kind="http",
        contract_runtime=native(),
    )
    assert plan.control_service == "agent"


def test_required_environment_is_process_scoped(tmp_path):
    (tmp_path / "server.py").write_text("print('source')")
    (tmp_path / "compose.yaml").write_text("""services:
  dialogue:
    build: .
    depends_on: [worker]
    environment: {SERVICE_TOKEN: public-value}
  worker:
    build: .
    command: [python, server.py]
""")
    with pytest.raises(BundleAuthorError, match="worker:SERVICE_TOKEN"):
        resolve_environment_plan(
            tmp_path,
            job(),
            contract_modality="chat",
            contract_interface_kind="http",
            contract_runtime={
                **native(),
                "execution": "compose",
                "compose_file": "compose.yaml",
                "service": "dialogue",
                "required_environment": {"worker": ["SERVICE_TOKEN"]},
            },
        )


def test_selected_compose_mandatory_substitution_blocks(tmp_path):
    (tmp_path / "server.py").write_text("print('source')")
    (tmp_path / "compose.yaml").write_text("""services:
  dialogue:
    build: .
    environment: {MANDATORY_TOKEN: '${MANDATORY_TOKEN:?required}'}
  omitted:
    build: .
    environment: {MANDATORY_TOKEN: public-value}
""")
    with pytest.raises(BundleAuthorError, match="dialogue:MANDATORY_TOKEN"):
        resolve_environment_plan(
            tmp_path,
            job(),
            contract_modality="chat",
            contract_interface_kind="http",
            contract_runtime={
                **native(),
                "execution": "compose",
                "compose_file": "compose.yaml",
                "service": "dialogue",
            },
        )


def test_required_environment_rejects_unknown_process(tmp_path):
    (tmp_path / "server.py").write_text("print('source')")
    with pytest.raises(BundleAuthorError, match="scope_unknown: omitted"):
        resolve_environment_plan(
            tmp_path,
            job(),
            contract_modality="chat",
            contract_interface_kind="http",
            contract_runtime={
                **native(),
                "required_environment": {"omitted": ["TOKEN"]},
            },
        )


def test_native_configuration_ref_must_reach_process(tmp_path):
    from fi.simulate.runtime.spec import SecretRef

    (tmp_path / "server.py").write_text("print('source')")
    configured = job()
    configured.agent.secret_refs["RUNTIME_TOKEN"] = SecretRef(
        manager="platform-vault", key="token", purpose="simulator_provider"
    )
    runtime = {**native(), "required_environment": {"agent": ["RUNTIME_TOKEN"]}}
    with pytest.raises(BundleAuthorError, match="agent:RUNTIME_TOKEN"):
        resolve_environment_plan(
            tmp_path,
            configured,
            contract_modality="chat",
            contract_interface_kind="http",
            contract_runtime=runtime,
        )
    configured.agent.secret_refs["RUNTIME_TOKEN"] = SecretRef(
        manager="platform-vault", key="token", purpose="target_provider"
    )
    plan = resolve_environment_plan(
        tmp_path,
        configured,
        contract_modality="chat",
        contract_interface_kind="http",
        contract_runtime=runtime,
    )
    assert plan.control_service == "agent"


@pytest.mark.parametrize("operator", [":?", "?"])
def test_aliased_compose_requirement_cannot_be_satisfied_by_sibling_literal(
    tmp_path, operator
):
    (tmp_path / "server.py").write_text("print('source')")
    (tmp_path / "compose.yaml").write_text(
        'services:\n  dialogue:\n    build: .\n    environment:\n      SOURCE: literal\n      DEST: "${SOURCE'
        + operator
        + 'required}"\n'
    )
    with pytest.raises(
        BundleAuthorError,
        match="compose_interpolation_unsupported: dialogue.DEST aliases SOURCE",
    ):
        resolve_environment_plan(
            tmp_path,
            job(),
            contract_modality="chat",
            contract_interface_kind="http",
            contract_runtime={
                **native(),
                "execution": "compose",
                "compose_file": "compose.yaml",
                "service": "dialogue",
            },
        )


def test_compose_default_satisfies_process_declaration(tmp_path):
    (tmp_path / "server.py").write_text("print('source')")
    (tmp_path / "compose.yaml").write_text(
        'services:\n  dialogue:\n    build: .\n    environment: {REGION: "${REGION:-west}"}\n'
    )
    plan = resolve_environment_plan(
        tmp_path,
        job(),
        contract_modality="chat",
        contract_interface_kind="http",
        contract_runtime={
            **native(),
            "execution": "compose",
            "compose_file": "compose.yaml",
            "service": "dialogue",
            "required_environment": {"dialogue": ["REGION"]},
        },
    )
    assert (
        next(p for p in plan.processes if p.name == "dialogue").environment["REGION"]
        == "west"
    )


def test_structural_interpolation_does_not_read_host_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SELECTED_IMAGE", "redis:7")
    (tmp_path / "compose.yaml").write_text(
        'services:\n  dialogue:\n    image: "${SELECTED_IMAGE}"\n'
    )
    with pytest.raises(
        BundleAuthorError,
        match="compose_structural_configuration_missing: SELECTED_IMAGE",
    ):
        resolve_environment_plan(
            tmp_path,
            job(),
            contract_runtime={
                "execution": "compose",
                "compose_file": "compose.yaml",
                "service": "dialogue",
            },
        )


def test_mongo_dependency_is_not_discarded(tmp_path):
    (tmp_path / "server.py").write_text("print('source')")
    (tmp_path / "compose.yaml").write_text("""services:
  dialogue:
    build: .
    depends_on: [storage]
  storage:
    image: mongo:7
""")
    with pytest.raises(
        BundleAuthorError, match="engine_unsupported: image-only service 'storage'"
    ):
        resolve_environment_plan(
            tmp_path,
            job(),
            contract_modality="chat",
            contract_interface_kind="http",
            contract_runtime={
                **native(),
                "execution": "compose",
                "compose_file": "compose.yaml",
                "service": "dialogue",
            },
        )


def test_native_local_dependency_is_reported(tmp_path):
    (tmp_path / "server.py").write_text("print('source')")
    with pytest.raises(
        BundleAuthorError, match="runtime_dependency_unsupported: storage"
    ):
        resolve_environment_plan(
            tmp_path,
            job(),
            contract_modality="chat",
            contract_interface_kind="http",
            contract_runtime=native(),
            runtime_dependencies=[
                {
                    "name": "storage",
                    "kind": "database",
                    "engine": "arbitrary-db",
                    "reached": {"dsn_env": "STORAGE_URL"},
                }
            ],
        )


def test_invalid_selection_and_required_names():
    for runtime in [
        {"execution": "compose"},
        {"service": "target"},
        {"compose_file": "../outside.yaml"},
        {"required_environment": {"agent": ["X", "X"]}},
        {"required_environment": {"agent": ["bad-name"]}},
        {"context_excludes": ["clients/[generated]"]},
        {"context_excludes": ["../source"]},
        {"context_excludes": ["tests", "tests"]},
    ]:
        with pytest.raises(ValueError):
            Runtime.model_validate(runtime)
    assert Runtime.model_validate(
        {"context_excludes": ["clients/**", "**/*.test.ts"]}
    ).context_excludes == ["clients/**", "**/*.test.ts"]


def test_explicit_native_survives_unselected_compose_ambiguity(tmp_path):
    from fi.alk.harness.bundle_author_v2 import _source_schema_artifacts

    (tmp_path / "server.py").write_text("print('source')")
    (tmp_path / "compose.yaml").write_text("invalid")
    (tmp_path / "docker-compose.yml").write_text("also invalid")
    plan = resolve_environment_plan(
        tmp_path,
        job(),
        contract_modality="chat",
        contract_interface_kind="http",
        contract_runtime=native(),
    )
    assert plan.control_service == "agent"
    assert _source_schema_artifacts(tmp_path, contract={"runtime": native()}) == []


def test_runtime_schema_fields_match_manual_tool_schema(tmp_path, monkeypatch):
    import fi.alk.harness.tools as contract_tools

    captured = {}

    def capture_tool(name, description, schema):
        captured[name] = schema
        return lambda function: function

    monkeypatch.setattr(contract_tools, "tool", capture_tool)
    monkeypatch.setattr(contract_tools, "tool_server", lambda **kwargs: kwargs)
    contract_tools.contract_tools(tmp_path)
    manual = captured["submit_contract"]["properties"]["runtime"]["properties"]
    canonical = Runtime.model_json_schema()["properties"]
    for name in (
        "execution", "service", "compose_file", "required_environment", "compose_image_services"
    ):
        assert manual[name]["type"] == [canonical[name]["type"], "null"]
    assert [
        value for value in manual["execution"]["enum"] if value is not None
    ] == canonical["execution"]["enum"]


def test_unselected_compose_interpolation_does_not_block_selected_service(tmp_path):
    (tmp_path / "server.py").write_text("print('source')")
    (tmp_path / "compose.yaml").write_text(
        'services:\n  dialogue:\n    build: .\n  omitted:\n    image: "${UNSELECTED_IMAGE}"\n'
    )
    plan = resolve_environment_plan(
        tmp_path,
        job(),
        contract_modality="chat",
        contract_interface_kind="http",
        contract_runtime={
            **native(),
            "execution": "compose",
            "compose_file": "compose.yaml",
            "service": "dialogue",
        },
    )
    assert {process.name for process in plan.processes} == {"world-db", "dialogue"}


def test_image_name_containing_engine_name_is_not_catalog_support(tmp_path):
    (tmp_path / "compose.yaml").write_text(
        "services:\n  dialogue:\n    image: arbitrary/redis-custom:1\n"
    )
    with pytest.raises(BundleAuthorError, match="engine_unsupported"):
        resolve_environment_plan(
            tmp_path,
            job(),
            contract_runtime={
                "execution": "compose",
                "compose_file": "compose.yaml",
                "service": "dialogue",
            },
        )


def test_dependency_cycle_is_reported(tmp_path):
    (tmp_path / "compose.yaml").write_text("""services:
  dialogue:
    build: .
    depends_on: [worker]
  worker:
    build: .
    depends_on: [dialogue]
""")
    with pytest.raises(BundleAuthorError, match="compose_dependency_cycle"):
        resolve_environment_plan(
            tmp_path,
            job(),
            contract_runtime={
                "execution": "compose",
                "compose_file": "compose.yaml",
                "service": "dialogue",
            },
        )


def test_selected_service_without_image_or_build_cannot_invent_source(tmp_path):
    (tmp_path / "compose.yaml").write_text(
        "services:\n  dialogue:\n    command: [python, absent.py]\n"
    )
    with pytest.raises(BundleAuthorError, match="source_service_unsupported"):
        resolve_environment_plan(
            tmp_path,
            job(),
            contract_runtime={
                "execution": "compose",
                "compose_file": "compose.yaml",
                "service": "dialogue",
            },
        )


@pytest.mark.parametrize("arguments", ["", "    command: [--port, '8111']\n"])
def test_selected_compose_entrypoint_cannot_be_replaced_with_invented_command(
    tmp_path, arguments
):
    (tmp_path / "server.py").write_text("print('source')")
    (tmp_path / "compose.yaml").write_text(
        "services:\n  dialogue:\n    build: .\n"
        "    entrypoint: [python, server.py]\n" + arguments
    )
    assert not (tmp_path / "agent.py").exists()
    with pytest.raises(
        BundleAuthorError, match="compose_entrypoint_unsupported: dialogue"
    ):
        resolve_environment_plan(
            tmp_path,
            job(),
            contract_runtime={
                "execution": "compose",
                "compose_file": "compose.yaml",
                "service": "dialogue",
            },
        )


def test_selected_build_without_source_launcher_cannot_infer_missing_agent(tmp_path):
    (tmp_path / "server.py").write_text("print('source')")
    (tmp_path / "compose.yaml").write_text("services:\n  dialogue:\n    build: .\n")
    with pytest.raises(
        BundleAuthorError, match="runtime_entrypoint_missing: dialogue:agent.py"
    ):
        resolve_environment_plan(
            tmp_path,
            job(),
            contract_runtime={
                "execution": "compose",
                "compose_file": "compose.yaml",
                "service": "dialogue",
            },
        )


@pytest.mark.parametrize("command", [None, ["python", "server.py"]])
def test_selected_build_preserves_existing_or_explicit_source_launcher(
    tmp_path, command
):
    import yaml

    entry = "agent.py" if command is None else "server.py"
    (tmp_path / entry).write_text("print('source')")
    service = {"build": "."}
    if command:
        service["command"] = command
    (tmp_path / "compose.yaml").write_text(
        yaml.safe_dump({"services": {"dialogue": service}})
    )
    plan = resolve_environment_plan(
        tmp_path,
        job(),
        contract_runtime={
            "execution": "compose",
            "compose_file": "compose.yaml",
            "service": "dialogue",
        },
    )
    process = next(process for process in plan.processes if process.name == "dialogue")
    assert process.run_command[-1] == entry
    if command:
        assert process.run_command == command


def test_generated_callback_adapter_does_not_need_conventional_agent_script(tmp_path):
    (tmp_path / "callbacks.py").write_text(
        'def agent_callback(messages):\n    return "ok"\n'
    )
    assert not (tmp_path / "agent.py").exists()
    plan = resolve_environment_plan(
        tmp_path,
        job(),
        contract_modality="chat",
        contract_interface_kind="callable",
        contract_runtime={"interface": {"kind": "callable"}},
    )
    process = next(process for process in plan.processes if process.name == "agent")
    assert process.run_command[-2] == "-c"
    assert process.environment["ALK_CALLBACK_ENTRYPOINT"] == "callbacks:agent_callback"


def test_required_environment_accepts_process_materialized_secret_path(tmp_path):
    from fi.simulate.runtime.spec import SecretRef

    (tmp_path / "server.py").write_text("print('source')")
    configured = job()
    configured.agent.secret_refs["GOOGLE_APPLICATION_CREDENTIALS_JSON"] = SecretRef(
        manager="mounted", key="fixture-document", purpose="target_provider"
    )
    plan = resolve_environment_plan(
        tmp_path,
        configured,
        contract_modality="chat",
        contract_interface_kind="http",
        contract_runtime={
            **native(),
            "required_environment": {"agent": ["GOOGLE_APPLICATION_CREDENTIALS"]},
        },
    )
    assert plan.control_service == "agent"


@pytest.mark.parametrize("purpose", [None, "simulator_provider"])
def test_materialized_secret_requires_an_input_routed_to_the_process(tmp_path, purpose):
    from fi.simulate.runtime.spec import SecretRef

    (tmp_path / "server.py").write_text("print('source')")
    configured = job()
    if purpose:
        configured.agent.secret_refs["GOOGLE_APPLICATION_CREDENTIALS_JSON"] = SecretRef(
            manager="mounted", key="fixture-document", purpose=purpose
        )
    with pytest.raises(BundleAuthorError, match="agent:GOOGLE_APPLICATION_CREDENTIALS"):
        resolve_environment_plan(
            tmp_path,
            configured,
            contract_modality="chat",
            contract_interface_kind="http",
            contract_runtime={
                **native(),
                "required_environment": {"agent": ["GOOGLE_APPLICATION_CREDENTIALS"]},
            },
        )


def test_materialized_secret_path_does_not_satisfy_sibling_process(tmp_path):
    from fi.simulate.runtime.spec import SecretRef

    (tmp_path / "server.py").write_text("print('source')")
    (tmp_path / "compose.yaml").write_text(
        "services:\n  dialogue:\n    build: .\n    depends_on: [worker]\n"
        "  worker:\n    build: .\n    command: [python, server.py]\n"
    )
    configured = job()
    configured.agent.secret_refs["GOOGLE_APPLICATION_CREDENTIALS_JSON"] = SecretRef(
        manager="mounted", key="fixture-document", purpose="target_provider"
    )
    with pytest.raises(
        BundleAuthorError, match="worker:GOOGLE_APPLICATION_CREDENTIALS"
    ):
        resolve_environment_plan(
            tmp_path,
            configured,
            contract_modality="chat",
            contract_interface_kind="http",
            contract_runtime={
                **native(),
                "execution": "compose",
                "compose_file": "compose.yaml",
                "service": "dialogue",
                "required_environment": {"worker": ["GOOGLE_APPLICATION_CREDENTIALS"]},
            },
        )


def test_local_editable_extra_must_be_selected(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='shell'\nversion='1'\n[tool.uv.workspace]\nmembers=['sdk', 'tools']\n"
    )
    for name in ("sdk", "tools"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "pyproject.toml").write_text(
            f"[project]\nname='{name}'\nversion='1'\n"
            + (
                "[project.optional-dependencies]\ncloud=['cloud-client']\n"
                if name == "sdk"
                else ""
            )
        )
    (tmp_path / "server.py").write_text("print('source')")
    runtime = native() | {"install": "pip install -e sdk[cloud] -e tools", "extras": []}
    with pytest.raises(
        BundleAuthorError, match="generated_runtime_extra_omitted.*cloud"
    ):
        resolve_environment_plan(
            tmp_path,
            job(),
            contract_modality="chat",
            contract_interface_kind="http",
            contract_runtime=runtime,
        )
    runtime["extras"] = ["cloud"]
    plan = resolve_environment_plan(
        tmp_path,
        job(),
        contract_modality="chat",
        contract_interface_kind="http",
        contract_runtime=runtime,
    )
    command = plan.processes[-1].build_commands[0]
    assert "--all-packages --extra cloud" in " ".join(command)


def test_process_lane_rejects_selected_dependency_image(tmp_path):
    (tmp_path / "server.py").write_text("print('source')")
    (tmp_path / "compose.yaml").write_text(
        "services:\n  dialogue:\n    build: .\n    depends_on: [cache]\n  cache:\n    image: redis:7\n"
    )
    with pytest.raises(
        BundleAuthorError, match="runtime_compose_image_requires_container_runtime"
    ):
        resolve_environment_plan(
            tmp_path,
            job(),
            contract_modality="chat",
            contract_interface_kind="http",
            contract_runtime={
                **native(),
                "execution": "compose",
                "compose_file": "compose.yaml",
                "service": "dialogue",
                "compose_image_services": ["cache"],
            },
        )
