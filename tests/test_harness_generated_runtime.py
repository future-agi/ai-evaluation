import json
import os
import time
from pathlib import Path

import pytest

from fi.alk.harness.contract import AgentContract, DataStore, Runtime, ToolSpec
from fi.alk.harness.bundle import export_session_bundle
from fi.alk.harness.generated_runtime import (
    GENERATED_DOCKERFILE,
    GeneratedRuntimeError,
    detect_generated_runtime,
    prepare_generated_runtime,
)
from fi.alk.harness import provision as provisioning
from fi.alk.harness.contract import Dependency
from fi.alk.harness.provision import reset, start_runtime, stop, stop_runtime


def _contract(**runtime) -> AgentContract:
    return AgentContract(
        agent="unpackaged-agent",
        tools=[ToolSpec(name="answer", args=["question"])],
        real_use_cases=["answer a caller"],
        runtime=Runtime(**runtime),
    )


def test_unambiguous_python_repository_gets_a_non_root_plan(tmp_path: Path) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    source.joinpath("requirements.txt").write_text("requests==2.32.4\n")
    source.joinpath("agent.py").write_text(
        'if __name__ == "__main__":\n    print("ready")\n'
    )

    plan = detect_generated_runtime(source, Runtime())

    assert plan.language == "python"
    assert plan.install_strategy == "pip-requirements"
    assert plan.command == ("python", "agent.py")
    assert "USER 10001:10001" in plan.dockerfile
    assert (
        "apt-get install -y --no-install-recommends build-essential" in plan.dockerfile
    )


def test_legacy_setup_py_repository_gets_generated_runtime(tmp_path: Path) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    source.joinpath("setup.py").write_text(
        "from setuptools import setup\nsetup(name='legacy-chat-agent', version='0.1')\n"
    )
    source.joinpath("run.py").write_text("print('ready')\n")

    plan = detect_generated_runtime(
        source, Runtime(language="python", command=["python", "run.py"])
    )

    assert plan.install_strategy == "pip-setup-project-unlocked"
    assert plan.command == ("python", "run.py")
    assert "RUN pip install --no-cache-dir ." in plan.dockerfile


def test_livekit_python_entrypoint_uses_worker_start_mode(tmp_path: Path) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    source.joinpath("requirements.txt").write_text("livekit-agents==1.2.0\n")
    source.joinpath("agent.py").write_text(
        "from livekit.agents import cli\ncli.run_app(object())\n"
    )

    plan = detect_generated_runtime(source, Runtime())

    assert plan.command == ("python", "agent.py", "start")


def test_lockless_uv_script_project_uses_uv_and_downloads_livekit_models(
    tmp_path: Path,
) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    source.joinpath("pyproject.toml").write_text(
        """[project]
name = "voice-script"
version = "0"
dependencies = [
  "livekit-agents>=1.6",
  "livekit-plugins-silero>=1.5.7",
]
[tool.uv]
package = false
"""
    )
    source.joinpath("agent.py").write_text(
        "from livekit.agents import cli\ncli.run_app(object())\n"
    )

    plan = detect_generated_runtime(source, Runtime())

    assert plan.install_strategy == "uv-script-project-unlocked"
    assert "uv sync --no-cache" in plan.dockerfile
    assert "pip install --no-cache-dir '.'" not in plan.dockerfile
    assert "python -m livekit.agents download-files" in plan.dockerfile


def test_python_compatibility_floor_does_not_become_an_exact_old_image(
    tmp_path: Path,
) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    source.joinpath("pyproject.toml").write_text(
        """[project]
name = "voice-script"
version = "0"
requires-python = ">=3.10"
dependencies = []
[tool.uv]
package = false
"""
    )
    source.joinpath("agent.py").write_text("print('ready')\n")

    plan = detect_generated_runtime(source, Runtime(version=">=3.10"))

    assert plan.version == "3.12"
    assert plan.dockerfile.startswith("FROM python:3.12-slim")


def test_selected_component_accepts_contract_absolute_monorepo_hint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "examples" / "drive_thru"
    source.mkdir(parents=True)
    source.joinpath("requirements.txt").write_text("")
    source.joinpath("agent.py").write_text("print('ready')\n")

    plan = detect_generated_runtime(
        source, Runtime(language="python", workdir="/examples/drive_thru")
    )

    assert plan.component == ""


def test_selected_component_accepts_contract_relative_monorepo_hint(
    tmp_path: Path,
) -> None:
    root = tmp_path / "examples" / "hotel_receptionist"
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "hotel"\nversion = "0"\n', encoding="utf-8"
    )
    (root / "agent.py").write_text("print('ready')\n", encoding="utf-8")

    runtime = Runtime(
        language="python",
        version="3.12",
        workdir="examples/hotel_receptionist",
        command=[],
        extras=[],
    )
    plan = detect_generated_runtime(root, runtime)

    assert plan.component == ""
    assert plan.command == ("python", "agent.py")


def test_repository_root_workdir_sentinel_uses_submitted_source(tmp_path: Path) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    source.joinpath("requirements.txt").write_text("")
    source.joinpath("agent.py").write_text("print('ready')\n")

    plan = detect_generated_runtime(source, Runtime(workdir="/"))

    assert plan.component == ""
    assert plan.command == ("python", "agent.py")


def test_unrelated_absolute_workdir_remains_rejected(tmp_path: Path) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    source.joinpath("requirements.txt").write_text("")
    source.joinpath("agent.py").write_text("print('ready')\n")

    with pytest.raises(GeneratedRuntimeError, match="workdir_escapes_repository"):
        detect_generated_runtime(source, Runtime(workdir="/etc"))


def test_node_repository_uses_lockfile_and_start_script(tmp_path: Path) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    source.joinpath("package.json").write_text(
        json.dumps({"scripts": {"start": "node src/agent.js"}})
    )
    source.joinpath("package-lock.json").write_text("{}")

    plan = detect_generated_runtime(source, Runtime(version="20"))

    assert plan.install_strategy == "npm-ci"
    assert plan.command == ("npm", "run", "start")
    assert plan.dockerfile.startswith("FROM node:20-slim")
    assert "USER node" in plan.dockerfile


def test_node_start_runtime_runs_declared_build_phase(tmp_path: Path) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    source.joinpath("package.json").write_text(
        json.dumps(
            {
                "scripts": {
                    "build": "tsc",
                    "start": "node dist/agent.js",
                }
            }
        )
    )
    source.joinpath("package-lock.json").write_text("{}")

    plan = detect_generated_runtime(source, Runtime())

    assert "RUN npm run build" in plan.dockerfile


def test_python_optional_group_is_inferred_from_grounded_entrypoint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    source.joinpath("pyproject.toml").write_text(
        """[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
[project]
name = "voice-agent"
version = "1.0.0"
[project.optional-dependencies]
voice = ["sounddevice>=0.4"]
dev = ["pytest"]
"""
    )
    source.joinpath("examples").mkdir()
    source.joinpath("examples/voice_demo.py").write_text("print('voice')\n")

    plan = detect_generated_runtime(
        source, Runtime(command=["python", "examples/voice_demo.py"])
    )

    assert plan.extras == ("voice",)
    assert "'.[voice]'" in plan.dockerfile


def test_ambiguous_entrypoint_stops_before_packaging(tmp_path: Path) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    source.joinpath("requirements.txt").write_text("requests==2.32.4\n")
    source.joinpath("agent.py").write_text("print('agent')\n")
    source.joinpath("main.py").write_text("print('main')\n")

    with pytest.raises(GeneratedRuntimeError, match="entrypoint_ambiguous"):
        detect_generated_runtime(source, Runtime())


def test_explicit_argv_resolves_entrypoint_ambiguity(tmp_path: Path) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    source.joinpath("requirements.txt").write_text("requests==2.32.4\n")
    source.joinpath("agent.py").write_text("print('agent')\n")
    source.joinpath("main.py").write_text("print('main')\n")

    plan = detect_generated_runtime(
        source, Runtime(command=["python", "main.py", "--worker"])
    )

    assert plan.command == ("python", "main.py", "--worker")


def test_generated_context_excludes_secrets_and_does_not_mutate_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    source.joinpath("requirements.txt").write_text("")
    source.joinpath("agent.py").write_text("print('ready')\n")
    source.joinpath(".env").write_text("OPENAI_API_KEY=secret\n")
    source.joinpath("private.pem").write_text("secret\n")
    before = sorted(path.name for path in source.iterdir())

    plan = prepare_generated_runtime(source, tmp_path / "session", Runtime())
    context = Path(plan.context_directory)

    assert sorted(path.name for path in source.iterdir()) == before
    assert context.joinpath(GENERATED_DOCKERFILE).is_file()
    assert not context.joinpath(".env").exists()
    assert not context.joinpath("private.pem").exists()
    recorded = json.loads(
        tmp_path.joinpath("session/generated-runtime.json").read_text()
    )
    assert recorded["fingerprint"] == plan.fingerprint


def test_symlink_in_generated_context_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    source.joinpath("requirements.txt").write_text("")
    source.joinpath("agent.py").write_text("print('ready')\n")
    source.joinpath("escape").symlink_to(tmp_path / "outside")

    with pytest.raises(GeneratedRuntimeError, match="symlink_forbidden"):
        prepare_generated_runtime(source, tmp_path / "session", Runtime())


def test_internal_symlink_is_materialized_without_preserving_link(
    tmp_path: Path,
) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    source.joinpath("requirements.txt").write_text("")
    source.joinpath("agent.py").write_text("print('ready')\n")
    data = source / "data"
    data.mkdir()
    data.joinpath("task.json").write_text('{"task": 1}\n')
    source.joinpath("public-data").symlink_to(data, target_is_directory=True)

    plan = prepare_generated_runtime(source, tmp_path / "session", Runtime())
    materialized = Path(plan.context_directory) / "public-data" / "task.json"

    assert materialized.read_text() == '{"task": 1}\n'
    assert not materialized.parent.is_symlink()


def test_generated_context_omits_explicit_large_output_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    source.joinpath("requirements.txt").write_text("")
    source.joinpath("agent.py").write_text("print('ready')\n")
    results = source / "data" / "results"
    results.mkdir(parents=True)
    results.joinpath("old.json").write_text("{}\n")

    plan = prepare_generated_runtime(
        source,
        tmp_path / "session",
        Runtime(context_excludes=["data/results"]),
    )

    assert plan.context_excludes == ("data/results",)
    assert not Path(plan.context_directory, "data/results").exists()


def test_generated_context_allows_excluding_not_yet_created_runtime_state(
    tmp_path: Path,
) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    source.joinpath("requirements.txt").write_text("")
    source.joinpath("agent.py").write_text("print('ready')\n")

    plan = prepare_generated_runtime(
        source,
        tmp_path / "session",
        Runtime(context_excludes=["fake_data/hotel.db"]),
    )

    assert plan.context_excludes == ()
    assert Path(plan.context_directory, "agent.py").is_file()


def test_generated_context_cannot_exclude_submitted_entrypoint(tmp_path: Path) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    source.joinpath("requirements.txt").write_text("")
    scripts = source / "scripts"
    scripts.mkdir()
    scripts.joinpath("agent.py").write_text("print('ready')\n")

    with pytest.raises(GeneratedRuntimeError, match="context_exclude_required"):
        detect_generated_runtime(
            source,
            Runtime(
                command=["python", "scripts/agent.py"],
                context_excludes=["scripts"],
            ),
        )


def test_provision_builds_generated_runtime_without_touching_source(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "agent"
    session = tmp_path / "session"
    source.mkdir()
    source.joinpath("requirements.txt").write_text("")
    source.joinpath("agent.py").write_text("print('ready')\n")
    calls: list[tuple[str, ...]] = []

    def run(_environment, *arguments, **_kwargs):
        calls.append(arguments)
        if "config" in arguments and "--format" in arguments:
            return json.dumps(
                {
                    "services": {
                        "agent-runtime": {
                            "profiles": ["harness-runtime"],
                            "build": {
                                "context": str(session / "generated-runtime-context"),
                                "dockerfile": GENERATED_DOCKERFILE,
                            },
                        }
                    }
                }
            )
        return ""

    monkeypatch.setattr(provisioning, "_run", run)
    environment = provisioning.provision(source, session, _contract())

    assert environment.managed and environment.runtime_fingerprint
    assert environment.generated_runtime_plan.endswith("generated-runtime.json")
    assert ("build", "agent-runtime") in calls
    assert not source.joinpath(GENERATED_DOCKERFILE).exists()
    generated = json.loads(session.joinpath("managed-compose.json").read_text())
    runtime = generated["services"]["agent-runtime"]
    assert runtime["command"] == ["python", "agent.py"]
    assert runtime["build"]["context"] == str(session / "generated-runtime-context")


def test_contract_command_changes_runtime_reuse_fingerprint(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "agent"
    session = tmp_path / "session"
    source.mkdir()
    source.joinpath("requirements.txt").write_text("")
    source.joinpath("agent.py").write_text("print('ready')\n")

    def run(_environment, *arguments, **_kwargs):
        if "config" in arguments and "--format" in arguments:
            return json.dumps(
                {"services": {"agent-runtime": {"profiles": ["harness-runtime"]}}}
            )
        return ""

    monkeypatch.setattr(provisioning, "_run", run)
    first = provisioning.provision(
        source, session, _contract(command=["python", "agent.py"])
    )
    second = provisioning.provision(
        source, session, _contract(command=["python", "agent.py", "--verbose"])
    )

    assert first.runtime_fingerprint != second.runtime_fingerprint


def test_external_model_provider_is_not_treated_as_managed_infrastructure(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "agent"
    session = tmp_path / "session"
    source.mkdir()
    source.joinpath("requirements.txt").write_text("")
    source.joinpath("agent.py").write_text("print('ready')\n")
    contract = _contract(command=["python", "agent.py"])
    contract.dependencies = [
        Dependency(
            name="Deepgram",
            kind="service",
            engine="deepgram nova-3",
            what="speech recognition provider",
            reached={"password_from": "DEEPGRAM_API_KEY"},
        )
    ]

    def run(_environment, *arguments, **_kwargs):
        if "config" in arguments and "--format" in arguments:
            return json.dumps(
                {"services": {"agent-runtime": {"profiles": ["harness-runtime"]}}}
            )
        return ""

    monkeypatch.setattr(provisioning, "_run", run)
    environment = provisioning.provision(source, session, contract)

    compose = json.loads(session.joinpath("managed-compose.json").read_text())
    assert set(compose["services"]) == {"agent-runtime"}
    assert environment.runtime_services == ["agent-runtime"]


def test_external_provider_dsn_environment_is_not_a_managed_sidecar(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "agent"
    session = tmp_path / "session"
    source.mkdir()
    source.joinpath("requirements.txt").write_text("")
    source.joinpath("agent.py").write_text("print('ready')\n")
    contract = _contract(command=["python", "agent.py"])
    contract.dependencies = [
        Dependency(
            name="OpenAI",
            kind="service",
            engine="gpt-4.1-mini",
            what="hosted model provider",
            reached={
                "dsn_env": "OPENAI_API_KEY",
                "host": "api.openai.com",
                "port": 443,
            },
        )
    ]

    def run(_environment, *arguments, **_kwargs):
        if "config" in arguments and "--format" in arguments:
            return json.dumps(
                {"services": {"agent-runtime": {"profiles": ["harness-runtime"]}}}
            )
        return ""

    monkeypatch.setattr(provisioning, "_run", run)
    environment = provisioning.provision(source, session, contract)

    compose = json.loads(session.joinpath("managed-compose.json").read_text())
    assert set(compose["services"]) == {"agent-runtime"}
    assert environment.runtime_services == ["agent-runtime"]


def test_importable_library_capability_is_not_a_managed_sidecar(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "agent"
    session = tmp_path / "session"
    source.mkdir()
    source.joinpath("requirements.txt").write_text("")
    source.joinpath("agent.py").write_text("print('ready')\n")
    contract = _contract(command=["python", "agent.py"])
    contract.dependencies = [
        Dependency(
            name="GetEmailTask",
            kind="service",
            engine="livekit-agents beta workflows",
            what="SDK-provided workflow",
            reached={
                "loader_module": "livekit.agents",
                "loader_function": "beta.workflows.GetEmailTask",
            },
        )
    ]

    def run(_environment, *arguments, **_kwargs):
        if "config" in arguments and "--format" in arguments:
            return json.dumps(
                {"services": {"agent-runtime": {"profiles": ["harness-runtime"]}}}
            )
        return ""

    monkeypatch.setattr(provisioning, "_run", run)
    environment = provisioning.provision(source, session, contract)

    compose = json.loads(session.joinpath("managed-compose.json").read_text())
    assert set(compose["services"]) == {"agent-runtime"}
    assert environment.runtime_services == ["agent-runtime"]


def test_generated_runtime_packages_are_not_treated_as_sidecars(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "agent"
    session = tmp_path / "session"
    source.mkdir()
    source.joinpath("pyproject.toml").write_text(
        """[project]
name = "hotel"
version = "0"
dependencies = ["livekit-plugins-silero>=1.6.0", "apsw>=3.46"]
"""
    )
    source.joinpath("agent.py").write_text("print('ready')\n")
    contract = _contract(command=["python", "agent.py"])
    contract.data_store = DataStore(kind="sqlite", host=":memory:")
    contract.dependencies = [
        Dependency(
            name="Silero VAD",
            kind="service",
            engine="livekit-plugins-silero>=1.6.0",
        ),
        Dependency(
            name="APSW SQLite",
            kind="datastore",
            engine="apsw>=3.46",
            reached={"host": ":memory:"},
        ),
    ]

    def run(_environment, *arguments, **_kwargs):
        if "config" in arguments and "--format" in arguments:
            return json.dumps(
                {"services": {"agent-runtime": {"profiles": ["harness-runtime"]}}}
            )
        return ""

    monkeypatch.setattr(provisioning, "_run", run)
    environment = provisioning.provision(source, session, contract)

    compose = json.loads(session.joinpath("managed-compose.json").read_text())
    assert set(compose["services"]) == {"agent-runtime"}
    assert environment.runtime_services == ["agent-runtime"]


def test_in_process_store_needs_no_managed_sidecar(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "agent"
    session = tmp_path / "session"
    source.mkdir()
    source.joinpath("requirements.txt").write_text("")
    source.joinpath("agent.py").write_text("print('ready')\n")
    contract = _contract(command=["python", "agent.py"])
    contract.data_store = {"kind": "in_process", "configured_by": "hardcoded"}

    def run(_environment, *arguments, **_kwargs):
        if "config" in arguments and "--format" in arguments:
            return json.dumps(
                {"services": {"agent-runtime": {"profiles": ["harness-runtime"]}}}
            )
        return ""

    monkeypatch.setattr(provisioning, "_run", run)
    environment = provisioning.provision(source, session, contract)

    compose = json.loads(session.joinpath("managed-compose.json").read_text())
    assert set(compose["services"]) == {"agent-runtime"}
    assert environment.services == []


def test_generated_runtime_persists_only_required_credential_names(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "agent"
    session = tmp_path / "session"
    source.mkdir()
    source.joinpath("requirements.txt").write_text("")
    source.joinpath("agent.py").write_text("print('ready')\n")
    contract = _contract(command=["python", "agent.py"])
    contract.dependencies = [
        Dependency(
            name="LiveKit",
            kind="service",
            reached={
                "dsn_env": "LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET",
            },
        )
    ]

    def run(_environment, *arguments, **_kwargs):
        if "config" in arguments and "--format" in arguments:
            return json.dumps(
                {"services": {"agent-runtime": {"profiles": ["harness-runtime"]}}}
            )
        return ""

    monkeypatch.setattr(provisioning, "_run", run)
    monkeypatch.setenv("LIVEKIT_API_SECRET", "must-never-be-persisted")
    environment = provisioning.provision(source, session, contract)

    assert environment.runtime_configuration_names == [
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "LIVEKIT_URL",
    ]
    persisted = session.joinpath("environment.json").read_text()
    assert "must-never-be-persisted" not in persisted


def test_generated_runtime_includes_runtime_dependency_configuration_names(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "agent"
    session = tmp_path / "session"
    source.mkdir()
    source.joinpath("requirements.txt").write_text("")
    source.joinpath("agent.py").write_text("print('ready')\n")
    contract = _contract(command=["python", "agent.py"])
    contract.runtime_dependencies = [
        Dependency(
            name="Vertex AI",
            kind="transport",
            reached={
                "dsn_env": "GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION",
            },
        )
    ]

    def run(_environment, *arguments, **_kwargs):
        if "config" in arguments and "--format" in arguments:
            return json.dumps(
                {"services": {"agent-runtime": {"profiles": ["harness-runtime"]}}}
            )
        return ""

    monkeypatch.setattr(provisioning, "_run", run)
    environment = provisioning.provision(source, session, contract)

    assert environment.runtime_configuration_names == [
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_CLOUD_PROJECT",
    ]


def test_generated_runtime_includes_ephemeral_uploaded_environment_names(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "agent"
    session = tmp_path / "session"
    source.mkdir()
    source.joinpath("requirements.txt").write_text("")
    source.joinpath("agent.py").write_text("print('ready')\n")
    contract = _contract(command=["python", "agent.py"])

    def run(_environment, *arguments, **_kwargs):
        if "config" in arguments and "--format" in arguments:
            return json.dumps(
                {"services": {"agent-runtime": {"profiles": ["harness-runtime"]}}}
            )
        return ""

    monkeypatch.setattr(provisioning, "_run", run)
    monkeypatch.setenv(
        "ALK_RUNTIME_CONFIGURATION_NAMES", "CUSTOM_REGION,FEATURE_FLAG,ALK_FORBIDDEN"
    )
    monkeypatch.setenv("CUSTOM_REGION", "must-never-be-persisted")
    environment = provisioning.provision(source, session, contract)

    assert environment.runtime_configuration_names == ["CUSTOM_REGION", "FEATURE_FLAG"]
    assert (
        "must-never-be-persisted"
        not in session.joinpath("environment.json").read_text()
    )


def test_generated_worker_translates_loopback_livekit_without_persisting_it(
    tmp_path: Path, monkeypatch
) -> None:
    environment = provisioning.ProvisionedEnvironment(
        source=str(tmp_path),
        compose_file=str(tmp_path / "compose.json"),
        project="generated-runtime-test",
        runtime_services=["agent-runtime"],
        runtime_configuration_names=["LIVEKIT_URL"],
        running=True,
    )
    environment.save(tmp_path)
    monkeypatch.setenv("ALK_RUNTIME_VALUE_LIVEKIT_URL", "ws://127.0.0.1:17880")
    monkeypatch.setenv("HARNESS_RUNTIME_STABLE_SECONDS", "0")
    seen: dict[str, str] = {}

    monkeypatch.setattr(
        provisioning,
        "_config",
        lambda _environment: {"services": {"agent-runtime": {"environment": {}}}},
    )

    def run(_environment, *arguments, **kwargs):
        seen.update(kwargs.get("process_overrides") or {})
        return ""

    monkeypatch.setattr(provisioning, "_run", run)
    monkeypatch.setattr(
        provisioning,
        "_docker",
        lambda *arguments, **_kwargs: (
            "running" if arguments and arguments[0] == "inspect" else ""
        ),
    )

    provisioning.start_runtime(tmp_path)

    assert seen["LIVEKIT_URL"] == "ws://host.docker.internal:17880"
    assert "127.0.0.1:17880" not in tmp_path.joinpath("environment.json").read_text()


def test_runtime_does_not_revalidate_container_target_for_mounted_google_credentials(
    tmp_path: Path, monkeypatch
) -> None:
    credential = tmp_path / "google.json"
    credential.write_text('{"type":"service_account"}\n')
    container_target = "/etc/vertex/creds.json"
    environment = provisioning.ProvisionedEnvironment(
        source=str(tmp_path),
        compose_file=str(tmp_path / "compose.json"),
        project="mounted-google-test",
        runtime_services=["agent-runtime"],
        running=True,
    )
    environment.save(tmp_path)
    monkeypatch.setenv("HARNESS_RUNTIME_STABLE_SECONDS", "0")
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        provisioning,
        "_config",
        lambda _environment: {
            "services": {
                "agent-runtime": {
                    "environment": {
                        "GOOGLE_APPLICATION_CREDENTIALS": container_target
                    },
                    "volumes": [
                        {"source": str(credential), "target": container_target}
                    ],
                }
            }
        },
    )

    def run(_environment, *arguments, **kwargs):
        seen["arguments"] = arguments
        seen["injected"] = kwargs.get("process_overrides") or {}
        return ""

    monkeypatch.setattr(provisioning, "_run", run)
    monkeypatch.setattr(
        provisioning,
        "_docker",
        lambda *arguments, **_kwargs: (
            "running" if arguments and arguments[0] == "inspect" else ""
        ),
    )

    provisioning.start_runtime(tmp_path)

    assert f"{credential.resolve()}:{container_target}:ro" in seen["arguments"]
    assert seen["injected"]["GOOGLE_APPLICATION_CREDENTIALS"] == container_target


def test_hosted_runtime_rejects_raw_google_host_path(tmp_path, monkeypatch) -> None:
    credential = tmp_path / "customer.json"
    credential.write_text('{"type":"service_account"}\n')
    monkeypatch.setenv("ALK_HOSTED_EXECUTION", "1")
    monkeypatch.setenv(
        "ALK_RUNTIME_VALUE_GOOGLE_APPLICATION_CREDENTIALS", str(credential)
    )
    monkeypatch.delenv("ALK_RUNTIME_SECRET_FILE_NAMES", raising=False)

    with pytest.raises(provisioning.ProvisionError, match="credential-file control"):
        provisioning._runtime_credential_mounts(
            {"GOOGLE_APPLICATION_CREDENTIALS": "/etc/vertex/creds.json"},
            {
                "volumes": [
                    {"source": "/dev/null", "target": "/etc/vertex/creds.json"}
                ]
            },
        )


def test_hosted_runtime_accepts_provider_authorized_google_file(
    tmp_path, monkeypatch
) -> None:
    credential = tmp_path / "customer.json"
    credential.write_text('{"type":"service_account"}\n')
    monkeypatch.setenv("ALK_HOSTED_EXECUTION", "1")
    monkeypatch.setenv(
        "ALK_RUNTIME_VALUE_GOOGLE_APPLICATION_CREDENTIALS", str(credential)
    )
    monkeypatch.setenv(
        "ALK_RUNTIME_SECRET_FILE_NAMES", "GOOGLE_APPLICATION_CREDENTIALS"
    )

    mounts = provisioning._runtime_credential_mounts(
        {"GOOGLE_APPLICATION_CREDENTIALS": "/etc/vertex/creds.json"},
        {"volumes": [{"source": "/dev/null", "target": "/etc/vertex/creds.json"}]},
    )

    assert mounts == [
        (
            "GOOGLE_APPLICATION_CREDENTIALS",
            credential.resolve(),
            f"/run/harness-secrets/{credential.name}",
        )
    ]


def test_managed_runtime_uses_generated_private_endpoint_over_submitted_default(
    tmp_path: Path, monkeypatch
) -> None:
    environment = provisioning.ProvisionedEnvironment(
        source=str(tmp_path),
        compose_file=str(tmp_path / "compose.json"),
        project="private-endpoint",
        managed=True,
        running=True,
        runtime_services=["agent-runtime"],
        internal_overrides={"TOOLS_API_URL": "http://tools-api:8080"},
        runtime_configuration_names=["TOOLS_API_URL"],
    )
    environment.save(tmp_path)
    monkeypatch.setenv("HARNESS_RUNTIME_STABLE_SECONDS", "0")
    monkeypatch.setenv("ALK_RUNTIME_CONFIGURATION_NAMES", "TOOLS_API_URL")
    monkeypatch.setenv("ALK_RUNTIME_VALUE_TOOLS_API_URL", "http://harness:8787")
    monkeypatch.setattr(
        provisioning,
        "_config",
        lambda _environment: {
            "services": {
                "agent-runtime": {
                    "environment": {"TOOLS_API_URL": "http://harness:8787"}
                }
            }
        },
    )
    seen: dict[str, object] = {}

    def run(_environment, *arguments, **kwargs):
        seen["arguments"] = arguments
        seen["injected"] = kwargs.get("process_overrides") or {}
        return ""

    monkeypatch.setattr(provisioning, "_run", run)
    monkeypatch.setattr(
        provisioning,
        "_docker",
        lambda *arguments, **_kwargs: (
            "running" if arguments and arguments[0] == "inspect" else ""
        ),
    )

    provisioning.start_runtime(tmp_path)

    assert seen["injected"]["TOOLS_API_URL"] == "http://tools-api:8080"
    arguments = seen["arguments"]
    assert "TOOLS_API_URL=http://tools-api:8080" in arguments


def test_compose_process_overrides_uploaded_runtime_endpoint(tmp_path: Path, monkeypatch) -> None:
    environment = provisioning.ProvisionedEnvironment(
        source=str(tmp_path),
        compose_file=str(tmp_path / "compose.json"),
        project="compose-precedence",
        runtime_configuration_names=["TOOLS_API_URL"],
    )
    monkeypatch.setenv("ALK_RUNTIME_CONFIGURATION_NAMES", "TOOLS_API_URL")
    monkeypatch.setenv("ALK_RUNTIME_VALUE_TOOLS_API_URL", "http://harness:8787")
    seen: dict[str, str] = {}

    def run(*_args, **kwargs):
        seen.update(kwargs["env"])
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(provisioning.subprocess, "run", run)

    provisioning._run(
        environment,
        "run",
        process_overrides={"TOOLS_API_URL": "http://tools-api:8080"},
    )

    assert seen["TOOLS_API_URL"] == "http://tools-api:8080"


def test_generated_runtime_bundle_is_portable_and_contains_no_host_context(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "agent"
    session = tmp_path / "session"
    source.mkdir()
    source.joinpath("requirements.txt").write_text("")
    source.joinpath("agent.py").write_text("print('ready')\n")

    def run(_environment, *arguments, **_kwargs):
        if "config" in arguments and "--format" in arguments:
            return json.dumps(
                {"services": {"agent-runtime": {"profiles": ["harness-runtime"]}}}
            )
        return ""

    monkeypatch.setattr(provisioning, "_run", run)
    provisioning.provision(source, session, _contract())
    root, bundle = export_session_bundle(source, session, name="unpackaged-agent")

    compose = json.loads(root.joinpath("compose.json").read_text())
    build = compose["services"]["agent-runtime"]["build"]
    assert build["context"] == "services/generated-runtime"
    assert build["dockerfile"] == GENERATED_DOCKERFILE
    assert root.joinpath("services/generated-runtime/agent.py").is_file()
    plan = json.loads(root.joinpath("generated-runtime.json").read_text())
    assert plan["context_directory"] == "services/generated-runtime"
    assert str(tmp_path) not in root.joinpath("compose.json").read_text()
    assert bundle.runtime.document == "compose.json"


def _wait_for_log(environment, expected: str, timeout: float = 30) -> str:
    deadline = time.monotonic() + timeout
    output = ""
    while time.monotonic() < deadline:
        output = provisioning._docker(
            "logs", environment.runtime_container, check=False, timeout=10
        )
        if expected in output:
            return output
        time.sleep(0.25)
    raise AssertionError(f"runtime did not emit {expected!r}:\n{output}")


@pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="requires Docker")
def test_real_unpackaged_python_agent_gets_redis_and_clean_reset(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HARNESS_RUNTIME_STABLE_SECONDS", "1")
    source = tmp_path / "python-agent"
    session = tmp_path / "python-session"
    source.mkdir()
    source.joinpath("requirements.txt").write_text("")
    source.joinpath("agent.py").write_text(
        """import os
import socket
import time
from urllib.parse import urlsplit

target = urlsplit(os.environ["REDIS_URL"])
sock = socket.create_connection((target.hostname, target.port), timeout=5)
sock.sendall(b"*2\\r\\n$3\\r\\nGET\\r\\n$9\\r\\nalk-proof\\r\\n")
before = sock.recv(1024)
sock.sendall(b"*3\\r\\n$3\\r\\nSET\\r\\n$9\\r\\nalk-proof\\r\\n$5\\r\\nready\\r\\n")
after = sock.recv(1024)
print(f"AGENT_READY before={before!r} set={after!r}", flush=True)
while True:
    time.sleep(60)
"""
    )
    contract = _contract()
    contract.dependencies = [Dependency(name="session-cache", engine="redis")]
    environment = provisioning.provision(source, session, contract)
    try:
        assert environment.services == ["redis"]
        assert environment.generated_runtime_plan
        bundle_root, bundle = export_session_bundle(
            source, session, name="unpackaged-python-redis"
        )
        assert bundle.runtime.document == "compose.json"
        assert bundle_root.joinpath(
            "services/generated-runtime/.alk-generated.Dockerfile"
        ).is_file()
        first = start_runtime(session)
        assert "before=b'$-1\\r\\n'" in _wait_for_log(first, "AGENT_READY")
        stop_runtime(session)
        reset(session)
        second = start_runtime(session)
        assert "before=b'$-1\\r\\n'" in _wait_for_log(second, "AGENT_READY")
    finally:
        stop_runtime(session)
        stop(session)


@pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="requires Docker")
def test_real_unpackaged_node_agent_builds_and_starts(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HARNESS_RUNTIME_STABLE_SECONDS", "1")
    source = tmp_path / "node-agent"
    session = tmp_path / "node-session"
    source.mkdir()
    source.joinpath("package.json").write_text(
        json.dumps(
            {
                "name": "unpackaged-node-agent",
                "version": "1.0.0",
                "scripts": {"start": "node agent.js"},
            }
        )
    )
    source.joinpath("package-lock.json").write_text(
        json.dumps(
            {
                "name": "unpackaged-node-agent",
                "version": "1.0.0",
                "lockfileVersion": 3,
                "requires": True,
                "packages": {"": {"name": "unpackaged-node-agent", "version": "1.0.0"}},
            }
        )
    )
    source.joinpath("agent.js").write_text(
        "console.log('AGENT_READY node=1'); setInterval(() => {}, 60000);\n"
    )
    environment = provisioning.provision(
        source, session, _contract(language="node", version="22")
    )
    try:
        assert environment.services == []
        bundle_root, bundle = export_session_bundle(
            source, session, name="unpackaged-node"
        )
        assert bundle.runtime.document == "compose.json"
        assert bundle_root.joinpath("services/generated-runtime/agent.js").is_file()
        running = start_runtime(session)
        assert "AGENT_READY node=1" in _wait_for_log(running, "AGENT_READY node=1")
    finally:
        stop_runtime(session)
        stop(session)
