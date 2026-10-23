from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from fi.alk.harness.bundle_author_v2 import (
    BundleAuthorError,
    _compile_source_tool_handlers,
    _contract_column_declarations,
    _contract_sql_type,
    _docker_python,
    _source_schema_artifacts,
    _sqlite_sql,
    _uv_sync_build_command,
    author_bundle_v2,
    reseal_scenarios,
    resolve_environment_plan,
)
from fi.alk.harness.authoring_runtime_validation import _artifact_digest
from fi.alk.harness.bundle_v2 import load_bundle_v2
from fi.alk.harness.certification import (
    CertificationAuthoring,
    CertificationChecks,
    CertificationCompiler,
    CertificationRuntime,
    CertificationSource,
    CertificationStatus,
    CheckStatus,
    HarnessCertification,
)
from fi.alk.harness.job import HarnessJob, ProviderExecutionMode
from fi.alk.harness.process_preflight import preflight_bundle
from fi.alk.harness.provision import source_fingerprint
from fi.alk.harness.repair_controller import RepairHistory
from fi.alk.harness.source_model import SourceModel
from fi.alk.harness.world.runtime import GeneratedWorld
from fi.alk.harness.world_ir import WorldIR


def test_source_tool_handler_preserves_safe_display_name(tmp_path: Path) -> None:
    written = _compile_source_tool_handlers(
        {
            "tool_entrypoints": [
                {
                    "tool": "Character Counter Tool",
                    "mode": "import",
                    "module": "example.tools",
                    "callable": "count",
                }
            ]
        },
        tmp_path,
    )

    assert written == ["handlers/Character Counter Tool.py"]
    assert (tmp_path / written[0]).is_file()


def test_source_tool_handler_rejects_path_separator(tmp_path: Path) -> None:
    with pytest.raises(BundleAuthorError, match="contract_tool_name_unsafe"):
        _compile_source_tool_handlers(
            {
                "tool_entrypoints": [
                    {
                        "tool": "../escape",
                        "mode": "import",
                        "module": "example.tools",
                        "callable": "count",
                    }
                ]
            },
            tmp_path,
        )


def _job(
    *,
    connector: str,
    with_secrets: bool = False,
    scenario_count: int = 1,
    secret_aliases: tuple[str, ...] = ("LIVEKIT_API_KEY",),
    metadata: dict[str, object] | None = None,
) -> HarnessJob:
    secret_refs = {}
    if with_secrets:
        # A source that really imports livekit declares the full credential set, where a toy
        # `print('agent')` entry declares none, so a realistic fixture needs more than the key.
        secret_refs = {
            alias: {
                "manager": "platform-vault",
                "key": alias.lower().replace("_", "-"),
                "purpose": "target_provider",
            }
            for alias in secret_aliases
        }
    return HarnessJob.model_validate(
        {
            "job_id": "job-v2",
            "run_id": "run-v2",
            "execution": "hosted",
            "source": {"kind": "archive", "archive_artifact_id": "source-1"},
            "agent": {"connector": connector, "secret_refs": secret_refs},
            "scenario_count": scenario_count,
            "metadata": metadata or {},
            "runtime": {
                "isolation": "dedicated_vm",
                "cpu_units": 2,
                "memory_mb": 4096,
                "parallelism": 1,
            },
        }
    )


def _authoring(root: Path) -> Path:
    artifact = root / "authoring"
    scenario = artifact / "scenarios" / "one"
    (scenario / "checks").mkdir(parents=True)
    (scenario / "scenario.json").write_text(
        json.dumps({"name": "one", "instruction": "Test one", "sub_goals": ["works"]}),
        encoding="utf-8",
    )
    (scenario / "setup.py").write_text(
        "def setup(world):\n    return None\n", encoding="utf-8"
    )
    (scenario / "ready.py").write_text(
        "def ready(world):\n    return None\n", encoding="utf-8"
    )
    (scenario / "checks" / "works.py").write_text(
        "def check(world, calls):\n    return None\n", encoding="utf-8"
    )
    return artifact


def _write_voice_contract(authoring: Path) -> None:
    (authoring / "contract.json").write_text(
        json.dumps({"modality": "voice"}), encoding="utf-8"
    )


def _write_callable_contract(authoring: Path) -> None:
    (authoring / "contract.json").write_text(
        json.dumps(
            {
                "modality": "chat",
                "runtime": {
                    "language": "python",
                    "interface": {
                        "kind": "callable",
                        "protocol": "fi.alk",
                        "include_tools": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _write_chat_contract(
    authoring: Path,
    *,
    command: list[str],
    workdir: str = ".",
    interface: dict[str, object] | None = None,
    data_store: str = "none",
) -> None:
    runtime: dict[str, object] = {
        "language": "python",
        "command": command,
        "workdir": workdir,
    }
    if interface is not None:
        runtime["interface"] = interface
    (authoring / "contract.json").write_text(
        json.dumps(
            {
                "modality": "chat",
                "runtime": runtime,
                "data_store": {"kind": data_store},
                "tools": [],
            }
        ),
        encoding="utf-8",
    )


def test_connect_only_provider_source_needs_no_agent_process(tmp_path: Path) -> None:
    source = tmp_path / "provider-target"
    source.mkdir()
    authoring = _authoring(tmp_path)
    _write_voice_contract(authoring)
    job = HarnessJob.model_validate(
        {
            "job_id": "provider-only-job",
            "run_id": "provider-only-run",
            "execution": "hosted",
            "source": {"kind": "provider", "visibility": "public"},
            "agent": {
                "connector": "retell",
                "mode": ProviderExecutionMode.CONNECT_ONLY.value,
                "config": {"agent_id": "agent_existing"},
                "secret_refs": {
                    "RETELL_API_KEY": {
                        "manager": "platform-vault",
                        "key": "retell-key",
                        "purpose": "target_provider",
                    }
                },
            },
            "scenario_count": 1,
            "runtime": {
                "isolation": "dedicated_vm",
                "cpu_units": 2,
                "memory_mb": 4096,
                "parallelism": 1,
            },
        }
    )

    plan = resolve_environment_plan(source, job, contract_modality="voice")
    assert plan.packaging == "provider_connect_only"
    assert plan.control_service is None
    assert [process.name for process in plan.processes] == ["world-db"]

    output = tmp_path / "bundle"
    bundle = author_bundle_v2(
        source=source,
        job=job,
        authoring=authoring,
        output=output,
    )

    assert bundle.runtime.control_service is None
    assert [process.name for process in bundle.processes] == ["world-db"]
    assert bundle.metadata["provider_connect_only"] == {"connector": "retell"}
    preflight_bundle(
        output,
        bundle,
        parallelism=1,
        secret_refs={"RETELL_API_KEY": "target_provider"},
    )


def test_phone_connect_only_uses_platform_telephony_without_target_secret(
    tmp_path: Path,
) -> None:
    source = tmp_path / "phone-target"
    source.mkdir()
    authoring = _authoring(tmp_path)
    _write_voice_contract(authoring)
    job = HarnessJob.model_validate(
        {
            "job_id": "phone-only-job",
            "run_id": "phone-only-run",
            "execution": "hosted",
            "source": {"kind": "provider", "visibility": "public"},
            "agent": {
                "connector": "phone",
                "mode": ProviderExecutionMode.CONNECT_ONLY.value,
                "config": {
                    "phone_number": "+14155551234",
                    "target_system_prompt": "Answer questions about a ride booking.",
                },
                "secret_refs": {},
            },
            "scenario_count": 1,
            "runtime": {
                "isolation": "dedicated_vm",
                "cpu_units": 2,
                "memory_mb": 4096,
                "parallelism": 1,
            },
        }
    )

    bundle = author_bundle_v2(
        source=source,
        job=job,
        authoring=authoring,
        output=tmp_path / "bundle",
    )

    assert bundle.metadata["provider_connect_only"] == {"connector": "phone"}
    preflight_bundle(tmp_path / "bundle", bundle, parallelism=1, secret_refs={})


def test_generic_connect_only_provider_uses_authored_world_schema(
    tmp_path: Path,
) -> None:
    source = tmp_path / "provider-target"
    source.mkdir()
    authoring = _authoring(tmp_path)
    _write_voice_contract(authoring)
    with sqlite3.connect(authoring / "world.sqlite") as database:
        database.execute("CREATE TABLE accounts (id TEXT PRIMARY KEY, status TEXT)")
        database.execute("INSERT INTO accounts VALUES ('acct-1', 'active')")
    body = {
        "job_id": "provider-generic-job",
        "run_id": "provider-generic-run",
        "execution": "hosted",
        "source": {"kind": "provider", "visibility": "public"},
        "agent": {
            "connector": "retell",
            "mode": "connect_only",
            "config": {"agent_id": "agent_existing"},
            "secret_refs": {
                "RETELL_API_KEY": {
                    "manager": "platform-vault",
                    "key": "retell-key",
                    "purpose": "target_provider",
                }
            },
        },
        "scenario_count": 1,
        "metadata": {"generic_harness_v1": True},
        "runtime": {
            "isolation": "dedicated_vm",
            "cpu_units": 2,
            "memory_mb": 4096,
            "parallelism": 1,
        },
    }

    bundle = author_bundle_v2(
        source=source,
        job=HarnessJob.model_validate(body),
        authoring=authoring,
        output=tmp_path / "bundle",
    )

    assert bundle.metadata["generic_harness"] == "v1"
    schema = (tmp_path / "bundle" / "seed" / "source-schema.sql").read_text()
    assert 'CREATE TABLE IF NOT EXISTS "accounts"' in schema
    assert "acct-1" not in schema


def test_connect_only_repository_source_still_compiles_uploaded_code(
    tmp_path: Path,
) -> None:
    source = tmp_path / "provider-target-with-code"
    source.mkdir()
    (source / "agent.py").write_text("print('agent')\n", encoding="utf-8")
    body = _job(connector="retell").model_dump(mode="json")
    body["agent"] = {
        "connector": "retell",
        "mode": ProviderExecutionMode.CONNECT_ONLY.value,
        "config": {"agent_id": "agent_existing"},
    }
    job = HarnessJob.model_validate(body)

    plan = resolve_environment_plan(source, job, contract_modality="voice")

    assert plan.packaging == "generated_python"
    assert plan.control_service == "agent"
    assert [process.name for process in plan.processes] == ["world-db", "agent"]


def test_bundle_compiles_discovered_source_tool_entrypoint(tmp_path: Path) -> None:
    source = tmp_path / "chat-agent"
    source.mkdir()
    (source / "agent.py").write_text("print('agent')\n", encoding="utf-8")
    (source / "customer_tools.py").write_text(
        "def lookup_account(email):\n    return {'email': email, 'status': 'active'}\n",
        encoding="utf-8",
    )
    authoring = _authoring(tmp_path)
    (authoring / "contract.json").write_text(
        json.dumps(
            {
                "modality": "chat",
                "runtime": {
                    "language": "python",
                    "interface": {
                        "kind": "openai_chat",
                        "protocol": "openai",
                        "include_tools": False,
                    },
                },
                "tools": [
                    {
                        "name": "lookup_account",
                        "args": ["email"],
                        "arg_types": {"email": "str"},
                    }
                ],
                "tool_entrypoints": [
                    {
                        "tool": "lookup_account",
                        "mode": "import",
                        "module": "customer_tools",
                        "callable": "lookup_account",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "bundle"
    author_bundle_v2(
        source=source,
        job=_job(connector="http"),
        authoring=authoring,
        output=output,
    )

    handler = (output / "handlers" / "lookup_account.py").read_text(encoding="utf-8")
    assert "from customer_tools import lookup_account" in handler
    assert "return settled(lookup_account(**args))" in handler
    load_bundle_v2(output)
    world = GeneratedWorld(":memory:")
    world.handlers["lookup_account"] = handler
    world.reach(str(source))
    call = world.call("lookup_account", {"email": "customer@example.com"})
    assert call.ok is True
    assert call.result == {
        "email": "customer@example.com",
        "status": "active",
    }


def test_tool_registration_is_not_observed_as_tool_execution(tmp_path: Path) -> None:
    source = tmp_path / "chat-agent"
    source.mkdir()
    (source / "agent.py").write_text("print('agent')\n", encoding="utf-8")
    (source / "tools.py").write_text(
        "class Editor:\n"
        "    @classmethod\n"
        "    def prepare(cls, context):\n"
        "        return [{'name': 'edit', 'description': 'Edit a file'}]\n",
        encoding="utf-8",
    )
    authoring = _authoring(tmp_path)
    _write_chat_contract(
        authoring,
        command=["python3", "agent.py"],
        interface={
            "kind": "http", "protocol": "openai_chat", "port": 8080,
            "path": "/chat", "include_tools": False,
        },
    )
    contract_path = authoring / "contract.json"
    contract = json.loads(contract_path.read_text())
    contract["tool_entrypoints"] = [
        {
            "tool": "edit", "mode": "construct", "module": "tools",
            "callable": "Editor.prepare", "factory": "Editor()",
        }
    ]
    contract_path.write_text(json.dumps(contract))
    bundle = author_bundle_v2(
        source=source, job=_job(connector="http"), authoring=authoring,
        output=tmp_path / "bundle",
    )
    process = next(p for p in bundle.processes if p.name == bundle.runtime.control_service)
    trace = tmp_path / "agent-tool-calls.jsonl"
    bootstrap = Path(__file__).resolve().parents[2] / "src/fi/alk/harness/livekit_tool_trace_bootstrap.py"
    shutil.copy2(bootstrap, source / "sitecustomize.py")
    result = subprocess.run(
        [sys.executable, "-c", "from tools import Editor; print(Editor.prepare({}))"],
        env={
            **os.environ,
            **{k: v.replace("{{WORLD_DIR}}", str(tmp_path)) for k, v in process.environment.items()},
            "PYTHONPATH": str(source),
        },
        capture_output=True, text=True, check=True,
    )
    assert "Edit a file" in result.stdout
    assert not trace.exists(), "Registering a tool does not prove the agent executed it"


def test_auto_voice_contract_compiles_livekit_process_runtime(tmp_path: Path) -> None:
    source = tmp_path / "voice-agent"
    source.mkdir()
    (source / "agent.py").write_text("print('agent')\n", encoding="utf-8")
    (source / "Dockerfile").write_text("FROM python:3.13\n", encoding="utf-8")
    (source / "pyproject.toml").write_text(
        "[project]\nname='agent'\nversion='1'\n", encoding="utf-8"
    )
    authoring = _authoring(tmp_path)
    _write_voice_contract(authoring)
    job = _job(
        connector="auto",
        with_secrets=True,
        secret_aliases=("LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LIVEKIT_URL"),
    )

    bundle = author_bundle_v2(
        source=source, job=job, authoring=authoring, output=tmp_path / "bundle"
    )

    agent = next(process for process in bundle.processes if process.name == "agent")
    assert agent.environment["LIVEKIT_AGENT_NAME"].endswith("-w{{WORLD_INDEX}}")
    assert agent.environment["HARNESS_MODE"] == "1"
    assert "target_provider" in agent.secret_purposes
    assert "target_http" not in bundle.capabilities


def test_livekit_cli_agent_without_a_dockerfile_is_started_with_a_subcommand(
    tmp_path: Path,
) -> None:
    """A repository that ships no Dockerfile has nothing to carry the subcommand.

    LiveKit's CLI prints its usage banner and exits when given none, so the worker never registers
    and the run fails at `spawn_failed` with the agent already dead.
    """
    source = tmp_path / "voice-agent"
    source.mkdir()
    (source / "agent.py").write_text(
        "from livekit.agents import cli, WorkerOptions\n"
        "if __name__ == '__main__':\n"
        "    cli.run_app(WorkerOptions(entrypoint_fnc=None))\n",
        encoding="utf-8",
    )
    (source / "requirements.txt").write_text("livekit-agents\n", encoding="utf-8")
    authoring = _authoring(tmp_path)
    _write_voice_contract(authoring)
    job = _job(
        connector="auto",
        with_secrets=True,
        secret_aliases=("LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LIVEKIT_URL"),
    )

    bundle = author_bundle_v2(
        source=source, job=job, authoring=authoring, output=tmp_path / "bundle"
    )

    agent = next(process for process in bundle.processes if process.name == "agent")
    assert agent.run_command[-1:] == ["start"]
    assert agent.fixed_port == 8081


def test_a_dockerfile_command_that_already_starts_a_worker_is_left_alone(
    tmp_path: Path,
) -> None:
    """The subcommand is added where one is missing, never doubled onto a command that has it."""
    source = tmp_path / "voice-agent"
    source.mkdir()
    (source / "agent.py").write_text(
        "from livekit.agents import cli, WorkerOptions\n"
        "cli.run_app(WorkerOptions(entrypoint_fnc=None))\n",
        encoding="utf-8",
    )
    (source / "pyproject.toml").write_text(
        "[project]\nname='agent'\nversion='1'\n", encoding="utf-8"
    )
    (source / "Dockerfile").write_text(
        'FROM python:3.13\nCMD ["python", "agent.py", "start"]\n', encoding="utf-8"
    )
    authoring = _authoring(tmp_path)
    _write_voice_contract(authoring)
    job = _job(
        connector="auto",
        with_secrets=True,
        secret_aliases=("LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LIVEKIT_URL"),
    )

    bundle = author_bundle_v2(
        source=source, job=job, authoring=authoring, output=tmp_path / "bundle"
    )

    agent = next(process for process in bundle.processes if process.name == "agent")
    assert agent.run_command.count("start") == 1
    assert agent.run_command[-1:] == ["start"]
    assert agent.fixed_port == 8081


def test_an_agent_that_starts_its_own_worker_gets_no_subcommand(tmp_path: Path) -> None:
    """Not every LiveKit agent is a CLI app; appending a subcommand to one of those breaks it."""
    source = tmp_path / "voice-agent"
    source.mkdir()
    (source / "agent.py").write_text(
        "from livekit.agents import Worker, WorkerOptions\n"
        "import asyncio\n"
        "asyncio.run(Worker(WorkerOptions(entrypoint_fnc=None)).run())\n",
        encoding="utf-8",
    )
    (source / "requirements.txt").write_text("livekit-agents\n", encoding="utf-8")
    authoring = _authoring(tmp_path)
    _write_voice_contract(authoring)
    job = _job(
        connector="auto",
        with_secrets=True,
        secret_aliases=("LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LIVEKIT_URL"),
    )

    bundle = author_bundle_v2(
        source=source, job=job, authoring=authoring, output=tmp_path / "bundle"
    )

    agent = next(process for process in bundle.processes if process.name == "agent")
    assert "start" not in agent.run_command


def test_bundle_rejects_missing_runtime_configuration_after_source_checkout(
    tmp_path: Path,
) -> None:
    source = tmp_path / "voice-agent"
    source.mkdir()
    (source / "agent.py").write_text(
        'import os\nproject = os.environ["GOOGLE_CLOUD_PROJECT"]\n',
        encoding="utf-8",
    )
    authoring = _authoring(tmp_path)
    _write_voice_contract(authoring)

    contract_path = authoring / "contract.json"
    contract = json.loads(contract_path.read_text())
    contract.setdefault("runtime", {})["required_environment"] = {
        "agent": ["GOOGLE_CLOUD_PROJECT"]
    }
    contract_path.write_text(json.dumps(contract))

    with pytest.raises(
        BundleAuthorError,
        match="target_runtime_configuration_missing: agent:GOOGLE_CLOUD_PROJECT",
    ):
        author_bundle_v2(
            source=source,
            job=_job(connector="auto", with_secrets=True),
            authoring=authoring,
            output=tmp_path / "bundle",
        )


def test_bundle_accepts_post_checkout_runtime_configuration_names(
    tmp_path: Path,
) -> None:
    source = tmp_path / "voice-agent"
    source.mkdir()
    (source / "agent.py").write_text(
        'import os\nproject = os.environ["GOOGLE_CLOUD_PROJECT"]\n',
        encoding="utf-8",
    )
    authoring = _authoring(tmp_path)
    _write_voice_contract(authoring)
    job = _job(
        connector="auto",
        with_secrets=True,
        secret_aliases=("LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LIVEKIT_URL"),
    ).model_copy(
        update={"metadata": {"environment_value_names": ["GOOGLE_CLOUD_PROJECT"]}}
    )

    bundle = author_bundle_v2(
        source=source, job=job, authoring=authoring, output=tmp_path / "bundle"
    )

    assert bundle.digest


@pytest.mark.parametrize(
    "project_dir,manifest",
    [(".", "pyproject.toml"), ("service", "pyproject.toml"), (".", "requirements.txt")],
)
def test_src_layout_keeps_nearest_project_manifest(tmp_path, project_dir, manifest):
    source = tmp_path / "source"
    project = source / project_dir
    (project / "src").mkdir(parents=True)
    (project / "src/agent.py").write_text("print('registered worker')\n")
    (project / manifest).write_text(
        "[project]\nname='agent'\nversion='1'\n"
        if manifest.endswith("toml")
        else "some-plugin\n"
    )
    (project / "Dockerfile").write_text(
        'FROM python:3.14-slim\nCMD ["uv", "run", "src/agent.py", "start"]\n'
        if manifest.endswith("toml")
        else 'FROM python:3.12\nCMD ["python", "src/agent.py", "start"]\n'
    )
    if project_dir != ".":
        # A monorepo's parent manifest must not replace the component's own environment.
        (source / "pyproject.toml").write_text(
            "[project]\nname='parent'\nversion='1'\n"
        )
    plan = resolve_environment_plan(
        source, _job(connector="livekit", with_secrets=True)
    )
    agent = next(p for p in plan.processes if p.name == "agent")
    assert agent.working_directory == project_dir
    assert "src/agent.py" in agent.run_command
    assert agent.run_command[-1:] == ["start"]
    assert agent.fixed_port == 8081
    if manifest.endswith("toml"):
        assert agent.build_commands[0] == [
            "uv",
            "sync",
            "--no-cache",
            "--python",
            "python3.14",
            "--no-dev",
        ]
        assert any("download-files" in command for command in agent.build_commands)
    else:
        assert any("requirements.txt" in command for command in agent.build_commands)


def test_repository_runtime_environment_is_sealed_into_control_process(
    tmp_path: Path,
) -> None:
    source = tmp_path / "voice-agent"
    source.mkdir()
    (source / "agent.py").write_text("print('agent')\n", encoding="utf-8")
    (source / "Dockerfile").write_text("FROM python:3.13\n", encoding="utf-8")
    (source / "pyproject.toml").write_text(
        "[project]\nname='agent'\nversion='1'\n", encoding="utf-8"
    )
    (source / "alk.yaml").write_text(
        'schema_version: "1"\nruntime:\n  environment:\n    HOTEL_TODAY: "2026-06-08"\n',
        encoding="utf-8",
    )
    authoring = _authoring(tmp_path)
    _write_voice_contract(authoring)

    bundle = author_bundle_v2(
        source=source,
        job=_job(connector="auto", with_secrets=True),
        authoring=authoring,
        output=tmp_path / "bundle",
    )

    agent = next(process for process in bundle.processes if process.name == "agent")
    assert agent.environment["HOTEL_TODAY"] == "2026-06-08"


def test_repository_runtime_environment_rejects_secrets(tmp_path: Path) -> None:
    source = tmp_path / "voice-agent"
    source.mkdir()
    (source / "agent.py").write_text("print('agent')\n", encoding="utf-8")
    (source / "alk.yaml").write_text(
        'schema_version: "1"\nruntime:\n  environment:\n    OPENAI_API_KEY: checked-in\n',
        encoding="utf-8",
    )
    authoring = _authoring(tmp_path)
    _write_voice_contract(authoring)

    with pytest.raises(RuntimeError, match="runtime_environment_secret_forbidden"):
        author_bundle_v2(
            source=source,
            job=_job(connector="auto", with_secrets=True),
            authoring=authoring,
            output=tmp_path / "bundle",
        )


@pytest.mark.parametrize("command", [None, ["python", "-m", "app.agent"]])
def test_callable_contract_compiles_repository_callback_adapter(
    tmp_path: Path, command: list[str] | None,
) -> None:
    source = tmp_path / "ava"
    app = source / "app"
    app.mkdir(parents=True)
    (app / "__init__.py").write_text("", encoding="utf-8")
    (app / "agent.py").write_text(
        "async def agent_callback(input):\n"
        "    return {'content': input.new_message['content']}\n\n"
        "if __name__ == '__main__':\n"
        "    print(input('you> '))\n",
        encoding="utf-8",
    )
    (source / "requirements.txt").write_text("agent-simulate\n", encoding="utf-8")
    authoring = _authoring(tmp_path)
    _write_callable_contract(authoring)
    if command:
        contract_path = authoring / "contract.json"
        contract = json.loads(contract_path.read_text())
        contract["runtime"]["command"] = command
        contract_path.write_text(json.dumps(contract))

    bundle = author_bundle_v2(
        source=source,
        job=_job(connector="auto", with_secrets=True),
        authoring=authoring,
        output=tmp_path / "bundle",
    )

    agent = next(process for process in bundle.processes if process.name == "agent")
    assert agent.source_origin == "bundle"
    assert agent.working_directory == "generated/source/agent"
    assert agent.build_commands[1][-1] == "requirements.txt"
    assert agent.run_command[:2] == [".venv/bin/python", "-c"]
    assert "ThreadingHTTPServer" in agent.run_command[2]
    assert agent.environment["ALK_CALLBACK_ENTRYPOINT"] == "app.agent:agent_callback"
    assert agent.environment["PORT"] == "{{PORT_agent}}"
    assert bundle.capabilities["target_http"].service == "agent"
    assert any(probe.capability == "target_http" for probe in bundle.readiness)
    preflight_bundle(
        tmp_path / "bundle",
        bundle,
        parallelism=1,
        secret_refs={
            alias: ref.purpose
            for alias, ref in _job(
                connector="auto", with_secrets=True
            ).agent.secret_refs.items()
        },
    )


def test_repository_callback_is_discovered_when_contract_omits_interface(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ava"
    app = source / "app"
    app.mkdir(parents=True)
    (app / "__init__.py").write_text("", encoding="utf-8")
    (app / "agent.py").write_text(
        "async def agent_callback(input):\n"
        "    return {'content': input.new_message['content']}\n",
        encoding="utf-8",
    )
    (source / "requirements.txt").write_text("agent-simulate\n", encoding="utf-8")
    authoring = _authoring(tmp_path)
    (authoring / "contract.json").write_text(
        json.dumps(
            {
                "modality": "chat",
                "runtime": {
                    "language": "python",
                    "interface": None,
                    "command": ["python", "-m", "app.agent"],
                },
            }
        ),
        encoding="utf-8",
    )

    bundle = author_bundle_v2(
        source=source,
        job=_job(connector="auto", with_secrets=True),
        authoring=authoring,
        output=tmp_path / "bundle",
    )

    agent = next(process for process in bundle.processes if process.name == "agent")
    assert agent.source_origin == "bundle"
    assert agent.working_directory == "generated/source/agent"
    assert agent.run_command[:2] == [".venv/bin/python", "-c"]
    assert agent.environment["ALK_CALLBACK_ENTRYPOINT"] == "app.agent:agent_callback"
    assert bundle.capabilities["target_http"].service == "agent"
    sealed_contract = json.loads(
        (tmp_path / "bundle" / "contract.json").read_text(encoding="utf-8")
    )
    assert sealed_contract["runtime"]["interface"] == {
        "health_path": "",
        "include_tools": True,
        "kind": "callable",
        "path": "",
        "protocol": "fi.alk",
    }


def test_callable_contract_rejects_missing_callback(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "agent.py").write_text("print('cli only')\n", encoding="utf-8")
    authoring = _authoring(tmp_path)
    _write_callable_contract(authoring)

    with pytest.raises(Exception, match="callback_entrypoint_missing"):
        author_bundle_v2(
            source=source,
            job=_job(connector="auto"),
            authoring=authoring,
            output=tmp_path / "bundle",
        )


@pytest.mark.parametrize(
    ("case", "connector", "packaging"),
    [
        ("uber-compose", "livekit", "compose"),
        ("packaged-chat", "http", "compose"),
        ("unpackaged-chat", "http", "generated_python"),
        ("frontdesk", "livekit", "dockerfile"),
        ("drive-thru", "livekit", "dockerfile"),
        ("hotel-receptionist", "livekit", "dockerfile"),
    ],
)
def test_six_supported_shapes_produce_preflight_clean_bundle(
    tmp_path: Path, case: str, connector: str, packaging: str
) -> None:
    source = tmp_path / case
    source.mkdir()
    if case == "uber-compose":
        (source / "agent" / "agent.py").parent.mkdir()
        (source / "agent" / "agent.py").write_text("print('agent')\n", encoding="utf-8")
        (source / "pyproject.toml").write_text(
            "[project]\nname='agent'\nversion='1'\n", encoding="utf-8"
        )
        tools = source / "tools-api"
        tools.mkdir()
        (tools / "agent.py").write_text("print('tools')\n", encoding="utf-8")
        (tools / "requirements.txt").write_text("\n", encoding="utf-8")
        (source / "compose.yml").write_text(
            """services:
  postgres:
    image: postgres:16
  tools-api:
    build: ./tools-api
    depends_on: {postgres: {condition: service_healthy}}
  agent:
    build: .
    depends_on: {tools-api: {condition: service_healthy}}
""",
            encoding="utf-8",
        )
    else:
        (source / "agent.py").write_text("print('agent')\n", encoding="utf-8")
        if case == "packaged-chat":
            (source / "Dockerfile").write_text("FROM python:3.12\n", encoding="utf-8")
            (source / "compose.yml").write_text(
                "services:\n  api:\n    build: .\n", encoding="utf-8"
            )
        elif packaging == "dockerfile":
            (source / "Dockerfile").write_text("FROM python:3.13\n", encoding="utf-8")
            (source / "pyproject.toml").write_text(
                "[project]\nname='agent'\nversion='1'\n", encoding="utf-8"
            )
        else:
            (source / "requirements.txt").write_text("\n", encoding="utf-8")

    job = _job(connector=connector, with_secrets=connector == "livekit")
    plan = resolve_environment_plan(source, job)
    assert plan.packaging == packaging
    if case == "uber-compose":
        proxy = next(
            process for process in plan.processes if process.name == "tool-proxy"
        )
        assert proxy.run_command == [sys.executable, "proxy.py"]
    output = tmp_path / "bundle"
    first = author_bundle_v2(
        source=source, job=job, authoring=_authoring(tmp_path), output=output
    )
    loaded = load_bundle_v2(output)
    assert loaded.digest == first.digest
    assert loaded.metadata["packaging"] == packaging
    assert loaded.provenance.source_digest
    if connector == "livekit":
        control = next(
            process
            for process in loaded.processes
            if process.name == loaded.runtime.control_service
        )
        dispatch_name = control.environment["LIVEKIT_AGENT_NAME"]
        assert "{{JOB_ID}}" in dispatch_name
        assert "{{WORLD_INDEX}}" in dispatch_name
        assert (
            control.environment["HARNESS_TOOL_TRACE"]
            == "{{WORLD_DIR}}/agent-tool-calls.jsonl"
        )
        assert control.started_check is not None
        assert control.started_check.log_marker == "registered worker"
    source_processes = [
        process
        for process in loaded.processes
        if hasattr(process, "environment") and process.name != "tool-proxy"
    ]
    assert source_processes
    assert all(
        process.environment.get("HARNESS_MODE") == "1" for process in source_processes
    )
    preflight_bundle(
        output,
        loaded,
        parallelism=1,
        secret_refs={
            alias: ref.purpose for alias, ref in job.agent.secret_refs.items()
        },
    )

    # The compiler is deterministic for identical source, authoring artifacts and job contract.
    second = author_bundle_v2(
        source=source, job=job, authoring=tmp_path / "authoring", output=output
    )
    assert second.digest == first.digest


def test_compose_multi_store_dependencies_are_rewired_from_declared_topology(
    tmp_path: Path,
) -> None:
    source = tmp_path / "multi-store"
    source.mkdir()
    (source / "agent.py").write_text("print('agent')\n", encoding="utf-8")
    (source / "requirements.txt").write_text("\n", encoding="utf-8")
    (source / "compose.yml").write_text(
        """services:
  postgres:
    image: postgres:16
  cache:
    image: redis:7-alpine
  agent:
    build: .
    environment:
      DATABASE_URL: postgresql://source:source@postgres:5432/app
      CACHE_URL: redis://cache:6379/2?decode_responses=true
      CACHE_HOST: cache
      PUBLIC_LABEL: unchanged
    depends_on:
      postgres: {condition: service_healthy}
      cache: {condition: service_healthy}
""",
        encoding="utf-8",
    )

    plan = resolve_environment_plan(source, _job(connector="http"))
    agent = next(process for process in plan.processes if process.name == "agent")

    assert agent.depends_on == ["world-db", "cache"]
    assert agent.environment == {
        "DATABASE_URL": "{{WORLD_DATABASE_URL}}",
        "CACHE_URL": "{{CACHE_URL}}/2?decode_responses=true",
        "CACHE_HOST": "{{HOST_cache}}",
        "PUBLIC_LABEL": "unchanged",
        "HARNESS_MODE": "1",
    }
    assert plan.capabilities["cache_redis"].service == "cache"
    assert plan.capabilities["cache_redis"].configuration_name == "CACHE_URL"
    assert plan.capabilities["target_http"].service == "agent"

    output = tmp_path / "bundle"
    manifest = author_bundle_v2(
        source=source,
        job=_job(connector="http"),
        authoring=_authoring(tmp_path),
        output=output,
    )
    preflight_bundle(output, manifest, parallelism=2, secret_refs={})


def test_fixed_multi_store_fixture_compiles_to_a_preflight_clean_bundle(
    tmp_path: Path,
) -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "harness"
        / "generic_multi_store_chat"
    )
    output = tmp_path / "bundle"

    manifest = author_bundle_v2(
        source=source,
        job=_job(connector="auto"),
        authoring=_authoring(tmp_path),
        output=output,
    )

    agent = next(process for process in manifest.processes if process.name == "agent")
    assert agent.depends_on == ["world-db", "cache"]
    assert agent.environment["DATABASE_URL"] == "{{WORLD_DATABASE_URL}}"
    assert agent.environment["CACHE_URL"] == "{{CACHE_URL}}/2"
    assert agent.environment["PORT"] == "{{PORT_agent}}"
    assert agent.fixed_port is None
    assert manifest.capabilities["target_http"].service == "agent"
    assert manifest.capabilities["cache_redis"].service == "cache"
    preflight_bundle(output, manifest, parallelism=2, secret_refs={})


def test_bundle_never_persists_resolved_secret(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "agent.py").write_text("print('ok')\n", encoding="utf-8")
    job = _job(connector="livekit", with_secrets=True)
    output = tmp_path / "bundle"
    author_bundle_v2(
        source=source, job=job, authoring=_authoring(tmp_path), output=output
    )
    manifest = (output / "manifest.json").read_text(encoding="utf-8")
    assert "livekit-key" not in manifest
    assert "target_provider" in manifest


def test_environment_backed_vapi_lifecycle_is_sealed_into_bundle_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vapi-source"
    source.mkdir()
    (source / "agent.py").write_text(
        "from fastapi import FastAPI\napp = FastAPI()\n", encoding="utf-8"
    )
    (source / "requirements.txt").write_text("fastapi==0.116.1\n", encoding="utf-8")
    (source / "alk.yaml").write_text(
        """
schema_version: "1"
provider:
  type: vapi
  scope: world
  process: agent
  public_capability: target_http
  event_path: /provider/events
  tool_path: /provider/tools
  required_secrets: [VAPI_API_KEY]
  provision: {command: [python, provider_target.py, provision]}
  destroy: {command: [python, provider_target.py, destroy]}
""",
        encoding="utf-8",
    )
    (source / "provider_target.py").write_text("pass\n", encoding="utf-8")
    job = HarnessJob.model_validate(
        {
            "job_id": "job-vapi-v2",
            "run_id": "run-vapi-v2",
            "execution": "hosted",
            "source": {"kind": "archive", "archive_artifact_id": "source-1"},
            "agent": {
                "connector": "vapi",
                "mode": "environment_backed",
                "config": {"lifecycle_manifest": "alk.yaml"},
                "secret_refs": {
                    "VAPI_API_KEY": {
                        "manager": "platform-vault",
                        "key": "vapi-key",
                        "purpose": "target_provider",
                    }
                },
            },
            "scenario_count": 1,
            "runtime": {
                "isolation": "dedicated_vm",
                "cpu_units": 2,
                "memory_mb": 4096,
                "parallelism": 1,
            },
        }
    )
    output = tmp_path / "vapi-bundle"
    authoring = _authoring(tmp_path)
    with sqlite3.connect(authoring / "world.sqlite") as database:
        database.execute("CREATE TABLE preferences (id TEXT PRIMARY KEY, value TEXT)")

    bundle = author_bundle_v2(
        source=source,
        job=job,
        authoring=authoring,
        output=output,
    )

    lifecycle = bundle.metadata["provider_lifecycle"]
    assert lifecycle["type"] == "vapi"
    assert lifecycle["required_secrets"] == ["VAPI_API_KEY"]
    assert lifecycle["public_capability"] == "target_http"
    preflight_bundle(
        output,
        bundle,
        parallelism=1,
        secret_refs={"VAPI_API_KEY": "target_provider"},
    )


def test_vapi_provider_import_is_sealed_with_detected_http_capability(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vapi-import-source"
    source.mkdir()
    (source / "agent.py").write_text(
        "from fastapi import FastAPI\napp = FastAPI()\n", encoding="utf-8"
    )
    (source / "requirements.txt").write_text("fastapi==0.116.1\n", encoding="utf-8")
    job = HarnessJob.model_validate(
        {
            "job_id": "job-vapi-import",
            "run_id": "run-vapi-import",
            "execution": "hosted",
            "source": {"kind": "archive", "archive_artifact_id": "source-1"},
            "agent": {
                "connector": "vapi",
                "mode": "provider_import",
                "config": {"assistant_id": "source-assistant"},
                "secret_refs": {
                    "VAPI_API_KEY": {
                        "manager": "platform-vault",
                        "key": "vapi-key",
                        "purpose": "target_provider",
                    }
                },
            },
            "scenario_count": 1,
            "runtime": {
                "isolation": "dedicated_vm",
                "cpu_units": 2,
                "memory_mb": 4096,
                "parallelism": 1,
            },
        }
    )
    output = tmp_path / "vapi-import-bundle"

    bundle = author_bundle_v2(
        source=source,
        job=job,
        authoring=_authoring(tmp_path),
        output=output,
    )

    imported = bundle.metadata["provider_import"]
    assert imported["type"] == "vapi"
    assert imported["source_target_id"] == "source-assistant"
    assert imported["public_capability"] == "target_http"
    preflight_bundle(
        output,
        bundle,
        parallelism=1,
        secret_refs={"VAPI_API_KEY": "target_provider"},
    )


def test_bundle_limits_authoring_scenarios_to_requested_count(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "agent.py").write_text("print('ok')\n", encoding="utf-8")
    authoring = _authoring(tmp_path)
    second = authoring / "scenarios" / "two"
    (second / "checks").mkdir(parents=True)
    (second / "scenario.json").write_text(
        json.dumps({"name": "two", "instruction": "Test two", "sub_goals": ["works"]}),
        encoding="utf-8",
    )
    (second / "setup.py").write_text(
        "def setup(world):\n    return None\n", encoding="utf-8"
    )
    (second / "ready.py").write_text(
        "def ready(world):\n    return None\n", encoding="utf-8"
    )
    (second / "checks" / "works.py").write_text(
        "def check(world, calls):\n    return None\n", encoding="utf-8"
    )

    output = tmp_path / "bundle"
    author_bundle_v2(
        source=source,
        job=_job(connector="http", scenario_count=1),
        authoring=authoring,
        output=output,
    )
    assert [path.name for path in (output / "scenarios").iterdir()] == ["one"]


def test_bundle_preserves_sqlite_scalar_types_and_boolean_values(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "agent.py").write_text("print('ok')\n", encoding="utf-8")
    authoring = _authoring(tmp_path)
    database = sqlite3.connect(authoring / "world.sqlite")
    try:
        database.execute(
            "CREATE TABLE payment_methods ("
            "id TEXT PRIMARY KEY, is_valid BOOLEAN, is_expired BOOLEAN, "
            "attempts INTEGER, score REAL)"
        )
        database.execute(
            "INSERT INTO payment_methods VALUES (?, ?, ?, ?, ?)",
            ("pm-1", True, False, 3, 0.75),
        )
        database.commit()
    finally:
        database.close()

    output = tmp_path / "bundle"
    author_bundle_v2(
        source=source,
        job=_job(connector="http"),
        authoring=authoring,
        output=output,
    )
    seed_sql = (output / "seed" / "world.sql").read_text(encoding="utf-8")
    assert (
        'CREATE TABLE IF NOT EXISTS "payment_methods" '
        '("id" text PRIMARY KEY, "is_valid" boolean, "is_expired" boolean, '
        '"attempts" bigint, "score" double precision);' in seed_sql
    )
    assert (
        'INSERT INTO "payment_methods" '
        '("id", "is_valid", "is_expired", "attempts", "score") '
        "VALUES ('pm-1', TRUE, FALSE, 3, 0.75);" in seed_sql
    )


def test_bundle_uses_contract_boolean_type_when_sqlite_erases_it(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "agent.py").write_text("print('ok')\n", encoding="utf-8")
    authoring = _authoring(tmp_path)
    (authoring / "contract.json").write_text(
        json.dumps(
            {
                "modality": "voice",
                "data_schema": {
                    "otp_codes": {
                        "phone": "TEXT PRIMARY KEY",
                        "issued_at": "TIMESTAMPTZ NOT NULL DEFAULT now()",
                        "attempts_left": "INT NOT NULL DEFAULT 3",
                        "verified": "BOOLEAN NOT NULL DEFAULT FALSE",
                    },
                    "market_config": {
                        "market": "TEXT PRIMARY KEY",
                        "available_products": "TEXT[] NOT NULL DEFAULT '{}'",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    database = sqlite3.connect(authoring / "world.sqlite")
    try:
        # This is how SQLite reports booleans from generated authoring worlds.
        database.execute(
            "CREATE TABLE otp_codes ("
            "phone TEXT PRIMARY KEY, issued_at TEXT, "
            "attempts_left INTEGER NOT NULL DEFAULT 3, "
            "verified INTEGER NOT NULL DEFAULT 0)"
        )
        database.execute(
            "INSERT INTO otp_codes VALUES (?, ?, ?, ?)",
            ("+14155550101", "2026-09-04T12:00:00Z", 3, 0),
        )
        database.execute(
            "CREATE TABLE market_config ("
            "market TEXT PRIMARY KEY, available_products TEXT NOT NULL DEFAULT '[]')"
        )
        database.execute(
            "INSERT INTO market_config VALUES (?, ?)",
            ("US-SF", '["uberx", "comfort"]'),
        )
        database.commit()
    finally:
        database.close()

    compiled_world = _sqlite_sql(
        authoring / "world.sqlite",
        contract_declarations=_contract_column_declarations(
            json.loads((authoring / "contract.json").read_text(encoding="utf-8"))
        ),
    )
    assert "\"available_products\" text[] NOT NULL DEFAULT '{}'" in compiled_world
    assert "VALUES ('US-SF', '{\"uberx\",\"comfort\"}');" in compiled_world

    output = tmp_path / "bundle"
    author_bundle_v2(
        source=source,
        job=_job(connector="http"),
        authoring=authoring,
        output=output,
    )

    seed_sql = (output / "seed" / "world.sql").read_text(encoding="utf-8")
    assert '"issued_at" timestamptz DEFAULT now()' in seed_sql
    assert '"attempts_left" bigint NOT NULL DEFAULT 3' in seed_sql
    assert '"verified" boolean NOT NULL DEFAULT FALSE' in seed_sql
    assert "'2026-09-04T12:00:00Z', 3, FALSE);" in seed_sql


def test_bundle_uses_widest_numeric_type_from_language_union(tmp_path: Path) -> None:
    authoring = _authoring(tmp_path)
    (authoring / "contract.json").write_text(
        json.dumps(
            {
                "modality": "chat",
                "data_schema": {"feedback": {"score": "int | float"}},
            }
        ),
        encoding="utf-8",
    )
    database = sqlite3.connect(authoring / "world.sqlite")
    try:
        database.execute("CREATE TABLE feedback (score INTEGER)")
        database.execute("INSERT INTO feedback VALUES (?)", (5.0,))
        database.commit()
    finally:
        database.close()

    compiled_world = _sqlite_sql(
        authoring / "world.sqlite",
        contract_declarations=_contract_column_declarations(
            json.loads((authoring / "contract.json").read_text(encoding="utf-8"))
        ),
    )

    assert '"score" double precision' in compiled_world


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ("bool, whether the rider is on the block list", "boolean"),
        ("number of items in the cart list", "double precision"),
    ],
)
def test_contract_scalar_type_ignores_container_words_in_prose(
    declaration: str, expected: str
) -> None:
    assert _contract_sql_type(declaration) == expected


def test_bundle_preserves_container_shape_from_descriptive_contract(
    tmp_path: Path,
) -> None:
    authoring = _authoring(tmp_path)
    (authoring / "contract.json").write_text(
        json.dumps(
            {
                "modality": "chat",
                "data_schema": {
                    "clients": {
                        "holdings": (
                            "list of dicts with ticker (str), qty (int), "
                            "cost_basis (float)"
                        )
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    database = sqlite3.connect(authoring / "world.sqlite")
    try:
        database.execute("CREATE TABLE clients (holdings TEXT)")
        database.execute(
            "INSERT INTO clients VALUES (?)",
            ('[{"ticker":"NVDA","qty":2,"cost_basis":100.5}]',),
        )
        database.commit()
    finally:
        database.close()

    compiled_world = _sqlite_sql(
        authoring / "world.sqlite",
        contract_declarations=_contract_column_declarations(
            json.loads((authoring / "contract.json").read_text(encoding="utf-8"))
        ),
    )

    assert '"holdings" jsonb' in compiled_world
    assert '"cost_basis": 100.5' in compiled_world
    assert '"ticker": "NVDA"' in compiled_world


def test_adopted_source_schema_applies_defaults_for_authored_nulls(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "db").mkdir(parents=True)
    (source / "agent.py").write_text("print('ok')\n", encoding="utf-8")
    (source / "db" / "schema.sql").write_text(
        "CREATE TABLE call_attempts ("
        "call_id TEXT PRIMARY KEY, "
        "room_name TEXT NOT NULL DEFAULT '', "
        "recording_url TEXT NOT NULL DEFAULT '', "
        "updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
        "optional_note TEXT);\n",
        encoding="utf-8",
    )
    authoring = _authoring(tmp_path)
    database = sqlite3.connect(authoring / "world.sqlite")
    try:
        database.execute(
            "CREATE TABLE call_attempts ("
            "call_id TEXT PRIMARY KEY, room_name TEXT, recording_url TEXT, "
            "updated_at TEXT, optional_note TEXT)"
        )
        database.execute(
            "INSERT INTO call_attempts VALUES (?, ?, ?, ?, ?)",
            ("call-1", None, None, None, None),
        )
        database.commit()
    finally:
        database.close()

    output = tmp_path / "bundle"
    author_bundle_v2(
        source=source,
        job=_job(connector="http"),
        authoring=authoring,
        output=output,
    )

    seed_sql = (output / "seed" / "world.sql").read_text(encoding="utf-8")
    assert 'INSERT INTO "call_attempts" ("call_id")' in seed_sql
    assert '"room_name", "recording_url", "updated_at", "optional_note"' not in seed_sql


def test_generic_pipeline_packages_source_schema_and_world_separately(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "db").mkdir(parents=True)
    (source / "agent.py").write_text("print('ok')\n", encoding="utf-8")
    (source / "db" / "schema.sql").write_text(
        "CREATE TABLE users (id TEXT PRIMARY KEY, tags TEXT[] NOT NULL);\n",
        encoding="utf-8",
    )
    authoring = _authoring(tmp_path)
    (authoring / "runtime-validation.json").write_text(
        '{"status":"certified"}\n', encoding="utf-8"
    )
    database = sqlite3.connect(authoring / "world.sqlite")
    try:
        database.execute("CREATE TABLE users (id TEXT, tags TEXT)")
        database.execute(
            "INSERT INTO users VALUES (?, ?)",
            ("user-1", '["priority"]'),
        )
        database.commit()
    finally:
        database.close()

    output = tmp_path / "bundle"
    manifest = author_bundle_v2(
        source=source,
        job=_job(connector="http", metadata={"generic_harness_v1": True}),
        authoring=authoring,
        output=output,
    )

    store = manifest.seed.stores[0]
    assert store.migrations == ["seed/source-schema.sql"]
    assert store.seed_files == ["seed/world.sqlite"]
    assert manifest.metadata["generic_harness"] == "v1"
    assert (tmp_path / "runtime-validation.json").read_text(encoding="utf-8") == (
        '{"status":"certified"}\n'
    )
    schema = (output / "seed" / "source-schema.sql").read_text(encoding="utf-8")
    assert "CREATE TABLE users" in schema
    assert "user-1" not in schema
    with sqlite3.connect(output / "seed" / "world.sqlite") as copied:
        assert copied.execute("SELECT tags FROM users").fetchone() == ('["priority"]',)
    preflight_bundle(output, manifest, parallelism=1, secret_refs={})


def test_documented_readme_schema_is_adopted_without_setup_or_fixture_sql(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text(
        """# Local setup

```sql
CREATE DATABASE ticketsdb;
\\c ticketsdb
CREATE TABLE tickets (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT
);
INSERT INTO tickets (title, description) VALUES ('Broken login', 'Fixture');
```

```sql
CREATE TABLE tickets (id BIGINT PRIMARY KEY, embedding vector(768));
```
""",
        encoding="utf-8",
    )

    artifacts = _source_schema_artifacts(source)

    assert len(artifacts) == 1
    assert artifacts[0].relative_path == "README.md#sql-1"
    assert "CREATE TABLE tickets" in artifacts[0].sql
    assert "title TEXT NOT NULL" in artifacts[0].sql
    assert "CREATE DATABASE" not in artifacts[0].sql
    assert "INSERT INTO" not in artifacts[0].sql
    assert "embedding" not in artifacts[0].sql


def test_documented_readme_schema_keeps_dependencies_across_sql_fences(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text(
        """# Local setup

```sql
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE TYPE ride_status AS ENUM ('requested', 'accepted');
CREATE SEQUENCE ride_number_seq;
CREATE TABLE rides (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status ride_status NOT NULL,
    ride_number BIGINT DEFAULT nextval('ride_number_seq')
);
```

```postgresql
CREATE TABLE riders (id UUID PRIMARY KEY);
ALTER TABLE rides ADD COLUMN rider_id UUID REFERENCES riders(id);
CREATE INDEX rides_rider_id_idx ON rides (rider_id);
INSERT INTO riders VALUES (gen_random_uuid());
```
""",
        encoding="utf-8",
    )

    artifacts = _source_schema_artifacts(source)

    assert [artifact.relative_path for artifact in artifacts] == [
        "README.md#sql-1",
        "README.md#sql-2",
    ]
    sql = "\n".join(artifact.sql for artifact in artifacts)
    assert "CREATE EXTENSION" in sql
    assert "CREATE TYPE ride_status" in sql
    assert "CREATE SEQUENCE ride_number_seq" in sql
    assert "CREATE TABLE rides" in sql
    assert "CREATE TABLE riders" in sql
    assert "ALTER TABLE rides" in sql
    assert "CREATE INDEX rides_rider_id_idx" in sql
    assert "INSERT INTO" not in sql


def test_docker_python_respects_project_requires_python(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile").write_text(
        "FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='agent'\nversion='1'\nrequires-python='>=3.10,<3.13'\n",
        encoding="utf-8",
    )

    assert _docker_python(tmp_path) == "python3.12"


@pytest.mark.parametrize("version", ["3.14", "3.14.2"])
@pytest.mark.parametrize("manifest", ["pyproject.toml", "requirements.txt", None])
def test_native_python_provisions_source_pin(tmp_path: Path, version: str, manifest: str | None) -> None:
    from fi.alk.harness.bundle_author_v2 import _plan_python

    (tmp_path / ".python-version").write_text(version + "\n")
    (tmp_path / "Dockerfile").write_text("FROM python:3.12\n")
    (tmp_path / "agent.py").write_text("print('hello')\n")
    if manifest:
        (tmp_path / manifest).write_text(
            "[project]\nname='agent'\nversion='1'\nrequires-python='>=3.11,<3.15'\n"
            if manifest == "pyproject.toml" else ""
        )
    process = _plan_python(
        tmp_path, name="agent", root=tmp_path, entry="agent.py", control=True,
        needs_secrets=False,
    )
    command = " ".join(process.build_commands[0])
    assert "uv " in command
    assert "--python" in command
    assert version in command
    if manifest != "pyproject.toml":
        assert process.run_command == [".venv/bin/python", "agent.py"]


def test_uv_sync_repairs_only_a_stale_lock_in_the_build_copy(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    binary = tmp_path / "bin"
    binary.mkdir()
    calls = tmp_path / "calls"
    uv = binary / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$UV_CALLS"\n'
        'case " $* " in\n'
        '  *" --locked "*) echo "The lockfile at uv.lock needs to be updated" >&2; exit 2;;\n'
        "  *) exit 0;;\n"
        "esac\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)

    completed = subprocess.run(
        _uv_sync_build_command(
            tmp_path, "python3.12", extra_arguments=["--all-packages", "--extra", "cloud"]
        ),
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{binary}:{os.environ['PATH']}",
            "UV_CALLS": str(calls),
        },
        check=False,
    )

    assert completed.returncode == 0
    attempts = calls.read_text(encoding="utf-8").splitlines()
    assert len(attempts) == 2
    assert "--locked" in attempts[0]
    assert "--locked" not in attempts[1]
    assert "--no-dev" in attempts[0]
    assert "--no-dev" in attempts[1]
    assert all("--all-packages --extra cloud" in attempt for attempt in attempts)


def test_uv_sync_does_not_reinterpret_other_install_failures(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    binary = tmp_path / "bin"
    binary.mkdir()
    calls = tmp_path / "calls"
    uv = binary / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$UV_CALLS"\n'
        'echo "Could not connect to package registry" >&2\n'
        "exit 7\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)

    completed = subprocess.run(
        _uv_sync_build_command(tmp_path, "python3.12"),
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{binary}:{os.environ['PATH']}",
            "UV_CALLS": str(calls),
        },
        check=False,
    )

    assert completed.returncode == 7
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 1


def test_generic_pipeline_prefers_canonical_world_ir(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "db").mkdir(parents=True)
    (source / "agent.py").write_text("print('ok')\n", encoding="utf-8")
    (source / "db" / "schema.sql").write_text(
        "CREATE TABLE users (id TEXT PRIMARY KEY);\n", encoding="utf-8"
    )
    authoring = _authoring(tmp_path)
    artifact_root = authoring / "generic-harness"
    artifact_root.mkdir()
    source_model = SourceModel.create(
        source_digest="sha256:" + source_fingerprint(source),
        engine="postgres",
        configuration_names=("APPLICATION_MODE",),
    )
    (artifact_root / "source-model.json").write_text(
        source_model.model_dump_json(), encoding="utf-8"
    )
    world = WorldIR.create(
        source_model_fingerprint=source_model.fingerprint,
        tables=(),
    )
    (artifact_root / "world-ir.json").write_text(
        world.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (artifact_root / "source-model.schema.json").write_text(
        '{"title":"SourceModel"}\n', encoding="utf-8"
    )
    (artifact_root / "world-ir.schema.json").write_text(
        '{"title":"WorldIR"}\n', encoding="utf-8"
    )
    # A stale compatibility artifact must not override the canonical semantic artifact.
    with sqlite3.connect(authoring / "world.sqlite") as database:
        database.execute("CREATE TABLE stale (id TEXT)")

    output = tmp_path / "bundle"
    manifest = author_bundle_v2(
        source=source,
        job=_job(connector="http", metadata={"generic_harness_v1": True}),
        authoring=authoring,
        output=output,
    )

    store = manifest.seed.stores[0]
    assert store.seed_files == ["seed/world-ir.json"]
    assert (output / "seed" / "world-ir.json").read_text() == (
        artifact_root / "world-ir.json"
    ).read_text()
    assert not (output / "seed" / "world.sqlite").exists()
    assert (output / "seed" / "source-model.json").read_bytes() == (
        artifact_root / "source-model.json"
    ).read_bytes()
    assert "generic-harness/source-model.json" in manifest.provenance.adopted_files
    assert any(item.path == "seed/source-model.json" for item in manifest.files)
    assert "generic-harness/world-ir.json" in manifest.provenance.adopted_files
    assert (
        "generic-harness/source-model.schema.json" in manifest.provenance.adopted_files
    )
    assert "generic-harness/world-ir.schema.json" in manifest.provenance.adopted_files
    assert (output / "contracts" / "source-model.schema.json").is_file()
    assert (output / "contracts" / "world-ir.schema.json").is_file()
    preflight_bundle(output, manifest, parallelism=1, secret_refs={})


@pytest.mark.parametrize("source_invariants", [CheckStatus.PASSED, CheckStatus.NOT_APPLICABLE])
def test_generic_pipeline_reuses_the_exact_runtime_certified_bundle(
    tmp_path: Path, source_invariants,
) -> None:
    source = tmp_path / "source"
    (source / "db").mkdir(parents=True)
    (source / "agent.py").write_text("print('ok')\n", encoding="utf-8")
    (source / "db" / "schema.sql").write_text(
        "CREATE TABLE users (id TEXT PRIMARY KEY);\n", encoding="utf-8"
    )
    authoring = _authoring(tmp_path)
    (authoring / "contract.json").write_text(
        json.dumps({"modality": "chat", "tools": []}), encoding="utf-8"
    )
    artifact_root = authoring / "generic-harness"
    artifact_root.mkdir()
    source_model = SourceModel.create(
        source_digest="sha256:" + source_fingerprint(source), engine="postgresql"
    )
    world = WorldIR.create(source_model_fingerprint=source_model.fingerprint, tables=())
    (artifact_root / "source-model.json").write_text(
        source_model.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (artifact_root / "world-ir.json").write_text(
        world.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    job = _job(connector="http", metadata={"generic_harness_v1": True})
    validated = tmp_path / "validated"
    manifest = author_bundle_v2(
        source=source, job=job, authoring=authoring, output=validated
    )
    shutil.copytree(validated, artifact_root / "certified-bundle")
    certificate = HarnessCertification.create(
        status=CertificationStatus.CERTIFIED,
        source=CertificationSource(
            digest=source_fingerprint(source), schema_hash=source_model.fingerprint
        ),
        authoring=CertificationAuthoring(
            contract_hash=_artifact_digest(authoring / "contract.json"),
            world_ir_hash=world.fingerprint,
            scenario_set_hash=_artifact_digest(authoring / "scenarios"),
        ),
        compiler=CertificationCompiler(
            version="test-compiler", bundle_digest=manifest.digest
        ),
        runtime=CertificationRuntime(snapshot="test", validation_attempts=1),
        checks=CertificationChecks(
            static=CheckStatus.PASSED,
            schema_and_seed=CheckStatus.PASSED,
            processes=CheckStatus.PASSED,
            chat_ingress=CheckStatus.PASSED,
            source_invariants=source_invariants,
            scenario_setup_ready="1/1",
            tool_contract="0/0",
            action_behavior="0/0",
            reset_equivalence=CheckStatus.PASSED,
            world_isolation=CheckStatus.PASSED,
        ),
        limitations=("No runtime SQL business tables; private state is not certified",),
        repairs=RepairHistory(
            decisions=(),
            results=(),
            compiler_candidates_used=0,
            environment_patches_used=0,
            scenario_patches_used=0,
        ),
    )
    (authoring / "runtime-validation.json").write_text(
        certificate.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    # Excluded interpreter caches must not replace the certified environment bytes.
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "generated.pyc").write_text(
        "def agent_callback(message):\n    return message\n", encoding="utf-8"
    )
    execution = tmp_path / "execution"
    reused = author_bundle_v2(
        source=source, job=job, authoring=authoring, output=execution
    )

    assert reused.digest == manifest.digest
    assert (tmp_path / "runtime-validation.json").is_file()
    assert load_bundle_v2(execution).digest == manifest.digest

    changed_world = WorldIR.create(
        source_model_fingerprint="sha256:" + "f" * 64, tables=()
    )
    (artifact_root / "world-ir.json").write_text(changed_world.model_dump_json())
    with pytest.raises(BundleAuthorError, match="certified_bundle_inputs_changed: world"):
        author_bundle_v2(
            source=source, job=job, authoring=authoring, output=tmp_path / "changed-world"
        )
    (artifact_root / "world-ir.json").write_text(world.model_dump_json())

    # A black-box hosted agent has no harness-owned schema/world. Those fingerprints are
    # intentionally synthetic and must not be compared to model-authored placeholder artifacts
    # when the certificate marks the corresponding checks not applicable.
    external_certificate = HarnessCertification.create(
        status=CertificationStatus.CERTIFIED,
        source=CertificationSource(
            digest=source_fingerprint(source), schema_hash="sha256:" + "a" * 64
        ),
        authoring=CertificationAuthoring(
            contract_hash=_artifact_digest(authoring / "contract.json"),
            world_ir_hash="sha256:" + "b" * 64,
            scenario_set_hash=_artifact_digest(authoring / "scenarios"),
        ),
        compiler=CertificationCompiler(
            version="external-provider-black-box", bundle_digest=manifest.digest
        ),
        runtime=CertificationRuntime(snapshot="test", validation_attempts=1),
        checks=CertificationChecks(
            static=CheckStatus.PASSED,
            schema_and_seed=CheckStatus.NOT_APPLICABLE,
            processes=CheckStatus.PASSED,
            chat_ingress=CheckStatus.PASSED,
            source_invariants=CheckStatus.NOT_APPLICABLE,
            scenario_setup_ready="1/1",
            tool_contract="0/0",
            action_behavior="0/0",
            reset_equivalence=CheckStatus.NOT_APPLICABLE,
            world_isolation=CheckStatus.NOT_APPLICABLE,
        ),
        repairs=RepairHistory(
            decisions=(),
            results=(),
            compiler_candidates_used=0,
            environment_patches_used=0,
            scenario_patches_used=0,
        ),
        limitations=("provider state is external",),
    )
    (authoring / "runtime-validation.json").write_text(
        external_certificate.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    external_execution = tmp_path / "external-execution"
    external_reused = author_bundle_v2(
        source=source, job=job, authoring=authoring, output=external_execution
    )
    assert external_reused.digest == manifest.digest

    from fi.alk.harness.certification import CertificationGateError

    (source / "build").mkdir()
    (source / "build" / "agent.py").write_text("print('new source')\n")
    with pytest.raises(CertificationGateError, match="source digest"):
        author_bundle_v2(
            source=source, job=job, authoring=authoring, output=tmp_path / "changed-source"
        )


def test_generic_pipeline_requires_source_schema_and_world(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "agent.py").write_text("print('ok')\n", encoding="utf-8")
    authoring = _authoring(tmp_path)

    with pytest.raises(BundleAuthorError, match="source_schema_required"):
        author_bundle_v2(
            source=source,
            job=_job(connector="http", metadata={"generic_harness_v1": True}),
            authoring=authoring,
            output=tmp_path / "bundle",
        )


def test_generic_pipeline_compiles_harness_schema_for_in_process_store(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "agent.py").write_text("print('ok')\n", encoding="utf-8")
    authoring = _authoring(tmp_path)
    (authoring / "contract.json").write_text(
        json.dumps(
            {
                "modality": "chat",
                "data_store": {"kind": "in_process"},
                "data_schema": {"accounts": {"id": "TEXT PRIMARY KEY"}},
            }
        ),
        encoding="utf-8",
    )
    database = sqlite3.connect(authoring / "world.sqlite")
    try:
        database.execute("CREATE TABLE accounts (id TEXT PRIMARY KEY, status TEXT)")
        database.execute("INSERT INTO accounts VALUES (?, ?)", ("acct-1", "active"))
        database.commit()
    finally:
        database.close()

    output = tmp_path / "bundle"
    manifest = author_bundle_v2(
        source=source,
        job=_job(connector="http", metadata={"generic_harness_v1": True}),
        authoring=authoring,
        output=output,
    )

    store = manifest.seed.stores[0]
    assert store.migrations == ["seed/source-schema.sql"]
    assert store.seed_files == ["seed/world.sqlite"]
    schema = (output / "seed" / "source-schema.sql").read_text(encoding="utf-8")
    assert 'CREATE TABLE IF NOT EXISTS "accounts"' in schema
    assert "acct-1" not in schema
    with sqlite3.connect(output / "seed" / "world.sqlite") as copied:
        assert copied.execute("SELECT status FROM accounts").fetchone() == ("active",)


def test_generic_pipeline_accepts_explicitly_tool_free_agent(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "agent.py").write_text("print('ok')\n", encoding="utf-8")
    authoring = _authoring(tmp_path)
    (authoring / "contract.json").write_text(
        json.dumps({"modality": "voice", "tools": []}),
        encoding="utf-8",
    )
    with sqlite3.connect(authoring / "world.sqlite") as database:
        database.execute("CREATE TABLE session_state (id TEXT PRIMARY KEY)")

    manifest = author_bundle_v2(
        source=source,
        job=_job(connector="livekit", metadata={"generic_harness_v1": True}),
        authoring=authoring,
        output=tmp_path / "bundle",
    )

    assert manifest.metadata["generic_harness"] == "v1"
    schema = (tmp_path / "bundle" / "seed" / "source-schema.sql").read_text()
    assert 'CREATE TABLE IF NOT EXISTS "session_state"' in schema


def test_generic_pipeline_still_requires_schema_for_declared_postgres(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "agent.py").write_text("print('ok')\n", encoding="utf-8")
    authoring = _authoring(tmp_path)
    (authoring / "contract.json").write_text(
        json.dumps({"modality": "chat", "data_store": {"kind": "postgres"}}),
        encoding="utf-8",
    )
    database = sqlite3.connect(authoring / "world.sqlite")
    database.close()

    with pytest.raises(BundleAuthorError, match="source_schema_required"):
        author_bundle_v2(
            source=source,
            job=_job(connector="http", metadata={"generic_harness_v1": True}),
            authoring=authoring,
            output=tmp_path / "bundle",
        )


def test_bundle_preserves_sqlite_unique_constraints_for_upserts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "agent.py").write_text("print('ok')\n", encoding="utf-8")
    authoring = _authoring(tmp_path)
    database = sqlite3.connect(authoring / "world.sqlite")
    try:
        database.execute(
            "CREATE TABLE users (rider_id TEXT PRIMARY KEY, phone TEXT UNIQUE NOT NULL)"
        )
        database.execute(
            "INSERT INTO users (rider_id, phone) VALUES (?, ?)",
            ("rider-1", "+14155550101"),
        )
        database.commit()
    finally:
        database.close()

    output = tmp_path / "bundle"
    author_bundle_v2(
        source=source,
        job=_job(connector="http"),
        authoring=authoring,
        output=output,
    )

    seed_sql = (output / "seed" / "world.sql").read_text(encoding="utf-8")
    assert (
        'CREATE TABLE IF NOT EXISTS "users" '
        '("rider_id" text PRIMARY KEY, "phone" text NOT NULL, UNIQUE ("phone"));'
        in seed_sql
    )


def test_bundle_preserves_composite_sqlite_primary_key(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "agent.py").write_text("print('ok')\n", encoding="utf-8")
    authoring = _authoring(tmp_path)
    database = sqlite3.connect(authoring / "world.sqlite")
    try:
        database.execute(
            "CREATE TABLE performance (client_id TEXT, period TEXT, value REAL, "
            "PRIMARY KEY (client_id, period))"
        )
        database.execute(
            "INSERT INTO performance (client_id, period, value) VALUES (?, ?, ?)",
            ("CLI-01", "YTD", 0.12),
        )
        database.commit()
    finally:
        database.close()

    output = tmp_path / "bundle"
    author_bundle_v2(
        source=source,
        job=_job(connector="http"),
        authoring=authoring,
        output=output,
    )

    seed_sql = (output / "seed" / "world.sql").read_text(encoding="utf-8")
    assert (
        'CREATE TABLE IF NOT EXISTS "performance" '
        '("client_id" text, "period" text, "value" double precision, '
        'PRIMARY KEY ("client_id", "period"));' in seed_sql
    )


def test_bundle_promotes_sqlite_json_text_to_postgres_jsonb(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "agent.py").write_text("print('ok')\n", encoding="utf-8")
    authoring = _authoring(tmp_path)
    database = sqlite3.connect(authoring / "world.sqlite")
    try:
        database.execute(
            "CREATE TABLE users (id TEXT PRIMARY KEY, accessibility_needs TEXT, note TEXT)"
        )
        database.execute(
            "INSERT INTO users VALUES (?, ?, ?)",
            ("rider-1", json.dumps(["wheelchair"]), "ordinary text"),
        )
        database.execute(
            "INSERT INTO users VALUES (?, ?, ?)",
            ("rider-2", json.dumps([]), "123"),
        )
        database.commit()
    finally:
        database.close()

    output = tmp_path / "bundle"
    author_bundle_v2(
        source=source,
        job=_job(connector="http"),
        authoring=authoring,
        output=output,
    )
    seed_sql = (output / "seed" / "world.sql").read_text(encoding="utf-8")
    assert (
        'CREATE TABLE IF NOT EXISTS "users" '
        '("id" text PRIMARY KEY, "accessibility_needs" jsonb, "note" text);' in seed_sql
    )
    assert "'[\"wheelchair\"]'" in seed_sql
    assert "'[]'" in seed_sql


def test_bundle_combines_schema_with_frozen_store_rows(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "agent.py").write_text("print('ok')\n", encoding="utf-8")
    authoring = _authoring(tmp_path)
    (authoring / "schema.sql").write_text(
        "SET search_path = '';\nCREATE TABLE public.users "
        "(id text PRIMARY KEY, tags text[], active boolean);\n",
        encoding="utf-8",
    )
    (authoring / "store.json").write_text(
        json.dumps(
            {
                "rows": {
                    "users": [
                        {"id": "rider-1", "tags": ["priority", "voice"], "active": True}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "bundle"
    manifest = author_bundle_v2(
        source=source,
        job=_job(connector="http"),
        authoring=authoring,
        output=output,
    )
    seed_sql = (output / "seed" / "world.sql").read_text(encoding="utf-8")
    assert "CREATE TABLE public.users" in seed_sql
    assert 'INSERT INTO public."users"' in seed_sql
    assert 'jsonb_populate_recordset(NULL::public."users"' in seed_sql
    assert '"priority"' in seed_sql
    assert "session_replication_role" not in seed_sql
    assert "EXCEPTION WHEN foreign_key_violation" in seed_sql
    assert "seed_dependency_unresolved" in seed_sql
    assert "schema.sql" in manifest.provenance.adopted_files
    assert "store.json" in manifest.provenance.adopted_files


def test_bundle_uses_source_schema_when_fresh_contract_omits_a_column(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "db").mkdir(parents=True)
    (source / "agent.py").write_text("print('ok')\n", encoding="utf-8")
    (source / "db" / "schema.sql").write_text(
        "CREATE TABLE bookings ("
        "booking_ref TEXT PRIMARY KEY, rider_id TEXT, phone_verified BOOLEAN);\n",
        encoding="utf-8",
    )
    # The model-authored representation is deliberately lossy: this reproduces the dev failure
    # where a fresh contract omitted booking_ref even though the submitted repository required it.
    authoring = _authoring(tmp_path)
    (authoring / "contract.json").write_text(
        json.dumps(
            {
                "modality": "voice",
                "data_store": {"schema_from": "db/schema.sql"},
                "data_schema": {
                    "bookings": {
                        "rider_id": "TEXT",
                        "phone_verified": "BOOLEAN",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    database = sqlite3.connect(authoring / "world.sqlite")
    try:
        database.execute(
            "CREATE TABLE bookings (rider_id TEXT, phone_verified INTEGER)"
        )
        database.execute("INSERT INTO bookings VALUES (?, ?)", ("rider-1", 1))
        database.commit()
    finally:
        database.close()

    output = tmp_path / "bundle"
    manifest = author_bundle_v2(
        source=source,
        job=_job(connector="http"),
        authoring=authoring,
        output=output,
    )

    seed_sql = (output / "seed" / "world.sql").read_text(encoding="utf-8")
    assert "booking_ref TEXT PRIMARY KEY" in seed_sql
    assert 'CREATE TABLE IF NOT EXISTS "bookings"' not in seed_sql
    assert (
        'INSERT INTO "bookings" ("rider_id", "phone_verified") '
        "VALUES (''''rider-1'''', TRUE);" in seed_sql
    )
    assert "source/db/schema.sql" in manifest.provenance.adopted_files
    assert "world.sqlite" in manifest.provenance.adopted_files


def test_bundle_discovers_compose_mounted_schema_without_model_hint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "database").mkdir(parents=True)
    (source / "agent.py").write_text("print('ok')\n", encoding="utf-8")
    (source / "database" / "001-ddl.sql").write_text(
        "CREATE TABLE accounts (id TEXT PRIMARY KEY, status TEXT NOT NULL);\n",
        encoding="utf-8",
    )
    (source / "database" / "002-seed.sql").write_text(
        "INSERT INTO accounts VALUES ('stale', 'stale');\n",
        encoding="utf-8",
    )
    (source / "compose.yml").write_text(
        "services:\n"
        "  postgres:\n"
        "    image: postgres:16\n"
        "    volumes:\n"
        "      - ./database/001-ddl.sql:/docker-entrypoint-initdb.d/01.sql:ro\n"
        "      - ./database/002-seed.sql:/docker-entrypoint-initdb.d/02.sql:ro\n"
        "  agent:\n"
        "    build: .\n",
        encoding="utf-8",
    )
    authoring = _authoring(tmp_path)
    database = sqlite3.connect(authoring / "world.sqlite")
    try:
        database.execute("CREATE TABLE accounts (id TEXT PRIMARY KEY, status TEXT)")
        database.execute("INSERT INTO accounts VALUES (?, ?)", ("fresh", "active"))
        database.commit()
    finally:
        database.close()

    output = tmp_path / "bundle"
    manifest = author_bundle_v2(
        source=source,
        job=_job(connector="http"),
        authoring=authoring,
        output=output,
    )

    seed_sql = (output / "seed" / "world.sql").read_text(encoding="utf-8")
    assert "CREATE TABLE accounts" in seed_sql
    assert "''''fresh'''', ''''active''''" in seed_sql
    assert "'stale', 'stale'" not in seed_sql
    assert "source/database/001-ddl.sql" in manifest.provenance.adopted_files


def test_bundle_rejects_missing_declared_source_schema(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "agent.py").write_text("print('ok')\n", encoding="utf-8")
    authoring = _authoring(tmp_path)
    (authoring / "contract.json").write_text(
        json.dumps(
            {
                "modality": "voice",
                "data_store": {"schema_from": "db/missing-schema.sql"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BundleAuthorError, match="source_schema_missing"):
        author_bundle_v2(
            source=source,
            job=_job(connector="http"),
            authoring=authoring,
            output=tmp_path / "bundle",
        )


# --- C1 (world-port-model v1.3): authoring the parallelism seams (Track A′) ------------------


def _livekit_compose_with_command_fixed_tools(source: Path) -> None:
    """A voice bundle whose tools-api pins its port in a Dockerfile exec-form CMD — the
    command-fixed shape C1 §1 names as the concrete `tools-api` rewrite target."""
    source.mkdir()
    agent = source / "agent"
    agent.mkdir()
    (agent / "agent.py").write_text("print('agent')\n", encoding="utf-8")
    (source / "pyproject.toml").write_text(
        "[project]\nname='agent'\nversion='1'\n", encoding="utf-8"
    )
    tools = source / "tools-api"
    tools.mkdir()
    (tools / "main.py").write_text("print('tools')\n", encoding="utf-8")
    (tools / "pyproject.toml").write_text(
        "[project]\nname='tools'\nversion='1'\n", encoding="utf-8"
    )
    (tools / "Dockerfile").write_text(
        'FROM python:3.12\nCMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]\n',
        encoding="utf-8",
    )
    (source / "compose.yml").write_text(
        """services:
  postgres:
    image: postgres:16
  tools-api:
    build: ./tools-api
    depends_on: {postgres: {condition: service_healthy}}
  agent:
    build: .
    depends_on: {tools-api: {condition: service_healthy}}
""",
        encoding="utf-8",
    )


def test_command_fixed_tools_api_is_rewritten_consumable(tmp_path: Path) -> None:
    # C1 §1 / checklist 2: mark tools-api consumable AND rewrite its verbatim-exec'd run_command
    # into the ONE valid wiring — `$FI_TOOLS_PORT` inside `sh -c`, fed by an env value carrying
    # `{{PORT_tools-api}}`. A bare token in argv would not work (run_command is never rendered).
    source = tmp_path / "voice-compose"
    _livekit_compose_with_command_fixed_tools(source)
    plan = resolve_environment_plan(
        source, _job(connector="livekit", with_secrets=True)
    )

    tools = next(p for p in plan.processes if p.name == "tools-api")
    assert tools.fixed_port == 8080
    assert tools.fixed_port_consumable is True
    assert tools.run_command == [
        "sh",
        "-c",
        "uvicorn main:app --host 0.0.0.0 --port $FI_TOOLS_PORT",
    ]
    assert tools.environment["FI_TOOLS_PORT"] == "{{PORT_tools-api}}"


def test_command_fixed_tools_api_stays_preflight_clean_and_round_trips(
    tmp_path: Path,
) -> None:
    source = tmp_path / "voice-compose"
    _livekit_compose_with_command_fixed_tools(source)
    job = _job(connector="livekit", with_secrets=True)
    output = tmp_path / "bundle"
    first = author_bundle_v2(
        source=source, job=job, authoring=_authoring(tmp_path), output=output
    )
    loaded = load_bundle_v2(output)
    assert loaded.digest == first.digest
    tools = next(p for p in loaded.processes if p.name == "tools-api")
    assert tools.fixed_port_consumable is True
    preflight_bundle(
        output,
        loaded,
        parallelism=1,
        secret_refs={
            alias: ref.purpose for alias, ref in job.agent.secret_refs.items()
        },
    )


def test_tools_api_without_a_port_command_stays_non_consumable(tmp_path: Path) -> None:
    # Honest wiring (C1 §1): a tools-api whose command carries no `--port` literal cannot be
    # wired to `$FI_TOOLS_PORT`, so authoring MUST NOT flag it consumable (that would be the
    # exact `fixed_port_consumable_unwired` lie). It stays code-fixed and degrades at W>1.
    source = tmp_path / "uber-compose"
    source.mkdir()
    (source / "agent" / "agent.py").parent.mkdir()
    (source / "agent" / "agent.py").write_text("print('agent')\n", encoding="utf-8")
    (source / "pyproject.toml").write_text(
        "[project]\nname='agent'\nversion='1'\n", encoding="utf-8"
    )
    tools = source / "tools-api"
    tools.mkdir()
    (tools / "agent.py").write_text("print('tools')\n", encoding="utf-8")
    (tools / "requirements.txt").write_text("\n", encoding="utf-8")
    (source / "compose.yml").write_text(
        """services:
  postgres:
    image: postgres:16
  tools-api:
    build: ./tools-api
    depends_on: {postgres: {condition: service_healthy}}
  agent:
    build: .
    depends_on: {tools-api: {condition: service_healthy}}
""",
        encoding="utf-8",
    )
    plan = resolve_environment_plan(
        source, _job(connector="livekit", with_secrets=True)
    )
    tools_proc = next(p for p in plan.processes if p.name == "tools-api")
    assert tools_proc.fixed_port == 8080
    assert tools_proc.fixed_port_consumable is False
    assert "FI_TOOLS_PORT" not in tools_proc.environment


def _livekit_compose_single_api_worker(source: Path) -> None:
    """A voice bundle whose ONLY source service is `api` (no separate agent): the LiveKit control
    worker is ALSO a command-fixed HTTP server. Track A′ D37: this ONE process would receive both
    FI_WORKER_HEALTH_PORT and FI_TOOLS_PORT bound to the SAME `{{PORT_api}}` token."""
    source.mkdir()
    api = source / "api"
    api.mkdir()
    (api / "agent.py").write_text("print('api')\n", encoding="utf-8")
    (api / "pyproject.toml").write_text(
        "[project]\nname='api'\nversion='1'\n", encoding="utf-8"
    )
    (api / "Dockerfile").write_text(
        'FROM python:3.12\nCMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]\n',
        encoding="utf-8",
    )
    (source / "compose.yml").write_text(
        """services:
  postgres:
    image: postgres:16
  api:
    build: ./api
    depends_on: {postgres: {condition: service_healthy}}
""",
        encoding="utf-8",
    )


def test_conflated_worker_and_http_server_stays_non_consumable(tmp_path: Path) -> None:
    # Track A′ D37: a single process that is BOTH the knob-bearing LiveKit control worker AND a
    # consumable HTTP server would get FI_WORKER_HEALTH_PORT and FI_TOOLS_PORT bound to the SAME
    # `{{PORT_api}}` token -> the worker health server and the HTTP server collide on one port at
    # ANY W. Such a conflated process MUST NOT be flagged consumable; it stays code-fixed and
    # degrades to W=1 honestly (at W=1 the default health port does not collide).
    source = tmp_path / "conflated-voice"
    _livekit_compose_single_api_worker(source)
    plan = resolve_environment_plan(
        source, _job(connector="livekit", with_secrets=True)
    )

    api = next(p for p in plan.processes if p.name == "api")
    # It IS the knob-bearing control worker...
    assert api.environment["FI_WORKER_HEALTH_PORT"] == "{{PORT_api}}"
    # ...so it is NOT flagged consumable and carries NO FI_TOOLS_PORT rewrite (no port collision).
    assert api.fixed_port == 8080
    assert api.fixed_port_consumable is False
    assert "FI_TOOLS_PORT" not in api.environment


def test_normal_separate_tools_api_is_still_consumable(tmp_path: Path) -> None:
    # Track A′ D37 anti-regression: the NORMAL topology (control=agent + a SEPARATE tools-api) is
    # unaffected — the separate tools-api still gets the consumable rewrite (it is not the
    # knob-bearing worker), and the agent control worker is not an HTTP server at all.
    source = tmp_path / "voice-compose"
    _livekit_compose_with_command_fixed_tools(source)
    plan = resolve_environment_plan(
        source, _job(connector="livekit", with_secrets=True)
    )

    tools = next(p for p in plan.processes if p.name == "tools-api")
    assert tools.fixed_port_consumable is True
    assert tools.environment["FI_TOOLS_PORT"] == "{{PORT_tools-api}}"
    assert "FI_WORKER_HEALTH_PORT" not in tools.environment
    control = next(p for p in plan.processes if p.name == "agent")
    assert control.environment["FI_WORKER_HEALTH_PORT"] == "{{PORT_agent}}"
    assert "FI_TOOLS_PORT" not in control.environment


def test_livekit_worker_carries_the_worker_knob_env(tmp_path: Path) -> None:
    # C1 §4: FI_WORKER_HEALTH_PORT is authored UNCONDITIONALLY into every LiveKit-worker process,
    # each fed its OWN `{{PORT_<name>}}`. Its presence IS the knob-bearing mark; it is the only
    # worker knob authored -- FI_LOAD_THRESHOLD / FI_NUM_IDLE_PROCESSES are hardcoded by the
    # harness's own sitecustomize shim at worker start instead, so there is nothing to author.
    source = tmp_path / "voice-compose"
    _livekit_compose_with_command_fixed_tools(source)
    plan = resolve_environment_plan(
        source, _job(connector="livekit", with_secrets=True)
    )

    control = next(p for p in plan.processes if p.name == "agent")
    assert control.environment["FI_WORKER_HEALTH_PORT"] == "{{PORT_agent}}"
    assert "FI_LOAD_THRESHOLD" not in control.environment
    assert "FI_NUM_IDLE_PROCESSES" not in control.environment
    # D32 / C3 §4.5: the dispatch-ack opt-in does NOT belong on the agent-under-test child. The
    # engine reads it from the GUEST MAIN PROCESS env (armed by `hosted_entrypoint`), not from the
    # spawned worker's environment, so the worker-knob authoring must not carry it.
    assert "FI_HOSTED_DISPATCH_ACK" not in control.environment
    # LIVEKIT_AGENT_NAME carries BOTH the job-id and world-index (C1 item 4 / §3).
    dispatch = control.environment["LIVEKIT_AGENT_NAME"]
    assert "{{JOB_ID}}" in dispatch and "{{WORLD_INDEX}}" in dispatch
    # The non-worker tools-api process must NOT be marked knob-bearing.
    tools = next(p for p in plan.processes if p.name == "tools-api")
    assert "FI_WORKER_HEALTH_PORT" not in tools.environment
    assert "FI_HOSTED_DISPATCH_ACK" not in tools.environment


def test_generated_python_livekit_worker_carries_the_worker_knob_env(
    tmp_path: Path,
) -> None:
    # The non-compose (generated_python) lane authors the same knobs onto its single worker.
    source = tmp_path / "voice-agent"
    source.mkdir()
    (source / "agent.py").write_text("print('agent')\n", encoding="utf-8")
    plan = resolve_environment_plan(
        source, _job(connector="livekit", with_secrets=True)
    )

    control = next(p for p in plan.processes if p.name == "agent")
    assert control.environment["FI_WORKER_HEALTH_PORT"] == "{{PORT_agent}}"
    assert "FI_LOAD_THRESHOLD" not in control.environment
    assert "FI_NUM_IDLE_PROCESSES" not in control.environment
    # D32 / C3 §4.5: the flag is guest-main-only, never authored onto the agent-under-test child.
    assert "FI_HOSTED_DISPATCH_ACK" not in control.environment
    dispatch = control.environment["LIVEKIT_AGENT_NAME"]
    assert "{{JOB_ID}}" in dispatch and "{{WORLD_INDEX}}" in dispatch


def test_non_livekit_control_has_no_worker_knob_env(tmp_path: Path) -> None:
    # The knobs are LiveKit-worker-only; a plain HTTP agent is not knob-bearing.
    source = tmp_path / "chat-agent"
    source.mkdir()
    (source / "agent.py").write_text("print('agent')\n", encoding="utf-8")
    plan = resolve_environment_plan(source, _job(connector="http"))

    control = next(p for p in plan.processes if p.name == "agent")
    assert "FI_WORKER_HEALTH_PORT" not in control.environment
    assert "FI_LOAD_THRESHOLD" not in control.environment
    assert "FI_HOSTED_DISPATCH_ACK" not in control.environment


def test_chat_command_runtime_compiles_generic_subprocess_bridge(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "requirements.txt").write_text("\n", encoding="utf-8")
    (source / "main.py").write_text("print('real target output')\n", encoding="utf-8")
    authoring = _authoring(tmp_path)
    interface = {"kind": "command", "protocol": "fi.alk"}
    _write_chat_contract(authoring, command=["python", "main.py"], interface=interface)
    with sqlite3.connect(authoring / "world.sqlite") as database:
        database.execute("CREATE TABLE fixture (id TEXT PRIMARY KEY)")

    output = tmp_path / "bundle"
    manifest = author_bundle_v2(
        source=source,
        job=_job(
            connector="auto",
            metadata={"generic_harness_v1": True},
        ),
        authoring=authoring,
        output=output,
    )

    agent = next(process for process in manifest.processes if process.name == "agent")
    assert agent.source_origin == "bundle"
    assert agent.working_directory == "generated/source/agent"
    assert agent.environment["ALK_SUBPROCESS_COMMAND"] == json.dumps(
        [".venv/bin/python", "main.py"]
    )
    assert "ThreadingHTTPServer" in agent.run_command[-1]
    assert manifest.capabilities["target_http"].service == "agent"
    sealed_contract = json.loads((output / "contract.json").read_text(encoding="utf-8"))
    assert sealed_contract["runtime"]["interface"] == interface
    preflight_bundle(output, manifest, parallelism=1, secret_refs={})


@pytest.mark.parametrize(
    "source_text",
    [
        "print('target')\n",
        "from http.server import HTTPServer, BaseHTTPRequestHandler\n"
        "HTTPServer(('127.0.0.1', 9119), BaseHTTPRequestHandler).serve_forever()\n",
    ],
)
@pytest.mark.parametrize("interface_kind", [None, "callable"])
def test_chat_command_without_supported_interface_requires_runtime_repair(
    tmp_path: Path, source_text: str, interface_kind: str | None,
) -> None:
    from fi.alk.harness.authoring_runtime_validation import (
        RuntimeValidationError,
        _fallback_diagnostic,
        _needs_runtime_plan_repair,
    )

    source = tmp_path / "source"
    source.mkdir()
    (source / "requirements.txt").write_text("\n", encoding="utf-8")
    (source / "main.py").write_text(source_text, encoding="utf-8")
    authoring = _authoring(tmp_path)
    _write_chat_contract(
        authoring,
        command=["python", "main.py"],
        interface={
            "kind": interface_kind,
            "protocol": "fi.alk",
            "path": "",
            "health_path": "",
            "include_tools": True,
        } if interface_kind else None,
    )
    with sqlite3.connect(authoring / "world.sqlite") as database:
        database.execute("CREATE TABLE fixture (id TEXT PRIMARY KEY)")

    output = tmp_path / "bundle"
    with pytest.raises(
        BundleAuthorError, match="generated_runtime_entrypoint_ambiguous"
    ) as failure:
        author_bundle_v2(
            source=source,
            job=_job(connector="auto", metadata={"generic_harness_v1": True}),
            authoring=authoring,
            output=output,
        )
    diagnostic = _fallback_diagnostic(
        RuntimeValidationError("environment", str(failure.value))
    )
    assert diagnostic.code == "generated_runtime_plan_invalid"
    assert "chat" in diagnostic.redacted_message
    assert _needs_runtime_plan_repair((diagnostic,))
    assert not (output / "contract.json").exists()


def test_declared_langgraph_graph_compiles_without_agent_script(tmp_path: Path) -> None:
    source = tmp_path / "source"
    graph_source = source / "src" / "example" / "graph.py"
    graph_source.parent.mkdir(parents=True)
    graph_source.write_text("graph = object()\n", encoding="utf-8")
    (source / "pyproject.toml").write_text(
        '[project]\nname = "example"\nversion = "0.1"\n', encoding="utf-8"
    )
    (source / "langgraph.json").write_text(
        json.dumps({"graphs": {"Example": "./src/example/graph.py:graph"}}),
        encoding="utf-8",
    )
    (source / ".env.example").write_text("UNUSED_PROVIDER_API_KEY=your-key\n")
    authoring = _authoring(tmp_path)
    _write_chat_contract(authoring, command=[])
    with sqlite3.connect(authoring / "world.sqlite") as database:
        database.execute("CREATE TABLE fixture (id TEXT PRIMARY KEY)")

    output = tmp_path / "bundle"
    manifest = author_bundle_v2(
        source=source,
        job=_job(connector="auto", metadata={"generic_harness_v1": True}),
        authoring=authoring,
        output=output,
    )

    agent = next(process for process in manifest.processes if process.name == "agent")
    assert agent.environment["ALK_LANGGRAPH_ENTRYPOINT"] == "src/example/graph.py:graph"
    assert "ThreadingHTTPServer" in agent.run_command[-1]
    sealed_contract = json.loads((output / "contract.json").read_text())
    assert sealed_contract["runtime"]["interface"]["kind"] == "callable"


def test_langgraph_graph_outside_source_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "langgraph.json").write_text(
        json.dumps({"graphs": {"Unsafe": "../outside.py:graph"}}), encoding="utf-8"
    )
    with pytest.raises(BundleAuthorError, match="langgraph_graph_invalid"):
        resolve_environment_plan(
            source,
            _job(connector="auto"),
            contract_modality="chat",
            contract_runtime={},
        )


def test_nested_script_runtime_uses_project_environment_without_agent_py(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "src" / "package").mkdir(parents=True)
    (source / "pyproject.toml").write_text(
        "[project]\nname='target'\nversion='1'\n"
        "[project.scripts]\nkickoff='package.main:main'\n",
        encoding="utf-8",
    )
    (source / "src" / "package" / "main.py").write_text(
        "def main():\n    print('target')\n",
        encoding="utf-8",
    )
    authoring = _authoring(tmp_path)
    _write_chat_contract(authoring, command=["kickoff"], workdir="src")

    plan = resolve_environment_plan(
        source,
        _job(connector="auto"),
        contract_modality="chat",
        contract_interface_kind="command",
        contract_runtime={"command": ["kickoff"], "workdir": "src"},
    )

    agent = next(process for process in plan.processes if process.name == "agent")
    assert agent.working_directory == "."
    assert agent.environment["ALK_SUBPROCESS_COMMAND"] == json.dumps(
        ["uv", "run", "--no-sync", "kickoff"]
    )
    assert "ThreadingHTTPServer" in agent.run_command[-1]


def test_agent_entrypoint_discovery_ignores_generated_virtualenv(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "app").mkdir(parents=True)
    (source / "app" / "agent.py").write_text("print('target')\n")
    dependency = source / ".venv" / "lib" / "site-packages" / "other"
    dependency.mkdir(parents=True)
    (dependency / "agent.py").write_text("print('dependency')\n")

    plan = resolve_environment_plan(
        source,
        _job(connector="auto"),
        contract_modality="chat",
        contract_interface_kind="http",
        contract_runtime={"command": [], "workdir": ""},
    )

    agent = next(process for process in plan.processes if process.name == "agent")
    assert agent.working_directory == "app"
    assert "agent.py" in agent.run_command


def test_http_runtime_uses_dockerfile_cmd_when_contract_omits_command(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "app").mkdir(parents=True)
    (source / "app" / "agent.py").write_text("print('not the server')\n")
    (source / "app" / "server.py").write_text("app = object()\n")
    (source / "Dockerfile").write_text(
        'FROM python:3.12\nCMD ["uv", "run", "uvicorn", "app.server:app", '
        '"--host", "0.0.0.0", "--port", "8080"]\n'
    )

    plan = resolve_environment_plan(
        source,
        _job(connector="auto"),
        contract_modality="chat",
        contract_interface_kind="http",
        contract_runtime={"command": [], "interface": {"kind": "http", "port": 8080}},
    )

    agent = next(process for process in plan.processes if process.name == "agent")
    assert agent.working_directory == "."
    assert agent.run_command[:4] == ["uv", "run", "uvicorn", "app.server:app"]


def test_declared_http_runtime_uses_contract_command_port_and_health(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "app").mkdir(parents=True)
    (source / "pyproject.toml").write_text(
        "[project]\nname='target'\nversion='1'\n", encoding="utf-8"
    )
    (source / "app" / "server.py").write_text("app = object()\n", encoding="utf-8")
    runtime = {
        "command": [
            "uv",
            "run",
            "uvicorn",
            "app.server:app",
            "--host",
            "0.0.0.0",
            "--port",
            "9090",
        ],
        "workdir": ".",
        "interface": {
            "kind": "http",
            "protocol": "custom",
            "path": "/invoke",
            "health_path": "/docs",
            "port": 9090,
        },
    }

    plan = resolve_environment_plan(
        source,
        _job(connector="auto"),
        contract_modality="chat",
        contract_interface_kind="http",
        contract_runtime=runtime,
    )

    agent = next(process for process in plan.processes if process.name == "agent")
    assert agent.working_directory == "."
    assert agent.run_command == runtime["command"]
    assert agent.fixed_port == 9090
    readiness = next(
        item for item in plan.readiness if item.capability == "target_http"
    )
    assert readiness.path == "/docs"


@pytest.mark.parametrize("declare_mount_root", [False, True])
def test_declared_http_runtime_selects_gpt_researcher_compose_backend(
    tmp_path: Path, monkeypatch, declare_mount_root: bool
) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "gpt_researcher_compose.yml"
    shutil.copyfile(fixture, tmp_path / "docker-compose.yml")
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.12\nCMD uvicorn main:app --port 8000\n"
    )
    (tmp_path / "main.py").write_text("print('server')\n")
    monkeypatch.setenv("PWD", "/ambient-private")
    if declare_mount_root:
        (tmp_path / "alk.yaml").write_text(
            'schema_version: "1"\nruntime:\n  environment:\n    PWD: "."\n'
        )
    else:
        with pytest.raises(
            BundleAuthorError, match="compose_structural_configuration_missing: PWD"
        ):
            resolve_environment_plan(
                tmp_path,
                _job(connector="auto"),
                contract_modality="chat",
                contract_interface_kind="http",
                contract_runtime={
                    "command": ["python", "main.py"],
                    "interface": {"kind": "http", "port": 8000, "health_path": "/"},
                },
            )
        return
    plan = resolve_environment_plan(
        tmp_path,
        _job(connector="auto"),
        contract_modality="chat",
        contract_interface_kind="http",
        contract_runtime={
            "command": ["python", "main.py"],
            "interface": {"kind": "http", "port": 8000, "health_path": "/"},
        },
    )
    assert plan.control_service == "gpt-researcher"
    assert [process.name for process in plan.processes] == [
        "world-db",
        "gpt-researcher",
    ]
    agent = plan.processes[1]
    assert agent.environment["PWD"] == "."
    assert agent.run_command == ["python", "main.py"]
    assert agent.fixed_port == 8000
    assert plan.capabilities["target_http"].container_port == 8000
    assert (
        next(
            probe for probe in plan.readiness if probe.capability == "target_http"
        ).path
        == "/"
    )


@pytest.mark.parametrize("command", [[], ["python", "agent.py", "start"]])
def test_voice_runtime_with_http_health_interface_keeps_worker_entrypoint(
    tmp_path: Path, command: list[str]
) -> None:
    (tmp_path / "agent.py").write_text("print('worker')\n")
    plan = resolve_environment_plan(
        tmp_path,
        _job(connector="livekit"),
        contract_modality="voice",
        contract_interface_kind="http",
        contract_runtime={
            "workdir": ".",
            "command": command,
            "interface": {"kind": "http", "port": 8081},
        },
    )
    agent = next(process for process in plan.processes if process.name == "agent")
    assert "agent.py" in agent.run_command
    assert agent.started_check.log_marker == "registered worker"


def test_http_workspace_requires_native_command_before_selecting_internal_agent(
    tmp_path: Path,
) -> None:
    backend = tmp_path / "backend"
    harness = backend / "packages" / "harness"
    extension = backend / "packages" / "extension-api"
    harness.mkdir(parents=True)
    extension.mkdir()
    (backend / "pyproject.toml").write_text(
        "[project]\nname='gateway'\nversion='1'\n"
        "[tool.uv.workspace]\nmembers=['packages/harness', 'packages/extension-api']\n"
    )
    (backend / "uv.lock").write_text("version = 1\n")
    (harness / "pyproject.toml").write_text("[project]\nname='harness'\nversion='1'\n")
    (extension / "pyproject.toml").write_text(
        "[project]\nname='extension-api'\nversion='1'\n"
    )
    (harness / "agent.py").write_text("agent = object()\n")
    runtime = {
        "workdir": "backend",
        "command": [],
        "interface": {"kind": "http", "port": 8001, "health_path": "/health"},
    }
    with pytest.raises(BundleAuthorError, match="runtime_http_command_missing"):
        resolve_environment_plan(
            tmp_path,
            _job(connector="auto"),
            contract_modality="chat",
            contract_interface_kind="http",
            contract_runtime=runtime,
        )

    runtime["command"] = ["uv", "run", "--no-sync", "uvicorn", "app.gateway.app:app"]
    plan = resolve_environment_plan(
        tmp_path,
        _job(connector="auto"),
        contract_modality="chat",
        contract_interface_kind="http",
        contract_runtime=runtime,
    )
    agent = next(process for process in plan.processes if process.name == "agent")
    assert agent.working_directory == "backend"
    assert agent.run_command == runtime["command"]
    assert "--locked" in agent.build_commands[0][-1]
    assert (
        tmp_path / agent.working_directory / "packages/harness/pyproject.toml"
    ).is_file()
    assert (
        tmp_path / agent.working_directory / "packages/extension-api/pyproject.toml"
    ).is_file()


def test_declared_compose_runtime_keeps_transitive_dependencies(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile").write_text(
        'FROM python:3.12\nCMD ["python", "main.py"]\n'
    )
    (tmp_path / "docker-compose.yml").write_text(
        "services:\n"
        "  backend:\n    build: .\n    ports: [{target: 8000, published: 8000}]\n"
        "    depends_on: {cache: {condition: service_started}}\n"
        "  cache:\n    image: redis:7\n    depends_on: [postgres]\n"
        "  postgres:\n    image: postgres:16\n"
        "  unrelated:\n    image: unsupported:latest\n"
    )
    plan = resolve_environment_plan(
        tmp_path,
        _job(connector="auto"),
        contract_modality="chat",
        contract_interface_kind="http",
        contract_runtime={
            "command": ["python", "main.py"],
            "interface": {"kind": "http", "port": 8000},
        },
    )
    assert {process.name for process in plan.processes} == {
        "world-db",
        "cache",
        "backend",
    }
    assert next(
        process for process in plan.processes if process.name == "backend"
    ).depends_on == ["cache"]


@pytest.mark.parametrize(
    ("postgres_image", "redis_image", "redis_version"),
    [
        ("postgres", "redis", "7"),
        ("registry.example/team/postgres:16", "bitnami/redis:7-alpine", "7"),
    ],
)
def test_declared_compose_runtime_accepts_catalog_images_without_a_tag_or_with_a_namespace(
    tmp_path: Path,
    postgres_image: str,
    redis_image: str,
    redis_version: str,
) -> None:
    (tmp_path / "Dockerfile").write_text(
        'FROM python:3.12\nCMD ["python", "main.py"]\n'
    )
    (tmp_path / "docker-compose.yml").write_text(
        "services:\n"
        "  backend:\n    build: .\n    ports: ['8000:8000']\n"
        "    depends_on: [cache, postgres]\n"
        f"  cache:\n    image: {redis_image}\n"
        f"  postgres:\n    image: {postgres_image}\n"
    )

    plan = resolve_environment_plan(
        tmp_path,
        _job(connector="auto"),
        contract_modality="chat",
        contract_interface_kind="http",
        contract_runtime={
            "command": ["python", "main.py"],
            "interface": {"kind": "http", "port": 8000},
        },
    )

    cache = next(process for process in plan.processes if process.name == "cache")
    assert cache.version == redis_version
    assert {process.name for process in plan.processes} == {
        "world-db",
        "cache",
        "backend",
    }


@pytest.mark.parametrize("ports", ["8000:8000", "8001:8001", "8000:8000/udp"])
def test_declared_compose_runtime_rejects_ambiguous_or_missing_match(
    tmp_path: Path, ports: str
) -> None:
    (tmp_path / "docker-compose.yml").write_text(
        "services:\n"
        f"  first:\n    build: .\n    ports: ['{ports}']\n"
        f"  second:\n    build: .\n    ports: ['{ports}']\n"
    )
    with pytest.raises(BundleAuthorError, match="control_service_ambiguous"):
        resolve_environment_plan(
            tmp_path,
            _job(connector="auto"),
            contract_modality="chat",
            contract_interface_kind="http",
            contract_runtime={
                "command": ["python", "main.py"],
                "interface": {"kind": "http", "port": 8000},
            },
        )


@pytest.mark.parametrize(
    ("dependency", "error"),
    [
        ("depends_on: [missing]", "compose_dependency_missing"),
        ("depends_on: worker", "compose_dependency_invalid"),
        ("depends_on: [{worker: {}}]", "compose_dependency_invalid"),
        ("links: [worker]", "compose_dependency_unsupported"),
        ("volumes_from: [worker]", "compose_dependency_unsupported"),
        ("network_mode: service:worker", "compose_dependency_unsupported"),
        ("ipc: service:worker", "compose_dependency_unsupported"),
        ("pid: service:worker", "compose_dependency_unsupported"),
        (
            "environment: {CACHE_URL: 'redis://worker:6379'}",
            "compose_dependency_undeclared",
        ),
        ("environment: [CACHE_HOST=worker]", "compose_dependency_undeclared"),
    ],
)
def test_declared_compose_runtime_does_not_drop_dependency_references(
    tmp_path: Path, dependency: str, error: str
) -> None:
    (tmp_path / "docker-compose.yml").write_text(
        "services:\n"
        "  backend:\n    build: .\n    ports: ['8000:8000']\n"
        f"    {dependency}\n"
        "  worker:\n    build: ./worker\n"
    )
    with pytest.raises(BundleAuthorError, match=error):
        resolve_environment_plan(
            tmp_path,
            _job(connector="auto"),
            contract_modality="chat",
            contract_interface_kind="http",
            contract_runtime={
                "command": ["python", "main.py"],
                "interface": {"kind": "http", "port": 8000},
            },
        )


def test_declared_compose_http_runtime_ignores_udp_port_match(tmp_path: Path) -> None:
    (tmp_path / "docker-compose.yml").write_text(
        "services:\n"
        "  tcp:\n    build: .\n    ports: ['8000:8000/tcp']\n"
        "  udp:\n    build: .\n    ports: [{target: 8000, protocol: udp}]\n"
    )
    plan = resolve_environment_plan(
        tmp_path,
        _job(connector="auto"),
        contract_modality="chat",
        contract_interface_kind="http",
        contract_runtime={
            "command": ["python", "main.py"],
            "interface": {"kind": "http", "port": 8000},
        },
    )
    assert plan.control_service == "tcp"
    assert [process.name for process in plan.processes] == ["world-db", "tcp"]


def test_adk_web_command_without_interface_uses_source_http_api(tmp_path: Path) -> None:
    source = tmp_path / "source"
    app = source / "software_bug_assistant"
    app.mkdir(parents=True)
    (source / "pyproject.toml").write_text(
        "[project]\nname='software-bug-assistant'\nversion='1'\n", encoding="utf-8"
    )
    (app / "__init__.py").write_text("", encoding="utf-8")
    (app / "agent.py").write_text("root_agent = object()\n", encoding="utf-8")
    authoring = _authoring(tmp_path)
    command = [
        "uv",
        "run",
        "adk",
        "web",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
    ]
    _write_chat_contract(authoring, command=command)
    with sqlite3.connect(authoring / "world.sqlite") as database:
        database.execute("CREATE TABLE fixture (id TEXT PRIMARY KEY)")

    output = tmp_path / "bundle"
    manifest = author_bundle_v2(
        source=source,
        job=_job(connector="auto", metadata={"generic_harness_v1": True}),
        authoring=authoring,
        output=output,
    )

    agent = next(process for process in manifest.processes if process.name == "agent")
    assert agent.run_command[:2] == ["sh", "-c"]
    assert "adk web" in agent.run_command[-1]
    assert "$PORT" in agent.run_command[-1]
    assert agent.fixed_port_consumable is True
    assert agent.environment["PORT"] == "{{PORT_agent}}"
    assert "ALK_SUBPROCESS_COMMAND" not in agent.environment
    assert "ThreadingHTTPServer" not in agent.run_command[-1]
    assert agent.fixed_port == 8080
    readiness = next(
        item for item in manifest.readiness if item.capability == "target_http"
    )
    assert readiness.path == "/docs"

    interface = json.loads((output / "contract.json").read_text())["runtime"][
        "interface"
    ]
    assert interface["kind"] == "http"
    assert interface["protocol"] == "json_template"
    assert interface["path"] == "/run_sse"
    assert interface["request_template"]["app_name"] == "software_bug_assistant"
    assert interface["setup_requests"][0]["path"] == (
        "/apps/software_bug_assistant/users/{{thread_id}}/sessions"
    )


def test_adk_source_toolbox_is_hosted_against_world_database(tmp_path: Path) -> None:
    source = tmp_path / "source"
    app = source / "bug_agent"
    tools = source / "deployment" / "mcp-toolbox"
    app.mkdir(parents=True)
    tools.mkdir(parents=True)
    (source / "pyproject.toml").write_text(
        "[project]\nname='bug-agent'\nversion='1'\n", encoding="utf-8"
    )
    (app / "__init__.py").write_text("", encoding="utf-8")
    (app / "agent.py").write_text("root_agent = object()\n", encoding="utf-8")
    (app / "tools.py").write_text(
        "from toolbox_core import ToolboxSyncClient\n"
        "tools = ToolboxSyncClient('http://127.0.0.1:5000').load_toolset('tickets')\n",
        encoding="utf-8",
    )
    (tools / "tools.yaml").write_text(
        "sources:\n"
        "  db:\n"
        "    kind: postgres\n"
        "tools:\n"
        "  get-ticket:\n"
        "    kind: postgres-sql\n"
        "    source: db\n"
        "    description: Get one ticket.\n"
        "    parameters:\n"
        "      - name: ticket_id\n"
        "        type: string\n"
        "    statement: SELECT * FROM tickets WHERE ticket_id = $1;\n"
        "toolsets:\n"
        "  tickets:\n"
        "    - get-ticket\n",
        encoding="utf-8",
    )
    authoring = _authoring(tmp_path)
    _write_chat_contract(
        authoring,
        command=[
            "uv",
            "run",
            "adk",
            "web",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
        ],
    )
    with sqlite3.connect(authoring / "world.sqlite") as database:
        database.execute("CREATE TABLE tickets (ticket_id TEXT PRIMARY KEY)")

    output = tmp_path / "bundle"
    manifest = author_bundle_v2(
        source=source,
        job=_job(connector="auto", metadata={"generic_harness_v1": True}),
        authoring=authoring,
        output=output,
    )

    adapter = next(
        process for process in manifest.processes if process.name == "mcp-toolbox"
    )
    assert adapter.depends_on == ["world-db"]
    assert adapter.environment["DATABASE_URL"] == "{{WORLD_DATABASE_URL}}"
    agent = next(process for process in manifest.processes if process.name == "agent")
    assert "mcp-toolbox" in agent.depends_on
    assert agent.environment["MCP_TOOLBOX_URL"] == "{{MCP_TOOLBOX_URL}}"
    assert (output / "generated" / "mcp-toolbox" / "server.py").is_file()
    definition = json.loads(
        (output / "generated" / "mcp-toolbox" / "toolbox.json").read_text()
    )
    assert definition["tools"]["get-ticket"]["statement"].endswith("$1;")


def test_explicit_adk_command_keeps_one_shot_adapter(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        "[project]\nname='target'\nversion='1'\n", encoding="utf-8"
    )
    authoring = _authoring(tmp_path)
    _write_chat_contract(
        authoring,
        command=["uv", "run", "adk", "run", "app"],
        interface={"kind": "command", "protocol": "fi.alk"},
    )
    with sqlite3.connect(authoring / "world.sqlite") as database:
        database.execute("CREATE TABLE fixture (id TEXT PRIMARY KEY)")

    output = tmp_path / "bundle"
    manifest = author_bundle_v2(
        source=source,
        job=_job(connector="auto", metadata={"generic_harness_v1": True}),
        authoring=authoring,
        output=output,
    )

    agent = next(process for process in manifest.processes if process.name == "agent")
    assert "ThreadingHTTPServer" in agent.run_command[-1]
    assert "ALK_SUBPROCESS_COMMAND" in agent.environment


def test_adk_web_command_repairs_incomplete_http_interface(tmp_path: Path) -> None:
    source = tmp_path / "source"
    app = source / "bug_agent"
    app.mkdir(parents=True)
    (source / "pyproject.toml").write_text(
        "[project]\nname='bug-agent'\nversion='1'\n", encoding="utf-8"
    )
    (app / "__init__.py").write_text("", encoding="utf-8")
    (app / "agent.py").write_text("root_agent = object()\n", encoding="utf-8")
    authoring = _authoring(tmp_path)
    _write_chat_contract(
        authoring,
        command=["uv", "run", "adk", "web", "--port=9090"],
        interface={"kind": "http", "port": 9090},
    )
    with sqlite3.connect(authoring / "world.sqlite") as database:
        database.execute("CREATE TABLE fixture (id TEXT PRIMARY KEY)")

    output = tmp_path / "bundle"
    manifest = author_bundle_v2(
        source=source,
        job=_job(connector="auto", metadata={"generic_harness_v1": True}),
        authoring=authoring,
        output=output,
    )

    agent = next(process for process in manifest.processes if process.name == "agent")
    assert agent.fixed_port == 9090
    assert agent.fixed_port_consumable is True
    interface = json.loads((output / "contract.json").read_text())["runtime"][
        "interface"
    ]
    assert interface["path"] == "/run_sse"
    assert interface["health_path"] == "/docs"
    assert interface["request_template"]["app_name"] == "bug_agent"


@pytest.mark.parametrize(
    ("lock", "knob", "expected"),
    [
        ('[[package]]\nname = "livekit-agents"\nversion = "1.6.10"\n', True, None),
        ('[[package]]\nname = "livekit-agents"\nversion = "1.3.12"\n', True, None),
        ('[[package]]\nname = "livekit-agents"\nversion = "1.2.18"\n', True, 8081),
        ('[[package]]\nname = "livekit-agents"\nversion = "1.6.10"\n', False, 8081),
        ("", True, 8081),
    ],
)
def test_livekit_health_port_is_fixed_only_when_the_shim_cannot_move_it(
    tmp_path, lock, knob, expected
):
    from fi.alk.harness.bundle_author_v2 import _livekit_cli_fixed_health_port

    (tmp_path / "uv.lock").write_text(lock)
    environment = {"FI_WORKER_HEALTH_PORT": "{{PORT_agent}}"} if knob else {}
    command = ["uv", "run", "agent.py", "start"]
    assert _livekit_cli_fixed_health_port(command, environment, tmp_path) == expected


def test_livekit_pinned_in_requirements_is_read(tmp_path):
    from fi.alk.harness.bundle_author_v2 import _livekit_isolatable

    (tmp_path / "requirements.txt").write_text("livekit-agents[silero]==1.4.0\n")
    assert _livekit_isolatable(tmp_path)


def test_livekit_lower_bound_in_pyproject_is_read(tmp_path):
    from fi.alk.harness.bundle_author_v2 import _livekit_isolatable

    (tmp_path / "pyproject.toml").write_text('dependencies = ["livekit-agents>=1.6.9,<2"]\n')
    assert _livekit_isolatable(tmp_path)
    (tmp_path / "pyproject.toml").write_text('dependencies = ["livekit-agents>=1.2,<2"]\n')
    assert not _livekit_isolatable(tmp_path)


def test_reseal_carries_an_added_scenario_into_the_sealed_bundle(tmp_path: Path) -> None:
    source = tmp_path / "provider-target"
    source.mkdir()
    authoring = _authoring(tmp_path)
    _write_voice_contract(authoring)
    job = HarnessJob.model_validate(
        {
            "job_id": "reseal-job",
            "run_id": "reseal-run",
            "execution": "hosted",
            "source": {"kind": "provider", "visibility": "public"},
            "agent": {
                "connector": "retell",
                "mode": ProviderExecutionMode.CONNECT_ONLY.value,
                "config": {"agent_id": "agent_existing"},
                "secret_refs": {
                    "RETELL_API_KEY": {
                        "manager": "platform-vault",
                        "key": "retell-key",
                        "purpose": "target_provider",
                    }
                },
            },
            "scenario_count": 1,
            "runtime": {
                "isolation": "dedicated_vm",
                "cpu_units": 2,
                "memory_mb": 4096,
                "parallelism": 1,
            },
        }
    )
    bundle = authoring / "environment-bundle"
    sealed = author_bundle_v2(source=source, job=job, authoring=authoring, output=bundle)
    added = authoring / "scenarios" / "two"
    shutil.copytree(authoring / "scenarios" / "one", added)
    (added / "scenario.json").write_text(
        json.dumps({"name": "two", "instruction": "Test two", "sub_goals": ["works"]}),
        encoding="utf-8",
    )

    digest = reseal_scenarios(authoring)

    resealed = load_bundle_v2(bundle)
    assert digest == resealed.digest != sealed.digest
    assert sorted(path.name for path in (bundle / "scenarios").iterdir()) == ["one", "two"]
    assert json.loads((bundle / "scenarios" / "two" / "scenario.json").read_text())[
        "scenario_key"
    ] == "two"
    preflight_bundle(
        bundle, resealed, parallelism=1, secret_refs={"RETELL_API_KEY": "target_provider"}
    )


def test_reseal_without_a_sealed_bundle_changes_nothing(tmp_path: Path) -> None:
    assert reseal_scenarios(_authoring(tmp_path)) is None


def test_a_changed_world_reaches_a_world_ir_seed_on_reseal(tmp_path: Path) -> None:
    from fi.alk.harness.source_model import LogicalType, SourceColumn, SourceTable
    from fi.alk.harness.world_import.sqlite import import_sqlite_world

    source = tmp_path / "source"
    (source / "db").mkdir(parents=True)
    (source / "agent.py").write_text("print('ok')\n", encoding="utf-8")
    (source / "db" / "schema.sql").write_text(
        "CREATE TABLE places (id TEXT PRIMARY KEY, label TEXT);\n", encoding="utf-8"
    )
    authoring = _authoring(tmp_path)
    artifact_root = authoring / "generic-harness"
    artifact_root.mkdir()
    columns = tuple(
        SourceColumn(name=name, logical_type=LogicalType.STRING, native_type="text", nullable=name != "id", has_default=False)
        for name in ("id", "label")
    )
    source_model = SourceModel.create(
        source_digest="sha256:" + source_fingerprint(source),
        engine="postgres",
        tables=(SourceTable(name="places", columns=columns, primary_key=("id",)),),
    )
    (artifact_root / "source-model.json").write_text(source_model.model_dump_json(), encoding="utf-8")
    with sqlite3.connect(authoring / "world.sqlite") as database:
        database.execute("CREATE TABLE places (id TEXT PRIMARY KEY, label TEXT)")
        database.execute("INSERT INTO places VALUES ('home', 'home')")
    with sqlite3.connect(authoring / "world.sqlite") as database:
        world = import_sqlite_world(database, source_model, validate=False).world
    (artifact_root / "world-ir.json").write_text(world.model_dump_json(indent=2) + "\n", encoding="utf-8")
    bundle = authoring / "environment-bundle"
    sealed = author_bundle_v2(
        source=source,
        job=_job(connector="http", metadata={"generic_harness_v1": True}),
        authoring=authoring,
        output=bundle,
    )

    with sqlite3.connect(authoring / "world.sqlite") as database:
        database.execute("UPDATE places SET label = 'renamed' WHERE id = 'home'")
    digest = reseal_scenarios(authoring)

    assert digest is not None and digest != sealed.digest
    assert "renamed" in (bundle / "seed" / "world-ir.json").read_text(encoding="utf-8")
    assert "renamed" in (artifact_root / "world-ir.json").read_text(encoding="utf-8")
    assert reseal_scenarios(authoring) == digest


def test_bundle_retains_unused_static_credential_inventory_without_blocking(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "agent.py").write_text("print('source')")
    (source / "optional.py").write_text(
        "import os\ndef unused(): return os.environ['UNUSED_TOKEN']\n"
    )
    authoring = _authoring(tmp_path)
    _write_voice_contract(authoring)
    output = tmp_path / "bundle"
    author_bundle_v2(
        source=source, job=_job(connector="auto"), authoring=authoring, output=output
    )
    inventory = json.loads((output / "credentials.json").read_text())
    unused = next(
        item
        for item in inventory["requirements"]
        if item["environment_name"] == "UNUSED_TOKEN"
    )
    assert unused["status"] == "missing"
    assert unused["required"] is True

@pytest.mark.parametrize("version", ["3.12", "3.12.8"])
def test_native_runtime_honors_declared_python_version(tmp_path: Path, version: str) -> None:
    (tmp_path / "agent.py").write_text("print('agent')\n")
    (tmp_path / "Dockerfile").write_text("FROM python:3.14\n")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='agent'\nversion='1'\nrequires-python='>=3.12'\n"
    )
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    plan = resolve_environment_plan(
        tmp_path, _job(connector="http"),
        contract_runtime={"language": "python", "version": version, "command": ["python", "agent.py"]},
    )
    process = next(p for p in plan.processes if p.name == "agent")
    command = process.build_commands[0]
    assert command[command.index("--python") + 1] == (
        "==3.12.*,>=3.12" if version == "3.12" else "python3.12.8"
    )
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before

@pytest.mark.parametrize("version", ["3.11.9", "nonsense", "3.12garbage"])
def test_native_runtime_rejects_invalid_declared_python(tmp_path: Path, version: str) -> None:
    (tmp_path / "agent.py").write_text("print('agent')\n")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='agent'\nversion='1'\nrequires-python='>=3.12'\n"
    )
    with pytest.raises(BundleAuthorError, match="generated_runtime_python_version_unsupported"):
        resolve_environment_plan(
            tmp_path, _job(connector="http"),
            contract_runtime={"language": "python", "version": version, "command": ["python", "agent.py"]},
        )


@pytest.mark.parametrize("child", [".", "packages/chat", "-chat"])
def test_native_bun_plan_installs_locked_root_and_preserves_child_command(
    tmp_path: Path, child: str
) -> None:
    (tmp_path / "package.json").write_text(
        json.dumps(
            {
                "name": "plain-chat" if child == "." else "workspace-bot",
                "packageManager": "bun@1.3.14",
            }
        )
    )
    (tmp_path / "bun.lock").write_text('{"lockfileVersion": 1}')
    component = tmp_path / child
    component.mkdir(parents=True, exist_ok=True)
    if child != ".":
        (component / "package.json").write_text('{"name":"chat"}')
    (component / "server.ts").write_text("Bun.serve({fetch: () => new Response('ok')})")
    plan = resolve_environment_plan(
        tmp_path,
        _job(connector="auto"),
        contract_modality="chat",
        contract_interface_kind="http",
        contract_runtime={
            "language": "bun",
            "version": "1.3.14",
            "execution": "native",
            "workdir": child,
            "command": ["bun", "run", "server.ts"],
            "install": "touch /tmp/must-not-execute",
            "workspaces": {"project": "files"},
            "interface": {"kind": "http", "port": 8001},
        },
    )
    agent = next(process for process in plan.processes if process.name == "agent")
    assert agent.working_directory == "."
    assert agent.workspaces == {
        "project": "files" if child == "." else child + "/files"
    }
    assert agent.build_commands[-1] == ["bun", "install", "--frozen-lockfile"]
    assert agent.run_command == (
        ["bun", "run", "server.ts"]
        if child == "."
        else [
            "sh",
            "-c",
            'cd "./$1" && shift && exec "$@"',
            "alk-source-process",
            child,
            "bun",
            "run",
            "server.ts",
        ]
    )
    assert "must-not-execute" not in repr(agent.build_commands)


@pytest.mark.parametrize(
    "manifest,lock,version,command,error",
    [
        ({}, True, "", ["bun", "run", "server.ts"], "package_manager_missing"),
        (
            {"packageManager": "bun@latest"},
            True,
            "",
            ["bun", "run", "server.ts"],
            "package_manager_unpinned",
        ),
        (
            {"packageManager": "npm@10.0.0"},
            True,
            "",
            ["bun", "run", "server.ts"],
            "package_manager_unpinned",
        ),
        (
            {"packageManager": "bun@1.3.14"},
            False,
            "",
            ["bun", "run", "server.ts"],
            "lock_required",
        ),
        (
            {"packageManager": "bun@1.3.14"},
            True,
            "1.3.13",
            ["bun", "run", "server.ts"],
            "version_mismatch",
        ),
        (
            {"packageManager": "bun@1.3.14"},
            True,
            "",
            ["sh", "-c", "bun server.ts"],
            "command_required",
        ),
        ([], True, "", ["bun", "run", "server.ts"], "manifest_invalid"),
    ],
)
def test_native_bun_rejects_invalid_source(
    tmp_path: Path, manifest, lock, version, command, error: str
) -> None:
    from fi.alk.harness.bundle_author_v2 import _plan_source_process

    (tmp_path / "package.json").write_text(json.dumps(manifest))
    if lock:
        (tmp_path / "bun.lock").write_text("{}")
    with pytest.raises(BundleAuthorError, match=error):
        _plan_source_process(
            tmp_path,
            language="bun",
            name="agent",
            root=tmp_path,
            entry=None,
            control=True,
            needs_secrets=False,
            version=version,
            run_override=command,
        )


def test_native_bun_requires_matching_installed_toolchain(tmp_path: Path) -> None:
    from fi.alk.harness.bundle_author_v2 import _plan_source_process

    (tmp_path / "package.json").write_text('{"packageManager":"bun@1.3.14"}')
    (tmp_path / "bun.lockb").write_bytes(b"frozen")
    process = _plan_source_process(
        tmp_path,
        language="bun",
        name="agent",
        root=tmp_path,
        entry=None,
        control=True,
        needs_secrets=False,
        run_override=["bun", "server.ts"],
    )
    command = [sys.executable, *process.build_commands[0][1:]]
    missing = subprocess.run(
        command, env={"PATH": str(tmp_path)}, capture_output=True, text=True
    )
    assert missing.returncode != 0
    assert "runtime_bun_toolchain_missing" in missing.stderr
    binary = tmp_path / "bun"
    binary.write_text("#!/bin/sh\necho 1.3.13\n")
    binary.chmod(0o755)
    wrong = subprocess.run(
        command, env={"PATH": str(tmp_path)}, capture_output=True, text=True
    )
    assert wrong.returncode != 0
    assert "runtime_bun_toolchain_version_mismatch" in wrong.stderr
    binary.write_text("#!/bin/sh\necho 1.3.14\n")
    matching = subprocess.run(
        command, env={"PATH": str(tmp_path)}, capture_output=True, text=True
    )
    assert matching.returncode == 0
    assert matching.stdout.strip() == "Bun toolchain 1.3.14"


def test_native_bun_rejects_ambiguous_and_symlink_lock(tmp_path: Path) -> None:
    from fi.alk.harness.bundle_author_v2 import _plan_source_process

    (tmp_path / "package.json").write_text('{"packageManager":"bun@1.3.14"}')
    (tmp_path / "bun.lock").write_text("{}")
    (tmp_path / "bun.lockb").write_bytes(b"frozen")
    kwargs = dict(
        language="bun",
        name="agent",
        root=tmp_path,
        entry=None,
        control=True,
        needs_secrets=False,
        run_override=["bun", "server.ts"],
    )
    with pytest.raises(BundleAuthorError, match="lock_required"):
        _plan_source_process(tmp_path, **kwargs)
    (tmp_path / "bun.lockb").unlink()
    (tmp_path / "bun.lock").unlink()
    (tmp_path / "bun.lock").symlink_to(tmp_path / "package.json")
    with pytest.raises(BundleAuthorError, match="manifest_unsafe"):
        _plan_source_process(tmp_path, **kwargs)


def test_native_process_runs_from_scanned_bundle_context(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "agent.py").write_text("from helper import answer\nprint(answer())\n")
    (source / "helper.py").write_text("def answer():\n    return 'chat-ready'\n")
    (source / "fixtures").mkdir()
    (source / "fixtures/token.txt").write_text("sk-unreviewedFixtureValue1234567890")
    (source / "manifest.json").write_text('{"application": "fixture"}')
    authoring = _authoring(tmp_path)
    _write_chat_contract(
        authoring,
        command=["python", "agent.py"],
        interface={"kind": "command", "protocol": "fi.alk"},
    )
    document = json.loads((authoring / "contract.json").read_text())
    document["runtime"]["execution"] = "native"
    (authoring / "contract.json").write_text(json.dumps(document))
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="generated_runtime_secret_material_detected"):
        author_bundle_v2(
            source=source,
            authoring=authoring,
            job=_job(connector="auto"),
            output=output,
        )
    document["runtime"]["context_excludes"] = ["fixtures"]
    (authoring / "contract.json").write_text(json.dumps(document))
    manifest = author_bundle_v2(
        source=source, authoring=authoring, job=_job(connector="auto"), output=output
    )
    process = next(
        p for p in manifest.processes if p.name == manifest.runtime.control_service
    )
    assert process.source_origin == "bundle"
    context = output / process.working_directory
    assert not (context / "fixtures").exists()
    assert (context / "manifest.json").read_text() == '{"application": "fixture"}'
    assert any(f.path.endswith("/manifest.json") for f in manifest.files)
    assert (source / "fixtures/token.txt").exists()
    shutil.rmtree(source)
    result = subprocess.run(
        [sys.executable, "agent.py"],
        cwd=context,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout == "chat-ready\n"
    load_bundle_v2(output)


@pytest.mark.parametrize("directory", ["vendor", "coverage", "target"])
def test_packaged_runtime_preserves_named_source_packages(tmp_path, directory):
    from fi.alk.harness.bundle_author_v2 import _package_process_sources
    from fi.alk.harness.contract import Runtime

    source = tmp_path / "source"
    (source / directory).mkdir(parents=True)
    (source / directory / "agent.py").write_text("print('source preserved')\n")
    runtime = Runtime(
        language="python",
        execution="native",
        command=["python", f"{directory}/agent.py"],
    )
    plan = resolve_environment_plan(
        source, _job(connector="auto"), contract_runtime=runtime.model_dump()
    )
    output = tmp_path / "bundle"
    packaged = _package_process_sources(source, output, plan, runtime)
    agent = next(p for p in packaged.processes if p.name == packaged.control_service)
    assert (
        output / agent.working_directory / directory / "agent.py"
    ).read_text() == "print('source preserved')\n"


@pytest.mark.parametrize("required", ["agent.py", "requirements.txt", "extra.txt"])
def test_packaging_cannot_exclude_declared_runtime_inputs(tmp_path, required):
    from fi.alk.harness.bundle_author_v2 import _package_process_sources
    from fi.alk.harness.contract import Runtime

    (tmp_path / "agent.py").write_text("print('ready')\n")
    (tmp_path / "requirements.txt").write_text("")
    (tmp_path / "extra.txt").write_text("")
    runtime = Runtime(
        language="python",
        execution="native",
        command=["python", "agent.py"],
        requirements_files=["extra.txt"],
    )
    plan = resolve_environment_plan(
        tmp_path, _job(connector="auto"), contract_runtime=runtime.model_dump()
    )
    runtime.context_excludes = [required]
    with pytest.raises(ValueError, match="context_exclude_required"):
        _package_process_sources(tmp_path, tmp_path.parent / "bundle", plan, runtime)


def test_workspace_exclusions_stay_relative_to_declared_component(tmp_path):
    from fi.alk.harness.bundle_author_v2 import _package_process_sources
    from fi.alk.harness.contract import Runtime

    app = tmp_path / "apps/chat"
    (app / "fixtures").mkdir(parents=True)
    (app / "fixtures/token.txt").write_text("sk-unreviewedFixtureValue1234567890")
    (app / "server.ts").write_text("Bun.serve({fetch: () => new Response('ready')})")
    (app / "package.json").write_text('{"name":"chat"}')
    (tmp_path / "package.json").write_text(
        '{"name":"workspace","packageManager":"bun@1.3.14","workspaces":["apps/*"]}'
    )
    (tmp_path / "bun.lock").write_text('{"lockfileVersion":1,"workspaces":{}}')
    runtime = Runtime(
        language="bun",
        execution="native",
        workdir="apps/chat",
        command=["bun", "server.ts"],
        context_excludes=["fixtures"],
        interface={"kind": "http", "port": 8080, "path": "/chat"},
    )
    plan = resolve_environment_plan(
        tmp_path,
        _job(connector="auto"),
        contract_modality="chat",
        contract_interface_kind="http",
        contract_runtime=runtime.model_dump(),
    )
    assert (
        next(
            p for p in plan.processes if p.name == plan.control_service
        ).working_directory
        == "."
    )
    output = tmp_path.parent / "bundle"
    packaged = _package_process_sources(tmp_path, output, plan, runtime)
    agent = next(p for p in packaged.processes if p.name == packaged.control_service)
    context = output / agent.working_directory
    assert not (context / "apps/chat/fixtures").exists()
    assert (context / "apps/chat/server.ts").is_file()
    assert (context / "package.json").is_file()


def test_python_workspace_packages_siblings_and_preserves_member_cwd(tmp_path):
    from fi.alk.harness.bundle_author_v2 import _package_process_sources
    from fi.alk.harness.contract import Runtime

    source = tmp_path / "source"
    member = source / "apps" / "server"
    sibling = source / "library"
    member.mkdir(parents=True)
    sibling.mkdir()
    (source / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers=["apps/*", "library"]\n'
    )
    (source / "uv.lock").write_text("version = 1\n")
    (member / "pyproject.toml").write_text(
        '[project]\nname="chat-server"\nversion="1"\n'
    )
    (sibling / "pyproject.toml").write_text(
        '[project]\nname="chat-library"\nversion="1"\n[project.optional-dependencies]\ntransport=[]\n'
    )
    (sibling / "value.txt").write_text("local source")
    (member / "app.py").write_text(
        'from pathlib import Path\nprint(Path("../../library/value.txt").read_text())\n'
    )
    (member / "fixtures").mkdir()
    (member / "fixtures" / "token.txt").write_text(
        "sk-unreviewedFixtureValue1234567890"
    )
    runtime = Runtime(
        language="python",
        execution="native",
        workdir="apps/server",
        command=[sys.executable, "app.py"],
        context_excludes=["fixtures"],
        workspaces={"files": "files"},
    )
    plan = resolve_environment_plan(
        source, _job(connector="auto"), contract_runtime=runtime.model_dump()
    )
    agent = next(p for p in plan.processes if p.name == plan.control_service)
    assert agent.working_directory == "."
    assert agent.workspaces == {"files": "apps/server/files"}
    assert "--package chat-server" in agent.build_commands[0][-1]
    assert "--locked" in agent.build_commands[0][-1]
    output = tmp_path / "bundle"
    packaged = _package_process_sources(source, output, plan, runtime)
    agent = next(p for p in packaged.processes if p.name == packaged.control_service)
    context = output / agent.working_directory
    assert (context / "uv.lock").is_file()
    assert not (context / "apps/server/fixtures").exists()
    shutil.rmtree(source)
    result = subprocess.run(
        agent.run_command, cwd=context, check=True, text=True, capture_output=True
    )
    assert result.stdout == "local source\n"


def test_compose_workspace_member_exclusions_without_runtime_workdir(tmp_path):
    from fi.alk.harness.bundle_author_v2 import _package_process_sources
    from fi.alk.harness.contract import Runtime

    member = tmp_path / "server"
    (member / "fixtures").mkdir(parents=True)
    (member / "fixtures/token.txt").write_text("sk-unreviewedFixtureValue1234567890")
    (member / "agent.py").write_text("print('ready')")
    (member / "pyproject.toml").write_text('[project]\nname="server"\nversion="1"\n')
    (member / "Dockerfile").write_text('FROM python:3.12\nCMD ["python", "agent.py"]\n')
    (tmp_path / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers=["server"]\n'
    )
    (tmp_path / "compose.yaml").write_text(
        "services:\n  agent:\n    build: ./server\n    command: [python, agent.py]\n"
    )
    runtime = Runtime(
        language="python",
        execution="compose",
        compose_file="compose.yaml",
        service="agent",
        context_excludes=["fixtures"],
    )
    plan = resolve_environment_plan(
        tmp_path, _job(connector="auto"), contract_runtime=runtime.model_dump()
    )
    output = tmp_path.parent / (tmp_path.name + "-bundle")
    packaged = _package_process_sources(tmp_path, output, plan, runtime)
    agent = next(p for p in packaged.processes if p.name == packaged.control_service)
    assert not (output / agent.working_directory / "server/fixtures").exists()
    assert (output / agent.working_directory / "server/agent.py").exists()
    assert 'source_component' not in agent.model_dump()


def test_environment_bundle_omits_public_suite_and_simulator(tmp_path):
    source = tmp_path / "agent"
    source.mkdir()
    (source / "agent.py").write_text("print('agent')\n")
    (source / "pyproject.toml").write_text("[project]\nname='agent'\nversion='1'\n")
    authoring = _authoring(tmp_path)
    _write_chat_contract(authoring, command=["python", "agent.py"], interface={
        "kind": "http", "port": 8080, "path": "/chat",
    })
    (authoring / "simulator_prompt.md").write_text("Public simulator prompt")
    shutil.rmtree(authoring / "scenarios")
    job = _job(connector="auto", with_secrets=False)
    output = tmp_path / "bundle"
    manifest = author_bundle_v2(
        source=source, job=job, authoring=authoring, output=output,
        include_scenarios=False,
    )
    assert (output / "contract.json").is_file()
    assert not (output / "scenarios").exists()
    assert not (output / "simulator_prompt.md").exists()
    assert "scenarios/" not in manifest.provenance.adopted_files
    with pytest.raises(BundleAuthorError, match="scenario_artifacts_missing"):
        author_bundle_v2(source=source, job=job, authoring=authoring, output=tmp_path / "full")


def test_native_plan_preserves_config_files_and_rejects_unknown_aliases(tmp_path):
    (tmp_path / "agent.py").write_text("print('hello')")
    runtime = {
        "execution": "native",
        "command": ["python", "agent.py"],
        "config_files": {
            ".env": {"values": {"MODE": "test"}, "secret_aliases": {"API_KEY": "KEY"}}
        },
        "directories": ["state"],
    }
    job = _job(connector="auto", with_secrets=True, secret_aliases=("KEY",))
    plan = resolve_environment_plan(
        tmp_path,
        job,
        contract_modality="chat",
        contract_interface_kind="command",
        contract_runtime=runtime,
    )
    process = next(p for p in plan.processes if p.name == "agent")
    assert process.config_files[".env"].secret_aliases == {"API_KEY": "KEY"}
    assert process.environment["MODE"] == "test"
    assert process.directories == ["state"]
    with pytest.raises(
        BundleAuthorError, match="runtime_config_target_reference_required"
    ):
        resolve_environment_plan(
            tmp_path,
            _job(connector="auto"),
            contract_modality="chat",
            contract_interface_kind="command",
            contract_runtime=runtime,
        )


@pytest.mark.parametrize("version", ["3.11", "3.11.9"])
def test_native_python_rejects_source_uv_environment_mismatch(tmp_path: Path, version: str) -> None:
    from fi.alk.harness.bundle_author_v2 import _plan_python

    (tmp_path / "agent.py").write_text("print('hello')\n")
    (tmp_path / ".python-version").write_text("3.14\n")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='agent'\nversion='1'\nrequires-python='>=3.11,<3.15'\n"
        "[tool.uv]\nenvironments=[\"python_version >= '3.14'\"]\n"
    )
    with pytest.raises(BundleAuthorError, match="generated_runtime_python_version_unsupported") as caught:
        _plan_python(tmp_path, name="agent", root=tmp_path, entry="agent.py", control=True,
                     needs_secrets=False, version=version)
    assert "python_version >= '3.14'" in str(caught.value)
    assert "source .python-version: 3.14" in str(caught.value)


@pytest.mark.parametrize("version", ["3.13", "3.13.8", ">=3.11"])
def test_native_python_selects_compatible_uv_environment_without_overriding_explicit_version(tmp_path: Path, version: str) -> None:
    from fi.alk.harness.bundle_author_v2 import _plan_python

    (tmp_path / "agent.py").write_text("print('hello')\n")
    (tmp_path / ".python-version").write_text("3.14\n")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='agent'\nversion='1'\nrequires-python='>=3.11,<3.15'\n"
        "[tool.uv]\nenvironments=[\"python_version >= '3.13'\"]\n"
    )
    plan = _plan_python(tmp_path, name="agent", root=tmp_path, entry="agent.py", control=True,
                        needs_secrets=False, version=version)
    command = plan.build_commands[0]
    assert command[command.index("--python") + 1] == (
        "python3.13.8" if version == "3.13.8" else
        "python3.13" if version == ">=3.11" else "<3.15,==3.13.*,>=3.11"
    )


@pytest.mark.parametrize("environments", [
    ["python_version == '3.11'", "python_version >= '3.14'"],
    ["python_version == '3.11' or python_version >= '3.14'"],
    ["python_version >= '3.14'", "sys_platform == 'linux'"],
    ["python_full_version >= '3.11.8'"],
])
def test_python_selection_preserves_uv_alternatives_and_target_only_constraints(tmp_path: Path, environments: list[str]) -> None:
    from fi.alk.harness.generated_runtime import _python_language_version

    (tmp_path / "pyproject.toml").write_text(
        "[project]\nrequires-python='>=3.11'\n[tool.uv]\n"
        f"environments={json.dumps(environments)}\n"
    )
    assert _python_language_version(tmp_path, "3.11", interpreter_selector=True) == "==3.11.*,>=3.11"
