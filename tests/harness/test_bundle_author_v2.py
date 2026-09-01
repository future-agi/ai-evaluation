from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from fi.alk.harness.bundle_author_v2 import (
    BundleAuthorError,
    _contract_column_declarations,
    _sqlite_sql,
    author_bundle_v2,
    resolve_environment_plan,
)
from fi.alk.harness.bundle_v2 import load_bundle_v2
from fi.alk.harness.job import HarnessJob
from fi.alk.harness.process_preflight import preflight_bundle
from fi.alk.harness.world.runtime import GeneratedWorld


def _job(
    *, connector: str, with_secrets: bool = False, scenario_count: int = 1
) -> HarnessJob:
    secret_refs = {}
    if with_secrets:
        secret_refs = {
            "LIVEKIT_API_KEY": {
                "manager": "platform-vault",
                "key": "livekit-key",
                "purpose": "target_provider",
            }
        }
    return HarnessJob.model_validate(
        {
            "job_id": "job-v2",
            "run_id": "run-v2",
            "execution": "hosted",
            "source": {"kind": "archive", "archive_artifact_id": "source-1"},
            "agent": {"connector": connector, "secret_refs": secret_refs},
            "scenario_count": scenario_count,
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
    job = _job(connector="auto", with_secrets=True)

    bundle = author_bundle_v2(
        source=source, job=job, authoring=authoring, output=tmp_path / "bundle"
    )

    agent = next(process for process in bundle.processes if process.name == "agent")
    assert agent.environment["LIVEKIT_AGENT_NAME"].endswith("-w{{WORLD_INDEX}}")
    assert agent.environment["HARNESS_MODE"] == "1"
    assert "target_provider" in agent.secret_purposes
    assert "target_http" not in bundle.capabilities


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

    with pytest.raises(
        BundleAuthorError,
        match="target_runtime_configuration_missing: environment:GOOGLE_CLOUD_PROJECT",
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
    job = _job(connector="auto", with_secrets=True).model_copy(
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
    assert agent.run_command[-1] == "start"
    if manifest.endswith("toml"):
        assert agent.build_commands[0] == [
            "uv",
            "sync",
            "--no-cache",
            "--python",
            "python3.14",
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


def test_callable_contract_compiles_repository_callback_adapter(tmp_path: Path) -> None:
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

    bundle = author_bundle_v2(
        source=source,
        job=_job(connector="auto", with_secrets=True),
        authoring=authoring,
        output=tmp_path / "bundle",
    )

    agent = next(process for process in bundle.processes if process.name == "agent")
    assert agent.working_directory == "."
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
    assert agent.working_directory == "."
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

    bundle = author_bundle_v2(
        source=source,
        job=job,
        authoring=_authoring(tmp_path),
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
    plan = resolve_environment_plan(source, _job(connector="livekit", with_secrets=True))

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
    plan = resolve_environment_plan(source, _job(connector="livekit", with_secrets=True))
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
    plan = resolve_environment_plan(source, _job(connector="livekit", with_secrets=True))

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
    plan = resolve_environment_plan(source, _job(connector="livekit", with_secrets=True))

    tools = next(p for p in plan.processes if p.name == "tools-api")
    assert tools.fixed_port_consumable is True
    assert tools.environment["FI_TOOLS_PORT"] == "{{PORT_tools-api}}"
    assert "FI_WORKER_HEALTH_PORT" not in tools.environment
    control = next(p for p in plan.processes if p.name == "agent")
    assert control.environment["FI_WORKER_HEALTH_PORT"] == "{{PORT_agent}}"
    assert "FI_TOOLS_PORT" not in control.environment


def test_livekit_worker_carries_the_worker_knob_env(tmp_path: Path) -> None:
    # C1 §4: the FI_* trio is authored UNCONDITIONALLY into every LiveKit-worker process, each
    # fed its OWN `{{PORT_<name>}}`. `FI_WORKER_HEALTH_PORT`'s presence IS the knob-bearing mark.
    source = tmp_path / "voice-compose"
    _livekit_compose_with_command_fixed_tools(source)
    plan = resolve_environment_plan(source, _job(connector="livekit", with_secrets=True))

    control = next(p for p in plan.processes if p.name == "agent")
    assert control.environment["FI_WORKER_HEALTH_PORT"] == "{{PORT_agent}}"
    assert control.environment["FI_LOAD_THRESHOLD"] == "inf"
    assert control.environment["FI_NUM_IDLE_PROCESSES"] == "1"
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
    plan = resolve_environment_plan(source, _job(connector="livekit", with_secrets=True))

    control = next(p for p in plan.processes if p.name == "agent")
    assert control.environment["FI_WORKER_HEALTH_PORT"] == "{{PORT_agent}}"
    assert control.environment["FI_LOAD_THRESHOLD"] == "inf"
    assert control.environment["FI_NUM_IDLE_PROCESSES"] == "1"
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
