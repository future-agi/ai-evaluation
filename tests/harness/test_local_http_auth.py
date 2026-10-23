from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from fi.alk.harness import chat_call_runner as chat
from fi.alk.harness import process_runtime as pr
from fi.alk.harness.authoring_runtime_validation import (
    RuntimeValidationError,
    _probe_runtime_readiness,
)
from fi.alk.harness.chat_readiness import probe_chat_ingress
from fi.alk.harness.bundle_v2 import EnvironmentBundleV2, SourceProcess
from fi.alk.harness.contract import (
    AgentContract, LocalHttpAuth, Runtime, RuntimeHttpSetupRequest, RuntimeInterface,
)
from fi.alk.harness.runtime_repair import (
    RuntimePlanPatch,
    _contract_hash,
    validate_runtime_plan_patch,
)
from test_chat_call_runner import _Adapter, _ToolWorld, _context, _single_exchange

AUTH = {"server_environment": "LOCAL_SERVER_KEY", "template_variable": "local_auth"}
SERVER = """import json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
secret = os.environ["LOCAL_SERVER_KEY"]
print("starting", flush=True)
sys.stdout.write("startup " + secret[:17]); sys.stdout.flush()
time.sleep(0.02)
print(secret[17:], flush=True)
class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        data = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        ok = self.headers.get("Authorization") == "Bearer " + secret
        body = json.dumps({"content": "Authenticated " + secret} if ok else {"error": secret}).encode()
        self.send_response(200 if ok else 401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args): pass
HTTPServer(("127.0.0.1", int(os.environ["PORT"])), Handler).serve_forever()
"""


def interface(**overrides):
    return RuntimeInterface(
        **{
            "kind": "http",
            "port": 8000,
            "path": "/chat",
            "include_tools": False,
            "local_auth": AUTH,
            "headers_template": {"Authorization": "Bearer {{local_auth}}"},
            **overrides,
        }
    )


def manifest(command=None):
    return EnvironmentBundleV2.model_validate(
        {
            "schema_version": "futureagi.environment-bundle.v2",
            "digest": "sha256:" + "a" * 64,
            "name": "local-auth",
            "runtime": {
                "kind": "process",
                "control_service": "agent",
                "evidence_seam": "tool_trace",
            },
            "processes": [
                {
                    "name": "agent",
                    "kind": "source",
                    "working_directory": ".",
                    "run_command": command or [sys.executable, "server.py"],
                    "environment": {"PORT": "{{PORT_agent}}"},
                    "local_auth": AUTH,
                    "user": "svc-agent",
                    "started_check": {"port": True, "timeout_seconds": 3},
                },
                {
                    "name": "sidecar",
                    "kind": "source",
                    "working_directory": ".",
                    "run_command": [
                        sys.executable,
                        "-c",
                        "import time; time.sleep(60)",
                    ],
                    "user": "svc-tools",
                },
            ],
            "capabilities": {
                "target_http": {
                    "protocol": "http",
                    "service": "agent",
                    "configuration_name": "TARGET_HTTP_URL",
                }
            },
            "readiness": [
                {
                    "capability": "target_http",
                    "path": "/private-health",
                    "timeout_seconds": 3,
                    "interval_seconds": 0.02,
                }
            ],
            "provenance": {
                "source_kind": "repository",
                "repository": "example/agent",
                "source_digest": "b" * 64,
            },
        }
    )


def provision(tmp_path, body, capture):
    source = tmp_path / "source"
    source.mkdir()
    (source / "server.py").write_text(SERVER)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    work = tmp_path / "work"

    def runner(argv, **kwargs):
        capture.append((kwargs["cwd"], dict(kwargs["env"])))
        return pr.default_process_runner(argv, **kwargs)

    provider = pr.ProcessRuntimeProvider(
        secrets_path=tmp_path / "secrets.json",
        runner=runner,
        user_resolver=lambda _: None,
        cpu_observer=lambda: 64,
        mem_observer=lambda: 64,
    )

    async def start():
        return await provider.provision(
            body,
            source=source,
            bundle_dir=bundle,
            work_directory=work,
            instances=2,
            require_declared_user=False,
        )

    return provider, start, work, source


def test_private_auth_real_child_readiness_hosted_chat_reset_and_cleanup(
    tmp_path, monkeypatch
):
    captures = []
    body = manifest()
    provider, start, work, source = provision(tmp_path, body, captures)
    context = _context(tmp_path)
    contract_path = context.bundle_dir / "contract.json"
    contract_body = json.loads(contract_path.read_text())
    contract_body["runtime"]["interface"] = interface().model_dump()
    contract_body["tools"] = []
    contract_path.write_text(json.dumps(contract_body))
    contract = AgentContract.model_validate(contract_body)
    monkeypatch.setattr(chat, "_tool_world", lambda *_: _ToolWorld())
    monkeypatch.setattr(chat, "_drive_conversation", _single_exchange)
    adapter = _Adapter()
    tokens = []
    runtimes = []
    try:
        runtimes = asyncio.run(start())
        assert len(runtimes) == 2
        tokens = [r._local_auth_values["LOCAL_SERVER_KEY"] for r in runtimes]
        assert len(set(tokens)) == 2 and all(len(v) == 64 for v in tokens)
        for runtime in runtimes:
            assert all(
                token not in runtime.model_dump_json() + repr(runtime)
                for token in tokens
            )
        assert all(
            "LOCAL_SERVER_KEY" not in env
            for cwd, env in captures
            if cwd.name == "sidecar" or "sidecar" in str(cwd)
        )
        live_logs = list((work / "worlds").rglob("*.log"))
        assert live_logs
        assert any("starting" in path.read_text() for path in live_logs)
        assert all(
            token not in path.read_text() for token in tokens for path in live_logs
        )
        first = runtimes[0]
        endpoint = first.endpoints["target_http"].address
        for header in ({}, {"Authorization": "Bearer wrong"}):
            with pytest.raises(HTTPError) as error:
                urlopen(
                    Request(endpoint + "/chat", data=b"{}", headers=header), timeout=3
                )
            assert error.value.code == 401
        with pytest.raises(RuntimeValidationError):
            asyncio.run(
                probe_chat_ingress(
                    interface=interface(local_auth=None, headers_template={}),
                    endpoint=endpoint,
                    authoring=context.bundle_dir,
                    candidate_hash="bad-auth",
                    bundle_digest=body.digest,
                    secret_values={},
                    redaction_values=first._local_auth_values,
                    timeout=3,
                )
            )
        asyncio.run(
            _probe_runtime_readiness(
                job=context.job,
                source=source,
                authoring=context.bundle_dir,
                manifest=body,
                contract=contract,
                provider=provider,
                runtime=first,
                factory=SimpleNamespace(create=AsyncMock(return_value=None)),
                work=work,
                values={},
                redactions={},
                compare_baseline=False,
            )
        )
        assert first._local_auth_values["LOCAL_SERVER_KEY"] == tokens[0]
        outcome = asyncio.run(
            chat.HostedChatCallRunner(adapter, context).run(
                SimpleNamespace(scenario_key="one", scenario_id="scenario-1"),
                first,
            )
        )
        assert outcome.transcript_artifact == "transcript-1"
        assert b"Authenticated <redacted>" in adapter.uploads[0]
        assert all(
            token.encode() not in upload
            for token in tokens
            for upload in adapter.uploads
        )
        for path in (context.bundle_dir / "generic-harness").rglob("*.json"):
            assert all(token not in path.read_text() for token in tokens)
        first.state = pr.RuntimeState.UNHEALTHY
        asyncio.run(start())
        assert first._local_auth_values["LOCAL_SERVER_KEY"] != tokens[0]
        tokens.append(first._local_auth_values["LOCAL_SERVER_KEY"])
    finally:
        asyncio.run(provider.close(work_directory=work))
    assert not (work / "worlds").exists()
    assert all(not runtime._local_auth_values for runtime in runtimes)
    logs = list((work / "diagnostics").rglob("*.log"))
    assert logs
    assert all(token not in path.read_text() for token in tokens for path in logs)
    assert all(token not in body.model_dump_json() for token in tokens)


def test_startup_failure_is_redacted_before_runtime_publication(tmp_path):
    captures = []
    body = manifest(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ['LOCAL_SERVER_KEY'], flush=True); raise SystemExit(1)",
        ]
    )
    provider, start, work, _ = provision(tmp_path, body, captures)
    try:
        with pytest.raises(pr.ProcessRuntimeError) as error:
            asyncio.run(start())
        token = next(
            env["LOCAL_SERVER_KEY"] for _, env in captures if "LOCAL_SERVER_KEY" in env
        )
        assert token not in str(error.value)
        assert "<redacted>" in str(error.value)
        assert not provider._runtimes
    finally:
        asyncio.run(provider.close(work_directory=work))
    assert all(
        token not in path.read_text() for path in (work / "diagnostics").rglob("*.log")
    )


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "command"},
        {"headers_template": {}},
        {"setup_secrets": {"local_auth": "API_KEY"}},
        {"local_auth": {**AUTH, "template_variable": "messages"}},
        {"setup_requests": [{"path": "/setup", "capture": {"local_auth": "id"}}]},
    ],
)
def test_auth_interface_rejects_ambiguous_bindings(change):
    with pytest.raises(ValueError):
        interface(**change)


def test_auth_interface_allows_private_credential_login_exchange():
    configured = interface(
        headers_template={"Authorization": "Bearer {{access_token}}"},
        setup_requests=[{
            "path": "/login", "encoding": "form",
            "body_template": {"username": "local-user", "password": "{{local_auth}}"},
            "capture": {"access_token": "access_token"},
        }],
    )
    assert configured.setup_requests[0].body_template["password"] == "{{local_auth}}"


@pytest.mark.parametrize("field", ["path", "request_template"])
def test_auth_interface_rejects_private_credential_in_urls_or_chat_body(field):
    change = ({"path": "/chat/{{local_auth}}"} if field == "path"
              else {"request_template": {"password": "{{local_auth}}"}})
    with pytest.raises(ValueError, match="url_or_chat_body_forbidden"):
        interface(**change)


def test_auth_rejects_public_config_and_external_alias_collisions(tmp_path):
    with pytest.raises(ValueError, match="name_collision"):
        Runtime(
            interface=interface(),
            config_files={"settings.env": {"values": {"LOCAL_SERVER_KEY": "public"}}},
        )
    with pytest.raises(ValueError, match="native_process"):
        Runtime(
            interface=interface(),
            execution="compose",
            compose_file="compose.yml",
            service="app",
        )
    with pytest.raises(ValueError, match="name_collision"):
        SourceProcess(
            name="app",
            working_directory=".",
            run_command=["app"],
            user="svc-agent",
            local_auth=AUTH,
            environment={"LOCAL_SERVER_KEY": "public"},
        )
    from fi.alk.harness.chat_transport import create_chat_wrapper

    with pytest.raises(ValueError, match="name_collision"):
        create_chat_wrapper(
            interface(),
            "http://localhost:8000",
            secret_values={"LOCAL_SERVER_KEY": "existing"},
            local_auth_values={"LOCAL_SERVER_KEY": "new"},
        )
    with pytest.raises(ValueError, match="token_unavailable"):
        create_chat_wrapper(interface(), "http://localhost:8000")


def test_local_auth_repair_is_additive_and_requires_source_evidence(tmp_path):
    (tmp_path / "app.py").write_text(SERVER)
    contract = AgentContract(
        agent="example",
        runtime=Runtime(
            language="python",
            command=[sys.executable, "app.py"],
            interface=interface(local_auth=None, headers_template={}),
        ),
    )

    def patch(runtime):
        return RuntimePlanPatch(
            base_contract_hash=_contract_hash(contract),
            runtime=runtime,
            evidence_paths=("app.py",),
            diagnostic_codes=("credential_missing",),
            summary="Wire source-owned local server auth.",
        )

    replacement = contract.runtime.model_copy(update={"interface": interface()})
    validate_runtime_plan_patch(
        tmp_path,
        contract,
        patch(replacement),
        allowed_diagnostic_codes={"credential_missing"},
    )
    with pytest.raises(ValueError, match="name_collision"):
        validate_runtime_plan_patch(
            tmp_path,
            contract,
            patch(replacement),
            allowed_diagnostic_codes={"credential_missing"},
            available_secret_aliases=frozenset({"LOCAL_SERVER_KEY"}),
        )
    (tmp_path / "app.py").write_text("pass\n")
    with pytest.raises(ValueError, match="source_evidence"):
        validate_runtime_plan_patch(
            tmp_path,
            contract,
            patch(replacement),
            allowed_diagnostic_codes={"credential_missing"},
        )


def test_local_auth_repair_can_append_source_owned_login_exchange(tmp_path):
    (tmp_path / "app.py").write_text(
        'import os\npassword = os.environ["LOCAL_PASSWORD"]\nLOGIN_PATH = "/login"\n'
    )
    contract = AgentContract(runtime=Runtime(
        language="python", command=["python", "app.py"],
        interface=RuntimeInterface(
            kind="http", protocol="json_template", port=8080, path="/chat",
            request_template={"text": "{{new_message_content}}"}, response_path="content",
        ),
    ))
    runtime = contract.runtime.model_copy(deep=True)
    runtime.interface.local_auth = LocalHttpAuth(
        server_environment="LOCAL_PASSWORD", template_variable="password",
    )
    runtime.interface.setup_requests.append(RuntimeHttpSetupRequest(
        path="/login", encoding="form",
        body_template={"username": "local-user", "password": "{{password}}"},
    ))
    patch = RuntimePlanPatch(
        base_contract_hash=_contract_hash(contract), runtime=runtime,
        evidence_paths=("app.py",), diagnostic_codes=("credential_missing",),
        summary="Use the source-owned local login route.",
    )
    repaired = validate_runtime_plan_patch(
        tmp_path, contract, patch, allowed_diagnostic_codes={"credential_missing"},
    )
    assert repaired.runtime.interface.setup_requests[-1].path == "/login"
    patch.runtime.interface.setup_requests[-1].path = "/invented"
    with pytest.raises(ValueError, match="login_evidence_required"):
        validate_runtime_plan_patch(
            tmp_path, contract, patch, allowed_diagnostic_codes={"credential_missing"},
        )


@pytest.mark.parametrize("required", [False, True])
def test_native_compiler_carries_only_auth_names_and_rejects_supplied_alias(tmp_path, required):
    from fi.alk.harness.bundle_author_v2 import (
        BundleAuthorError,
        resolve_environment_plan,
    )
    from test_bundle_author_v2 import _job

    (tmp_path / "server.py").write_text(SERVER)
    runtime = Runtime(
        language="python", command=["python", "server.py"], interface=interface(),
        required_environment={"agent": ["LOCAL_SERVER_KEY"]} if required else {},
    ).model_dump()
    plan = resolve_environment_plan(
        tmp_path,
        _job(connector="auto"),
        contract_modality="chat",
        contract_interface_kind="http",
        contract_runtime=runtime,
    )
    targets = [
        p for p in plan.processes if isinstance(p, SourceProcess) and p.local_auth
    ]
    assert len(targets) == 1
    assert targets[0].name == "agent"
    assert targets[0].local_auth.model_dump() == AUTH
    assert "LOCAL_SERVER_KEY" not in targets[0].environment
    assert "LOCAL_SERVER_KEY" not in (targets[0].build_environment or {})
    with pytest.raises(BundleAuthorError, match="target_runtime_configuration_missing.*EXTERNAL_API_KEY"):
        resolve_environment_plan(
            tmp_path, _job(connector="auto"), contract_modality="chat",
            contract_interface_kind="http",
            contract_runtime={**runtime, "required_environment": {"agent": ["EXTERNAL_API_KEY"]}},
        )
    with pytest.raises(BundleAuthorError, match="name_collision"):
        resolve_environment_plan(
            tmp_path,
            _job(
                connector="auto",
                with_secrets=True,
                secret_aliases=("LOCAL_SERVER_KEY",),
            ),
            contract_modality="chat",
            contract_interface_kind="http",
            contract_runtime=runtime,
        )
    legacy = targets[0].model_copy(update={"local_auth": None})
    assert "local_auth" not in legacy.model_dump()


def test_missing_external_credential_cannot_be_repaired_by_changing_runtime(tmp_path):
    (tmp_path / "app.py").write_text(
        "import os\nexternal_key = os.environ['EXTERNAL_KEY']\n"
    )
    contract = AgentContract(
        agent="example",
        runtime=Runtime(
            command=["python", "app.py"],
            interface=interface(local_auth=None, headers_template={}),
        ),
    )
    patch = RuntimePlanPatch(
        base_contract_hash=_contract_hash(contract),
        runtime=contract.runtime,
        evidence_paths=("app.py",),
        diagnostic_codes=("credential_missing",),
        summary="Cannot invent external credentials.",
    )
    with pytest.raises(ValueError, match="addition_required"):
        validate_runtime_plan_patch(
            tmp_path, contract, patch, allowed_diagnostic_codes={"credential_missing"}
        )


def test_supplied_alias_collision_stops_before_token_generation_or_spawn(tmp_path):
    body = manifest()
    calls = []
    context = pr.SpawnContext(
        work_directory=tmp_path,
        port_plan=pr.plan_ports(body, instances=1),
        credentials={},
        secret_values={"LOCAL_SERVER_KEY": "existing-account-secret"},
        secret_purposes={"LOCAL_SERVER_KEY": "simulator"},
        runner=lambda *args, **kwargs: calls.append(args),
    )
    with pytest.raises(ValueError, match="name_collision"):
        pr.spawn_world(body, world_index=0, context=context)
    assert not calls
    assert not context.local_auth_redactions


@pytest.mark.parametrize("split", range(1, 64))
def test_log_redaction_handles_every_split_before_persistence(split):
    import io

    token = "0123456789abcdef" * 4

    class Input(io.BytesIO):
        def __init__(self):
            super().__init__()
            self.chunks = iter(
                [
                    b"ready\n",
                    token[:split].encode(),
                    token[split:].encode() + b"\n",
                    b"last line\n",
                    b"",
                ]
            )

        def read1(self, _size):
            return next(self.chunks)

    class Output(io.BytesIO):
        def __init__(self):
            super().__init__()
            self.snapshots = []

        def flush(self):
            self.snapshots.append(self.getvalue())

        def close(self):
            self.result = self.getvalue()
            super().close()

    output = Output()
    pr._copy_redacted_output(Input(), output, {"local": token})
    assert output.snapshots[0] == b"ready\n"
    assert output.result == b"ready\n<redacted>\nlast line\n"
    assert all(token.encode() not in value for value in output.snapshots)
