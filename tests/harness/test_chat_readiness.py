from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from fi.alk.harness.bundle_v2 import BundleRuntimeV2, EvidenceSeam, RuntimeKindV2
from fi.simulate.runtime.spec import RuntimeRequirements
from urllib.request import urlopen

import pytest

from fi.alk.harness.authoring_runtime_validation import RuntimeValidationError
from fi.alk.harness.chat_readiness import probe_chat_ingress
from fi.alk.harness.contract import AgentContract, Runtime, RuntimeInterface
from fi.alk.harness.runtime_repair import (
    RuntimePlanPatch,
    _contract_hash,
    apply_runtime_plan_patch,
)


@pytest.fixture
def profile_app():
    state = {
        "profile": False,
        "calls": [],
        "status": None,
        "body": {"content": "Hello"},
        "delay": 0,
        "completed": threading.Event(),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"healthy")

        def do_POST(self):
            payload = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
            )
            state["calls"].append((self.path, payload))
            time.sleep(state["delay"])
            if self.path == "/profiles":
                state["profile"] = True
                status, body = 201, {"id": "profile"}
            else:
                status = state["status"] or (200 if state["profile"] else 404)
                body = (
                    state["body"]
                    if status == 200
                    else {"error": "missing profile secret123"}
                )
            self.send_response(status)
            self.send_header("Content-Length", str(len(json.dumps(body).encode())))
            self.end_headers()
            try:
                self.wfile.write(json.dumps(body).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                state["completed"].set()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", state
    server.shutdown()
    thread.join()
    server.server_close()


def interface(**changes):
    return RuntimeInterface(kind="http", path="/chat", port=8080, **changes)


def probe(tmp_path, url, **changes):
    args = dict(
        interface=interface(),
        endpoint=url,
        document={
            "scenario_key": "one",
            "instruction": "Do not send this",
            "persona": {"initial_message": "Hi there"},
        },
        authoring=tmp_path,
        candidate_hash="candidate",
        bundle_digest="bundle",
        secret_values={},
        redaction_values={"key": "secret123"},
        timeout=1,
    )
    args.update(changes)
    return probe_chat_ingress(**args)


def receipts(path):
    return [
        json.loads(p.read_text())
        for p in (path / "generic-harness/chat-readiness").glob("*.json")
    ]


@pytest.mark.parametrize("stream", ["stdout", "stderr", "both"])
@pytest.mark.parametrize("verbose", [False, True])
def test_command_failure_retains_both_streams_without_leaking_credentials(
    tmp_path, monkeypatch, capsys, stream, verbose
):
    import sys

    from fi.alk.harness.bundle_author_v2 import _subprocess_adapter_source

    secret = "first-secret-line\nsecond-secret-line"
    output = (
        "build progress\n" * 400 + "noise" * 1000 + "\n" + "X" * 1990 + secret + "\n"
        if verbose else secret + "\n"
    )
    statements = ["import sys"]
    for name in ("stdout", "stderr"):
        if stream in (name, "both"):
            statements.append(
                f"sys.{name}.write({output!r} + 'Configure the local provider ({name}).\\n')"
            )
    statements.append("sys.exit(1)")
    monkeypatch.setenv(
        "ALK_SUBPROCESS_COMMAND", json.dumps([sys.executable, "-c", ";".join(statements)])
    )
    namespace = {"__name__": "compiled_command_bridge"}
    exec(_subprocess_adapter_source(), namespace)
    server = ThreadingHTTPServer(("127.0.0.1", 0), namespace["Handler"])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(RuntimeValidationError) as failure:
            asyncio.run(probe(
                tmp_path, f"http://127.0.0.1:{server.server_port}",
                interface=RuntimeInterface(kind="command"),
                redaction_values={"credential": secret},
            ))
        evidence = str(failure.value) + json.dumps(receipts(tmp_path))
        assert "submitted command exited 1" in evidence
        for name in ("stdout", "stderr"):
            if stream in (name, "both"):
                assert f"Configure the local provider ({name})." in evidence
                assert f"{name}:" in evidence
        assert "first-secret-line" not in evidence
        assert "second-secret-line" not in evidence
        process_log = capsys.readouterr().err
        assert "first-secret-line" not in process_log
        assert "second-secret-line" not in process_log
        assert len(str(failure.value)) <= 4000
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_healthy_server_requires_profile_then_typed_runtime_patch_passes(
    tmp_path, profile_app
):
    url, state = profile_app
    with urlopen(url + "/health") as response:
        assert response.status == 200
    with pytest.raises(RuntimeValidationError) as error:
        asyncio.run(probe(tmp_path, url))
    assert error.value.diagnostics, str(error.value)
    assert error.value.diagnostics[0].code == "chat_ingress_configuration_invalid"
    assert receipts(tmp_path)[0]["trace"]["status_code"] == 404
    assert "secret123" not in json.dumps(receipts(tmp_path))
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text('PROFILE_ROUTE = "/profiles"\n')
    contract = AgentContract(
        runtime=Runtime(
            language="python", command=["python", "app.py"], interface=interface()
        )
    )
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(contract.model_dump_json())
    patched = contract.runtime.model_copy(
        update={
            "interface": interface(
                setup_requests=[{"path": "/profiles", "body_template": {}}]
            )
        }
    )
    apply_runtime_plan_patch(
        source,
        contract_path,
        RuntimePlanPatch(
            base_contract_hash=_contract_hash(contract),
            runtime=patched,
            evidence_paths=("app.py",),
            diagnostic_codes=("chat_ingress_configuration_invalid",),
            summary="Provision the source-owned profile before chat",
        ),
        allowed_diagnostic_codes={"chat_ingress_configuration_invalid"},
    )
    repaired = AgentContract.model_validate_json(contract_path.read_text())
    asyncio.run(probe(tmp_path, url, interface=repaired.runtime.interface))
    assert sorted(r["status"] for r in receipts(tmp_path)) == ["failed", "passed"]
    assert [path for path, _ in state["calls"]] == ["/chat", "/profiles", "/chat"]
    assert state["calls"][-1][1]["new_message"]["content"] == "Hi there"
    assert state["calls"][-1][1]["thread_id"] == "one"


@pytest.mark.parametrize(
    "status,code",
    [
        (400, "chat_ingress_configuration_invalid"),
        (404, "chat_ingress_configuration_invalid"),
        (405, "chat_ingress_configuration_invalid"),
        (415, "chat_ingress_configuration_invalid"),
        (422, "chat_ingress_configuration_invalid"),
        (401, "credential_missing"),
        (403, "credential_missing"),
        (408, "external_service_unavailable"),
        (429, "external_service_unavailable"),
        (500, "chat_ingress_failed"),
    ],
)
def test_failure_routing(tmp_path, profile_app, status, code):
    url, state = profile_app
    state["status"] = status
    with pytest.raises(RuntimeValidationError) as error:
        asyncio.run(probe(tmp_path, url))
    assert error.value.diagnostics[0].code == code


@pytest.mark.parametrize(
    "body,passed",
    [
        ({"content": "  "}, False),
        ({"error": "failed"}, False),
        ({"content": "I cannot do that"}, True),
        ({"tool_calls": [{"name": "lookup", "arguments": {}}]}, True),
    ],
)
def test_response_acceptance_without_tool_replay(tmp_path, profile_app, body, passed):
    url, state = profile_app
    state.update(profile=True, body=body)
    if passed:
        asyncio.run(probe(tmp_path, url))
    else:
        with pytest.raises(RuntimeValidationError):
            asyncio.run(probe(tmp_path, url))
    assert len(state["calls"]) == 1


def test_cancellation_drains_http_worker_before_return(tmp_path, profile_app):
    url, state = profile_app
    state.update(profile=True, delay=0.08)

    async def run():
        task = asyncio.create_task(probe(tmp_path, url))
        while not state["calls"]:
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert state["completed"].is_set()

    asyncio.run(run())
    assert receipts(tmp_path)[0]["status"] == "cancelled"


def test_setup_and_chat_share_deadline(tmp_path, profile_app):
    url, state = profile_app
    state["delay"] = 0.07
    with pytest.raises(RuntimeValidationError) as error:
        asyncio.run(
            probe(
                tmp_path,
                url,
                interface=interface(setup_requests=[{"path": "/profiles"}]),
                timeout=0.1,
            )
        )
    assert error.value.diagnostics[0].code == "chat_ingress_timeout"
    assert error.value.diagnostics[0].retryable is True
    assert "0.1" in str(error.value)
    receipt = receipts(tmp_path)[0]
    assert receipt["status"] == "failed"
    assert receipt["timeout_seconds"] == 0.1
    assert receipt["trace"]["timeout_error"] is True


@pytest.mark.parametrize("readiness_only", [False, True])
@pytest.mark.parametrize("required_workflow", [False, True])
def test_validation_rejects_healthy_unprovisioned_chat_before_invariants(
    tmp_path, monkeypatch, profile_app, readiness_only, required_workflow
):
    from fi.alk.harness import authoring_runtime_validation as validation
    from fi.alk.harness import (
        bundle_author_v2,
        certification,
        hosted_entrypoint,
        process_preflight,
        process_runtime,
        scenario_source,
        source_data_invariants,
    )
    from fi.alk.harness.certification import CheckStatus

    url, state = profile_app
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text('PROFILE_ROUTE = "/profiles"\n')
    authoring = tmp_path / "authoring"
    authoring.mkdir()
    secrets = tmp_path / "secrets.json"
    secrets.write_text('{"key": "secret123"}')
    log_dir = authoring / "generic-harness/process-logs"
    log_dir.mkdir(parents=True)
    (log_dir / "old.log").write_text("unrelated old attempt")
    contract = AgentContract(
        runtime=Runtime(
            language="python", command=["python", "app.py"], interface=interface()
        )
    )
    contract_path = authoring / "contract.json"
    contract_path.write_text(contract.model_dump_json())
    calls = []
    evidence = {}

    class Store:
        RUNTIME_EVIDENCE = "runtime-evidence.json"
        CERTIFICATION = "certification.json"

        def __init__(self, root):
            pass

        def read_source_model(self):
            return SimpleNamespace(actions=(), fingerprint="schema", engine="none")

        def read_world_ir(self):
            return SimpleNamespace(fingerprint="world")

        def write_runtime_evidence(self, value):
            evidence["value"] = value

        def write_tool_certification(self, value):
            pass

        def write_action_certification(self, value):
            pass

        def write_repair_history(self, value):
            pass

    class Provider:
        def __init__(self, **kwargs):
            pass

        async def provision(self, manifest, **kwargs):
            calls.append("provision")
            artifacts = kwargs["work_directory"] / "artifacts"
            artifacts.mkdir()
            (artifacts / "build.json").write_text('{"conformance":true}')
            return [
                SimpleNamespace(world_index=0, endpoints={"target_http": SimpleNamespace(address=url)})
            ]

        async def reset(self, *args, **kwargs):
            calls.append("reset")
            state["profile"] = False

        async def clone_world_database(self, *_args, **_kwargs):
            return None

        async def close(self, **kwargs):
            from fi.alk.harness.process_diagnostics import retain_process_log

            calls.append("close")
            work = kwargs["work_directory"]
            log = work / "process.log"
            log.write_text("current attempt detail secret123")
            retain_process_log(log, work=work, destination=authoring / "generic-harness", secrets={"key": "secret123"})

    class World:
        def read_only(self):
            return self

        def state(self):
            return {}

        def checkpoint(self):
            return "baseline"

    class Factory:
        def __init__(self, work, bundle):
            pass

        async def create(self, *args, **kwargs):
            return World()

    def bundle(**kwargs):
        target = kwargs["output"]
        assert kwargs["include_scenarios"] is not readiness_only
        target.mkdir(parents=True)
        if not readiness_only:
            keys = (
                [("00-refusal", "refusal"), ("01-one", "one")]
                if required_workflow
                else [("one", "one")]
            )
            for folder, key in keys:
                scenario = target / f"scenarios/{folder}"
                scenario.mkdir(parents=True)
                (scenario / "scenario.json").write_text(
                    json.dumps({"scenario_key": key})
                )
        return SimpleNamespace(
            digest="bundle", provenance=SimpleNamespace(source_digest="source"),
            runtime=BundleRuntimeV2(kind=RuntimeKindV2.PROCESS, evidence_seam=EvidenceSeam.HTTP_TOOL),
        )

    async def invariants(*args, **kwargs):
        calls.append("invariants")
        assert state["profile"] is False
        return []

    async def check(*args, **kwargs):
        pass

    def freeze(*args):
        calls.append("freeze")
        assert state["profile"] is False
        assert evidence["value"].checks.chat_ingress is CheckStatus.PASSED

    monkeypatch.setattr(certification, "GenericHarnessArtifactStore", Store)
    monkeypatch.setattr(process_runtime, "ProcessRuntimeProvider", Provider)
    monkeypatch.setattr(hosted_entrypoint, "ProcessWorldFactory", Factory)
    monkeypatch.setattr(bundle_author_v2, "author_bundle_v2", bundle)
    monkeypatch.setattr(process_preflight, "preflight_bundle", lambda *a, **k: None)
    monkeypatch.setattr(
        scenario_source,
        "load_scenarios",
        lambda *a: ([SimpleNamespace(
            scenario_key="refusal", scenario_id="refusal-id", requires_tool_evidence=False,
            setup=lambda w: calls.append("refusal-setup"), ready=lambda w: True,
        )] if required_workflow else []) + [
            SimpleNamespace(
                scenario_key="one",
                scenario_id="scenario-id",
                requires_tool_evidence=required_workflow,
                setup=lambda w: calls.append("setup"),
                ready=lambda w: True,
            )
        ],
    )
    monkeypatch.setattr(source_data_invariants, "author_invariants", invariants)
    monkeypatch.setattr(source_data_invariants, "check_invariants", check)
    monkeypatch.setattr(validation, "_freeze_certified_bundle", freeze)
    workflow_selections = []

    async def workflow(**kwargs):
        workflow_selections.append(kwargs["scenario"].scenario_key)
        assert kwargs["scenario"].requires_tool_evidence
        current = AgentContract.model_validate_json(contract_path.read_text())
        await probe(
            authoring, url, interface=current.runtime.interface,
            scenario_key=kwargs["scenario"].scenario_key,
        )

    from fi.alk.harness import chat_readiness
    monkeypatch.setattr(chat_readiness, "probe_chat_workflow", workflow)
    job = SimpleNamespace(
        source=SimpleNamespace(kind=None),
        agent=SimpleNamespace(secret_refs={}),
        metadata={"generic_harness_v1": True},
        scenario_count=2 if required_workflow else 1,
        runtime=RuntimeRequirements(),
        seed=0,
    )

    async def validate(*args):
        return await validation.validate_once(
            job, source, authoring, secrets_path=secrets, local_runtime=True,
            readiness_only=readiness_only,
        )

    with pytest.raises(RuntimeValidationError) as error:
        asyncio.run(validate())
    assert error.value.diagnostics, str(error.value)
    assert error.value.diagnostics[0].code == "chat_ingress_configuration_invalid"
    assert calls == (["provision", "reset", "close"] if readiness_only
                     else ["provision", "reset", "setup", "close"])
    assert "current attempt detail" in error.value.process_evidence
    assert "secret123" not in error.value.process_evidence
    assert "unrelated old attempt" not in error.value.process_evidence
    assert len(receipts(authoring)) == 1
    repaired_runtime = contract.runtime.model_copy(
        update={"interface": interface(setup_requests=[{"path": "/profiles"}])}
    )
    from fi.alk.harness import runtime_repair

    repairs = []

    async def request_patch(source_path, current_contract, diagnostics, **kwargs):
        assert "current attempt detail" in kwargs["guidance"]
        assert "secret123" not in kwargs["guidance"]
        assert "unrelated old attempt" not in kwargs["guidance"]
        repairs.append(tuple(item.code for item in diagnostics))
        return RuntimePlanPatch(
            base_contract_hash=_contract_hash(current_contract),
            runtime=repaired_runtime,
            evidence_paths=("app.py",),
            diagnostic_codes=("chat_ingress_configuration_invalid",),
            summary="Provision source profile",
        )

    monkeypatch.setattr(runtime_repair, "request_runtime_plan_patch", request_patch)
    asyncio.run(
        validation.validate_and_repair(
            job, source, authoring, validate=validate, readiness_only=readiness_only
        )
    )
    assert repairs == [("chat_ingress_configuration_invalid",)]
    if readiness_only:
        assert calls[-2:] == ["reset", "close"]
        assert "setup" not in calls and "invariants" not in calls and "freeze" not in calls
        assert evidence == {}
        assert not (authoring / "runtime-validation.json").exists()
        assert not (authoring / "generic-harness/certification.json").exists()
        assert not (authoring / "scenarios").exists()
        assert all("scenario_key" not in item for item in receipts(authoring))
        assert json.loads((authoring / "generic-harness/runtime-readiness.json").read_text())["reset"] == "equivalent"
        assert (authoring / "generic-harness/readiness-repair-history.json").exists()
    else:
        assert calls[-2:] == ["freeze", "close"]
    assert workflow_selections == (["one", "one", "one"] if required_workflow and not readiness_only else [])
    assert sorted(item["status"] for item in receipts(authoring)) == [
        "failed",
        "failed",
        "passed",
    ]


def test_declared_application_error_and_extraction_fail_terminal(tmp_path, profile_app):
    url, state = profile_app
    state.update(profile=True, body={"status": "error", "message": "failed secret123"})
    for target in (
        interface(
            failure_rule={
                "status_path": "status",
                "failure_values": ["error"],
                "message_path": "message",
            }
        ),
        interface(
            protocol="json_template", request_template={}, response_path="missing"
        ),
    ):
        with pytest.raises(RuntimeValidationError) as error:
            asyncio.run(probe(tmp_path, url, interface=target))
        assert error.value.diagnostics[0].code == "chat_ingress_failed"
    assert "secret123" not in json.dumps(receipts(tmp_path))


def test_response_path_must_select_chat_content_not_message_envelope(tmp_path, profile_app):
    url, state = profile_app
    state.update(profile=True, body={"response": {"role": "assistant", "content": "Hello"}})
    target = interface(protocol="json_template", request_template={}, response_path="response")
    with pytest.raises(RuntimeValidationError) as error:
        asyncio.run(probe(tmp_path, url, interface=target))
    assert error.value.diagnostics[0].code == "chat_ingress_failed"
    target = target.model_copy(update={"response_path": "response.content"})
    asyncio.run(probe(tmp_path, url, interface=target))
    assert len(state["calls"]) == 2


def test_receipt_redaction_preserves_controller_identity(tmp_path, profile_app):
    url, state = profile_app
    with pytest.raises(RuntimeValidationError):
        asyncio.run(
            probe(
                tmp_path,
                url,
                candidate_hash="sha256:111",
                bundle_digest="sha256:123",
                redaction_values={"config": "1", "token": "secret123"},
            )
        )
    receipt = receipts(tmp_path)[0]
    assert receipt["candidate_hash"] == "sha256:111"
    assert receipt["bundle_digest"] == "sha256:123"
    assert receipt["diagnostic"]["code"] == "chat_ingress_configuration_invalid"
    assert "secret123" not in json.dumps(receipt)


@pytest.mark.parametrize(
    "calls,passed",
    [
        ([{}], False),
        ([{"name": " "}], False),
        ([{"name": "lookup", "arguments": "not-json"}], False),
        ([{"name": "lookup", "arguments": "[]"}], False),
        ([{"name": "lookup", "arguments": []}], False),
        ([{"name": "lookup"}, {}], False),
        ([{"name": "lookup"}], True),
        ([{"name": "lookup", "arguments": {"id": 1}}], True),
        ([{"function": {"name": "lookup", "arguments": '{"id":1}'}}], True),
    ],
)
@pytest.mark.parametrize("content", ["", "Here is the answer"])
def test_all_tool_calls_must_be_usable_even_with_text(
    tmp_path, profile_app, calls, passed, content
):
    url, state = profile_app
    state.update(profile=True, body={"content": content, "tool_calls": calls})
    if passed:
        asyncio.run(probe(tmp_path, url))
    else:
        with pytest.raises(RuntimeValidationError) as error:
            asyncio.run(probe(tmp_path, url))
        assert error.value.diagnostics[0].code == "chat_ingress_failed"
    assert len(state["calls"]) == 1


@pytest.mark.parametrize("recovers", [True, False])
def test_dependency_patch_repeats_identical_http_request_and_stops_without_new_evidence(tmp_path, profile_app, recovers):
    from fi.alk.harness.authoring_runtime_validation import validate_and_repair

    url, state = profile_app
    state.update(profile=True)
    source = tmp_path / "source"
    authoring = tmp_path / "authoring"
    source.mkdir()
    authoring.mkdir()
    (source / "app.py").write_text("# immutable source")
    (source / "pyproject.toml").write_text(
        '[project]\nname="fixture"\nversion="1"\n'
        '[project.optional-dependencies]\nrenamed_transport=["declared-package"]\n'
    )
    (authoring / "scenarios.json").write_text('{"original": true}')
    target = interface(protocol="json_template", request_template={"messages": "{{messages}}"}, response_path="response")
    contract = AgentContract(agent="fixture", runtime=Runtime(
        language="python", execution="native", command=["python", "app.py"], interface=target,
    ))
    (authoring / "contract.json").write_text(contract.model_dump_json())
    snapshots = {path: path.read_bytes() for path in source.iterdir()}
    repairs = []
    builds = []

    async def validate(_job, _source, _authoring):
        current = AgentContract.model_validate_json((authoring / "contract.json").read_text())
        builds.append(current.runtime.extras)
        state["body"] = (
            {"response": "Ready"} if current.runtime.extras and recovers
            else {"error": "Cannot import declared-package secret123"}
        )
        await probe(authoring, url, interface=current.runtime.interface, candidate_hash=_contract_hash(current))
        return 1

    async def repair(_phase, guidance):
        assert "Cannot import declared-package" in guidance
        assert "secret123" not in guidance
        repairs.append(guidance)
        current = AgentContract.model_validate_json((authoring / "contract.json").read_text())
        if current.runtime.extras:
            return None
        return RuntimePlanPatch(
            base_contract_hash=_contract_hash(contract),
            runtime=contract.runtime.model_copy(update={"extras": ["renamed_transport"]}),
            evidence_paths=("pyproject.toml",), diagnostic_codes=("chat_ingress_failed",),
            summary="Investigate the declared transport dependency group.",
        )

    run = validate_and_repair(SimpleNamespace(metadata={"generic_harness_v1": True}), source, authoring, validate=validate, repair=repair)
    if recovers:
        asyncio.run(run)
    else:
        with pytest.raises(RuntimeValidationError):
            asyncio.run(run)
    assert len(repairs) == (1 if recovers else 2)
    assert builds == [[], ["renamed_transport"]]
    assert len(state["calls"]) == 2
    assert state["calls"][0] == state["calls"][1]
    assert sorted(item["status"] for item in receipts(authoring)) == (["failed", "passed"] if recovers else ["failed", "failed"])
    assert (authoring / "generic-harness/runtime-plan-repair-1.json").is_file()
    assert all(path.read_bytes() == body for path, body in snapshots.items())
    assert (authoring / "scenarios.json").read_text() == '{"original": true}'


@pytest.mark.parametrize("kind,code", [
    ("timeout", "chat_ingress_timeout"),
    ("wrapped_timeout", "chat_ingress_timeout"),
    ("url_error_text", "external_service_unavailable"),
    ("connection", "external_service_unavailable"),
    ("value_error_text", "chat_ingress_failed"),
])
def test_ingress_classifies_timeout_type_not_message(tmp_path, monkeypatch, kind, code):
    import urllib.error
    import urllib.request

    errors = {
        "timeout": TimeoutError("deadline reached"),
        "wrapped_timeout": urllib.error.URLError(TimeoutError("deadline reached")),
        "url_error_text": urllib.error.URLError("timed out"),
        "connection": ConnectionError("timed out"),
        "value_error_text": ValueError("timed out"),
    }

    def fail(*args, **kwargs):
        raise errors[kind]

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", fail)
    with pytest.raises(RuntimeValidationError) as failure:
        asyncio.run(probe(tmp_path, "http://127.0.0.1:1"))
    assert failure.value.diagnostics[0].code == code
    receipt = receipts(tmp_path)[0]
    assert receipt["timeout_seconds"] == 1
    assert receipt["trace"]["timeout_error"] is (code == "chat_ingress_timeout")


@pytest.mark.parametrize("reset_mode", ["restore", "leave_mutated", "error"])
def test_internal_readiness_initializes_real_chat_and_requires_reset(tmp_path, profile_app, reset_mode):
    from fi.alk.harness.authoring_runtime_validation import _probe_runtime_readiness

    url, state = profile_app
    source = tmp_path / "source"
    source.mkdir()
    authoring = tmp_path / "authoring"
    authoring.mkdir()
    contract = AgentContract(runtime=Runtime(
        language="python", command=["python", "app.py"],
        interface=interface(setup_requests=[{"path": "/profiles"}]),
    ))

    class World:
        def read_only(self):
            return self

        def state(self):
            return {"profiles": [{"present": state["profile"]}]}

    class Factory:
        async def create(self, *args, **kwargs):
            return World()

    class Provider:
        async def reset(self, *args, **kwargs):
            if reset_mode == "error":
                raise RuntimeError("cannot reset")
            if reset_mode == "restore":
                state["profile"] = False

    async def run():
        await _probe_runtime_readiness(
            job=SimpleNamespace(seed=0), source=source, authoring=authoring,
            manifest=SimpleNamespace(digest="bundle"), contract=contract,
            provider=Provider(), runtime=SimpleNamespace(
                endpoints={"target_http": SimpleNamespace(address=url)}
            ), factory=Factory(), work=tmp_path, values={}, redactions={},
        )

    if reset_mode == "restore":
        asyncio.run(run())
    else:
        with pytest.raises(RuntimeValidationError, match="reset"):
            asyncio.run(run())
    assert [path for path, _ in state["calls"]] == ["/profiles", "/chat"]
    assert all("scenario_key" not in item for item in receipts(authoring))
    receipt = json.loads((authoring / "generic-harness/runtime-readiness.json").read_text())
    assert receipt["status"] == ("passed" if reset_mode == "restore" else "failed")
    assert not (authoring / "scenarios").exists()
    assert not (authoring / "runtime-validation.json").exists()


def workflow_probe(tmp_path, url, **changes):
    from fi.alk.harness.chat_readiness import probe_chat_workflow

    bundle = tmp_path / "bundle"
    bundle.mkdir(exist_ok=True)
    (bundle / "contract.json").write_text(AgentContract(
        runtime=Runtime(language="python", command=["python", "app.py"], interface=interface()),
    ).model_dump_json())
    scenario_dir = bundle / "scenarios" / "private"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    (scenario_dir / "scenario.json").write_text(json.dumps({
        "scenario_key": "private", "name": "private", "instruction": "Look up the account",
    }))
    args = dict(
        job=SimpleNamespace(metadata={}), source=tmp_path, bundle=bundle, authoring=tmp_path,
        runtime=SimpleNamespace(
            metadata={"observation_mode": "native"}, world_index=0,
            endpoints={"target_http": SimpleNamespace(address=url, protocol="http")},
        ),
        scenario=SimpleNamespace(scenario_key="private", scenario_id="private-id"),
        candidate_hash="candidate", bundle_digest="bundle", secret_values={},
        redaction_values={"key": "secret123"}, timeout=1,
    )
    args.update(changes)
    return probe_chat_workflow(**args)


def workflow_receipt(tmp_path):
    paths = list((tmp_path / "generic-harness/chat-readiness").glob("*/receipt.json"))
    assert len(paths) == 1
    return paths[0], json.loads(paths[0].read_text())


@pytest.mark.parametrize("credential", ["known-value", '{"enabled": true}'])
def test_private_tool_artifacts_preserve_json_lines_during_redaction(tmp_path, credential):
    from fi.alk.harness.chat_readiness import _PrivateWorkflowArtifacts
    from fi.alk.harness.outbound import ArtifactKind

    rows = [
        {"name": "lookup", "result": {"detail": "Client(api_key='unlisted-key', value='known-value')",
                                     "password": "unlisted-password", "active": True}},
        {"name": "finish", "arguments": ["known-value", {"access_token": "unlisted-token"}],
         "result": False},
    ]
    artifacts = _PrivateWorkflowArtifacts(tmp_path, {"credential": credential})
    path = asyncio.run(artifacts.upload_artifact(
        "\n".join(json.dumps(row) for row in rows).encode(), kind=ArtifactKind.TOOL_TRACE,
    ))
    text = Path(path).read_text()
    saved = [json.loads(line) for line in text.splitlines()]
    assert [row["name"] for row in saved] == ["lookup", "finish"]
    assert saved[0]["result"]["active"] == ("<redacted>" if "enabled" in credential else True)
    assert saved[1]["result"] is False
    assert saved[0]["result"]["password"] == "<redacted>"
    assert saved[1]["arguments"][1]["access_token"] == "<redacted>"
    assert all(value not in text for value in ("unlisted-key", "unlisted-password", "unlisted-token"))
    if credential == "known-value":
        assert "known-value" not in text
    assert Path(path).stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("kind", ["trace", "transcript"])
@pytest.mark.parametrize("redactions", [None, {"credential": "x" * (1024 * 1024 + 1)}])
def test_private_json_artifacts_fail_closed_without_safe_credential_context(tmp_path, kind, redactions):
    from fi.alk.harness.chat_readiness import _PrivateWorkflowArtifacts
    from fi.alk.harness.outbound import ArtifactKind

    artifacts = _PrivateWorkflowArtifacts(tmp_path, redactions)
    path = asyncio.run(artifacts.upload_artifact(
        json.dumps({"messages": [{"content": "unknown-secret", "credential": "other-secret"}],
                    "success": True}, indent=2).encode(),
        kind=ArtifactKind(kind),
    ))
    text = Path(path).read_text()
    json.loads(text)
    assert "unknown-secret" not in text
    assert "other-secret" not in text
    assert "credential context" in text


@pytest.mark.parametrize("completed", [False, True])
def test_private_workflow_checks_executed_tools_after_clarification(
    tmp_path, profile_app, monkeypatch, completed
):
    from fi.alk.harness import chat_call_runner
    from fi.alk.harness.run.conversation import Exchange, Transcript

    url, state = profile_app
    state.update(profile=True, body={"content": "Which account? secret123"})
    observed = []

    async def conversation(target, *_args):
        first = await target.say("Look up my account")
        state["body"] = {"content": "The account is active secret123"}
        if completed:
            state["body"].update(
                tool_calls=[{"id": "lookup-1", "name": "lookup_account", "arguments": {"account_id": "A-1"}}],
                tool_responses=[{"tool_call_id": "lookup-1", "result": {"active": True}}],
            )
        second = await target.say("Account A-1")
        observed.extend(target.world.calls)
        return Transcript(
            exchanges=[Exchange("customer", "Look up my account"), Exchange("agent", first),
                       Exchange("customer", "Account A-1"), Exchange("agent", second)],
            calls=list(target.world.calls), ended="finished",
        )

    monkeypatch.setattr(chat_call_runner, "_drive_conversation", conversation)
    if completed:
        asyncio.run(workflow_probe(tmp_path, url))
        assert [(call.name, call.arguments, call.result) for call in observed] == [
            ("lookup_account", {"account_id": "A-1"}, {"active": True})
        ]
    else:
        with pytest.raises(RuntimeValidationError) as failure:
            asyncio.run(workflow_probe(tmp_path, url))
        assert failure.value.diagnostics[0].code == "chat_tool_evidence_unverified"
        assert "Which account?" in str(failure.value)
        assert "secret123" not in str(failure.value)
    assert [request[1]["new_message"]["content"] for request in state["calls"]] == [
        "Look up my account", "Account A-1",
    ]
    receipt_path, receipt = workflow_receipt(tmp_path)
    assert receipt["status"] == ("passed" if completed else "failed")
    assert receipt["tool_call_count"] == int(completed)
    assert receipt["candidate_hash"] == "candidate"
    assert receipt["bundle_digest"] == "bundle"
    assert receipt["scenario_key"] == "private"
    assert receipt["turns"] == 4
    assert receipt_path.parent.stat().st_mode & 0o777 == 0o700
    for path in receipt_path.parent.iterdir():
        assert "secret123" not in path.read_text()
        assert path.stat().st_mode & 0o777 == 0o600
    assert any("transcript" in name for name in receipt["artifacts"])
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("status,code", [(401, "credential_missing"), (422, "chat_ingress_configuration_invalid")])
def test_private_workflow_preserves_transport_classification(tmp_path, profile_app, monkeypatch, status, code):
    from fi.alk.harness import chat_call_runner

    url, state = profile_app
    state["status"] = status

    async def conversation(target, *_args):
        await target.say("Look up my account")

    monkeypatch.setattr(chat_call_runner, "_drive_conversation", conversation)
    with pytest.raises(RuntimeValidationError) as failure:
        asyncio.run(workflow_probe(tmp_path, url))
    assert failure.value.diagnostics[0].code == code
    receipt_path, receipt = workflow_receipt(tmp_path)
    assert receipt["status"] == "failed"
    assert any("trace" in name for name in receipt["artifacts"])
    assert all("secret123" not in path.read_text() for path in receipt_path.parent.iterdir())


@pytest.mark.parametrize("cancel", [False, True])
def test_private_workflow_deadline_and_cancellation_keep_completed_evidence(tmp_path, profile_app, monkeypatch, cancel):
    import importlib

    conversation = importlib.import_module("fi.alk.harness.run.conversation")
    url, state = profile_app
    state.update(profile=True, body={
        "content": "The account is active secret123",
        "tool_calls": [{"id": "lookup-1", "name": "lookup_account", "arguments": {"account_id": "A-1"}}],
        "tool_responses": [{"tool_call_id": "lookup-1", "result": {"active": True, "credential": "secret123"}}],
    })
    started = asyncio.Event()
    closed = []

    class CustomerStage:
        spent_usd = 0.0
        simulator_tokens = (12, 4)

        def __init__(self, *_args, **_kwargs):
            self.opening = True

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            closed.append(True)

        async def say(self, _message):
            if self.opening:
                self.opening = False
                return SimpleNamespace(text="Look up account A-1")
            started.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(conversation, "Stage", CustomerStage)

    async def run():
        task = asyncio.create_task(workflow_probe(tmp_path, url, timeout=2))
        if cancel:
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(RuntimeValidationError) as failure:
                await task
            assert failure.value.diagnostics[0].code == "chat_ingress_timeout"

    asyncio.run(run())
    assert len(state["calls"]) == 1
    assert closed == [True]
    assert started.is_set()
    receipt_path, receipt = workflow_receipt(tmp_path)
    assert receipt["status"] == ("cancelled" if cancel else "failed")
    assert receipt["tool_call_count"] == 1
    assert receipt["turns"] == 2
    transcript = next(receipt_path.parent.glob("transcript-*.json"))
    assert [
        {key: message[key] for key in ("role", "content")}
        for message in json.loads(transcript.read_text())["messages"]
    ] == [
        {"role": "user", "content": "Look up account A-1"},
        {"role": "assistant", "content": "The account is active <redacted>"},
    ]
    tool_trace = next(receipt_path.parent.glob("tool_trace-*.json"))
    call = json.loads(tool_trace.read_text())
    assert call["name"] == "lookup_account"
    assert call["arguments"] == {"account_id": "A-1"}
    assert call["result"]["active"] is True
    assert all("secret123" not in path.read_text() for path in receipt_path.parent.iterdir())


def test_native_http_200_failure_rule_unlocks_existing_setup_repair(tmp_path, profile_app):
    from fi.alk.harness.contract import RuntimeConfigFile, RuntimeHttpFailureRule
    from fi.alk.harness.runtime_repair import validate_runtime_plan_patch

    url, state = profile_app
    state.update(profile=True, body={
        "choices": [{"message": {"role": "assistant", "content": "Configuration failed"},
                     "finish_reason": "error"}],
        "error": {"message": "Wrong region secret123"},
    })
    (tmp_path / "app.py").write_text(
        'REGION = "global"\n'
        'FAILED_RESPONSE = {"choices": [{"finish_reason": "error"}], '
        '"error": {"message": "Wrong region"}}\n'
    )
    contract = AgentContract(runtime=Runtime(
        language="python", command=["python", "app.py"],
        config_files={".env": RuntimeConfigFile(values={"REGION": "wrong"})},
        interface=interface(protocol="openai_chat", model="existing-profile"),
    ))
    asyncio.run(probe(tmp_path, url, interface=contract.runtime.interface))
    runtime = contract.runtime.model_copy(deep=True)
    runtime.interface.failure_rule = RuntimeHttpFailureRule(
        status_path="choices.0.finish_reason", failure_values=["error"],
        message_path="error.message",
    )
    patch = RuntimePlanPatch(
        base_contract_hash=_contract_hash(contract), runtime=runtime,
        evidence_paths=("app.py",), diagnostic_codes=("chat_tool_evidence_unverified",),
        summary="Recognize the source-defined failure response before repairing setup.",
    )
    observed = validate_runtime_plan_patch(
        tmp_path, contract, patch, allowed_diagnostic_codes={"chat_tool_evidence_unverified"},
    )
    with pytest.raises(RuntimeValidationError) as error:
        asyncio.run(probe(tmp_path, url, interface=observed.runtime.interface))
    assert error.value.diagnostics[0].code == "chat_ingress_failed"
    assert state["calls"][0][1] == state["calls"][1][1]
    runtime = observed.runtime.model_copy(deep=True)
    runtime.config_files[".env"].values["REGION"] = "global"
    patch = patch.model_copy(update={
        "base_contract_hash": _contract_hash(observed), "runtime": runtime,
        "diagnostic_codes": (error.value.diagnostics[0].code,),
        "summary": "Correct the source-defined region without changing the agent.",
    })
    repaired = validate_runtime_plan_patch(
        tmp_path, observed, patch, allowed_diagnostic_codes={"chat_ingress_failed"},
    )
    assert repaired.runtime.config_files[".env"].values == {"REGION": "global"}
    assert repaired.runtime.interface == observed.runtime.interface
    assert repaired.runtime.command == contract.runtime.command
    assert "secret123" not in json.dumps(receipts(tmp_path))
    assert sorted(item["status"] for item in receipts(tmp_path)) == ["failed", "passed"]
