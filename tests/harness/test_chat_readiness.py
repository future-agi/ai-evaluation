from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
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


def test_healthy_server_requires_profile_then_typed_runtime_patch_passes(
    tmp_path, profile_app
):
    url, state = profile_app
    with urlopen(url + "/health") as response:
        assert response.status == 200
    with pytest.raises(RuntimeValidationError) as error:
        asyncio.run(probe(tmp_path, url))
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
    assert error.value.diagnostics[0].code == "external_service_unavailable"
    assert receipts(tmp_path)[0]["status"] == "failed"


def test_validation_rejects_healthy_unprovisioned_chat_before_invariants(
    tmp_path, monkeypatch, profile_app
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
                SimpleNamespace(endpoints={"target_http": SimpleNamespace(address=url)})
            ]

        async def reset(self, *args, **kwargs):
            calls.append("reset")
            state["profile"] = False

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
        def __init__(self, work):
            pass

        async def create(self, *args, **kwargs):
            return World()

    def bundle(**kwargs):
        target = kwargs["output"]
        scenario = target / "scenarios/one"
        scenario.mkdir(parents=True)
        (scenario / "scenario.json").write_text(json.dumps({"scenario_key": "one"}))
        return SimpleNamespace(
            digest="bundle", provenance=SimpleNamespace(source_digest="source")
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
        lambda *a: [
            SimpleNamespace(
                scenario_key="one",
                scenario_id="scenario-id",
                setup=lambda w: calls.append("setup"),
                ready=lambda w: True,
            )
        ],
    )
    monkeypatch.setattr(source_data_invariants, "author_invariants", invariants)
    monkeypatch.setattr(source_data_invariants, "check_invariants", check)
    monkeypatch.setattr(validation, "_freeze_certified_bundle", freeze)
    job = SimpleNamespace(
        source=SimpleNamespace(kind=None),
        agent=SimpleNamespace(secret_refs={}),
        metadata={"generic_harness_v1": True},
        scenario_count=1,
        seed=0,
    )

    async def validate(*args):
        return await validation.validate_once(
            job, source, authoring, secrets_path=secrets, local_runtime=True
        )

    with pytest.raises(RuntimeValidationError) as error:
        asyncio.run(validate())
    assert error.value.diagnostics[0].code == "chat_ingress_configuration_invalid"
    assert calls == ["provision", "reset", "setup", "close"]
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
        validation.validate_and_repair(job, source, authoring, validate=validate)
    )
    assert repairs == [("chat_ingress_configuration_invalid",)]
    assert calls[-2:] == ["freeze", "close"]
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
def test_dependency_patch_repeats_identical_http_request_once(tmp_path, profile_app, recovers):
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
    assert len(repairs) == 1
    assert builds == [[], ["renamed_transport"]]
    assert len(state["calls"]) == 2
    assert state["calls"][0] == state["calls"][1]
    assert sorted(item["status"] for item in receipts(authoring)) == (["failed", "passed"] if recovers else ["failed", "failed"])
    assert (authoring / "generic-harness/runtime-plan-repair-1.json").is_file()
    assert all(path.read_bytes() == body for path, body in snapshots.items())
    assert (authoring / "scenarios.json").read_text() == '{"original": true}'
