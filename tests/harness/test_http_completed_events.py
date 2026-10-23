import asyncio
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from fi.alk.harness.chat_call_runner import _HostedChatTarget
from fi.alk.harness.contract import RuntimeInterface
from fi.simulate.agent.wrapper import AgentInput
from fi.simulate.agent.wrappers.http import HTTPAgentWrapper

SPEC = dict(
    correlation_header="X-Conversation-ID",
    path="/events/{{correlation_id}}",
    items_path="items",
    next_page_path="next",
    page_parameter="page",
    kind_path="kind",
    action_kind="action",
    observation_kind="completed",
    action_id_path="id",
    observation_action_id_path="action_id",
    name_path="name",
    arguments_path="arguments",
    result_path="result",
    error_path="failed",
)
ACTION = dict(kind="action", id="a", name="edit", arguments='{"path":"a.py"}')
DONE = dict(kind="completed", action_id="a", result={"bytes": 42}, failed=False)


@contextmanager
def server(*, pages=None, correlation="conversation-1", redirect=False, event_status=200, malformed=False):
    requests = []
    pages = (
        pages
        if pages is not None
        else [{"items": [ACTION], "next": "second"}, {"items": [DONE], "next": None}]
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            requests.append(("POST", self.path))
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            if correlation is not None:
                self.send_header("X-Conversation-ID", correlation)
            self.end_headers()
            self.wfile.write(
                json.dumps({"choices": [{"message": {"content": "Edited."}}]}).encode()
            )

        def do_GET(self):
            requests.append(("GET", self.path))
            if redirect:
                self.send_response(302)
                self.send_header("Location", "/leaked")
                self.end_headers()
                return
            index = 1 if "?page=" in self.path else 0
            self.send_response(event_status if self.path.startswith("/events/") else 404)
            self.end_headers()
            self.wfile.write(b"not json" if malformed else json.dumps(pages[min(index, len(pages) - 1)]).encode())

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}", requests
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def wrapper(base, spec=None):
    return HTTPAgentWrapper(
        endpoint=base + "/chat",
        protocol="openai_chat",
        completed_tool_events=spec or SPEC,
    )


def invoke(target):
    return asyncio.run(target.call(AgentInput(thread_id="one", messages=[])))


def test_missing_event_route_is_repairable_after_successful_chat(tmp_path):
    from fi.alk.harness.authoring_runtime_validation import RuntimeValidationError
    from fi.alk.harness.chat_readiness import probe_chat_ingress
    from fi.alk.harness.contract import AgentContract, Runtime
    from fi.alk.harness.runtime_repair import RuntimePlanPatch, _contract_hash, validate_runtime_plan_patch

    (tmp_path / "app.py").write_text('EVENT_ROUTE = "/events/{conversation_id}"\n')
    contract = AgentContract(runtime=Runtime(
        language="python", command=["python", "app.py"],
        interface=RuntimeInterface(
            kind="http", port=8080, path="/chat", protocol="openai_chat",
            completed_tool_events={**SPEC, "path": "/missing/{{correlation_id}}"},
        ),
    ))
    with server() as (base, requests):
        with pytest.raises(RuntimeValidationError) as error:
            asyncio.run(probe_chat_ingress(
                interface=contract.runtime.interface, endpoint=base, authoring=tmp_path,
                candidate_hash="candidate", bundle_digest="bundle", secret_values={}, redaction_values={},
            ))
        diagnostic = error.value.diagnostics[0]
        assert diagnostic.code == "chat_tool_evidence_unverified"
        receipt = json.loads(next((tmp_path / "generic-harness/chat-readiness").glob("*.json")).read_text())
        assert receipt["trace"]["status_code"] == 404
        assert receipt["trace"]["failure_stage"] == "tool_events"
        assert requests == [("POST", "/chat"), ("GET", "/missing/conversation-1")]
        runtime = contract.runtime.model_copy(deep=True)
        runtime.interface.completed_tool_events.path = SPEC["path"]
        patch = RuntimePlanPatch(
            base_contract_hash=_contract_hash(contract), runtime=runtime,
            evidence_paths=("app.py",), diagnostic_codes=(diagnostic.code,),
            summary="Correct the source-owned event route.",
        )
        repaired = validate_runtime_plan_patch(
            tmp_path, contract, patch, allowed_diagnostic_codes={diagnostic.code},
        )
        response = invoke(wrapper(base, repaired.runtime.interface.completed_tool_events.model_dump()))
        assert response.metadata["external_agent"]["success"] is True
        assert len(response.metadata["native_completed_tools"]) == 1
        assert response.metadata["native_completed_tools"][0]["result"] == {"bytes": 42}


@pytest.mark.parametrize("status,code", [
    (401, "credential_missing"), (403, "credential_missing"),
    (429, "external_service_unavailable"), (503, "external_service_unavailable"),
])
def test_event_http_failure_retains_status_and_auth_or_transient_classification(status, code):
    from fi.alk.harness.chat_readiness import _failure_code

    with server(event_status=status) as (base, _):
        trace = invoke(wrapper(base)).metadata["external_agent"]
    assert trace["status_code"] == status
    assert trace["failure_stage"] == "tool_events"
    assert _failure_code(trace) == code
    assert trace["response_excerpt"] == ""


def test_malformed_event_payload_is_an_observation_failure():
    from fi.alk.harness.chat_readiness import _failure_code

    with server(malformed=True) as (base, _):
        trace = invoke(wrapper(base)).metadata["external_agent"]
    assert trace["status_code"] == 200
    assert trace["failure_stage"] == "tool_events"
    assert _failure_code(trace) == "chat_tool_evidence_unverified"


@pytest.mark.parametrize("timeout", [False, True])
def test_event_network_failure_does_not_report_prior_chat_status(monkeypatch, timeout):
    import urllib.error
    from fi.alk.harness.chat_readiness import _failure_code

    with server() as (base, requests):
        target = wrapper(base)
        open_request = target._opener.open

        def fail_event(request, **kwargs):
            if request.get_method() == "GET":
                if timeout:
                    raise TimeoutError("event deadline")
                raise urllib.error.URLError("event connection failed")
            return open_request(request, **kwargs)

        monkeypatch.setattr(target._opener, "open", fail_event)
        trace = invoke(target).metadata["external_agent"]
        assert requests == [("POST", "/chat")]
    assert trace["status_code"] == 0
    assert trace["failure_stage"] == "tool_events"
    assert _failure_code(trace) == ("chat_ingress_timeout" if timeout else "external_service_unavailable")


def test_native_paginated_tools_recorded_once_without_execution_or_continuation():
    with server() as (base, requests):
        world = SimpleNamespace(calls=[])
        contract = SimpleNamespace(
            runtime=SimpleNamespace(interface=SimpleNamespace(include_tools=False))
        )
        target = _HostedChatTarget(
            wrapper=wrapper(base),
            contract=contract,
            world=world,
            scenario_key="case",
            scenario_id="one",
        )
        assert asyncio.run(target.say("edit")) == "Edited."
        assert len(world.calls) == 1
        call = world.calls[0]
        assert (call.name, call.arguments, call.result, call.ok) == (
            "edit",
            {"path": "a.py"},
            {"bytes": 42},
            True,
        )
        assert requests == [
            ("POST", "/chat"),
            ("GET", "/events/conversation-1"),
            ("GET", "/events/conversation-1?page=second"),
        ]
        assert asyncio.run(target.say("thanks")) == "Edited."
        assert len(world.calls) == 1
        assert [method for method, _ in requests].count("POST") == 2


@pytest.mark.parametrize(
    "pages",
    [
        [{"items": [ACTION], "next": None}],
        [{"items": [DONE], "next": None}],
        [{"items": [ACTION, ACTION, DONE], "next": None}],
        [{"items": [ACTION, DONE, DONE], "next": None}],
        [{"items": [ACTION, dict(DONE, failed="false")], "next": None}],
        [{"items": [], "next": "again"}],
    ],
)
def test_invalid_event_evidence_fails_closed(pages):
    with server(pages=pages) as (base, _):
        response = invoke(wrapper(base))
        assert response.metadata["external_agent"]["success"] is False
        assert response.metadata["external_agent"]["failure_stage"] == "tool_events"
        assert not response.metadata.get("native_completed_tools")
        assert response.metadata["external_agent"]["response_excerpt"] == ""


def test_missing_header_and_redirect_do_not_follow_or_leak():
    for kwargs in ({"correlation": None}, {"redirect": True}):
        with server(**kwargs) as (base, requests):
            response = invoke(wrapper(base))
            assert response.metadata["external_agent"]["success"] is False
            assert all(path != "/leaked" for _, path in requests)
            assert "conversation-1" not in json.dumps(response.metadata)


def test_declared_failed_native_tool_is_not_success():
    with server(pages=[{"items": [ACTION, dict(DONE, failed=True)], "next": None}]) as (
        base,
        _,
    ):
        response = invoke(wrapper(base))
        assert response.metadata["external_agent"]["success"] is True
        assert response.metadata["native_completed_tools"][0]["ok"] is False


def test_page_budget_fails_instead_of_partial_evidence():
    with server() as (base, requests):
        response = invoke(wrapper(base, dict(SPEC, max_pages=1)))
        assert response.metadata["external_agent"]["success"] is False
        assert len(requests) == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"path": "https://outside/{{correlation_id}}"},
        {"path": "//outside/{{correlation_id}}"},
        {"path": "/events/{{unknown}}"},
        {"path": "/events/{{correlation_id}}#ignored"},
        {"correlation_header": "Set-Cookie"},
        {"correlation_header": "Authorization"},
        {"correlation_header": "X-API-Key"},
        {"correlation_header": "X-Auth\r\nInjected"},
        {"max_pages": 21},
        {"observation_kind": "action"},
        {"error_path": "", "result_error_path": "error"},
    ],
)
def test_contract_rejects_invalid_observation_config(changes):
    with pytest.raises(ValidationError):
        RuntimeInterface(
            kind="http",
            port=8000,
            path="/chat",
            completed_tool_events={**SPEC, **changes},
        )


def test_endpoint_rejection_explains_required_placeholder():
    with pytest.raises(ValidationError, match=r"\{\{correlation_id\}\}"):
        RuntimeInterface(
            kind="http", port=8000, path="/chat",
            completed_tool_events={**SPEC, "path": "/events/{conversation_id}"},
        )


def test_contract_preserves_declared_observation_config():
    interface = RuntimeInterface(
        kind="http", port=8000, path="/chat", completed_tool_events=SPEC
    )
    assert interface.completed_tool_events.correlation_header == "X-Conversation-ID"


def test_correlation_is_encoded_and_isolated_across_conversations():
    with server(correlation="one/two?private=1") as (base, requests):
        first = invoke(wrapper(base))
        assert first.metadata["native_completed_tools"][0]["name"] == "edit"
        assert requests[1] == ("GET", "/events/one%2Ftwo%3Fprivate%3D1")
        assert "one/two" not in json.dumps(first.metadata)
    with server(correlation="another") as (base, _):
        second = invoke(wrapper(base))
        assert len(second.metadata["native_completed_tools"]) == 1


def test_non_http_contract_rejects_event_collection():
    with pytest.raises(ValidationError):
        RuntimeInterface(kind="callable", completed_tool_events=SPEC)


def test_native_serialized_tool_call_uses_flat_arguments():
    spec = {**SPEC, 'action_kind': 'ActionEvent', 'observation_kind': 'ObservationEvent',
            'name_path': 'tool_name', 'arguments_path': 'tool_call.arguments',
            'result_path': 'observation', 'error_path': 'observation.is_error'}
    pages = [{'items': [
        {'kind': 'ActionEvent', 'id': 'action-1', 'tool_name': 'file_editor',
         'tool_call': {'id': 'call-1', 'name': 'file_editor',
                       'arguments': '{"command":"view","path":"a.py"}'}},
        {'kind': 'ObservationEvent', 'action_id': 'action-1', 'tool_call_id': 'call-1',
         'observation': {'is_error': False, 'content': 'original bytes'}}], 'next': None}]
    with server(pages=pages) as (base, requests):
        world = SimpleNamespace(calls=[])
        target = _HostedChatTarget(wrapper=wrapper(base, spec),
            contract=SimpleNamespace(runtime=SimpleNamespace(interface=SimpleNamespace(include_tools=False))),
            world=world, scenario_key='case', scenario_id='one')
        assert asyncio.run(target.say('view')) == 'Edited.'
        assert len(world.calls) == 1
        assert world.calls[0].arguments == {'command': 'view', 'path': 'a.py'}
        assert world.calls[0].result == {'is_error': False, 'content': 'original bytes'}
        assert world.calls[0].ok is True
        assert requests == [('POST', '/chat'), ('GET', '/events/conversation-1')]


def test_correlation_dot_segment_cannot_change_declared_path():
    with server(correlation='..') as (base, requests):
        result = invoke(wrapper(base))
        assert result.metadata['external_agent']['success'] is True
        assert requests[1] == ('GET', '/events/%2E%2E')


def test_native_validation_failure_and_retry_are_recorded_once():
    spec = {
        **SPEC,
        "action_id_path": "tool_call_id",
        "observation_action_id_path": "tool_call_id",
        "arguments_path": "tool_call.arguments",
        "failure_events": [{"kind": "rejected", "action_id_path": "tool_call_id",
                            "result_path": "message"}],
    }
    events = [
        {"kind": "action", "tool_call_id": "first", "name": "edit",
         "tool_call": {"arguments": '{"path":"a.py"}'}},
        {"kind": "completed", "tool_call_id": "first", "result": "read", "failed": False},
        {"kind": "action", "tool_call_id": "invalid", "name": "edit",
         "tool_call": {"arguments": '{"path":7}'}},
        {"kind": "rejected", "tool_call_id": "invalid", "message": "path must be a string"},
        {"kind": "action", "tool_call_id": "retry", "name": "edit",
         "tool_call": {"arguments": '{"path":"a.py"}'}},
        {"kind": "completed", "tool_call_id": "retry", "result": "written", "failed": False},
    ]
    with server(pages=[{"items": events, "next": None}]) as (base, requests):
        world = SimpleNamespace(calls=[])
        target = _HostedChatTarget(
            wrapper=wrapper(base, spec),
            contract=SimpleNamespace(runtime=SimpleNamespace(
                interface=SimpleNamespace(include_tools=False))),
            world=world, scenario_key="case", scenario_id="one",
        )
        assert asyncio.run(target.say("edit")) == "Edited."
        assert [(call.arguments, call.result, call.ok) for call in world.calls] == [
            ({"path": "a.py"}, "read", True),
            ({"path": 7}, "path must be a string", False),
            ({"path": "a.py"}, "written", True),
        ]
        assert asyncio.run(target.say("thanks")) == "Edited."
        assert len(world.calls) == 3
        assert requests == [("POST", "/chat"), ("GET", "/events/conversation-1")] * 2


FAILURE = {"kind": "rejected", "action_id_path": "action_id", "result_path": "message"}
REJECTED = {"kind": "rejected", "action_id": "a", "message": "invalid arguments"}


@pytest.mark.parametrize("events", [
    [REJECTED],
    [ACTION],
    [ACTION, REJECTED, REJECTED],
    [ACTION, DONE, REJECTED],
    [ACTION, REJECTED, DONE],
    [ACTION, {"kind": "rejected", "action_id": "a"}],
    [ACTION, dict(REJECTED, message=None)],
    [ACTION, dict(REJECTED, message=float("nan"))],
    [dict(ACTION, arguments="bad json"), REJECTED],
    [dict(ACTION, arguments="[]"), REJECTED],
    [dict(ACTION, id="a" * 8193), dict(REJECTED, action_id="a" * 8193)],
    [ACTION, dict(REJECTED, action_id="")],
    [ACTION, dict(REJECTED, action_id=7)],
])
def test_failure_event_evidence_fails_closed(events):
    with server(pages=[{"items": events, "next": None}]) as (base, _):
        response = invoke(wrapper(base, {**SPEC, "failure_events": [FAILURE]}))
    assert response.metadata["external_agent"]["success"] is False
    assert response.metadata["external_agent"]["failure_stage"] == "tool_events"
    assert not response.metadata.get("native_completed_tools")
    assert response.metadata["external_agent"]["response_excerpt"] == ""


@pytest.mark.parametrize("length", [5887, 8192])
@pytest.mark.parametrize("failed", [False, True])
def test_opaque_event_join_keys_accept_bounded_long_values(length, failed):
    key = "x" * length
    outcome = REJECTED if failed else DONE
    events = [dict(ACTION, id=key), dict(outcome, action_id=key)]
    with server(pages=[{"items": events, "next": None}]) as (base, _):
        target = wrapper(base, {**SPEC, "failure_events": [FAILURE]})
        response = invoke(target)
        assert response.metadata["external_agent"]["success"] is True
        assert response.metadata["native_completed_tools"][0]["ok"] is not failed
        assert invoke(target).metadata["native_completed_tools"] == []


def test_undeclared_failure_kind_does_not_infer_completion():
    with server(pages=[{"items": [ACTION, REJECTED], "next": None}]) as (base, _):
        response = invoke(wrapper(base))
    assert response.metadata["external_agent"]["success"] is False
    assert not response.metadata.get("native_completed_tools")


@pytest.mark.parametrize("failures", [
    [FAILURE, FAILURE],
    [dict(FAILURE, kind="action")],
    [dict(FAILURE, kind="completed")],
    [dict(FAILURE, kind="")],
    [dict(FAILURE, action_id_path="")],
    [dict(FAILURE, result_path="")],
    [dict(FAILURE, ok=True)],
    [dict(FAILURE, kind=str(index)) for index in range(9)],
])
def test_contract_rejects_invalid_failure_event_mapping(failures):
    with pytest.raises(ValidationError):
        RuntimeInterface(
            kind="http", port=8000, path="/chat",
            completed_tool_events={**SPEC, "failure_events": failures},
        )


def test_distinct_failure_kinds_share_completion_identity_space():
    events = [ACTION, REJECTED, dict(REJECTED, kind="execution_error")]
    failures = [FAILURE, dict(FAILURE, kind="execution_error")]
    with server(pages=[{"items": events, "next": None}]) as (base, _):
        response = invoke(wrapper(base, {**SPEC, "failure_events": failures}))
    assert response.metadata["external_agent"]["success"] is False
    assert not response.metadata.get("native_completed_tools")


def test_long_event_keys_do_not_expand_correlation_or_cursor_limits():
    for kwargs in (
        {"correlation": "x" * 1025},
        {"pages": [{"items": [ACTION, DONE], "next": "x" * 1025}]},
    ):
        with server(**kwargs) as (base, _):
            response = invoke(wrapper(base))
        assert response.metadata["external_agent"]["success"] is False
        assert not response.metadata.get("native_completed_tools")


def test_multiple_failure_kinds_use_declared_payloads_and_always_fail():
    failures = [FAILURE, {"kind": "execution_error", "action_id_path": "call", "result_path": "error"}]
    events = [
        ACTION, dict(REJECTED, failed=False), dict(ACTION, id="b"),
        {"kind": "execution_error", "call": "b", "error": {"reason": "tool unavailable"}},
    ]
    with server(pages=[{"items": events, "next": None}]) as (base, _):
        response = invoke(wrapper(base, {**SPEC, "failure_events": failures}))
    assert response.metadata["external_agent"]["success"] is True
    assert [(call["result"], call["ok"]) for call in response.metadata["native_completed_tools"]] == [
        ("invalid arguments", False), ({"reason": "tool unavailable"}, False),
    ]
