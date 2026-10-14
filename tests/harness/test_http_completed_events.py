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
def server(*, pages=None, correlation="conversation-1", redirect=False):
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
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(pages[min(index, len(pages) - 1)]).encode())

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
