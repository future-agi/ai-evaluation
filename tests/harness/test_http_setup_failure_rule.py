import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from fi.alk.harness.chat_transport import create_chat_wrapper
from fi.alk.harness.contract import RuntimeHttpSetupRequest, RuntimeInterface
from fi.simulate.agent.wrapper import AgentInput


@pytest.fixture
def setup_app():
    state = {"response": {"code": 0}, "status": 200, "calls": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            state["calls"].append(self.path)
            status, body = (
                (state["status"], state["response"])
                if self.path == "/configure"
                else (200, {"content": "ready"})
            )
            if self.path == "/login":
                body = {"token": state["login_token"]}
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", state
    server.shutdown()
    thread.join()
    server.server_close()


def invoke(base, rule, *, first_changes=None, secrets=None):
    first = {"path": "/configure", **(first_changes or {})}
    if rule is not None:
        first["failure_rule"] = rule
    config = RuntimeInterface(
        kind="http",
        port=8000,
        path="/chat",
        protocol="json_template",
        request_template={},
        response_path="content",
        setup_requests=[first, {"path": "/next"}],
    )
    wrapper = create_chat_wrapper(config, base, secret_values=secrets or {})
    return asyncio.run(
        wrapper.call(
            AgentInput(thread_id="trial", messages=[], new_message={"content": "hello"})
        )
    )


def test_http200_application_error_stops_before_capture_and_next_call(setup_app):
    base, state = setup_app
    state["response"] = {"code": 102}
    response = invoke(
        base,
        {"status_path": "code", "success_values": [0]},
        first_changes={
            "capture": {"new_id": "missing_id"},
            "capture_headers": {"session": "X-Missing"},
        },
    )
    trace = response.metadata["external_agent"]
    assert trace["success"] is False
    assert trace["status_code"] == 200
    assert trace["failure_stage"] == "setup"
    assert trace["error"] == "Declared application failure"
    assert state["calls"] == ["/configure"]


@pytest.mark.parametrize("value", [102, False, "0", 0.0, None])
def test_setup_success_predicate_preserves_scalar_types(setup_app, value):
    base, state = setup_app
    state["response"] = {"code": value}
    response = invoke(base, {"status_path": "code", "success_values": [0]})
    assert response.metadata["external_agent"]["success"] is False
    assert state["calls"] == ["/configure"]


def test_missing_setup_status_fails_before_later_requests(setup_app):
    base, state = setup_app
    state["response"] = {}
    response = invoke(base, {"status_path": "code", "success_values": [0]})
    assert response.metadata["external_agent"]["failure_stage"] == "setup"
    assert response.metadata["external_agent"]["success"] is False
    assert state["calls"] == ["/configure"]


@pytest.mark.parametrize(
    "rule",
    [
        {"status_path": "code", "success_values": [0]},
        {"status_path": "code", "failure_values": [102]},
        None,
    ],
)
def test_success_or_undeclared_rule_preserves_existing_setup_flow(setup_app, rule):
    base, state = setup_app
    response = invoke(base, rule)
    assert response.metadata["external_agent"]["success"] is True
    assert response.content == "ready"
    assert state["calls"] == ["/configure", "/next", "/chat"]


def test_failure_predicate_is_applied_to_setup(setup_app):
    base, state = setup_app
    state["response"] = {"code": 102}
    response = invoke(base, {"status_path": "code", "failure_values": [102]})
    assert response.metadata["external_agent"]["success"] is False
    assert state["calls"] == ["/configure"]


def test_http_failure_precedes_application_rule(setup_app):
    base, state = setup_app
    state.update(status=503, response={"code": 0})
    response = invoke(base, {"status_path": "code", "success_values": [0]})
    trace = response.metadata["external_agent"]
    assert trace["status_code"] == 503
    assert trace["failure_stage"] == "setup"
    assert state["calls"] == ["/configure"]


def test_setup_application_error_redacts_private_values(setup_app):
    base, state = setup_app
    secret = "fixture-private-credential-do-not-log"
    state["response"] = {"code": 102, "detail": "configuration rejected " + secret}
    response = invoke(
        base,
        {"status_path": "code", "success_values": [0], "message_path": "detail"},
        secrets={"PRIVATE_KEY_ALIAS": secret},
    )
    trace = response.metadata["external_agent"]
    assert trace["failure_stage"] == "setup"
    assert "configuration rejected" in trace["error"]
    assert secret not in json.dumps(response.metadata)
    assert state["calls"] == ["/configure"]


@pytest.mark.parametrize(
    "rule",
    [
        {},
        {"status_path": "code"},
        {"status_path": "code", "success_values": []},
        {"status_path": "code", "success_values": [0], "failure_values": [102]},
        {"status_path": "code", "success_values": [{}]},
    ],
)
def test_setup_rule_uses_shared_rule_validation(rule):
    with pytest.raises(ValueError):
        RuntimeHttpSetupRequest(path="/configure", failure_rule=rule)


def test_setup_rule_survives_contract_serialization():
    request = RuntimeHttpSetupRequest(
        path="/configure", failure_rule={"status_path": "code", "success_values": [0]}
    )
    assert request.model_dump()["failure_rule"]["success_values"] == [0]
    assert RuntimeHttpSetupRequest.model_json_schema()["properties"]["failure_rule"]


def test_authoring_setup_rule_survives_vertex_schema_conversion(tmp_path):
    from google.genai import types
    from fi.alk.harness.backends.vertex_gemini import _json_schema
    from fi.alk.harness.tools import contract_tools

    submit = next(
        tool
        for tool in contract_tools(tmp_path).tools
        if tool.name == "submit_contract"
    )
    declaration = types.FunctionDeclaration(
        name=submit.name, parameters=_json_schema(submit.input_schema)
    )
    interface = declaration.parameters.properties["runtime"].properties["interface"]
    rule = interface.properties["setup_requests"].items.properties["failure_rule"]
    assert rule.type == types.Type.OBJECT
    assert "status_path" in rule.required
    assert rule.properties["status_path"].type == types.Type.STRING
    for name in ["failure_values", "success_values"]:
        choices = rule.properties[name].any_of
        assert {choice.type for choice in choices} == {
            types.Type.ARRAY,
            types.Type.NULL,
        }
        values = next(choice for choice in choices if choice.type == types.Type.ARRAY)
        assert values.min_items == 1
        assert values.max_items == 32
        assert {choice.type for choice in values.items.any_of} == {
            types.Type.STRING,
            types.Type.INTEGER,
            types.Type.NUMBER,
            types.Type.BOOLEAN,
            types.Type.NULL,
        }
    assert rule.properties["message_path"].type == types.Type.STRING


def test_setup_error_redacts_a_previously_captured_token(setup_app):
    base, state = setup_app
    token = "fixture-captured-private+/token"
    state.update(
        login_token=token, response={"code": 102, "detail": "rejected " + token}
    )
    config = RuntimeInterface(
        kind="http",
        port=8000,
        path="/chat",
        protocol="json_template",
        request_template={},
        response_path="content",
        setup_requests=[
            {"path": "/login", "capture": {"access_token": "token"}},
            {
                "path": "/configure",
                "headers_template": {"Authorization": "Bearer {{access_token}}"},
                "failure_rule": {
                    "status_path": "code",
                    "success_values": [0],
                    "message_path": "detail",
                },
            },
            {"path": "/next"},
        ],
    )
    wrapper = create_chat_wrapper(config, base, secret_values={})
    response = asyncio.run(
        wrapper.call(
            AgentInput(thread_id="trial", messages=[], new_message={"content": "hello"})
        )
    )
    assert response.metadata["external_agent"]["failure_stage"] == "setup"
    assert token not in json.dumps(response.metadata)
    assert state["calls"] == ["/login", "/configure"]
