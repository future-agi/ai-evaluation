import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

import pytest

from fi.alk.harness.chat_readiness import probe_chat_ingress
from fi.alk.harness.chat_transport import create_chat_wrapper
from fi.alk.harness.contract import AgentContract, Runtime, RuntimeInterface
from fi.alk.harness.runtime_repair import (
    RuntimePlanPatch, _contract_hash, validate_runtime_plan_patch,
)
from fi.simulate.agent.wrapper import AgentInput


@pytest.fixture
def auth_app():
    state = {"calls": [], "cookies": {}, "fail": False, "token": None, "unknown": ""}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            content_type = self.headers.get("Content-Type")
            body = (
                parse_qs(raw.decode())
                if content_type == "application/x-www-form-urlencoded"
                else json.loads(raw)
            )
            state["calls"].append((self.path, body, dict(self.headers)))
            status, response, headers = 200, {}, {}
            if self.path == "/register":
                response = {"id": body["username"]}
            elif self.path == "/login-form":
                if content_type != "application/x-www-form-urlencoded" or body[
                    "password"
                ] != ["fixture password"]:
                    status = 400
                else:
                    user = body["username"][0]
                    cookie = "session-" + user
                    state["cookies"][cookie] = user
                    headers["Set-Cookie"] = f"session={cookie}; Path=/; HttpOnly"
            elif self.path in {"/login-json", "/login-header"}:
                token = state["token"] or "token-" + body["username"]
                if self.path == "/login-json":
                    response = {"token": token}
                else:
                    headers["Authorization"] = token
            elif self.path == "/redirect":
                status, headers = 302, {"Location": "/steal"}
            elif self.path in {"/configure", "/chat"}:
                user = body["user"]
                cookie = self.headers.get("Cookie", "")
                authorization = self.headers.get("Authorization", "")
                if cookie != "session=session-" + user and authorization not in {
                    "token-" + user,
                    "Bearer token-" + user,
                }:
                    status, response = 401, {"error": "authentication missing"}
                elif state["fail"]:
                    status, response = (
                        500,
                        {
                            "error": "echo "
                            + cookie
                            + " "
                            + authorization
                            + " "
                            + state["unknown"]
                        },
                    )
                else:
                    response = {"content": "hello " + user}
                    headers["X-Conversation-ID"] = user
            encoded = json.dumps(response).encode()
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            state["calls"].append((self.path, {}, dict(self.headers)))
            user = self.path.rsplit("/", 1)[-1]
            authenticated = self.headers.get(
                "Cookie"
            ) == "session=session-" + user or self.headers.get("Authorization") in {
                "Bearer token-" + user,
                "token-" + user,
            }
            body = (
                {
                    "items": [
                        {
                            "kind": "action",
                            "id": "read-1",
                            "name": "read_file",
                            "arguments": {"path": "fixture.txt"},
                        },
                        {
                            "kind": "completed",
                            "action_id": "read-1",
                            "result": {"content": "fixture 84"},
                            "failed": False,
                        },
                    ],
                    "next": None,
                }
                if authenticated
                else {"error": "event authentication missing"}
            )
            encoded = json.dumps(body).encode()
            self.send_response(200 if authenticated else 401)
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


@pytest.fixture
def local_setup_auth_patch(tmp_path):
    (tmp_path / "app.py").write_text(
        'import os\nsecret = os.environ["LOCAL_SERVER_KEY"]\n'
        'def authenticated(headers):\n    return headers.get("Authorization") == "Bearer " + secret\n'
    )
    contract = AgentContract(runtime=Runtime(
        language="python", execution="native", command=["python", "app.py"],
        interface=RuntimeInterface(
            kind="http", protocol="json_template", port=8080, path="/chat",
            local_auth={"server_environment": "LOCAL_SERVER_KEY", "template_variable": "local_auth"},
            headers_template={
                "Authorization": "Bearer {{local_auth}}", "X-Setup-Result": "{{configured}}",
            },
            setup_requests=[
                {"path": "/configure", "body_template": {"user": "user"},
                 "headers_template": {"X-Preserve": "existing"}, "capture": {"configured": "content"}},
                {"path": "/configure", "body_template": {"user": "user"}},
            ],
            request_template={"user": "user", "text": "{{new_message_content}}"},
            response_path="content",
        ),
    ))
    runtime = contract.runtime.model_copy(deep=True)
    for step in runtime.interface.setup_requests:
        step.headers_template["Authorization"] = "Bearer {{local_auth}}"
    patch = RuntimePlanPatch(
        base_contract_hash=_contract_hash(contract), runtime=runtime,
        evidence_paths=("app.py",), diagnostic_codes=("credential_missing",),
        summary="Bind existing local authentication to the source-required setup headers.",
    )
    return contract, patch


def test_existing_local_auth_repair_authenticates_setup_before_captured_chat_headers(
    tmp_path, auth_app, local_setup_auth_patch,
):
    url, state = auth_app
    contract, patch = local_setup_auth_patch
    auth_values = {"LOCAL_SERVER_KEY": "token-user"}
    original = create_chat_wrapper(contract.runtime.interface, url, local_auth_values=auth_values)
    failed = asyncio.run(original.call(AgentInput(thread_id="user", messages=[])))
    assert failed.metadata["external_agent"]["success"] is False
    assert [path for path, _, _ in state["calls"]] == ["/configure"]
    assert "Authorization" not in state["calls"][0][2]
    repaired = validate_runtime_plan_patch(
        tmp_path, contract, patch, allowed_diagnostic_codes={"credential_missing"},
    )
    wrapper = create_chat_wrapper(repaired.runtime.interface, url, local_auth_values=auth_values)
    response = asyncio.run(wrapper.call(AgentInput(thread_id="user", messages=[])))
    assert response.content == "hello user"
    assert [path for path, _, _ in state["calls"]] == ["/configure", "/configure", "/configure", "/chat"]
    for _, _, headers in state["calls"][1:]:
        assert headers["Authorization"] == "Bearer token-user"
    assert all("X-Setup-Result" not in headers for _, _, headers in state["calls"][:-1])
    assert state["calls"][-1][2]["X-Setup-Result"] == "hello user"


def test_generated_local_credential_can_login_by_form_and_use_cookie(auth_app):
    url, state = auth_app
    config = RuntimeInterface(
        kind="http", port=8080, path="/chat", protocol="json_template",
        local_auth={"server_environment": "LOCAL_PASSWORD", "template_variable": "password"},
        setup_requests=[{
            "path": "/login-form", "encoding": "form",
            "body_template": {"username": "fixture", "password": "{{password}}"},
        }],
        request_template={"user": "fixture"}, response_path="content",
    )
    wrapper = create_chat_wrapper(
        config, url, local_auth_values={"LOCAL_PASSWORD": "fixture password"},
    )
    response = asyncio.run(wrapper.call(AgentInput(thread_id="one", messages=[])))
    assert response.content == "hello fixture"
    assert state["calls"][0][1]["password"] == ["fixture password"]
    assert state["calls"][-1][2]["Cookie"] == "session=session-fixture"


@pytest.mark.parametrize("change", [
    "body", "path", "remove_header", "replace_header", "unbound_token",
    "auth_identity", "global_headers", "config", "no_evidence",
])
def test_existing_local_auth_setup_repair_rejects_other_changes(
    tmp_path, local_setup_auth_patch, change,
):
    contract, patch = local_setup_auth_patch
    step = patch.runtime.interface.setup_requests[0]
    if change == "body":
        step.body_template = {"user": "different"}
    elif change == "path":
        step.path = "/different"
    elif change == "remove_header":
        del step.headers_template["X-Preserve"]
    elif change == "replace_header":
        step.headers_template["X-Preserve"] = "{{local_auth}}"
    elif change == "unbound_token":
        step.headers_template["Authorization"] = "literal-token"
    elif change == "auth_identity":
        patch.runtime.interface.local_auth.server_environment = "OTHER_KEY"
    elif change == "global_headers":
        patch.runtime.interface.headers_template["X-New"] = "{{local_auth}}"
    elif change == "config":
        patch.runtime.command = ["python", "other.py"]
    else:
        patch = patch.model_copy(update={"evidence_paths": ()})
    with pytest.raises(ValueError, match="runtime_repair_"):
        validate_runtime_plan_patch(
            tmp_path, contract, patch, allowed_diagnostic_codes={"credential_missing"},
        )


def interface(mode):
    setup = [
        {
            "path": "/register",
            "body_template": {"username": "{{thread_id}}"},
            "capture": {"user": "id"},
        }
    ]
    login = {
        "path": "/login-" + mode,
        "body_template": {"username": "{{user}}", "password": "fixture password"},
    }
    headers = {}
    if mode == "form":
        login["encoding"] = "form"
    elif mode == "json":
        login["capture"] = {"session_token": "token"}
        headers = {"Authorization": "Bearer {{session_token}}"}
    else:
        login["capture_headers"] = {"session_token": "Authorization"}
        headers = {"Authorization": "{{session_token}}"}
    setup.extend(
        [
            login,
            {
                "path": "/configure",
                "body_template": {"user": "{{user}}"},
                "headers_template": headers,
            },
        ]
    )
    return RuntimeInterface(
        kind="http",
        port=8080,
        path="/chat",
        protocol="json_template",
        request_template={"user": "{{user}}"},
        response_path="content",
        headers_template=headers,
        setup_requests=setup,
    )


@pytest.mark.parametrize("mode", ["form", "json", "header"])
def test_real_auth_setup_then_chat_and_wrapper_isolation(auth_app, monkeypatch, mode):
    url, state = auth_app
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("no_proxy", "")
    first, second = [create_chat_wrapper(interface(mode), url) for _ in range(2)]
    for wrapper, user in [(first, "first"), (second, "second"), (first, "first")]:
        response = asyncio.run(wrapper.call(AgentInput(thread_id=user, messages=[])))
        assert response.content == "hello " + user
        assert response.metadata["external_agent"]["success"] is True
    assert [path for path, _, _ in state["calls"]].count("/register") == 2
    assert [path for path, _, _ in state["calls"]].count("/configure") == 2


@pytest.mark.parametrize("mode", ["form", "json", "header"])
@pytest.mark.parametrize("secret_values", [None, {}])
def test_auth_failure_diagnostics_redact_dynamic_values_without_secret_context(
    auth_app, mode, secret_values
):
    url, state = auth_app
    wrapper = create_chat_wrapper(interface(mode), url, secret_values=secret_values)
    user = "unique-private-user-3849"
    assert asyncio.run(wrapper.call(AgentInput(thread_id=user, messages=[]))).content
    state["fail"] = True
    response = asyncio.run(wrapper.call(AgentInput(thread_id=user, messages=[])))
    trace = json.dumps(response.metadata)
    assert "session-" + user not in trace
    assert "token-" + user not in trace
    assert response.metadata["external_agent"]["success"] is False


def test_authenticated_requests_do_not_follow_redirects(auth_app):
    url, state = auth_app
    config = interface("form")
    config.setup_requests[-1].path = "/redirect"
    response = asyncio.run(
        create_chat_wrapper(config, url).call(AgentInput(thread_id="user", messages=[]))
    )
    assert response.metadata["external_agent"]["success"] is False
    assert all(path not in {"/steal", "/chat"} for path, _, _ in state["calls"])


def test_rendered_token_cannot_inject_headers(auth_app):
    url, state = auth_app
    state["token"] = "private\r\nInjected: value"
    response = asyncio.run(
        create_chat_wrapper(interface("json"), url).call(
            AgentInput(thread_id="user", messages=[])
        )
    )
    assert response.metadata["external_agent"]["success"] is False
    assert all(path not in {"/configure", "/chat"} for path, _, _ in state["calls"])
    assert "Injected" not in json.dumps(response.metadata)


def test_readiness_uses_same_form_cookie_authenticated_chat(auth_app, tmp_path):
    url, state = auth_app
    asyncio.run(
        probe_chat_ingress(
            interface=interface("form"),
            endpoint=url,
            document={
                "scenario_key": "one",
                "instruction": "hello",
                "persona": {"initial_message": "Hello"},
            },
            authoring=tmp_path,
            candidate_hash="candidate",
            bundle_digest="bundle",
            secret_values={},
            redaction_values={},
            timeout=3,
        )
    )
    receipt = next((tmp_path / "generic-harness" / "chat-readiness").glob("*.json"))
    assert json.loads(receipt.read_text())["status"] == "passed"
    assert [path for path, _, _ in state["calls"]] == [
        "/register",
        "/login-form",
        "/configure",
        "/chat",
    ]


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "other"},
        {"Content-Length": "100"},
        {"X-Key": "a\r\nb"},
        {"Authorization": "x", "authorization": "y"},
    ],
)
def test_contract_rejects_unsafe_headers(headers):
    with pytest.raises(ValueError):
        RuntimeInterface(kind="http", port=8080, path="/chat", headers_template=headers)


def test_contract_header_placeholders_obey_setup_order():
    config = interface("header")
    raw = config.model_dump()
    raw["setup_requests"][0]["headers_template"] = {
        "Authorization": "{{session_token}}"
    }
    with pytest.raises(ValueError, match="unknown_placeholders"):
        RuntimeInterface.model_validate(raw)


def test_public_capture_does_not_enable_unknown_credential_diagnostics(auth_app):
    url, state = auth_app
    wrapper = create_chat_wrapper(interface("form"), url)
    assert asyncio.run(
        wrapper.call(AgentInput(thread_id="public-user", messages=[]))
    ).content
    state["fail"] = True
    state["unknown"] = "unrelated-opaque-sensitive-value"
    response = asyncio.run(
        wrapper.call(AgentInput(thread_id="public-user", messages=[]))
    )
    assert state["unknown"] not in json.dumps(response.metadata)
    assert (
        "credential context unavailable"
        in response.metadata["external_agent"]["response_excerpt"]
    )


@pytest.mark.parametrize("mode", ["form", "json", "header"])
def test_native_completed_events_use_authenticated_session(auth_app, mode):
    url, state = auth_app
    raw = interface(mode).model_dump()
    raw["completed_tool_events"] = {
        "correlation_header": "X-Conversation-ID",
        "path": "/events/{{correlation_id}}",
        "items_path": "items",
        "next_page_path": "next",
        "page_parameter": "page",
        "kind_path": "kind",
        "action_kind": "action",
        "observation_kind": "completed",
        "action_id_path": "id",
        "observation_action_id_path": "action_id",
        "name_path": "name",
        "arguments_path": "arguments",
        "result_path": "result",
        "error_path": "failed",
    }
    wrapper = create_chat_wrapper(RuntimeInterface.model_validate(raw), url)
    response = asyncio.run(
        wrapper.call(AgentInput(thread_id="event-user", messages=[]))
    )
    assert response.content == "hello event-user"
    tools = response.metadata["native_completed_tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "read_file"
    assert tools[0]["result"] == {"content": "fixture 84"}
    assert state["calls"][-1][0] == "/events/event-user"


def test_existing_explicit_cookie_auth_and_bare_value_redaction(auth_app):
    from fi.simulate.agent.wrappers.http import HTTPAgentWrapper

    url, state = auth_app
    user = "preauthenticated"
    token = "session-" + user
    wrapper = HTTPAgentWrapper(
        endpoint=url + "/chat",
        protocol="json_template",
        headers={"Cookie": "session=" + token},
        request_template={"user": user},
        response_path="content",
        secret_values={},
    )
    assert (
        asyncio.run(wrapper.call(AgentInput(thread_id="one", messages=[]))).content
        == "hello " + user
    )
    state["fail"] = True
    state["unknown"] = token
    response = asyncio.run(wrapper.call(AgentInput(thread_id="one", messages=[])))
    assert response.metadata["external_agent"]["success"] is False
    assert token not in json.dumps(response.metadata)
    assert "echo" in response.metadata["external_agent"]["response_excerpt"]
