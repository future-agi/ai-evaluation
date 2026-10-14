from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from fi.simulate.agent.wrapper import AgentInput
from fi.simulate.agent.wrappers.http import HTTPAgentWrapper


@pytest.fixture
def endpoint():
    state = {"status": 500, "body": b'{"error":"upstream refused"}'}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(state["status"])
            self.end_headers()
            self.wfile.write(state["body"])

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/call", state
    server.shutdown()
    thread.join()
    server.server_close()


@pytest.mark.parametrize(
    "status,body,stage",
    [
        (500, b'{"error":"upstream refused"}', "http"),
        (502, b"upstream refused", "http"),
        (200, b"not JSON", "decode"),
        (200, b'{"other":"value"}', "extract"),
    ],
)
def test_http_failure_preserves_status_without_assistant(endpoint, status, body, stage):
    url, state = endpoint
    state.update(status=status, body=body)
    response = asyncio.run(
        HTTPAgentWrapper(
            endpoint=url,
            protocol="json_template",
            request_template={},
            response_path="messages",
            secret_values={},
        ).call(AgentInput(thread_id="t", messages=[], new_message={}))
    )
    trace = response.metadata["external_agent"]
    assert trace["success"] is False
    assert trace["status_code"] == status, trace
    assert trace["failure_stage"] == stage
    assert trace["response_excerpt"]
    assert response.content == ""
    if status >= 400:
        assert "upstream refused" in trace["error"]
        assert "response_path" not in trace["error"]


def test_application_failure_is_explicit(endpoint):
    url, state = endpoint
    state.update(
        status=200,
        body=b'{"state":"failed","error":"legitimate data","messages":"answer"}',
    )
    options = dict(
        endpoint=url,
        protocol="json_template",
        request_template={},
        response_path="messages",
        secret_values={},
    )
    normal = asyncio.run(
        HTTPAgentWrapper(**options).call(
            AgentInput(thread_id="t", messages=[], new_message={})
        )
    )
    assert normal.content == "answer"
    failed = asyncio.run(
        HTTPAgentWrapper(
            **options,
            failure_rule={
                "status_path": "state",
                "failure_values": ["failed"],
                "message_path": "error",
            },
        ).call(AgentInput(thread_id="t", messages=[], new_message={}))
    )
    assert failed.metadata["external_agent"]["failure_stage"] == "application"
    assert failed.content == ""


def test_diagnostic_redacts_admitted_and_structured_secrets(endpoint):
    url, state = endpoint
    secret = 'line one\nline "two"'
    state.update(
        body=json.dumps(
            {
                "error": secret,
                "Authorization": "Bearer unknown",
                "url": "https://user:password@host/path?token=other",
            }
        ).encode()
    )
    response = asyncio.run(
        HTTPAgentWrapper(
            endpoint=url,
            secret_values={"credential": json.dumps({"private_key": secret})},
        ).call(AgentInput(thread_id="t", messages=[], new_message={}))
    )
    evidence = json.dumps(response.metadata)
    assert "line one" not in evidence
    assert "unknown" not in evidence
    assert "password" not in evidence
    assert "token=other" not in evidence


def test_response_size_and_missing_credential_context(endpoint):
    url, state = endpoint
    state.update(status=200, body=b"x" * (1024 * 1024 + 1))
    result = asyncio.run(
        HTTPAgentWrapper(endpoint=url, secret_values={}).call(
            AgentInput(thread_id="t", messages=[], new_message={})
        )
    )
    trace = result.metadata["external_agent"]
    assert trace["failure_stage"] == "size"
    assert trace["status_code"] == 200
    assert trace["response_excerpt"] == ""
    state.update(status=502, body=b"unknown credentials should not be retained")
    result = asyncio.run(
        HTTPAgentWrapper(endpoint=url).call(
            AgentInput(thread_id="t", messages=[], new_message={})
        )
    )
    assert "unknown credentials" not in json.dumps(result.metadata)
    assert result.metadata["external_agent"]["status_code"] == 502


def test_setup_and_transport_failure_stages(endpoint, monkeypatch):
    url, state = endpoint
    result = asyncio.run(
        HTTPAgentWrapper(
            endpoint=url,
            secret_values={},
            setup_requests=[{"path": "/setup", "accepted_statuses": [201]}],
        ).call(AgentInput(thread_id="t", messages=[], new_message={}))
    )
    trace = result.metadata["external_agent"]
    assert trace["failure_stage"] == "setup"
    assert trace["status_code"] == 500

    import urllib.request

    def timeout(*args, **kwargs):
        raise TimeoutError("timed out password=secret123")

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", timeout)
    result = asyncio.run(
        HTTPAgentWrapper(endpoint=url, secret_values={"token": "secret123"}).call(
            AgentInput(thread_id="t", messages=[], new_message={})
        )
    )
    assert result.metadata["external_agent"]["failure_stage"] == "transport"
    assert "secret123" not in json.dumps(result.metadata)


@pytest.mark.parametrize("values", [[{}], [float("nan")], [], ["a"] * 33])
def test_declared_failure_rejects_invalid_scalars(values):
    from fi.alk.harness.contract import RuntimeHttpFailureRule

    with pytest.raises(ValueError):
        RuntimeHttpFailureRule(status_path="status", failure_values=values)


def test_authoring_rule_schema_matches_model_and_preserves_null(tmp_path):
    from fi.alk.harness.contract import RuntimeHttpFailureRule
    from fi.alk.harness.tools import contract_tools, _without_nulls

    submit = next(
        tool
        for tool in contract_tools(tmp_path).tools
        if tool.name == "submit_contract"
    )
    interface = submit.input_schema["properties"]["runtime"]["properties"]["interface"][
        "properties"
    ]
    declared = interface["failure_rule"]
    model = RuntimeHttpFailureRule.model_json_schema()
    assert set(declared["properties"]) == set(model["properties"])
    assert set(declared["required"]) == set(model["required"])
    assert (
        declared["properties"]["failure_values"]["maxItems"]
        == model["properties"]["failure_values"]["maxItems"]
    )
    parsed = RuntimeHttpFailureRule.model_validate(
        _without_nulls(
            {"status_path": "status", "failure_values": [None, False, 0, "failed"]}
        )
    )
    assert parsed.failure_values == [None, False, 0, "failed"]
    assert [type(value) for value in parsed.failure_values] == [
        type(None),
        bool,
        int,
        str,
    ]


def test_sanitizer_handles_unicode_json_escaping_and_short_credentials():
    from fi._diagnostics import sanitize_diagnostic

    secret = 'páss\n"word"'
    for ascii_only in (True, False):
        raw = json.dumps(
            {"error": secret, "token": "unadmitted"}, ensure_ascii=ascii_only
        )
        safe = sanitize_diagnostic(raw, {"credential": secret, "short": "xy"})
        assert (
            "páss" not in safe and "p\\u00e1ss" not in safe and "unadmitted" not in safe
        )
    assert "xy" not in sanitize_diagnostic("value xy", {"short": "xy"})


@pytest.mark.parametrize("body", [b"not JSON", b"x" * (1024 * 1024 + 1)])
def test_setup_malformed_or_oversized_body_keeps_status(endpoint, body):
    url, state = endpoint
    state.update(status=500, body=body)
    result = asyncio.run(
        HTTPAgentWrapper(
            endpoint=url,
            secret_values={},
            setup_requests=[{"path": "/setup", "accepted_statuses": [201]}],
        ).call(AgentInput(thread_id="t", messages=[], new_message={}))
    )
    trace = result.metadata["external_agent"]
    assert trace["failure_stage"] == "setup"
    assert trace["status_code"] == 500
    assert result.content == ""


def test_bare_custom_header_token_is_redacted(endpoint):
    url, state = endpoint
    state["body"] = b'{"error":"custom-token-value"}'
    result = asyncio.run(
        HTTPAgentWrapper(
            endpoint=url,
            secret_values={},
            headers={"Authorization": "Bearer custom-token-value"},
        ).call(AgentInput(thread_id="t", messages=[], new_message={}))
    )
    assert "custom-token-value" not in json.dumps(result.metadata)
