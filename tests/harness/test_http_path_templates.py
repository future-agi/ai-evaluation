import asyncio
import json
from urllib.parse import urlparse

import pytest

from fi.simulate.agent.wrapper import AgentInput
from fi.simulate.agent.wrappers.http import HTTPAgentWrapper


def test_captured_paths_render_after_setup_with_fixed_authority(monkeypatch):
    wrapper = HTTPAgentWrapper(
        endpoint="http://fixture.invalid:8080/threads/{{session_key}}/turns?mode=sync",
        protocol="json_template",
        request_template={"text": "{{new_message_content}}"},
        response_path="answer",
        secret_values={},
        setup_requests=[
            {"path": "/threads", "capture": {"session_key": "id"}},
            {"path": "/threads/{{session_key}}/activate"},
        ],
    )
    calls = []
    identifier = "../other/path?x=1#frag%"

    def request(endpoint, *, method, payload, headers):
        calls.append((endpoint, method, payload))
        return 200, {"id": identifier, "answer": "done"}

    monkeypatch.setattr(wrapper, "_request_json", request)
    original = wrapper.endpoint
    for text in ("first", "second"):
        response = asyncio.run(
            wrapper.call(
                AgentInput(thread_id="t", messages=[], new_message={"content": text})
            )
        )
        assert response.content == "done"
    segment = "%2E%2E%2Fother%2Fpath%3Fx%3D1%23frag%25"
    assert calls == [
        ("http://fixture.invalid:8080/threads", "POST", {}),
        (f"http://fixture.invalid:8080/threads/{segment}/activate", "POST", {}),
        (
            f"http://fixture.invalid:8080/threads/{segment}/turns?mode=sync",
            "POST",
            {"text": "first"},
        ),
        (
            f"http://fixture.invalid:8080/threads/{segment}/turns?mode=sync",
            "POST",
            {"text": "second"},
        ),
    ]
    assert all(urlparse(url).netloc == "fixture.invalid:8080" for url, _, _ in calls)
    assert wrapper.endpoint == original
    assert identifier not in json.dumps(response.metadata)


@pytest.mark.parametrize("capture", [{}, {"key": None}, {"key": ""}])
def test_missing_path_capture_stops_before_main_request(monkeypatch, capture):
    wrapper = HTTPAgentWrapper(
        endpoint="http://fixture.invalid/threads/{{key}}", secret_values={}
    )
    wrapper._setup_values = capture
    calls = []
    monkeypatch.setattr(
        wrapper, "_request_json", lambda *args, **kwargs: calls.append(args)
    )
    response = asyncio.run(wrapper.call(AgentInput(thread_id="t", messages=[])))
    assert calls == []
    assert response.metadata["external_agent"]["success"] is False
    assert "path placeholder" in response.metadata["external_agent"]["error"]


def test_static_endpoint_keeps_exact_url(monkeypatch):
    endpoint = "http://fixture.invalid/static;param?x=a%2Fb"
    wrapper = HTTPAgentWrapper(endpoint=endpoint)
    calls = []

    def request(url, **kwargs):
        calls.append(url)
        return 200, {"content": "done"}

    monkeypatch.setattr(wrapper, "_request_json", request)
    asyncio.run(wrapper.call(AgentInput(thread_id="t", messages=[])))
    assert calls == [endpoint]


def test_failed_request_redacts_raw_and_encoded_capture(monkeypatch):
    wrapper = HTTPAgentWrapper(
        endpoint="http://fixture.invalid/{{key}}", secret_values={}
    )
    wrapper._setup_values = {"key": "private/key-value"}

    def request(endpoint, **kwargs):
        raise RuntimeError("failed " + endpoint + " private/key-value")

    monkeypatch.setattr(wrapper, "_request_json", request)
    response = asyncio.run(wrapper.call(AgentInput(thread_id="t", messages=[])))
    trace = json.dumps(response.metadata)
    assert "private/key-value" not in trace
    assert "private%2Fkey-value" not in trace
    assert "{{key}}" in trace


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://{{host}}/turn",
        "http://fixture.invalid/turn?key={{key}}",
    ],
)
def test_substitution_outside_path_is_rejected(monkeypatch, endpoint):
    wrapper = HTTPAgentWrapper(endpoint=endpoint, secret_values={})
    wrapper._setup_values = {"key": "value", "host": "other.invalid"}
    calls = []
    monkeypatch.setattr(
        wrapper, "_request_json", lambda *args, **kwargs: calls.append(args)
    )
    response = asyncio.run(wrapper.call(AgentInput(thread_id="t", messages=[])))
    assert calls == []
    assert response.metadata["external_agent"]["success"] is False


def test_setup_without_capture_value_does_not_send_main_request(monkeypatch):
    wrapper = HTTPAgentWrapper(
        endpoint="http://fixture.invalid/{{key}}",
        secret_values={},
        setup_requests=[{"path": "/setup", "capture": {"key": "missing"}}],
    )
    calls = []

    def request(endpoint, **kwargs):
        calls.append(endpoint)
        return 200, {}

    monkeypatch.setattr(wrapper, "_request_json", request)
    response = asyncio.run(wrapper.call(AgentInput(thread_id="t", messages=[])))
    assert calls == ["http://fixture.invalid/setup"]
    assert response.metadata["external_agent"]["success"] is False


@pytest.mark.parametrize("protocol", ["fi.alk", "openai_chat", "json_template"])
def test_contract_path_allows_previous_capture_but_rejects_unknown_name(protocol):
    from fi.alk.harness.contract import RuntimeInterface

    options = dict(
        kind="http",
        protocol=protocol,
        port=8080,
        path="/threads/{{key}}",
        request_template={},
        setup_requests=[{"path": "/setup", "capture": {"key": "id"}}],
    )
    assert RuntimeInterface(**options).path == "/threads/{{key}}"
    options["path"] = "/threads/{{missing}}"
    with pytest.raises(ValueError, match="runtime_http_request_unknown_placeholders"):
        RuntimeInterface(**options)


@pytest.mark.parametrize("protocol", ["fi.alk", "openai_chat", "json_template"])
def test_contract_setup_path_requires_earlier_capture(protocol):
    from fi.alk.harness.contract import RuntimeInterface

    options = dict(
        kind="http",
        protocol=protocol,
        port=8080,
        path="/turn",
        request_template={},
        setup_requests=[
            {"path": "/threads", "capture": {"key": "id"}},
            {"path": "/threads/{{key}}/activate"},
        ],
    )
    assert RuntimeInterface(**options).setup_requests[1].path == "/threads/{{key}}/activate"
    options["setup_requests"] = [
        {"path": "/threads/{{key}}", "capture": {"key": "id"}}
    ]
    with pytest.raises(ValueError, match="runtime_http_setup_unknown_placeholders"):
        RuntimeInterface(**options)


@pytest.mark.parametrize("protocol", ["fi.alk", "openai_chat", "json_template"])
def test_contract_body_template_validation_stays_protocol_scoped(protocol):
    from fi.alk.harness.contract import RuntimeInterface

    options = dict(
        kind="http",
        protocol=protocol,
        port=8080,
        path="/turn",
        request_template={"value": "{{unknown_body_value}}"},
    )
    if protocol == "json_template":
        with pytest.raises(
            ValueError, match="runtime_http_request_unknown_placeholders"
        ):
            RuntimeInterface(**options)
    else:
        assert RuntimeInterface(**options).path == "/turn"
