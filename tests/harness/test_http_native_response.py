import asyncio
import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from fi.alk.harness.chat_call_runner import _HostedChatTarget
from fi.alk.harness.chat_transport import create_chat_wrapper
from fi.alk.harness.contract import RuntimeInterface, RuntimeHttpFailureRule
from fi.simulate.agent.wrapper import AgentInput


ROWS = {
    "items_path": "tool_calls",
    "name_path": "tool_name",
    "arguments_path": "tool_arguments",
    "result_path": "tool_result",
    "id_path": "tool_id",
}
EVENTS = {
    "source": "response",
    "items_path": "output",
    "kind_path": "type",
    "action_kind": "function_call",
    "observation_kind": "function_call_output",
    "action_id_path": "call_id",
    "observation_action_id_path": "call_id",
    "name_path": "name",
    "arguments_path": "arguments",
    "result_path": "output",
    "status_path": "status",
    "completed_status": "completed",
}
PAYLOAD = {
    "answer": "The fixture contains 84.",
    "error_msg": None,
    "tool_calls": [
        {
            "tool_id": "tool-1",
            "tool_name": "read_document",
            "tool_arguments": {"path": "fixture.txt"},
            "tool_result": "fixture 84",
        }
    ],
}
PAIR_PAYLOAD = {
    "answer": "The fixture contains 84.",
    "error_msg": None,
    "output": [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "read_document",
            "arguments": '{"path":"fixture.txt"}',
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "fixture 84",
            "status": "completed",
        },
        {"type": "message", "content": [{"text": "The fixture contains 84."}]},
    ],
}


@pytest.fixture
def native_app():
    state = {
        "payload": copy.deepcopy(PAYLOAD),
        "calls": [],
        "setup_status": 200,
        "setup_body": {"id": "created"},
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["calls"].append((self.path, body, dict(self.headers)))
            status, response = (
                (state["setup_status"], state["setup_body"])
                if self.path == "/setup"
                else (200, state["payload"])
            )
            encoded = json.dumps(response).encode()
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


def interface(**changes):
    raw = dict(
        kind="http",
        port=8080,
        path="/chat",
        protocol="json_template",
        include_tools=False,
        request_template={"message": "{{new_message_content}}"},
        response_path="answer",
        completed_tools=ROWS,
        failure_rule={
            "status_path": "error_msg",
            "success_values": [None],
            "message_path": "error_msg",
        },
    )
    raw.update(changes)
    return RuntimeInterface(**raw)


def invoke(wrapper):
    return asyncio.run(
        wrapper.call(
            AgentInput(
                thread_id="one", messages=[], new_message={"content": "read fixture"}
            )
        )
    )


@pytest.mark.parametrize("paired", [False, True])
def test_response_tools_recorded_once_and_never_reexecuted(native_app, paired):
    base, state = native_app
    config = interface()
    if paired:
        state["payload"] = copy.deepcopy(PAIR_PAYLOAD)
        config = interface(
            completed_tools=None,
            completed_tool_events=EVENTS,
            response_path="output.content.0.text",
        )
    wrapper = create_chat_wrapper(config, base, secret_values={})
    world = SimpleNamespace(calls=[])
    contract = SimpleNamespace(runtime=SimpleNamespace(interface=config))
    target = _HostedChatTarget(
        wrapper=wrapper,
        contract=contract,
        world=world,
        scenario_key="one",
        scenario_id="one",
    )
    assert asyncio.run(target.say("read fixture")) == "The fixture contains 84."
    assert len(world.calls) == 1
    call = world.calls[0]
    assert (call.name, call.arguments, call.result, call.ok) == (
        "read_document",
        {"path": "fixture.txt"},
        "fixture 84",
        True,
    )
    assert [path for path, _, _ in state["calls"]] == ["/chat"]


def test_cumulative_completed_rows_are_reported_once_per_thread(native_app):
    base, state = native_app
    wrapper = create_chat_wrapper(interface(), base, secret_values={})
    first = invoke(wrapper)
    state["payload"]["tool_calls"].append(
        {
            "tool_id": "tool-2",
            "tool_name": "read_document",
            "tool_arguments": {"path": "second.txt"},
            "tool_result": "second fixture",
        }
    )
    second = invoke(wrapper)
    assert [row["result"] for row in first.metadata["native_completed_tools"]] == [
        "fixture 84"
    ]
    assert [row["result"] for row in second.metadata["native_completed_tools"]] == [
        "second fixture"
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_result",
        "arguments_list",
        "blank_name",
        "too_many",
        "not_array",
        "nonfinite",
    ],
)
def test_malformed_completed_rows_fail_closed(native_app, mutation):
    base, state = native_app
    row = state["payload"]["tool_calls"][0]
    if mutation == "missing_result":
        row.pop("tool_result")
    elif mutation == "arguments_list":
        row["tool_arguments"] = []
    elif mutation == "blank_name":
        row["tool_name"] = " "
    elif mutation == "too_many":
        state["payload"]["tool_calls"] = [row] * 129
    elif mutation == "not_array":
        state["payload"]["tool_calls"] = {"unexpected": row}
    else:
        row["tool_arguments"] = {"value": float("nan")}
    result = invoke(create_chat_wrapper(interface(), base, secret_values={}))
    assert result.metadata["external_agent"]["success"] is False
    assert not result.tool_calls
    assert not result.metadata.get("native_completed_tools")


@pytest.mark.parametrize(
    "mutation",
    ["missing_pair", "duplicate", "pending", "missing_status", "invalid_arguments"],
)
def test_response_event_pairs_require_completed_matching_records(native_app, mutation):
    base, state = native_app
    state["payload"] = copy.deepcopy(PAIR_PAYLOAD)
    rows = state["payload"]["output"]
    if mutation == "missing_pair":
        rows.pop(1)
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "pending":
        rows[1]["status"] = "running"
    elif mutation == "missing_status":
        rows[0].pop("status")
    else:
        rows[0]["arguments"] = "not JSON"
    result = invoke(
        create_chat_wrapper(
            interface(completed_tools=None, completed_tool_events=EVENTS),
            base,
            secret_values={},
        )
    )
    assert result.metadata["external_agent"]["success"] is False
    assert not result.tool_calls
    assert not result.metadata.get("native_completed_tools")


@pytest.mark.parametrize(
    "status", ["arbitrary upstream error", "", False, 0, "missing"]
)
def test_success_predicate_rejects_every_other_status(native_app, status):
    base, state = native_app
    if status == "missing":
        state["payload"].pop("error_msg")
    else:
        state["payload"]["error_msg"] = status
    result = invoke(create_chat_wrapper(interface(), base, secret_values={}))
    assert result.metadata["external_agent"]["success"] is False
    assert result.metadata["external_agent"]["failure_stage"] == "application"
    assert not result.metadata.get("native_completed_tools")


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"failure_values": [], "success_values": [None]},
        {"failure_values": ["failed"], "success_values": [None]},
    ],
)
def test_failure_rule_requires_exactly_one_nonempty_predicate(values):
    with pytest.raises(ValueError):
        RuntimeHttpFailureRule(status_path="status", **values)


def test_completed_modes_cannot_be_mixed():
    with pytest.raises(ValueError):
        interface(completed_tool_events=EVENTS)


def test_private_setup_value_is_sent_only_to_declared_setup_and_redacted(native_app):
    base, state = native_app
    private = 'opaque-provider-value/with"quotes'
    config = interface(
        setup_secrets={"provider_key": "TARGET_KEY"},
        setup_requests=[
            {
                "path": "/setup",
                "body_template": {"api_key": "{{provider_key}}"},
                "capture": {"echoed": "returned"},
            }
        ],
    )
    captured = 'opaque-captured-value/with"different-quotes'
    state["setup_body"] = {"returned": captured}
    state["payload"]["error_msg"] = "unrelated label " + private + " and " + captured
    wrapper = create_chat_wrapper(config, base, secret_values={"TARGET_KEY": private})
    result = invoke(wrapper)
    assert state["calls"][0][1] == {"api_key": private}
    assert private not in json.dumps(state["calls"][1][1])
    assert private not in config.model_dump_json()
    for value in (private, captured):
        assert value not in result.metadata["external_agent"]["error"]
        assert value not in result.metadata["external_agent"]["response_excerpt"]
        assert json.dumps(value)[1:-1] not in json.dumps(result.metadata)
    assert result.metadata["external_agent"]["success"] is False
    assert "unrelated label" in result.metadata["external_agent"]["error"]


def test_private_bearer_header_authenticates_each_chat_without_leaking(native_app):
    base, state = native_app
    private = 'private-gateway-token/with"quotes'
    config = interface(
        setup_secrets={"gateway_token": "TARGET_KEY"},
        headers_template={"Authorization": "Bearer {{gateway_token}}"},
    )
    state["payload"]["error_msg"] = "authentication failed: " + private
    wrapper = create_chat_wrapper(config, base, secret_values={"TARGET_KEY": private})
    for _ in range(2):
        result = invoke(wrapper)
        assert private not in json.dumps(result.metadata)
        assert json.dumps(private)[1:-1] not in json.dumps(result.metadata)
    assert len(state["calls"]) == 2
    for path, body, headers in state["calls"]:
        assert path == "/chat"
        assert headers["Authorization"] == "Bearer " + private
        assert private not in json.dumps(body)
    assert private not in config.model_dump_json()


@pytest.mark.parametrize("secrets", [None, {}, {"TARGET_KEY": ""}])
def test_missing_setup_alias_fails_before_request(native_app, secrets):
    base, state = native_app
    config = interface(
        setup_secrets={"provider_key": "TARGET_KEY"},
        setup_requests=[
            {"path": "/setup", "body_template": {"key": "{{provider_key}}"}}
        ],
    )
    with pytest.raises(ValueError, match="target secret missing"):
        create_chat_wrapper(config, base, secret_values=secrets)
    assert state["calls"] == []


@pytest.mark.parametrize(
    "changes",
    [
        {"path": "/{{provider_key}}"},
        {"request_template": {"key": "{{provider_key}}"}},
        {"setup_requests": [{"path": "/{{provider_key}}"}]},
        {"setup_requests": [{"path": "/setup", "capture": {"provider_key": "id"}}]},
    ],
)
def test_private_placeholders_cannot_enter_paths_turns_or_be_shadowed(changes):
    with pytest.raises(ValueError):
        interface(setup_secrets={"provider_key": "TARGET_KEY"}, **changes)


@pytest.mark.parametrize("paired", [False, True])
def test_completed_mapping_rejects_separate_tool_requests(native_app, paired):
    base, state = native_app
    config = interface()
    if paired:
        state["payload"] = copy.deepcopy(PAIR_PAYLOAD)
        config = interface(completed_tools=None, completed_tool_events=EVENTS)
    state["payload"]["other"] = {
        "tool_calls": [
            {"id": "requested", "function": {"name": "delete_file", "arguments": "{}"}}
        ]
    }
    result = invoke(create_chat_wrapper(config, base, secret_values={}))
    assert result.metadata["external_agent"]["success"] is False
    assert not result.tool_calls
    assert not result.metadata.get("native_completed_tools")


def test_response_event_ids_deduplicate_history_per_thread(native_app):
    base, state = native_app
    state["payload"] = copy.deepcopy(PAIR_PAYLOAD)
    wrapper = create_chat_wrapper(
        interface(completed_tools=None, completed_tool_events=EVENTS),
        base,
        secret_values={},
    )
    first, repeated = invoke(wrapper), invoke(wrapper)
    assert len(first.metadata["native_completed_tools"]) == 1
    assert repeated.metadata["native_completed_tools"] == []
    other_thread = asyncio.run(
        wrapper.call(
            AgentInput(thread_id="other", messages=[], new_message={"content": "read"})
        )
    )
    assert len(other_thread.metadata["native_completed_tools"]) == 1


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("missing_pair", "tool events are missing a paired action or completion"),
        ("invalid_data", "invalid completed tool data"),
        ("unfinished", "tool event is not completed"),
    ],
)
def test_tool_failure_reports_fixed_reason_without_payload(native_app, defect, reason):
    base, state = native_app
    payload = copy.deepcopy(PAIR_PAYLOAD)
    payload["private_value"] = "unregistered-secret-value"
    if defect == "missing_pair":
        del payload["output"][1]
    elif defect == "invalid_data":
        payload["output"][0]["name"] = ""
    else:
        payload["output"][0]["status"] = "in_progress"
    state["payload"] = payload
    wrapper = create_chat_wrapper(
        interface(completed_tools=None, completed_tool_events=EVENTS),
        base,
        secret_values={},
    )
    response = invoke(wrapper)
    trace = response.metadata["external_agent"]
    assert trace["success"] is False
    assert trace["failure_stage"] == "tool_events"
    assert trace["error"] == "Completed tool evidence collection failed: " + reason
    assert trace["response_excerpt"] == ""
    assert "unregistered-secret-value" not in json.dumps(response.metadata)


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_unrecognized_tool_error_message_stays_private(
    native_app, monkeypatch, error_type
):
    base, state = native_app
    state["payload"] = copy.deepcopy(PAIR_PAYLOAD)
    wrapper = create_chat_wrapper(
        interface(completed_tools=None, completed_tool_events=EVENTS),
        base,
        secret_values={},
    )

    def fail(*args, **kwargs):
        raise error_type("unregistered-secret-value")

    monkeypatch.setattr(wrapper, "_completed_response_events", fail)
    response = invoke(wrapper)
    trace = response.metadata["external_agent"]
    assert trace["success"] is False
    assert (
        trace["error"]
        == "Completed tool evidence collection failed: " + error_type.__name__
    )
    assert trace["response_excerpt"] == ""
    assert "unregistered-secret-value" not in json.dumps(response.metadata)
