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
from fi.simulate.agent.wrappers.http import HTTPAgentWrapper
from fi.simulate.agent.wrappers.http_events import CompletedToolEvents, CompletedToolRows


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
    "error_path": "failed",
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
            "failed": False,
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


@pytest.mark.parametrize("status", ["completed", "in_progress"])
def test_response_failure_events_preserve_status_and_dedup_guards(native_app, status):
    base, state = native_app
    state["payload"] = copy.deepcopy(PAIR_PAYLOAD)
    state["payload"]["output"][1] = {
        "type": "rejected", "call_id": "call-1", "message": "invalid path", "status": status,
    }
    events = {**EVENTS, "error_path": "", "failure_events": [
        {"kind": "rejected", "action_id_path": "call_id", "result_path": "message"},
    ]}
    wrapper = create_chat_wrapper(
        interface(completed_tools=None, completed_tool_events=events), base, secret_values={},
    )
    first = invoke(wrapper)
    assert first.metadata["external_agent"]["success"] is (status == "completed")
    if status == "completed":
        assert first.metadata["native_completed_tools"] == [{
            "name": "read_document", "arguments": {"path": "fixture.txt"},
            "result": "invalid path", "ok": False,
        }]
        assert invoke(wrapper).metadata["native_completed_tools"] == []
    else:
        assert not first.metadata.get("native_completed_tools")


@pytest.mark.parametrize("paired", [False, True])
def test_declared_result_errors_classify_completed_tools(native_app, paired):
    base, state = native_app
    bodies = [
        json.dumps({"status": "error", "error": "unattended execution blocked"}),
        json.dumps({"stdout": "42", "exit_code": 0}),
        {"revision": 2, "todos": [{"title": "calculate", "status": "completed"}], "error": None},
        json.dumps({"error": "search backend unavailable"}),
    ]
    names = ["execute_code", "terminal", "tool_call", "search_files"]
    arguments = [{"code": "print(42)"}, {"command": "echo 42"},
                 {"calls": [{"name": "todo_list", "arguments": {"status": "completed"}}]},
                 {"pattern": "fixture"}]
    if paired:
        state["payload"] = copy.deepcopy(PAIR_PAYLOAD)
        state["payload"]["output"] = [
            event
            for index, (name, args, body) in enumerate(zip(names, arguments, bodies))
            for event in [
                {"type": "function_call", "call_id": str(index), "name": name,
                 "arguments": json.dumps(args), "status": "completed"},
                {"type": "function_call_output", "call_id": str(index),
                 "output": body, "status": "completed"},
            ]
        ]
        config = interface(completed_tools=None,
                           completed_tool_events={**EVENTS, "error_path": "", "result_error_path": "error"})
    else:
        state["payload"]["tool_calls"] = [
            {
                "tool_id": str(index),
                "tool_name": name,
                "tool_arguments": args,
                "tool_result": body,
            }
            for index, (name, args, body) in enumerate(zip(names, arguments, bodies))
        ]
        config = interface(completed_tools={**ROWS, "result_error_path": "error"}, failure_rule=None)
    target = create_chat_wrapper(config, base, secret_values={})
    response = invoke(target)
    assert response.metadata["external_agent"]["success"] is True
    tools = response.metadata["native_completed_tools"]
    assert [call["ok"] for call in tools] == [False, True, True, False]
    assert [call["result"] for call in tools] == bodies
    assert [call["name"] for call in tools] == names
    assert [call["arguments"] for call in tools] == arguments
    if paired:
        assert invoke(target).metadata["native_completed_tools"] == []


@pytest.mark.parametrize("paired", [False, True])
def test_result_error_values_and_paths_are_strict(native_app, paired):
    base, state = native_app
    for body, expected in [
        ({"error": None}, True), ({"error": False}, True), ({"error": ""}, True),
        ({}, True), ({"error": True}, False), ({"error": "blocked"}, False),
        ({"error": 0}, None), ({"error": 1}, None), ({"error": {}}, None),
        ({"error": []}, None), ({"error": float("nan")}, None),
        (None, None), ([], None), ("broken JSON", None),
        (json.dumps(json.dumps({"error": "blocked"})), None),
    ]:
        for result in [body, json.dumps(body)]:
            payload = copy.deepcopy(PAIR_PAYLOAD if paired else PAYLOAD)
            if paired:
                payload["output"][1]["output"] = result
                config = interface(completed_tools=None, completed_tool_events={
                    **EVENTS, "error_path": "", "result_error_path": "error"})
            else:
                payload["tool_calls"][0]["tool_result"] = result
                config = interface(completed_tools={**ROWS, "result_error_path": "error"},
                                   failure_rule=None)
            state["payload"] = payload
            response = invoke(create_chat_wrapper(config, base, secret_values={}))
            assert response.metadata["external_agent"]["success"] is (expected is not None), result
            if expected is None:
                assert not response.metadata.get("native_completed_tools")
                assert response.metadata["external_agent"]["response_excerpt"] == ""
            else:
                assert response.metadata["native_completed_tools"][0]["ok"] is expected
                assert response.metadata["native_completed_tools"][0]["result"] == result


@pytest.mark.parametrize("paired", [False, True])
def test_result_errors_do_not_weaken_boolean_failure(native_app, paired):
    base, state = native_app
    state["payload"] = copy.deepcopy(PAIR_PAYLOAD if paired else PAYLOAD)
    if paired:
        state["payload"]["output"][1].update(output='{"error":null}', failed=True)
        config = interface(completed_tools=None, completed_tool_events={
            **EVENTS, "error_path": "failed", "result_error_path": "error"})
    else:
        state["payload"]["tool_calls"][0].update(tool_result='{"error":null}', failed=True)
        config = interface(completed_tools={
            **ROWS, "error_path": "failed", "result_error_path": "error"}, failure_rule=None)
    response = invoke(create_chat_wrapper(config, base, secret_values={}))
    assert response.metadata["external_agent"]["success"] is True
    assert response.metadata["native_completed_tools"][0]["ok"] is False


def test_result_error_paths_do_not_guess_nested_shapes_or_consume_bad_events(native_app):
    base, state = native_app
    state["payload"] = copy.deepcopy(PAIR_PAYLOAD)
    target = create_chat_wrapper(interface(completed_tools=None, completed_tool_events={
        **EVENTS, "error_path": "", "result_error_path": "outcome.error"}), base, secret_values={})
    for body in [{"outcome": None}, {"outcome": []}, {"outcome": "nested JSON"}]:
        state["payload"]["output"][1]["output"] = body
        response = invoke(target)
        assert response.metadata["external_agent"]["success"] is False
        assert not response.metadata.get("native_completed_tools")
    state["payload"]["output"][1]["output"] = {"outcome": {"error": "failed"}}
    response = invoke(target)
    assert response.metadata["external_agent"]["success"] is True
    assert response.metadata["native_completed_tools"][0]["ok"] is False
    assert invoke(target).metadata["native_completed_tools"] == []


def test_terminal_failure_bypasses_different_result_shape(native_app):
    base, state = native_app
    state["payload"] = copy.deepcopy(PAIR_PAYLOAD)
    state["payload"]["output"][1] = {
        "type": "rejected", "call_id": "call-1", "message": "not JSON", "status": "completed",
    }
    config = interface(completed_tools=None, completed_tool_events={
        **EVENTS, "error_path": "", "result_error_path": "error", "failure_events": [
            {"kind": "rejected", "action_id_path": "call_id", "result_path": "message"},
        ],
    })
    response = invoke(create_chat_wrapper(config, base, secret_values={}))
    assert response.metadata["external_agent"]["success"] is True
    assert response.metadata["native_completed_tools"][0]["ok"] is False
    assert response.metadata["native_completed_tools"][0]["result"] == "not JSON"


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("application_success", [False, True])
def test_response_event_completion_requires_a_tool_outcome_mapping(direct, application_success):
    events = {key: value for key, value in EVENTS.items() if key != "error_path"}
    failure_rule = {"status_path": "error_msg", "success_values": [None]} if application_success else None
    with pytest.raises(ValueError, match="response events require a tool outcome mapping"):
        if direct:
            HTTPAgentWrapper(
                endpoint="http://localhost:8080/chat", protocol="json_template",
                request_template={"input": "{{messages}}"},
                completed_tool_events=events, failure_rule=failure_rule,
            )
        else:
            interface(completed_tools=None, completed_tool_events=events,
                      failure_rule=failure_rule)


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("path", [
    " ", " error", "error ", "out come.error", "outcome.\nerror", "outcome[0].error",
    "$['error']", ".error", "error.", "outcome..error", "$", "$.", "$error",
])
def test_malformed_result_error_selectors_rejected_before_chat(paired, path):
    model, spec = (CompletedToolEvents, EVENTS) if paired else (CompletedToolRows, ROWS)
    with pytest.raises(ValueError, match="invalid tool result error path"):
        model.model_validate({**spec, "result_error_path": path})


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("path", ["out-comes.0.érror", "$.out-comes.0.érror"])
def test_result_error_selector_preserves_keys_and_numeric_indices(native_app, paired, path):
    base, state = native_app
    state["payload"] = copy.deepcopy(PAIR_PAYLOAD if paired else PAYLOAD)
    result = {"out-comes": [{"érror": "blocked"}]}
    if paired:
        state["payload"]["output"][1]["output"] = result
        config = interface(completed_tools=None, completed_tool_events={
            **EVENTS, "error_path": "", "result_error_path": path})
    else:
        state["payload"]["tool_calls"][0]["tool_result"] = result
        config = interface(completed_tools={**ROWS, "result_error_path": path}, failure_rule=None)
    response = invoke(create_chat_wrapper(config, base, secret_values={}))
    assert response.metadata["external_agent"]["success"] is True
    assert response.metadata["native_completed_tools"][0]["ok"] is False
    assert response.metadata["native_completed_tools"][0]["result"] == result
