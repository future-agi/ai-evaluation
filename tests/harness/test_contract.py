import pytest
from pydantic import ValidationError

from fi.alk.harness.contract import AgentContract, RuntimeInterface, validate_contract


def test_response_path_schema_describes_supported_selector_in_authoring_and_repair(tmp_path):
    from fi.alk.harness.runtime_repair import _RuntimeSubmission
    from fi.alk.harness.tools import contract_tools
    from fi.simulate.agent.wrappers.http import _select_response_value

    description = RuntimeInterface.model_json_schema()["properties"]["response_path"]["description"]
    submit = next(tool for tool in contract_tools(tmp_path).tools if tool.name == "submit_contract")
    initial = submit.input_schema["properties"]["runtime"]["properties"]["interface"]["properties"]
    repair = _RuntimeSubmission.model_json_schema()["$defs"]["RuntimeInterface"]["properties"]
    assert initial["response_path"]["description"] == repair["response_path"]["description"] == description
    assert "dot-separated" in description
    assert "Negative indexes" in description
    assert "Bracket indexing and JSONPath expressions are not supported" in description
    assert "output.-1.content.0.text" in description
    payload = {"output": [{"content": [{"text": "earlier"}]}, {"content": [{"text": "answer"}]}]}
    assert _select_response_value(payload, "output.-1.content.0.text") == "answer"
    with pytest.raises(ValueError, match="response_path segment not found"):
        _select_response_value(payload, "output[-1].content[0].text")


def test_generic_http_contract_rejects_unbound_or_forward_template_variables():
    with pytest.raises(
        ValidationError, match="runtime_http_setup_unknown_placeholders"
    ):
        RuntimeInterface.model_validate(
            {
                "kind": "http",
                "port": 8080,
                "path": "/run",
                "protocol": "json_template",
                "setup_requests": [
                    {
                        "path": "/users/{{user_id}}/sessions",
                        "capture": {"session_id": "id"},
                    }
                ],
                "request_template": {"session_id": "{{session_id}}"},
            }
        )
    with pytest.raises(
        ValidationError, match="runtime_http_request_unknown_placeholders"
    ):
        RuntimeInterface.model_validate(
            {
                "kind": "http",
                "port": 8080,
                "path": "/run",
                "protocol": "json_template",
                "request_template": {"session_id": "{{session_id}}"},
            }
        )
    assert RuntimeInterface.model_validate(
        {
            "kind": "http",
            "port": 8080,
            "path": "/run",
            "protocol": "json_template",
            "setup_requests": [
                {
                    "path": "/users/{{thread_id}}/sessions",
                    "capture": {"session_id": "id"},
                }
            ],
            "request_template": {"session_id": "{{session_id}}"},
        }
    ).setup_requests[0].capture == {"session_id": "id"}


def test_an_import_entrypoint_without_module_and_callable_is_rejected_at_contract_time():
    """Bundling compiles a binding from exactly these two fields, so a half-filled entry is dead.

    Observed on a dev run: the model recorded `lookup_policy` as `construct` with no module or
    callable, the contract was accepted, and the job died three runtime-validation attempts later
    with `contract_tool_entry_incomplete`. Catching it here tells the model while it can still fix
    the entry.
    """
    contract = AgentContract.model_validate(
        {
            "agent": "hotel",
            "real_use_cases": ["book a room"],
            "tools": [{"name": "lookup_policy", "args": []}],
            "tool_entrypoints": [{"tool": "lookup_policy", "mode": "construct"}],
        }
    )
    problems = validate_contract(contract)
    assert any(
        "lookup_policy" in p and "needs-module-and-callable" in p for p in problems
    ), problems


def test_a_complete_entrypoint_and_an_unreachable_one_both_pass():
    contract = AgentContract.model_validate(
        {
            "agent": "hotel",
            "real_use_cases": ["book a room"],
            "tools": [{"name": "a", "args": []}, {"name": "b", "args": []}],
            "tool_entrypoints": [
                {
                    "tool": "a",
                    "mode": "construct",
                    "module": "pkg.mod",
                    "callable": "Klass.method",
                },
                {"tool": "b", "mode": "unreachable"},
            ],
        }
    )
    assert not [
        p for p in validate_contract(contract) if "needs-module-and-callable" in p
    ]


def test_failure_event_schema_is_available_in_authoring_and_repair(tmp_path):
    from fi.alk.harness.runtime_repair import _RuntimeSubmission
    from fi.alk.harness.tools import contract_tools
    from fi.simulate.agent.wrappers.http_events import CompletedToolFailureEvent

    submit = next(tool for tool in contract_tools(tmp_path).tools if tool.name == "submit_contract")
    initial = submit.input_schema["properties"]["runtime"]["properties"]["interface"]["properties"]["completed_tool_events"]
    repair = _RuntimeSubmission.model_json_schema()
    field = initial["properties"]["failure_events"]
    assert field["maxItems"] == 8
    assert field["items"] == CompletedToolFailureEvent.model_json_schema()
    assert "$defs" not in initial
    assert repair["$defs"]["CompletedToolEvents"]["properties"]["failure_events"]["items"] == {
        "$ref": "#/$defs/CompletedToolFailureEvent",
    }
    assert repair["$defs"]["CompletedToolFailureEvent"] == field["items"]
    assert field["items"]["required"] == ["kind", "action_id_path", "result_path"]


@pytest.mark.parametrize("argument", [
    "{{new_message}}",
    "{{new_message_content}}",
    "{{messages}}",
    "{{ new_message }}",
    "--query={{new_message_content}}",
])
def test_runtime_command_rejects_unexpanded_message_templates(argument):
    from fi.alk.harness.contract import Runtime

    with pytest.raises(ValueError, match="runtime_command_message_template_unsupported"):
        Runtime.model_validate({
            "execution": "native",
            "command": ["agent", "chat", "--query", argument],
            "interface": {"kind": "command", "protocol": "fi.alk"},
        })


@pytest.mark.parametrize("argument", [
    '{"message":"hello", "options":{}}',
    "{message}",
    "{{source_template}}",
    "{{PORT_agent}}",
    "{1..4}",
])
def test_runtime_command_preserves_static_braces(argument):
    from fi.alk.harness.contract import Runtime

    command = ["agent", "--format", argument]
    assert Runtime(command=command, interface={"kind": "command", "protocol": "fi.alk"}).command == command


def test_runtime_command_rejects_message_template_in_normalized_shell_string():
    from fi.alk.harness.contract import Runtime

    with pytest.raises(ValueError, match="runtime_command_message_template_unsupported"):
        Runtime(command="agent chat -q '{{new_message}}'", interface={"kind": "command", "protocol": "fi.alk"})


def test_http_server_keeps_source_owned_message_template_argument():
    from fi.alk.harness.contract import Runtime

    command = ["server", "--chat-template", "{{messages}}"]
    runtime = Runtime(command=command, interface={"kind": "http", "protocol": "openai_chat", "port": 8080, "path": "/v1/chat/completions"})
    assert runtime.command == command
