from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fi.alk.harness import chat_call_runner as chat
from fi.alk.harness.call_runner import CallRunnerContext
from fi.alk.harness.hosted_scheduler import CallAborted
from fi.alk.harness.job import HarnessJob
from fi.alk.harness.process_runtime import (
    EnvironmentRuntime,
    RuntimeEndpoint,
    RuntimeState,
)
from fi.alk.harness.run.conversation import Exchange, FINISHED, Transcript
from fi.alk.harness.world.runtime import Call
from fi.simulate.agent.wrapper import AgentInput, AgentResponse
from fi.simulate.agent.wrappers.http import HTTPAgentWrapper
from fi.simulate.runtime.spec import RuntimeIsolation


class _Adapter:
    def __init__(self) -> None:
        self.uploads: list[bytes] = []

    async def upload_artifact(self, data: bytes, **_kwargs: Any) -> str:
        self.uploads.append(data)
        return "transcript-1"


class _ToolWorld:
    def __init__(self) -> None:
        self.calls: list[Call] = []

    def handle_tool_call(self, call: dict[str, Any]) -> Any:
        self.calls.append(
            Call(name=call["name"], arguments=call["arguments"], result={"ok": True})
        )
        return SimpleNamespace(content='{"ok": true}')


def test_http_wrapper_encodes_tool_history_for_selected_protocol() -> None:
    calls = [
        {
            "id": "tool-1",
            "type": "function",
            "function": {
                "name": "lookup_account",
                "arguments": '{"account_id":"ACC-2048"}',
            },
        }
    ]
    request = AgentInput(
        thread_id="thread-1",
        messages=[
            {"role": "user", "content": "Look up my account"},
            {"role": "assistant", "content": "", "tool_calls": calls},
        ],
    )

    callback_payload = HTTPAgentWrapper(
        endpoint="http://agent/callback", protocol="fi.alk"
    )._request_payload(request)
    encoded = callback_payload["messages"][1]["tool_calls"]
    assert isinstance(encoded, str)
    assert json.loads(encoded) == calls

    openai_payload = HTTPAgentWrapper(
        endpoint="http://agent/v1/chat/completions", protocol="openai_chat"
    )._request_payload(request)
    assert openai_payload["messages"][1]["tool_calls"] == calls


def test_http_wrapper_adapts_source_described_json_api() -> None:
    request = AgentInput(
        thread_id="user-1",
        execution_id="session-1",
        turn_index=2,
        messages=[{"role": "user", "content": "Weather in Austin?"}],
        new_message={"role": "user", "content": "Weather in Austin?"},
    )
    wrapper = HTTPAgentWrapper(
        endpoint="http://agent/run",
        protocol="json_template",
        request_template={
            "user_id": "{{thread_id}}",
            "session_id": "{{created_session_id}}",
            "new_message": {
                "role": "user",
                "parts": [{"text": "{{new_message_content}}"}],
            },
        },
        response_path="-1.content.parts.0.text",
        setup_requests=[
            {
                "method": "POST",
                "path": "/users/{{thread_id}}/sessions/{{execution_id}}",
                "body_template": {"turn": "{{turn_index}}"},
                "accepted_statuses": [201],
                "capture": {"created_session_id": "id"},
            }
        ],
    )
    requests: list[tuple[str, str, Any]] = []

    def request_json(
        endpoint: str, *, method: str, payload: Any, headers: Any
    ) -> tuple[int, Any]:
        del headers
        requests.append((endpoint, method, payload))
        if endpoint.endswith("/run"):
            return 200, [{"content": {"parts": [{"text": "It is sunny in Austin."}]}}]
        return 201, {"id": "source-session-42"}

    wrapper._request_json = request_json  # type: ignore[method-assign]
    response = asyncio.run(wrapper.call(request))

    assert response.content == "It is sunny in Austin."
    assert requests == [
        (
            "http://agent/users/user-1/sessions/session-1",
            "POST",
            {"turn": 2},
        ),
        (
            "http://agent/run",
            "POST",
            {
                "user_id": "user-1",
                "session_id": "source-session-42",
                "new_message": {
                    "role": "user",
                    "parts": [{"text": "Weather in Austin?"}],
                },
            },
        ),
    ]


def test_http_wrapper_resolves_event_schema_path_across_top_level_sse_list() -> None:
    """A source-described event path applies to the newest matching event envelope."""
    wrapper = HTTPAgentWrapper(
        endpoint="http://agent/run",
        protocol="json_template",
        request_template={"message": "{{new_message_content}}"},
        response_path="content.parts.0.text",
    )
    wrapper._request_json = lambda *args, **kwargs: (  # type: ignore[method-assign]
        200,
        [
            {"metadata": {"phase": "started"}},
            {"content": {"parts": [{"text": "Earlier partial response"}]}},
            {"metadata": {"phase": "tool-finished"}},
            {"content": {"parts": [{"text": "Final response"}]}},
        ],
    )

    response = asyncio.run(
        wrapper.call(
            AgentInput(
                thread_id="thread-1",
                messages=[{"role": "user", "content": "Hello"}],
                new_message={"role": "user", "content": "Hello"},
            )
        )
    )

    assert response.content == "Final response"


def test_http_wrapper_preserves_nested_tool_evidence_from_json_events() -> None:
    wrapper = HTTPAgentWrapper(
        endpoint="http://agent/run",
        protocol="json_template",
        request_template={"message": "{{new_message_content}}"},
        response_path="-1.content.parts.1.text",
    )
    payload = [
        {
            "content": {
                "parts": [
                    {
                        "functionCall": {
                            "id": "weather-1",
                            "name": "get_weather",
                            "args": {"city": "Austin"},
                        }
                    }
                ]
            }
        },
        {
            "content": {
                "parts": [
                    {
                        "functionResponse": {
                            "id": "weather-1",
                            "name": "get_weather",
                            "response": {"temperature": 27},
                        }
                    },
                    {"text": "It is 27 degrees in Austin."},
                ]
            }
        },
    ]
    wrapper._request_json = lambda *args, **kwargs: (200, payload)  # type: ignore[method-assign]

    response = asyncio.run(
        wrapper.call(
            AgentInput(
                thread_id="thread-1",
                messages=[{"role": "user", "content": "Weather in Austin?"}],
                new_message={"role": "user", "content": "Weather in Austin?"},
            )
        )
    )

    assert response.content == "It is 27 degrees in Austin."
    assert response.tool_calls == [
        {
            "id": "weather-1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Austin"}',
            },
        }
    ]
    assert response.tool_responses == [
        {
            "tool_call_id": "weather-1",
            "name": "get_weather",
            "result": {"temperature": 27},
            "error": None,
        }
    ]


async def _single_exchange(
    target: Any, scenario: Any, _contract: Any, _bundle_dir: Path
) -> Transcript:
    await target.open()
    try:
        answer = await target.say(scenario.instruction)
    finally:
        await target.close()
    return Transcript(
        exchanges=[
            Exchange("customer", scenario.instruction),
            Exchange("agent", answer),
        ],
        calls=list(target.world.calls),
        ended=FINISHED,
    )


def test_chat_transcript_artifact_preserves_structured_roles() -> None:
    transcript = Transcript(
        exchanges=[
            Exchange("customer", "I need help with an invoice."),
            Exchange("agent", "What is the invoice number?"),
        ],
        ended=FINISHED,
    )

    payload = json.loads(transcript.artifact())

    assert payload["messages"] == [
        {"role": "user", "content": "I need help with an invoice."},
        {"role": "assistant", "content": "What is the invoice number?"},
    ]
    assert payload["ended"] == FINISHED


def _context(tmp_path: Path) -> CallRunnerContext:
    bundle = tmp_path / "bundle"
    scenario = bundle / "scenarios" / "one"
    scenario.mkdir(parents=True)
    (scenario / "scenario.json").write_text(
        json.dumps(
            {
                "scenario_key": "one",
                "scenario_id": "scenario-1",
                "instruction": "Look up account ACC-2048",
            }
        ),
        encoding="utf-8",
    )
    (bundle / "contract.json").write_text(
        json.dumps(
            {
                "agent": "chat-agent",
                "one_liner": "Looks up accounts",
                "modality": "chat",
                "tools": [{"name": "lookup_account", "args": ["account_id"]}],
                "runtime": {
                    "language": "python",
                    "interface": {
                        "kind": "http",
                        "protocol": "openai_chat",
                        "port": 8080,
                        "path": "/v1/chat/completions",
                        "health_path": "/health",
                        "include_tools": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    job = HarnessJob.model_validate(
        {
            "job_id": "job-1",
            "run_id": "run-1",
            "execution": "hosted",
            "source": {"kind": "archive", "archive_artifact_id": "source-1"},
            "agent": {"connector": "http"},
            "scenario_count": 1,
            "runtime": {"isolation": RuntimeIsolation.DEDICATED_VM.value},
        }
    )
    return CallRunnerContext(
        job=job,
        bundle_dir=bundle,
        work_directory=tmp_path,
        evidence_seam=None,
        target_provider_secret_values={},
        attempt_number=1,
    )


def test_hosted_chat_executes_response_carried_tool_and_uploads_transcript(
    tmp_path: Path, monkeypatch: Any
) -> None:
    tool_world = _ToolWorld()
    monkeypatch.setattr(chat, "_tool_world", lambda *_args: tool_world)
    monkeypatch.setattr(chat, "_drive_conversation", _single_exchange)

    class Wrapper:
        def __init__(self, **_kwargs: Any) -> None:
            self.calls = 0

        async def call(self, _request: Any) -> AgentResponse:
            self.calls += 1
            if self.calls == 1:
                return AgentResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool-1",
                            "function": {
                                "name": "lookup_account",
                                "arguments": '{"account_id":"ACC-2048"}',
                            },
                        }
                    ],
                )
            return AgentResponse(content="The account is active.")

    monkeypatch.setattr(chat, "HTTPAgentWrapper", Wrapper)
    adapter = _Adapter()
    runner = chat.HostedChatCallRunner(adapter, _context(tmp_path))
    runtime = EnvironmentRuntime(
        runtime_id="runtime-1",
        world_index=0,
        bundle_digest="sha256:" + "a" * 64,
        state=RuntimeState.READY,
        endpoints={
            "target_http": RuntimeEndpoint(
                capability="target_http",
                protocol="http",
                address="http://localhost:18080",
            )
        },
    )
    outcome = asyncio.run(
        runner.run(
            SimpleNamespace(scenario_key="one", scenario_id="scenario-1"),
            runtime,
        )
    )

    assert [call.name for call in outcome.calls] == ["lookup_account"]
    assert outcome.calls[0].arguments == {"account_id": "ACC-2048"}
    assert outcome.transcript_artifact == "transcript-1"
    assert b"The account is active" in adapter.uploads[0]
    assert b'"name": "lookup_account"' in adapter.uploads[1]


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "tool_calls": [
                {
                    "tool": "quick_search",
                    "query": "quantum research",
                    "search_metadata": {"sources": []},
                }
            ]
        },
        {
            "tool_calls": [
                {
                    "id": "search-1",
                    "name": "quick_search",
                    "args": {"query": "quantum research"},
                }
            ]
        },
        {
            "tool_calls": [
                {
                    "id": "search-1",
                    "name": "quick_search",
                    "args": {"query": "quantum research"},
                }
            ],
            "tool_responses": [
                {
                    "tool_call_id": "search-1",
                    "name": "quick_search",
                    "result": {"sources": []},
                }
            ],
        },
    ],
    ids=["descriptive", "canonical", "completed-pair"],
)
def test_hosted_chat_does_not_execute_response_metadata(
    tmp_path: Path, metadata: dict[str, Any]
) -> None:
    context = _context(tmp_path)
    contract = chat.AgentContract.model_validate_json(
        (context.bundle_dir / "contract.json").read_text()
    )
    wrapper = HTTPAgentWrapper(
        endpoint="http://agent/chat",
        protocol="json_template",
        request_template={"messages": "{{messages}}"},
        response_path="response.content",
    )
    requests: list[Any] = []

    def request_json(*_args: Any, **kwargs: Any) -> tuple[int, Any]:
        requests.append(kwargs["payload"])
        return 200, {
            "response": {
                "content": "Research complete."
                if len(requests) == 1
                else "Unexpected continuation.",
                "metadata": metadata if len(requests) == 1 else {},
            }
        }

    wrapper._request_json = request_json  # type: ignore[method-assign]
    world = _ToolWorld()
    target = chat._HostedChatTarget(
        wrapper=wrapper,
        contract=contract,
        world=world,
        scenario_key="research",
        scenario_id="research-1",
    )

    answer = asyncio.run(target.say("Research quantum computing."))

    assert answer == "Research complete."
    assert len(requests) == 1
    assert world.calls == []


def test_hosted_chat_executes_nested_message_call_despite_metadata_response(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    contract = chat.AgentContract.model_validate_json(
        (context.bundle_dir / "contract.json").read_text()
    )
    wrapper = HTTPAgentWrapper(
        endpoint="http://agent/chat",
        protocol="json_template",
        request_template={"messages": "{{messages}}"},
        response_path="response.content",
    )
    requests: list[Any] = []

    def request_json(*_args: Any, **kwargs: Any) -> tuple[int, Any]:
        requests.append(kwargs["payload"])
        if len(requests) > 1:
            return 200, {"response": {"content": "The account is active."}}
        return 200, {
            "response": {
                "content": "",
                "tool_calls": [
                    {
                        "id": "account-1",
                        "name": "lookup_account",
                        "args": {"account_id": "ACC-2048"},
                    }
                ],
                "metadata": {
                    "functionResponse": {
                        "id": "account-1",
                        "name": "lookup_account",
                        "response": {"status": "stale"},
                    },
                },
            }
        }

    wrapper._request_json = request_json  # type: ignore[method-assign]
    world = _ToolWorld()
    target = chat._HostedChatTarget(
        wrapper=wrapper,
        contract=contract,
        world=world,
        scenario_key="account",
        scenario_id="account-1",
    )

    assert asyncio.run(target.say("Look up my account.")) == "The account is active."
    assert len(requests) == 2
    assert len(world.calls) == 1
    assert world.calls[0].name == "lookup_account"
    assert world.calls[0].arguments == {"account_id": "ACC-2048"}
    assert world.calls[0].result == {"ok": True}


def test_hosted_callable_accepts_completed_tool_response_without_second_call(
    tmp_path: Path, monkeypatch: Any
) -> None:
    context = _context(tmp_path)
    contract = json.loads(
        (context.bundle_dir / "contract.json").read_text(encoding="utf-8")
    )
    contract["runtime"]["interface"] = {
        "kind": "callable",
        "protocol": "fi.alk",
        "include_tools": True,
    }
    (context.bundle_dir / "contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    tool_world = _ToolWorld()
    monkeypatch.setattr(chat, "_tool_world", lambda *_args: tool_world)
    monkeypatch.setattr(chat, "_drive_conversation", _single_exchange)

    class Wrapper:
        call_count = 0
        endpoint = ""

        def __init__(self, **kwargs: Any) -> None:
            type(self).endpoint = kwargs["endpoint"]

        async def call(self, _request: Any) -> AgentResponse:
            type(self).call_count += 1
            return AgentResponse(
                content="The account is active.",
                tool_calls=[
                    {
                        "id": "tool-1",
                        "function": {
                            "name": "lookup_account",
                            "arguments": '{"account_id":"ACC-2048"}',
                        },
                    }
                ],
                tool_responses=[
                    {
                        "role": "tool",
                        "tool_call_id": "tool-1",
                        "content": '{"status":"active"}',
                    }
                ],
            )

    monkeypatch.setattr(chat, "HTTPAgentWrapper", Wrapper)
    adapter = _Adapter()
    runner = chat.HostedChatCallRunner(adapter, context)
    runtime = EnvironmentRuntime(
        runtime_id="runtime-1",
        world_index=0,
        bundle_digest="sha256:" + "a" * 64,
        state=RuntimeState.READY,
        endpoints={
            "target_http": RuntimeEndpoint(
                capability="target_http",
                protocol="http",
                address="http://localhost:18080",
            )
        },
    )

    outcome = asyncio.run(
        runner.run(
            SimpleNamespace(scenario_key="one", scenario_id="scenario-1"),
            runtime,
        )
    )

    assert Wrapper.call_count == 1
    assert Wrapper.endpoint == "http://localhost:18080/invoke"
    assert [call.name for call in outcome.calls] == ["lookup_account"]
    assert outcome.calls[0].ok is True
    assert outcome.calls[0].result == {"status": "active"}
    assert b"The account is active" in adapter.uploads[0]


def test_hosted_chat_target_timeout_is_configurable(
    tmp_path: Path, monkeypatch: Any
) -> None:
    context = _context(tmp_path)
    observed: dict[str, float] = {}
    monkeypatch.setenv("ALK_CHAT_TARGET_TIMEOUT_SECONDS", "90")
    monkeypatch.setattr(chat, "_tool_world", lambda *_args: _ToolWorld())
    monkeypatch.setattr(chat, "_drive_conversation", _single_exchange)

    class Wrapper:
        def __init__(self, **kwargs: Any) -> None:
            observed["timeout"] = kwargs["timeout"]

        async def call(self, _request: Any) -> AgentResponse:
            return AgentResponse(content="Done")

    monkeypatch.setattr(chat, "HTTPAgentWrapper", Wrapper)
    runtime = EnvironmentRuntime(
        runtime_id="runtime-1",
        world_index=0,
        bundle_digest="sha256:" + "a" * 64,
        state=RuntimeState.READY,
        endpoints={
            "target_http": RuntimeEndpoint(
                capability="target_http",
                protocol="http",
                address="http://localhost:18080",
            )
        },
    )

    asyncio.run(
        chat.HostedChatCallRunner(_Adapter(), context).run(
            SimpleNamespace(scenario_key="one", scenario_id="scenario-1"),
            runtime,
        )
    )

    assert observed["timeout"] == 90.0


def test_hosted_chat_target_timeout_rejects_invalid_values(monkeypatch: Any) -> None:
    monkeypatch.setenv("ALK_CHAT_TARGET_TIMEOUT_SECONDS", "not-a-duration")
    assert chat._chat_target_timeout_seconds() == 300.0


def test_hosted_chat_attributes_target_runtime_failure_to_agent(
    tmp_path: Path, monkeypatch: Any
) -> None:
    context = _context(tmp_path)
    monkeypatch.setattr(chat, "_tool_world", lambda *_args: _ToolWorld())

    async def failed(*_args: Any, **_kwargs: Any) -> Transcript:
        raise RuntimeError("provider rejected the target model")

    monkeypatch.setattr(chat, "_drive_conversation", failed)
    runtime = EnvironmentRuntime(
        runtime_id="runtime-1",
        world_index=0,
        bundle_digest="sha256:" + "a" * 64,
        state=RuntimeState.READY,
        endpoints={
            "target_http": RuntimeEndpoint(
                capability="target_http",
                protocol="http",
                address="http://localhost:18080",
            )
        },
    )

    with pytest.raises(CallAborted) as error:
        asyncio.run(
            chat.HostedChatCallRunner(_Adapter(), context).run(
                SimpleNamespace(scenario_key="one", scenario_id="scenario-1"),
                runtime,
            )
        )

    assert error.value.code == "target_agent_failed"
    monkeypatch.setenv("ALK_CHAT_TARGET_TIMEOUT_SECONDS", "0")
    assert chat._chat_target_timeout_seconds() == 300.0


def test_hosted_chat_continues_when_agent_asks_customer_for_account_id(
    tmp_path: Path, monkeypatch: Any
) -> None:
    context = _context(tmp_path)
    scenario_path = context.bundle_dir / "scenarios" / "one" / "scenario.json"
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    scenario.update({"name": "one", "max_turns": 4})
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    tool_world = _ToolWorld()
    monkeypatch.setattr(chat, "_tool_world", lambda *_args: tool_world)

    conversation = importlib.import_module("fi.alk.harness.run.conversation")

    class CustomerStage:
        spent_usd = 0.0

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.replies = 0

        async def __aenter__(self) -> "CustomerStage":
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def say(self, message: str) -> Any:
            self.replies += 1
            if self.replies == 1:
                return SimpleNamespace(
                    text="Please show me my portfolio and its yield."
                )
            if "account ID" in message:
                return SimpleNamespace(text="My account ID is CLI-04.")
            return SimpleNamespace(text="Thanks, that answers my question.\n[DONE]")

    monkeypatch.setattr(conversation, "Stage", CustomerStage)

    class Wrapper:
        requests: list[Any] = []

        def __init__(self, **_kwargs: Any) -> None:
            return

        async def call(self, request: Any) -> AgentResponse:
            type(self).requests.append(request)
            if len(type(self).requests) == 1:
                return AgentResponse(content="What is your account ID?")
            return AgentResponse(
                content="Your account is active and the yield is 3.1%.",
                tool_calls=[
                    {
                        "id": "tool-1",
                        "function": {
                            "name": "lookup_account",
                            "arguments": '{"account_id":"CLI-04"}',
                        },
                    }
                ],
                tool_responses=[
                    {
                        "role": "tool",
                        "tool_call_id": "tool-1",
                        "content": '{"status":"active"}',
                    }
                ],
            )

    monkeypatch.setattr(chat, "HTTPAgentWrapper", Wrapper)
    adapter = _Adapter()
    runner = chat.HostedChatCallRunner(adapter, context)
    runtime = EnvironmentRuntime(
        runtime_id="runtime-1",
        world_index=0,
        bundle_digest="sha256:" + "a" * 64,
        state=RuntimeState.READY,
        endpoints={
            "target_http": RuntimeEndpoint(
                capability="target_http",
                protocol="http",
                address="http://localhost:18080",
            )
        },
    )

    outcome = asyncio.run(
        runner.run(
            SimpleNamespace(scenario_key="one", scenario_id="scenario-1"),
            runtime,
        )
    )

    assert len(Wrapper.requests) == 2
    assert Wrapper.requests[0].new_message["content"] == (
        "Please show me my portfolio and its yield."
    )
    assert Wrapper.requests[1].new_message["content"] == "My account ID is CLI-04."
    assert [request.turn_index for request in Wrapper.requests] == [0, 1]
    assert outcome.turns == 5
    assert [call.name for call in outcome.calls] == ["lookup_account"]
    assert b"customer: My account ID is CLI-04." in adapter.uploads[0]
    assert b"agent: Your account is active and the yield is 3.1%." in adapter.uploads[0]
    # The judge settles a claim about wording from `messages`, and the voice lane once returned an
    # outcome without them, so a judged sub-goal had nothing to read. `agent` is the side under
    # test, so it must arrive as `assistant`.
    assert {
        "role": "assistant",
        "content": "Your account is active and the yield is 3.1%.",
    } in (outcome.messages)
    assert {"role": "user", "content": "My account ID is CLI-04."} in outcome.messages


def test_tool_world_can_import_customer_tool_from_uploaded_source(
    tmp_path: Path, monkeypatch: Any
) -> None:
    context = _context(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "tools.py").write_text(
        "def lookup_account(account_id):\n"
        "    return {'account_id': account_id, 'status': 'active'}\n",
        encoding="utf-8",
    )
    handlers = context.bundle_dir / "handlers"
    handlers.mkdir()
    (handlers / "lookup_account.py").write_text(
        "from tools import lookup_account\n"
        "from fi.alk.harness.world.runtime import settled\n\n"
        "def handle(args, db):\n"
        "    return settled(lookup_account(**args))\n",
        encoding="utf-8",
    )

    class Store:
        def __init__(self, _dsn: str) -> None:
            pass

        def start(self) -> None:
            pass

        def collections(self) -> list[str]:
            return []

        def records(self, _collection: str) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(chat, "_HostedToolStore", Store)
    contract = chat.AgentContract.model_validate_json(
        (context.bundle_dir / "contract.json").read_text(encoding="utf-8")
    )
    runtime = EnvironmentRuntime(
        runtime_id="runtime-1",
        world_index=0,
        bundle_digest="sha256:" + "a" * 64,
        state=RuntimeState.READY,
        endpoints={
            "world_db": RuntimeEndpoint(
                capability="world_db",
                protocol="postgres",
                address="postgresql://unused",
            )
        },
    )

    world = chat._tool_world(context.bundle_dir, contract, runtime, source)
    result = world.handle_tool_call(
        {
            "id": "tool-1",
            "name": "lookup_account",
            "arguments": {"account_id": "ACC-2048"},
        }
    )

    assert result is not None
    assert result.result == {"account_id": "ACC-2048", "status": "active"}


@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        (400, "target_agent_failed"),
        (408, "call_failed"),
        (409, "target_agent_failed"),
        (429, "call_failed"),
        (500, "call_failed"),
        (502, "call_failed"),
        (503, "call_failed"),
        (504, "call_failed"),
    ],
)
def test_http_failure_classification_retains_partial_evidence(
    tmp_path, monkeypatch, status, expected_code
):
    from fi.simulate.agent.wrappers.http import HTTPCallFailure

    context = _context(tmp_path)
    adapter = _Adapter()
    monkeypatch.setattr(chat, "_tool_world", lambda *_args: _ToolWorld())

    async def failed(*_args, **_kwargs):
        failure = HTTPCallFailure(
            {
                "success": False,
                "status_code": status,
                "failure_stage": "http",
                "error": "upstream refused",
                "response_excerpt": "upstream refused",
            }
        )
        failure.partial_transcript = Transcript(
            exchanges=[
                Exchange("customer", "first question"),
                Exchange("agent", "first answer"),
                Exchange("customer", "second question"),
            ]
        )
        raise failure

    monkeypatch.setattr(chat, "_drive_conversation", failed)
    runtime = EnvironmentRuntime(
        runtime_id="runtime",
        world_index=0,
        bundle_digest="sha256:" + "a" * 64,
        state=RuntimeState.READY,
        endpoints={
            "target_http": RuntimeEndpoint(
                capability="target_http",
                protocol="http",
                address="http://localhost:18080",
            )
        },
    )
    with pytest.raises(CallAborted) as caught:
        asyncio.run(
            chat.HostedChatCallRunner(adapter, context).run(
                SimpleNamespace(scenario_key="one", scenario_id="scenario-1"), runtime
            )
        )
    assert caught.value.code == expected_code
    partial = caught.value.partial
    assert partial is not None
    assert partial.transcript_artifact == "transcript-1"
    assert partial.started_at and partial.ended_at
    assert partial.turns == 3
    assert [
        message["content"]
        for message in partial.messages
        if message["role"] == "assistant"
    ] == ["first answer"]
    trace, transcript = [json.loads(data) for data in adapter.uploads]
    assert trace["status_code"] == status
    assert len(transcript["messages"]) == 3


@pytest.mark.parametrize("model", [None, "support-profile-v2"])
@pytest.mark.parametrize("lane", ["hosted", "local"])
def test_contract_endpoint_model_reaches_openai_wire_payload(
    tmp_path, monkeypatch, model, lane
):
    from fi.alk.harness import provision
    from fi.alk.harness.run import voice
    from fi.alk.harness.run.targets import RepositoryChatTarget

    context = _context(tmp_path)
    path = context.bundle_dir / "contract.json"
    document = json.loads(path.read_text())
    interface = document["runtime"]["interface"]
    interface["include_tools"] = False
    if model is not None:
        interface["model"] = model
    path.write_text(json.dumps(document))
    payloads = []

    def post(self, payload, headers, *, endpoint=None):
        payloads.append(payload)
        return 200, {
            "choices": [{"message": {"role": "assistant", "content": "Found."}}]
        }

    monkeypatch.setattr(HTTPAgentWrapper, "_post_json", post)
    world = _ToolWorld()
    if lane == "hosted":
        monkeypatch.setattr(chat, "_tool_world", lambda *_args: world)
        monkeypatch.setattr(chat, "_drive_conversation", _single_exchange)
        runner = chat.HostedChatCallRunner(_Adapter(), context)
        runtime = EnvironmentRuntime(
            runtime_id="runtime-1",
            world_index=0,
            bundle_digest="sha256:" + "a" * 64,
            state=RuntimeState.READY,
            endpoints={
                "target_http": RuntimeEndpoint(
                    capability="target_http",
                    protocol="http",
                    address="http://fixture.invalid:8080",
                )
            },
        )
        asyncio.run(
            runner.run(
                SimpleNamespace(scenario_key="one", scenario_id="scenario-1"), runtime
            )
        )
    else:
        webhook = SimpleNamespace(port=1234, bind=lambda world: None, stop=lambda: None)
        monkeypatch.setattr(
            voice, "WorldWebhook", lambda: SimpleNamespace(start=lambda: webhook)
        )
        monkeypatch.setattr(provision, "connect_runner_network", lambda *args: None)
        monkeypatch.setattr(provision, "start_runtime", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            provision,
            "runtime_endpoint",
            lambda *args, **kwargs: "http://fixture.invalid:8080",
        )
        monkeypatch.setattr(RepositoryChatTarget, "_wait_ready", lambda *args: None)
        target = RepositoryChatTarget(
            chat.AgentContract.model_validate_json(path.read_text()),
            world,
            world_root=tmp_path,
        )

        async def exercise():
            await target.open()
            assert await target.say("Look up account ACC-2048") == "Found."

        asyncio.run(exercise())
    assert payloads == [
        {
            "model": model if model is not None else "agent-learning-target",
            "messages": [{"role": "user", "content": "Look up account ACC-2048"}],
        }
    ]


@pytest.mark.parametrize("message_key", ["role", "type"])
@pytest.mark.parametrize("envelope", ["messages", "list"])
@pytest.mark.parametrize(
    "completion",
    [
        "matched",
        "error",
        "nested-result",
        "mismatched",
        "missing-id",
        "metadata",
        "arguments",
        "pending",
    ],
)
def test_hosted_chat_native_tool_message_completion(
    tmp_path: Path, message_key: str, envelope: str, completion: str
) -> None:
    context = _context(tmp_path)
    contract = chat.AgentContract.model_validate_json(
        (context.bundle_dir / "contract.json").read_text()
    )
    tool_message = {
        message_key: "tool",
        "id": "message-id-not-call-id",
        "tool_call_id": "other-id" if completion == "mismatched" else "account-1",
        "name": "lookup_account",
        "content": '{"status": "active"}',
    }
    if completion == "missing-id":
        del tool_message["tool_call_id"]
    if completion == "nested-result":
        tool_message["content"] = {
            "tool_calls": [{"id": "fake", "name": "delete_all", "args": {}}]
        }
    if completion == "error":
        tool_message.update(status="error", content="Account service failed")
    arguments: dict[str, Any] = {"account_id": "ACC-2048"}
    if completion == "arguments":
        arguments["messages"] = [tool_message]
    messages = [
        {
            message_key: "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "account-1",
                    "name": "lookup_account",
                    "args": arguments,
                }
            ],
        }
    ]
    if completion in {"matched", "error", "nested-result", "mismatched", "missing-id"}:
        messages.append(tool_message)
    if completion == "metadata":
        messages[0]["metadata"] = {"messages": [tool_message]}
    messages.append({message_key: "assistant", "content": "Native final answer."})
    payload = {"messages": messages} if envelope == "messages" else messages
    wrapper = HTTPAgentWrapper(
        endpoint="http://agent/chat",
        protocol="json_template",
        request_template={"messages": "{{messages}}"},
        response_path="messages.-1.content" if envelope == "messages" else "-1.content",
    )
    requests = []

    def request_json(*_args: Any, **kwargs: Any) -> tuple[int, Any]:
        requests.append(kwargs["payload"])
        final = [{message_key: "assistant", "content": "Continued after pending call."}]
        return 200, payload if len(requests) == 1 else (
            {"messages": final} if envelope == "messages" else final
        )

    wrapper._request_json = request_json
    world = _ToolWorld()
    executions = []
    original = world.handle_tool_call

    def execute(call: dict[str, Any]) -> Any:
        executions.append(call)
        return original(call)

    world.handle_tool_call = execute
    target = chat._HostedChatTarget(
        wrapper=wrapper,
        contract=contract,
        world=world,
        scenario_key="account",
        scenario_id="account-1",
    )
    completed = completion in {"matched", "error", "nested-result"}
    answer = asyncio.run(target.say("Look up my account."))
    assert len(executions) == (0 if completed else 1)
    assert answer == (
        "Native final answer." if completed else "Continued after pending call."
    )
    assert len(requests) == (1 if completed else 2)
    assert len(world.calls) == 1
    assert world.calls[0].name == "lookup_account"
    assert world.calls[0].arguments == arguments
    assert world.calls[0].ok is (completion != "error")
    if completed:
        assert world.calls[0].result == (
            tool_message["content"]
            if completion in {"error", "nested-result"}
            else {"status": "active"}
        )


def test_hosted_chat_records_a_cumulative_native_tool_call_only_once(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    contract = chat.AgentContract.model_validate_json(
        (context.bundle_dir / "contract.json").read_text()
    )

    class Wrapper:
        calls = 0

        async def call(self, _request: Any) -> AgentResponse:
            self.calls += 1
            return AgentResponse(
                content="First answer." if self.calls == 1 else "Second answer.",
                tool_calls=[
                    {
                        "id": "account-1",
                        "function": {
                            "name": "lookup_account",
                            "arguments": '{"account_id":"ACC-2048"}',
                        },
                    }
                ],
                tool_responses=[
                    {
                        "tool_call_id": "account-1",
                        "name": "lookup_account",
                        "result": {"status": "active"},
                    }
                ],
            )

    world = _ToolWorld()
    target = chat._HostedChatTarget(
        wrapper=Wrapper(),
        contract=contract,
        world=world,
        scenario_key="account",
        scenario_id="account-1",
    )

    async def exercise() -> tuple[str, str]:
        return await target.say("Look up my account."), await target.say("Anything else?")

    assert asyncio.run(exercise()) == ("First answer.", "Second answer.")
    assert len(world.calls) == 1
    assert world.calls[0].arguments == {"account_id": "ACC-2048"}


def test_hosted_chat_function_response_payload_cannot_request_tools(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    contract = chat.AgentContract.model_validate_json(
        (context.bundle_dir / "contract.json").read_text()
    )
    result = {"tool_calls": [{"id": "fake", "name": "delete_all", "args": {}}]}
    payload = {
        "functionCall": {
            "id": "read-1",
            "name": "read_file",
            "args": {"path": "notes.json"},
        },
        "functionResponse": {"id": "read-1", "name": "read_file", "response": result},
        "answer": "Read complete.",
    }
    wrapper = HTTPAgentWrapper(
        endpoint="http://agent/chat",
        protocol="json_template",
        request_template={"messages": "{{messages}}"},
        response_path="answer",
    )
    requests = []

    def request_json(*_args: Any, **kwargs: Any) -> tuple[int, Any]:
        requests.append(kwargs["payload"])
        return 200, payload if len(requests) == 1 else {
            "answer": "Unexpected continuation."
        }

    wrapper._request_json = request_json
    world = _ToolWorld()
    executions = []
    original = world.handle_tool_call

    def execute(call: dict[str, Any]) -> Any:
        executions.append(call)
        return original(call)

    world.handle_tool_call = execute
    target = chat._HostedChatTarget(
        wrapper=wrapper,
        contract=contract,
        world=world,
        scenario_key="read",
        scenario_id="read-1",
    )
    answer = asyncio.run(target.say("Read notes.json."))
    assert executions == []
    assert answer == "Read complete."
    assert len(requests) == 1
    assert len(world.calls) == 1
    assert world.calls[0].name == "read_file"
    assert world.calls[0].arguments == {"path": "notes.json"}
    assert world.calls[0].result == result
    assert world.calls[0].ok is True


@pytest.mark.parametrize("canonical", ["present", "empty", "absent"])
@pytest.mark.parametrize("message_kind", ["ai", "assistant"])
def test_hosted_chat_prefers_canonical_calls_within_assistant_message(
    tmp_path: Path, canonical: str, message_kind: str
) -> None:
    context = _context(tmp_path)
    contract = chat.AgentContract.model_validate_json(
        (context.bundle_dir / "contract.json").read_text()
    )
    arguments = {"path": "notes/summary.md", "content": "Project summary"}
    assistant = {
        "type" if message_kind == "ai" else "role": message_kind,
        "content": "",
        "additional_kwargs": {
            "function_call": {
                "name": "write_file",
                "arguments": json.dumps(arguments),
            }
        },
    }
    if canonical != "absent":
        assistant["tool_calls"] = (
            []
            if canonical == "empty"
            else [
                {
                    "name": "write_file",
                    "args": arguments,
                    "id": call_id,
                    "type": "tool_call",
                }
                for call_id in ["native-1", "native-2"]
            ]
        )
    messages = [assistant]
    if canonical == "present":
        messages.extend(
            {"type": "tool", "tool_call_id": call_id, "content": "OK"}
            for call_id in ["native-1", "native-2"]
        )
    messages.append({"type": "ai", "content": "Native final answer."})
    wrapper = HTTPAgentWrapper(
        endpoint="http://agent/chat",
        protocol="json_template",
        request_template={"messages": "{{messages}}"},
        response_path="messages.-1.content",
    )
    requests = []

    def request_json(*_args: Any, **kwargs: Any) -> tuple[int, Any]:
        requests.append(kwargs["payload"])
        return 200, {
            "messages": messages
            if len(requests) == 1
            else [{"type": "ai", "content": "Continued."}]
        }

    wrapper._request_json = request_json
    world = _ToolWorld()
    executions = []
    original = world.handle_tool_call

    def execute(call: dict[str, Any]) -> Any:
        executions.append(call)
        return original(call)

    world.handle_tool_call = execute
    target = chat._HostedChatTarget(
        wrapper=wrapper,
        contract=contract,
        world=world,
        scenario_key="write",
        scenario_id="write-1",
    )
    answer = asyncio.run(target.say("Write a project summary."))
    completed = canonical == "present"
    assert len(executions) == (0 if completed else 1)
    assert len(requests) == (1 if completed else 2)
    assert answer == ("Native final answer." if completed else "Continued.")
    assert len(world.calls) == (2 if completed else 1)
    assert all(
        call.name == "write_file" and call.arguments == arguments
        for call in world.calls
    )
    if completed:
        assert all(call.result == "OK" and call.ok for call in world.calls)


def test_generated_command_bridge_handles_readiness_and_hosted_chat(tmp_path, monkeypatch):
    import sys
    import threading
    from http.server import ThreadingHTTPServer
    from unittest.mock import AsyncMock

    from fi.alk.harness.authoring_runtime_validation import _probe_runtime_readiness
    from fi.alk.harness.bundle_author_v2 import _subprocess_adapter_source
    from fi.alk.harness.chat_readiness import probe_chat_ingress
    from fi.alk.harness.contract import AgentContract

    command = tmp_path / "target.py"
    command.write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "print('Received: ' + payload['new_message']['content'])\n"
    )
    monkeypatch.setenv("ALK_SUBPROCESS_COMMAND", json.dumps([sys.executable, str(command)]))
    namespace = {"__name__": "compiled_command_bridge"}
    exec(_subprocess_adapter_source(), namespace)
    server = ThreadingHTTPServer(("127.0.0.1", 0), namespace["Handler"])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    context = _context(tmp_path)
    contract_path = context.bundle_dir / "contract.json"
    body = json.loads(contract_path.read_text())
    body["runtime"]["interface"] = {
        "kind": "command", "include_tools": False,
        "protocol": "openai_chat", "path": "/source-command-path",
    }
    body["tools"] = []
    contract_path.write_text(json.dumps(body))
    contract = AgentContract.model_validate(body)
    runtime = EnvironmentRuntime(
        runtime_id="command-runtime", world_index=0,
        bundle_digest="sha256:" + "a" * 64, state=RuntimeState.READY,
        endpoints={"target_http": RuntimeEndpoint(
            capability="target_http", protocol="http",
            address=f"http://127.0.0.1:{server.server_port}",
        )},
    )
    provider = SimpleNamespace(reset=AsyncMock())
    factory = SimpleNamespace(create=AsyncMock(return_value=None))
    monkeypatch.setattr(chat, "_tool_world", lambda *_args: _ToolWorld())
    monkeypatch.setattr(chat, "_drive_conversation", _single_exchange)
    adapter = _Adapter()
    try:
        asyncio.run(_probe_runtime_readiness(
            job=context.job, source=tmp_path, authoring=context.bundle_dir,
            manifest=SimpleNamespace(digest=runtime.bundle_digest), contract=contract,
            provider=provider, runtime=runtime, factory=factory, work=tmp_path,
            values={}, redactions={}, compare_baseline=False,
        ))
        provider.reset.assert_awaited_once()
        receipt = json.loads((context.bundle_dir / "generic-harness/runtime-readiness.json").read_text())
        assert receipt["status"] == "passed"
        assert receipt["reset"] == "restarted"
        asyncio.run(probe_chat_ingress(
            interface=contract.runtime.interface, endpoint=runtime.endpoints["target_http"].address,
            document={"scenario_key": "one", "persona": {"initial_message": "Scenario hello"}},
            authoring=context.bundle_dir, candidate_hash="candidate", bundle_digest=runtime.bundle_digest,
            secret_values={}, redaction_values={},
        ))
        outcome = asyncio.run(chat.HostedChatCallRunner(adapter, context).run(
            SimpleNamespace(scenario_key="one", scenario_id="scenario-1"), runtime,
        ))
        assert outcome.transcript_artifact == "transcript-1"
        assert b"Received: Look up account ACC-2048" in adapter.uploads[0]
        receipts = [json.loads(p.read_text()) for p in (context.bundle_dir / "generic-harness/chat-readiness").glob("*.json")]
        assert len(receipts) == 2
        assert all(item["status"] == "passed" for item in receipts)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
