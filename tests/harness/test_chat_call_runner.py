from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fi.alk.harness import chat_call_runner as chat
from fi.alk.harness.call_runner import CallRunnerContext
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
    assert chat._chat_target_timeout_seconds() == 120.0
    monkeypatch.setenv("ALK_CHAT_TARGET_TIMEOUT_SECONDS", "0")
    assert chat._chat_target_timeout_seconds() == 120.0


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
    assert {"role": "assistant", "content": "Your account is active and the yield is 3.1%."} in (
        outcome.messages
    )
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
