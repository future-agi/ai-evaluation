from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from fi.alk.harness.backends import SessionSpec, ToolServer, ToolSpec
from fi.alk.harness.chat_policy import (
    ChatPolicyError,
    apply_chat_policy,
    load_chat_policy,
)
from fi.alk.harness.hosted_chat_entrypoint import HostedChatRuntime
from fi.alk.harness.hosted_conversation import (
    ConversationCapabilities,
    ConversationClient,
)
from fi.alk.harness.session import Event as RuntimeEvent


def _capabilities() -> ConversationCapabilities:
    conversation_id = "4f1ddaa9-569e-4ddd-bef0-74926bb42b16"
    prefix = (
        f"https://platform.example/simulate/api/harness/conversations/{conversation_id}"
    )
    return ConversationCapabilities.model_validate(
        {
            "schema_version": "futureagi.harness-conversation.v1",
            "conversation_id": conversation_id,
            "job_id": "job-1",
            "token": "token",
            "fence": "fence",
            "expires_at": (
                datetime.now(timezone.utc) + timedelta(minutes=10)
            ).isoformat(),
            "identity": {
                "app_name": "alk-harness",
                "user_id": "conversation-user",
            },
            "turn_context": {
                "conversation_id": conversation_id,
                "stage": "run",
                "event_watermark": 0,
                "command_watermark": 0,
            },
            "endpoints": {
                "commands": f"{prefix}/commands/",
                "events": f"{prefix}/events/",
                "rerun": f"{prefix}/rerun/",
                "adjust": f"{prefix}/adjust/",
                "run_status": f"{prefix}/run-status/",
                "workspace": f"{prefix}/workspace/",
                "session_store": f"{prefix}/session-store/",
                "session_store_append": f"{prefix}/session-store/append/",
            },
        }
    )


def test_conversation_turn_does_not_require_undeclared_history():
    from fi.alk.harness.chat import Conversation

    class StageStub:
        spent_usd = 0.25

        async def say(self, _message, on_event=None):
            del on_event

        async def __aexit__(self, *_args):
            return None

    conversation = Conversation(stage=StageStub(), stage_name="run")

    asyncio.run(conversation.say("what is happening?"))
    asyncio.run(conversation.close())
    assert conversation.spent_usd == 0.25


def test_conversation_client_starts_from_durable_command_watermark():
    capabilities = _capabilities()
    capabilities.turn_context["command_watermark"] = 7

    client = ConversationClient(capabilities)

    assert client.command_watermark == 7


def test_conversation_client_bounds_long_provider_call_ids():
    capabilities = _capabilities()

    class Transport:
        def __init__(self):
            self.body = None

        def request(self, _method, _url, *, headers, json_body=None, timeout):
            del headers, timeout
            self.body = json_body
            return type(
                "Response",
                (),
                {
                    "status_code": 200,
                    "body": {"acked_through_sequence": 1},
                },
            )()

    transport = Transport()
    client = ConversationClient(capabilities, transport=transport)

    asyncio.run(
        client.emit(
            "tool_started",
            function_call_id="provider-signature-" + ("x" * 500),
        )
    )

    call_id = transport.body["events"][0]["function_call_id"]
    assert len(call_id) <= 255
    assert call_id.startswith("call-")


def test_authoring_activity_pump_does_not_replay_existing_lines(tmp_path):
    events_path = tmp_path / "harness-events.jsonl"
    events_path.write_text(
        json.dumps(
            {
                "event_type": "harness.stage.done",
                "payload": {"stage": "scenarios", "status": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class ActivityClient:
        def __init__(self):
            self.events = []

        async def emit(self, kind, **kwargs):
            self.events.append((kind, kwargs))

    runtime = object.__new__(HostedChatRuntime)
    runtime.workspace = tmp_path
    runtime.client = ActivityClient()
    runtime.stopping = asyncio.Event()

    async def exercise():
        task = asyncio.create_task(runtime._pump_authoring_activity())
        await asyncio.sleep(0)
        assert runtime.client.events == []
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "event_type": "harness.stage.done",
                        "payload": {"stage": "run", "status": 0},
                    }
                )
                + "\n"
            )
        for _ in range(20):
            if runtime.client.events:
                break
            await asyncio.sleep(0.05)
        runtime.stopping.set()
        await task

    asyncio.run(exercise())
    assert len(runtime.client.events) == 1
    assert runtime.client.events[0][0] == "authoring_activity"


def test_chat_policy_gates_stages_without_redeclaring_stage_tools(tmp_path):
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        """schema_version: 1
harness_chat:
  enabled: true
  stages:
    run:
      enabled: true
  lifecycle:
    request_user_input: true
    interrupt_response: true
""",
        encoding="utf-8",
    )
    policy = load_chat_policy(policy_path)

    async def handler(_args):
        return {"content": []}

    spec = SessionSpec(
        system_prompt="run",
        servers={
            "run": ToolServer(
                name="run",
                tools=[
                    ToolSpec("read_results", "read", {}, handler),
                    ToolSpec("run_simulation", "run", {}, handler),
                ],
            ),
            "flow": ToolServer(
                name="flow",
                tools=[ToolSpec("route_to_stage", "route", {}, handler)],
            ),
        },
        builtins=("Read", "AskUserQuestion"),
        ask=handler,
    )
    filtered = apply_chat_policy(spec, stage="run", policy=policy)
    assert filtered.granted() == [
        "Read",
        "AskUserQuestion",
        "mcp__run__read_results",
        "mcp__run__run_simulation",
        "mcp__flow__route_to_stage",
    ]


def test_active_authoring_chat_answers_read_only_and_requests_adjustments(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ALK_HARNESS", "claude")
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        """schema_version: 1
harness_chat:
  enabled: true
  stages:
    understand:
      enabled: true
  lifecycle:
    request_user_input: true
""",
        encoding="utf-8",
    )

    class ControlClient:
        def __init__(self):
            self.capabilities = _capabilities()
            self.turn_context = {
                **self.capabilities.turn_context,
                "capabilities": {"control_only": True},
            }
            self.adjustments = []

        async def adjust(self, instruction):
            self.adjustments.append(instruction)
            return {"status": "pending", "instruction": instruction}

        async def run_status(self):
            return {"state": "running", "stage": "generating_scenarios"}

    async def mutation_tool(_args):
        return {"content": []}

    client = ControlClient()
    runtime = HostedChatRuntime(
        client=client,  # type: ignore[arg-type]
        policy=load_chat_policy(policy_path),
        workspace=tmp_path,
        source=tmp_path,
        job={"metadata": {"agent_name": "demo"}, "scenario_count": 1},
    )
    spec = SessionSpec(
        system_prompt="understand",
        servers={
            "mutations": ToolServer(
                name="mutations",
                tools=[ToolSpec("save_contract", "write", {}, mutation_tool)],
            )
        },
        builtins=("Read", "Glob", "Grep", "Write", "Bash"),
    )

    configured = runtime._configure_stage("understand", spec)

    assert set(configured.granted()) == {
        "Read",
        "Glob",
        "Grep",
        "AskUserQuestion",
        "mcp__platform-control__get_run_status",
        "mcp__platform-control__request_adjustment",
    }
    adjustment_tool = next(
        tool
        for tool in configured.servers["platform-control"].tools
        if tool.name == "request_adjustment"
    )
    result = asyncio.run(
        adjustment_tool.handler({"instruction": "Add a failed payment scenario"})
    )
    assert client.adjustments == ["Add a failed payment scenario"]
    assert '"status": "pending"' in result["content"][0]["text"]


def test_chat_policy_rejects_unknown_stages(tmp_path):
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        """schema_version: 1
harness_chat:
  enabled: true
  stages:
    invent:
      enabled: true
""",
        encoding="utf-8",
    )
    with pytest.raises(ChatPolicyError, match="unknown hosted chat stages"):
        load_chat_policy(policy_path)


class QuestionClient:
    def __init__(self) -> None:
        self.capabilities = _capabilities()
        self.turn_context = self.capabilities.turn_context
        self.command_watermark = 0
        self.events: list[dict[str, Any]] = []

    async def emit(self, kind: str, **kwargs):
        self.events.append({"kind": kind, **kwargs})
        return self.events[-1]

    async def commands(self):
        question = next(
            event for event in self.events if event["kind"] == "question_requested"
        )
        return [
            {
                "sequence": 2,
                "message_id": "answer-1",
                "kind": "user_response",
                "payload": {
                    "content": "Use app/api.py",
                    "reply_to": question["message_id"],
                },
            }
        ]

    async def checkpoint(self, workspace: Path):
        del workspace
        return {"digest": "sha256:" + "0" * 64, "size": 0}


def test_blocking_question_waits_for_matching_conversation_response(tmp_path):
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        """schema_version: 1
harness_chat:
  enabled: true
  stages:
    reception:
      enabled: true
  lifecycle:
    request_user_input: true
    interrupt_response: true
""",
        encoding="utf-8",
    )
    client = QuestionClient()
    runtime = HostedChatRuntime(
        client=client,  # type: ignore[arg-type]
        policy=load_chat_policy(policy_path),
        workspace=tmp_path,
        source=tmp_path,
        job={"metadata": {"agent_name": "demo"}, "scenario_count": 1},
    )
    result = asyncio.run(
        runtime._ask(
            "AskUserQuestion",
            {"question": "Which entrypoint?", "options": ["app/api.py", "worker.py"]},
            None,
        )
    )
    assert result == {
        "content": [{"type": "text", "text": "Use app/api.py"}],
        "is_error": False,
    }
    question = client.events[0]
    assert question["kind"] == "question_requested"
    assert question["payload"]["options"] == ["app/api.py", "worker.py"]


def test_runtime_reuses_warm_state_and_does_not_reexecute_cold_replays(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ALK_HARNESS", "vertex-gemini")
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        """schema_version: 1
harness_chat:
  enabled: true
  stages:
    reception:
      enabled: true
""",
        encoding="utf-8",
    )

    class TurnClient:
        def __init__(self):
            self.capabilities = _capabilities()
            self.turn_context = self.capabilities.turn_context
            self.command_watermark = 0
            self.events = []
            self.checkpoints = 0

        async def emit(self, kind, **kwargs):
            self.events.append({"kind": kind, **kwargs})
            acknowledged = kwargs.get("acknowledge_through")
            if acknowledged is not None:
                self.command_watermark = max(self.command_watermark, acknowledged)
            return self.events[-1]

        async def checkpoint(self, workspace):
            assert workspace == tmp_path
            self.checkpoints += 1
            return {
                "digest": f"sha256:{self.checkpoints:064x}",
                "size": self.checkpoints,
            }

    class TurnConversation:
        stage = None
        stage_name = "run"

        def __init__(self):
            self.messages = []
            self.resumed = []

        async def say(self, message, receive):
            self.messages.append(message)
            receive(
                RuntimeEvent(
                    kind="text",
                    text=f"Reply to {message}",
                    detail={"stage": "run", "invocation_id": f"reply-{message}"},
                )
            )

        async def resume(self, invocation_id, receive):
            self.resumed.append(invocation_id)
            receive(
                RuntimeEvent(
                    kind="text",
                    text="Resumed reply",
                    detail={"stage": "run", "invocation_id": invocation_id},
                )
            )

        async def close(self):
            return None

    def command(sequence, content):
        return {
            "sequence": sequence,
            "message_id": f"message-{sequence}",
            "kind": "user_message",
            "payload": {"content": content},
        }

    warm_client = TurnClient()
    warm_runtime = HostedChatRuntime(
        client=warm_client,  # type: ignore[arg-type]
        policy=load_chat_policy(policy_path),
        workspace=tmp_path,
        source=tmp_path,
        job={"metadata": {"agent_name": "demo"}, "scenario_count": 1},
    )
    warm_conversation = TurnConversation()
    warm_runtime.conversation = warm_conversation  # type: ignore[assignment]

    async def exercise_warm_turns():
        await warm_runtime._handle(command(1, "first"))
        await warm_runtime._handle(command(2, "second"))

    asyncio.run(exercise_warm_turns())
    assert warm_conversation.messages == ["first", "second"]
    assert warm_client.command_watermark == 2
    assert warm_client.checkpoints == 4
    assert [event["kind"] for event in warm_client.events].count(
        "assistant_message"
    ) == 2

    cold_client = TurnClient()
    cold_runtime = HostedChatRuntime(
        client=cold_client,  # type: ignore[arg-type]
        policy=load_chat_policy(policy_path),
        workspace=tmp_path,
        source=tmp_path,
        job={"metadata": {"agent_name": "demo"}, "scenario_count": 1},
    )
    cold_conversation = TurnConversation()
    cold_runtime.conversation = cold_conversation  # type: ignore[assignment]
    asyncio.run(cold_runtime._handle(command(1, "first")))

    assert cold_conversation.messages == []
    assert cold_client.command_watermark == 1
    assert cold_client.events == [
        {
            "kind": "turn_completed",
            "stage": "run",
            "payload": {"command_message_id": "message-1", "replayed": True},
            "acknowledge_through": 1,
        }
    ]

    journal_path = tmp_path / ".futureagi-conversation.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["message-3"] = "started"
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    resumed_client = TurnClient()
    resumed_client.turn_context["active_invocation_id"] = "provider-invocation-3"
    resumed_runtime = HostedChatRuntime(
        client=resumed_client,  # type: ignore[arg-type]
        policy=load_chat_policy(policy_path),
        workspace=tmp_path,
        source=tmp_path,
        job={"metadata": {"agent_name": "demo"}, "scenario_count": 1},
    )
    resumed_conversation = TurnConversation()
    resumed_runtime.conversation = resumed_conversation  # type: ignore[assignment]
    asyncio.run(resumed_runtime._handle(command(3, "third")))
    assert resumed_conversation.messages == []
    assert resumed_conversation.resumed == ["provider-invocation-3"]
    assert resumed_client.command_watermark == 3


def test_interrupt_stops_the_active_turn_and_acknowledges_control(tmp_path):
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        """schema_version: 1
harness_chat:
  enabled: true
  stages:
    reception:
      enabled: true
""",
        encoding="utf-8",
    )

    class InterruptClient:
        def __init__(self):
            self.capabilities = _capabilities()
            self.turn_context = self.capabilities.turn_context
            self.command_watermark = 0
            self.events = []

        async def commands(self):
            return [
                {
                    "sequence": 2,
                    "message_id": "interrupt-2",
                    "kind": "interrupt",
                    "payload": {},
                }
            ]

        async def emit(self, kind, **kwargs):
            self.events.append({"kind": kind, **kwargs})
            acknowledged = kwargs.get("acknowledge_through")
            if acknowledged is not None:
                self.command_watermark = max(self.command_watermark, acknowledged)
            return self.events[-1]

        async def checkpoint(self, workspace):
            assert workspace == tmp_path
            return {"digest": "sha256:" + "0" * 64, "size": 0}

    class InterruptibleConversation:
        stage = None
        stage_name = "run"

        def __init__(self):
            self.closed = False

        async def say(self, _message, receive):
            receive(
                RuntimeEvent(
                    kind="text",
                    text="Working",
                    detail={"stage": "run", "invocation_id": "active-turn"},
                )
            )
            await asyncio.Event().wait()

        async def close(self):
            self.closed = True

    client = InterruptClient()
    runtime = HostedChatRuntime(
        client=client,  # type: ignore[arg-type]
        policy=load_chat_policy(policy_path),
        workspace=tmp_path,
        source=tmp_path,
        job={"metadata": {"agent_name": "demo"}, "scenario_count": 1},
    )
    conversation = InterruptibleConversation()
    runtime.conversation = conversation  # type: ignore[assignment]
    asyncio.run(
        runtime._handle(
            {
                "sequence": 1,
                "message_id": "message-1",
                "kind": "user_message",
                "payload": {"content": "Run the suite"},
            }
        )
    )

    assert conversation.closed
    assert client.command_watermark == 2
    assert any(event["kind"] == "turn_interrupted" for event in client.events)
    completed = client.events[-1]
    assert completed["kind"] == "turn_completed"
    assert completed["payload"]["outcome"] == "interrupted"


def test_answer_does_not_acknowledge_queued_followup(tmp_path, monkeypatch):
    monkeypatch.setenv("ALK_HARNESS", "claude")
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "schema_version: 1\nharness_chat:\n  enabled: true\n"
        "  stages:\n    reception:\n      enabled: true\n"
    )

    class QueuedClient(QuestionClient):
        async def commands(self):
            answer = (await super().commands())[0]
            answer["sequence"] = 4
            return [
                {
                    "sequence": 2,
                    "message_id": "followup",
                    "kind": "user_message",
                    "payload": {"content": "Explain the scenario"},
                },
                answer,
            ]

    client = QueuedClient()
    runtime = HostedChatRuntime(
        client=client,
        policy=load_chat_policy(policy_path),
        workspace=tmp_path,
        source=tmp_path,
        job={},
    )
    runtime._turn_acknowledged_through = 1
    result = asyncio.run(
        runtime._ask("AskUserQuestion", {"question": "Which file?"}, None)
    )
    assert result["content"][0]["text"] == "Use app/api.py"
    assert runtime._turn_acknowledged_through == 1
    asyncio.run(runtime._handle((asyncio.run(client.commands()))[-1]))
    assert client.events[-1]["acknowledge_through"] == 4


def test_vertex_coordinator_opens_durable_session(tmp_path, monkeypatch):
    monkeypatch.setenv("ALK_HARNESS", "vertex-gemini")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "schema_version: 1\nharness_chat:\n  enabled: true\n"
        "  stages:\n    reception:\n      enabled: true\n"
    )
    client = QuestionClient()
    client.turn_context = {
        **client.turn_context,
        "capabilities": {"control_only": True},
    }
    runtime = HostedChatRuntime(
        client=client,
        policy=load_chat_policy(policy_path),
        workspace=tmp_path,
        source=tmp_path,
        job={},
    )

    async def exercise():
        stage = runtime.conversation.stage
        async with stage:
            context = stage.spec.conversation
            stored = await runtime.event_store.get_session(
                app_name=context.app_name,
                user_id=context.user_id,
                session_id=context.session_id,
            )
            assert stored.id == context.session_id

    asyncio.run(exercise())


def test_control_coordinator_restores_journal_without_resuming_missing_provider_session(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ALK_HARNESS", "claude")
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "schema_version: 1\nharness_chat:\n  enabled: true\n"
        "  stages:\n    reception:\n      enabled: true\n"
    )

    class PersistentClient(QuestionClient):
        def __init__(self):
            super().__init__()
            self.turn_context = {
                **self.turn_context,
                "capabilities": {"control_only": True},
            }
            self.entries = []

        async def append_session(self, key, entries):
            self.entries.extend(entries)

        async def load_session(self, key):
            return {"entries": self.entries}

        async def checkpoint(self, workspace):
            raise AssertionError(
                "Control-only chat must not overwrite authoring archives"
            )

    client = PersistentClient()
    warm = tmp_path / "warm"
    cold = tmp_path / "cold"
    warm.mkdir()
    cold.mkdir()
    (warm / ".futureagi-provider-sessions.json").write_text(
        json.dumps({"claude:reception": "provider-session-123"})
    )
    (warm / "contract.json").write_text('{"authoring": "untouched"}')
    runtime = HostedChatRuntime(
        client=client,
        policy=load_chat_policy(policy_path),
        workspace=warm,
        source=warm,
        job={},
    )
    runtime._write_journal({"message-1": "completed"})
    asyncio.run(runtime._checkpoint())
    restored = HostedChatRuntime(
        client=client,
        policy=load_chat_policy(policy_path),
        workspace=cold,
        source=cold,
        job={},
    )
    asyncio.run(restored._restore_coordinator_state())
    assert restored.conversation.stage.spec.conversation.resume_session_id is None
    assert restored._read_journal() == {"message-1": "completed"}
    assert not (cold / ".futureagi-provider-sessions.json").exists()
    assert not (cold / "contract.json").exists()


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
def test_definite_conversation_errors_do_not_retry(status, monkeypatch):
    from types import SimpleNamespace
    from fi.alk.harness.hosted_conversation import ConversationTransportError

    class Transport:
        calls = 0

        def request(self, *args, **kwargs):
            self.calls += 1
            return SimpleNamespace(status_code=status, body={})

    async def unexpected_sleep(_delay):
        pytest.fail("A definite error must not be retried")

    monkeypatch.setattr(asyncio, "sleep", unexpected_sleep)
    transport = Transport()
    client = ConversationClient(_capabilities(), transport=transport)
    with pytest.raises(ConversationTransportError, match=f"HTTP {status}"):
        asyncio.run(client._request("GET", client.capabilities.endpoints.commands))
    assert transport.calls == 1


def test_unavailable_requests_stop_before_platform_stale_window(monkeypatch):
    from types import SimpleNamespace
    from fi.alk.harness.hosted_conversation import ConversationTransportError

    clock = [0.0]
    timeouts = []

    class Transport:
        def request(self, *args, timeout, **kwargs):
            timeouts.append(timeout)
            clock[0] += timeout
            return SimpleNamespace(status_code=503, body={})

    async def exercise():
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "time", lambda: clock[0])

        async def sleep(delay):
            clock[0] += delay

        monkeypatch.setattr(asyncio, "sleep", sleep)
        client = ConversationClient(_capabilities(), transport=Transport())
        with pytest.raises(ConversationTransportError, match="remained unavailable"):
            await client._request("GET", client.capabilities.endpoints.commands)

    asyncio.run(exercise())
    assert 35 < clock[0] <= 60
    assert timeouts[-1] < timeouts[0]
