from __future__ import annotations

import argparse
import asyncio
import traceback
import json
import signal
import uuid
from collections.abc import Awaitable
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any

from .backends import (
    ASK_TOOL,
    ConversationSession,
    SessionSpec,
    resolve,
    tool,
    tool_server,
)
from .chat import Conversation, open_conversation
from .chat_policy import EffectiveChatPolicy, apply_chat_policy, load_chat_policy
from .hosted_conversation import (
    ConversationClient,
    ConversationTransportError,
    PlatformSessionStore,
    load_conversation_capabilities,
)
from .tools import schema
from .session import ARTIFACT, DONE, RESULT, TEXT, TOOL, Event, Stage

_PENDING_QUESTION = ".futureagi-pending-question.json"
_JOURNAL = ".futureagi-conversation.json"
_PROVIDER_SESSIONS = ".futureagi-provider-sessions.json"

# Both the foreground coordinator and resumed stage conversations need this boundary:
# authoring tools can edit harness artifacts, but cannot update the target configuration
# used by the platform or deploy changes to an externally hosted agent.
_HOSTED_CHAT_CAPABILITIES = """

## What this chat can change

Check whether the requested action is supported BEFORE acknowledging work or calling tools.
This chat manages the test environment and scenarios for the existing target agent. Available
tools can inspect progress/results, author or edit environment fixtures and scenarios, correct
the contract's interpretation of existing agent behavior, and submit simulations when the run
tool is available. A correction to the contract changes ALK's understanding, not the agent.

This chat CANNOT change the saved target or connection configuration: the phone number dialed
by a simulation, provider/assistant/agent IDs, connection URLs, credentials, or target model and
voice settings. It also CANNOT edit or deploy the target agent's actual system prompt,
instructions, source code, or provider configuration. Updating a local artifact, a copied source
file, or the contract does not apply any of those changes. Stage routing and request_adjustment
do not grant these capabilities. Do not use them as a workaround or queue unsupported requests.

For an unsupported request, reply immediately with a clear, brief refusal, naming the limit.
Do not first say "working on it", "I'll update that", or promise to investigate or apply it.
Do not ask for a new phone number, prompt, or credentials when you cannot save that change.
For example:
- "Change the number we call to +1 202 555 0123" -> "I can't change the simulation's target
  phone number through this chat. It needs to be changed in the agent configuration."
- "Update the agent prompt so it always reads full addresses" -> "I can't update the target
  agent's prompt through this chat. Update it where the agent is configured. I can help add a
  scenario to test whether it reads the full address."
Only suggest alternatives this chat actually supports. Drafting or explaining a prompt is
allowed, but explicitly distinguish a proposed draft from an applied change. Do not invent UI
buttons, settings paths, or claim that an external change has been made.

Phone numbers in scenario fixture data and the simulated caller's instructions/personality are
editable test content; they are different from the target phone number and target agent prompt.
Do not refuse those supported edits just because they mention a number or a prompt. If the user
only says "change the phone number" or "update the prompt" and the conversation does not make
the intended object clear, ask which one they mean before promising work. For a mixed request,
state which part is unsupported and act only on the supported part the user asked for.

For any other requested action, use only capabilities actually exposed by the available tools.
If there is no tool that can apply the change, explain that it is not possible through this chat.
A queued adjustment is pending, not applied: only report completion when a tool result or
observable artifact proves that the requested change has taken effect.
"""


class CoordinatorConversation:
    """One foreground model that owns the user conversation and delegates stage work."""

    def __init__(self, stage: Stage) -> None:
        self.stage = stage
        self.stage_name = "reception"
        self.opened = False

    async def say(self, message: str, on_event: Any | None = None) -> Any:
        if not self.opened:
            await self.stage.__aenter__()
            self.opened = True
        return await self.stage.say(message, on_event=on_event)

    async def resume(self, invocation_id: str, on_event: Any | None = None) -> Any:
        if not self.opened:
            await self.stage.__aenter__()
            self.opened = True
        return await self.stage.resume_turn(invocation_id, on_event=on_event)

    async def interrupt_response(self) -> bool:
        return await self.stage.interrupt_response()

    async def close(self) -> None:
        if self.opened:
            await self.stage.__aexit__()
            self.opened = False


class HostedChatRuntime:
    def __init__(
        self,
        *,
        client: ConversationClient,
        policy: EffectiveChatPolicy,
        workspace: Path,
        source: Path,
        job: dict[str, Any],
    ) -> None:
        self.client = client
        self.policy = policy
        self.workspace = workspace
        self.source = source
        self.job = job
        backend = resolve()
        self.backend_name = backend.name
        self.event_store: Any = None
        if backend.name == "vertex-gemini":
            from google.adk.sessions import DatabaseSessionService

            session_db = (workspace / ".futureagi-adk.db").resolve()
            self.event_store = DatabaseSessionService(
                db_url=f"sqlite+aiosqlite:///{session_db.as_posix()}"
            )
        self.transcript_store = PlatformSessionStore(client)
        self.stopping = asyncio.Event()
        self._turn_acknowledged_through = 0
        self._tool_names: dict[str, str] = {}
        self.conversation = self._conversation()

    def _conversation(self) -> Conversation | CoordinatorConversation:
        metadata = self.job.get("metadata") or {}
        name = str(metadata.get("agent_name") or self.job.get("name") or "agent")
        if not self._control_only():
            return open_conversation(
                name=name,
                path=str(self.source),
                kind="repo",
                out=self.workspace,
                wanted=int(self.job.get("scenario_count") or 10),
                workspace=self.source,
                ask=self._ask,
                configure_stage=self._configure_stage,
                control_only=False,
            )
        identity = self.client.capabilities.identity
        spec = SessionSpec(
            system_prompt=(
                "You are the foreground harness coordinator. The authoring stages run as "
                "background workers in this same sandbox. Answer the user immediately from "
                "run status and observable progress; do not reconstruct or execute Understand, "
                "Build, Scenarios, or Run yourself. Use get_run_status for progress. Supported "
                "authoring changes can be queued through request_adjustment for the worker to "
                "apply at a safe boundary. AskUserQuestion is available only when the worker "
                "reports a genuine missing requirement; never invent a question."
                + _HOSTED_CHAT_CAPABILITIES
            ),
            servers={"platform-control": self._control_server()},
            builtins=("Read", "Glob", "Grep", ASK_TOOL),
            max_turns=12,
            ask=self._ask,
            conversation=ConversationSession(
                app_name=identity.app_name,
                user_id=identity.user_id,
                session_id=f"{identity.app_name}-{self.client.capabilities.conversation_id}-coordinator",
                transcript_store=self.transcript_store,
                event_store=self.event_store,
                resume_session_id=self._provider_sessions().get(
                    f"{self.backend_name}:reception"
                ),
                config_dir=str(self.workspace / ".claude-coordinator"),
                turn_context=dict(self.client.turn_context),
            ),
        )
        configured = apply_chat_policy(spec, stage="reception", policy=self.policy)
        return CoordinatorConversation(Stage(configured, name="coordinator"))

    def _control_only(self) -> bool:
        capabilities = self.client.turn_context.get("capabilities") or {}
        return bool(capabilities.get("control_only"))

    def _control_server(self):
        @tool(
            "get_run_status",
            "Read the current authoring or simulation run status.",
            schema({}, []),
        )
        async def get_run_status(_args: dict[str, Any]) -> dict[str, Any]:
            status = await self.client.run_status()
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(status, sort_keys=True),
                    }
                ]
            }

        @tool(
            "request_adjustment",
            "Queue a supported authoring change to environment fixtures, scenarios, or the "
            "contract's interpretation of existing agent behavior. This does not change the "
            "target phone number, saved connection settings, target agent prompt, source code, "
            "or provider configuration. Refuse those requests directly without this tool. "
            "Answer questions directly. A pending response means queued, not applied.",
            schema({"instruction": str}, ["instruction"]),
        )
        async def request_adjustment(args: dict[str, Any]) -> dict[str, Any]:
            instruction = str(args.get("instruction") or "").strip()
            if not instruction:
                return {
                    "content": [{"type": "text", "text": "instruction is required"}],
                    "is_error": True,
                }
            result = await self.client.adjust(instruction)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, sort_keys=True),
                    }
                ]
            }

        return tool_server(
            name="platform-control",
            version="0.1.0",
            tools=[get_run_status, request_adjustment],
        )

    def _configure_stage(self, stage: str, spec: SessionSpec) -> SessionSpec:
        capabilities = self.client.capabilities
        turn_context = {**self.client.turn_context, "stage": stage}
        session_id = f"{capabilities.conversation_id}-{stage}"
        resume_session_id = self._provider_sessions().get(
            f"{self.backend_name}:{stage}"
        )
        control_only = self._control_only()
        hosted_prompt = (
            "\n\n## Hosted conversation behavior\n"
            "Reply directly to the user in this conversation. First check the capability "
            "boundaries below. For a supported actionable request, say one short truthful "
            "acknowledgement before calling a tool. Refuse unsupported changes immediately. "
            "Never describe requested work as completed until its tool result proves "
            "completion. Ask one concise question only when a required fact cannot be "
            "derived from the workspace. Use route_to_stage when another harness "
            "stage owns a supported request; stages are an internal detail."
            + _HOSTED_CHAT_CAPABILITIES
        )
        if control_only:
            hosted_prompt += (
                "\n\nThe initial authoring run is active. This is a side conversation: "
                "answer questions without interrupting or changing the authoring workspace. "
                "When the user explicitly requests a supported authoring change, call "
                "request_adjustment; the active runner can apply it at a safe stage boundary. "
                "A pending adjustment has not been applied yet. Use get_run_status for "
                "current progress. Never treat a question as an adjustment."
            )
        servers = spec.servers
        builtins = tuple(dict.fromkeys((*spec.builtins, ASK_TOOL)))
        if control_only:
            servers = {"platform-control": self._control_server()}
            builtins = tuple(
                name for name in builtins if name in {"Read", "Glob", "Grep", ASK_TOOL}
            )
        elif stage == "run":
            servers = {
                server_name: replace(
                    server,
                    tools=[
                        (
                            replace(tool, handler=self._run_platform_simulation)
                            if tool.name == "run_simulation"
                            else replace(tool, handler=self._read_platform_results)
                            if tool.name in {"read_run", "read_results"}
                            else tool
                        )
                        for tool in server.tools
                        if tool.name != "run_scenario"
                    ],
                )
                for server_name, server in spec.servers.items()
            }
        configured = replace(
            spec,
            servers=servers,
            system_prompt=spec.system_prompt + hosted_prompt,
            builtins=builtins,
            ask=self._ask,
            conversation=ConversationSession(
                app_name=capabilities.identity.app_name,
                user_id=capabilities.identity.user_id,
                session_id=session_id,
                event_store=self.event_store,
                transcript_store=self.transcript_store,
                resume_session_id=resume_session_id,
                config_dir=str(self.workspace / ".claude"),
                turn_context=turn_context,
                streaming=True,
            ),
        )
        return apply_chat_policy(configured, stage=stage, policy=self.policy)

    async def _pump_authoring_activity(self) -> None:
        """Forward the active harness's safe observable events through this same lease."""
        path = self.workspace / "harness-events.jsonl"
        try:
            offset = len(path.read_text(encoding="utf-8").splitlines())
        except OSError:
            offset = 0
        while not self.stopping.is_set():
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                lines = []
            while offset < len(lines):
                line = lines[offset]
                offset += 1
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    offset -= 1
                    break
                if not isinstance(event, dict):
                    continue
                payload = event.get("payload") or {}
                if not isinstance(payload, dict):
                    payload = {"value": payload}
                try:
                    await self.client.emit(
                        "authoring_activity",
                        stage=str(payload.get("stage") or "authoring"),
                        payload={
                            "event_type": event.get("event_type"),
                            "event": payload,
                        },
                    )
                except ConversationTransportError:
                    offset -= 1
                    break
            try:
                await asyncio.wait_for(self.stopping.wait(), timeout=0.5)
            except TimeoutError:
                continue

    async def run(self) -> None:
        if self._control_only():
            await self._restore_coordinator_state()
        activity_task = asyncio.create_task(self._pump_authoring_activity())
        try:
            while not self.stopping.is_set():
                commands = await self.client.commands()
                pending = [
                    command
                    for command in commands
                    if int(command["sequence"]) > self.client.command_watermark
                ]
                if not pending:
                    try:
                        await asyncio.wait_for(self.stopping.wait(), timeout=0.5)
                    except TimeoutError:
                        continue
                    break
                await self._handle(pending[0])
        finally:
            self.stopping.set()
            activity_task.cancel()
            with suppress(asyncio.CancelledError):
                await activity_task
            await self.conversation.close()

    async def _handle(self, command: dict[str, Any]) -> None:
        sequence = int(command["sequence"])
        message_id = str(command["message_id"])
        kind = str(command["kind"])
        payload = command.get("payload") or {}
        if kind in {"user_response", "approval"}:
            await self.client.emit(
                "turn_completed",
                stage=self.conversation.stage_name,
                payload={
                    "command_message_id": message_id,
                    "ignored": "no matching suspended question",
                },
                acknowledge_through=sequence,
            )
            return
        if kind in {"interrupt", "cancel_operation"}:
            await self.client.emit(
                "turn_interrupted",
                stage=self.conversation.stage_name,
                payload={"command_message_id": message_id, "kind": kind},
                acknowledge_through=sequence,
            )
            return

        journal = self._read_journal()
        state = str(journal.get(message_id) or "")
        if state == "completed":
            await self.client.emit(
                "turn_completed",
                stage=self.conversation.stage_name,
                payload={"command_message_id": message_id, "replayed": True},
                acknowledge_through=sequence,
            )
            return
        resume_invocation_id: str | None = None
        if state == "started":
            resume_invocation_id = (
                str(self.client.turn_context.get("active_invocation_id") or "") or None
            )
            if self.backend_name != "vertex-gemini":
                resume_invocation_id = None
            if resume_invocation_id is None:
                await self.client.emit(
                    "turn_interrupted",
                    stage=self.conversation.stage_name,
                    payload={
                        "command_message_id": message_id,
                        "reason": "previous sandbox stopped before the turn committed",
                    },
                )
                journal[message_id] = "completed"
                self._write_journal(journal)
                await self._checkpoint()
                await self.client.emit(
                    "turn_completed",
                    stage=self.conversation.stage_name,
                    payload={"command_message_id": message_id, "interrupted": True},
                    acknowledge_through=sequence,
                )
                return
        else:
            journal[message_id] = "started"
            self._write_journal(journal)
            await self._checkpoint()
        invocation_id = resume_invocation_id or f"turn-{uuid.uuid4().hex}"
        assistant_message_id = str(uuid.uuid4())
        self._turn_acknowledged_through = sequence
        self._tool_names.clear()
        await self.client.emit(
            "turn_started",
            stage=self.conversation.stage_name,
            invocation_id=resume_invocation_id,
            payload={
                "command_message_id": message_id,
                "resumed": resume_invocation_id is not None,
            },
        )
        emitted_text: list[str] = []
        turn_failures: list[str] = []
        queue: asyncio.Queue[Event | None] = asyncio.Queue()

        def receive(event: Event) -> None:
            queue.put_nowait(event)

        pump = asyncio.create_task(
            self._pump_events(
                queue,
                assistant_message_id=assistant_message_id,
                fallback_invocation_id=invocation_id,
                emitted_text=emitted_text,
                turn_failures=turn_failures,
            )
        )
        try:
            if self.conversation.stage is not None:
                context = self.conversation.stage.spec.conversation
                if context is not None:
                    context.turn_context = {
                        **self.client.turn_context,
                        "stage": self.conversation.stage_name,
                        "command_message_id": message_id,
                    }
            turn = (
                self.conversation.resume(resume_invocation_id, receive)
                if resume_invocation_id is not None
                else self.conversation.say(str(payload.get("content") or ""), receive)
            )
            interrupted_by = await self._turn_until_interrupted(
                turn,
                command_sequence=sequence,
            )
        except Exception as exc:  # noqa: BLE001 - turn boundary reports provider/tool failures.
            traceback.print_exc()
            await queue.put(None)
            await pump
            message = f"I couldn't complete that turn: {type(exc).__name__}."
            await self.client.emit(
                "assistant_message",
                message_id=assistant_message_id,
                stage=self.conversation.stage_name,
                invocation_id=invocation_id,
                payload={"text": message, "error": type(exc).__name__},
            )
            journal[message_id] = "completed"
            self._write_journal(journal)
            await self._checkpoint()
            await self.client.emit(
                "turn_completed",
                stage=self.conversation.stage_name,
                invocation_id=invocation_id,
                payload={"command_message_id": message_id, "outcome": "failed"},
                acknowledge_through=self._turn_acknowledged_through,
            )
            return

        await queue.put(None)
        await pump
        if interrupted_by is not None:
            if emitted_text:
                await self.client.emit(
                    "assistant_message",
                    message_id=assistant_message_id,
                    stage=self.conversation.stage_name,
                    invocation_id=invocation_id,
                    payload={
                        "text": "".join(emitted_text).strip(),
                        "interrupted": True,
                    },
                )
            await self.client.emit(
                "turn_interrupted",
                stage=self.conversation.stage_name,
                invocation_id=invocation_id,
                payload={
                    "command_message_id": message_id,
                    "interrupt_message_id": str(interrupted_by["message_id"]),
                },
            )
            journal[message_id] = "completed"
            self._write_journal(journal)
            checkpoint = await self._checkpoint()
            await self.client.emit(
                "checkpoint_committed",
                stage=self.conversation.stage_name,
                invocation_id=invocation_id,
                payload=checkpoint,
            )
            await self.client.emit(
                "turn_completed",
                stage=self.conversation.stage_name,
                invocation_id=invocation_id,
                payload={"command_message_id": message_id, "outcome": "interrupted"},
                acknowledge_through=self._turn_acknowledged_through,
            )
            return
        if turn_failures:
            detail = turn_failures[-1] or "the model turn failed"
            message = f"I couldn't complete that turn: {detail[:400]}"
            await self.client.emit(
                "assistant_message",
                message_id=assistant_message_id,
                stage=self.conversation.stage_name,
                invocation_id=invocation_id,
                payload={"text": message, "error": "stage_failed"},
            )
            journal[message_id] = "completed"
            self._write_journal(journal)
            await self._checkpoint()
            await self.client.emit(
                "turn_completed",
                stage=self.conversation.stage_name,
                invocation_id=invocation_id,
                payload={"command_message_id": message_id, "outcome": "failed"},
                acknowledge_through=self._turn_acknowledged_through,
            )
            return
        final_text = "".join(emitted_text).strip()
        if final_text:
            await self.client.emit(
                "assistant_message",
                message_id=assistant_message_id,
                stage=self.conversation.stage_name,
                invocation_id=invocation_id,
                payload={"text": final_text},
            )
        journal[message_id] = "completed"
        self._write_journal(journal)
        checkpoint = await self._checkpoint()
        await self.client.emit(
            "checkpoint_committed",
            stage=self.conversation.stage_name,
            invocation_id=invocation_id,
            payload=checkpoint,
        )
        await self.client.emit(
            "turn_completed",
            stage=self.conversation.stage_name,
            invocation_id=invocation_id,
            payload={"command_message_id": message_id, "outcome": "success"},
            acknowledge_through=self._turn_acknowledged_through,
        )

    async def _turn_until_interrupted(
        self,
        turn: Awaitable[None],
        *,
        command_sequence: int,
    ) -> dict[str, Any] | None:
        turn = asyncio.create_task(turn)
        while True:
            done, _pending = await asyncio.wait({turn}, timeout=0.5)
            if done:
                await turn
                return None
            try:
                pending = [
                    command
                    for command in await self.client.commands()
                    if int(command["sequence"]) > command_sequence
                ]
            except ConversationTransportError:
                continue
            if not pending or str(pending[0]["kind"]) != "interrupt":
                continue
            interrupt = pending[0]
            interrupt_response = getattr(self.conversation, "interrupt_response", None)
            gracefully_interrupted = (
                bool(await interrupt_response())
                if interrupt_response is not None
                else False
            )
            if gracefully_interrupted:
                try:
                    await asyncio.wait_for(turn, timeout=30)
                except TimeoutError:
                    await self.conversation.close()
            else:
                turn.cancel()
                with suppress(asyncio.CancelledError):
                    await turn
                await self.conversation.close()
            self._turn_acknowledged_through = max(
                self._turn_acknowledged_through, int(interrupt["sequence"])
            )
            return interrupt

    async def _pump_events(
        self,
        queue: asyncio.Queue[Event | None],
        *,
        assistant_message_id: str,
        fallback_invocation_id: str,
        emitted_text: list[str],
        turn_failures: list[str],
    ) -> None:
        visible_stage = self.conversation.stage_name
        while True:
            event = await queue.get()
            try:
                if event is None:
                    return
                stage = str(event.detail.get("stage") or self.conversation.stage_name)
                invocation_id = str(
                    event.detail.get("invocation_id") or fallback_invocation_id
                )
                if event.detail.get("invocation_id"):
                    self.client.turn_context["active_invocation_id"] = invocation_id
                if stage != visible_stage:
                    await self.client.emit(
                        "stage_changed",
                        stage=stage,
                        invocation_id=invocation_id,
                        payload={"from": visible_stage, "to": stage},
                    )
                    visible_stage = stage
                if event.kind == TEXT and event.text:
                    emitted_text.append(event.text)
                    await self.client.emit(
                        "assistant_delta",
                        message_id=assistant_message_id,
                        stage=stage,
                        invocation_id=invocation_id,
                        payload={
                            "text": event.text,
                            "partial": bool(event.detail.get("partial")),
                        },
                    )
                elif event.kind == TOOL:
                    call_id = str(event.detail.get("call_id") or "")
                    tool = event.tool.rsplit("__", 1)[-1]
                    self._tool_names[call_id] = tool
                    await self.client.emit(
                        "tool_started",
                        stage=stage,
                        invocation_id=invocation_id,
                        function_call_id=call_id or None,
                        payload={"tool": tool, "label": event.detail.get("label")},
                    )
                    checkpoint = await self._checkpoint()
                    await self.client.emit(
                        "checkpoint_committed",
                        stage=stage,
                        invocation_id=invocation_id,
                        payload=checkpoint,
                    )
                elif event.kind == RESULT:
                    call_id = str(event.detail.get("call_id") or "")
                    tool = self._tool_names.get(call_id, "")
                    failed = bool(event.detail.get("is_error"))
                    await self.client.emit(
                        "tool_result",
                        stage=stage,
                        invocation_id=invocation_id,
                        function_call_id=call_id or None,
                        payload={"tool": tool, "text": event.text, "is_error": failed},
                    )
                elif event.kind == ARTIFACT:
                    await self.client.emit(
                        "tool_result",
                        stage=stage,
                        invocation_id=invocation_id,
                        payload={"artifact": event.detail.get("path")},
                    )
                elif event.kind == DONE and event.detail.get("outcome") == "failed":
                    turn_failures.append(
                        str(event.detail.get("error") or "stage failed")
                    )
            finally:
                queue.task_done()

    async def _run_platform_simulation(self, _args: dict[str, Any]) -> dict[str, Any]:
        status = await self.client.rerun(self.workspace)
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Hosted run {status['state']} at {status['stage']}; "
                        "its durable results are available through read_results."
                    ),
                }
            ]
        }

    async def _read_platform_results(self, _args: dict[str, Any]) -> dict[str, Any]:
        status = await self.client.run_status()
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(status, sort_keys=True),
                }
            ]
        }

    async def _ask(
        self, _tool_name: str, payload: dict[str, Any], _context: Any
    ) -> dict[str, Any]:
        question = str(payload.get("question") or "What information should I use?")
        options = [str(option) for option in payload.get("options") or []]
        pending_path = self.workspace / _PENDING_QUESTION
        try:
            pending = json.loads(pending_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pending = {}
        if (
            pending.get("question") == question
            and pending.get("options") == options
            and pending.get("message_id")
        ):
            message_id = str(pending["message_id"])
        else:
            message_id = str(uuid.uuid4())
            pending = {
                "message_id": message_id,
                "question": question,
                "options": options,
            }
            temporary = pending_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(pending, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            temporary.replace(pending_path)
        await self._checkpoint()
        await self.client.emit(
            "question_requested",
            message_id=message_id,
            stage=self.conversation.stage_name,
            payload={"prompt": question, "options": options},
        )
        while not self.stopping.is_set():
            commands = await self.client.commands()
            for command in commands:
                command_payload = command.get("payload") or {}
                if str(command_payload.get("reply_to") or "") != message_id:
                    continue
                answer = str(command_payload.get("content") or "").strip()
                try:
                    pending_path.unlink()
                except FileNotFoundError:
                    pass
                await self._checkpoint()
                answer_sequence = int(command["sequence"])
                if not any(
                    self._turn_acknowledged_through
                    < int(queued["sequence"])
                    < answer_sequence
                    for queued in commands
                ):
                    self._turn_acknowledged_through = max(
                        self._turn_acknowledged_through, answer_sequence
                    )
                return {
                    "content": [{"type": "text", "text": answer}],
                    "is_error": not bool(answer),
                }
            await asyncio.sleep(0.5)
        return {
            "content": [{"type": "text", "text": "The conversation was stopped."}],
            "is_error": True,
        }

    async def _checkpoint(self) -> dict[str, Any]:
        self._record_provider_session()
        if self._control_only():
            files = {}
            for name in (_JOURNAL, _PENDING_QUESTION):
                path = self.workspace / name
                if path.is_file():
                    files[name] = json.loads(path.read_text(encoding="utf-8"))
            await self.client.append_session(
                self._coordinator_state_key(),
                [
                    {
                        "type": "coordinator_state",
                        "uuid": str(uuid.uuid4()),
                        "files": files,
                    }
                ],
            )
            return {"control_only": True, "stored": True}
        return await self.client.checkpoint(self.workspace)

    def _coordinator_state_key(self) -> dict[str, str]:
        return {
            "project_key": "futureagi-coordinator-state",
            "session_id": self.client.capabilities.conversation_id,
            "subpath": "",
        }

    async def _restore_coordinator_state(self) -> None:
        body = await self.client.load_session(self._coordinator_state_key())
        entries = body.get("entries") or []
        if not entries:
            return
        files = entries[-1]["files"]
        for name in (_JOURNAL, _PENDING_QUESTION):
            path = self.workspace / name
            if name in files:
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(files[name]), encoding="utf-8")
                temporary.replace(path)
            else:
                path.unlink(missing_ok=True)
        self.conversation = self._conversation()

    def _provider_sessions(self) -> dict[str, str]:
        path = self.workspace / _PROVIDER_SESSIONS
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {
            str(key): str(value)
            for key, value in body.items()
            if isinstance(value, str) and value
        }

    def _record_provider_session(self) -> None:
        stage = self.conversation.stage
        if stage is None or not stage.session_id:
            return
        sessions = self._provider_sessions()
        key = f"{self.backend_name}:{self.conversation.stage_name}"
        if sessions.get(key) == stage.session_id:
            return
        sessions[key] = stage.session_id
        path = self.workspace / _PROVIDER_SESSIONS
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(sessions, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _read_journal(self) -> dict[str, str]:
        path = self.workspace / _JOURNAL
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {str(key): str(value) for key, value in body.items()}

    def _write_journal(self, body: dict[str, str]) -> None:
        path = self.workspace / _JOURNAL
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(body, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one hosted ALK conversation")
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    return parser


async def _run(args: argparse.Namespace) -> None:
    capabilities = load_conversation_capabilities(args.capabilities)
    policy = load_chat_policy(args.policy)
    job = json.loads(args.job.read_text(encoding="utf-8"))
    runtime = HostedChatRuntime(
        client=ConversationClient(capabilities),
        policy=policy,
        workspace=args.workspace,
        source=args.source,
        job=job,
    )
    loop = asyncio.get_running_loop()
    for name in ("SIGTERM", "SIGINT"):
        signum = getattr(signal, name, None)
        if signum is not None:
            try:
                loop.add_signal_handler(signum, runtime.stopping.set)
            except NotImplementedError:
                pass
    await runtime.run()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        asyncio.run(_run(args))
    except (ConversationTransportError, ValueError, OSError) as exc:
        print(f"hosted conversation stopped: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
