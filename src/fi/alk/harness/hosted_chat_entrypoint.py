from __future__ import annotations

import argparse
import asyncio
import hashlib
import traceback
import json
import os
import signal
import time
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
from .authoring_entrypoint import _load_provider_import_profile, agent_source_kind
from .understand import PROVIDER_IMPORT_PROFILE_PATH_ENV
from .chat_policy import EffectiveChatPolicy, apply_chat_policy, load_chat_policy
from .hosted_conversation import (
    _UNAVAILABLE_BUDGET_SECONDS,
    ConversationClient,
    ConversationTransportError,
    PlatformSessionStore,
    load_conversation_capabilities,
)
from .tools import schema
from .session import ARTIFACT, DONE, RESULT, STAGE_IDLE_TIMEOUT_SECONDS, TEXT, TOOL, Event, Stage

_NUMBERS = {"type": "array", "items": {"type": "integer"}}
_RANGES = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"from": {"type": "integer"}, "to": {"type": "integer"}},
        "required": ["from", "to"],
    },
}
# Commands that steer the conversation rather than say something to it.
_NOT_MESSAGES = {"interrupt", "cancel_operation", "user_response", "approval"}
_PENDING_QUESTION = ".futureagi-pending-question.json"
_BUILD_FINISHED = ".futureagi-build-finished.json"
_PROVIDER_PROFILE = "provider-import-profile.json"
_JOURNAL = ".futureagi-conversation.json"
_PROVIDER_SESSIONS = ".futureagi-provider-sessions.json"
_AUTHORING_BASIS = ".futureagi-authoring-basis.json"
_PUBLISHED = ".futureagi-published.json"
_LEDGER = ".futureagi-ledger.json"
_LEDGER_NOTES = 40
# Tools whose effect the person confirms before it happens.
_NEEDS_CONFIRMATION = frozenset(
    {
        "submit_scenario",
        "drop_scenario",
        "drop_scenarios",
        "change_world",
        "declare_tool",
        "fix_tool",
        "amend_contract",
        "add_rule",
        "drop_rule",
        "add_sub_goal",
    }
)
_CONFIRM = "Yes, make this change"
_RUN_FINISHED = frozenset({"completed", "failed", "canceled"})
_RUN_POLL_SECONDS = 20.0
# A wait, with its last poll and that poll's retries, ends before the turn would be stopped for being idle.
_RUN_WAIT_SECONDS = max(0.0, STAGE_IDLE_TIMEOUT_SECONDS - _RUN_POLL_SECONDS - _UNAVAILABLE_BUDGET_SECONDS - 30)
# What the environment is made of, for telling whether anything is left unsaved.
_PUBLISHED_PARTS = (
    "scenarios",
    "scenarios.json",
    "sub_goals.json",
    "contract.json",
    "simulator_prompt.md",
    "world.sqlite",
)
# Only these change the workspace, so only these are worth a checkpoint before they run.
_WRITING_TOOLS = frozenset(
    {
        "submit_scenario",
        "drop_scenario",
        "drop_scenarios",
        "save_scenarios",
        "add_sub_goal",
        "amend_contract",
        "add_rule",
        "drop_rule",
        "fix_tool",
        "save_changes",
        "change_world",
        "declare_tool",
        "remember",
    }
)

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


class BuilderConversation:
    """The one builder session, opened on the first message and kept for the conversation."""

    stage_name = "builder"

    def __init__(self, open_stage: Any) -> None:
        self._open_stage = open_stage
        self.stage: Stage | None = None
        self._said = False
        self.on_health: Any = None

    async def _opened(self, *, resume: bool = True) -> Stage:
        if self.stage is None:
            self.stage = await self._open_stage(resume=resume)
            await self.stage.__aenter__()
        return self.stage

    async def say(self, message: str, on_event: Any | None = None) -> Any:
        opening = self.stage is None
        stage = await self._opened()
        health = list(getattr(stage, "health_at_open", None) or []) if opening else []
        if health and not message.lstrip().startswith("/"):
            # Put first, so a resumed session that remembers an older, healthy suite cannot miss it.
            listed = "\n- ".join(health[:40]) + ("\n- and more" if len(health) > 40 else "")
            message = (
                "[The health check taken as this conversation opened found these broken:"
                f"\n- {listed}]\n\n" + message
            )
            if self.on_health is not None:
                self.on_health("\n".join(health))
        try:
            said = await stage.say(message, on_event=on_event)
        except Exception:
            if self._said:
                raise
            # A stored session that cannot be resumed (its transcript never reached the
            # platform) must not cost the conversation: start a fresh session; the kept
            # references still arrive with every message.
            traceback.print_exc()
            await self.close()
            said = await (await self._opened(resume=False)).say(message, on_event=on_event)
        self._said = True
        return said

    async def resume(self, invocation_id: str, on_event: Any | None = None) -> Any:
        return await (await self._opened()).resume_turn(invocation_id, on_event=on_event)

    async def interrupt_response(self) -> bool:
        return await self.stage.interrupt_response() if self.stage is not None else False

    async def close(self) -> None:
        if self.stage is not None:
            await self.stage.__aexit__()
            self.stage = None


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
        self._from_platform = False
        self._turn_acknowledged_through = 0
        self._tool_names: dict[str, str] = {}
        self._checkpoint_lock = asyncio.Lock()
        self._restart_after_turn = False
        self._last_health = ""
        # Set when the person answered a question in this turn, or when the opening health check
        # asked for a small repair: then consequential writes are allowed.
        self._opened_with: tuple[bool, bool] | None = None
        self._last_progress: Any = None
        self._idle_turns = 0
        self.conversation = self._conversation()
        if isinstance(self.conversation, BuilderConversation):
            # What the sandbox started from is what the environment holds, unless an earlier
            # sandbox of this conversation already recorded otherwise.
            self._published_fingerprint()

    def _conversation(self) -> BuilderConversation | CoordinatorConversation:
        if self._side_chat():
            return self._coordinator()
        # One harness for the environment's whole life: it builds, talks, changes and repairs.
        builder = BuilderConversation(self._open_builder)
        builder.on_health = self._note_health
        return builder

    async def _open_builder(self, *, resume: bool = True) -> Stage:
        from .builder import BUILDER
        from .poc_guest_booking import guest_booking_policy_job
        from .understand import load as load_contract

        policy_job = guest_booking_policy_job(self.job)
        contract = load_contract(self.workspace)
        self._opened_with = self._what_exists()
        if not self._opened_with[1]:
            stage = self._open_for_building(contract)
        else:
            try:
                overview = await self.client.overview()
            except ConversationTransportError:
                overview = None
            stage = self._open_built(contract, overview, policy_job)
        if not self._control_only():
            self._guard(stage.spec)
        capabilities = self.client.capabilities
        stage.configure(
            apply_chat_policy(
                replace(
                    stage.spec,
                    conversation=ConversationSession(
                        app_name=capabilities.identity.app_name,
                        user_id=capabilities.identity.user_id,
                        session_id=f"{capabilities.conversation_id}-{BUILDER}",
                        event_store=self.event_store,
                        transcript_store=self.transcript_store,
                        resume_session_id=(
                            self._provider_sessions().get(f"{self.backend_name}:{BUILDER}")
                            if resume
                            else None
                        ),
                        config_dir=str(self.workspace / ".claude"),
                        turn_context={**self.client.turn_context, "stage": BUILDER},
                        streaming=True,
                    ),
                ),
                stage=BUILDER,
                policy=self.policy,
            )
        )
        return stage

    def _hosted_prompt(self) -> str:
        return (
            "\n\n## Hosted conversation behavior\n"
            "Reply directly to the user in this conversation. First check the capability "
            "boundaries below. Refuse unsupported changes immediately. Never describe "
            "requested work as completed until a tool result proves it."
            + _HOSTED_CHAT_CAPABILITIES
        )

    def _open_built(self, contract: Any, overview: Any, policy_job: Any) -> Stage:
        from .builder import environment_brief, open_builder
        from .poc_guest_booking import guest_booking_pin_scenario_problem

        return open_builder(
            contract,
            brief=environment_brief(contract, self.workspace, overview),
            out=self.workspace,
            source_root=self.source if self.source.is_dir() else None,
            platform=self._builder_platform_server(),
            ask=self._ask,
            scenario_problem=(
                (lambda args: guest_booking_pin_scenario_problem(policy_job, args))
                if policy_job is not None
                else None
            ),
            hosted_prompt=self._hosted_prompt(),
            sealed=not self._control_only(),
            # While building, the world itself (tools, data, caller prompt) can still change.
            more_servers=self._world_servers(contract) if self._control_only() else None,
        )

    def _world_servers(self, contract: Any) -> dict[str, Any]:
        from .build import open_stage as build_stage

        stage, _ = build_stage(
            contract,
            out=self.workspace,
            source_root=str(self.source if self.source.is_dir() else ""),
            deferred_runtime=True,
        )
        return dict(stage.spec.servers)

    def _open_for_building(self, contract: Any) -> Stage:
        from .builder import open_for_building
        from .job import HarnessJob
        from .sources import resolve

        metadata = self.job.get("metadata") or {}
        name = str(metadata.get("agent_name") or self.job.get("name") or "agent")
        kind = str(metadata.get("source_kind") or "repo")
        try:
            job = HarnessJob.model_validate(self.job)
        except Exception:  # noqa: BLE001 - the evals offered are optional context
            job = None
        profile = self._provider_profile(job) if job is not None else None
        if job is not None and agent_source_kind(job, profile) == "provider":
            source = resolve("provider", name=name, profile=profile or {}, scratch=str(self.source))
        else:
            source = resolve(kind if kind in {"repo", "spec"} else "repo", name=name, root=str(self.source))
        return open_for_building(
            contract,
            out=self.workspace,
            source=source,
            job=job,
            platform=self._builder_platform_server(),
            ask=self._ask,
            hosted_prompt=self._hosted_prompt(),
        )

    def _provider_profile(self, job: Any) -> dict[str, Any] | None:
        """The provider's definition the build process fetched, shared with this one through its cache.

        The fetch needs the provider credential, which only the build process holds, so this
        waits briefly for its cache rather than opening without the agent's definition.
        """
        cache = self.source.parent / _PROVIDER_PROFILE
        for _ in range(60):
            try:
                profile = _load_provider_import_profile(job, None, cache)
                break
            except RuntimeError:
                time.sleep(1)
        else:
            raise RuntimeError("the provider's definition was not fetched by the build process")
        if profile is not None:
            if not cache.is_file():
                cache.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.environ[PROVIDER_IMPORT_PROFILE_PATH_ENV] = str(cache)
        return profile

    def _what_exists(self) -> tuple[bool, bool]:
        """Whether the workspace has a contract, and a world: what the harness's tools follow."""
        from .understand import load as load_contract
        from .world.snapshot import saved as world_saved

        return load_contract(self.workspace) is not None, bool(world_saved(self.workspace))

    def _guard(self, spec: SessionSpec) -> None:
        """Once the environment is published, consequential changes wait for the person's yes,
        whoever makes them: the builder or a sub-agent it briefed."""

        def guarded(servers: dict[str, Any]) -> dict[str, Any]:
            return {
                name: replace(server, tools=[self._confirmed(one) for one in server.tools])
                for name, server in servers.items()
            }

        spec.servers = guarded(spec.servers)
        spec.workers = {
            name: replace(worker, servers=guarded(worker.servers))
            for name, worker in spec.workers.items()
        }

    def _confirmed(self, spec: Any) -> Any:
        """A consequential tool that runs only once the person has said yes to this very call.

        A change to what a scenario tests, a removal, a replaced sub-goal, or a change to the world
        or contract does not run on the strength of the request alone: the person is asked to
        confirm the exact call, and only that answer runs it. Adding scenarios and surface edits
        do not ask.
        """
        if spec.name not in _NEEDS_CONFIRMATION:
            return spec
        handler = spec.handler

        async def guarded(args: dict[str, Any]) -> Any:
            if self._from_platform:
                if spec.name in ("drop_scenarios", "drop_scenario") or (
                    spec.name == "submit_scenario" and self._changes_situation(args)
                ):
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": "Not done: during the build and its check a scenario keeps the "
                                "situation it tests. Correct its answer key (solution, fixed values, "
                                "sub-goals, setup) for that same situation; if the agent fails it, that "
                                "is the agent's fault: report it in your summary.",
                            }
                        ],
                        "is_error": True,
                    }
                return await handler(args)
            if spec.name == "submit_scenario" and not self._changes_meaning(args):
                return await handler(args)
            if spec.name == "add_sub_goal" and not self._replaces_sub_goal(args):
                return await handler(args)
            answer = await self._ask(
                spec.name,
                {
                    "question": f"Confirm this change: {spec.name} "
                    + json.dumps(args, sort_keys=True, default=str, indent=1),
                    "options": [_CONFIRM, "No"],
                },
                None,
            )
            said = " ".join(
                str(part.get("text") or "") for part in (answer.get("content") or [])
            ).strip() if isinstance(answer, dict) else ""
            if said == _CONFIRM:
                return await handler(args)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Not done: the person did not confirm this change (they said: "
                        f"{said or 'nothing'}). Leave it undone and follow what they said.",
                    }
                ],
                "is_error": True,
            }

        return replace(spec, handler=guarded)

    def _changes_situation(self, args: dict[str, Any]) -> bool:
        """Whether a submission changes what an existing scenario tests, not how it is graded."""
        return self._changes(
            args, ("instruction", "use_case", "branch", "tests", "coverage", "fixture"), whole_persona=True
        )

    def _replaces_sub_goal(self, args: dict[str, Any]) -> bool:
        from .catalogue import load_catalogue

        return any(one.name == args.get("name") for one in load_catalogue(self.workspace).sub_goals)

    def _changes_meaning(self, args: dict[str, Any]) -> bool:
        """Whether a submission rewrites an existing scenario's test rather than its surface."""
        return self._changes(args, ("instruction", "use_case", "branch", "sub_goals", "solution"))

    def _changes(self, args: dict[str, Any], fields: tuple[str, ...], whole_persona: bool = False) -> bool:
        """Whether a submission changes any of these fields of an existing scenario, or who its person is.

        `whole_persona` compares the persona the submission would save, so a changed language or a
        dropped persona counts: a submission replaces the scenario rather than patching it.
        """
        path = self.workspace / "scenarios" / str(args.get("name") or "") / "scenario.json"
        try:
            held = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if any(field in args and args.get(field) != held.get(field) for field in fields):
            return True
        persona, before = args.get("persona") or {}, held.get("persona") or {}
        if whole_persona:
            from .scenario import Persona

            def saved(one: dict[str, Any]) -> dict[str, Any]:
                return Persona.model_validate(one).model_dump(exclude_defaults=True) if one else {}

            try:
                return saved(persona) != saved(before)
            except ValueError:  # an unreadable persona is never a safe unattended change
                return True
        return any(
            key in persona and persona.get(key) != before.get(key)
            for key in ("name", "gender", "age_group")
        )

    def _builder_platform_server(self):
        def text(value: Any, *, error: bool = False) -> dict[str, Any]:
            body = value if isinstance(value, str) else json.dumps(value, indent=1, sort_keys=True)
            return {"content": [{"type": "text", "text": body}], "is_error": error}

        async def call(request: Awaitable[dict[str, Any]]) -> dict[str, Any]:
            try:
                return text(await request)
            except ConversationTransportError as exc:
                return text(str(exc), error=True)

        @tool(
            "find_scenarios",
            "Read the environment's scenarios exactly as the Scenarios tab shows them, with each "
            "one's tab number. Returns total_scenarios (the whole suite), matching (how many fit "
            "what you asked) and one page of rows. Narrow with any of: `numbers` (tab numbers, "
            "e.g. [12, 250, 499]), `ranges` (e.g. [{\"from\": 3, \"to\": 7}]), `names`, `search` (every word must match a "
            "scenario's name, situation, use case or branch), `filters` (field to a list of "
            "values, e.g. {\"persona.accent\": [\"Indian\"]}; fields: use_case, persona.name, "
            "persona.accent, persona.languages, persona.age_group, persona.gender, "
            "persona.location, persona.personality, persona.communication_style, "
            "background_noise, coverage.overlay, coverage.overlay_intensity, coverage.task, "
            "sub_goals, keywords; append _not to exclude). `full` returns every field. Pages "
            "hold up to 100 rows.",
            schema(
                {
                    "numbers": _NUMBERS,
                    "ranges": _RANGES,
                    "names": {"type": "array", "items": {"type": "string"}},
                    "search": str,
                    "filters": {"type": "object"},
                    "page": int,
                    "page_size": int,
                    "full": bool,
                },
                [],
            ),
        )
        async def find_scenarios(args: dict[str, Any]) -> dict[str, Any]:
            query = {key: value for key, value in args.items() if value not in (None, "", [], {})}
            return await call(self.client.scenarios(query))

        @tool(
            "save_changes",
            "Put every change made in this conversation into the environment: scenarios added, "
            "rewritten or dropped, sub-goals and contract corrections. Seals them so runs play "
            "them, and returns exactly what changed by tab number. Until this says applied, "
            "nothing you changed is in the environment. `force` reseals and publishes even when "
            "nothing changed here, to repair a sealed bundle or index out of step.",
            schema({"force": bool}, []),
        )
        async def save_changes(args: dict[str, Any]) -> dict[str, Any]:
            result = await self._save_changes(force=bool(args.get("force")))
            return text(result, error=not result.get("applied"))

        @tool(
            "read_runs",
            "The environment's runs, newest first, or with `run_id` one run's calls: each "
            "scenario's number and name, pass or fail, every sub-goal with whether it held and "
            "why, turns, duration and how the call ended. `wait` with a `run_id` holds until "
            "that run has finished or a few minutes have passed; a run still going says so, and "
            "you wait on it again.",
            schema({"run_id": str, "limit": int, "wait": bool}, []),
        )
        async def read_runs(args: dict[str, Any]) -> dict[str, Any]:
            run_id, limit = str(args.get("run_id") or ""), int(args.get("limit") or 10)
            loop = asyncio.get_running_loop()
            deadline = loop.time() + _RUN_WAIT_SECONDS
            try:
                result = await self.client.runs(run_id, limit)
                while (
                    args.get("wait")
                    and run_id
                    and result.get("state") not in _RUN_FINISHED | {None}
                    and loop.time() < deadline
                    and not self.stopping.is_set()
                ):
                    await asyncio.sleep(min(_RUN_POLL_SECONDS, deadline - loop.time()))
                    result = await self.client.runs(run_id, limit)
            except ConversationTransportError as exc:
                return text(str(exc), error=True)
            return text(result)

        @tool(
            "run_scenarios",
            "Start a run of the numbered scenarios against the agent, each played `repeats` times "
            "(1 to 20). Every call costs time and money: confirm the numbers and repeats with the "
            "person first unless they said exactly what to run or this is your check of a finished "
            "build. Unsaved changes are saved first.",
            schema({"numbers": _NUMBERS, "ranges": _RANGES, "repeats": int}, []),
        )
        async def run_scenarios(args: dict[str, Any]) -> dict[str, Any]:
            if not args.get("numbers") and not args.get("ranges"):
                return text({"started": False, "because": "name the scenarios by `numbers` or `ranges`"}, error=True)
            saved = await self._save_changes()
            if not saved.get("applied"):
                return text({"started": False, "because_not_saved": saved}, error=True)
            return await call(
                self.client.start_run(
                    [int(one) for one in args.get("numbers") or []],
                    list(args.get("ranges") or []),
                    max(1, int(args.get("repeats") or 1)),
                )
            )

        @tool(
            "remember",
            "Record something this conversation must not lose when it is summarised: a decision "
            "taken, a confirmation you are waiting for, an open question, or a fact the person "
            "told you. Kept notes come back at the start of every message. `resolve` takes the "
            "number of a note that no longer applies.",
            schema(
                {
                    "kind": {"type": "string", "enum": ["decision", "pending", "question", "fact"]},
                    "note": str,
                    "resolve": int,
                },
                [],
            ),
        )
        async def remember(args: dict[str, Any]) -> dict[str, Any]:
            ledger = self._ledger()
            notes = ledger.setdefault("notes", [])
            if args.get("resolve"):
                index = int(args["resolve"]) - 1
                if not 0 <= index < len(notes):
                    return text(f"there is no note {args['resolve']}", error=True)
                notes.pop(index)
            if str(args.get("note") or "").strip():
                notes.append({"kind": str(args.get("kind") or "fact"), "note": str(args["note"]).strip()})
                del notes[:-_LEDGER_NOTES]
            await self._keep_ledger(ledger)
            return text(self._ledger_note() or "no notes kept")

        @tool(
            "finish_build",
            "Say the build is finished. It checks what a finished environment must have (a "
            "contract, a world, the scenarios, every one proved against the world and its "
            "sub-goals) and lists what is still missing; only when nothing is does it hand the "
            "environment on to be validated against the running agent and sealed. `note` says "
            "anything the person should know. The build finishes with exactly the number of "
            "scenarios it was asked for.",
            schema({"note": str}, []),
        )
        async def finish_build(args: dict[str, Any]) -> dict[str, Any]:
            from .builder import suite_health
            from .scenario_tools import load_scenarios
            from .understand import load as load_contract

            contract = load_contract(self.workspace)
            has_contract, has_world = self._what_exists()
            kept = load_scenarios(self.workspace) if has_world else []
            asked = int(self.job.get("scenario_count") or 0)
            missing = (
                ["no contract is saved"] if not has_contract
                else ["no world is saved"] if not has_world
                else ["no scenarios are saved"] if not kept
                else suite_health(contract, self.workspace, kept, prove_all=True, sealed=False)
            )
            if not missing and len(kept) < asked:
                missing = [
                    f"{len(kept)} of the {asked} scenarios the build asked for are saved; "
                    "write and prove the rest"
                ]
            if not missing and asked and len(kept) > asked:
                missing = [
                    f"{len(kept)} scenarios are saved but the build asked for {asked}; drop the "
                    "extra ones, keeping the spread, then finish"
                ]
            if missing:
                return text("Not finished:\n- " + "\n- ".join(missing[:60]), error=True)
            # The bundle is sealed from the world as it is now, after every change made during the build.
            await asyncio.to_thread(self._after_saved, world_changed=True)
            (self.workspace / _BUILD_FINISHED).write_text(
                json.dumps({"scenarios": len(kept), "asked": asked, "note": str(args.get("note") or "")}),
                encoding="utf-8",
            )
            return text(
                f"Finished: {len(kept)} scenarios (the build asked for {asked}). It is now being "
                "validated against the running agent and sealed; the environment opens when that "
                "is done, and this conversation carries on there."
            )

        tools = [find_scenarios, save_changes, read_runs, run_scenarios, remember]
        if self._control_only():
            tools.append(finish_build)
        return tool_server(name="platform", version="0.1.0", tools=tools)

    def _note_health(self, problems: str) -> None:
        self._last_health = problems

    def _ledger(self) -> dict[str, Any]:
        try:
            body = json.loads((self.workspace / _LEDGER).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            body = {}
        return body if isinstance(body, dict) else {}

    def _ledger_note(self) -> str:
        """The conversation's kept references, put at the head of every message so a summary
        of the history can never lose them."""
        ledger = self._ledger()
        touched = ledger.get("touched") or {}
        notes = ledger.get("notes") or []
        if not touched and not notes:
            return ""
        parts = [f"environment {self.client.capabilities.job_id}"]
        if touched:
            shown = sorted(touched.items(), key=lambda pair: int(pair[0]))[-40:]
            parts.append(
                "scenarios changed so far: "
                + ", ".join(f"#{number} {name} ({what})" for number, (name, what) in shown)
            )
        parts.extend(
            f"{index}. {one.get('kind')}: {one.get('note')}" for index, one in enumerate(notes, 1)
        )
        return "[Kept from earlier in this conversation: " + "; ".join(parts) + "]\n\n"

    async def _record_touched(self, result: dict[str, Any]) -> None:
        ledger = self._ledger()
        touched = ledger.setdefault("touched", {})
        for what in ("added", "removed", "changed"):
            for row in result.get(what) or []:
                if row.get("number") is not None:
                    touched[str(row["number"])] = [row.get("name"), what]
        await self._keep_ledger(ledger)

    async def _keep_ledger(self, ledger: dict[str, Any]) -> None:
        """Write the ledger locally and to the platform, so a fresh workspace still has it."""
        self._write_dotfile(_LEDGER, ledger)
        key = {**self._provider_sessions_key(), "project_key": "futureagi-ledger"}
        try:
            await self.client.append_session(
                key, [{"type": "ledger", "uuid": str(uuid.uuid4()), "ledger": ledger}]
            )
        except ConversationTransportError:
            pass

    def _environment_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for part in _PUBLISHED_PARTS:
            root = self.workspace / part
            paths = sorted(root.rglob("*")) if root.is_dir() else [root]
            for path in paths:
                if path.is_file() and "__pycache__" not in path.parts:
                    digest.update(path.relative_to(self.workspace).as_posix().encode())
                    digest.update(path.read_bytes())
        return digest.hexdigest()

    def _published_fingerprint(self) -> str:
        path = self.workspace / _PUBLISHED
        try:
            return str(json.loads(path.read_text(encoding="utf-8")).get("fingerprint") or "")
        except (OSError, ValueError):
            fingerprint = self._environment_fingerprint()
            self._write_dotfile(_PUBLISHED, {"fingerprint": fingerprint})
            return fingerprint

    def _write_dotfile(self, name: str, body: dict[str, Any]) -> None:
        path = self.workspace / name
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
        temporary.replace(path)

    async def _save_changes(self, *, force: bool = False) -> dict[str, Any]:
        """Seal and publish the workspace now, and return what the platform says changed."""
        fingerprint = self._environment_fingerprint()
        if fingerprint == self._published_fingerprint() and not force:
            return {"applied": True, "unchanged": True, "message": "nothing is waiting to be saved"}
        from .bundle_author_v2 import reseal_scenarios

        try:
            await asyncio.to_thread(reseal_scenarios, self.workspace)
        except Exception as exc:  # noqa: BLE001 - reported to the model, never swallowed
            traceback.print_exc()
            return {
                "applied": False,
                "code": "reseal_failed",
                "message": f"the edited scenarios could not be sealed for runs: {exc}"[:600],
            }
        result: dict[str, Any] = {}
        for _attempt in range(3):
            try:
                async with self._checkpoint_lock:
                    stored = await self.client.checkpoint(self.workspace)
                    result = await self.client.publish(str(stored.get("digest") or ""))
            except ConversationTransportError as exc:
                return {"applied": False, "code": "publish_unreachable", "message": str(exc)}
            if result.get("code") != "conversation_checkpoint_not_latest":
                break
        if result.get("code") == "conversation_checkpoint_stale":
            # The environment moved on underneath this copy; a fresh one comes after this turn.
            self._restart_after_turn = True
            result["message"] = (
                "The environment was changed outside this chat after this working copy was "
                "loaded, so these changes were not saved. The chat reloads the current "
                "environment after this reply; the change has to be made again on it."
            )
        if result.get("applied"):
            await self._record_touched(result)
            if result.get("revision"):
                self._write_dotfile(_AUTHORING_BASIS, {"authoring_revision": result["revision"]})
            self._write_dotfile(_PUBLISHED, {"fingerprint": self._environment_fingerprint()})
        return result

    async def _publish_if_changed(self) -> str:
        """A line for the reply when the turn left changes the model did not save itself."""
        if self._control_only():
            # While building, the build's own sealing publishes; there is nothing to save yet.
            return ""
        if self._environment_fingerprint() == self._published_fingerprint():
            return ""
        result = await self._save_changes()
        if not result.get("applied"):
            return (
                "These changes were NOT saved to the environment: "
                f"{result.get('message') or result.get('code') or 'unknown reason'}."
            )

        def named(rows: list[dict[str, Any]]) -> str:
            return ", ".join(f"#{one.get('number')} {one.get('name')}" for one in rows)

        parts = [
            f"{label} {named(result.get(key) or [])}"
            for label, key in (("added", "added"), ("removed", "removed"), ("changed", "changed"))
            if result.get(key)
        ]
        return (
            "Saved to the environment: "
            + ("; ".join(parts) or "no scenario changed")
            + f". It now has {result.get('scenario_count')} scenarios."
        )

    async def _selection_note(self, payload: dict[str, Any]) -> str:
        ids = [str(one) for one in payload.get("scenario_ids") or [] if str(one)]
        if not ids or not isinstance(self.conversation, BuilderConversation):
            return ""
        try:
            found = await self.client.scenarios({"ids": ids, "page_size": 100})
        except ConversationTransportError:
            return f"[{len(ids)} scenarios are selected on the Scenarios tab; they could not be read]\n\n"
        rows = ", ".join(f"#{row.get('number')} {row.get('name')}" for row in found.get("rows") or [])
        more = len(ids) - len(found.get("rows") or [])
        return (
            f"[Selected on the Scenarios tab ({len(ids)}): {rows}"
            + (f", and {more} more" if more > 0 else "")
            + "]\n\n"
        )

    def _control_only(self) -> bool:
        capabilities = self.client.turn_context.get("capabilities") or {}
        return bool(capabilities.get("control_only"))

    def _side_chat(self) -> bool:
        capabilities = self.client.turn_context.get("capabilities") or {}
        return self._control_only() and capabilities.get("builder_session") is False

    def _coordinator(self) -> CoordinatorConversation:
        # Without the builder session, the build sandbox keeps the side chat it has on dev.
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
                            "event_type": event.get("type") or event.get("event_type"),
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
        await self._restore_provider_sessions()
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
                    if await self._carry_on():
                        continue
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

    async def _carry_on(self) -> bool:
        """While the environment is being built, keep the harness at it until it calls
        finish_build: when it has saved something that opens more tools, it is reopened with
        them and resumes; otherwise it is told the build is not finished. Waits on the person
        while a question is open, and gives up after a few turns that change nothing."""
        if not self._control_only() or self._side_chat() or (self.workspace / _BUILD_FINISHED).is_file():
            return False
        if (self.workspace / _PENDING_QUESTION).is_file():
            return False
        exists = self._what_exists()
        progress = (exists, len(list((self.workspace / "scenarios").glob("*/scenario.json"))))
        if progress != self._last_progress:
            self._last_progress, self._idle_turns = progress, 0
        elif self._idle_turns >= 3:
            return False
        self._idle_turns += 1
        await asyncio.to_thread(self._after_saved)
        if self._opened_with is not None and exists != self._opened_with:
            await self.conversation.close()
            said = "[What you saved opened the next part of the build: your tools now cover it. Carry on.]"
        else:
            asked = int(self.job.get("scenario_count") or 0)
            said = (
                f"[The build is not finished: the test environment for this agent, with {asked} "
                "scenarios. Carry on, and call finish_build when it is.]"
                if asked
                else "[The build is not finished. Carry on, and call finish_build when it is.]"
            )
        await self._handle(
            {
                "sequence": self.client.command_watermark,
                "message_id": str(uuid.uuid4()),
                "kind": "continue",
                "payload": {"content": said},
            }
        )
        return True

    def _after_saved(self, *, world_changed: bool = False) -> None:
        """The work that follows a saved contract or world, the same as a staged build does it,
        done for whatever exists and has not had it yet, so a restart in between loses nothing."""
        from .cli import after_contract, after_world
        from .job import HarnessJob
        from .understand import load as load_contract

        has_contract, has_world = self._what_exists()
        try:
            job = HarnessJob.model_validate(self.job)
        except Exception:  # noqa: BLE001 - only a generic-harness job needs it
            job = None
        if has_contract and not (self.workspace / "generic-harness" / "source-model.json").is_file():
            after_contract(self.workspace, self.source, job)
        if has_world and (world_changed or not (self.workspace / "environment-bundle" / "manifest.json").is_file()):
            after_world(self.source if self.source.is_dir() else None, self.workspace, load_contract(self.workspace))

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
        # The build and its check are sent by the platform; nobody is there to answer during them.
        self._from_platform = bool(payload.get("from_platform"))

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
                # Answer once more; the cut-off attempt may already have changed things.
                journal[message_id] = "retried"
                self._write_journal(journal)
                await self._checkpoint()
        if state == "retried":
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
        elif state != "started":
            journal[message_id] = "started"
            self._write_journal(journal)
            await self._checkpoint()
        retry_note = (
            "[Your earlier attempt at this message was cut off when the sandbox restarted. "
            "Tools it called may already have taken effect: check before repeating a change.]\n\n"
            if state == "started" and resume_invocation_id is None
            else ""
        )
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
                else self.conversation.say(
                    str(payload.get("content") or "")
                    if str(payload.get("content") or "").lstrip().startswith("/")
                    else retry_note
                    + self._ledger_note()
                    + await self._selection_note(payload)
                    + str(payload.get("content") or ""),
                    receive,
                )
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
                    **(
                        {"interrupt_message_id": str(interrupted_by["message_id"])}
                        if str(interrupted_by["kind"]) == "interrupt"
                        else {}
                    ),
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
        saved = await self._publish_if_changed()
        if saved:
            final_text = f"{final_text}\n\n{saved}".strip()
        if self._last_health and not self._last_health.startswith("Healthy") and not self._control_only():
            problems = [line for line in self._last_health.splitlines() if line.strip()]
            final_text = (
                f"{final_text}\n\nStill open from the last health check ({len(problems)}): "
                + "; ".join(problems[:3])
                + ("; and more" if len(problems) > 3 else "")
            ).strip()
        self._last_health = ""
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
        if self._restart_after_turn:
            self.stopping.set()

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
            if not pending:
                continue
            said = str(pending[0]["kind"]) not in _NOT_MESSAGES
            # While building, the person speaking stops the turn so they are answered now;
            # the build carries on after. Afterwards a new message waits for the turn to end.
            if str(pending[0]["kind"]) != "interrupt" and not (said and self._control_only()):
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
            if not said:
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
                    if tool in _WRITING_TOOLS or not isinstance(
                        self.conversation, BuilderConversation
                    ):
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
                    if tool == "check_health":
                        self._last_health = str(event.text or "")
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
    ) -> Any:
        if self._from_platform:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Nobody answers during the build or its check. Decide it yourself from "
                        "the agent and the contract, and say what you decided in your summary.",
                    }
                ]
            }
        asked = [one for one in payload.get("questions") or [] if isinstance(one, dict)]
        question = str(
            payload.get("question")
            or "\n".join(str(one.get("question") or "") for one in asked if one.get("question"))
            or "What information should I use?"
        )
        options = [
            (
                f"{option.get('label')}: {option.get('description')}"
                if option.get("description")
                else str(option.get("label"))
            )
            if isinstance(option, dict)
            else str(option)
            for option in (payload.get("options") or (asked[0].get("options") if asked else None) or [])
        ]
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
                answered = answer
                if not any(
                    self._turn_acknowledged_through
                    < int(queued["sequence"])
                    < answer_sequence
                    for queued in commands
                ):
                    self._turn_acknowledged_through = max(
                        self._turn_acknowledged_through, answer_sequence
                    )
                if asked and self.backend_name == "claude":
                    # The CLI's own question tool takes its answers as an allowed, rewritten input.
                    from claude_agent_sdk.types import PermissionResultAllow

                    return PermissionResultAllow(
                        updated_input={
                            "questions": asked,
                            "answers": {
                                str(one.get("question") or ""): answered for one in asked
                            },
                        }
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

    def _provider_sessions_key(self) -> dict[str, str]:
        return {
            "project_key": "futureagi-provider-sessions",
            "session_id": self.client.capabilities.conversation_id,
            "subpath": "",
        }

    async def _restore_provider_sessions(self) -> None:
        """Bring back the model sessions when a fresh workspace arrived without them."""
        if (self.workspace / _PROVIDER_SESSIONS).is_file():
            return
        try:
            body = await self.client.load_session(self._provider_sessions_key())
        except ConversationTransportError:
            return
        entries = body.get("entries") or []
        sessions = entries[-1].get("sessions") if entries and isinstance(entries[-1], dict) else None
        if isinstance(sessions, dict) and sessions:
            self._write_dotfile(_PROVIDER_SESSIONS, sessions)
        if (self.workspace / _LEDGER).is_file():
            return
        try:
            kept = await self.client.load_session(
                {**self._provider_sessions_key(), "project_key": "futureagi-ledger"}
            )
        except ConversationTransportError:
            return
        entries = kept.get("entries") or []
        ledger = entries[-1].get("ledger") if entries and isinstance(entries[-1], dict) else None
        if isinstance(ledger, dict):
            self._write_dotfile(_LEDGER, ledger)

    async def _checkpoint(self) -> dict[str, Any]:
        if self._record_provider_session():
            try:
                await self.client.append_session(
                    self._provider_sessions_key(),
                    [
                        {
                            "type": "provider_sessions",
                            "uuid": str(uuid.uuid4()),
                            "sessions": self._provider_sessions(),
                        }
                    ],
                )
            except ConversationTransportError:
                pass
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
        async with self._checkpoint_lock:
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

    def _record_provider_session(self) -> bool:
        stage = self.conversation.stage
        if stage is None or not stage.session_id:
            return False
        sessions = self._provider_sessions()
        key = f"{self.backend_name}:{self.conversation.stage_name}"
        if sessions.get(key) == stage.session_id:
            return False
        sessions[key] = stage.session_id
        path = self.workspace / _PROVIDER_SESSIONS
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(sessions, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(path)
        return True

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
