from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fi.alk.harness import builder
from fi.alk.harness.chat_policy import load_chat_policy
from fi.alk.harness.hosted_chat_entrypoint import BuilderConversation, HostedChatRuntime
from fi.alk.harness.hosted_conversation import (
    ConversationCapabilities,
    ConversationClient,
    ConversationTransportError,
)
from fi.alk.harness.session import Event as RuntimeEvent


def _capabilities() -> ConversationCapabilities:
    conversation_id = "4f1ddaa9-569e-4ddd-bef0-74926bb42b16"
    prefix = f"https://platform.example/simulate/api/harness/conversations/{conversation_id}"
    names = ("commands", "events", "rerun", "adjust", "workspace", "session_store")
    endpoints = {name: f"{prefix}/{name.replace('_', '-')}/" for name in names}
    endpoints["run_status"] = f"{prefix}/run-status/"
    endpoints["session_store_append"] = f"{prefix}/session-store/append/"
    return ConversationCapabilities.model_validate(
        {
            "schema_version": "futureagi.harness-conversation.v1",
            "conversation_id": conversation_id,
            "job_id": "job-1",
            "token": "token",
            "fence": "fence",
            "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
            "identity": {"app_name": "alk-harness", "user_id": "conversation-user"},
            "turn_context": {"conversation_id": conversation_id, "stage": "builder"},
            "endpoints": endpoints,
        }
    )


class BuilderClient:
    def __init__(self, *, applied: bool = True):
        self.capabilities = _capabilities()
        self.turn_context = self.capabilities.turn_context
        self.command_watermark = 0
        self.events: list[dict] = []
        self.checkpoints = 0
        self.published: list[str] = []
        self.queries: list[dict] = []
        self.applied = applied

    async def emit(self, kind, **kwargs):
        self.events.append({"kind": kind, **kwargs})
        acknowledged = kwargs.get("acknowledge_through")
        if acknowledged is not None:
            self.command_watermark = max(self.command_watermark, acknowledged)
        return self.events[-1]

    async def checkpoint(self, workspace):
        self.checkpoints += 1
        return {"digest": f"sha256:{self.checkpoints:064x}", "size": 1}

    async def publish(self, digest):
        self.published.append(digest)
        if not self.applied:
            return {"applied": False, "code": "conversation_checkpoint_invalid", "message": "bad"}
        return {
            "applied": True,
            "revision": "sha256:new",
            "scenario_count": 41,
            "added": [{"number": 41, "name": "late_driver"}],
            "removed": [],
            "changed": [],
        }

    async def scenarios(self, query):
        self.queries.append(query)
        return {"rows": [{"number": 7, "name": "seven"}, {"number": 12, "name": "twelve"}]}

    async def append_session(self, key, entries):
        return None

    async def load_session(self, key):
        return {"entries": []}


class EditingConversation(BuilderConversation):
    """A builder whose model writes one scenario file and never calls save_changes."""

    def __init__(self, workspace: Path, *, writes: bool):
        super().__init__(lambda: None)
        self.workspace = workspace
        self.writes = writes
        self.messages: list[str] = []

    async def say(self, message, on_event=None):
        self.messages.append(message)
        if self.writes:
            (self.workspace / "scenarios" / "late_driver").mkdir(parents=True)
            (self.workspace / "scenarios" / "late_driver" / "scenario.json").write_text("{}")
        on_event(RuntimeEvent(kind="text", text="Added it.", detail={"stage": "builder"}))


def _runtime(tmp_path: Path, client: BuilderClient, monkeypatch) -> HostedChatRuntime:
    monkeypatch.setenv("ALK_HARNESS", "vertex-gemini")
    (tmp_path / "scenarios" / "one").mkdir(parents=True)
    (tmp_path / "scenarios" / "one" / "scenario.json").write_text("{}")
    policy = tmp_path / "policy.yaml"
    policy.write_text("schema_version: 1\nharness_chat:\n  enabled: true\n", encoding="utf-8")
    runtime = HostedChatRuntime(
        client=client,  # type: ignore[arg-type]
        policy=load_chat_policy(policy),
        workspace=tmp_path,
        source=tmp_path,
        job={"metadata": {"agent_name": "demo"}, "scenario_count": 1},
    )
    return runtime


def _turn(runtime: HostedChatRuntime, payload: dict) -> None:
    command = {"sequence": 1, "message_id": "m-1", "kind": "user_message", "payload": payload}
    asyncio.run(runtime._handle(command))


def _reply(client: BuilderClient) -> str:
    return next(e for e in client.events if e["kind"] == "assistant_message")["payload"]["text"]


def test_selected_rows_reach_the_builder_by_tab_number(tmp_path, monkeypatch):
    client = BuilderClient()
    runtime = _runtime(tmp_path, client, monkeypatch)
    conversation = EditingConversation(tmp_path, writes=False)
    runtime.conversation = conversation
    runtime._published_fingerprint()

    _turn(runtime, {"content": "make these harder", "scenario_ids": ["a", "b"]})

    assert client.queries == [{"ids": ["a", "b"], "page_size": 100}]
    assert conversation.messages == [
        "[Selected on the Scenarios tab (2): #7 seven, #12 twelve]\n\nmake these harder"
    ]
    assert client.published == []


def test_unsaved_changes_are_published_and_reported_by_number(tmp_path, monkeypatch):
    client = BuilderClient()
    runtime = _runtime(tmp_path, client, monkeypatch)
    runtime.conversation = EditingConversation(tmp_path, writes=True)
    runtime._published_fingerprint()
    monkeypatch.setattr("fi.alk.harness.bundle_author_v2.reseal_scenarios", lambda root: None)

    _turn(runtime, {"content": "add one"})

    assert len(client.published) == 1
    assert "Saved to the environment: added #41 late_driver" in _reply(client)
    basis = json.loads((tmp_path / ".futureagi-authoring-basis.json").read_text())
    assert basis == {"authoring_revision": "sha256:new"}
    # Saved once; nothing is left for the next turn to publish again.
    assert asyncio.run(runtime._save_changes())["unchanged"] is True


def test_a_change_to_the_world_data_alone_is_saved(tmp_path, monkeypatch):
    client = BuilderClient()
    runtime = _runtime(tmp_path, client, monkeypatch)
    (tmp_path / "world.sqlite").write_bytes(b"before")
    runtime._published_fingerprint()
    monkeypatch.setattr("fi.alk.harness.bundle_author_v2.reseal_scenarios", lambda root: None)

    (tmp_path / "world.sqlite").write_bytes(b"after")
    result = asyncio.run(runtime._save_changes())

    assert result.get("unchanged") is not True
    assert len(client.published) == 1


def test_a_rejected_publish_is_never_reported_as_done(tmp_path, monkeypatch):
    client = BuilderClient(applied=False)
    runtime = _runtime(tmp_path, client, monkeypatch)
    runtime.conversation = EditingConversation(tmp_path, writes=True)
    runtime._published_fingerprint()
    monkeypatch.setattr("fi.alk.harness.bundle_author_v2.reseal_scenarios", lambda root: None)

    _turn(runtime, {"content": "add one"})

    assert "These changes were NOT saved to the environment: bad." in _reply(client)
    assert not (tmp_path / ".futureagi-authoring-basis.json").exists()


def test_read_only_tools_skip_the_workspace_checkpoint(tmp_path, monkeypatch):
    client = BuilderClient()
    runtime = _runtime(tmp_path, client, monkeypatch)
    runtime._published_fingerprint()

    class ReadingConversation(EditingConversation):
        async def say(self, message, on_event=None):
            for name in ("find_scenarios", "submit_scenario"):
                on_event(
                    RuntimeEvent(
                        kind="tool", tool=f"mcp__x__{name}", detail={"call_id": name}
                    )
                )
            on_event(RuntimeEvent(kind="text", text="ok", detail={}))

    runtime.conversation = ReadingConversation(tmp_path, writes=False)
    _turn(runtime, {"content": "look"})

    # One before the turn, one for the write, one at the end; none for the read.
    assert client.checkpoints == 3


def test_older_platforms_without_builder_endpoints_still_load(monkeypatch):
    body = _capabilities().model_dump(mode="json")
    capabilities = ConversationCapabilities.model_validate(body)
    client = ConversationClient(capabilities)
    with pytest.raises(ConversationTransportError, match="does not offer"):
        asyncio.run(client.scenarios({}))


def test_every_skill_is_loadable_by_the_builder_except_its_own():
    index = builder.skill_index()
    assert "write-scenarios" in index and "platform" in index
    assert builder.BUILDER not in index
    assert all(index.values())


def test_the_secret_guard_refuses_env_reads_and_scrubs_the_shell():
    from fi.alk.harness.config import scrubbed_command, secret_refusal

    for tool, arguments in (
        ("Bash", {"command": "printenv"}),
        ("Bash", {"command": "echo $SOME_API_KEY"}),
        ("Bash", {"command": "cat /proc/1/environ"}),
        ("Read", {"file_path": "/work/authoring/.env"}),
        ("Read", {"file_path": "/run/futureagi/conversation.json"}),
        ("Write", {"file_path": "/work/authoring/scenarios/a/scenario.json"}),
        ("Read", {"file_path": "/work/authoring/../../run/futureagi/simulator-secrets.json"}),
        ("Read", {"file_path": "/work/.authoring-credentials/google.json"}),
        ("Grep", {"pattern": "KEY", "path": "/run"}),
        ("Grep", {"pattern": "KEY", "path": "/"}),
    ):
        assert secret_refusal(tool, arguments), (tool, arguments)
    for tool, arguments in (
        ("Bash", {"command": "ls scenarios | wc -l"}),
        ("Read", {"file_path": "/work/authoring/contract.json"}),
        ("Write", {"file_path": "/tmp/analysis.py"}),
        ("Grep", {"pattern": "refund", "path": "/work/authoring/scenarios"}),
    ):
        assert not secret_refusal(tool, arguments), (tool, arguments)
    assert scrubbed_command("ls").startswith("env -i ")


def test_world_change_is_all_or_nothing_and_reports_counts(tmp_path):
    import sqlite3

    from fi.alk.harness.builder import world_change

    db = sqlite3.connect(tmp_path / "world.sqlite")
    db.executescript(
        "PRAGMA foreign_keys=ON; CREATE TABLE a(id TEXT PRIMARY KEY);"
        "CREATE TABLE b(id TEXT, a_id TEXT REFERENCES a(id));"
        "INSERT INTO a VALUES ('x'); INSERT INTO b VALUES ('1', 'x');"
    )
    db.close()

    assert "1 rows changed" in world_change(tmp_path, "INSERT INTO a VALUES ('y')")
    assert "refused" in world_change(tmp_path, "DELETE FROM a WHERE id = 'x'")
    assert "exactly one" in world_change(tmp_path, "DROP TABLE a")
    assert "a: 2, b: 1" in world_change(tmp_path, "UPDATE b SET id = '2'")


def test_meaning_changes_wait_for_a_yes_to_that_call_but_surface_edits_do_not(tmp_path, monkeypatch):
    from fi.alk.harness.backends import ToolSpec
    from fi.alk.harness.hosted_chat_entrypoint import _CONFIRM

    client = BuilderClient()
    runtime = _runtime(tmp_path, client, monkeypatch)
    (tmp_path / "scenarios" / "one" / "scenario.json").write_text(
        json.dumps({"name": "one", "instruction": "ask for a refund", "persona": {"name": "Ana"}})
    )
    (tmp_path / "sub_goals.json").write_text(json.dumps({"sub_goals": [{"name": "refunded", "description": "x"}]}))
    calls, asked, answers = [], [], []

    async def handler(args):
        calls.append(args)
        return {"content": [{"type": "text", "text": "kept"}]}

    async def ask(tool, payload, context):
        asked.append(payload["question"])
        return {"content": [{"type": "text", "text": answers.pop(0)}]}

    runtime._ask = ask
    submit = runtime._confirmed(ToolSpec("submit_scenario", "", {}, handler))
    drop = runtime._confirmed(ToolSpec("drop_scenarios", "", {}, handler))
    goal = runtime._confirmed(ToolSpec("add_sub_goal", "", {}, handler))

    surface = {"name": "one", "instruction": "ask for a refund", "persona": {"name": "Ana", "accent": "Indian"}}
    meaning = {"name": "one", "instruction": "ask for a cancellation", "persona": {"name": "Ana"}}
    assert asyncio.run(submit.handler(surface)).get("is_error") is None
    assert asyncio.run(submit.handler({"name": "two", "instruction": "anything"})).get("is_error") is None
    assert asyncio.run(goal.handler({"name": "new-goal"})).get("is_error") is None
    assert asked == []

    answers.extend(["No, keep it", "Which scenarios use it?"])
    assert asyncio.run(drop.handler({"names": ["one"]}))["is_error"] is True
    assert asyncio.run(submit.handler(meaning))["is_error"] is True
    answers.append(_CONFIRM)
    assert asyncio.run(goal.handler({"name": "refunded"})).get("is_error") is None
    answers.append("No")
    assert asyncio.run(drop.handler({"names": ["one"]}))["is_error"] is True
    assert "drop_scenarios" in asked[0] and "add_sub_goal" in asked[2]

    runtime._note_health("one is broken")
    answers.append("No")
    assert asyncio.run(submit.handler(meaning))["is_error"] is True
    assert asyncio.run(submit.handler(surface)).get("is_error") is None
    assert len(calls) == 5


def test_the_harness_builds_from_an_empty_workspace_with_the_tools_for_what_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("ALK_HARNESS", "claude")
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "schema_version: 1\nharness_chat:\n  enabled: true\n  stages:\n    builder:\n      enabled: true\n"
        "  lifecycle:\n    request_user_input: true\n",
        encoding="utf-8",
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "agent.py").write_text("def answer():\n    return 'hi'\n", encoding="utf-8")
    workspace = tmp_path / "authoring"
    workspace.mkdir()

    class LiveBuildClient(BuilderClient):
        def __init__(self):
            super().__init__()
            self.turn_context = {**self.capabilities.turn_context, "capabilities": {"control_only": True}}

        async def overview(self):
            return {}

    client = LiveBuildClient()
    runtime = HostedChatRuntime(
        client=client,  # type: ignore[arg-type]
        policy=load_chat_policy(policy),
        workspace=workspace,
        source=source,
        job={"metadata": {"agent_name": "demo"}, "scenario_count": 3},
    )
    assert isinstance(runtime.conversation, BuilderConversation)
    stage = asyncio.run(runtime._open_builder())
    granted = set(stage.spec.granted())
    assert "mcp__platform__finish_build" in granted and "mcp__knowledge__read_skill" in granted
    assert any(name.endswith("submit_contract") for name in granted)
    assert "Sub-goals are shared" in stage.spec.system_prompt or "shared sub-goals" in stage.spec.system_prompt

    finish = next(one for one in stage.spec.servers["platform"].tools if one.name == "finish_build")
    refused = asyncio.run(finish.handler({}))
    assert refused["is_error"] and "no contract is saved" in refused["content"][0]["text"]

    from fi.alk.harness import builder, scenario_tools

    monkeypatch.setattr(runtime, "_what_exists", lambda: (True, True))
    monkeypatch.setattr(scenario_tools, "load_scenarios", lambda path: ["one", "two"])
    monkeypatch.setattr(builder, "suite_health", lambda *args, **kwargs: [])
    short = asyncio.run(finish.handler({}))
    assert short["is_error"] and "2 of the 3 scenarios" in short["content"][0]["text"]
    assert not (workspace / ".futureagi-build-finished.json").exists()
    monkeypatch.setattr(scenario_tools, "load_scenarios", lambda path: ["one", "two", "three", "four"])
    over = asyncio.run(finish.handler({}))
    assert over["is_error"] and "the build asked for 3" in over["content"][0]["text"]
    assert not (workspace / ".futureagi-build-finished.json").exists()


    async def commands():
        return [{"sequence": 7, "message_id": "m7", "kind": "user_message", "payload": {"content": "add a lapsed policy"}}]

    client.commands = commands
    stopped = []

    class Working:
        async def interrupt_response(self):
            stopped.append(True)
            return True

        async def close(self):
            pass

    runtime.conversation = Working()

    async def building():
        while not stopped:
            await asyncio.sleep(0.05)

    yielded = asyncio.run(runtime._turn_until_interrupted(building(), command_sequence=1))
    assert yielded["message_id"] == "m7" and stopped
    # Answered next as its own turn, so it is not acknowledged here.
    assert runtime._turn_acknowledged_through == 0



def test_opening_a_chat_on_a_built_environment_does_not_reprove_the_whole_suite(tmp_path, monkeypatch):
    import pytest

    from fi.alk.harness import builder
    from fi.alk.harness.backends import ToolServer
    from fi.alk.harness.contract import AgentContract

    class Opened(Exception):
        pass

    def health(*args, **kwargs):
        raise Opened(kwargs)

    empty = ToolServer(name="stub", version="1", tools=[])
    monkeypatch.setattr(builder, "load_scenarios", lambda out: ["saved"])
    monkeypatch.setattr(builder, "scenario_tools", lambda *args, **kwargs: (empty, ["saved"]))
    monkeypatch.setattr(builder, "knowledge_server", lambda *args, **kwargs: empty)
    monkeypatch.setattr(builder, "suite_server", lambda *args, **kwargs: empty)
    monkeypatch.setattr(builder, "suite_health", health)
    with pytest.raises(Opened) as opened:
        builder.open_builder(AgentContract(agent_name="demo"), out=tmp_path, source_root=None)
    assert not opened.value.args[0].get("prove_all")


def _job(source_kind: str, connector: str, mode: str, config: dict) -> dict:
    return {
        "job_id": "job-1",
        "run_id": "run-1",
        "execution": "hosted",
        "source": {"kind": source_kind, "visibility": "public"}
        if source_kind == "provider"
        else {"kind": "archive", "archive_artifact_id": "source-1"},
        "agent": {
            "connector": connector,
            "mode": mode,
            "config": config,
            "secret_refs": {
                "VAPI_API_KEY": {"manager": "platform-vault", "key": "vapi-key", "purpose": "target_provider"}
            }
            if mode == "provider_import"
            else {},
        },
        "scenario_count": 1,
        "runtime": {"isolation": "dedicated_vm", "cpu_units": 2, "memory_mb": 4096, "parallelism": 1},
        "metadata": {"agent_name": "demo"},
    }


def _opened_source(tmp_path: Path, monkeypatch, job: dict):
    opened = {}
    monkeypatch.setattr(
        "fi.alk.harness.builder.open_for_building",
        lambda contract, **kwargs: opened.update(kwargs),
    )
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    runtime = _runtime(tmp_path / "authoring", BuilderClient(), monkeypatch)
    runtime.source = source
    runtime.job = job
    runtime._open_for_building(None)
    return opened["source"]


def test_a_phone_target_opens_the_builder_on_its_supplied_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("ALK_PROVIDER_IMPORT_PROFILE_PATH", "")
    monkeypatch.delenv("ALK_PROVIDER_IMPORT_PROFILE_PATH")
    job = _job(
        "provider",
        "phone",
        "connect_only",
        {"phone_number": "+14155551234", "target_system_prompt": "Answer ride questions."},
    )

    source = _opened_source(tmp_path, monkeypatch, job)

    assert source.kind == "provider"
    assert "Answer ride questions." in source.briefing()


def test_an_imported_provider_reaches_the_builder_from_the_build_process_cache(
    tmp_path, monkeypatch
):
    import os

    from fi.alk.harness.understand import _provider_import_briefing

    monkeypatch.setenv("ALK_PROVIDER_IMPORT_PROFILE_PATH", "")
    monkeypatch.delenv("ALK_PROVIDER_IMPORT_PROFILE_PATH")
    (tmp_path / "provider-import-profile.json").write_text(
        json.dumps({"provider": "vapi", "system_prompt": "Imported assistant prompt."})
    )
    job = _job("archive", "vapi", "provider_import", {"assistant_id": "asst-1"})

    source = _opened_source(tmp_path, monkeypatch, job)

    assert source.kind == "repo"
    assert os.environ["ALK_PROVIDER_IMPORT_PROFILE_PATH"] == str(tmp_path / "provider-import-profile.json")
    assert "Imported assistant prompt." in _provider_import_briefing()


def test_a_repository_opens_the_builder_on_its_source(tmp_path, monkeypatch):
    monkeypatch.setenv("ALK_PROVIDER_IMPORT_PROFILE_PATH", "")
    monkeypatch.delenv("ALK_PROVIDER_IMPORT_PROFILE_PATH")
    job = _job("archive", "livekit", "connect_only", {})

    source = _opened_source(tmp_path, monkeypatch, job)

    assert source.kind == "repo"
    assert "ALK_PROVIDER_IMPORT_PROFILE_PATH" not in __import__("os").environ


def test_a_retried_run_start_sends_the_same_operation_id(monkeypatch):
    from types import SimpleNamespace

    body = _capabilities().model_dump(mode="json")
    body["endpoints"]["run_start"] = body["endpoints"]["run_status"].replace("run-status", "run-start")
    sent = []

    class Transport:
        def request(self, method, url, **kwargs):
            sent.append(kwargs.get("json_body"))
            return SimpleNamespace(status_code=503 if len(sent) == 1 else 200, body={"run_id": "r-1"})

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    client = ConversationClient(ConversationCapabilities.model_validate(body), transport=Transport())

    asyncio.run(client.start_run([1], [], 1))

    assert len(sent) == 2 and sent[0]["operation_id"] == sent[1]["operation_id"]


def test_a_sub_agent_replacing_a_shared_sub_goal_waits_for_a_yes(tmp_path, monkeypatch):
    from fi.alk.harness.backends import SessionSpec, ToolSpec, WorkerSpec, tool_server

    runtime = _runtime(tmp_path, BuilderClient(), monkeypatch)
    (tmp_path / "sub_goals.json").write_text(json.dumps({"sub_goals": [{"name": "refunded", "description": "x"}]}))
    ran, asked = [], []

    async def handler(args):
        ran.append(args)
        return {"content": [{"type": "text", "text": "kept"}]}

    async def ask(tool, payload, context):
        asked.append(tool)
        return {"content": [{"type": "text", "text": "No"}]}

    runtime._ask = ask
    server = tool_server(name="scenarios", version="1", tools=[ToolSpec("add_sub_goal", "", {}, handler)])
    spec = SessionSpec(
        system_prompt="", servers={"scenarios": server},
        workers={"scenario_writer": WorkerSpec(description="", instructions="", servers={"scenarios": server})},
    )
    runtime._guard(spec)
    replace_goal = spec.workers["scenario_writer"].servers["scenarios"].tools[0].handler

    assert asyncio.run(replace_goal({"name": "refunded"}))["is_error"] is True
    assert asyncio.run(replace_goal({"name": "a-new-goal"})).get("is_error") is None
    assert asked == ["add_sub_goal"] and ran == [{"name": "a-new-goal"}]


def test_file_tools_on_any_backend_never_return_credentials(tmp_path):
    from fi.alk.harness.backends.files import file_tools
    from fi.alk.harness.backends.vertex_gemini import _spec_tool

    work = tmp_path / "work"
    (work / "scenarios").mkdir(parents=True)
    (work / "scenarios" / "a.json").write_text("SENTINEL-free")
    (work / ".env").write_text("SENTINEL=1")
    (work / "service_account.json").write_text('{"SENTINEL": 1}')
    tools = {spec.name: _spec_tool(spec.name, spec) for spec in file_tools(str(work / "scenarios"))}

    def run(name, **args):
        result = asyncio.run(tools[name].run_async(args=args, tool_context=None))
        return result["content"][0]["text"]

    assert "SENTINEL=1" not in run("Read", file_path="../.env")
    assert "SENTINEL" not in run("Read", file_path=str(work / "service_account.json")).replace("SENTINEL-free", "")
    found = run("Grep", pattern="SENTINEL", path="..")
    assert "SENTINEL-free" in found and "SENTINEL=1" not in found and '{"SENTINEL' not in found
    assert ".env" not in run("Glob", pattern="*", path="..")
    assert "a.json" in run("Glob", pattern="*.json")
    assert "kept out of this session" in run("Grep", pattern="x", path="/")


class RebuildClient(BuilderClient):
    def __init__(self, workspace: Path):
        super().__init__()
        self.workspace = workspace
        self.results: list[dict | None] = []

    async def overview(self):
        return {}

    async def checkpoint(self, workspace):
        from fi.alk.harness.nudge import NUDGE_RESULT

        path = self.workspace / NUDGE_RESULT
        self.results.append(json.loads(path.read_text()) if path.is_file() else None)
        return await super().checkpoint(workspace)


class RebuildConversation(BuilderConversation):
    """A builder whose model turn is ``act``, run against the stage the runtime opens."""

    def __init__(self, runtime: HostedChatRuntime, act):
        super().__init__(runtime._open_builder)
        self.act = act
        self.messages: list[str] = []

    async def say(self, message, on_event=None):
        self.messages.append(message)
        self.stage = await self._open_stage(resume=True)
        self.stage.emit_event = on_event
        await self.act(self.stage)
        on_event(
            RuntimeEvent(kind="text", text="Updated.", detail={"stage": "builder"})
        )

    async def resume(self, invocation_id, on_event=None):
        await self.say("", on_event)


def _rebuild(
    tmp_path: Path,
    monkeypatch,
    stage: str,
    act,
    *,
    stale_result: bool = False,
    draft_keys: tuple[str, ...] = (),
    journal: dict | None = None,
    redeliver: bool = False,
    resumed: bool = False,
):
    from test_nudge_turn import _suite

    from fi.alk.harness import spend

    monkeypatch.setenv("ALK_HARNESS", "vertex-gemini" if resumed else "claude")
    monkeypatch.setattr(spend, "_stages", {})
    workspace = tmp_path / "work"
    workspace.mkdir()
    _suite(workspace, "adds-a-big-mac")
    if stale_result:
        from fi.alk.harness.nudge import NUDGE_RESULT

        (workspace / NUDGE_RESULT).write_text('{"nudge_id": "an-older-turn"}')
    if journal is not None:
        (workspace / ".futureagi-conversation.json").write_text(json.dumps(journal))
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "schema_version: 1\nharness_chat:\n  enabled: true\n  stages:\n    builder:\n"
        "      enabled: true\n",
        encoding="utf-8",
    )
    client = RebuildClient(workspace)
    if resumed:
        client.turn_context = {**client.turn_context, "active_invocation_id": "inv-1"}
    runtime = HostedChatRuntime(
        client=client,  # type: ignore[arg-type]
        policy=load_chat_policy(policy),
        workspace=workspace,
        source=tmp_path / "source",
        job={"metadata": {"agent_name": "cart"}, "scenario_count": 1},
    )
    asked: list[str] = []

    async def ask(tool, payload, context):
        asked.append(tool)
        return {"content": [{"type": "text", "text": "No"}]}

    runtime._ask = ask
    conversation = RebuildConversation(runtime, act)
    runtime.conversation = conversation
    command = {
        "sequence": 3,
        "message_id": "m-3",
        "kind": "nudge",
        "payload": {
            "content": "The agent moved to v2: add was renamed.",
            "nudge_id": "n-1",
            "stage": stage,
            "user_draft_keys": list(draft_keys),
        },
    }
    asyncio.run(runtime._handle(command))
    if redeliver:
        # The turn_completed was lost: the same command reaches the restored workspace.
        client.redelivered_at = len(client.events)
        asyncio.run(runtime._handle(command))
    return runtime, client, conversation, asked


def _tool(stage, server: str, name: str):
    return next(one for one in stage.spec.servers[server].tools if one.name == name)


def _called(stage, name: str) -> None:
    stage.emit_event(
        RuntimeEvent(kind="tool", tool=f"mcp__x__{name}", detail={"call_id": name})
    )


def test_an_environment_rebuild_turn_runs_its_own_tools_and_ends_the_sandbox(
    tmp_path, monkeypatch
):
    from fi.alk.harness import spend
    from fi.alk.harness.catalogue import load_catalogue
    from fi.alk.harness.nudge import NUDGE_RESULT
    from fi.alk.harness.source_data_invariants import ARTIFACT

    seen: dict = {}

    async def act(stage):
        seen["result_at_start"] = (tmp_path / "work" / NUDGE_RESULT).exists()
        seen["granted"] = set(stage.spec.granted())
        seen["workers"] = dict(stage.spec.workers)
        seen["rule"] = await _tool(stage, "scenarios", "add_rule").handler(
            {"rule": "Always read the item back.", "why": "v2 says so in agent.py"}
        )
        await _tool(stage, "nudge", "retire_sub_goal").handler(
            {"name": "right-item", "why": "gone in v2"}
        )
        seen["added"] = await _tool(stage, "world", "add_sub_goal").handler(
            {
                "name": "cart-exists",
                "what": "the cart table is still there",
                "check": "def check(world, calls):\n    return None if 'cart' in world.state() else 'no cart'\n",
            }
        )
        seen["added_too"] = await _tool(stage, "scenarios", "add_sub_goal").handler(
            {
                "name": "one-row",
                "what": "the cart holds at most one row",
                "check": "def check(world, calls):\n    return None if len(world.state()['cart']) <= 1 else 'more'\n",
            }
        )
        spend.record(
            "builder", 0.25, turns=1, models={"m2"}, tokens_in=40, tokens_out=6
        )
        (tmp_path / "source").mkdir()
        (tmp_path / "source" / "agent.py").write_text("def add_item(item_id):\n")
        review = {
            "name": "items-exist",
            "evidence": [{"path": "agent.py", "quote": "def add("}],
        }
        (tmp_path / "work" / ARTIFACT).write_text(json.dumps({"checks": [review]}))
        await _tool(stage, "nudge", "report_update").handler(
            {"changed": True, "notes": "rule added"}
        )

    runtime, client, conversation, asked = _rebuild(
        tmp_path, monkeypatch, "environment", act, stale_result=True
    )

    granted = seen["granted"]
    assert seen["result_at_start"] is False
    assert {
        "mcp__nudge__report_update",
        "mcp__nudge__retire_sub_goal",
        "mcp__world__save_world",
    } <= granted
    assert (
        not {
            "mcp__scenarios__submit_scenario",
            "mcp__platform__save_changes",
            "Delegate",
        }
        & granted
    )
    # One world writer: change_world would bypass the world save_world keeps.
    assert not any(name.endswith("__change_world") for name in granted)
    assert seen["workers"] == {}
    # No confirmation in a rebuild turn: nobody is there to answer.
    assert asked == [] and not seen["rule"].get("is_error"), seen["rule"]
    assert not seen["added"].get("is_error"), seen["added"]
    assert not seen["added_too"].get("is_error"), seen["added_too"]
    assert load_catalogue(tmp_path / "work").names() == {
        "item-added",
        "cart-exists",
        "one-row",
    }
    assert "add was renamed" in conversation.messages[0]
    assert "builder/agent-update" in conversation.messages[0]

    kinds = [one["kind"] for one in client.events]
    committed = kinds.index("checkpoint_committed")
    completed = client.events[-1]
    assert kinds[-1] == "turn_completed" and committed < len(kinds) - 1
    assert client.events[committed]["invocation_id"] == completed["invocation_id"]
    result = client.results[-1]
    assert result["nudge_id"] == "n-1" and result["checks"]["passed"]
    # The agent's source at /work/source is what the stale-review warning reads.
    assert "items-exist" in result["checks"]["warnings"][0]
    assert client.published, "the turn's changes are published before turn_completed"
    assert completed["payload"]["outcome"] == "success"
    assert completed["payload"]["usage"] == {
        "tokens_in": 40,
        "tokens_out": 6,
        "tokens_cached": 0,
        "usd": 0.25,
        "unpriced_turns": 0,
        "models": ["m2"],
    }
    assert runtime.stopping.is_set()


def test_a_suite_rebuild_turn_cannot_amend_the_contract_or_rename(
    tmp_path, monkeypatch
):
    from test_nudge_turn import _scenario

    from fi.alk.harness.proofs import PROOFS, load_proofs

    seen: dict = {}

    async def act(stage):
        seen["granted"] = set(stage.spec.granted())
        (tmp_path / "work" / PROOFS).unlink(missing_ok=True)
        seen["renamed"] = await _tool(stage, "scenarios", "submit_scenario").handler(
            _scenario("orders-a-big-mac", scenario_key="adds-a-big-mac")
        )
        await _tool(stage, "nudge", "report_update").handler(
            {"changed": False, "notes": "fits"}
        )

    runtime, client, _conversation, asked = _rebuild(
        tmp_path, monkeypatch, "suite", act
    )

    granted = seen["granted"]
    assert {"mcp__scenarios__submit_scenario", "mcp__suite__prove_scenarios"} <= granted
    assert not {"mcp__scenarios__amend_contract", "mcp__world__save_world"} & granted
    assert (
        seen["renamed"]["is_error"]
        and "rename" in seen["renamed"]["content"][0]["text"]
    )
    assert load_proofs(tmp_path / "work")["adds-a-big-mac"]["holds"] is True
    assert client.results[-1]["stage"] == "suite" and asked == []
    assert client.events[-1]["payload"]["usage"]["tokens_in"] == 0
    assert runtime.stopping.is_set()


def test_a_failed_rebuild_turn_still_reports_usage_and_ends_the_sandbox(
    tmp_path, monkeypatch
):
    from fi.alk.harness import spend

    async def act(stage):
        spend.record("builder", 0.1, turns=1, models={"m2"}, tokens_in=5)
        raise RuntimeError("model unavailable")

    runtime, client, _conversation, _asked = _rebuild(
        tmp_path, monkeypatch, "environment", act
    )

    completed = client.events[-1]
    assert (
        completed["kind"] == "turn_completed"
        and completed["payload"]["outcome"] == "failed"
    )
    assert completed["payload"]["usage"]["tokens_in"] == 5
    assert runtime.stopping.is_set()


def test_a_chat_turn_carries_no_usage_and_keeps_the_sandbox(tmp_path, monkeypatch):
    client = BuilderClient()
    runtime = _runtime(tmp_path, client, monkeypatch)
    runtime.conversation = EditingConversation(tmp_path, writes=False)

    _turn(runtime, {"content": "hello"})

    assert "usage" not in client.events[-1]["payload"]
    assert not runtime.stopping.is_set()


def test_prove_scenarios_records_each_proof(tmp_path):
    from test_nudge_turn import _suite

    from fi.alk.harness.proofs import PROOFS, load_proofs
    from fi.alk.harness.scenario_tools import load_scenarios

    root = _suite(tmp_path, "adds-a-big-mac", "adds-one-big-mac")
    (root / PROOFS).unlink(missing_ok=True)
    server = builder.suite_server(root, load_scenarios(root))
    prove = next(one for one in server.tools if one.name == "prove_scenarios")

    said = asyncio.run(prove.handler({"names": ["adds-a-big-mac", "not-there"]}))

    proofs = load_proofs(root)
    assert "adds-a-big-mac: holds" in said["content"][0]["text"]
    assert set(proofs) == {"adds-a-big-mac"}
    assert (
        proofs["adds-a-big-mac"]["holds"] is True
        and proofs["adds-a-big-mac"]["proved_at"]
    )


def test_a_write_after_the_last_report_runs_the_checks_again(tmp_path, monkeypatch):
    from test_nudge_turn import _break_scenario

    async def act(stage):
        _called(stage, "report_update")
        await _tool(stage, "nudge", "report_update").handler(
            {"changed": False, "notes": "fits"}
        )
        _called(stage, "submit_scenario")
        _break_scenario(tmp_path / "work", "adds-a-big-mac")
        _called(stage, "find_scenarios")

    _runtime, client, _conversation, _asked = _rebuild(
        tmp_path, monkeypatch, "suite", act
    )

    result = client.results[-1]
    assert result["checks"]["passed"] is False
    assert [one["item"] for one in result["checks"]["problems"]] == ["adds-a-big-mac"]
    assert result["changed"] is True and result["notes"] == (
        "fits (changed after this report; checks re-run at the end of the turn)"
    )


def test_a_tool_added_without_saving_the_world_fails_the_report(
    tmp_path, monkeypatch
):
    from test_nudge_turn import _add_tool

    async def act(stage):
        _add_tool(tmp_path / "work", "send_otp")
        await _tool(stage, "nudge", "report_update").handler(
            {"changed": True, "notes": "added send_otp"}
        )

    _runtime, client, _conversation, _asked = _rebuild(
        tmp_path, monkeypatch, "environment", act
    )

    result = client.results[-1]
    assert result["checks"]["passed"] is False
    # The turn added it, so it also needs a sub-goal checking it.
    assert [(one["item"], one["kind"]) for one in result["checks"]["problems"]] == [
        ("send_otp", "world"),
        ("send_otp", "tool"),
    ]


def test_world_changes_fail_the_report_until_the_world_is_saved(tmp_path, monkeypatch):
    import test_nudge_turn

    from fi.alk.harness.simulator import save_simulator_prompt

    suite = test_nudge_turn._suite

    def _checked_suite(root: Path, *names: str) -> Path:
        suite(root, *names)
        test_nudge_turn._set_manifest(
            root,
            world_checks={
                "menu-has-items": "def check(world):\n"
                "    return None if world.state().get('menu') else 'no menu'\n"
            },
        )
        save_simulator_prompt(
            "You are a customer described by {{ persona }}. Follow {{ instruction }} in a "
            "natural multi-turn conversation. Never invent the assistant's responses or claim "
            "a tool result you have not heard. Ask a concise follow-up when needed and end "
            "once the task resolves.",
            root,
        )
        return root

    monkeypatch.setattr(test_nudge_turn, "_suite", _checked_suite)
    seen: dict = {}

    async def act(stage):
        seeded = await _tool(stage, "world", "seed").handler(
            {"table": "menu", "rows": [{"id": "fries"}]}
        )
        assert not seeded.get("is_error"), seeded
        seen["unsaved"] = await _tool(stage, "nudge", "report_update").handler(
            {"changed": True, "notes": "seeded fries"}
        )
        seen["saved"] = await _tool(stage, "world", "save_world").handler({})
        await _tool(stage, "nudge", "report_update").handler(
            {"changed": True, "notes": "seeded fries"}
        )

    _runtime, client, _conversation, _asked = _rebuild(
        tmp_path, monkeypatch, "environment", act
    )

    assert seen["unsaved"].get("is_error")
    assert '"kind": "world"' in seen["unsaved"]["content"][0]["text"]
    assert not seen["saved"].get("is_error"), seen["saved"]
    assert client.results[-1]["checks"]["passed"] is True


_OLD_SEED_CHECKS = {
    "seed_matches_source": "def check(world):\n"
    "    ids = sorted(row['id'] for row in world.state().get('menu') or [])\n"
    "    return None if ids == ['big_mac'] else f'menu is {ids}'\n",
    "key_balances": "def check(world):\n"
    "    rows = world.state().get('wallet') or []\n"
    "    dana = [row['balance'] for row in rows if row['rider'] == 'dana']\n"
    "    return None if dana == [60.0] else f'dana holds {dana}'\n",
}
_NEW_SEED = (
    "INSERT INTO menu (id) VALUES ('big_mac');\n"
    "INSERT INTO menu (id) VALUES ('fries');\n"
    "INSERT INTO wallet (rider, balance) VALUES ('dana', 5.00);\n"
)


def _seed_rebuild(tmp_path, monkeypatch, act, checks=_OLD_SEED_CHECKS):
    """An environment rebuild whose world checks pin the old seed: big_mac only, dana 60."""
    import test_nudge_turn

    from fi.alk.harness.simulator import save_simulator_prompt
    from fi.alk.harness.world.snapshot import read_manifest, restore, save

    suite = test_nudge_turn._suite

    def _old_seed_suite(root: Path, *names: str) -> Path:
        suite(root, *names)
        world = restore(root)
        world.connection.executescript(
            "CREATE TABLE wallet(rider TEXT PRIMARY KEY, balance REAL);"
            "INSERT INTO wallet VALUES ('dana', 60.0);"
        )
        world.connection.commit()
        save(
            world,
            root,
            notes="old seed",
            sequences=list(read_manifest(root).get("sequences") or []),
            world_checks=checks,
        )
        save_simulator_prompt(
            "You are a customer described by {{ persona }}. Follow {{ instruction }} in a "
            "natural multi-turn conversation. Never invent the assistant's responses or claim "
            "a tool result you have not heard. Ask a concise follow-up when needed and end "
            "once the task resolves.",
            root,
        )
        return root

    monkeypatch.setattr(test_nudge_turn, "_suite", _old_seed_suite)
    return _rebuild(tmp_path, monkeypatch, "environment", act)


def _reseed(stage):
    async def run():
        seeded = await _tool(stage, "world", "seed").handler(
            {"table": "menu", "rows": [{"id": "fries"}]}
        )
        changed = await _tool(stage, "world", "change_data").handler(
            {"sql": "UPDATE wallet SET balance = 5.0 WHERE rider = 'dana'"}
        )
        assert not seeded.get("is_error") and not changed.get("is_error"), changed

    return run()


def test_a_seed_change_rewrites_the_world_checks_that_pin_the_old_seed(
    tmp_path, monkeypatch
):
    from fi.alk.harness.world.snapshot import read_manifest

    seen: dict = {}
    menu = (
        "def check(world):\n"
        "    ids = sorted(row['id'] for row in world.state().get('menu') or [])\n"
        "    return None if ids == ['big_mac', 'fries'] else f'menu is {ids}'\n"
    )
    balances = _OLD_SEED_CHECKS["key_balances"].replace("[60.0]", "[5.0]")

    async def act(stage):
        (tmp_path / "source" / "db").mkdir(parents=True)
        (tmp_path / "source" / "db" / "seed.sql").write_text(_NEW_SEED)
        seen["granted"] = set(stage.spec.granted())
        await _reseed(stage)
        seen["refused"] = await _tool(stage, "world", "save_world").handler({})
        for name, code, quote in (
            ("seed_matches_source", menu, "INSERT INTO menu (id) VALUES ('fries');"),
            ("key_balances", balances, "VALUES ('dana', 5.00);"),
        ):
            edited = await _tool(stage, "world", "add_world_check").handler(
                {
                    "name": name,
                    "code": code,
                    "source_path": "db/seed.sql",
                    "source_quote": quote,
                }
            )
            assert not edited.get("is_error"), edited
        seen["saved"] = await _tool(stage, "world", "save_world").handler({})
        seen["report"] = await _tool(stage, "nudge", "report_update").handler(
            {"changed": True, "notes": "new seed rows"}
        )

    _runtime, client, _conversation, _asked = _seed_rebuild(tmp_path, monkeypatch, act)

    assert "mcp__world__add_world_check" in seen["granted"]
    refused = seen["refused"]["content"][0]["text"]
    assert seen["refused"].get("is_error")
    assert "key_balances" in refused and "seed_matches_source" in refused
    assert not seen["saved"].get("is_error"), seen["saved"]
    assert not seen["report"].get("is_error"), seen["report"]
    checks = read_manifest(tmp_path / "work")["world_checks"]
    assert checks == {"seed_matches_source": menu, "key_balances": balances}
    result = client.results[-1]
    assert result["checks"]["passed"] is True
    assert result["checks"]["warnings"] == [
        "world checks changed: key_balances (cites db/seed.sql), "
        "seed_matches_source (cites db/seed.sql)"
    ]


def test_a_world_check_edit_that_cites_no_seed_change_is_refused(tmp_path, monkeypatch):
    from fi.alk.harness.world.snapshot import read_manifest

    seen: dict = {}
    weakened = (
        "def check(world):\n    return None if world.state().get('menu') else 'no'\n"
    )

    async def act(stage):
        (tmp_path / "source" / "db").mkdir(parents=True)
        (tmp_path / "source" / "db" / "seed.sql").write_text(
            "INSERT INTO menu (id) VALUES ('big_mac');\n"
        )
        await _reseed(stage)
        tool = _tool(stage, "world", "add_world_check")
        seen["uncited"] = await tool.handler(
            {"name": "seed_matches_source", "code": weakened}
        )
        seen["unchanged"] = await tool.handler(
            {
                "name": "seed_matches_source",
                "code": weakened,
                "source_path": "db/seed.sql",
                "source_quote": "INSERT INTO menu (id) VALUES ('fries');",
            }
        )
        seen["saved"] = await _tool(stage, "world", "save_world").handler({})
        await _tool(stage, "nudge", "report_update").handler(
            {"changed": True, "notes": "fries"}
        )

    _runtime, client, _conversation, _asked = _seed_rebuild(tmp_path, monkeypatch, act)

    assert seen["uncited"].get("is_error") and seen["unchanged"].get("is_error")
    assert "does not occur in db/seed.sql" in seen["unchanged"]["content"][0]["text"]
    assert seen["saved"].get("is_error")
    assert read_manifest(tmp_path / "work")["world_checks"] == _OLD_SEED_CHECKS
    result = client.results[-1]
    assert result["checks"]["passed"] is False
    assert not result["checks"]["warnings"]


def test_a_rebuild_that_leaves_out_a_seeded_row_fails_until_it_is_seeded(
    tmp_path, monkeypatch
):
    seen: dict = {}

    async def act(stage):
        (tmp_path / "source" / "db").mkdir(parents=True)
        (tmp_path / "source" / "db" / "seed.sql").write_text(
            "INSERT INTO menu (id) VALUES ('big_mac');\n"
            "INSERT INTO bookings (ref, status) VALUES ('UB7F3A2C91', 'matched');\n"
        )
        schema = await _tool(stage, "world", "create_schema").handler(
            {"sql": "CREATE TABLE bookings(ref TEXT PRIMARY KEY, status TEXT);"}
        )
        assert not schema.get("is_error"), schema
        for key in ("left_out", "seeded"):
            if key == "seeded":
                await _tool(stage, "world", "seed").handler(
                    {
                        "table": "bookings",
                        "rows": [{"ref": "UB7F3A2C91", "status": "matched"}],
                    }
                )
            saved = await _tool(stage, "world", "save_world").handler({})
            assert not saved.get("is_error"), saved
            seen[key] = await _tool(stage, "nudge", "report_update").handler(
                {"changed": True, "notes": key}
            )

    _runtime, client, conversation, _asked = _seed_rebuild(tmp_path, monkeypatch, act)

    assert "every row the agent's current seed inserts" in conversation.messages[0]
    assert seen["left_out"].get("is_error")
    left_out = seen["left_out"]["content"][0]["text"]
    assert "UB7F3A2C91" in left_out and "db/seed.sql" in left_out
    assert not seen["seeded"].get("is_error"), seen["seeded"]
    assert client.results[-1]["checks"]["passed"] is True


def test_a_seed_row_in_a_table_the_old_world_kept_empty_is_reproduced(
    tmp_path, monkeypatch
):
    from fi.alk.harness.world.snapshot import read_manifest

    seen: dict = {}
    empty_cart = {
        "seed_matches_source": "def check(world):\n"
        "    st = world.state()\n"
        "    if sorted(row['id'] for row in st.get('menu') or []) != ['big_mac']:\n"
        "        return 'menu is not the seed menu'\n"
        "    return 'cart is not empty at the start' if st.get('cart') else None\n"
    }
    seeded_cart = (
        "def check(world):\n"
        "    st = world.state()\n"
        "    if sorted(row['id'] for row in st.get('menu') or []) != ['big_mac']:\n"
        "        return 'menu is not the seed menu'\n"
        "    cart = [row['item_id'] for row in st.get('cart') or []]\n"
        "    return None if cart == ['big_mac'] else f'cart is {cart}'\n"
    )

    async def act(stage):
        (tmp_path / "source" / "db").mkdir(parents=True)
        (tmp_path / "source" / "db" / "seed.sql").write_text(
            "INSERT INTO menu (id) VALUES ('big_mac');\n"
            "INSERT INTO cart (item_id) VALUES ('big_mac');\n"
        )
        seeded = await _tool(stage, "world", "seed").handler(
            {"table": "cart", "rows": [{"item_id": "big_mac"}]}
        )
        assert not seeded.get("is_error"), seeded
        seen["refused"] = await _tool(stage, "world", "save_world").handler({})
        edited = await _tool(stage, "world", "add_world_check").handler(
            {
                "name": "seed_matches_source",
                "code": seeded_cart,
                "source_path": "db/seed.sql",
                "source_quote": "INSERT INTO cart (item_id) VALUES ('big_mac');",
            }
        )
        assert not edited.get("is_error"), edited
        seen["saved"] = await _tool(stage, "world", "save_world").handler({})
        await _tool(stage, "nudge", "report_update").handler(
            {"changed": True, "notes": "seeded the open cart"}
        )

    _runtime, client, _conversation, _asked = _seed_rebuild(
        tmp_path, monkeypatch, act, checks=empty_cart
    )

    assert seen["refused"].get("is_error")
    assert "seed_matches_source" in seen["refused"]["content"][0]["text"]
    assert not seen["saved"].get("is_error"), seen["saved"]
    manifest = read_manifest(tmp_path / "work")
    assert manifest["tables"]["cart"] == 1
    assert manifest["world_checks"] == {"seed_matches_source": seeded_cart}
    assert client.results[-1]["checks"]["passed"] is True


def test_a_report_after_the_last_write_stands(tmp_path, monkeypatch):
    async def act(stage):
        _called(stage, "submit_scenario")
        _called(stage, "report_update")
        await _tool(stage, "nudge", "report_update").handler(
            {"changed": False, "notes": "fits"}
        )
        _called(stage, "check_health")

    _runtime, client, _conversation, _asked = _rebuild(
        tmp_path, monkeypatch, "suite", act
    )

    result = client.results[-1]
    assert result["checks"]["passed"] is True and result["changed"] is False


def test_a_late_write_whose_checks_cannot_run_lands_no_result(tmp_path, monkeypatch):
    from fi.alk.harness import nudge

    async def act(stage):
        await _tool(stage, "nudge", "report_update").handler(
            {"changed": True, "notes": "fits"}
        )
        _called(stage, "submit_scenario")

        def broken(*_args, **_kwargs):
            raise RuntimeError("gate failed")

        monkeypatch.setattr(nudge, "report_update", broken)

    _runtime, client, _conversation, _asked = _rebuild(
        tmp_path, monkeypatch, "suite", act
    )

    assert client.results[-1] is None
    assert client.events[-1]["payload"]["outcome"] == "success"


def test_a_suite_rebuild_is_told_which_scenarios_are_the_persons(tmp_path, monkeypatch):
    async def act(stage):
        return None

    _runtime, _client, conversation, _asked = _rebuild(
        tmp_path, monkeypatch, "suite", act, draft_keys=("adds-a-big-mac",)
    )

    assert conversation.messages[0].endswith(
        "Scenarios the person is still editing (keep what they wrote; only carry over a "
        "renamed tool or argument): adds-a-big-mac"
    )
    assert "No scenario is being edited" not in conversation.messages[0]


def test_a_suite_rebuild_with_no_draft_says_earlier_edits_have_landed(
    tmp_path, monkeypatch
):
    async def act(stage):
        return None

    _runtime, _client, conversation, _asked = _rebuild(
        tmp_path, monkeypatch, "suite", act
    )

    assert conversation.messages[0].endswith(
        "No scenario is being edited by the person now: no list of edits in an earlier "
        "message applies any more, so repair every scenario this update breaks."
    )


def test_an_environment_rebuild_says_nothing_about_edits(tmp_path, monkeypatch):
    async def act(stage):
        return None

    _runtime, _client, conversation, _asked = _rebuild(
        tmp_path, monkeypatch, "environment", act
    )

    assert "being edited" not in conversation.messages[0]
    assert "still editing" not in conversation.messages[0]


def test_a_suite_rebuild_can_add_a_scenario(tmp_path, monkeypatch):
    from test_nudge_turn import _scenario

    from fi.alk.harness.scenario_tools import load_scenarios

    seen: dict = {}

    async def act(stage):
        await _tool(stage, "scenarios", "aim_for").handler({"count": 2})
        seen["added"] = await _tool(stage, "scenarios", "submit_scenario").handler(
            _scenario("adds-one-big-mac")
        )

    _rebuild(tmp_path, monkeypatch, "suite", act)

    assert not seen["added"].get("is_error"), seen["added"]
    assert {one.name for one in load_scenarios(tmp_path / "work")} == {
        "adds-a-big-mac",
        "adds-one-big-mac",
    }


def test_a_replayed_rebuild_commits_the_workspace_holding_its_result(
    tmp_path, monkeypatch
):
    from fi.alk.harness.nudge import NUDGE_RESULT

    async def act(stage):
        raise AssertionError("a replayed turn does not run again")

    _runtime, client, _conversation, _asked = _rebuild(
        tmp_path,
        monkeypatch,
        "suite",
        act,
        stale_result=True,
        journal={"m-3": "completed"},
    )

    kinds = [one["kind"] for one in client.events]
    assert kinds[-2:] == ["checkpoint_committed", "turn_completed"]
    assert client.events[-1]["payload"]["replayed"] is True
    assert client.events[-2]["payload"]["digest"]
    assert client.results[-1] == {"nudge_id": "an-older-turn"}
    assert (tmp_path / "work" / NUDGE_RESULT).is_file()


@pytest.mark.parametrize("ending", ["raised", "turn_failures", "interrupted"])
def test_a_rebuild_turn_that_did_not_succeed_replays_as_failed(
    tmp_path, monkeypatch, ending
):
    async def act(stage):
        await _tool(stage, "nudge", "report_update").handler(
            {"changed": False, "notes": "fits"}
        )
        _called(stage, "submit_scenario")
        if ending == "raised":
            raise RuntimeError("model unavailable")
        if ending == "turn_failures":
            stage.emit_event(
                RuntimeEvent(kind="done", detail={"outcome": "failed", "error": "x"})
            )

    async def interrupted(self, turn, *, command_sequence):
        await turn
        return {"kind": "interrupt", "message_id": "m-4", "sequence": 4}

    if ending == "interrupted":
        monkeypatch.setattr(HostedChatRuntime, "_turn_until_interrupted", interrupted)

    _runtime, client, _conversation, _asked = _rebuild(
        tmp_path, monkeypatch, "suite", act, redeliver=True
    )

    journal = json.loads(
        (tmp_path / "work" / ".futureagi-conversation.json").read_text()
    )
    assert journal == {"m-3": "ended"}
    # The turn's last workspace holds no result, so nothing of it can land.
    assert client.results[-1] is None
    replay = client.events[client.redelivered_at :]
    assert [one["kind"] for one in replay] == ["turn_completed"]
    assert replay[0]["payload"] == {"command_message_id": "m-3", "outcome": "failed"}


def test_a_cut_off_rebuild_turn_keeps_no_result(tmp_path, monkeypatch):
    async def act(stage):
        raise AssertionError("a retried rebuild turn does not run again")

    _runtime, client, _conversation, _asked = _rebuild(
        tmp_path,
        monkeypatch,
        "suite",
        act,
        stale_result=True,
        journal={"m-3": "retried"},
    )

    journal = json.loads(
        (tmp_path / "work" / ".futureagi-conversation.json").read_text()
    )
    assert journal == {"m-3": "ended"} and client.results[-1] is None
    assert client.events[-1]["payload"]["interrupted"] is True


def test_a_resumed_rebuild_turn_starts_without_its_earlier_result(
    tmp_path, monkeypatch
):
    from fi.alk.harness.nudge import NUDGE_RESULT

    seen: dict = {}

    async def act(stage):
        seen["result_at_start"] = (tmp_path / "work" / NUDGE_RESULT).exists()
        _called(stage, "submit_scenario")

    _runtime, client, _conversation, _asked = _rebuild(
        tmp_path,
        monkeypatch,
        "suite",
        act,
        stale_result=True,
        journal={"m-3": "started"},
        resumed=True,
    )

    assert client.events[0]["payload"]["resumed"] is True
    # The report lived in the lost process: without a new one nothing lands.
    assert seen["result_at_start"] is False
    assert client.results[-1] is None


def test_aim_for_in_a_suite_rebuild_only_raises_the_count(tmp_path, monkeypatch):
    from fi.alk.harness.nudge import REBUILD_ADDS_AT_MOST, aim_refusal

    said: dict = {}

    async def act(stage):
        aim = _tool(stage, "scenarios", "aim_for").handler
        said["alike"] = await aim({"count": 2, "people": "alike"})
        said["asked"] = await aim({"count": 2, "asked_for": {"task": ["add"]}})
        said["far"] = await aim({"count": 2 + REBUILD_ADDS_AT_MOST})
        said["near"] = await aim({"count": 1 + REBUILD_ADDS_AT_MOST})

    _rebuild(tmp_path, monkeypatch, "suite", act)

    for refused in ("alike", "asked", "far"):
        assert said[refused].get("is_error"), said[refused]
    assert not said["near"].get("is_error"), said["near"]
    assert aim_refusal({"count": 2}, 3, 3)


@pytest.mark.parametrize("kind", ["builder", "coordinator"])
def test_a_stored_session_the_provider_will_not_resume_opens_fresh(kind):
    from fi.alk.harness.backends.base import ConversationSession, SessionSpec
    from fi.alk.harness.hosted_chat_entrypoint import CoordinatorConversation
    from fi.alk.harness.session import Stage

    starts: list = []

    class Session:
        def __init__(self, spec):
            self.resuming = spec.conversation.resume_session_id

        async def start(self):
            starts.append(self.resuming)
            if self.resuming:
                raise RuntimeError(f"No conversation found with session ID {self.resuming}")

        async def stop(self):
            pass

    class Backend:
        def create(self, spec):
            return Session(spec)

    conversation = ConversationSession(
        app_name="alk", user_id="u", session_id="s", resume_session_id="sdk-1"
    )
    stage = Stage(
        SessionSpec(system_prompt="", conversation=conversation),
        name=kind,
        backend=Backend(),
    )

    async def say(message, on_event=None):
        return f"heard {message}"

    stage.say = say  # type: ignore[method-assign]

    async def open_stage(*, resume=True):
        return stage

    chat = (
        BuilderConversation(open_stage)
        if kind == "builder"
        else CoordinatorConversation(stage)
    )
    assert asyncio.run(chat.say("hi")) == "heard hi"
    assert starts == ["sdk-1", None]
    assert conversation.resume_session_id is None
