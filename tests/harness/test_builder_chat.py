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
    long_meaning = {**meaning, "setup_code": "def setup(world):\n" + "    pass\n" * 200 + "    world.done()"}
    answers.append("No")
    assert asyncio.run(submit.handler(long_meaning))["is_error"] is True
    assert "world.done()" in asked[-1]

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


def test_reading_a_run_can_wait_until_it_has_finished(tmp_path, monkeypatch):
    client = BuilderClient()
    states = iter(["running", "running", "completed"])

    async def runs(run_id, limit):
        if run_id == "unknown":
            return {"error": "no Run with that id belongs to this environment"}
        return {"run_id": run_id, "state": next(states), "calls": []}

    async def no_sleep(_delay):
        return None

    client.runs = runs
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    runtime = _runtime(tmp_path, client, monkeypatch)
    read = next(spec for spec in runtime._builder_platform_server().tools if spec.name == "read_runs").handler

    result = asyncio.run(read({"run_id": "r-1", "wait": True}))

    assert '"completed"' in result["content"][0]["text"]
    assert "no Run" in asyncio.run(read({"run_id": "unknown", "wait": True}))["content"][0]["text"]


def test_waiting_on_a_run_never_outlasts_the_turn_it_waits_in():
    from fi.alk.harness.hosted_chat_entrypoint import _RUN_POLL_SECONDS, _RUN_WAIT_SECONDS
    from fi.alk.harness.hosted_conversation import _UNAVAILABLE_BUDGET_SECONDS
    from fi.alk.harness.session import STAGE_IDLE_TIMEOUT_SECONDS

    assert _RUN_WAIT_SECONDS + _RUN_POLL_SECONDS + _UNAVAILABLE_BUDGET_SECONDS < STAGE_IDLE_TIMEOUT_SECONDS


def test_the_build_and_its_check_change_without_asking_and_a_question_gets_no_one(tmp_path, monkeypatch):
    from fi.alk.harness.backends import ToolSpec

    client = BuilderClient()
    runtime = _runtime(tmp_path, client, monkeypatch)
    calls = []

    async def handler(args):
        calls.append(args)
        return {"content": [{"type": "text", "text": "kept"}]}

    (tmp_path / "scenarios" / "one").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scenarios" / "one" / "scenario.json").write_text(
        json.dumps({"name": "one", "branch": "the account is on hold", "solution": [{"tool": "a"}]})
    )
    drop = runtime._confirmed(ToolSpec("drop_scenarios", "", {}, handler))
    submit = runtime._confirmed(ToolSpec("submit_scenario", "", {}, handler))
    runtime._from_platform = True
    corrected = {"name": "one", "branch": "the account is on hold", "solution": [{"tool": "b"}]}
    assert asyncio.run(submit.handler(corrected)).get("is_error") is None
    assert asyncio.run(submit.handler({**corrected, "branch": "a lost item"}))["is_error"] is True
    assert asyncio.run(drop.handler({"names": ["one"]}))["is_error"] is True
    answer = asyncio.run(runtime._ask("AskUserQuestion", {"question": "Which?"}, None))
    assert "Nobody answers" in answer["content"][0]["text"]
    assert calls == [corrected]
