"""Native sub-agents: what tools they are actually given."""

import asyncio
from types import SimpleNamespace

from claude_agent_sdk import AssistantMessage, ResultMessage, ToolResultBlock, ToolUseBlock

from fi.alk.harness.backends.base import (
    Call,
    ModelReply,
    SessionSpec,
    StageDone,
    ToolServer,
    ToolSpec,
    WorkerSpec,
)
from fi.alk.harness.backends.claude import ClaudeBackend, ClaudeSession, _definition


def _tool(name: str) -> ToolSpec:
    return ToolSpec(name=name, description=name, input_schema={}, handler=lambda _a: {})


def test_every_workers_tools_survive_the_merge_onto_one_server_name() -> None:
    server = "scenarios"
    spec = SessionSpec(
        system_prompt="parent",
        servers={server: ToolServer(name=server, tools=[_tool("aim_for")])},
        workers={
            "scenario_writer": WorkerSpec(
                description="writes",
                instructions="write",
                servers={server: ToolServer(name=server, tools=[_tool("submit_scenario")])},
            ),
            "suite_reviewer": WorkerSpec(
                description="reviews",
                instructions="review",
                servers={server: ToolServer(name=server, tools=[_tool("inspect_scenario")])},
            ),
        },
    )

    options = ClaudeBackend()._options(spec)
    allowed = set(options.allowed_tools or [])
    assert {
        f"mcp__{server}__aim_for",
        f"mcp__{server}__submit_scenario",
        f"mcp__{server}__inspect_scenario",
    } <= allowed, sorted(allowed)


def test_a_worker_does_not_inherit_the_stages_delegation_tools() -> None:
    server = "scenarios"
    spec = SessionSpec(
        system_prompt="parent",
        builtins=("AskUserQuestion", "Delegate"),
        servers={server: ToolServer(name=server, tools=[_tool("aim_for")])},
        workers={
            "scenario_writer": WorkerSpec(
                description="writes",
                instructions="write",
                servers={server: ToolServer(name=server, tools=[_tool("submit_scenario")])},
            )
        },
    )
    tools = _definition(spec.workers["scenario_writer"], spec).tools
    assert "Agent" not in tools and "Task" not in tools, tools
    assert f"mcp__{server}__submit_scenario" in tools, tools


def test_a_stage_that_delegates_waits_for_its_writers() -> None:
    from pathlib import Path

    source = Path("src/fi/alk/harness/backends/claude.py").read_text(encoding="utf-8")
    assert 'environment["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] = "1"' in source
    assert "background=False" in source


def test_a_successful_terminal_save_interrupts_and_ignores_later_tool_calls() -> None:
    terminal = "mcp__runtime_repair__submit_runtime_plan_patch"
    received = [
        AssistantMessage(
            content=[ToolUseBlock(id="first", name=terminal, input={})],
            model="gemini-3.7-flash",
        ),
        SimpleNamespace(content=[ToolResultBlock(tool_use_id="first", content="accepted")]),
        AssistantMessage(
            content=[ToolUseBlock(id="second", name=terminal, input={})],
            model="gemini-3.7-flash",
        ),
        ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="session",
        ),
    ]

    class Client:
        interrupted = 0

        async def receive_response(self):
            for item in received:
                yield item

        async def interrupt(self):
            self.interrupted += 1

    session = ClaudeSession(None)
    client = Client()
    session._client = client

    async def collect():
        return [event async for event in session.replies()]

    events = asyncio.run(collect())
    assert client.interrupted == 1
    calls = [
        part
        for event in events
        if isinstance(event, ModelReply)
        for part in event.parts
        if isinstance(part, Call)
    ]
    assert [call.name for call in calls] == [terminal]
    assert isinstance(events[-1], StageDone)
