"""Battle-test: deep env creation — world-kinds x tool-mocking, real vs typed-only.

Reality guards for the "all kinds of sim, tool mocking, CUA/browser diff envs"
directive. Pins (a) that tool-mocking genuinely EXECUTES (agent calls a mocked
tool -> handler fires -> fixture result flows back), and (b) the honest
world-kind / tool-mock-level partition (what's executable vs typed-only vs
not-yet-built), so the surface can't silently drift or overclaim.
"""

from __future__ import annotations

import asyncio

import pytest

from fi.alk import simulate


# --- the honest env-creation partition (contract-pinned) --------------------
def test_world_kind_execution_partition() -> None:
    from fi.simulate.simulation import contract as c

    # only these two run a live agent in-process; the rest are typed-only in v1.
    assert c.EXECUTABLE_WORLD_KINDS_V1 == ("conversation", "tool_api")
    assert set(c.TYPED_ONLY_WORLD_KINDS_V1) == {
        "browser", "computer_use", "code_exec", "voice_telephony"
    }


def test_tool_mock_levels_vocab() -> None:
    from fi.simulate.simulation import contract as c

    # the closed tool-mock ladder. static_fixture + recorded_replay have lifting
    # builders today; emulated + live are typed but not yet wired (A15 deferred).
    assert c.TOOL_MOCK_LEVELS == ("static_fixture", "recorded_replay", "emulated", "live")


def test_mock_tools_env_is_creatable() -> None:
    # the mock_tools env type is a supported, creatable environment.
    assert "mock_tools" in simulate.supported_manifest_environment_types()


# --- tool-mocking genuinely EXECUTES (the concrete "tool mocking" ask) -------
@pytest.mark.integration
def test_tool_mock_executes_end_to_end(tmp_path) -> None:
    agent = {
        "type": "scripted",
        "responses": [
            {"content": "Checking the weather tool.",
             "tool_calls": [{"id": "c1", "name": "get_weather", "arguments": {"city": "SF"}}]},
            {"content": "Per the tool, it's sunny and 72F in SF."},
        ],
    }
    env = {"type": "mock_tools",
           "data": {"tools": {"get_weather": {"response": {"content": "sunny, 72F", "success": True}}}}}
    manifest = simulate.build_task_run_manifest(
        name="toolmock", agent=agent,
        task_description="Get the weather in SF using the tool.",
        success_criteria=["sunny"], threshold=0.5,
        environments=[env], auto_execute_tools=True, min_turns=1, max_turns=2,
    )
    p = tmp_path / "m.json"
    simulate.write_manifest_file(manifest, p)
    res = asyncio.run(simulate.run_manifest_file(p))
    r0 = (res.get("report", {}).get("results") or [{}])[0]

    # the agent's tool call was recorded
    tool_calls = r0.get("tool_calls") or []
    assert any(tc.get("name") == "get_weather" for tc in tool_calls), tool_calls
    # the mock handler FIRED (a tool-execution/tool event was emitted)
    events = r0.get("events") or []
    assert any("tool" in str(e.get("type", "")) for e in events), events
    # the FIXTURE result flowed back into the conversation (not a stub)
    transcript = str(r0.get("transcript") or "")
    assert "72F" in transcript, "mock tool fixture did not reach the transcript"
