"""Battle-test: the model-driven TOOL-CALLING agent (real agentic action loop).

`agent.type=llm_tool_calling` is the canonical agent-takes-actions loop: the MODEL
decides whether to call the environment's tools (function-calling), the engine
executes them (mock or real) and feeds results back. This is what makes the kit's
env run REAL agents, credential-free + multi-modal + tool-mocked.

Unit tests (credential-free) pin the format conversions; the live end-to-end loop
is key-gated (skips without OPENAI_API_KEY).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from fi.simulate.cli import _to_openai_tools


def test_to_openai_tools_normalizes_plain_specs() -> None:
    out = _to_openai_tools([
        {"name": "get_weather", "description": "weather",
         "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}},
    ])
    assert out == [{
        "type": "function",
        "function": {
            "name": "get_weather", "description": "weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }]


def test_to_openai_tools_passes_through_function_shape() -> None:
    spec = {"type": "function", "function": {"name": "x", "description": "d", "parameters": {}}}
    assert _to_openai_tools([spec]) == [spec]


def test_to_openai_tools_skips_garbage() -> None:
    assert _to_openai_tools([{"no_name": 1}, "bad", None]) == []


def test_llm_tool_calling_agent_builds() -> None:
    # building requires only instructions (no key); only CALLING needs a key.
    from fi.simulate.cli import _build_agent_callback

    cb = _build_agent_callback(
        {"type": "llm_tool_calling", "model": "gpt-4o-mini", "instructions": "be helpful"},
        Path("."),
    )
    assert callable(cb)


def test_llm_tool_calling_requires_instructions() -> None:
    from fi.simulate.cli import ManifestError, _build_agent_callback

    with pytest.raises(ManifestError):
        _build_agent_callback({"type": "llm_tool_calling"}, Path("."))


@pytest.mark.integration
def test_llm_tool_calling_live_loop(tmp_path) -> None:
    """The full live agentic loop: model decides to call a mocked tool, the env
    executes it, the result feeds back. Key-gated."""
    if not (os.environ.get("OPENAI_API_KEY") or "").strip():
        pytest.skip("OPENAI_API_KEY not set")
    from fi.alk import simulate

    agent = {"type": "llm_tool_calling", "model": "gpt-4o-mini",
             "instructions": "You are a weather assistant. When asked about weather you MUST "
                             "call the get_weather tool, then report its result."}
    env = {"type": "mock_tools", "data": {"tools": {"get_weather": {
        "schema": {"description": "Get current weather for a city",
                   "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                  "required": ["city"]}},
        "response": {"content": "It is 19C and raining in London.", "success": True}}}}}
    m = simulate.build_task_run_manifest(
        name="toolcall", agent=agent, task_description="What is the weather in London?",
        success_criteria=["raining"], threshold=0.5,
        environments=[env], auto_execute_tools=True, min_turns=1, max_turns=4)
    p = tmp_path / "m.json"
    simulate.write_manifest_file(m, p)
    res = asyncio.run(simulate.run_manifest_file(p))
    r0 = (res.get("report", {}).get("results") or [{}])[0]
    tool_calls = r0.get("tool_calls") or []
    assert any(tc.get("name") == "get_weather" for tc in tool_calls), tool_calls
    assert "raining" in str(r0.get("transcript") or "")
