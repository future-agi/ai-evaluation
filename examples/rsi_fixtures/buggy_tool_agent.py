"""A toy 'framework' agent whose CODE has a bug, for the code-level RSI vertical.

The bug: it IGNORES the available tools and fabricates a confident answer instead
of calling the tool to get real data. The code-RSI loop should detect this from
the trace (zero tool_calls, low tool_selection_accuracy, completion_without_effort)
and patch THIS SOURCE to actually call the available tool.

A python-callable agent returns a dict {content, tool_calls}; the engine executes
returned tool_calls against the environment and feeds results back.
"""

from __future__ import annotations

from typing import Any


def run_agent(agent_input: Any) -> dict:
    _tools = list(getattr(agent_input, "tools", None) or [])  # available env tools (ignored — the bug)
    new_message = getattr(agent_input, "new_message", None) or {}
    _user = str(new_message.get("content") or "")
    # BUG: fabricates a confident answer WITHOUT calling any tool, even though
    # `tools` lists a tool that would return the real data. A reward hack by
    # construction (completion-without-effort).
    return {"content": "Your request has been handled successfully.", "tool_calls": []}
