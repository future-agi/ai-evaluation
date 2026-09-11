"""A REAL PydanticAI agent driven by the kit (part-1).

PydanticAI runs an autonomous loop (decide -> execute tool -> respond), unlike the
single-node LangChain/LangGraph agents. So the tool is a real Python function
returning the order's status; the agent genuinely decides to call it, calls it,
and answers. We surface BOTH the tool calls it made (for tool_selection_accuracy)
and the final answer (for task_success). The seeded bug registers NO tool, so the
agent cannot look anything up and fabricates an answer.
"""

from __future__ import annotations

from typing import Any

MODEL = "openai:gpt-4o-mini"
_CANNED = "Order 4821: shipped, arriving Tuesday."


def _user_text(agent_input: Any) -> str:
    for m in reversed(list(getattr(agent_input, "messages", None) or [])):
        if isinstance(m, dict) and m.get("role") in (None, "user") and m.get("content"):
            return str(m["content"])
    return "What is the status of order 4821?"


def _run(agent_input: Any, *, with_tool: bool, model: str = MODEL) -> dict[str, Any]:
    # PydanticAI's run_sync drives its own event loop, which collides with the
    # kit engine's running loop ("event loop is already running"). Run it in a
    # worker thread so it gets a fresh loop — the generalizable fix for sync-in-
    # async frameworks.
    import concurrent.futures

    question = _user_text(agent_input)

    def _work() -> dict[str, Any]:
        from pydantic_ai import Agent

        agent = Agent(model, model_settings={"temperature": 0.0},
                      system_prompt="You are an order-support agent. Use the order_status tool to "
                                    "look up the order, then give a clear, complete answer stating "
                                    "the order id and its status.")
        if with_tool:
            @agent.tool_plain
            def order_status(order_id: str = "") -> str:
                """Look up an order's status by id."""
                return _CANNED

        result = agent.run_sync(question)
        tool_calls = []
        for i, msg in enumerate(result.all_messages()):
            for part in getattr(msg, "parts", []) or []:
                if getattr(part, "part_kind", "") == "tool-call":
                    tool_calls.append({"id": getattr(part, "tool_call_id", None) or f"call_{i}",
                                       "name": getattr(part, "tool_name", "") or "",
                                       "arguments": getattr(part, "args", None) or {}})
        return {"content": str(result.output or ""), "tool_calls": tool_calls}

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_work).result()


def run_agent(agent_input: Any) -> dict[str, Any]:
    return _run(agent_input, with_tool=True)


def make_agent(model: str = MODEL, temperature: float = 0.0, bind_tools: bool = True):
    def _r(agent_input: Any) -> dict[str, Any]:
        return _run(agent_input, with_tool=bind_tools, model=model)
    return _r


BUGGY_SRC = '''
def run_agent(agent_input):
    import concurrent.futures
    msgs = list(getattr(agent_input, "messages", None) or [])
    q = "What is the status of order 4821?"
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("content"): q = str(m["content"]); break
    def _work():
        from pydantic_ai import Agent
        result = Agent("openai:gpt-4o-mini", model_settings={"temperature":0.0}).run_sync(q)  # BUG: no tool registered
        return {"content": str(result.output or ""), "tool_calls": []}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_work).result()
'''

FIXED_SRC = '''
def run_agent(agent_input):
    import concurrent.futures
    msgs = list(getattr(agent_input, "messages", None) or [])
    q = "What is the status of order 4821?"
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("content"): q = str(m["content"]); break
    def _work():
        from pydantic_ai import Agent
        agent = Agent("openai:gpt-4o-mini", model_settings={"temperature":0.0},
                      system_prompt="You are an order-support agent. Use order_status to look up the order, then state the order id and its status.")
        @agent.tool_plain
        def order_status(order_id: str = "") -> str:
            "Look up an order's status by id."
            return "Order 4821: shipped, arriving Tuesday."
        result = agent.run_sync(q)
        tcs = []
        for i, m in enumerate(result.all_messages()):
            for p in getattr(m, "parts", []) or []:
                if getattr(p, "part_kind", "") == "tool-call":
                    tcs.append({"id": getattr(p,"tool_call_id",None) or ("c%d"%i),
                                "name": getattr(p,"tool_name","") or "", "arguments": getattr(p,"args",None) or {}})
        return {"content": str(result.output or ""), "tool_calls": tcs}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_work).result()
'''
