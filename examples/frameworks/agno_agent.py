"""A REAL Agno agent driven by the kit (part-1).

Autonomous framework: Agno's Agent.run() drives its own internal loops, so we run
it in a worker thread (concurrent.futures) to avoid colliding with the kit engine's
asyncio loop ("event loop is already running"). Temperature 0 via an explicit
OpenAIChat model (Agno requires a Model instance or a "provider:id" string for
temperature control). The tool RECORDS ITS OWN invocation into a closure list (a
real signal that it genuinely ran) and that list is surfaced as tool_calls, with
RunOutput.tools as a fallback. The seeded bug gives the agent NO tools.
"""

from __future__ import annotations

from typing import Any

MODEL = "gpt-4o-mini"
_CANNED = "Order 4821: shipped, arriving Tuesday."


def _user_text(agent_input: Any) -> str:
    for m in reversed(list(getattr(agent_input, "messages", None) or [])):
        if isinstance(m, dict) and m.get("role") in (None, "user") and m.get("content"):
            return str(m["content"])
    return "What is the status of order 4821?"


def _surface_tool_calls(result: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for i, t in enumerate(getattr(result, "tools", None) or []):
        name = getattr(t, "tool_name", None) or "tool"
        calls.append({"id": getattr(t, "tool_call_id", None) or f"call_{i}",
                      "name": str(name), "arguments": getattr(t, "tool_args", None) or {}})
    return calls


def _run(agent_input: Any, *, with_tool: bool, model: str = MODEL,
         temperature: float = 0.0) -> dict[str, Any]:
    import concurrent.futures

    question = _user_text(agent_input)

    def _work() -> dict[str, Any]:
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat

        recorded: list[dict[str, Any]] = []
        tools = []
        if with_tool:
            def order_status(order_id: str = "") -> str:
                """Look up an order's status by id."""
                recorded.append({"id": f"c{len(recorded)}", "name": "order_status",
                                 "arguments": {"order_id": order_id}})
                return _CANNED
            tools = [order_status]

        agent = Agent(
            model=OpenAIChat(id=model, temperature=temperature),
            tools=tools,
            instructions="You are an order-support agent. Use the order_status tool to look up the order, then give the customer a clear, complete answer stating the order id and its status.",
        )
        result = agent.run(question)
        content = getattr(result, "content", None)
        # tool-records-itself (clean signal) with RunOutput.tools as a fallback.
        return {"content": str(content or ""),
                "tool_calls": recorded or _surface_tool_calls(result)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_work).result()


def run_agent(agent_input: Any) -> dict[str, Any]:
    return _run(agent_input, with_tool=True)


def make_agent(model: str = MODEL, temperature: float = 0.0, bind_tools: bool = True):
    def _r(agent_input: Any) -> dict[str, Any]:
        return _run(agent_input, with_tool=bind_tools, model=model, temperature=temperature)
    return _r


BUGGY_SRC = '''
def run_agent(agent_input):
    import concurrent.futures
    q = "What is the status of order 4821?"
    for m in reversed(list(getattr(agent_input, "messages", None) or [])):
        if isinstance(m, dict) and m.get("content"): q = str(m["content"]); break
    def _work():
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat
        agent = Agent(model=OpenAIChat(id="gpt-4o-mini", temperature=0.0),
                      tools=[],  # BUG: no tools
                      instructions="You are an order-support agent. Help the customer.")
        result = agent.run(q)
        return {"content": str(getattr(result, "content", "") or ""), "tool_calls": []}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_work).result()
'''

FIXED_SRC = '''
def run_agent(agent_input):
    import concurrent.futures
    q = "What is the status of order 4821?"
    for m in reversed(list(getattr(agent_input, "messages", None) or [])):
        if isinstance(m, dict) and m.get("content"): q = str(m["content"]); break
    def _work():
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat
        recorded = []
        def order_status(order_id: str = "") -> str:
            "Look up an order's status by id."
            recorded.append({"id":"c%d"%len(recorded),"name":"order_status","arguments":{"order_id":order_id}})
            return "Order 4821: shipped, arriving Tuesday."
        agent = Agent(model=OpenAIChat(id="gpt-4o-mini", temperature=0.0),
                      tools=[order_status],
                      instructions="You are an order-support agent. Use the order_status tool to look up the order, then give a clear, complete answer stating the order id and its status.")
        result = agent.run(q)
        return {"content": str(getattr(result, "content", "") or ""), "tool_calls": recorded}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_work).result()
'''
