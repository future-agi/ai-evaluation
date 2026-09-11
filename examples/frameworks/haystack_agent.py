"""A REAL Haystack (haystack-ai) Agent driven by the kit (part-1).

Autonomous framework: the `haystack.components.agents.Agent` runs its own internal
pipeline loop (chat-generator -> ToolInvoker -> chat-generator ... until a text-only
turn). We run it in a worker thread (parity with the other autonomous adapters), the
tool RECORDS ITS OWN invocation into a closure list (a real signal — the tool genuinely
ran), and the LLM temperature is pinned to 0 via the generator's generation_kwargs.
The seeded bug gives the agent NO tools.
"""

from __future__ import annotations

from typing import Any

import haystack  # noqa: F401  (the adapter must import haystack)
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import tool

MODEL = "gpt-4o-mini"
_CANNED = "Order 4821: shipped, arriving Tuesday."
_INSTRUCTIONS = (
    "You are an order-support agent. Use the order_status tool to look up the order, "
    "then give a clear, complete answer stating the order id and its status."
)


def _user_text(agent_input: Any) -> str:
    for m in reversed(list(getattr(agent_input, "messages", None) or [])):
        if isinstance(m, dict) and m.get("role") in (None, "user") and m.get("content"):
            return str(m["content"])
    return "What is the status of order 4821?"


def _surface_tool_calls(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Fallback: read tool calls off the assistant messages in the run output."""
    calls: list[dict[str, Any]] = []
    for msg in result.get("messages", []) or []:
        for i, tc in enumerate(getattr(msg, "tool_calls", None) or []):
            calls.append({"id": getattr(tc, "id", None) or f"call_{len(calls)}",
                          "name": getattr(tc, "tool_name", "tool"),
                          "arguments": getattr(tc, "arguments", None) or {}})
    return calls


def _run(agent_input: Any, *, with_tool: bool, model: str = MODEL) -> dict[str, Any]:
    import concurrent.futures

    question = _user_text(agent_input)

    def _work() -> dict[str, Any]:
        recorded: list[dict[str, Any]] = []
        tools = []
        if with_tool:
            @tool
            def order_status(order_id: str = "") -> str:
                """Look up an order's status by id."""
                recorded.append({"id": f"c{len(recorded)}", "name": "order_status",
                                 "arguments": {"order_id": order_id}})
                return _CANNED
            tools = [order_status]

        agent = Agent(
            chat_generator=OpenAIChatGenerator(model=model,
                                               generation_kwargs={"temperature": 0.0}),
            tools=tools,
            system_prompt=_INSTRUCTIONS,
        )
        agent.warm_up()
        result = agent.run(messages=[ChatMessage.from_user(question)])
        last = result.get("last_message")
        content = getattr(last, "text", None) or ""
        # tool-records-itself (clean signal) with the run messages as a fallback.
        return {"content": str(content), "tool_calls": recorded or _surface_tool_calls(result)}

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
    q = "What is the status of order 4821?"
    for m in reversed(list(getattr(agent_input, "messages", None) or [])):
        if isinstance(m, dict) and m.get("content"): q = str(m["content"]); break
    def _work():
        from haystack.components.agents import Agent
        from haystack.components.generators.chat import OpenAIChatGenerator
        from haystack.dataclasses import ChatMessage
        agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini", generation_kwargs={"temperature": 0.0}),
            tools=[],  # BUG: no tools
            system_prompt="You are an order-support agent. Help the customer.")
        agent.warm_up()
        result = agent.run(messages=[ChatMessage.from_user(q)])
        last = result.get("last_message")
        return {"content": str(getattr(last, "text", None) or ""), "tool_calls": []}
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
        from haystack.components.agents import Agent
        from haystack.components.generators.chat import OpenAIChatGenerator
        from haystack.dataclasses import ChatMessage
        from haystack.tools import tool
        recorded = []
        @tool
        def order_status(order_id: str = "") -> str:
            "Look up an order's status by id."
            recorded.append({"id": "c%d"%len(recorded), "name": "order_status", "arguments": {"order_id": order_id}})
            return "Order 4821: shipped, arriving Tuesday."
        agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini", generation_kwargs={"temperature": 0.0}),
            tools=[order_status],
            system_prompt="You are an order-support agent. Use the order_status tool to look up the order, then give a clear, complete answer stating the order id and its status.")
        agent.warm_up()
        result = agent.run(messages=[ChatMessage.from_user(q)])
        last = result.get("last_message")
        return {"content": str(getattr(last, "text", None) or ""), "tool_calls": recorded}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_work).result()
'''
