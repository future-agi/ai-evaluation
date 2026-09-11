"""A REAL BeeAI (beeai-framework) agent driven by the kit (part-1).

Autonomous async framework: the RequirementAgent drives its own asyncio loop, so
we run it in a worker thread + asyncio.run (avoids "event loop is already
running"). beeai-framework doesn't surface raw tool calls on its output, so the
order_status tool RECORDS ITS OWN invocation into a closure list (a real signal
-- the tool genuinely ran). LLM temperature 0. The seeded bug gives the agent NO
tools.
"""

from __future__ import annotations

from typing import Any

MODEL = "gpt-4o-mini"
_CANNED = "Order 4821: shipped, arriving Tuesday."
_INSTRUCTIONS = (
    "You are an order-support agent. Use the order_status tool to look up the "
    "order, then give a clear, complete answer stating the order id and its status."
)


def _user_text(agent_input: Any) -> str:
    for m in reversed(list(getattr(agent_input, "messages", None) or [])):
        if isinstance(m, dict) and m.get("role") in (None, "user") and m.get("content"):
            return str(m["content"])
    return "What is the status of order 4821?"


def _run(agent_input: Any, *, with_tool: bool, model: str = MODEL) -> dict[str, Any]:
    import asyncio
    import concurrent.futures

    question = _user_text(agent_input)

    def _work() -> dict[str, Any]:
        from beeai_framework.adapters.openai import OpenAIChatModel
        from beeai_framework.agents.requirement import RequirementAgent
        from beeai_framework.backend.types import ChatModelParameters
        from beeai_framework.tools.tool import tool

        recorded: list[dict[str, Any]] = []
        tools = []
        if with_tool:
            @tool(name="order_status", description="Look up an order's status by id.")
            def order_status(order_id: str = "") -> str:
                recorded.append({"id": f"c{len(recorded)}", "name": "order_status",
                                 "arguments": {"order_id": order_id}})
                return _CANNED
            tools = [order_status]

        llm = OpenAIChatModel(model, parameters=ChatModelParameters(temperature=0.0))
        agent = RequirementAgent(llm=llm, tools=tools, instructions=_INSTRUCTIONS)

        async def _arun() -> Any:
            return await agent.run(question)

        result = asyncio.run(_arun())
        content = ""
        last = getattr(result, "last_message", None)
        if last is not None:
            content = getattr(last, "text", "") or ""
        return {"content": content, "tool_calls": recorded}

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
    import asyncio, concurrent.futures
    q = "What is the status of order 4821?"
    for m in reversed(list(getattr(agent_input, "messages", None) or [])):
        if isinstance(m, dict) and m.get("content"): q = str(m["content"]); break
    def _work():
        from beeai_framework.adapters.openai import OpenAIChatModel
        from beeai_framework.agents.requirement import RequirementAgent
        from beeai_framework.backend.types import ChatModelParameters
        llm = OpenAIChatModel("gpt-4o-mini", parameters=ChatModelParameters(temperature=0.0))
        agent = RequirementAgent(llm=llm, tools=[],  # BUG: no tools
                                 instructions="You are an order-support agent. Answer the order question.")
        async def _arun(): return await agent.run(q)
        result = asyncio.run(_arun())
        last = getattr(result, "last_message", None)
        content = (getattr(last, "text", "") or "") if last is not None else ""
        return {"content": content, "tool_calls": []}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_work).result()
'''

FIXED_SRC = '''
def run_agent(agent_input):
    import asyncio, concurrent.futures
    q = "What is the status of order 4821?"
    for m in reversed(list(getattr(agent_input, "messages", None) or [])):
        if isinstance(m, dict) and m.get("content"): q = str(m["content"]); break
    def _work():
        from beeai_framework.adapters.openai import OpenAIChatModel
        from beeai_framework.agents.requirement import RequirementAgent
        from beeai_framework.backend.types import ChatModelParameters
        from beeai_framework.tools.tool import tool
        recorded = []
        @tool(name="order_status", description="Look up an order's status by id.")
        def order_status(order_id: str = "") -> str:
            recorded.append({"id":"c%d"%len(recorded),"name":"order_status","arguments":{"order_id":order_id}})
            return "Order 4821: shipped, arriving Tuesday."
        llm = OpenAIChatModel("gpt-4o-mini", parameters=ChatModelParameters(temperature=0.0))
        agent = RequirementAgent(llm=llm, tools=[order_status],
                                 instructions="You are an order-support agent. Use the order_status tool to look up the order, then state the order id and its status.")
        async def _arun(): return await agent.run(q)
        result = asyncio.run(_arun())
        last = getattr(result, "last_message", None)
        content = (getattr(last, "text", "") or "") if last is not None else ""
        return {"content": content, "tool_calls": recorded}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_work).result()
'''
