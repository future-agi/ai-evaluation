"""A REAL LlamaIndex agent (FunctionAgent) driven by the kit (part-1).

Autonomous async framework: worker thread + asyncio.run, the tool records its own
invocation, temperature 0. The seeded bug gives the agent NO tools.
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


def _run(agent_input: Any, *, with_tool: bool, model: str = MODEL) -> dict[str, Any]:
    import asyncio
    import concurrent.futures

    question = _user_text(agent_input)

    def _work() -> dict[str, Any]:
        from llama_index.core.agent.workflow import FunctionAgent
        from llama_index.llms.openai import OpenAI

        recorded: list[dict[str, Any]] = []
        tools = []
        if with_tool:
            def order_status(order_id: str = "") -> str:
                """Look up an order's status by id."""
                recorded.append({"id": f"c{len(recorded)}", "name": "order_status",
                                 "arguments": {"order_id": order_id}})
                return _CANNED
            tools = [order_status]

        agent = FunctionAgent(tools=tools, llm=OpenAI(model=model, temperature=0.0),
                              system_prompt="Help the customer; use tools when available.")

        async def _arun() -> Any:
            # agent.run must be CALLED inside a running loop (it returns a handler
            # to await); calling it as asyncio.run's arg raised "no running event loop".
            return await agent.run(question)

        resp = asyncio.run(_arun())
        return {"content": str(resp), "tool_calls": recorded}

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
        from llama_index.core.agent.workflow import FunctionAgent
        from llama_index.llms.openai import OpenAI
        agent = FunctionAgent(tools=[], llm=OpenAI(model="gpt-4o-mini", temperature=0.0),
                              system_prompt="Help the customer.")  # BUG: no tools
        async def _arun(): return await agent.run(q)
        resp = asyncio.run(_arun())
        return {"content": str(resp), "tool_calls": []}
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
        from llama_index.core.agent.workflow import FunctionAgent
        from llama_index.llms.openai import OpenAI
        recorded = []
        def order_status(order_id: str = "") -> str:
            "Look up an order's status by id."
            recorded.append({"id":"c%d"%len(recorded),"name":"order_status","arguments":{"order_id":order_id}})
            return "Order 4821: shipped, arriving Tuesday."
        agent = FunctionAgent(tools=[order_status], llm=OpenAI(model="gpt-4o-mini", temperature=0.0),
                              system_prompt="Help the customer; use tools.")
        async def _arun(): return await agent.run(q)
        resp = asyncio.run(_arun())
        return {"content": str(resp), "tool_calls": recorded}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_work).result()
'''
