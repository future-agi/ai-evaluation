"""A REAL AutoGen (autogen-agentchat) agent driven by the kit (part-1).

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
        from autogen_agentchat.agents import AssistantAgent
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        recorded: list[dict[str, Any]] = []

        async def _arun() -> Any:
            # close the client INSIDE the loop (else "Event loop is closed" at teardown)
            client = OpenAIChatCompletionClient(model=model, temperature=0.0)
            tools = []
            if with_tool:
                def order_status(order_id: str = "") -> str:
                    """Look up an order's status by id."""
                    recorded.append({"id": f"c{len(recorded)}", "name": "order_status",
                                     "arguments": {"order_id": order_id}})
                    return _CANNED
                tools = [order_status]
            agent = AssistantAgent("support", model_client=client, tools=tools,
                                   reflect_on_tool_use=True)
            res = await agent.run(task=question)
            await client.close()
            return res

        result = asyncio.run(_arun())
        content = ""
        for msg in reversed(getattr(result, "messages", None) or []):
            c = getattr(msg, "content", None)
            if isinstance(c, str) and c.strip():
                content = c
                break
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
        from autogen_agentchat.agents import AssistantAgent
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        async def _arun():
            client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.0)
            agent = AssistantAgent("support", model_client=client, tools=[])  # BUG: no tools
            res = await agent.run(task=q); await client.close(); return res
        result = asyncio.run(_arun())
        content = ""
        for msg in reversed(getattr(result, "messages", None) or []):
            c = getattr(msg, "content", None)
            if isinstance(c, str) and c.strip(): content = c; break
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
        from autogen_agentchat.agents import AssistantAgent
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        recorded = []
        async def _arun():
            client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.0)
            def order_status(order_id: str = "") -> str:
                "Look up an order's status by id."
                recorded.append({"id":"c%d"%len(recorded),"name":"order_status","arguments":{"order_id":order_id}})
                return "Order 4821: shipped, arriving Tuesday."
            agent = AssistantAgent("support", model_client=client, tools=[order_status], reflect_on_tool_use=True)
            res = await agent.run(task=q); await client.close(); return res
        result = asyncio.run(_arun())
        content = ""
        for msg in reversed(getattr(result, "messages", None) or []):
            c = getattr(msg, "content", None)
            if isinstance(c, str) and c.strip(): content = c; break
        return {"content": content, "tool_calls": recorded}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_work).result()
'''
