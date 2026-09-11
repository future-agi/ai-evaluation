"""A REAL Semantic Kernel agent driven by the kit (part-1).

Async framework (ChatCompletionAgent). Worker thread + asyncio.run; a kernel
plugin whose function records its own invocation. The seeded bug gives the agent
no plugins, so it cannot look anything up.
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
        from semantic_kernel.agents import ChatCompletionAgent
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
        from semantic_kernel.functions import kernel_function

        recorded: list[dict[str, Any]] = []

        class OrderPlugin:
            @kernel_function(description="Look up an order's status by id.", name="order_status")
            def order_status(self, order_id: str = "") -> str:
                recorded.append({"id": f"c{len(recorded)}", "name": "order_status",
                                 "arguments": {"order_id": order_id}})
                return _CANNED

        async def _arun() -> Any:
            # create AND close the SK service/client INSIDE the loop (else the
            # underlying AsyncOpenAI httpx client finalizes after asyncio.run
            # returns -> "RuntimeError: Event loop is closed").
            service = OpenAIChatCompletion(ai_model_id=model)
            agent = ChatCompletionAgent(
                service=service,
                name="support",
                instructions="You are an order-support agent. Use the order_status function to "
                             "look up the order, then give a clear, complete answer.",
                plugins=[OrderPlugin()] if with_tool else [],
            )
            try:
                return await agent.get_response(messages=question)
            finally:
                await service.client.close()

        resp = asyncio.run(_arun())
        content = getattr(getattr(resp, "message", None), "content", None) or str(resp)
        return {"content": str(content or ""), "tool_calls": recorded}

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_work).result()


def run_agent(agent_input: Any) -> dict[str, Any]:
    return _run(agent_input, with_tool=True)


def make_agent(model: str = MODEL, temperature: float = 0.0, bind_tools: bool = True):
    def _r(agent_input: Any) -> dict[str, Any]:
        return _run(agent_input, with_tool=bind_tools, model=model)
    return _r


_REC = '''
        class OrderPlugin:
            @kernel_function(description="Look up an order's status by id.", name="order_status")
            def order_status(self, order_id: str = "") -> str:
                recorded.append({"id":"c%d"%len(recorded),"name":"order_status","arguments":{"order_id":order_id}})
                return "Order 4821: shipped, arriving Tuesday."'''

BUGGY_SRC = '''
def run_agent(agent_input):
    import asyncio, concurrent.futures
    q = "What is the status of order 4821?"
    for m in reversed(list(getattr(agent_input, "messages", None) or [])):
        if isinstance(m, dict) and m.get("content"): q = str(m["content"]); break
    def _work():
        from semantic_kernel.agents import ChatCompletionAgent
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
        async def _a():
            svc = OpenAIChatCompletion(ai_model_id="gpt-4o-mini")  # create+close inside loop
            agent = ChatCompletionAgent(service=svc,
                name="support", instructions="Help the customer.", plugins=[])  # BUG: no plugins
            try: return await agent.get_response(messages=q)
            finally: await svc.client.close()
        resp = asyncio.run(_a())
        content = getattr(getattr(resp,"message",None),"content",None) or str(resp)
        return {"content": str(content or ""), "tool_calls": []}
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
        from semantic_kernel.agents import ChatCompletionAgent
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
        from semantic_kernel.functions import kernel_function
        recorded = []
        class OrderPlugin:
            @kernel_function(description="Look up an order's status by id.", name="order_status")
            def order_status(self, order_id: str = "") -> str:
                recorded.append({"id":"c%d"%len(recorded),"name":"order_status","arguments":{"order_id":order_id}})
                return "Order 4821: shipped, arriving Tuesday."
        async def _a():
            svc = OpenAIChatCompletion(ai_model_id="gpt-4o-mini")  # create+close inside loop
            agent = ChatCompletionAgent(service=svc,
                name="support", instructions="Help the customer; use order_status.", plugins=[OrderPlugin()])
            try: return await agent.get_response(messages=q)
            finally: await svc.client.close()
        resp = asyncio.run(_a())
        content = getattr(getattr(resp,"message",None),"content",None) or str(resp)
        return {"content": str(content or ""), "tool_calls": recorded}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_work).result()
'''
