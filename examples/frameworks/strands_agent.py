"""A REAL Strands (strands-agents) agent driven by the kit (part-1).

Autonomous framework: Strands' ``Agent.__call__`` runs its own event loop
(isolated in a worker thread via ``strands._async.run_async``), so it is fully
synchronous from the caller's view. We still drive it inside a
ThreadPoolExecutor worker (crewai-style) to stay clear of the kit engine's own
running loop. Temperature 0; the tool RECORDS ITS OWN invocation into a closure
list (a real signal — the tool genuinely ran) and we surface that as tool_calls.
The seeded bug gives the agent NO tools.

Note: Strands' ``Agent(model=...)`` defaults to **Bedrock** when given a bare
string, so we wire an explicit ``OpenAIModel`` (model_id + params.temperature).
"""

from __future__ import annotations

from typing import Any

# Module-level import so this adapter literally imports strands at import time
# (the framework objects are still constructed lazily inside _work()).
import strands  # noqa: F401

MODEL = "gpt-4o-mini"
_CANNED = "Order 4821: shipped, arriving Tuesday."

_INSTRUCTION = (
    "You are an order-support agent. Use the order_status tool to look up the "
    "order, then give the customer a clear, complete answer stating the order "
    "id and its status."
)


def _user_text(agent_input: Any) -> str:
    for m in reversed(list(getattr(agent_input, "messages", None) or [])):
        if isinstance(m, dict) and m.get("role") in (None, "user") and m.get("content"):
            return str(m["content"])
    return "What is the status of order 4821?"


def _run(agent_input: Any, *, with_tool: bool, model: str = MODEL) -> dict[str, Any]:
    import concurrent.futures

    question = _user_text(agent_input)

    def _work() -> dict[str, Any]:
        from strands import Agent, tool
        from strands.models.openai import OpenAIModel

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

        # Bare-string model => Bedrock default; wire OpenAI explicitly.
        model_obj = OpenAIModel(model_id=model, params={"temperature": 0.0})
        agent = Agent(model=model_obj, tools=tools, system_prompt=_INSTRUCTION,
                      callback_handler=None)
        result = agent(question)
        return {"content": str(result), "tool_calls": recorded}

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
        from strands import Agent
        from strands.models.openai import OpenAIModel
        model_obj = OpenAIModel(model_id="gpt-4o-mini", params={"temperature": 0.0})
        agent = Agent(model=model_obj, tools=[],  # BUG: no tools
                      system_prompt="You are an order-support agent. Help the customer.",
                      callback_handler=None)
        result = agent(q)
        return {"content": str(result), "tool_calls": []}
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
        from strands import Agent, tool
        from strands.models.openai import OpenAIModel
        recorded = []
        @tool
        def order_status(order_id: str = "") -> str:
            "Look up an order's status by id."
            recorded.append({"id": "c%d"%len(recorded), "name": "order_status", "arguments": {"order_id": order_id}})
            return "Order 4821: shipped, arriving Tuesday."
        model_obj = OpenAIModel(model_id="gpt-4o-mini", params={"temperature": 0.0})
        agent = Agent(model=model_obj, tools=[order_status],
                      system_prompt="You are an order-support agent. Use the order_status tool to look up the order, then give a clear, complete answer stating the order id and its status.",
                      callback_handler=None)
        result = agent(q)
        return {"content": str(result), "tool_calls": recorded}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_work).result()
'''
