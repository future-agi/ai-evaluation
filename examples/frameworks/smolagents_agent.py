"""A REAL smolagents agent driven by the kit (part-1).

Autonomous framework (ToolCallingAgent runs the loop). Worker thread, temp 0, the
tool records its own invocation. The seeded bug gives the agent NO tools.
"""

from __future__ import annotations

from typing import Any

MODEL = "gpt-4o-mini"
_CANNED = "Order 4821: shipped, arriving Tuesday."
_DIRECTIVE = (
    "You are an order-support agent. You MUST call the order_status tool to look up "
    "the order, then state the order id and its status. Do not ask for clarification."
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
        from smolagents import LiteLLMModel, ToolCallingAgent, tool

        recorded: list[dict[str, Any]] = []
        tools = []
        if with_tool:
            @tool
            def order_status(order_id: str = "") -> str:
                """Look up an order's status by id.

                Args:
                    order_id: The order id to look up.
                """
                recorded.append({"id": f"c{len(recorded)}", "name": "order_status",
                                 "arguments": {"order_id": order_id}})
                return _CANNED
            tools = [order_status]

        agent = ToolCallingAgent(tools=tools, model=LiteLLMModel(model_id=model, temperature=0.0))
        answer = agent.run(f"{_DIRECTIVE}\n\n{question}")
        return {"content": str(answer or ""), "tool_calls": recorded}

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
        from smolagents import ToolCallingAgent, LiteLLMModel
        agent = ToolCallingAgent(tools=[], model=LiteLLMModel(model_id="gpt-4o-mini", temperature=0.0))  # BUG: no tools
        return {"content": str(agent.run(q) or ""), "tool_calls": []}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_work).result()
'''

FIXED_SRC = '''
def run_agent(agent_input):
    import concurrent.futures
    q = "What is the status of order 4821?"
    for m in reversed(list(getattr(agent_input, "messages", None) or [])):
        if isinstance(m, dict) and m.get("content"): q = str(m["content"]); break
    directive = ("You are an order-support agent. You MUST call the order_status tool to look up "
                 "the order, then state the order id and its status. Do not ask for clarification.")
    def _work():
        from smolagents import ToolCallingAgent, LiteLLMModel, tool
        recorded = []
        @tool
        def order_status(order_id: str = "") -> str:
            """Look up an order's status by id.

            Args:
                order_id: The order id to look up.
            """
            recorded.append({"id":"c%d"%len(recorded),"name":"order_status","arguments":{"order_id":order_id}})
            return "Order 4821: shipped, arriving Tuesday."
        agent = ToolCallingAgent(tools=[order_status], model=LiteLLMModel(model_id="gpt-4o-mini", temperature=0.0))
        return {"content": str(agent.run(directive + "\\n\\n" + q) or ""), "tool_calls": recorded}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_work).result()
'''
