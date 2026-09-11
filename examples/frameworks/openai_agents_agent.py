"""A REAL OpenAI Agents SDK agent driven by the kit (part-1).

Autonomous-loop framework (same template as PydanticAI): a worker thread for the
SDK's event loop, temperature 0, a real tool returning the order status, and the
tool calls surfaced from the run items. The seeded bug registers NO tool.
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
    for i, item in enumerate(getattr(result, "new_items", None) or []):
        if type(item).__name__ in ("ToolCallItem",) or getattr(item, "type", "") == "tool_call_item":
            raw = getattr(item, "raw_item", None)
            name = getattr(raw, "name", None) or getattr(item, "name", None) or "tool"
            args = getattr(raw, "arguments", None) or {}
            calls.append({"id": getattr(raw, "call_id", None) or f"call_{i}",
                          "name": str(name), "arguments": args})
    return calls


def _run(agent_input: Any, *, with_tool: bool, model: str = MODEL) -> dict[str, Any]:
    import concurrent.futures

    question = _user_text(agent_input)

    def _work() -> dict[str, Any]:
        from agents import Agent, ModelSettings, Runner, function_tool

        recorded: list[dict[str, Any]] = []
        tools = []
        if with_tool:
            @function_tool
            def order_status(order_id: str = "") -> str:
                """Look up an order's status by id."""
                recorded.append({"id": f"c{len(recorded)}", "name": "order_status",
                                 "arguments": {"order_id": order_id}})
                return _CANNED
            tools = [order_status]

        agent = Agent(name="support", instructions="You are an order-support agent. Use the order_status tool to look up the order, then give the customer a clear, complete answer stating the order id and its status.",
                      model=model, tools=tools, model_settings=ModelSettings(temperature=0.0))
        result = Runner.run_sync(agent, question)
        # tool-records-itself (clean signal) with new_items as a fallback.
        return {"content": str(getattr(result, "final_output", "") or ""),
                "tool_calls": recorded or _surface_tool_calls(result)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_work).result()


def run_agent(agent_input: Any) -> dict[str, Any]:
    return _run(agent_input, with_tool=True)


def make_agent(model: str = MODEL, temperature: float = 0.0, bind_tools: bool = True):
    def _r(agent_input: Any) -> dict[str, Any]:
        return _run(agent_input, with_tool=bind_tools, model=model)
    return _r


_TOOL_DEF = '''
        @function_tool
        def order_status(order_id: str = "") -> str:
            "Look up an order's status by id."
            return "Order 4821: shipped, arriving Tuesday."
        tools = [order_status]'''

BUGGY_SRC = '''
def run_agent(agent_input):
    import concurrent.futures
    q = "What is the status of order 4821?"
    for m in reversed(list(getattr(agent_input, "messages", None) or [])):
        if isinstance(m, dict) and m.get("content"): q = str(m["content"]); break
    def _work():
        from agents import Agent, ModelSettings, Runner
        agent = Agent(name="support", instructions="Help the user.", model="gpt-4o-mini",
                      tools=[], model_settings=ModelSettings(temperature=0.0))  # BUG: no tool
        result = Runner.run_sync(agent, q)
        return {"content": str(getattr(result, "final_output", "") or ""), "tool_calls": []}
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
        from agents import Agent, ModelSettings, Runner, function_tool
        @function_tool
        def order_status(order_id: str = "") -> str:
            "Look up an order's status by id."
            return "Order 4821: shipped, arriving Tuesday."
        agent = Agent(name="support", instructions="Help the user; use tools.", model="gpt-4o-mini",
                      tools=[order_status], model_settings=ModelSettings(temperature=0.0))
        result = Runner.run_sync(agent, q)
        tcs = []
        for i, item in enumerate(getattr(result, "new_items", None) or []):
            if type(item).__name__ == "ToolCallItem" or getattr(item, "type", "") == "tool_call_item":
                raw = getattr(item, "raw_item", None)
                tcs.append({"id": getattr(raw,"call_id",None) or ("c%d"%i),
                            "name": str(getattr(raw,"name",None) or "tool"),
                            "arguments": getattr(raw,"arguments",None) or {}})
        return {"content": str(getattr(result, "final_output", "") or ""), "tool_calls": tcs}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_work).result()
'''
