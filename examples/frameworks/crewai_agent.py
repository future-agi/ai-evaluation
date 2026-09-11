"""A REAL CrewAI agent driven by the kit (part-1).

CrewAI doesn't surface tool calls in its result, so the tool RECORDS ITS OWN
invocation into a closure list (a real signal — the tool genuinely ran). Worker
thread for CrewAI's internal loops, LLM temperature 0, real tool returning the
order status. The seeded bug gives the agent NO tools.
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
    import concurrent.futures

    question = _user_text(agent_input)

    def _work() -> dict[str, Any]:
        from crewai import LLM, Agent, Crew, Task
        from crewai.tools import tool

        recorded: list[dict[str, Any]] = []
        tools = []
        if with_tool:
            @tool("order_status")
            def order_status(order_id: str = "") -> str:
                """Look up an order's status by id."""
                recorded.append({"id": f"c{len(recorded)}", "name": "order_status",
                                 "arguments": {"order_id": order_id}})
                return _CANNED
            tools = [order_status]

        agent = Agent(role="Support agent", goal="Answer the customer's order question",
                      backstory="You help customers check orders.", tools=tools,
                      llm=LLM(model=model, temperature=0.0), verbose=False)
        task = Task(description=question, expected_output="The order's status.", agent=agent)
        result = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
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
        from crewai import Agent, Task, Crew, LLM
        agent = Agent(role="Support agent", goal="Answer the order question",
                      backstory="You help customers.", tools=[],  # BUG: no tools
                      llm=LLM(model="gpt-4o-mini", temperature=0.0), verbose=False)
        task = Task(description=q, expected_output="The order's status.", agent=agent)
        result = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
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
        from crewai import Agent, Task, Crew, LLM
        from crewai.tools import tool
        recorded = []
        @tool("order_status")
        def order_status(order_id: str = "") -> str:
            "Look up an order's status by id."
            recorded.append({"id": "c%d"%len(recorded), "name": "order_status", "arguments": {"order_id": order_id}})
            return "Order 4821: shipped, arriving Tuesday."
        agent = Agent(role="Support agent", goal="Answer the order question",
                      backstory="You help customers.", tools=[order_status],
                      llm=LLM(model="gpt-4o-mini", temperature=0.0), verbose=False)
        task = Task(description=q, expected_output="The order's status.", agent=agent)
        result = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
        return {"content": str(result), "tool_calls": recorded}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_work).result()
'''
