"""A REAL DSPy agent driven by the kit (part-1).

DSPy is a genuinely different paradigm — declarative *signatures* + modules
(`dspy.ReAct`), compiled rather than prompt-strung. Proving the loop on it shows
the kit's adapter contract generalizes ACROSS paradigms.

Uses the thread-safe ``dspy.context(lm=...)`` (NOT the global ``dspy.configure``,
whose settings are locked to the thread that first configured them — the kit runs
the agent across tasks, so a per-call global reconfigure raises "dspy.settings can
only be changed by the thread that initially configured it"). DSPy is synchronous,
so no worker thread is needed. The tool records its own invocation; the seeded
bug uses a tool-less ``dspy.Predict`` so it cannot look anything up.
"""

from __future__ import annotations

from typing import Any

MODEL = "openai/gpt-4o-mini"
_CANNED = "Order 4821: shipped, arriving Tuesday."
_INSTR = ("You are an order-support agent. Use the order_status tool to look up the "
          "order, then give a clear, complete answer stating the order id and its status.")


def _user_text(agent_input: Any) -> str:
    for m in reversed(list(getattr(agent_input, "messages", None) or [])):
        if isinstance(m, dict) and m.get("role") in (None, "user") and m.get("content"):
            return str(m["content"])
    return "What is the status of order 4821?"


def _run(agent_input: Any, *, with_tool: bool, model: str = MODEL) -> dict[str, Any]:
    import dspy

    question = _user_text(agent_input)
    recorded: list[dict[str, Any]] = []
    with dspy.context(lm=dspy.LM(model, temperature=0.0, cache=False)):
        if with_tool:
            def order_status(order_id: str = "") -> str:
                """Look up an order's status by its id."""
                recorded.append({"id": f"c{len(recorded)}", "name": "order_status",
                                 "arguments": {"order_id": order_id}})
                return _CANNED
            sig = dspy.Signature("question -> answer: str", _INSTR)
            pred = dspy.ReAct(sig, tools=[order_status])(question=question)
        else:
            pred = dspy.Predict("question -> answer: str")(question=question)  # BUG path: no tools
    return {"content": str(getattr(pred, "answer", "") or ""), "tool_calls": recorded}


def run_agent(agent_input: Any) -> dict[str, Any]:
    return _run(agent_input, with_tool=True)


def make_agent(model: str = MODEL, temperature: float = 0.0, bind_tools: bool = True):
    def _r(agent_input: Any) -> dict[str, Any]:
        return _run(agent_input, with_tool=bind_tools, model=model)
    return _r


BUGGY_SRC = '''
def run_agent(agent_input):
    import dspy
    q = "What is the status of order 4821?"
    for m in reversed(list(getattr(agent_input, "messages", None) or [])):
        if isinstance(m, dict) and m.get("content"): q = str(m["content"]); break
    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", temperature=0.0, cache=False)):
        pred = dspy.Predict("question -> answer: str")(question=q)  # BUG: no tools
    return {"content": str(getattr(pred, "answer", "") or ""), "tool_calls": []}
'''

FIXED_SRC = '''
def run_agent(agent_input):
    import dspy
    q = "What is the status of order 4821?"
    for m in reversed(list(getattr(agent_input, "messages", None) or [])):
        if isinstance(m, dict) and m.get("content"): q = str(m["content"]); break
    recorded = []
    def order_status(order_id: str = "") -> str:
        "Look up an order's status by its id."
        recorded.append({"id":"c%d"%len(recorded),"name":"order_status","arguments":{"order_id":order_id}})
        return "Order 4821: shipped, arriving Tuesday."
    instr = ("You are an order-support agent. Use the order_status tool to look up the "
             "order, then give a clear, complete answer stating the order id and its status.")
    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", temperature=0.0, cache=False)):
        sig = dspy.Signature("question -> answer: str", instr)
        pred = dspy.ReAct(sig, tools=[order_status])(question=q)
    return {"content": str(getattr(pred, "answer", "") or ""), "tool_calls": recorded}
'''
