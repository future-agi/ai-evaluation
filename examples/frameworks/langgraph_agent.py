"""A REAL LangGraph agent (a compiled StateGraph) driven by the kit (part-1).

A single model node binds the task's tools and emits the tool call for the kit's
engine to execute turn-by-turn (we don't use a prebuilt react loop, which would
try to run tools the kit owns). Reuses the LangChain message/tool converters.
"""

from __future__ import annotations

from typing import Any

from examples.frameworks.langchain_agent import _to_lc_messages, _to_openai_tools

MODEL = "gpt-4o-mini"


def _graph(bind_tools: bool, model: str, temperature: float, tools_raw: Any):
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import MessagesState

    llm: Any = ChatOpenAI(model=model, temperature=temperature)
    if bind_tools:
        tools = _to_openai_tools(tools_raw)
        if tools:
            llm = llm.bind_tools(tools)

    def call_model(state: MessagesState) -> dict[str, Any]:
        return {"messages": [llm.invoke(state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("model", call_model)
    g.add_edge(START, "model")
    g.add_edge("model", END)
    return g.compile()


def _invoke(agent_input: Any, *, bind_tools: bool, model: str = MODEL,
            temperature: float = 0.0) -> dict[str, Any]:
    messages = _to_lc_messages(getattr(agent_input, "messages", None)) or \
        _to_lc_messages([{"role": "user", "content": "Help the user."}])
    graph = _graph(bind_tools, model, temperature, getattr(agent_input, "tools", None))
    out = graph.invoke({"messages": messages})
    ai = out["messages"][-1]
    tcs = [
        {"id": tc.get("id") or f"call_{i}", "name": tc.get("name") or "",
         "arguments": tc.get("args") or {}}
        for i, tc in enumerate(getattr(ai, "tool_calls", None) or [])
    ]
    return {"content": str(getattr(ai, "content", "") or ""), "tool_calls": tcs}


def run_agent(agent_input: Any) -> dict[str, Any]:
    return _invoke(agent_input, bind_tools=True)


def make_agent(model: str = MODEL, temperature: float = 0.0, bind_tools: bool = True):
    def _run(agent_input: Any) -> dict[str, Any]:
        return _invoke(agent_input, bind_tools=bind_tools, model=model, temperature=temperature)
    return _run


BUGGY_SRC = '''
def run_agent(agent_input):
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from langgraph.graph import START, END, StateGraph
    from langgraph.graph.message import MessagesState
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # BUG: tools never bound
    def call_model(state):
        return {"messages": [llm.invoke(state["messages"])]}
    g = StateGraph(MessagesState); g.add_node("m", call_model)
    g.add_edge(START, "m"); g.add_edge("m", END); graph = g.compile()
    msgs = [HumanMessage(content=str(m.get("content") if hasattr(m,"get") else m))
            for m in (getattr(agent_input,"messages",None) or []) if (m.get("content") if hasattr(m,"get") else m)]
    if not msgs: msgs = [HumanMessage(content="Help the user.")]
    ai = graph.invoke({"messages": msgs})["messages"][-1]
    return {"content": str(getattr(ai,"content","") or ""), "tool_calls": []}
'''

FIXED_SRC = '''
def run_agent(agent_input):
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from langgraph.graph import START, END, StateGraph
    from langgraph.graph.message import MessagesState
    raw = list(getattr(agent_input,"tools",None) or [])
    tools = []
    for spec in raw:
        if not hasattr(spec,"get"): continue
        if spec.get("type")=="function" and spec.get("function"): tools.append(dict(spec)); continue
        name = spec.get("name")
        if name: tools.append({"type":"function","function":{"name":name,
            "description":spec.get("description") or name,
            "parameters":spec.get("parameters") or {"type":"object","properties":{}}}})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    if tools: llm = llm.bind_tools(tools)
    def call_model(state):
        return {"messages": [llm.invoke(state["messages"])]}
    g = StateGraph(MessagesState); g.add_node("m", call_model)
    g.add_edge(START, "m"); g.add_edge("m", END); graph = g.compile()
    msgs = [HumanMessage(content=str(m.get("content") if hasattr(m,"get") else m))
            for m in (getattr(agent_input,"messages",None) or []) if (m.get("content") if hasattr(m,"get") else m)]
    if not msgs: msgs = [HumanMessage(content="Help the user.")]
    ai = graph.invoke({"messages": msgs})["messages"][-1]
    tcs = [{"id": tc.get("id") or ("c%d"%i), "name": tc.get("name") or "", "arguments": tc.get("args") or {}}
           for i,tc in enumerate(getattr(ai,"tool_calls",None) or [])]
    return {"content": str(getattr(ai,"content","") or ""), "tool_calls": tcs}
'''
