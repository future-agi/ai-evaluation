"""A REAL LangChain agent driven by the kit's simulation engine (part-1 live loop).

The kit runs an agent spec ``{"type":"python","callable":"<this file>:run_agent"}``.
Each call receives an ``agent_input`` with ``.messages`` (history) + ``.tools``
(available tool specs) and must return ``{"content": str, "tool_calls": [...]}``.

``run_agent`` builds a genuine LangChain ``ChatOpenAI`` agent, binds the task's
tools, and invokes a real LLM — so eval/sim/opt/code-RSI exercise actual
framework execution, not a kit-native stub. ``run_agent_buggy`` is the seeded
"forgets-the-tool" bug for the code-RSI lane (it ignores the tools and fabricates
an answer); the proven fix is to bind + call the tool.
"""

from __future__ import annotations

from typing import Any, Mapping

MODEL = "gpt-4o-mini"  # cheap, OpenAI key; default across frameworks (cost-bounded)


def _to_openai_tools(raw_tools: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for spec in list(raw_tools or []):
        if not isinstance(spec, Mapping):
            continue
        if spec.get("type") == "function" and isinstance(spec.get("function"), Mapping):
            out.append(dict(spec))
            continue
        name = str(spec.get("name") or "")
        if not name:
            continue
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": str(spec.get("description") or f"Tool {name}."),
                "parameters": dict(spec.get("parameters") or {"type": "object", "properties": {}}),
            },
        })
    return out


def _lc_tool_calls(raw: Any) -> list[dict[str, Any]]:
    """Convert kit/OpenAI tool_call shapes -> LangChain ``{name, args, id}``."""
    import json

    out: list[dict[str, Any]] = []
    for i, tc in enumerate(list(raw or [])):
        if not isinstance(tc, Mapping):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), Mapping) else tc
        args = fn.get("arguments", fn.get("args", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {"_raw": args}
        out.append({
            "name": str(fn.get("name") or tc.get("name") or ""),
            "args": dict(args or {}),
            "id": str(tc.get("id") or f"call_{i}"),
        })
    return out


def _to_lc_messages(messages: Any) -> list[Any]:
    """Convert the kit's history into LangChain messages, preserving the
    tool-call protocol: an assistant turn that issued tool calls becomes an
    ``AIMessage`` carrying those ``tool_calls``, so the following ``ToolMessage``
    has a valid antecedent (OpenAI 400s on an orphaned tool message). A tool
    result whose id we never saw is folded into a human turn instead of dropped.
    """
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    out: list[Any] = []
    known_tool_ids: set[str] = set()
    for m in list(messages or []):
        if not isinstance(m, Mapping):
            out.append(HumanMessage(content=str(m)))
            continue
        role = str(m.get("role") or "user")
        content = str(m.get("content") or "")
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role in ("assistant", "ai"):
            tcs = _lc_tool_calls(m.get("tool_calls"))
            if tcs:
                known_tool_ids.update(tc["id"] for tc in tcs)
                out.append(AIMessage(content=content, tool_calls=tcs))
            else:
                out.append(AIMessage(content=content))
        elif role == "tool":
            tid = str(m.get("tool_call_id") or m.get("id") or "")
            if tid and tid in known_tool_ids:
                out.append(ToolMessage(content=content, tool_call_id=tid))
            else:  # no valid antecedent -> fold into a human turn (avoid OpenAI 400)
                out.append(HumanMessage(content=f"Tool result: {content}"))
        else:
            out.append(HumanMessage(content=content))
    return out


def run_agent(agent_input: Any) -> dict[str, Any]:
    """REAL LangChain agent: bind the task's tools and call a live LLM."""
    from langchain_openai import ChatOpenAI

    messages = _to_lc_messages(getattr(agent_input, "messages", None))
    tools = _to_openai_tools(getattr(agent_input, "tools", None))
    if not messages:
        messages = _to_lc_messages([{"role": "user", "content": "Help the user."}])

    llm: Any = ChatOpenAI(model=MODEL, temperature=0)
    if tools:
        llm = llm.bind_tools(tools)
    ai = llm.invoke(messages)

    tool_calls = [
        {"id": tc.get("id") or f"call_{i}", "name": tc.get("name") or "",
         "arguments": tc.get("args") or {}}
        for i, tc in enumerate(getattr(ai, "tool_calls", None) or [])
    ]
    return {"content": str(getattr(ai, "content", "") or ""), "tool_calls": tool_calls}


def make_agent(model: str = MODEL, temperature: float = 0.0, bind_tools: bool = True):
    """Factory for the optimize lane (``agent.type=framework`` + ``factory`` +
    search over ``factory_kwargs``). Returns a plain callable so the kit's generic
    ``callable`` adapter (``input_mode=agent_input``) drives it. The searchable
    config is real LangChain behavior: ``bind_tools`` toggles whether the agent
    can call the task's tools — the optimizer should pick ``True`` (a real,
    interpretable lift), and ``model``/``temperature`` are also searchable."""

    def _run(agent_input: Any) -> dict[str, Any]:
        from langchain_openai import ChatOpenAI

        messages = _to_lc_messages(getattr(agent_input, "messages", None)) or \
            _to_lc_messages([{"role": "user", "content": "Help the user."}])
        llm: Any = ChatOpenAI(model=model, temperature=temperature)
        if bind_tools:
            tools = _to_openai_tools(getattr(agent_input, "tools", None))
            if tools:
                llm = llm.bind_tools(tools)
        ai = llm.invoke(messages)
        tcs = [
            {"id": tc.get("id") or f"call_{i}", "name": tc.get("name") or "",
             "arguments": tc.get("args") or {}}
            for i, tc in enumerate(getattr(ai, "tool_calls", None) or [])
        ]
        return {"content": str(getattr(ai, "content", "") or ""), "tool_calls": tcs}

    return _run


# --- code-RSI lane: self-contained buggy -> fixed source (the loop rewrites the
#     agent's ACTUAL source). BUGGY forgets to bind/call the tool; FIXED binds it. ---
BUGGY_SRC = '''
def run_agent(agent_input):
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    msgs = []
    for m in (getattr(agent_input, "messages", None) or []):
        c = m.get("content") if hasattr(m, "get") else str(m)
        if c:
            msgs.append(HumanMessage(content=str(c)))
    if not msgs:
        msgs = [HumanMessage(content="Help the user.")]
    ai = ChatOpenAI(model="gpt-4o-mini", temperature=0).invoke(msgs)  # BUG: tools never bound
    return {"content": str(getattr(ai, "content", "") or ""), "tool_calls": []}
'''

FIXED_SRC = '''
def run_agent(agent_input):
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    raw = list(getattr(agent_input, "tools", None) or [])
    tools = []
    for spec in raw:
        if not hasattr(spec, "get"):
            continue
        if spec.get("type") == "function" and spec.get("function"):
            tools.append(dict(spec)); continue
        name = spec.get("name")
        if name:
            tools.append({"type": "function", "function": {"name": name,
                "description": spec.get("description") or name,
                "parameters": spec.get("parameters") or {"type": "object", "properties": {}}}})
    msgs = []
    for m in (getattr(agent_input, "messages", None) or []):
        c = m.get("content") if hasattr(m, "get") else str(m)
        if c:
            msgs.append(HumanMessage(content=str(c)))
    if not msgs:
        msgs = [HumanMessage(content="Help the user.")]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    if tools:
        llm = llm.bind_tools(tools)
    ai = llm.invoke(msgs)
    tcs = [{"id": tc.get("id") or ("c%d" % i), "name": tc.get("name") or "",
            "arguments": tc.get("args") or {}}
           for i, tc in enumerate(getattr(ai, "tool_calls", None) or [])]
    return {"content": str(getattr(ai, "content", "") or ""), "tool_calls": tcs}
'''
