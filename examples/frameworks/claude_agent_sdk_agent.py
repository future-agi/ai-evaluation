"""A REAL Claude Agent SDK agent driven by the kit (part-1).

claude_agent_sdk is an autonomous framework: ``query()`` drives the bundled
``claude`` CLI over a subprocess transport inside its own asyncio event loop, so
we run it in a worker thread + ``asyncio.run`` (avoids "event loop is already
running" against the kit engine's loop).

Tools are in-process MCP tools: ``@tool(...)`` defines an async handler, and
``create_sdk_mcp_server`` exposes them; the model reaches them as the namespaced
name ``mcp__<server>__<tool>``. The SDK doesn't hand back a tidy tool-call list,
so the tool RECORDS ITS OWN invocation into a closure list (a real signal — the
handler genuinely ran), with a scan of the streamed ToolUseBlocks as a fallback.
The seeded bug registers NO MCP server/tool, so the agent can't look the order up.

NOTE: the Claude Agent SDK only runs Claude models via the CLI; the contract's
``gpt-4o-mini`` default is meaningless to it, so a non-Claude model is mapped to a
Claude default (a Claude id is passed through untouched). Import + PREP need no key;
a live run needs Claude Code auth / an Anthropic key and the bundled CLI.
"""

from __future__ import annotations

from typing import Any

MODEL = "gpt-4o-mini"
_CLAUDE_DEFAULT = "claude-sonnet-4-5"
_CANNED = "Order 4821: shipped, arriving Tuesday."
_SERVER = "order"
_TOOL = "order_status"
_QUALIFIED_TOOL = f"mcp__{_SERVER}__{_TOOL}"
_INSTRUCTION = (
    "You are an order-support agent. Use the order_status tool to look up the "
    "order, then give the customer a clear, complete answer stating the order id "
    "and its status."
)


def _user_text(agent_input: Any) -> str:
    for m in reversed(list(getattr(agent_input, "messages", None) or [])):
        if isinstance(m, dict) and m.get("role") in (None, "user") and m.get("content"):
            return str(m["content"])
    return "What is the status of order 4821?"


def _claude_model(model: str) -> str:
    # The SDK drives Claude only; pass a Claude id through, else fall back.
    return model if isinstance(model, str) and model.startswith("claude") else _CLAUDE_DEFAULT


def _run(agent_input: Any, *, with_tool: bool, model: str = MODEL) -> dict[str, Any]:
    import asyncio
    import concurrent.futures

    question = _user_text(agent_input)
    claude_model = _claude_model(model)

    def _work() -> dict[str, Any]:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
            create_sdk_mcp_server,
            query,
            tool,
        )

        recorded: list[dict[str, Any]] = []

        mcp_servers: dict[str, Any] = {}
        allowed_tools: list[str] = []
        if with_tool:
            @tool(_TOOL, "Look up an order's status by id.", {"order_id": str})
            async def order_status(args: dict[str, Any]) -> dict[str, Any]:
                order_id = str(args.get("order_id", ""))
                recorded.append({"id": f"c{len(recorded)}", "name": _TOOL,
                                 "arguments": {"order_id": order_id}})
                return {"content": [{"type": "text", "text": _CANNED}]}

            server = create_sdk_mcp_server(_SERVER, tools=[order_status])
            mcp_servers = {_SERVER: server}
            allowed_tools = [_QUALIFIED_TOOL]

        options = ClaudeAgentOptions(
            model=claude_model,
            system_prompt=_INSTRUCTION,
            tools=[],   # disable built-in tools (Bash/Read/WebSearch/...) so order_status is the only signal
            mcp_servers=mcp_servers,
            allowed_tools=allowed_tools,
            permission_mode="bypassPermissions",
            setting_sources=[],   # SDK isolation: ignore filesystem settings/CLAUDE.md
            max_turns=4,
        )

        async def _arun() -> dict[str, Any]:
            final = ""
            text_parts: list[str] = []
            scanned: list[dict[str, Any]] = []
            async for msg in query(prompt=question, options=options):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock) and block.text:
                            text_parts.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            scanned.append({"id": block.id, "name": str(block.name),
                                            "arguments": dict(block.input or {})})
                elif isinstance(msg, ResultMessage):
                    if msg.result:
                        final = msg.result
            content = final or "\n".join(text_parts)
            return {"content": content, "tool_calls": recorded or scanned}

        return asyncio.run(_arun())

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
        from claude_agent_sdk import (
            AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query,
        )
        options = ClaudeAgentOptions(
            model="claude-sonnet-4-5",
            system_prompt="You are an order-support agent. Answer the customer's order question.",
            tools=[], mcp_servers={}, allowed_tools=[],  # BUG: no tool registered
            permission_mode="bypassPermissions", setting_sources=[], max_turns=4)
        async def _arun():
            final = ""; parts = []
            async for msg in query(prompt=q, options=options):
                if isinstance(msg, AssistantMessage):
                    for b in msg.content:
                        if isinstance(b, TextBlock) and b.text: parts.append(b.text)
                elif isinstance(msg, ResultMessage):
                    if msg.result: final = msg.result
            return {"content": final or "\\n".join(parts), "tool_calls": []}
        return asyncio.run(_arun())
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
        from claude_agent_sdk import (
            AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, ToolUseBlock,
            create_sdk_mcp_server, query, tool,
        )
        recorded = []
        @tool("order_status", "Look up an order's status by id.", {"order_id": str})
        async def order_status(args):
            oid = str(args.get("order_id", ""))
            recorded.append({"id": "c%d" % len(recorded), "name": "order_status", "arguments": {"order_id": oid}})
            return {"content": [{"type": "text", "text": "Order 4821: shipped, arriving Tuesday."}]}
        server = create_sdk_mcp_server("order", tools=[order_status])
        options = ClaudeAgentOptions(
            model="claude-sonnet-4-5",
            system_prompt="You are an order-support agent. Use the order_status tool to look up the order, then give a clear answer with the order id and its status.",
            tools=[], mcp_servers={"order": server}, allowed_tools=["mcp__order__order_status"],
            permission_mode="bypassPermissions", setting_sources=[], max_turns=4)
        async def _arun():
            final = ""; parts = []; scanned = []
            async for msg in query(prompt=q, options=options):
                if isinstance(msg, AssistantMessage):
                    for b in msg.content:
                        if isinstance(b, TextBlock) and b.text: parts.append(b.text)
                        elif isinstance(b, ToolUseBlock):
                            scanned.append({"id": b.id, "name": str(b.name), "arguments": dict(b.input or {})})
                elif isinstance(msg, ResultMessage):
                    if msg.result: final = msg.result
            return {"content": final or "\\n".join(parts), "tool_calls": recorded or scanned}
        return asyncio.run(_arun())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_work).result()
'''
