"""A REAL Google ADK (google-adk) agent driven by the kit (part-1).

Autonomous async framework. ADK has no one-call `agent.run()`: you build an
`Agent`, wrap it in an `InMemoryRunner` (its own SessionService), create a
session, feed a `types.Content` user message, and read results by iterating the
async event stream (`event.is_final_response()` -> `event.content.parts[].text`).
ADK is Gemini-native, so the OpenAI model is wrapped with `LiteLlm`. Worker
thread + asyncio.run (the framework drives its own event loop, which collides
with the kit engine's). Temperature 0 via GenerateContentConfig. The tool
records its own invocation. The seeded bug gives the agent NO tools.
"""

from __future__ import annotations

from typing import Any

import google  # noqa: F401  (the adapter must import google)

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
    import asyncio
    import concurrent.futures

    question = _user_text(agent_input)

    def _work() -> dict[str, Any]:
        from google.adk.agents import Agent
        from google.adk.models.lite_llm import LiteLlm
        from google.adk.runners import InMemoryRunner
        from google.genai import types

        recorded: list[dict[str, Any]] = []
        tools = []
        if with_tool:
            def order_status(order_id: str = "") -> str:
                """Look up an order's status by id."""
                recorded.append({"id": f"c{len(recorded)}", "name": "order_status",
                                 "arguments": {"order_id": order_id}})
                return _CANNED
            tools = [order_status]

        agent = Agent(
            name="support",
            model=LiteLlm(model=model),
            instruction=_INSTRUCTION,
            tools=tools,
            generate_content_config=types.GenerateContentConfig(temperature=0.0),
        )

        async def _arun() -> str:
            runner = InMemoryRunner(agent=agent, app_name="kit_order_support")
            await runner.session_service.create_session(
                app_name="kit_order_support", user_id="kit_user", session_id="kit_session")
            message = types.Content(role="user", parts=[types.Part(text=question)])
            content = ""
            async for event in runner.run_async(
                user_id="kit_user", session_id="kit_session", new_message=message):
                if event.is_final_response() and event.content and event.content.parts:
                    text = "".join(p.text for p in event.content.parts if getattr(p, "text", None))
                    if text.strip():
                        content = text
            await runner.close()
            return content

        content = asyncio.run(_arun())
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
        from google.adk.agents import Agent
        from google.adk.models.lite_llm import LiteLlm
        from google.adk.runners import InMemoryRunner
        from google.genai import types
        agent = Agent(name="support", model=LiteLlm(model="gpt-4o-mini"),
                      instruction="Help the customer.", tools=[],  # BUG: no tools
                      generate_content_config=types.GenerateContentConfig(temperature=0.0))
        async def _arun():
            runner = InMemoryRunner(agent=agent, app_name="kit_order_support")
            await runner.session_service.create_session(
                app_name="kit_order_support", user_id="kit_user", session_id="kit_session")
            msg = types.Content(role="user", parts=[types.Part(text=q)])
            content = ""
            async for event in runner.run_async(user_id="kit_user", session_id="kit_session", new_message=msg):
                if event.is_final_response() and event.content and event.content.parts:
                    t = "".join(p.text for p in event.content.parts if getattr(p, "text", None))
                    if t.strip(): content = t
            await runner.close()
            return content
        return {"content": asyncio.run(_arun()), "tool_calls": []}
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
        from google.adk.agents import Agent
        from google.adk.models.lite_llm import LiteLlm
        from google.adk.runners import InMemoryRunner
        from google.genai import types
        recorded = []
        def order_status(order_id: str = "") -> str:
            "Look up an order's status by id."
            recorded.append({"id":"c%d"%len(recorded),"name":"order_status","arguments":{"order_id":order_id}})
            return "Order 4821: shipped, arriving Tuesday."
        agent = Agent(name="support", model=LiteLlm(model="gpt-4o-mini"),
                      instruction="You are an order-support agent. Use the order_status tool to look up the order, then give a clear answer with the order id and status.",
                      tools=[order_status],
                      generate_content_config=types.GenerateContentConfig(temperature=0.0))
        async def _arun():
            runner = InMemoryRunner(agent=agent, app_name="kit_order_support")
            await runner.session_service.create_session(
                app_name="kit_order_support", user_id="kit_user", session_id="kit_session")
            msg = types.Content(role="user", parts=[types.Part(text=q)])
            content = ""
            async for event in runner.run_async(user_id="kit_user", session_id="kit_session", new_message=msg):
                if event.is_final_response() and event.content and event.content.parts:
                    t = "".join(p.text for p in event.content.parts if getattr(p, "text", None))
                    if t.strip(): content = t
            await runner.close()
            return content
        return {"content": asyncio.run(_arun()), "tool_calls": recorded}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_work).result()
'''
