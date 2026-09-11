"""Bring-your-own / 3rd-party framework proof (part-1).

The kit works with ANY framework via its generic `callable`/`custom`/`framework`
adapters — not just the famous names. To prove that isn't special-cased, this
file defines a SYNTHETIC framework nobody has ever heard of ("Acme") with its OWN
idiosyncratic API (``AcmeAgent(...).act(...) -> AcmeResult`` with ``.reply`` and
``.actions_taken``), then adapts it to the kit's agent contract in ~10 lines.

If the full eval/sim + optimize + code-RSI loop runs green on Acme, the contract
is universal: a user drops in their own framework the same way. Acme calls a real
LLM via litellm (real key), so this is genuine execution, not a stub.
"""

from __future__ import annotations

import json
from typing import Any

MODEL = "gpt-4o-mini"
_CANNED = "Order 4821: shipped, arriving Tuesday."


# ===========================================================================
# THE "3RD-PARTY FRAMEWORK" — pretend this is `pip install acme-agents`.
# Its API looks nothing like LangChain/OpenAI; the kit has never seen it.
# ===========================================================================
class AcmeAction:
    def __init__(self, tool: str, params: dict) -> None:
        self.tool = tool
        self.params = params


class AcmeResult:
    def __init__(self, reply: str, actions_taken: list[AcmeAction]) -> None:
        self.reply = reply
        self.actions_taken = actions_taken


class AcmeAgent:
    """A made-up agent framework with its own quirky surface."""

    def __init__(self, model: str = MODEL, abilities: list[dict] | None = None,
                 use_abilities: bool = True) -> None:
        self.model = model
        self.abilities = abilities or []
        self.use_abilities = use_abilities

    def act(self, instruction: str) -> AcmeResult:
        import litellm

        tools = None
        if self.use_abilities and self.abilities:
            tools = [{"type": "function", "function": {
                "name": a["name"], "description": a.get("description") or a["name"],
                "parameters": a.get("parameters") or {"type": "object", "properties": {}}}}
                for a in self.abilities]
        litellm.drop_params = True
        resp = litellm.completion(model=self.model, temperature=0,
                                  messages=[{"role": "user", "content": instruction}],
                                  tools=tools, tool_choice="auto" if tools else None)
        msg = resp.choices[0].message
        actions = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            fn = tc.function
            try:
                params = json.loads(fn.arguments or "{}")
            except ValueError:
                params = {}
            actions.append(AcmeAction(fn.name, params))
        return AcmeResult(msg.content or "", actions)


# ===========================================================================
# THE KIT ADAPTER (bring-your-own) — the only glue a 3rd-party user writes.
# ===========================================================================
def _abilities(agent_input: Any) -> list[dict]:
    out = []
    for t in list(getattr(agent_input, "tools", None) or []):
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            f = t["function"]
            out.append({"name": f.get("name"), "description": f.get("description"),
                        "parameters": f.get("parameters")})
        elif t.get("name"):
            out.append({"name": t["name"], "description": t.get("description"),
                        "parameters": t.get("parameters")})
    return out


def _user_text(agent_input: Any) -> str:
    for m in reversed(list(getattr(agent_input, "messages", None) or [])):
        if isinstance(m, dict) and m.get("role") in (None, "user") and m.get("content"):
            return str(m["content"])
    return "What is the status of order 4821?"


def _adapt(result: AcmeResult) -> dict[str, Any]:
    return {"content": str(result.reply or ""),
            "tool_calls": [{"id": f"c{i}", "name": a.tool, "arguments": a.params}
                           for i, a in enumerate(result.actions_taken)]}


def run_agent(agent_input: Any) -> dict[str, Any]:
    agent = AcmeAgent(abilities=_abilities(agent_input), use_abilities=True)
    return _adapt(agent.act(_user_text(agent_input)))


def make_agent(model: str = MODEL, temperature: float = 0.0, bind_tools: bool = True):
    def _r(agent_input: Any) -> dict[str, Any]:
        agent = AcmeAgent(model=model, abilities=_abilities(agent_input), use_abilities=bind_tools)
        return _adapt(agent.act(_user_text(agent_input)))
    return _r


BUGGY_SRC = '''
def run_agent(agent_input):
    import json, litellm
    q = "What is the status of order 4821?"
    for m in reversed(list(getattr(agent_input, "messages", None) or [])):
        if isinstance(m, dict) and m.get("content"): q = str(m["content"]); break
    # BUG: ignores the agent's abilities (tools) entirely -> fabricates an answer
    litellm.drop_params = True
    resp = litellm.completion(model="gpt-4o-mini", temperature=0,
                              messages=[{"role":"user","content":q}])
    return {"content": resp.choices[0].message.content or "", "tool_calls": []}
'''

FIXED_SRC = '''
def run_agent(agent_input):
    import json, litellm
    q = "What is the status of order 4821?"
    for m in reversed(list(getattr(agent_input, "messages", None) or [])):
        if isinstance(m, dict) and m.get("content"): q = str(m["content"]); break
    tools = []
    for t in list(getattr(agent_input, "tools", None) or []):
        if not isinstance(t, dict): continue
        if t.get("type")=="function" and t.get("function"): tools.append(dict(t)); continue
        if t.get("name"): tools.append({"type":"function","function":{"name":t["name"],
            "description":t.get("description") or t["name"],
            "parameters":t.get("parameters") or {"type":"object","properties":{}}}})
    litellm.drop_params = True
    resp = litellm.completion(model="gpt-4o-mini", temperature=0,
                              messages=[{"role":"user","content":q}],
                              tools=tools or None, tool_choice="auto" if tools else None)
    msg = resp.choices[0].message
    tcs = []
    for i, tc in enumerate(getattr(msg, "tool_calls", None) or []):
        try: args = json.loads(tc.function.arguments or "{}")
        except Exception: args = {}
        tcs.append({"id": "c%d"%i, "name": tc.function.name, "arguments": args})
    return {"content": msg.content or "", "tool_calls": tcs}
'''
