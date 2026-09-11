# Run the kit's loop on ANY framework (incl. your own 3rd-party agent)

The kit's `eval/sim → optimize → code-RSI` loop is **framework-agnostic**. It
drives your agent through one tiny contract; everything else (scoring, optimize,
code-RSI, dashboard) is the same regardless of framework.

## The contract (the only thing you write)

A Python callable:

```python
def run_agent(agent_input) -> dict:
    # agent_input.messages : list[{role, content, tool_calls?}]  (the turn history)
    # agent_input.tools    : list[ tool specs ]                  (available tools)
    return {
        "content": "<the agent's text answer>",
        "tool_calls": [{"id": "c0", "name": "<tool>", "arguments": {...}}],
    }
```

Point the kit at it:

```python
agent = {"type": "python", "callable": "/abs/path/my_agent.py:run_agent"}
tasks.run_benchmark(dataset, agent)            # eval/sim
```

That's it. Your framework can be **anything** — the kit never imports it.

## Three ways to register

| `agent.type` | when | how |
|---|---|---|
| `python` | simplest — any callable | `{"type":"python","callable":"file.py:run_agent"}` |
| `framework` + `framework:"callable"` | optimize over a config | add `"factory":true,"target":"file.py:make_agent","factory_kwargs":{...}`; the kit searches `factory_kwargs` |
| `framework` + a **registered** name | native trace labels | 58 names pre-registered (langchain, crewai, dspy, …) with method/input-mode; see `src/fi/simulate/agent/frameworks.py` |

For optimize, also expose a factory:

```python
def make_agent(model="gpt-4o-mini", temperature=0.0, bind_tools=True):
    def _run(agent_input): ...   # build your framework agent with these knobs
    return _run
```

For code-RSI, provide the buggy source + a proposer that returns the fixed source
(see any `*_agent.py` `BUGGY_SRC`/`FIXED_SRC`).

## Worked examples in this folder

- **`acme_thirdparty_agent.py`** — a **synthetic framework nobody has heard of**
  ("Acme", its own `AcmeAgent().act() -> AcmeResult` API) adapted in ~10 lines.
  Proof the path is **not** special-cased to famous names. Full loop runs green.
- `langchain_agent.py`, `langgraph_agent.py` — *kit-owned tool* shape (single
  model node; the kit executes the tool turn-by-turn).
- `pydantic_ai_agent.py`, `crewai_agent.py`, `openai_agents_agent.py`,
  `autogen_agent.py`, `llamaindex_agent.py` — *agent-owned tool* shape
  (autonomous frameworks: worker thread for their event loop + the tool records
  its own call).

## Run one

```bash
python examples/frameworks/run_live_loop.py <framework>     # e.g. acme_thirdparty
```

## Framework categories (not all are "agent frameworks")

The 58 registered adapters partition into:
- **Agent frameworks** — the loop runs *on* them (the examples above).
- **Model-provider clients** (anthropic, cohere, groq, ollama, bedrock, …) — the
  loop runs *through* them; they're the `model=` argument, not an agent. Swap the
  provider, same loop.
- **Protocol/env** (a2a, mcp, openenv, gymnasium) — step/reset & tool-session
  contracts; a different task shape.
- **Voice / CUA / browser** (livekit, pipecat, vapi, …, computer_use,
  browser_use) — infra-gated (audio / browser / computer), a separate proof tier.
