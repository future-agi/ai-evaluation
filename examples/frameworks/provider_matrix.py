"""Model-provider matrix: the SAME kit loop, the SAME agent, swap the model=
across providers (openai/anthropic/groq/xai) via litellm. Proves the loop runs
THROUGH any provider. eval/sim per provider (opt/code-RSI are provider-agnostic,
proven on openai). Emits to CH."""
from __future__ import annotations
import json
import sys
from dotenv import load_dotenv
load_dotenv("/Users/nikhilpareek/Documents/futureAGI/code/core/future-agi/futureagi/.env", override=False)
from fi.alk import tasks  # noqa: E402

PROVIDERS = [
    ("openai", "gpt-4o-mini"),
    ("anthropic", "anthropic/claude-haiku-4-5-20251001"),
    ("groq", "groq/llama-3.3-70b-versatile"),
    ("xai", "xai/grok-3-mini"),
]

def _task(tid):
    return {"id": tid, "title": tid, "world": {"kind": "tool_api", "spec": {"max_turns": 3}},
        "difficulty": "medium",
        "objective": {"source": "declared", "evals": [
            {"eval": "task_success", "weight": 1.0, "anchor": True},
            {"eval": "tool_selection_accuracy", "weight": 0.8, "anchor": True}],
            "guards": {"sentinel_rows": [{"id": "s"}], "min_guard_count": 1}},
        "scenario": {"name": tid, "kind": "task", "dataset": [{"persona": {"name": "Pat"},
            "situation": "What is the status of order 4821?", "outcome": "Calls order_status."}]},
        "verification": {"checks": [{"type": "contains", "value": "order"}], "threshold": 0.5},
        "environments": [{"type": "mock_tools", "data": {"tools": {"order_status": {
            "schema": {"description": "Look up an order's status by id.",
                       "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}}},
            "response": {"content": "Order 4821: shipped, arriving Tuesday.", "success": True}}}}}]}

ds = tasks.compile_task_dataset({"name": "provider-matrix", "tasks": [_task("t1"), _task("t2")]})

rows = []
for provider, model in PROVIDERS:
    try:
        agent = {"type": "framework", "framework": "callable", "factory": True,
                 "target": "/Users/nikhilpareek/Documents/futureAGI/code/core/agent-learning-kit/examples/frameworks/acme_thirdparty_agent.py:make_agent",
                 "factory_kwargs": {"model": model, "bind_tools": True}}
        b = tasks.run_benchmark(ds, agent, seed=7, project_name="agent-learning")
        agg = b["aggregate"]
        rows.append({"provider": provider, "model": model,
                     "status": "PASS" if agg["pass_rate"] >= 0.5 else "FAIL",
                     "pass_rate": agg["pass_rate"],
                     "tool_calls": sum(len(r.get("tool_calls") or []) for r in b["per_task"]),
                     "tele": b["telemetry"]["status"]})
    except Exception as e:  # noqa: BLE001
        rows.append({"provider": provider, "model": model, "status": "FAIL",
                     "error": f"{type(e).__name__}: {e}"[:160]})

print("PROVIDER_MATRIX " + json.dumps(rows))
sys.exit(0)
