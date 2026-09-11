"""Generic part-1 live loop: eval/sim + optimize + code-RSI for ONE framework.

Usage: fw_loop.py <framework_name>
Imports examples.frameworks.<framework_name>_agent, which must expose:
  run_agent(agent_input) -> {content, tool_calls}     # eval/sim
  make_agent(**kw) -> callable                          # optimize factory (bind_tools kw)
  BUGGY_SRC / FIXED_SRC : str                           # code-RSI source rewrite

Real LLM key from futureagi/.env; local FI key + sync/base_url/project from env.
Prints one JSON line: the framework x stage matrix row (PASS/FAIL + metrics)."""
from __future__ import annotations

import importlib
import json
import sys

from dotenv import load_dotenv

load_dotenv("/Users/nikhilpareek/Documents/futureAGI/code/core/future-agi/futureagi/.env", override=False)

from fi.alk import improve, tasks  # noqa: E402

KIT = "/Users/nikhilpareek/Documents/futureAGI/code/core/agent-learning-kit"


def _task(tid: str) -> dict:
    return {
        "id": tid, "title": tid,
        "world": {"kind": "tool_api", "spec": {"max_turns": 3}}, "difficulty": "medium",
        "objective": {"source": "declared", "evals": [
            {"eval": "task_success", "weight": 1.0, "anchor": True},
            {"eval": "tool_selection_accuracy", "weight": 0.8, "anchor": True}],
            "guards": {"sentinel_rows": [{"id": "s"}], "min_guard_count": 1}},
        "scenario": {"name": tid, "kind": "task", "dataset": [{"persona": {"name": "Pat"},
            "situation": "What is the status of order 4821?",
            "outcome": "Calls order_status and reports the status."}]},
        "verification": {"checks": [{"type": "contains", "value": "order"}], "threshold": 0.5},
        "environments": [{"type": "mock_tools", "data": {"tools": {"order_status": {
            "schema": {"description": "Look up an order's status by id.",
                       "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}}},
            "response": {"content": "Order 4821: shipped, arriving Tuesday.", "success": True}}}}}],
    }


def _ds() -> dict:
    return tasks.compile_task_dataset({
        "name": f"{FW}-live", "tasks": [_task("tr1"),_task("tr2"),_task("te1"),_task("te2"),_task("rg1"),_task("rg2")],
        "splits": {"train": ["tr1","tr2"], "test": ["te1","te2"], "regression": ["rg1","rg2"]}})


def main(fw: str) -> int:
    global FW
    FW = fw
    mod = importlib.import_module(f"examples.frameworks.{fw}_agent")
    ds = _ds()
    row: dict = {"framework": fw, "stages": {}}

    # eval/sim
    try:
        agent = {"type": "python", "callable": f"{KIT}/examples/frameworks/{fw}_agent.py:run_agent"}
        b = tasks.run_benchmark(ds, agent, seed=7, project_name="agent-learning")
        agg = b["aggregate"]
        row["stages"]["eval_sim"] = {"status": "PASS" if agg["pass_rate"] >= 0.5 else "FAIL",
            "pass_rate": agg["pass_rate"], "mean": agg["mean_score"],
            "tool_calls": sum(len(r.get("tool_calls") or []) for r in b["per_task"]),
            "url": b["telemetry"]["dashboard_url"], "tele": b["telemetry"]["status"]}
    except Exception as e:  # noqa: BLE001
        row["stages"]["eval_sim"] = {"status": "FAIL", "error": f"{type(e).__name__}: {e}"[:200]}

    # optimize (factory over bind_tools, detector ON; weak no-tool baseline)
    try:
        base = {"type": "framework", "framework": "callable", "factory": True,
                "target": f"{KIT}/examples/frameworks/{fw}_agent.py:make_agent",
                "factory_kwargs": {"bind_tools": False}}
        o = tasks.optimize_against_dataset(ds, base,
            {"agent.factory_kwargs": [{"bind_tools": True}, {"bind_tools": False}]},
            max_candidates=2, seed=7, project_name="agent-learning", detect_reward_hacks=True)
        ho = o["held_out"]
        row["stages"]["optimize"] = {"status": "PASS" if ho["improved"] else "FAIL",
            "winner": o["winner"]["assignment"], "baseline": ho["baseline_mean_score"],
            "winner_score": ho["winner_mean_score"], "lift": ho["lift"],
            "url": o["telemetry"]["dashboard_url"], "tele": o["telemetry"]["status"]}
    except Exception as e:  # noqa: BLE001
        row["stages"]["optimize"] = {"status": "FAIL", "error": f"{type(e).__name__}: {e}"[:200]}

    # code-RSI (buggy -> fixed source rewrite)
    try:
        rep = improve.improve_agent_code(source_text=mod.BUGGY_SRC, symbol="run_agent", dataset=ds,
            propose_patch=lambda d: mod.FIXED_SRC, objective=ds["tasks"][0]["objective"],
            threshold=0.5, project_name="agent-learning")
        lift = round(rep["held_out_final"] - rep["held_out_baseline"], 4)
        row["stages"]["code_rsi"] = {"status": "PASS" if (rep["fixed"] and lift > 0) else "FAIL",
            "fixed": rep["fixed"], "baseline": rep["held_out_baseline"],
            "final": rep["held_out_final"], "lift": lift, "regression_held": rep["regression_held"],
            "url": rep["telemetry"]["dashboard_url"], "tele": rep["telemetry"]["status"]}
    except Exception as e:  # noqa: BLE001
        row["stages"]["code_rsi"] = {"status": "FAIL", "error": f"{type(e).__name__}: {e}"[:200]}

    print("MATRIX_ROW " + json.dumps(row))
    return 0


if __name__ == "__main__":
    FW = ""
    sys.exit(main(sys.argv[1]))
