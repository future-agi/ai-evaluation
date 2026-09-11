"""Code-level RSI — fix a framework agent's actual SOURCE CODE (not config).

The general self-improvement loop: run the agent's real source in sim → trace →
diagnose (detector) → PATCH THE SOURCE → re-run → keep only if held-out improves
AND a regression split holds. This is ACTUAL CODE MODIFICATION (the buggy source
is rewritten), distinct from config-selection (optimize_against_dataset).

Mechanics tests use a deterministic proposer (the loop closes on a real source
rewrite, held-out + no-regression verified); the LLM-finds-the-fix path is
key-gated (skips without OPENAI_API_KEY).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fi.alk import improve, tasks

BUGGY = Path(__file__).parent.parent / "examples" / "rsi_fixtures" / "buggy_tool_agent.py"

FIXED_SRC = (
    "def run_agent(agent_input):\n"
    "    tools = list(getattr(agent_input, 'tools', None) or [])\n"
    "    if tools:\n"
    "        name = tools[0].get('name') or (tools[0].get('function') or {}).get('name')\n"
    "        return {'content': 'Checking the tool.', 'tool_calls': [{'id': 'c1', 'name': name, 'arguments': {}}]}\n"
    "    return {'content': 'no tools', 'tool_calls': []}\n"
)


def _task(tid: str) -> dict:
    return {
        "id": tid, "title": tid, "world": {"kind": "tool_api", "spec": {"max_turns": 3}},
        "difficulty": "medium",
        "objective": {"source": "declared", "evals": [
            {"eval": "task_success", "weight": 1.0, "anchor": True},
            {"eval": "tool_selection_accuracy", "weight": 0.8, "anchor": True}],
            "guards": {"sentinel_rows": [{"id": "s"}], "min_guard_count": 1}},
        "scenario": {"name": tid, "kind": "task", "dataset": [{"persona": {"name": "P"},
            "situation": "Look up order 4821 status.", "outcome": "Calls the tool and reports status."}]},
        "verification": {"checks": [{"type": "contains", "value": "order"}], "threshold": 0.5},
        "environments": [{"type": "mock_tools", "data": {"tools": {"order_status": {
            "schema": {"description": "order status", "parameters": {"type": "object", "properties": {}}},
            "response": {"content": "Order 4821: shipped.", "success": True}}}}}],
    }


def _dataset() -> dict:
    return tasks.compile_task_dataset({
        "name": "code-rsi", "tasks": [_task("tr1"), _task("te1"), _task("rg1")],
        "splits": {"train": ["tr1"], "test": ["te1"], "regression": ["rg1"]}})


# --- the detector signal that makes the no-tool bug detectable ---------------
def test_tool_anchor_unmet_signal_fires() -> None:
    from fi.alk import rewardhack
    obj = {"evals": [{"eval": "tool_selection_accuracy", "anchor": True}]}
    # tool-anchored objective + ZERO tool calls -> caught (vacuous tool_selection
    # _accuracy=1.0 would otherwise hide it)
    hacked = {"metric_averages": {"task_completion": 0.2, "tool_selection_accuracy": 1.0},
              "tool_calls": [], "score": 0.55}
    v = rewardhack.score_trajectory(hacked, objective=obj)
    assert v["hacked"] is True
    assert "tool_anchor_unmet" in [s["kind"] for s in v["signals"]]
    # made a tool call -> not flagged
    ok = {"metric_averages": {"task_completion": 0.8, "tool_selection_accuracy": 1.0},
          "tool_calls": [{"name": "x"}], "score": 0.8}
    assert "tool_anchor_unmet" not in [s["kind"] for s in rewardhack.score_trajectory(ok, objective=obj)["signals"]]


# --- the loop closes on a real source rewrite (deterministic proposer) -------
@pytest.mark.integration
def test_code_rsi_fixes_real_source_with_deterministic_patch() -> None:
    ds = _dataset()
    obj = ds["tasks"][0]["objective"]
    report = improve.improve_agent_code(
        source_text=BUGGY.read_text(), symbol="run_agent", dataset=ds,
        propose_patch=lambda diagnosis: FIXED_SRC, objective=obj, threshold=0.5)
    assert report["fixed"] is True
    assert report["held_out_final"] > report["held_out_baseline"] + 0.2   # real held-out lift
    assert report["regression_held"] is True
    assert "def run_agent" in report["accepted_source"]                   # ACTUAL code change


def test_code_rsi_rejects_a_noop_patch() -> None:
    # a proposer that returns the same buggy source -> no fix, honest null.
    ds = _dataset()
    obj = ds["tasks"][0]["objective"]
    report = improve.improve_agent_code(
        source_text=BUGGY.read_text(), symbol="run_agent", dataset=ds,
        propose_patch=lambda diagnosis: BUGGY.read_text(), objective=obj, max_rounds=1)
    assert report["fixed"] is False


@pytest.mark.integration
def test_code_rsi_llm_finds_the_fix_from_the_trace() -> None:
    """The real RSI claim: the MODEL (not the test) derives the code fix from the
    trace + eval, with error-feedback across rounds. Key-gated."""
    if not (os.environ.get("OPENAI_API_KEY") or "").strip():
        pytest.skip("OPENAI_API_KEY not set")
    ds = _dataset()
    obj = ds["tasks"][0]["objective"]
    report = improve.improve_agent_code(
        source_text=BUGGY.read_text(), symbol="run_agent", dataset=ds,
        propose_patch=improve.propose_patch_via_llm("gpt-4o-mini"),
        objective=obj, threshold=0.5, max_rounds=3)
    assert report["fixed"] is True
    assert report["held_out_final"] > report["held_out_baseline"] + 0.2
    assert report["regression_held"] is True
