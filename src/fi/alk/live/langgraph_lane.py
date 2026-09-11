"""LangChain/LangGraph live lane (3D) — real compiled graphs, checkpoints.

Two execution paths, selected by what the caller passes (P3-D1):

- **In-process** when the caller passes a live Python graph object (a
  ``CompiledStateGraph``) — the existing ``wrap_agent`` contract users
  already accept. Framework access happens through the object the caller
  built; any framework import here is lazy, inside function bodies only.
- **Subprocess** via ``_workers/langgraph_worker.py`` when the lane boots
  from a factory path (a dotted ``module:factory`` string): the worker
  imports the factory, compiles the graph, and runs the same turn script
  under the scrubbed-env subprocess model. The artifact records which
  execution model ran.
"""

from __future__ import annotations

import tempfile
import traceback as _traceback
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ._contract import lane_budget_s, require_lane_enabled
from ._runner import run_worker_once, version_preflight
from ._stats import (
    lane_run_payload,
    primary_transcript_events,
    run_repeated,
    step_signature_from_events,
)

_WORKERS = Path(__file__).resolve().parent / "_workers"
_RUNG_LABELS = {1: "scripted_local_model", 2: "credentialed_model"}

_DEFAULT_TURNS = (
    {"user": "Hello - what can you do?"},
    {"user": "Summarize our conversation so far."},
)


def _scenario_turns(scenario: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = scenario.get("turns") or scenario.get("user_messages")
    if not raw:
        return [dict(turn) for turn in _DEFAULT_TURNS]
    turns: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            turns.append({"user": item})
        elif isinstance(item, Mapping):
            turns.append(dict(item))
    return turns or [dict(turn) for turn in _DEFAULT_TURNS]


def _langgraph_version() -> str | None:
    try:
        import importlib.metadata

        return importlib.metadata.version("langgraph")
    except Exception:
        return None


def _turn_input(turn: Mapping[str, Any]) -> Any:
    if "input" in turn:
        return turn["input"]
    return {"messages": [{"role": "user", "content": str(turn.get("user") or "")}]}


def _last_message_text(output: Any) -> str:
    if isinstance(output, Mapping):
        messages = output.get("messages")
        if isinstance(messages, Sequence) and messages:
            last = messages[-1]
            content = getattr(last, "content", None)
            if content is None and isinstance(last, Mapping):
                content = last.get("content")
            if content is not None:
                return str(content)
        return str(output)
    return str(output)


def _turn_check(turn: Mapping[str, Any], reply: str) -> bool:
    expect = turn.get("expect")
    if isinstance(expect, Mapping) and isinstance(expect.get("contains"), str):
        return expect["contains"].lower() in reply.lower()
    return bool(reply.strip())


def _workflow_state(
    events: Sequence[Mapping[str, Any]], *, execution_model: str
) -> dict[str, Any]:
    items = []
    for index, event in enumerate(events, start=1):
        if event.get("channel") in ("user", "agent", "tool"):
            payload = event.get("payload")
            payload = payload if isinstance(payload, Mapping) else {}
            items.append(
                {
                    "index": index,
                    "channel": event.get("channel"),
                    "item_type": event.get("type"),
                    "text": payload.get("text"),
                }
            )
    return {
        "engine": "live_lane_langgraph",
        "execution_model": execution_model,
        "item_count": len(items),
        "items": items[:200],
    }


def run_langgraph_lane(
    graph_or_factory: Any,           # CompiledStateGraph object → in-process;
                                     # "pkg.module:make_graph" → subprocess
                                     #   via _workers/langgraph_worker.py (P3-D1)
    scenario: Mapping[str, Any],
    *,
    repeats: int = 8,
    checkpointer: Any | None = None, # in-process: a live checkpointer object;
                                     # subprocess: "memory" | "sqlite"
    cross_session_probe: bool = True,
    rung: int = 1,
    required_env: Optional[Sequence[str]] = None,
    version_requirement: str | None = None,
    budget_s: float | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    require_lane_enabled("langchain")
    if rung >= 2:
        require_lane_enabled("credentialed")
    if rung not in _RUNG_LABELS:
        raise ValueError(f"rung must be one of {sorted(_RUNG_LABELS)}, got {rung}")

    required = tuple(required_env) if required_env is not None else ()
    turns = _scenario_turns(scenario)
    base_dir = (
        Path(artifacts_dir)
        if artifacts_dir is not None
        else Path(tempfile.mkdtemp(prefix="agent-learning-live-langgraph-"))
    )
    run_id = uuid.uuid4().hex
    resolved_budget = (
        float(budget_s) if budget_s is not None else lane_budget_s("langchain")
    )
    subprocess_path = isinstance(graph_or_factory, str)
    execution_model = "subprocess" if subprocess_path else "in_process"

    if subprocess_path:
        if checkpointer is not None and not isinstance(checkpointer, str):
            raise ValueError(
                "the subprocess (factory) path takes checkpointer as a string "
                "('memory' or 'sqlite'); live checkpointer objects cannot "
                "cross the process boundary"
            )
        boot = {
            "type": "boot",
            "lane": "langchain",
            "rung": rung,
            "scenario": {"name": str(scenario.get("name") or "langgraph-smoke")},
            "turns": turns,
            "config": {
                "factory": graph_or_factory,
                "checkpointer": checkpointer or "memory",
                "cross_session_probe": bool(cross_session_probe),
                "probe": scenario.get("probe"),
                "thread_id": f"live-{run_id[:8]}",
            },
        }
        worker = _WORKERS / "langgraph_worker.py"

        def _run_once(index: int, transcript: Any) -> dict[str, Any]:
            return run_worker_once(
                worker,
                boot,
                lane="langchain",
                required_env=required,
                cwd=base_dir,
                timeout_s=resolved_budget,
                transcript=transcript,
                version_requirement=version_requirement,
            )

    else:
        graph = graph_or_factory

        def _run_once(index: int, transcript: Any) -> dict[str, Any]:
            # In-process path: the caller's live graph object, the accepted
            # wrap_agent contract. Verification is programmatic per turn.
            version = _langgraph_version()
            preflight = version_preflight(
                version_requirement,
                {
                    "framework": "langgraph",
                    "framework_version": version,
                    "capability_hash": None,
                },
            )
            transcript.record(
                "lane",
                "framework_ready",
                {
                    "framework": "langgraph",
                    "framework_version": version,
                    "capability_hash": None,
                    "package_paths": [],
                    "execution_model": "in_process",
                },
            )
            row: dict[str, Any] = {
                "transcript_path": str(transcript.path),
                "version": preflight,
            }
            if not preflight["version_ok"]:
                row.update(
                    passed=None,
                    score=None,
                    failure_layer="lane_infra",
                    void_reason=preflight["void_reason"],
                    detail=str(preflight["void_reason"]),
                )
                return row
            thread_id = f"live-{run_id[:8]}-r{index}"
            config = {"configurable": {"thread_id": thread_id}}
            checks: list[bool] = []
            try:
                for turn_index, turn in enumerate(turns):
                    transcript.record(
                        "user",
                        "message",
                        {"turn": turn_index, "text": str(turn.get("user") or "")},
                    )
                    output = graph.invoke(_turn_input(turn), config=config)
                    reply = _last_message_text(output)
                    transcript.record(
                        "agent", "message", {"turn": turn_index, "text": reply}
                    )
                    checks.append(_turn_check(turn, reply))
                probe = scenario.get("probe")
                if cross_session_probe and isinstance(probe, Mapping):
                    # Same-object cross-session probe: state must survive a
                    # second session on the same thread. The full
                    # discard-and-rebuild probe needs a factory — that is
                    # the subprocess path's job (guide §3.3).
                    inject = str(probe.get("inject") or "")
                    question = str(probe.get("question") or "What do you remember?")
                    if inject:
                        transcript.record(
                            "user", "message", {"session": 1, "text": inject}
                        )
                        graph.invoke(_turn_input({"user": inject}), config=config)
                    transcript.record(
                        "user", "message", {"session": 2, "text": question}
                    )
                    output = graph.invoke(_turn_input({"user": question}), config=config)
                    reply = _last_message_text(output)
                    transcript.record(
                        "agent", "message", {"session": 2, "text": reply}
                    )
                    fired = (
                        str(probe.get("assert_contains") or "").lower()
                        in reply.lower()
                        if probe.get("assert_contains")
                        else bool(reply.strip())
                    )
                    contained = (
                        str(probe.get("assert_not_contains") or "").lower()
                        not in reply.lower()
                        if probe.get("assert_not_contains")
                        else True
                    )
                    checks.append(fired and contained)
                    transcript.record(
                        "lane",
                        "cross_session_probe",
                        {
                            "probe_mode": "same_object",
                            "fired": fired,
                            "contained": contained,
                        },
                    )
            except Exception as exc:
                transcript.record(
                    "lane",
                    "worker_error",
                    {"traceback": _traceback.format_exc()},
                )
                row.update(
                    passed=False,
                    score=0.0,
                    failure_layer="framework_runtime",
                    detail=f"graph invoke raised: {exc}",
                    step_signature=step_signature_from_events(transcript.events),
                )
                return row
            passed = bool(checks) and all(checks)
            transcript.record(
                "lane", "verification", {"passed": passed, "checks": checks}
            )
            row.update(
                passed=passed,
                score=1.0 if passed else 0.0,
                failure_layer=None if passed else "agent_behavior",
                detail="" if passed else "programmatic turn checks failed",
                step_signature=step_signature_from_events(transcript.events),
            )
            return row

    result = run_repeated(
        _run_once,
        lane="langchain",
        evidence_class="live_lane",
        repeats=repeats,
        budget_s=budget_s,
        required_env=required,
        artifacts_dir=base_dir,
        run_id=run_id,
        rung=_RUNG_LABELS[rung],
        framework="langgraph",
        version_requirement=version_requirement,
    )

    events = primary_transcript_events(result)
    payload = lane_run_payload(
        result,
        name=f"live-langgraph-{run_id[:8]}",
        scenario=scenario,
        states={
            "workflow_trace": _workflow_state(events, execution_model=execution_model)
        },
        metadata={
            "execution_model": execution_model,
            "rung": _RUNG_LABELS[rung],
            "cross_session_probe": bool(cross_session_probe),
        },
    )
    return payload
