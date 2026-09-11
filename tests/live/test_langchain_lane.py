"""LangChain/LangGraph live lane suite (3D) — opt-in, env-gated (guide §5.2).

Collected in every env, SKIPPED unless AGENT_LEARNING_LIVE_LANGCHAIN=1 (the
conftest three-fact reason). The real-graph tests need the `langchain` extra
(real langgraph import, real checkpointer); the stub-graph round-trip test
exercises the in-process wrap_agent contract without any framework import.
"""

from __future__ import annotations

import dataclasses
import json
import os
import textwrap
from typing import Any

import pytest

pytestmark = [pytest.mark.live_lane, pytest.mark.live_langchain]

_PROBE = {
    "inject": "Remember this passphrase: teal-anchor-42.",
    "question": "What passphrase do you remember?",
    "assert_contains": "teal-anchor-42",
    "assert_not_contains": "REFUSED-CANARY",
}

_SCRIPTED_SCENARIO = {
    "name": "langgraph-rung1-smoke",
    "turns": [
        {"user": "Hello - what can you do?", "expect": {"contains": "what can you do"}},
        {"user": "Summarize our conversation so far.", "expect": {"contains": "summarize"}},
    ],
    "probe": _PROBE,
}

_FACTORY_MODULE = textwrap.dedent(
    '''
    """Deterministic memory-echo LangGraph factory for the lane subprocess."""

    from typing import Any

    from langgraph.graph import END, START, MessagesState, StateGraph


    def _respond(state):
        messages = state["messages"]
        last = getattr(messages[-1], "content", str(messages[-1]))
        human = [
            str(getattr(message, "content", ""))
            for message in messages
            if getattr(message, "type", "") == "human"
        ]
        reply = "echo: %s | memory: %s" % (last, " ; ".join(human))
        return {"messages": [{"role": "assistant", "content": reply}]}


    def make_graph(checkpointer: Any = None):
        builder = StateGraph(MessagesState)
        builder.add_node("respond", _respond)
        builder.add_edge(START, "respond")
        builder.add_edge("respond", END)
        return builder  # the worker compiles it against ITS checkpointer
    '''
)


class _StubGraph:
    """Framework-free stand-in honoring the CompiledStateGraph invoke shape:
    per-thread memory keyed by configurable.thread_id (the wrap_agent
    contract the in-process path accepts)."""

    def __init__(self) -> None:
        self._threads: dict[str, list[str]] = {}

    def invoke(self, value: Any, config: Any = None) -> dict[str, Any]:
        thread = str(
            ((config or {}).get("configurable") or {}).get("thread_id", "t")
        )
        history = self._threads.setdefault(thread, [])
        messages = list((value or {}).get("messages") or [])
        text = str(messages[-1].get("content", "")) if messages else ""
        history.append(text)
        reply = "echo: " + text + " | memory: " + " ; ".join(history)
        return {
            "messages": [*messages, {"role": "assistant", "content": reply}],
        }


def _build_real_graph():
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph

    def respond(state):
        messages = state["messages"]
        last = getattr(messages[-1], "content", str(messages[-1]))
        human = [
            str(getattr(message, "content", ""))
            for message in messages
            if getattr(message, "type", "") == "human"
        ]
        reply = "echo: %s | memory: %s" % (last, " ; ".join(human))
        return {"messages": [{"role": "assistant", "content": reply}]}

    builder = StateGraph(MessagesState)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=MemorySaver())


def _transcript_events(result: dict[str, Any]) -> list[dict[str, Any]]:
    from fi.alk.live._transcript import read_transcript

    rows = [
        row
        for row in result["live_lane"]["per_repeat"]
        if not row.get("quarantined")
    ] or list(result["live_lane"]["per_repeat"])
    return read_transcript(str(rows[0]["transcript_path"]))


def test_lane_refuses_without_env_flag(monkeypatch):
    from fi.alk.live import _contract, langgraph_lane

    monkeypatch.delenv("AGENT_LEARNING_LIVE_LANGCHAIN", raising=False)
    with pytest.raises(_contract.LaneDisabledError):
        langgraph_lane.run_langgraph_lane(_StubGraph(), {"name": "smoke"})


def test_rung1_in_process_real_graph_repeats_and_attributes():
    from fi.alk.live import _contract, langgraph_lane

    graph = _build_real_graph()  # real langgraph import, real checkpointer
    result = langgraph_lane.run_langgraph_lane(
        graph, _SCRIPTED_SCENARIO, repeats=3, cross_session_probe=True
    )

    assert result["live_lane"]["evidence_class"] == "live_lane"
    assert result["live_lane"]["verdict"] == "pass"
    assert result["live_lane"]["repeats"] == 3
    assert result["live_lane"]["icc"] == 1.0  # deterministic echo graph
    assert result["live_lane"]["framework"] == "langgraph"
    assert result["live_lane"]["framework_version"]
    assert result["metadata"]["execution_model"] == "in_process"
    for repeat in result["live_lane"]["per_repeat"]:
        assert repeat["passed"] is True
        assert repeat.get("failure_layer") in (None, *_contract.FAILURE_LAYERS)
    assert all(
        repeat["failure_layer"] != "lane_infra" or repeat.get("quarantined")
        for repeat in result["live_lane"]["per_repeat"]
    )
    # the same-object cross-session probe fired and contained
    probes = [
        event
        for event in _transcript_events(result)
        if event.get("type") == "cross_session_probe"
    ]
    assert probes and probes[-1]["payload"]["probe_mode"] == "same_object"
    assert probes[-1]["payload"]["fired"] is True
    assert probes[-1]["payload"]["contained"] is True


def test_factory_subprocess_rung1_real_sqlite_checkpointer(
    tmp_path, monkeypatch
):
    import langgraph  # noqa: F401 — flag set + extra missing must ERROR, not skip

    from fi.alk.live import _runner, langgraph_lane

    factory_dir = tmp_path / "factory"
    factory_dir.mkdir()
    (factory_dir / "live_lane_factory_mod.py").write_text(
        _FACTORY_MODULE, encoding="utf-8"
    )
    real_pythonpath = _runner.kit_pythonpath()
    monkeypatch.setattr(
        _runner,
        "kit_pythonpath",
        lambda: os.pathsep.join([real_pythonpath, str(factory_dir)]),
    )

    result = langgraph_lane.run_langgraph_lane(
        "live_lane_factory_mod:make_graph",
        _SCRIPTED_SCENARIO,
        repeats=2,
        checkpointer="sqlite",
        cross_session_probe=True,
        artifacts_dir=tmp_path / "artifacts",
    )

    assert result["live_lane"]["verdict"] == "pass"
    assert result["metadata"]["execution_model"] == "subprocess"
    assert result["live_lane"]["framework"] == "langgraph"
    assert result["live_lane"]["framework_version"]
    for repeat in result["live_lane"]["per_repeat"]:
        assert repeat["passed"] is True
        assert repeat["quarantined"] is False
    # end-state diff of the REAL checkpoint store (R§1 #14)
    diff = result["live_lane"]["end_state_diff"]
    assert diff is not None
    assert diff["checkpoint_store"] == "sqlite"
    assert (tmp_path / "artifacts" / "checkpoints.sqlite").is_file()


def test_cross_session_probe_rebuilt_graph_fires_and_contains(
    tmp_path, monkeypatch
):
    import langgraph  # noqa: F401 — flag set + extra missing must ERROR, not skip

    from fi.alk.live import _runner, langgraph_lane

    factory_dir = tmp_path / "factory"
    factory_dir.mkdir()
    (factory_dir / "live_lane_factory_mod.py").write_text(
        _FACTORY_MODULE, encoding="utf-8"
    )
    real_pythonpath = _runner.kit_pythonpath()
    monkeypatch.setattr(
        _runner,
        "kit_pythonpath",
        lambda: os.pathsep.join([real_pythonpath, str(factory_dir)]),
    )

    result = langgraph_lane.run_langgraph_lane(
        "live_lane_factory_mod:make_graph",
        _SCRIPTED_SCENARIO,
        repeats=1,
        checkpointer="sqlite",
        cross_session_probe=True,
        artifacts_dir=tmp_path / "artifacts",
    )

    assert result["live_lane"]["verdict"] == "pass"
    events = _transcript_events(result)
    probes = [
        event for event in events if event.get("type") == "cross_session_probe"
    ]
    # The worker DISCARDED and REBUILT the graph against the same
    # checkpointer before session 2 (probe_mode "rebuilt", R§1 #6).
    assert probes and probes[-1]["payload"]["probe_mode"] == "rebuilt"
    assert probes[-1]["payload"]["fired"] is True
    assert probes[-1]["payload"]["contained"] is True
    session2 = [
        event
        for event in events
        if event.get("channel") == "agent"
        and (event.get("payload") or {}).get("session") == 2
    ]
    assert session2 and "teal-anchor-42" in str(session2[-1]["payload"]["text"])


def test_captured_fixture_round_trip_offline_stub(tmp_path):
    """live run -> capture candidate -> simulated review -> replay green
    (guide §5.4 pattern; the stub graph keeps this framework-free)."""

    from fi.alk.live import _capture, _stats, langgraph_lane

    result = langgraph_lane.run_langgraph_lane(
        _StubGraph(),
        _SCRIPTED_SCENARIO,
        repeats=2,
        cross_session_probe=True,
        artifacts_dir=tmp_path / "artifacts",
    )
    assert result["live_lane"]["verdict"] == "pass"

    fields = {field.name for field in dataclasses.fields(_stats.LaneRunResult)}
    lane_result = _stats.LaneRunResult(
        **{
            key: value
            for key, value in result["live_lane"].items()
            if key in fields
        }
    )
    candidate = _capture.capture_to_fixture(
        lane_result,
        output=tmp_path / "candidates" / "langgraph.fixture.json",
        scenario=result.get("scenario"),
    )
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    assert payload["evidence_class"] == "live_lane"  # source class kept
    assert payload["capture"]["reviewed"] is False

    # Simulated review: rewrite reviewed:true into a tmp copy and replay.
    payload["evidence_class"] = "captured_fixture"
    payload["capture"]["reviewed"] = True
    payload["capture"]["reviewer"] = "test-reviewer"
    reviewed_copy = tmp_path / "reviewed" / "langgraph.fixture.json"
    reviewed_copy.parent.mkdir(parents=True)
    reviewed_copy.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    replay = _capture.replay_fixture(reviewed_copy)
    assert replay["verdict"] == "pass"
    assert replay["evidence_class"] == "captured_fixture"
    assert all(replay["checks"].values())
