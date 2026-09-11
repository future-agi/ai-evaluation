"""LangGraph lane worker (3D, manifest/factory path) — untrusted subprocess
entry (P3-D1).

Imports the caller's ``module:factory``, compiles the REAL graph against a
real checkpoint store (MemorySaver or SqliteSaver in the run tempdir), runs
the turn script via ``invoke`` on the same thread_id, and executes the
cross-session probe (R§1 #6): session 1 injects via the persistence channel,
the graph object is DISCARDED and REBUILT against the same checkpointer,
session 2 asserts firing/containment on the same thread. End-state diffs of
the checkpoint store are emitted as ``lane/end_state_diff`` (R§1 #14).

IPC: see livekit_worker.py — same one-boot-line / JSONL-events contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from typing import Any


def _emit(channel: str, type_: str, payload: dict[str, Any]) -> None:
    print(
        json.dumps(
            {"channel": channel, "type": type_, "payload": payload},
            ensure_ascii=False,
            default=str,
        ),
        flush=True,
    )


def _read_boot() -> dict[str, Any]:
    line = sys.stdin.readline()
    if not line.strip():
        raise RuntimeError("missing boot message on stdin")
    boot = json.loads(line)
    if not isinstance(boot, dict) or boot.get("type") != "boot":
        raise RuntimeError("first stdin line must be a boot message")
    return boot


def _capability_hash(framework: str, version: str) -> str:
    return hashlib.sha256(f"{framework}:{version}".encode("utf-8")).hexdigest()


def _package_paths(module: Any) -> list[str]:
    """Filesystem roots of a framework package for traceback attribution.

    langgraph (like livekit) ships as a NAMESPACE package: ``__file__`` is
    None and the roots live on ``__path__`` instead.
    """

    file = getattr(module, "__file__", None)
    if file:
        return [os.path.dirname(file)]
    return [str(path) for path in getattr(module, "__path__", None) or []]


def _turn_input(turn: dict[str, Any]) -> Any:
    if "input" in turn:
        return turn["input"]
    return {"messages": [{"role": "user", "content": str(turn.get("user") or "")}]}


def _last_message_text(output: Any) -> str:
    if isinstance(output, dict):
        messages = output.get("messages")
        if isinstance(messages, (list, tuple)) and messages:
            last = messages[-1]
            content = getattr(last, "content", None)
            if content is None and isinstance(last, dict):
                content = last.get("content")
            if content is not None:
                return str(content)
        return str(output)
    return str(output)


def _run(boot: dict[str, Any]) -> None:
    import importlib.metadata

    import langgraph

    version = importlib.metadata.version("langgraph")
    package_paths = _package_paths(langgraph)
    try:
        import langchain_core

        package_paths.extend(_package_paths(langchain_core))
    except ImportError:
        pass
    _emit(
        "lane",
        "framework_ready",
        {
            "framework": "langgraph",
            "framework_version": version,
            "capability_hash": _capability_hash("langgraph", version),
            "package_paths": package_paths,
            "execution_model": "subprocess",
        },
    )
    config = boot.get("config") or {}
    factory_path = str(config.get("factory") or "")
    module_name, _, attr = factory_path.partition(":")
    if not module_name or not attr:
        raise RuntimeError(f"factory must be 'module:attr', got {factory_path!r}")
    factory = getattr(importlib.import_module(module_name), attr)

    checkpointer_kind = str(config.get("checkpointer") or "memory")
    if checkpointer_kind == "sqlite":
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        connection = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
        checkpointer = SqliteSaver(connection)
    else:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()

    def _build_graph() -> Any:
        try:
            candidate = factory(checkpointer=checkpointer)
        except TypeError:
            candidate = factory()
        if hasattr(candidate, "compile"):
            candidate = candidate.compile(checkpointer=checkpointer)
        return candidate

    thread_id = str(config.get("thread_id") or "live-thread")
    invoke_config = {"configurable": {"thread_id": thread_id}}

    def _checkpoint_count() -> int | None:
        try:
            return sum(1 for _ in checkpointer.list(invoke_config))
        except Exception:
            return None

    graph = _build_graph()
    checkpoints_before = _checkpoint_count()
    checks: list[bool] = []
    turns = boot.get("turns") or []
    for index, turn in enumerate(turns):
        turn = turn or {}
        text = str(turn.get("user") or "")
        _emit("user", "message", {"turn": index, "text": text, "session": 1})
        output = graph.invoke(_turn_input(turn), config=invoke_config)
        reply = _last_message_text(output)
        _emit("agent", "message", {"turn": index, "text": reply, "session": 1})
        expect = turn.get("expect")
        ok = bool(reply.strip())
        if isinstance(expect, dict) and isinstance(expect.get("contains"), str):
            ok = ok and expect["contains"].lower() in reply.lower()
        checks.append(ok)

    probe = config.get("probe")
    if config.get("cross_session_probe") and isinstance(probe, dict):
        inject = str(probe.get("inject") or "")
        question = str(probe.get("question") or "What do you remember?")
        if inject:
            _emit("user", "message", {"session": 1, "text": inject, "probe": True})
            graph.invoke(_turn_input({"user": inject}), config=invoke_config)
        # Discard and REBUILD against the same checkpointer — the process
        # crosses a real persistence boundary, not an in-memory alias.
        del graph
        graph = _build_graph()
        _emit("user", "message", {"session": 2, "text": question, "probe": True})
        output = graph.invoke(_turn_input({"user": question}), config=invoke_config)
        reply = _last_message_text(output)
        _emit("agent", "message", {"session": 2, "text": reply, "probe": True})
        fired = True
        if isinstance(probe.get("assert_contains"), str):
            fired = probe["assert_contains"].lower() in reply.lower()
        contained = True
        if isinstance(probe.get("assert_not_contains"), str):
            contained = probe["assert_not_contains"].lower() not in reply.lower()
        _emit(
            "lane",
            "cross_session_probe",
            {"probe_mode": "rebuilt", "fired": fired, "contained": contained},
        )
        checks.append(fired and contained)

    checkpoints_after = _checkpoint_count()
    _emit(
        "lane",
        "end_state_diff",
        {
            "checkpoint_store": checkpointer_kind,
            "checkpoints_before": checkpoints_before,
            "checkpoints_after": checkpoints_after,
            "thread_id": thread_id,
        },
    )
    passed = bool(checks) and all(checks)
    _emit("lane", "verification", {"passed": passed, "checks": checks})


def main() -> int:
    boot = _read_boot()
    try:
        _run(boot)
    except Exception:
        _emit("lane", "worker_error", {"traceback": traceback.format_exc()})
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
