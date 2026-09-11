"""MCP live lane (3E) — real MCP server processes over the real protocol.

Framework imports: NONE at module top (P3-D1). The default tier spawns the
shipped loopback stdio server (``_workers/mcp_loopback_server.py`` — a real
``FastMCP`` process, credential-free but genuinely separate and speaking the
real protocol over the wire: that IS the live graduation, P3-D6/R§1 #12).
The client side is ``_workers/mcp_worker.py`` (a ``ClientSession`` over
stdio). Every artifact carries the server-behavior snapshot stamp
``{server_name, server_version, capability_hash}`` (R§1 #11).
"""

from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ._contract import lane_budget_s, require_lane_enabled
from ._runner import run_worker_once
from ._stats import lane_run_payload, primary_transcript_events, run_repeated

_WORKERS = Path(__file__).resolve().parent / "_workers"
_RUNG_LABELS = {1: "loopback_servers", 2: "third_party_servers"}

# Deterministic, credential-free default tool script against the loopback
# server (claim-level expectations, tolerant of alternative trajectories).
_DEFAULT_CALLS = (
    {"tool": "echo", "arguments": {"text": "hello loopback"}, "expect": {"contains": "hello loopback"}},
    {"tool": "add", "arguments": {"a": 2, "b": 3}, "expect": {"contains": "5"}},
)


def _scenario_calls(scenario: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = scenario.get("calls")
    if not raw:
        return [dict(call) for call in _DEFAULT_CALLS]
    return [dict(call) for call in raw if isinstance(call, Mapping)]


def _server_snapshot(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    for event in events:
        if event.get("type") == "server_snapshot":
            payload = event.get("payload")
            if isinstance(payload, Mapping):
                return dict(payload)
    return None


def _tool_session_state(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items = []
    for index, event in enumerate(events, start=1):
        if event.get("channel") == "tool":
            payload = event.get("payload")
            payload = payload if isinstance(payload, Mapping) else {}
            items.append(
                {
                    "index": index,
                    "item_type": event.get("type"),
                    "tool": payload.get("name"),
                    "ok": payload.get("ok"),
                }
            )
    return {
        "engine": "live_lane_mcp",
        "item_count": len(items),
        "items": items[:200],
    }


def run_mcp_lane(
    scenario: Mapping[str, Any],
    *,
    server: Optional[Mapping[str, Any]] = None,
    repeats: int = 8,
    required_env: Optional[Sequence[str]] = None,
    version_requirement: str | None = None,
    budget_s: float | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Default tier (server=None): loopback stdio server fixture + client.
    Third-party tier (server={"command": [...], "env_names": [...]}) is
    ``live_credentialed`` with server-specific names (P3-D6)."""

    require_lane_enabled("mcp")
    rung = 1 if server is None else 2
    if rung >= 2:
        require_lane_enabled("credentialed")

    if server is None:
        server_command = [sys.executable, str(_WORKERS / "mcp_loopback_server.py")]
        server_env_names: list[str] = []
    else:
        command = server.get("command")
        if not isinstance(command, Sequence) or not command:
            raise ValueError(
                "third-party server spec needs a non-empty 'command' list"
            )
        server_command = [str(part) for part in command]
        server_env_names = [str(name) for name in server.get("env_names") or []]
    required = tuple(
        required_env if required_env is not None else server_env_names
    )

    base_dir = (
        Path(artifacts_dir)
        if artifacts_dir is not None
        else Path(tempfile.mkdtemp(prefix="agent-learning-live-mcp-"))
    )
    run_id = uuid.uuid4().hex
    resolved_budget = float(budget_s) if budget_s is not None else lane_budget_s("mcp")
    boot = {
        "type": "boot",
        "lane": "mcp",
        "rung": rung,
        "scenario": {"name": str(scenario.get("name") or "mcp-loopback-smoke")},
        "config": {
            "server_command": server_command,
            "server_env_names": server_env_names,
            "calls": _scenario_calls(scenario),
        },
    }
    worker = _WORKERS / "mcp_worker.py"

    def _run_once(index: int, transcript: Any) -> dict[str, Any]:
        return run_worker_once(
            worker,
            boot,
            lane="mcp",
            required_env=required,
            cwd=base_dir,
            timeout_s=resolved_budget,
            transcript=transcript,
            version_requirement=version_requirement,
        )

    result = run_repeated(
        _run_once,
        lane="mcp",
        evidence_class="live_lane",
        repeats=repeats,
        budget_s=budget_s,
        required_env=required,
        artifacts_dir=base_dir,
        run_id=run_id,
        rung=_RUNG_LABELS[rung],
        framework="mcp",
        version_requirement=version_requirement,
    )

    events = primary_transcript_events(result)
    payload = lane_run_payload(
        result,
        name=f"live-mcp-{run_id[:8]}",
        scenario=scenario,
        states={
            "framework_runtime": {
                "framework": "mcp",
                "engine": "live_lane_mcp",
                "rung": _RUNG_LABELS[rung],
            },
            "mcp_tool_session": _tool_session_state(events),
        },
        metadata={"execution_model": "subprocess", "rung": _RUNG_LABELS[rung]},
    )
    snapshot = _server_snapshot(events)
    if snapshot is not None:
        payload["live_lane"]["server_snapshot"] = snapshot
    return payload
