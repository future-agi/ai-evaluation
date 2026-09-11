"""A2A live lane (3E) — one adapter over heterogeneous A2A peers.

Framework imports: NONE at module top (P3-D1). The default tier is a
loopback peer pair: ``_workers/a2a_worker.py`` in client mode spawns its own
peer-mode sibling on 127.0.0.1 and walks the protocol stages — card
discovery → task lifecycle → artifact exchange (R§1 #18). Remote peers are
``live_credentialed``. Live red-team scenarios point the existing corpus at
these targets; there is NO separate red-team marker.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ._contract import lane_budget_s, require_lane_enabled
from ._runner import run_worker_once
from ._stats import lane_run_payload, primary_transcript_events, run_repeated

_WORKERS = Path(__file__).resolve().parent / "_workers"
_RUNG_LABELS = {1: "loopback_peers", 2: "external_peers"}

_DEFAULT_STAGES = ("card_discovery", "task_lifecycle", "artifact_exchange")


def _scenario_stages(scenario: Mapping[str, Any]) -> list[str]:
    raw = scenario.get("stages")
    if not raw:
        return list(_DEFAULT_STAGES)
    stages = [str(stage) for stage in raw if str(stage) in _DEFAULT_STAGES]
    return stages or list(_DEFAULT_STAGES)


def _protocol_state(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items = []
    for index, event in enumerate(events, start=1):
        if event.get("channel") in ("agent", "tool", "user"):
            payload = event.get("payload")
            payload = payload if isinstance(payload, Mapping) else {}
            items.append(
                {
                    "index": index,
                    "channel": event.get("channel"),
                    "item_type": event.get("type"),
                    "stage": payload.get("stage"),
                    "ok": payload.get("ok"),
                }
            )
    return {
        "engine": "live_lane_a2a",
        "item_count": len(items),
        "items": items[:200],
    }


def run_a2a_lane(
    scenario: Mapping[str, Any],
    *,
    peer: Optional[str] = None,
    repeats: int = 8,
    required_env: Optional[Sequence[str]] = None,
    version_requirement: str | None = None,
    budget_s: float | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Default tier (peer=None): loopback peer pair. A remote peer URL is
    the ``live_credentialed`` tier."""

    require_lane_enabled("a2a")
    rung = 1 if peer is None else 2
    if rung >= 2:
        require_lane_enabled("credentialed")

    required = tuple(required_env) if required_env is not None else ()
    base_dir = (
        Path(artifacts_dir)
        if artifacts_dir is not None
        else Path(tempfile.mkdtemp(prefix="agent-learning-live-a2a-"))
    )
    run_id = uuid.uuid4().hex
    resolved_budget = float(budget_s) if budget_s is not None else lane_budget_s("a2a")
    boot = {
        "type": "boot",
        "lane": "a2a",
        "rung": rung,
        "mode": "client",
        "scenario": {"name": str(scenario.get("name") or "a2a-loopback-smoke")},
        "config": {
            "peer_url": peer,
            "stages": _scenario_stages(scenario),
            "message": str(scenario.get("message") or "ping from the harness"),
        },
    }
    worker = _WORKERS / "a2a_worker.py"

    def _run_once(index: int, transcript: Any) -> dict[str, Any]:
        return run_worker_once(
            worker,
            boot,
            lane="a2a",
            required_env=required,
            cwd=base_dir,
            timeout_s=resolved_budget,
            transcript=transcript,
            version_requirement=version_requirement,
        )

    result = run_repeated(
        _run_once,
        lane="a2a",
        evidence_class="live_lane",
        repeats=repeats,
        budget_s=budget_s,
        required_env=required,
        artifacts_dir=base_dir,
        run_id=run_id,
        rung=_RUNG_LABELS[rung],
        framework="a2a-sdk",
        version_requirement=version_requirement,
    )

    events = primary_transcript_events(result)
    return lane_run_payload(
        result,
        name=f"live-a2a-{run_id[:8]}",
        scenario=scenario,
        states={
            "framework_runtime": {
                "framework": "a2a",
                "engine": "live_lane_a2a",
                "rung": _RUNG_LABELS[rung],
            },
            "protocol_trace": _protocol_state(events),
        },
        metadata={"execution_model": "subprocess", "rung": _RUNG_LABELS[rung]},
    )
