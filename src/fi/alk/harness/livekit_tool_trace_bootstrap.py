"""ALK-owned ``sitecustomize`` hook for LiveKit function-tool executions.

The hosted simulator cannot observe an agent's private ``AgentSession`` events over the
LiveKit room. Repository agent interpreters therefore import this tiny hook when the bundle
declares ``HARNESS_TOOL_TRACE``. Using Python's ``sitecustomize`` mechanism is important:
LiveKit starts job executors in child Python processes, so a wrapper applied only to the
parent worker misses the ``AgentSession`` that actually executes tools. Tracing is strictly
best effort and must never alter agent behaviour.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _record(event: Any) -> None:
    destination = os.environ.get("HARNESS_TOOL_TRACE", "").strip()
    if not destination:
        return
    records: list[dict[str, Any]] = []
    try:
        pairs = event.zipped()
    except Exception:  # noqa: BLE001 - observability must never affect the target
        return
    for call, output in pairs:
        records.append(
            {
                "name": str(getattr(call, "name", "")),
                "arguments": getattr(call, "arguments", {}) or {},
                "output": getattr(output, "output", "") if output is not None else "",
                "is_error": bool(
                    output is not None and getattr(output, "is_error", False)
                ),
            }
        )
    if not records:
        return
    try:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as trace:
            for record in records:
                trace.write(json.dumps(record, default=str, sort_keys=True) + "\n")
    except OSError:
        return


def _install() -> None:
    try:
        from livekit.agents import AgentSession
    except Exception:  # noqa: BLE001 - non-LiveKit targets continue unchanged
        return
    original = AgentSession.__init__
    if getattr(original, "__alk_tool_trace__", False):
        return

    def traced_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        try:
            self.on("function_tools_executed", _record)
        except Exception:  # noqa: BLE001 - observability must never affect the target
            return

    traced_init.__alk_tool_trace__ = True  # type: ignore[attr-defined]
    AgentSession.__init__ = traced_init


_install()
