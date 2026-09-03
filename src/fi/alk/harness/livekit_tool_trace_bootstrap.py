"""ALK-owned ``sitecustomize`` hook for LiveKit function-tool executions.

The hosted simulator cannot observe an agent's private ``AgentSession`` events over the
LiveKit room. Repository agent interpreters therefore import this tiny hook when the bundle
declares ``HARNESS_TOOL_TRACE``. Using Python's ``sitecustomize`` mechanism is important:
LiveKit starts job executors in child Python processes, so a wrapper applied only to the
parent worker misses the ``AgentSession`` that actually executes tools. Tracing is strictly
best effort and must never alter agent behaviour.

Second responsibility: W>1 LiveKit workers sharing one sandbox each try to bind the same
worker health port and each believe themselves solely responsible for load, so a naive
multi-worker run collides on startup and mis-reports availability. Rather than modify the
agent under test, this hook also flips the worker into livekit-agents' own side-by-side
mode (the same mode the library's CLI uses for ``start --simulation``) whenever the
harness's own ``FI_WORKER_HEALTH_PORT`` env var is present; outside the harness (that var
absent), this isolation flip is a no-op and ``run()`` behaves exactly as the library defines
it. Simulation mode also disables the worker's own load shedding (it never reports itself
``WS_FULL``) -- intended, since the harness routes each world to exactly one worker, but an
operator reading worker logs should know that report is suppressed by design, not a bug.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import os
from collections.abc import Mapping
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


_WORKER_ISOLATION_LOGGER = logging.getLogger("fi.alk.harness.worker_isolation")


def _apply_worker_isolation(server: Any, env: Mapping[str, str]) -> str:
    """Pure, unit-testable core of the isolation shim: nudge one already-constructed
    ``AgentServer`` into livekit-agents' own side-by-side mode using only the
    harness's single ``FI_WORKER_HEALTH_PORT`` env knob (presence marks a
    harness-managed worker; there is nothing else to read, so both isolation
    modes below hardcode the rest). Never raises and never leaves ``server`` in a
    worse state than it started in.

    Two guards, not one, gate every attribute write:

    - `hasattr` first -- this shim does not own `server`'s class, so it must never
      CREATE an attribute that was never there. A plain, slot-free Python object
      accepts `server._simulation = True` unconditionally (that assignment just adds
      the attribute), so skipping this check on a wheel shape that genuinely has no
      `_simulation` would silently invent one, report "simulation", and never try the
      `_port` fallback that wheel actually needs -- wrong, and unobservable from the
      return value alone.
    - A post-write re-read second -- a wheel CAN carry `_simulation` yet reject
      assignment (a property with a raising or no-op setter), and stopping at `hasattr`
      alone would report "simulation" for a write that silently did nothing. So, once
      `hasattr` allows the attempt, the value is re-read to confirm it actually landed
      before committing to that mode -- otherwise execution falls through to the next,
      weaker path.
    """
    port_raw = env.get("FI_WORKER_HEALTH_PORT", "").strip()
    if not port_raw:
        return "noop"

    mode = "unsupported"

    # Primary path: the library's own CLI uses this for `start --simulation`, so it is
    # the best-supported way to disable the health server and force `_is_available()`
    # to always report true -- exactly what a harness-managed, log-driven-readiness
    # worker needs.
    if hasattr(server, "_simulation"):
        try:
            server._simulation = True
        except Exception:  # noqa: BLE001 - isolation must never break the worker
            pass
        if getattr(server, "_simulation", None) is True:
            mode = "simulation"

    if mode == "unsupported" and hasattr(server, "_port") and hasattr(server, "_load_threshold"):
        # Fallback for wheels where `_simulation` doesn't exist or didn't take: move the
        # health port to the harness-assigned one (so W>1 workers don't collide on the
        # default) and disable load-shedding, since this worker is the only one the
        # harness routes to.
        try:
            port = int(port_raw)
        except (TypeError, ValueError):
            port = None
        if port is not None:
            try:
                server._port = port
                server._load_threshold = math.inf
            except Exception:  # noqa: BLE001 - isolation must never break the worker
                pass
            if (
                getattr(server, "_port", None) == port
                and getattr(server, "_load_threshold", None) == math.inf
            ):
                mode = "port_override"

    if mode != "unsupported" and hasattr(server, "_num_idle_processes"):
        # Every harness-managed world runs exactly one worker process per world, so
        # this is a fixed fact of the harness's topology, not a per-world value --
        # there is nothing to read from env. No re-read needed: a rejected write here
        # only costs an idle-process count the library defaults for anyway.
        try:
            server._num_idle_processes = 1
        except Exception:  # noqa: BLE001 - isolation must never break the worker
            pass

    if mode == "unsupported":
        # Keys only, never values. An operator seeing repeated stray health-port binds at
        # W>1 needs to know this world's worker was never actually isolated.
        _WORKER_ISOLATION_LOGGER.warning(
            "livekit worker isolation could not be applied to this worker (unsupported "
            "livekit-agents version or worker shape); at parallelism>1 this world's worker "
            "will bind the default health port"
        )
    elif mode != "noop":
        _WORKER_ISOLATION_LOGGER.info("livekit worker isolation applied: mode=%s", mode)
    return mode


def _install_worker_isolation() -> None:
    if not os.environ.get("FI_WORKER_HEALTH_PORT", "").strip():
        # Local, unmanaged runs (no harness env at all) leave `run()` behaving exactly
        # as the library defines it -- no wrapper is installed for them.
        return
    try:
        from livekit.agents.worker import AgentServer
    except Exception:  # noqa: BLE001 - non-LiveKit / unpinned wheels continue unchanged
        return

    # `getattr` rather than a bare attribute access: a wheel shape without a `run` method
    # at all must not turn into a `sitecustomize` import error printed by every interpreter
    # this hook loads into, including every per-job child process it spawns.
    original_run = getattr(AgentServer, "run", None)
    if original_run is None or not callable(original_run):
        return
    if getattr(original_run, "__alk_worker_isolation__", False):
        return

    @functools.wraps(original_run)
    async def isolated_run(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Re-derive from `os.environ` here (not a closed-over value) so the check is
        # live at call time too, not just at import time -- cheap, and keeps this
        # wrapper inert if the module is ever imported outside the harness.
        _apply_worker_isolation(self, os.environ)
        return await original_run(self, *args, **kwargs)

    isolated_run.__alk_worker_isolation__ = True  # type: ignore[attr-defined]
    try:
        AgentServer.run = isolated_run
    except Exception:  # noqa: BLE001 - a wheel that forbids the reassignment must not crash
        return


_install()
_install_worker_isolation()
