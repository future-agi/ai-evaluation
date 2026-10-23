"""Run scenario code in a disposable interpreter against its assigned world."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import random
import sys
from pathlib import Path
from typing import Any


def tuples(value: Any) -> Any:
    return tuple(tuples(item) for item in value) if isinstance(value, list) else value


async def main() -> None:
    from .world.errors import (
        WorldError,
        WorldQueryRejected,
        WorldReadOnly,
        WorldReservedName,
        WorldStateTooLarge,
        WorldUnavailable,
        WorldUsageError,
    )
    from .world.handle import HostedWorld
    from .world.runtime import Call
    from .world.stores.postgres import AttachedPostgresStore

    payload = json.load(sys.stdin)
    state = payload["world"]
    rng = random.Random()
    rng.setstate(tuples(state["rng"]))
    world = None
    try:
        if state.get("kind") == "native_observation":
            from .world.observation import ObservationWorld
            world = ObservationWorld(state["world_index"], rng, state.get("endpoints", {}))
            world._restore_private_http_state(payload.get("private_http", {}))
        else:
            world = HostedWorld(
                AttachedPostgresStore(state["dsn"]),
                state["world_index"],
                rng,
                state["baseline"],
                # The parent already validated this handle before scenario execution.
                _validate_baseline=False,
                workspace_roots=state.get("workspace_roots", {}),
            )
        handle = world.read_only() if state["read_only"] else world
        namespace: dict[str, Any] = {}
        exec(compile(payload["source"], "<scenario-phase>", "exec"), namespace)  # noqa: S102 - isolated scenario execution
        args = [handle]
        if "calls" in payload:
            args.append(tuple(Call(**call) for call in payload["calls"]))
        if "reference" in payload:
            args.append(payload["reference"])
        value = namespace[payload["entry"]](*args)
        if inspect.isawaitable(value):
            value = await value
        valid_verdict = value is None or isinstance(value, (bool, str))
        result = {
            "value": value if valid_verdict else None,
            "invalid_verdict": not valid_verdict,
            "rng": rng.getstate(),
        }
    except BaseException as exc:  # noqa: BLE001 - scenario exit must become a reported phase fault
        result = {
            "error": type(exc).__name__,
            "world_error": next(
                (
                    kind.__name__
                    for kind in (
                        WorldUnavailable,
                        WorldStateTooLarge,
                        WorldReadOnly,
                        WorldReservedName,
                        WorldQueryRejected,
                        WorldUsageError,
                        WorldError,
                    )
                    if isinstance(exc, kind)
                ),
                None,
            ),
            "rng": rng.getstate(),
        }
    if state.get("kind") == "native_observation" and world is not None:
        if isinstance(result.get("value"), str):
            result["value"] = world._redact_http_text(result["value"])
        try:
            result["private_http"] = world._private_http_state()
        except WorldUsageError:
            result = {
                "error": "WorldUsageError",
                "world_error": "WorldUsageError",
                "rng": rng.getstate(),
                "private_http": {},
            }
        result["http_events"] = world._drain_http_events()
    fd = os.open(Path(sys.argv[1]), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(result, stream)


if __name__ == "__main__":
    asyncio.run(main())
