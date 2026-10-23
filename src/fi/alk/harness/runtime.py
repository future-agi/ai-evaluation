"""Runtime-provider boundary for ALK-owned test environments.

Providers decide *where* a sealed environment runs.  They do not decide how an agent is
understood, how scenarios are written, or how results are graded.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field, JsonValue

from .bundle import EnvironmentBundle


class RuntimeState(str, Enum):
    PREPARING = "preparing"
    READY = "ready"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"


class RuntimeEndpoint(BaseModel):
    capability: str
    protocol: str
    address: str
    configuration_name: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class EnvironmentRuntime(BaseModel):
    runtime_id: str
    provider: str
    bundle_digest: str
    state: RuntimeState
    endpoints: dict[str, RuntimeEndpoint] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class RuntimeProvider(Protocol):
    """Execution-location port implemented locally and by the hosted sandbox fleet."""

    name: str

    async def provision(
        self,
        bundle: EnvironmentBundle,
        *,
        source: Path,
        work_directory: Path,
        contract: Any | None = None,
    ) -> EnvironmentRuntime: ...

    async def reset(
        self,
        runtime: EnvironmentRuntime,
        *,
        work_directory: Path,
        keep_processes: bool = False,
    ) -> None: ...

    async def healthy(
        self, runtime: EnvironmentRuntime, *, work_directory: Path
    ) -> bool: ...

    async def close(
        self, runtime: EnvironmentRuntime, *, work_directory: Path
    ) -> None: ...


__all__ = [
    "EnvironmentRuntime",
    "RuntimeEndpoint",
    "RuntimeProvider",
    "RuntimeState",
]
