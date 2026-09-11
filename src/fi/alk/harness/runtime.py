"""Runtime-provider boundary for ALK-owned test environments.

Providers decide *where* a sealed environment runs.  They do not decide how an agent is
understood, how scenarios are written, or how results are graded.  The local provider below
adapts the proven repository/Compose provisioner; the hosted sandbox implements the same port.
"""

from __future__ import annotations

import asyncio
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
        self, runtime: EnvironmentRuntime, *, work_directory: Path
    ) -> None: ...

    async def healthy(
        self, runtime: EnvironmentRuntime, *, work_directory: Path
    ) -> bool: ...

    async def close(
        self, runtime: EnvironmentRuntime, *, work_directory: Path
    ) -> None: ...


class LocalComposeRuntimeProvider:
    """Run one repository environment as an isolated Docker Compose project.

    The adapter delegates lifecycle mechanics to ``harness.provision``, which gives each run a
    unique project, allocates ports, waits for declared health checks, fingerprints source reuse,
    and removes volumes during cleanup.  Only endpoint names and addresses cross this boundary;
    resolved credentials remain process-local.
    """

    name = "local-compose"

    async def provision(
        self,
        bundle: EnvironmentBundle,
        *,
        source: Path,
        work_directory: Path,
        contract: Any | None = None,
    ) -> EnvironmentRuntime:
        from .provision import provision

        environment = await asyncio.to_thread(
            provision, source, work_directory, contract
        )
        endpoints: dict[str, RuntimeEndpoint] = {}
        overrides = dict(environment.overrides)
        for capability, definition in bundle.capabilities.items():
            address = ""
            if definition.configuration_name:
                address = overrides.get(definition.configuration_name, "")
            if not address and len(overrides) == 1:
                address = next(iter(overrides.values()))
            discovered = next(
                (
                    endpoint
                    for endpoint in environment.service_endpoints
                    if endpoint["service"] == definition.service
                    and endpoint["container_port"] == definition.container_port
                ),
                None,
            )
            if not address and discovered:
                address = str(discovered["external_address"])
            if address:
                endpoints[capability] = RuntimeEndpoint(
                    capability=capability,
                    protocol=definition.protocol.value,
                    address=address,
                    configuration_name=definition.configuration_name,
                    metadata=(
                        {
                            "service": str(discovered["service"]),
                            "kind": str(discovered["kind"]),
                        }
                        if discovered
                        else {}
                    ),
                )
        return EnvironmentRuntime(
            runtime_id=environment.project,
            provider=self.name,
            bundle_digest=bundle.digest,
            state=RuntimeState.READY,
            endpoints=endpoints,
            metadata={
                "services": environment.services,
                "provision_seconds": environment.provision_seconds,
                "managed": environment.managed,
            },
        )

    async def reset(self, runtime: EnvironmentRuntime, *, work_directory: Path) -> None:
        from .provision import reset

        environment = await asyncio.to_thread(reset, work_directory)
        runtime.state = (
            RuntimeState.READY if environment.running else RuntimeState.UNHEALTHY
        )

    async def healthy(
        self, runtime: EnvironmentRuntime, *, work_directory: Path
    ) -> bool:
        from .provision import healthy

        is_healthy = await asyncio.to_thread(healthy, work_directory)
        runtime.state = RuntimeState.READY if is_healthy else RuntimeState.UNHEALTHY
        return is_healthy

    async def close(self, runtime: EnvironmentRuntime, *, work_directory: Path) -> None:
        from .provision import stop

        await asyncio.to_thread(stop, work_directory)
        runtime.state = RuntimeState.STOPPED


__all__ = [
    "EnvironmentRuntime",
    "LocalComposeRuntimeProvider",
    "RuntimeEndpoint",
    "RuntimeProvider",
    "RuntimeState",
]
