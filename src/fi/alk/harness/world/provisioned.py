"""A world backed by the services the submitted repository actually ships."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from ..contract import AgentContract
from ..provision import (
    ProvisionedEnvironment,
    ProvisionError,
    attached_postgres_store,
    reachable_overrides,
)
from .runtime import Call, GeneratedWorld


class ProvisionedWorld(GeneratedWorld):
    """Forward tool effects to a shipped HTTP service and grade its real datastore."""

    def __init__(
        self,
        contract: AgentContract,
        destination: Path,
        *,
        source_root: str,
    ) -> None:
        environment = ProvisionedEnvironment.load(destination)
        if environment is None or not environment.running:
            raise ProvisionError(f"no running source environment at {destination}")
        overrides = reachable_overrides(environment)
        http_overrides = {
            name: value
            for name, value in overrides.items()
            if urlsplit(value).scheme.lower() in {"http", "https"}
        }
        if not http_overrides:
            raise ProvisionError(
                "the source environment publishes no HTTP endpoint that the agent can be "
                "pointed at"
            )
        http_endpoints = {value.rstrip("/") for value in http_overrides.values()}
        if len(http_endpoints) != 1:
            raise ProvisionError(
                "more than one HTTP source endpoint was discovered ("
                + ", ".join(sorted(http_overrides))
                + "); each service-backed tool must name which configuration variable reaches "
                "its service"
            )
        super().__init__(store=attached_postgres_store(destination))
        self.name = contract.agent
        self.destination = Path(destination)
        self.source_root = str(source_root)
        self.base_url = next(iter(http_endpoints))
        self.refusal_signature = contract.refusal_signature
        self.endpoints = self._discover_endpoints()
        self.endpoint_for: dict[str, str] = {}
        # The contract describes the agent-facing tools, not the raw dependency API. Even when a
        # tool and an HTTP endpoint share a name, the agent commonly injects session state,
        # renames arguments, enforces ordering, or records local effects before forwarding it.
        # Probing the endpoint with the model-facing arguments therefore tests the dependency's
        # transport schema, not the submitted agent. Every contract tool executes in the real
        # worker during scenarios and is marked as such here. Matching endpoints remain available
        # as environment handlers for setup/health sequences only.
        self.runtime_tools: set[str] = set(contract.tool_names())
        self.local_runtime_tools: set[str] = set()
        for spec in contract.tools:
            entry = contract.entry_for(spec.name)
            endpoint = str(getattr(entry, "endpoint", "") or spec.name).strip("/")
            if endpoint in self.endpoints:
                self.endpoint_for[spec.name] = endpoint
            elif entry is not None and entry.mode in {"import", "construct"}:
                # The submitted worker executes this tool inside its own process. Some workers
                # mirror that completed action to TOOLS_API_URL so ALK can observe it. That
                # mirror is telemetry, not a request for ALK to invent a second implementation.
                self.local_runtime_tools.add(spec.name)
        # Marker source is persisted only as provenance/readability. ``call`` below performs the
        # forwarding, so no generated implementation executes.
        self.handlers = {
            name: f"# forwarded unchanged to submitted service endpoint /{endpoint}\n"
            for name, endpoint in self.endpoint_for.items()
        }

    def _discover_endpoints(self) -> set[str]:
        try:
            response = requests.get(f"{self.base_url}/openapi.json", timeout=10)
            response.raise_for_status()
            document = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise ProvisionError(
                f"could not read the submitted service's OpenAPI document: {exc}"
            ) from exc
        return {
            str(path).strip("/")
            for path, methods in (document.get("paths") or {}).items()
            if isinstance(methods, dict) and "post" in methods
        }

    def forward(
        self,
        endpoint: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        record_as: str = "",
        record: bool = True,
        session_id: str = "harness",
    ) -> Call:
        endpoint = endpoint.strip("/")
        name = record_as or endpoint
        args = dict(arguments or {})
        if endpoint not in self.endpoints:
            call = Call(
                name=name,
                arguments=args,
                ok=False,
                refused=True,
                error=f"the submitted service has no endpoint /{endpoint}",
            )
            return self._record(call) if record else call
        try:
            response = requests.post(
                f"{self.base_url}/{endpoint}",
                json=args,
                headers={"x-session-id": session_id},
                timeout=15,
            )
            body: Any
            try:
                body = response.json()
            except ValueError:
                body = response.text
            refused = response.status_code >= 400
            call = Call(
                name=name,
                arguments=args,
                result=body,
                ok=not refused,
                refused=refused,
                error=(
                    json.dumps(body, default=str)
                    if refused and not isinstance(body, str)
                    else str(body)
                    if refused
                    else ""
                ),
            )
        except requests.RequestException as exc:
            call = Call(
                name=name,
                arguments=args,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        return self._record(call) if record else call

    def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> Call:
        if name in self.local_runtime_tools and name not in self.endpoint_for:
            return self._record(
                Call(
                    name=name,
                    arguments=dict(arguments or {}),
                    result={
                        "observed": True,
                        "execution": "submitted_agent_runtime",
                    },
                    ok=True,
                )
            )
        endpoint = self.endpoint_for.get(name, name)
        semantic_name = next(
            (tool for tool, path in self.endpoint_for.items() if path == name), name
        )
        return self.forward(endpoint, arguments, record_as=semantic_name)


def open_provisioned_world(
    destination: str | Path,
    contract: AgentContract,
    *,
    source_root: str,
) -> ProvisionedWorld:
    return ProvisionedWorld(
        contract,
        Path(destination),
        source_root=source_root,
    )
