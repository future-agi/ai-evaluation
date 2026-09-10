"""Agent endpoint adapter tree (plan §4.1 + §13).

Each adapter conforms to ``AgentEndpoint``. Existing runtime paths keep
running through ``LiveKitEngine`` — the adapters here give hosted-runner
and matrix-runner callers a stable spec-level surface without waiting
for a full engine rewrite.
"""

from __future__ import annotations

from .base import (
    AgentEndpoint,
    AgentEndpointManifest,
    DiscoveryRequest,
    DiscoverySnapshot,
    EndpointHandle,
    ReadinessResult,
    ReconciliationResult,
)
from .callable import CallableAgentEndpoint
from .http import HttpAgentEndpoint
from .livekit import LiveKitAgentEndpoint
from .retell import RetellAgentEndpoint, RetellCall, RetellCallOriginator
from .vapi import VapiAgentEndpoint, VapiCall, VapiCallOriginator
from .websocket import WebSocketAgentEndpoint

__all__ = [
    "AgentEndpoint",
    "AgentEndpointManifest",
    "CallableAgentEndpoint",
    "DiscoveryRequest",
    "DiscoverySnapshot",
    "EndpointHandle",
    "HttpAgentEndpoint",
    "LiveKitAgentEndpoint",
    "ReadinessResult",
    "ReconciliationResult",
    "RetellAgentEndpoint",
    "RetellCall",
    "RetellCallOriginator",
    "VapiAgentEndpoint",
    "VapiCall",
    "VapiCallOriginator",
    "WebSocketAgentEndpoint",
]
