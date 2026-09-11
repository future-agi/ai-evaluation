from __future__ import annotations

from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field, JsonValue


class EvidenceClass(str, Enum):
    CALLER_OBSERVED = "caller_observed"
    PROVIDER_REPORTED = "provider_reported"
    AGENT_INSTRUMENTED = "agent_instrumented"
    PLATFORM_VERIFIED = "platform_verified"


class EvidenceCapabilities(BaseModel):
    transcript: bool = False
    audio: bool = False
    tool_calls: bool = False
    tool_results: bool = False
    usage: bool = False
    internal_latency: bool = False
    configuration_snapshot: bool = False

    def supported(self) -> set[str]:
        return {
            name
            for name, enabled in self.model_dump().items()
            if enabled
        }


class EvidenceSourceSpec(BaseModel):
    source_id: str
    adapter: str
    adapter_version: str = "1"
    evidence_class: EvidenceClass
    capabilities: EvidenceCapabilities = Field(default_factory=EvidenceCapabilities)
    config: dict[str, JsonValue] = Field(default_factory=dict)


class EvidenceSourceSummary(BaseModel):
    source_id: str
    adapter: str
    evidence_class: EvidenceClass
    capabilities: EvidenceCapabilities = Field(default_factory=EvidenceCapabilities)
    available: bool = True
    redactions: list[str] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class AgentEvidenceSource(Protocol):
    capabilities: EvidenceCapabilities

    async def connect(self, context: object) -> None: ...

    async def fetch_final(self) -> object: ...

    async def close(self) -> None: ...
