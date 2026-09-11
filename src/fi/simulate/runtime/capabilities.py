from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class EndpointCapabilities(BaseModel):
    audio: bool = False
    text: bool = False
    streaming: bool = False
    interruption: bool = False
    dtmf: bool = False
    transfer: bool = False
    transcript_events: bool = False
    tool_events: bool = False
    usage_events: bool = False
    internal_metrics: bool = False
    recording: bool = False
    web_rtc: bool = False
    sip: bool = False

    def supported(self) -> set[str]:
        return {
            name
            for name, enabled in self.model_dump().items()
            if enabled
        }


class CapabilitySet(BaseModel):
    required: list[str] = Field(default_factory=list)
    supported: list[str] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)

    @field_validator("required", "supported", "degraded")
    @classmethod
    def _normalize(cls, values: list[str]) -> list[str]:
        return sorted(set(values))

    def missing(self) -> list[str]:
        return sorted(set(self.required) - set(self.supported))
