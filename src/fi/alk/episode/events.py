"""The Episode event envelope and the payload of each closed event type."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
)

EVENT_SCHEMA_VERSION = "futureagi.episode-event.v1"
EPISODE_EVENT_TYPES = (
    "episode.started",
    "utterance",
    "tool.started",
    "tool.completed",
    "tool.failed",
    "tool.unresolved",
    "transfer.started",
    "transfer.completed",
    "state.delta",
    "episode.ended",
)

# allow_inf_nan=False: the JSON form never carries NaN, and pydantic would
# otherwise write a NaN inside a payload as null, so the round trip would stop
# giving back an equal Episode. JSON mode ignores it inside JsonValue, so the
# Episode checks its free-form values itself.
# strict=True: a coerced value (True for 1, "yes" for true) would pass the model
# yet fail the exported schema, so the two would disagree on one Episode. Data
# in JSON form is therefore always validated in JSON mode.
# hide_input_in_errors=True: a validation error must never quote a record value.
_MODEL_CONFIG = ConfigDict(
    extra="forbid", frozen=True, allow_inf_nan=False, strict=True, hide_input_in_errors=True
)


def _as_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


# Naive times are refused rather than assumed UTC: the JSON form promises
# RFC 3339 in UTC, and guessing a zone is the converter's call, not the model's.
UtcDatetime = Annotated[AwareDatetime, AfterValidator(_as_utc)]


class EpisodeEvent(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["futureagi.episode-event.v1"]
    event_id: str
    episode_id: str
    seq: int = Field(ge=0)
    type: str = Field(max_length=64, pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")
    source: str = Field(min_length=1)
    unit_id: str = Field(min_length=1)
    wall_time: UtcDatetime | None
    payload: dict[str, JsonValue]
    provider_raw_ref: str | None


class EmptyPayload(BaseModel):
    model_config = _MODEL_CONFIG


class UtterancePayload(BaseModel):
    model_config = _MODEL_CONFIG

    role: Literal["agent", "caller", "human_agent"]
    text: str
    duration_ms: int | None = Field(ge=0)


class ToolStartedPayload(BaseModel):
    model_config = _MODEL_CONFIG

    tool_call_id: str = Field(min_length=1)
    name: str | None
    arguments: dict[str, JsonValue] | None


class ToolCompletedPayload(BaseModel):
    model_config = _MODEL_CONFIG

    tool_call_id: str = Field(min_length=1)
    result: JsonValue


class ToolFailedPayload(BaseModel):
    model_config = _MODEL_CONFIG

    tool_call_id: str = Field(min_length=1)
    error: str = Field(min_length=1)
    result: JsonValue


class ToolUnresolvedPayload(BaseModel):
    model_config = _MODEL_CONFIG

    tool_call_id: str = Field(min_length=1)
    reason: Literal["no_result_in_record"]


class TransferStartedPayload(BaseModel):
    model_config = _MODEL_CONFIG

    transfer_id: str = Field(min_length=1)
    destination: str | None


class TransferCompletedPayload(BaseModel):
    model_config = _MODEL_CONFIG

    transfer_id: str = Field(min_length=1)
    outcome: Literal["completed", "failed", "cancelled"]


class StateDeltaPayload(BaseModel):
    model_config = _MODEL_CONFIG

    delta: dict[str, JsonValue]


EVENT_PAYLOAD_MODELS: dict[str, type[BaseModel]] = {
    "episode.started": EmptyPayload,
    "utterance": UtterancePayload,
    "tool.started": ToolStartedPayload,
    "tool.completed": ToolCompletedPayload,
    "tool.failed": ToolFailedPayload,
    "tool.unresolved": ToolUnresolvedPayload,
    "transfer.started": TransferStartedPayload,
    "transfer.completed": TransferCompletedPayload,
    "state.delta": StateDeltaPayload,
    "episode.ended": EmptyPayload,
}
