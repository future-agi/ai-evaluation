"""The Episode record, its header models, and the value helpers converters share."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Context, Decimal, InvalidOperation, localcontext
from typing import Literal, get_args

from pydantic import BaseModel, Field, JsonValue, ValidationError, model_validator

from .events import EVENT_PAYLOAD_MODELS, EpisodeEvent, UtcDatetime, _MODEL_CONFIG
from .ids import derive_episode_event_id

EPISODE_SCHEMA_VERSION = "futureagi.episode.v1"

CapabilityName = Literal[
    "transcript",
    "turn_timing",
    "tool_calls",
    "latency",
    "cost_breakdown",
    "recordings",
    "transfers",
    "agent_version",
    "end_reason",
    "direction",
    "provider_analysis",
    "raw_log",
]
CoverageReason = Literal[
    "not_in_record",
    "not_mapped",
    "call_in_progress",
    "some_entries_untimed",
    "result_missing",
    "total_only",
    "no_version_on_record",
]
# Derived from the Literals so the type and the published list cannot drift apart.
VOICE_CAPABILITIES: tuple[str, ...] = get_args(CapabilityName)
COVERAGE_REASONS: tuple[str, ...] = get_args(CoverageReason)

_TOOL_CLOSING_TYPES = ("tool.completed", "tool.failed", "tool.unresolved")


class EpisodeVersion(BaseModel):
    model_config = _MODEL_CONFIG

    ended: bool
    event_count: int = Field(ge=1)

    def ordering_key(self) -> tuple[bool, int]:
        """Sort key between two copies of one Episode: not ended before ended, then fewer events first."""
        return (self.ended, self.event_count)


class ProviderRef(BaseModel):
    model_config = _MODEL_CONFIG

    provider: str
    id: str
    version: str | None


class CostLine(BaseModel):
    model_config = _MODEL_CONFIG

    component: Literal["stt", "llm", "tts", "platform", "transport", "other"]
    label: str | None
    amount_micros: int = Field(ge=0)
    tokens_in: int | None
    tokens_out: int | None


class Cost(BaseModel):
    model_config = _MODEL_CONFIG

    amount_micros: int = Field(ge=0)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    breakdown: tuple[CostLine, ...]


class Latency(BaseModel):
    model_config = _MODEL_CONFIG

    metric: Literal["e2e", "stt", "llm", "tts", "endpointing", "kb"]
    avg_ms: int | None
    p50_ms: int | None
    p90_ms: int | None
    p99_ms: int | None

    @model_validator(mode="after")
    def _has_a_number(self) -> Latency:
        if (self.avg_ms, self.p50_ms, self.p90_ms, self.p99_ms) == (None, None, None, None):
            raise ValueError("a latency entry needs at least one of avg_ms, p50_ms, p90_ms, p99_ms")
        return self


class ProviderAnalysis(BaseModel):
    model_config = _MODEL_CONFIG

    summary: str | None
    success: bool | None
    sentiment: str | None
    structured: dict[str, JsonValue] | None


class Recording(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["mono", "stereo", "multi_channel"]
    channel: Literal["combined", "agent", "caller"] | None
    uri: str
    expires_at: UtcDatetime | None


class CoverageEntry(BaseModel):
    model_config = _MODEL_CONFIG

    capability: CapabilityName
    status: Literal["observed", "partial", "unavailable"]
    reason: CoverageReason | None

    @model_validator(mode="after")
    def _reason_matches_status(self) -> CoverageEntry:
        if (self.status == "observed") != (self.reason is None):
            raise ValueError("a coverage entry has a reason exactly when its status is not observed")
        return self


class VoiceEpisode(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["futureagi.episode.v1"]
    type: Literal["voice"]
    sdk_version: str
    id: str
    version: EpisodeVersion
    provider: str = Field(min_length=1)
    provider_call_id: str = Field(min_length=1)
    provider_ref: ProviderRef | None
    status: Literal["in_progress", "ended", "errored"]
    started_at: UtcDatetime | None
    ended_at: UtcDatetime | None
    direction: Literal["inbound", "outbound", "unknown"]
    end_reason: str | None
    cost: Cost | None
    latency: tuple[Latency, ...]
    analysis: ProviderAnalysis | None
    recordings: tuple[Recording, ...]
    provider_log_url: str | None
    coverage: tuple[CoverageEntry, ...]
    unmapped_entries: dict[str, int]
    events: tuple[EpisodeEvent, ...]

    # Runs at construction and at parse alike, so no Episode that breaks an
    # invariant can exist in memory or be read back from JSON.
    @model_validator(mode="after")
    def _invariants(self) -> VoiceEpisode:
        events = self.events
        for index, event in enumerate(events):
            if event.seq != index:
                raise ValueError("every event's seq must equal its position in events")
            if event.episode_id != self.id:
                raise ValueError("every event's episode_id must equal the Episode id")
            if event.event_id != derive_episode_event_id(self.id, event.source, index):
                raise ValueError("every event_id must be derived from the Episode id, source and position")
        if self.version.event_count != len(events):
            raise ValueError("version.event_count must equal the number of events")
        if self.version.ended != (self.ended_at is not None):
            raise ValueError("version.ended must be true exactly when ended_at is set")

        types = [event.type for event in events]
        if not types or types[0] != "episode.started" or types.count("episode.started") != 1:
            raise ValueError("the first event, and only the first, must be episode.started")
        if self.status == "in_progress":
            if "episode.ended" in types:
                raise ValueError("an in-progress Episode must not have an episode.ended event")
        elif types[-1] != "episode.ended" or types.count("episode.ended") != 1:
            raise ValueError("the last event, and only the last, must be episode.ended once the call is over")

        for event in events:
            model = EVENT_PAYLOAD_MODELS.get(event.type)
            if model is None:
                continue
            try:
                model.model_validate(event.payload)
            except ValidationError:
                raise ValueError("an event's payload does not match the payload of its type") from None

        started: set[str] = set()
        open_calls: set[str] = set()
        for event in events:
            if event.type == "tool.started":
                call_id = event.payload["tool_call_id"]
                if call_id in started:
                    raise ValueError("two tool.started events share a tool_call_id")
                started.add(call_id)
                open_calls.add(call_id)
            elif event.type in _TOOL_CLOSING_TYPES:
                call_id = event.payload["tool_call_id"]
                if call_id not in open_calls:
                    raise ValueError("a tool call is closed without an open tool.started before it")
                open_calls.discard(call_id)
        if open_calls:
            raise ValueError("every tool.started needs a later tool.completed, tool.failed or tool.unresolved")

        if len({entry.capability for entry in self.coverage}) != len(self.coverage):
            raise ValueError("coverage names a capability more than once")

        # JSON mode lets a NaN or Infinity token into a JsonValue, and writing it
        # out gives null, so the Episode read back would differ from this one.
        structured = self.analysis.structured if self.analysis is not None else None
        if _has_non_finite_number(structured) or any(_has_non_finite_number(event.payload) for event in events):
            raise ValueError("an event payload or analysis.structured holds a number that is not finite")
        return self


def _has_non_finite_number(value: object) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_has_non_finite_number(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_non_finite_number(item) for item in value)
    return False


_MICROS_PER_UNIT = Decimal(1_000_000)
_MICROS_PER_CENT = Decimal(10_000)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def money_to_micros(value: object, *, in_cents: bool) -> int | None:
    """Integer micros of the currency, rounded half up; None when ``value`` is not a finite number >= 0.

    Going through ``str`` first keeps the decimal the provider wrote (0.0165 stays
    0.0165) instead of the nearest binary float, which would round the wrong way.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value < 0:
        return None
    scale = _MICROS_PER_CENT if in_cents else _MICROS_PER_UNIT
    # A fixed context: the host's own decimal settings would otherwise round the
    # product first, and equal inputs must always give equal micros.
    with localcontext(Context(prec=28)):
        try:
            return int((Decimal(str(value)) * scale).quantize(Decimal(1), rounding=ROUND_HALF_UP))
        except InvalidOperation:
            # Past the context's 28 digits: an absurd amount becomes null like any other bad value.
            return None


def parse_record_time(value: object) -> datetime | None:
    """A record's time as an aware UTC datetime; None when it does not parse.

    Strings are ISO 8601 with a trailing ``Z`` accepted (Python 3.10's parser
    refuses it), naive results are UTC, numbers are epoch milliseconds.
    """
    if isinstance(value, str):
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            # Shifting a time at the edge of the calendar into UTC can leave the range.
            return parsed.astimezone(timezone.utc)
        except (ValueError, OverflowError):
            return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return _EPOCH + timedelta(milliseconds=value)
    except (OverflowError, ValueError):
        return None
