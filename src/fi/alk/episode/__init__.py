"""The Episode: one record shape for a voice call, simulated or in production.

The platform never imports this package; it reads the JSON form and the schema
that ``export_schema()`` publishes.
"""

from __future__ import annotations

from .conformance import check_conformance
from .convert import (
    DEFAULT_UNIT_ID,
    ConvertContext,
    Converter,
    EpisodeConversionError,
    EpisodeError,
    UnknownProviderError,
    convert_call,
    convert_record,
    episode_from_json,
    episode_to_json,
    register_converter,
    registered_providers,
)
from .events import (
    EPISODE_EVENT_TYPES,
    EVENT_PAYLOAD_MODELS,
    EVENT_SCHEMA_VERSION,
    EpisodeEvent,
)
from .ids import EPISODE_ID_NAMESPACE, derive_episode_event_id, derive_episode_id
from .models import (
    COVERAGE_REASONS,
    EPISODE_SCHEMA_VERSION,
    VOICE_CAPABILITIES,
    CapabilityName,
    Cost,
    CostLine,
    CoverageEntry,
    CoverageReason,
    EpisodeVersion,
    Latency,
    ProviderAnalysis,
    ProviderRef,
    Recording,
    VoiceEpisode,
)
from .schema import export_schema

# Imported last and for its side effect: each converter module registers itself.
from . import converters  # noqa: F401

__all__ = [
    "COVERAGE_REASONS",
    "DEFAULT_UNIT_ID",
    "EPISODE_EVENT_TYPES",
    "EPISODE_ID_NAMESPACE",
    "EPISODE_SCHEMA_VERSION",
    "EVENT_PAYLOAD_MODELS",
    "EVENT_SCHEMA_VERSION",
    "VOICE_CAPABILITIES",
    "CapabilityName",
    "ConvertContext",
    "Converter",
    "Cost",
    "CostLine",
    "CoverageEntry",
    "CoverageReason",
    "EpisodeConversionError",
    "EpisodeError",
    "EpisodeEvent",
    "EpisodeVersion",
    "Latency",
    "ProviderAnalysis",
    "ProviderRef",
    "Recording",
    "UnknownProviderError",
    "VoiceEpisode",
    "check_conformance",
    "convert_call",
    "convert_record",
    "derive_episode_event_id",
    "derive_episode_id",
    "episode_from_json",
    "episode_to_json",
    "export_schema",
    "register_converter",
    "registered_providers",
]
