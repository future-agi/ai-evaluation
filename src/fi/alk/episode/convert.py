"""The converter registry and the one entry point that turns a provider's call record into an Episode."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator

from .models import VOICE_CAPABILITIES, VoiceEpisode

DEFAULT_UNIT_ID = "main"


class EpisodeError(ValueError):
    """A conversion failure whose message is only its code.

    The code is all that crosses process boundaries and lands in logs, and it
    can never carry a value from the provider's record.
    """

    code: str

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class UnknownProviderError(EpisodeError):
    pass


class EpisodeConversionError(EpisodeError):
    pass


class ConvertContext(BaseModel):
    # Strict, so bytes are not read as a tenant id and a set, which has no
    # order, is not read as the ordered capability list.
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tenant_id: str = Field(min_length=1)
    unit_id: str = Field(default=DEFAULT_UNIT_ID, min_length=1)
    declared_capabilities: tuple[str, ...]

    # A JSON caller can only send an array, and strict alone would refuse it.
    @field_validator("declared_capabilities", mode="before")
    @classmethod
    def _list_as_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


Converter = Callable[[Mapping[str, Any], ConvertContext], VoiceEpisode]

# Module-level so a test can swap the whole registry and leave nothing behind.
_CONVERTERS: dict[str, Converter] = {}


def register_converter(provider: str, converter: Converter) -> None:
    if provider in _CONVERTERS:
        raise ValueError("converter_already_registered")
    _CONVERTERS[provider] = converter


def registered_providers() -> tuple[str, ...]:
    return tuple(sorted(_CONVERTERS))


def _converter_for(provider: str) -> Converter:
    converter = _CONVERTERS.get(provider) if isinstance(provider, str) else None
    if converter is None:
        raise UnknownProviderError("unknown_provider")
    return converter


def convert_record(provider: str, record: Mapping[str, Any], context: ConvertContext) -> VoiceEpisode:
    converter = _converter_for(provider)
    if not isinstance(context, ConvertContext):
        raise EpisodeConversionError("context_invalid")
    if not isinstance(record, Mapping):
        raise EpisodeConversionError("record_not_object")
    declared = context.declared_capabilities
    if any(word not in VOICE_CAPABILITIES for word in declared):
        raise EpisodeConversionError("unknown_capability")
    if len(set(declared)) != len(declared):
        raise EpisodeConversionError("duplicate_capability")

    # The converter's own exception is dropped, not chained: a pydantic error
    # quotes the values it rejected, and no record value may reach a log or a
    # traceback. Raising after the except block leaves no __context__ either.
    try:
        episode = converter(record, context)
    except EpisodeError:
        raise
    except Exception:
        episode = None
    if not isinstance(episode, VoiceEpisode):
        raise EpisodeConversionError("converter_error")
    return episode


def convert_call(provider: str, record: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, JsonValue]:
    _converter_for(provider)
    parsed: ConvertContext | None = None
    if isinstance(context, Mapping):
        try:
            parsed = ConvertContext.model_validate(dict(context))
        except ValidationError:
            parsed = None
    if parsed is None:
        raise EpisodeConversionError("context_invalid")
    return json.loads(episode_to_json(convert_record(provider, record, parsed)))


def episode_to_json(episode: VoiceEpisode) -> str:
    return episode.model_dump_json()


def episode_from_json(data: str | bytes) -> VoiceEpisode:
    return VoiceEpisode.model_validate_json(data)
