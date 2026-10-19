"""The JSON Schema export that schema publication stores, unchanged, per SDK version."""

from __future__ import annotations

from pydantic import JsonValue

from .events import EPISODE_EVENT_TYPES, EVENT_PAYLOAD_MODELS
from .models import VOICE_CAPABILITIES, VoiceEpisode


def export_schema() -> dict[str, JsonValue]:
    """The four keys publication reads: the Episode schema, the event types, each payload's schema, the capabilities.

    Pydantic emits draft 2020-12 with local ``#/$defs`` references and no
    ``$schema`` key, which is the form consumers expect: they always name the
    2020-12 validator themselves.
    """
    episode_schema = VoiceEpisode.model_json_schema()
    definitions = episode_schema["$defs"]
    rules = []
    for name in EPISODE_EVENT_TYPES:
        model = EVENT_PAYLOAD_MODELS[name]
        document = model.model_json_schema()
        for key, value in {**document.pop("$defs", {}), model.__name__: document}.items():
            if definitions.setdefault(key, value) != value:
                raise RuntimeError("two different schemas were generated under one definition name")
        # The envelope keeps type open, so the payload rule is conditional:
        # a known type must carry its payload, any other type passes.
        rules.append(
            {
                "if": {"properties": {"type": {"const": name}}, "required": ["type"]},
                "then": {"properties": {"payload": {"$ref": f"#/$defs/{model.__name__}"}}},
            }
        )
    definitions["EpisodeEvent"]["allOf"] = rules
    return {
        "episode_schema": episode_schema,
        "event_types": list(EPISODE_EVENT_TYPES),
        "payload_schemas": {name: EVENT_PAYLOAD_MODELS[name].model_json_schema() for name in EPISODE_EVENT_TYPES},
        "capabilities": list(VOICE_CAPABILITIES),
    }
