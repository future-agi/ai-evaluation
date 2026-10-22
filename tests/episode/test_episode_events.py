from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from fi.alk.episode import (
    EPISODE_EVENT_TYPES,
    EVENT_PAYLOAD_MODELS,
    EVENT_SCHEMA_VERSION,
    EpisodeEvent,
)

ENVELOPE_FIELDS = (
    "schema_version",
    "event_id",
    "episode_id",
    "seq",
    "type",
    "source",
    "unit_id",
    "wall_time",
    "payload",
    "provider_raw_ref",
)

PAYLOAD_KEYS = {
    "episode.started": (),
    "utterance": ("role", "text", "duration_ms"),
    "tool.started": ("tool_call_id", "name", "arguments"),
    "tool.completed": ("tool_call_id", "result"),
    "tool.failed": ("tool_call_id", "error", "result"),
    "tool.unresolved": ("tool_call_id", "reason"),
    "transfer.started": ("transfer_id", "destination"),
    "transfer.completed": ("transfer_id", "outcome"),
    "state.delta": ("delta",),
    "episode.ended": (),
}

VALID_PAYLOADS = {
    "episode.started": {},
    "utterance": {"role": "caller", "text": "", "duration_ms": None},
    "tool.started": {"tool_call_id": "call-1", "name": None, "arguments": None},
    "tool.completed": {"tool_call_id": "call-1", "result": None},
    "tool.failed": {"tool_call_id": "call-1", "error": "boom", "result": None},
    "tool.unresolved": {"tool_call_id": "call-1", "reason": "no_result_in_record"},
    "transfer.started": {"transfer_id": "t-1", "destination": None},
    "transfer.completed": {"transfer_id": "t-1", "outcome": "completed"},
    "state.delta": {"delta": {}},
    "episode.ended": {},
}


def _event(**overrides):
    fields = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": "event_0",
        "episode_id": "episode_0",
        "seq": 0,
        "type": "utterance",
        "source": "vapi",
        "unit_id": "main",
        "wall_time": None,
        "payload": {},
        "provider_raw_ref": None,
    }
    fields.update(overrides)
    return fields


def test_envelope_fields_exact():
    assert tuple(EpisodeEvent.model_fields) == ENVELOPE_FIELDS
    assert all(field.is_required() for field in EpisodeEvent.model_fields.values())
    event = EpisodeEvent.model_validate(_event())
    with pytest.raises(ValidationError):
        EpisodeEvent.model_validate(_event(run_id="run_1"))
    with pytest.raises(ValidationError):
        event.seq = 1  # frozen
    for name in ENVELOPE_FIELDS:
        fields = _event()
        del fields[name]
        with pytest.raises(ValidationError):
            EpisodeEvent.model_validate(fields)
    # JSON mode, where a time string is accepted at all, so the refusal is for the missing zone.
    assert EpisodeEvent.model_validate_json(json.dumps(_event(wall_time="2026-10-12T05:00:00Z"))).wall_time
    with pytest.raises(ValidationError):
        EpisodeEvent.model_validate_json(json.dumps(_event(wall_time="2026-10-12T05:00:00")))  # naive time


def test_type_is_open_on_envelope():
    assert EpisodeEvent.model_validate(_event(type="vendor.custom_thing")).type == "vendor.custom_thing"
    for bad in ("Utterance", "tool..started", "1tool", "tool.", "x" * 65):
        with pytest.raises(ValidationError):
            EpisodeEvent.model_validate(_event(type=bad))


def test_payload_keys_exact_per_type():
    assert tuple(EVENT_PAYLOAD_MODELS) == EPISODE_EVENT_TYPES
    for name, keys in PAYLOAD_KEYS.items():
        model = EVENT_PAYLOAD_MODELS[name]
        assert tuple(model.model_fields) == keys, name
        model.model_validate(VALID_PAYLOADS[name])
        # An unknown value is null, never an omitted key.
        for key in keys:
            payload = dict(VALID_PAYLOADS[name])
            del payload[key]
            with pytest.raises(ValidationError):
                model.model_validate(payload)


@pytest.mark.parametrize("name", EPISODE_EVENT_TYPES)
def test_unknown_payload_key_rejected(name):
    payload = {**VALID_PAYLOADS[name], "unexpected": 1}
    with pytest.raises(ValidationError):
        EVENT_PAYLOAD_MODELS[name].model_validate(payload)
