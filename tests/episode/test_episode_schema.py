from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from test_episode_events import VALID_PAYLOADS

from fi.alk.episode import (
    COVERAGE_REASONS,
    DEFAULT_UNIT_ID,
    EPISODE_EVENT_TYPES,
    EPISODE_SCHEMA_VERSION,
    EVENT_PAYLOAD_MODELS,
    EVENT_SCHEMA_VERSION,
    VOICE_CAPABILITIES,
    ConvertContext,
    EpisodeEvent,
    VoiceEpisode,
    check_conformance,
    episode_from_json,
    episode_to_json,
    export_schema,
    register_converter,
    registered_providers,
)
from fi.alk.episode import convert as convert_module

TENANT = "0b7e6a52-3c1d-4f6e-9a61-2f1c5d8e9a10"
EXAMPLE_CONTEXT = {
    "tenant_id": TENANT,
    "unit_id": "main",
    "declared_capabilities": ["transcript", "turn_timing", "tool_calls", "cost_breakdown", "direction"],
}
EMPTY_DECLARED = {"tenant_id": TENANT, "unit_id": "main", "declared_capabilities": []}
UTTERANCE = ("utterance", {"role": "caller", "text": "hi", "duration_ms": 1200}, "2026-10-12T05:00:01Z")


def _walk(node):
    yield node
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def test_constants_and_vocabularies_are_the_published_ones():
    # Written out by hand: the export is built from these same tuples, so only a
    # literal catches a reordered or renamed word.
    assert EPISODE_SCHEMA_VERSION == "futureagi.episode.v1"
    assert EVENT_SCHEMA_VERSION == "futureagi.episode-event.v1"
    assert DEFAULT_UNIT_ID == "main"
    assert EPISODE_EVENT_TYPES == (
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
    assert VOICE_CAPABILITIES == (
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
    )
    assert COVERAGE_REASONS == (
        "not_in_record",
        "not_mapped",
        "call_in_progress",
        "some_entries_untimed",
        "result_missing",
        "total_only",
        "no_version_on_record",
    )
    assert ConvertContext(tenant_id="t", declared_capabilities=()).unit_id == "main"


def test_registered_providers():
    # Rises with each converter: ("vapi",) once the Vapi converter lands, then
    # ("retell", "vapi") once Retell's does.
    assert registered_providers() == ()


def test_register_converter_rejects_a_duplicate(monkeypatch):
    monkeypatch.setattr(convert_module, "_CONVERTERS", {})
    register_converter("fake", lambda record, context: None)
    with pytest.raises(ValueError) as raised:
        register_converter("fake", lambda record, context: None)
    assert str(raised.value) == "converter_already_registered"
    assert registered_providers() == ("fake",)


def test_registered_providers_are_sorted(monkeypatch):
    monkeypatch.setattr(convert_module, "_CONVERTERS", {})
    register_converter("b", lambda record, context: None)
    register_converter("a", lambda record, context: None)
    assert registered_providers() == ("a", "b")


def test_conformance_names_each_broken_rule(build_episode, example_episode_json):
    assert check_conformance(json.loads(example_episode_json), EXAMPLE_CONTEXT) == ()
    clean = build_episode(body=[UTTERANCE])
    assert check_conformance(clean, EMPTY_DECLARED) == ()

    broken_invariant = copy.deepcopy(clean)
    broken_invariant["events"][1]["seq"] = 9
    assert check_conformance(broken_invariant, EMPTY_DECLARED) == ("invariants",)

    other_tenant = {**EMPTY_DECLARED, "tenant_id": "another-organization"}
    assert check_conformance(clean, other_tenant) == ("id_matches_context",)

    declared_transcript = {**EMPTY_DECLARED, "declared_capabilities": ["transcript"]}
    assert check_conformance(clean, declared_transcript) == ("coverage_matches_declared",)
    # Right names, wrong status for what the Episode holds.
    misreported = json.loads(example_episode_json)
    misreported["coverage"][4] = {"capability": "direction", "status": "observed", "reason": None}
    assert check_conformance(misreported, EXAMPLE_CONTEXT) == ("coverage_matches_declared",)

    open_type = build_episode(body=[("vendor.custom_event", {"anything": [1, 2]})])
    assert check_conformance(open_type, EMPTY_DECLARED) == ("closed_event_types",)


def test_json_round_trip_and_all_keys_present(example_episode_json):
    episode = episode_from_json(example_episode_json)
    text = episode_to_json(episode)
    assert json.loads(text) == json.loads(example_episode_json)
    assert episode_from_json(text) == episode
    assert episode_from_json(text.encode("utf-8")) == episode
    assert "NaN" not in text and "Infinity" not in text

    data = json.loads(text)
    assert list(data) == list(VoiceEpisode.model_fields)
    assert all(list(event) == list(EpisodeEvent.model_fields) for event in data["events"])
    assert data["analysis"] is None and data["provider_log_url"] is None
    assert data["started_at"] == "2026-10-12T05:00:00Z"
    assert data["ended_at"] == "2026-10-12T05:00:42.500000Z"
    assert isinstance(data["latency"], list) and isinstance(data["events"], list)

    shifted = json.loads(example_episode_json)
    shifted["started_at"] = "2026-10-12T07:00:00+02:00"
    assert json.loads(episode_to_json(episode_from_json(json.dumps(shifted))))["started_at"] == "2026-10-12T05:00:00Z"


def test_validation_errors_quote_no_record_value(example_episode_json, build_episode):
    bad_header = json.loads(example_episode_json)
    bad_header["end_reason"] = 123
    with pytest.raises(ValidationError) as raised:
        episode_from_json(json.dumps(bad_header))
    assert "123" not in str(raised.value)

    payload = {"role": "robot-7f3a", "text": "secret-text-91", "duration_ms": None}
    with pytest.raises(ValidationError) as raised:
        EVENT_PAYLOAD_MODELS["utterance"].model_validate(payload)
    assert "robot-7f3a" not in str(raised.value) and "secret-text-91" not in str(raised.value)
    with pytest.raises(ValidationError) as raised:
        episode_from_json(json.dumps(build_episode(body=[("utterance", payload)])))
    assert "robot-7f3a" not in str(raised.value) and "secret-text-91" not in str(raised.value)


def test_export_has_exactly_four_keys():
    export = export_schema()
    assert list(export) == ["episode_schema", "event_types", "payload_schemas", "capabilities"]
    assert export["event_types"] == list(EPISODE_EVENT_TYPES)
    assert export["capabilities"] == list(VOICE_CAPABILITIES)
    assert list(export["payload_schemas"]) == list(EPISODE_EVENT_TYPES)
    for name, document in export["payload_schemas"].items():
        assert document == EVENT_PAYLOAD_MODELS[name].model_json_schema()


def test_export_is_stable_local_refs_no_dialect_key():
    first = json.dumps(export_schema(), sort_keys=True)
    assert json.dumps(export_schema(), sort_keys=True) == first
    probe = "import json, fi.alk.episode as e; print(json.dumps(e.export_schema(), sort_keys=True))"
    other = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": "12345"},
    )
    assert other.stdout.strip() == first
    nodes = list(_walk(export_schema()))
    assert not any(isinstance(node, dict) and "$schema" in node for node in nodes)
    refs = [node["$ref"] for node in nodes if isinstance(node, dict) and "$ref" in node]
    assert refs and all(ref.startswith("#") for ref in refs)


def test_bad_known_payload_fails_both_schemas(build_episode, example_episode_json):
    export = export_schema()
    episode_validator = Draft202012Validator(export["episode_schema"])
    assert episode_validator.is_valid(json.loads(example_episode_json))
    good = build_episode(body=[UTTERANCE])
    assert episode_validator.is_valid(good)
    assert Draft202012Validator(export["payload_schemas"]["utterance"]).is_valid(good["events"][1]["payload"])

    bad = copy.deepcopy(good)
    bad["events"][1]["payload"]["role"] = "robot"
    assert not episode_validator.is_valid(bad)
    assert not Draft202012Validator(export["payload_schemas"]["utterance"]).is_valid(bad["events"][1]["payload"])

    for name in EPISODE_EVENT_TYPES:
        wrong = {"unexpected": 1}
        assert not episode_validator.is_valid(build_episode(body=[(name, wrong)])), name
        assert not Draft202012Validator(export["payload_schemas"][name]).is_valid(wrong), name


@pytest.mark.parametrize("name", EPISODE_EVENT_TYPES)
def test_valid_known_payload_passes_episode_schema(name, build_episode):
    # The schema has no pairing rule, so a lone close is fine; a payload checked
    # against another type's schema fails here.
    episode = build_episode(body=[(name, VALID_PAYLOADS[name])])
    assert Draft202012Validator(export_schema()["episode_schema"]).is_valid(episode)


def _nan_in_unknown_payload(build):
    return build(body=[("vendor.custom_event", {"x": float("nan")})])


def _nan_in_structured(build):
    episode = build()
    episode["analysis"] = {"summary": None, "success": None, "sentiment": None, "structured": {"k": float("nan")}}
    return episode


def _infinity_nested_in_unknown_payload(build):
    return build(body=[("vendor.custom_event", {"x": [1, {"y": float("inf")}]})])


# The unknown type also breaks the closed list, which is reported beside the invariants.
NON_FINITE_CASES = {
    "nan_in_unknown_payload": (_nan_in_unknown_payload, ("invariants", "closed_event_types")),
    "nan_in_structured": (_nan_in_structured, ("invariants",)),
    "infinity_nested_in_unknown_payload": (
        _infinity_nested_in_unknown_payload,
        ("invariants", "closed_event_types"),
    ),
}


@pytest.mark.parametrize("make, broken", NON_FINITE_CASES.values(), ids=NON_FINITE_CASES.keys())
def test_non_finite_number_is_refused_in_json_mode(make, broken, build_episode):
    episode = make(build_episode)
    # json.dumps writes the non-standard NaN and Infinity tokens by default.
    with pytest.raises(ValidationError):
        episode_from_json(json.dumps(episode))
    assert check_conformance(episode, EMPTY_DECLARED) == broken


def test_unknown_type_passes_episode_schema(build_episode):
    episode = build_episode(body=[("vendor.custom_event", {"anything": [1, 2], "nested": {"ok": None}})])
    assert Draft202012Validator(export_schema()["episode_schema"]).is_valid(episode)


def test_every_key_required(example_episode_json):
    export = export_schema()
    documents = [export["episode_schema"], *export["payload_schemas"].values()]
    objects = [
        node
        for document in documents
        for node in _walk(document)
        if isinstance(node, dict) and isinstance(node.get("properties"), dict) and node.get("type") == "object"
    ]
    assert objects
    for node in objects:
        assert sorted(node.get("required", [])) == sorted(node["properties"]), node.get("title")

    validator = Draft202012Validator(export["episode_schema"])
    example = json.loads(example_episode_json)
    for key in example:
        missing = {k: v for k, v in example.items() if k != key}
        assert not validator.is_valid(missing), key
    for key in example["events"][0]:
        missing = copy.deepcopy(example)
        del missing["events"][0][key]
        assert not validator.is_valid(missing), key


def test_export_meets_the_provider_list_rules():
    export = export_schema()
    assert isinstance(export, dict)
    Draft202012Validator.check_schema(export["episode_schema"])
    refs = [n["$ref"] for n in _walk(export["episode_schema"]) if isinstance(n, dict) and "$ref" in n]
    assert all(ref.startswith("#") for ref in refs)
    event_types = export["event_types"]
    assert isinstance(event_types, list) and len(event_types) >= 1
    assert all(isinstance(name, str) for name in event_types) and len(set(event_types)) == len(event_types)
    capabilities = export["capabilities"]
    assert isinstance(capabilities, list)
    assert all(isinstance(word, str) for word in capabilities) and len(set(capabilities)) == len(capabilities)
    assert all(word in capabilities for word in VOICE_CAPABILITIES)
    # The provider list's PROVIDER_LIST_MAX_BYTES.
    assert len(json.dumps(export, sort_keys=True).encode("utf-8")) < 2_000_000
