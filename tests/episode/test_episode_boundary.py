from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

import fi.alk.episode
from fi.alk.episode import (
    DEFAULT_UNIT_ID,
    ConvertContext,
    EpisodeConversionError,
    EpisodeError,
    UnknownProviderError,
    VoiceEpisode,
    convert_call,
    convert_record,
)
from fi.alk.episode import convert as convert_module
from fi.simulate.runtime.events import CanonicalEvent
from fi.simulate.runtime.ids import derive_event_id

CONTEXT = {"tenant_id": "0b7e6a52-3c1d-4f6e-9a61-2f1c5d8e9a10", "unit_id": "main", "declared_capabilities": []}

# pydantic's own runtime dependencies; anything else outside the standard
# library that the import loads breaks the package's import boundary.
PYDANTIC_OWN = {"pydantic", "pydantic_core", "annotated_types", "typing_extensions", "typing_inspection"}
FORBIDDEN = ("fi.simulate", "fi.alk.providers", "jsonschema", "httpx", "requests", "aiohttp")


def test_canonical_event_fields_and_event_id_unchanged():
    assert tuple(CanonicalEvent.model_fields) == (
        "schema_version",
        "event_id",
        "run_id",
        "test_case_id",
        "session_id",
        "type",
        "source",
        "provider",
        "wall_time",
        "monotonic_ns",
        "reliability",
        "payload",
        "trace_id",
        "correlation_id",
        "provider_raw_ref",
    )
    assert derive_event_id("case_a", "runtime", 0) == "event_235166e1d10c57d2949ba0d3dbd63c60"


def test_import_loads_no_simulate_no_providers_no_http():
    # The baseline is taken inside the fresh interpreter, so what the
    # interpreter and site loaded before the import (__main__, a virtualenv's
    # .pth hook) is not charged to the package.
    probe = (
        "import json, sys\n"
        "before = set(sys.modules)\n"
        "import fi.alk.episode\n"
        "print(json.dumps({'new': sorted(set(sys.modules) - before), 'all': sorted(sys.modules)}))\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    loaded = json.loads(result.stdout)
    top_level = {name.split(".")[0] for name in loaded["new"]}
    outside_stdlib = {
        name
        for name in top_level
        if name not in sys.stdlib_module_names
        # sysconfig's generated per-platform data module: standard library, but not listed.
        and not name.startswith("_sysconfigdata")
    }
    assert outside_stdlib <= {"fi"} | PYDANTIC_OWN, sorted(outside_stdlib - {"fi"} - PYDANTIC_OWN)
    for name in FORBIDDEN:
        assert name not in loaded["all"], name


def test_package_source_has_no_trace_or_top_span_id():
    package_dir = Path(fi.alk.episode.__file__).parent
    sources = sorted(package_dir.rglob("*.py"))
    assert len(sources) >= 9
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for needle in ("_provider_collector_", "trace:", "4d61d4e2"):
            assert needle not in text, f"{needle} in {path.name}"


def test_default_unit_id_equals_main_unit_id():
    from fi.alk.providers.agent import MAIN_UNIT_ID

    assert DEFAULT_UNIT_ID == MAIN_UNIT_ID == "main"


def _invalid_episode(record, context):
    # Building a model from bad values raises pydantic.ValidationError inside the converter.
    return VoiceEpisode.model_validate({"schema_version": "futureagi.episode.v1", "type": "voice"})


def test_pydantic_error_never_leaves_convert_call(monkeypatch):
    monkeypatch.setattr(convert_module, "_CONVERTERS", {"fake": _invalid_episode})
    with pytest.raises(EpisodeConversionError) as raised:
        convert_call("fake", {"id": "call-1"}, CONTEXT)
    assert raised.value.code == "converter_error"
    assert str(raised.value) == "converter_error"
    # Not chained either: a pydantic error quotes the values it rejected.
    assert raised.value.__cause__ is None and raised.value.__context__ is None

    with pytest.raises(EpisodeConversionError) as raised:
        convert_record("fake", {"id": "call-1"}, ConvertContext(**CONTEXT))
    assert raised.value.code == "converter_error"
    assert raised.value.__cause__ is None and raised.value.__context__ is None


def test_converter_errors_are_wrapped_or_passed_through(monkeypatch):
    def raises_runtime_error(record, context):
        raise RuntimeError("boom")

    def returns_wrong_type(record, context):
        return {"not": "an episode"}

    def own_episode_error(record, context):
        raise EpisodeConversionError("call_id_missing")

    monkeypatch.setattr(
        convert_module,
        "_CONVERTERS",
        {"runtime": raises_runtime_error, "wrong": returns_wrong_type, "own": own_episode_error},
    )
    for provider in ("runtime", "wrong"):
        with pytest.raises(EpisodeConversionError) as raised:
            convert_call(provider, {}, CONTEXT)
        assert raised.value.code == "converter_error"
    with pytest.raises(EpisodeConversionError) as raised:
        convert_call("own", {}, CONTEXT)
    assert raised.value.code == "call_id_missing"


def test_context_is_strict_and_keeps_list_order(monkeypatch):
    seen = []
    monkeypatch.setattr(convert_module, "_CONVERTERS", {"fake": lambda record, context: seen.append(context)})
    for context in (
        {"tenant_id": b"t", "declared_capabilities": []},
        {"tenant_id": "t", "declared_capabilities": {"transcript", "turn_timing"}},
    ):
        with pytest.raises(EpisodeConversionError) as raised:
            convert_call("fake", {}, context)
        assert raised.value.code == "context_invalid"
        # convert_record takes a built context, so the refusal happens at construction.
        with pytest.raises(ValidationError):
            ConvertContext(**context)
    assert seen == []

    ordered = ["turn_timing", "transcript"]
    assert ConvertContext(tenant_id="t", declared_capabilities=ordered).declared_capabilities == tuple(ordered)
    # The fake returns no Episode; reaching it shows the list was accepted.
    with pytest.raises(EpisodeConversionError) as raised:
        convert_call("fake", {}, {"tenant_id": "t", "declared_capabilities": ordered})
    assert raised.value.code == "converter_error"
    assert seen[0].declared_capabilities == ("turn_timing", "transcript")


def test_convert_error_codes_in_order(monkeypatch):
    calls = []
    monkeypatch.setattr(convert_module, "_CONVERTERS", {"fake": lambda record, context: calls.append(1)})

    def code_of(provider, record, context):
        with pytest.raises(EpisodeError) as raised:
            convert_call(provider, record, context)
        assert str(raised.value) == raised.value.code
        return type(raised.value), raised.value.code

    bad_context = {"tenant_id": 5, "declared_capabilities": ["bogus"]}
    assert code_of("vapi", "not a mapping", bad_context) == (UnknownProviderError, "unknown_provider")
    for context in (
        {"tenant_id": "t"},
        {"declared_capabilities": []},
        {"tenant_id": "t", "declared_capabilities": [], "origin": "simulation"},
        {"tenant_id": 5, "declared_capabilities": []},
        {"tenant_id": "t", "declared_capabilities": "transcript"},
        {"tenant_id": "", "declared_capabilities": []},
        ["tenant_id"],
    ):
        assert code_of("fake", "not a mapping", context) == (EpisodeConversionError, "context_invalid")
    with_bogus = {"tenant_id": "t", "declared_capabilities": ["bogus", "transcript", "transcript"]}
    assert code_of("fake", ["not", "a", "mapping"], with_bogus) == (EpisodeConversionError, "record_not_object")
    assert code_of("fake", {}, with_bogus) == (EpisodeConversionError, "unknown_capability")
    repeated = {"tenant_id": "t", "declared_capabilities": ["transcript", "transcript"]}
    assert code_of("fake", {}, repeated) == (EpisodeConversionError, "duplicate_capability")
    assert calls == []
