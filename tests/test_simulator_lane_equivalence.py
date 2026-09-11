"""The two lanes must build the same caller from the same inputs.

They drifted once: the hosted lane hand-mirrored the local spec builder and silently lost the
objective fix, the numbered conduct rules, Cartesia voice selection, persona STT language and
background noise. This asserts the equivalence directly so the next added field cannot repeat it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from fi.alk.harness.call_runner import _build_spec
from fi.alk.harness.simulator_voice import caller_scenario, fixture_caller_phone
from fi.alk.harness.run.sdk_voice import build_spec as local_build_spec

PERSONA = {
    "name": "Noor",
    "gender": "female",
    "accent": "Indian",
    "languages": ["Hindi"],
    "metadata": {"caller_phone": "+14155550107"},
}

# A developer shell carrying any of these would silence the comparison.
OVERRIDES = (
    "SIMULATOR_LLM_PROVIDER",
    "SIMULATOR_STT_PROVIDER",
    "SIMULATOR_TTS_PROVIDER",
    "SIMULATOR_STT_LANGUAGE",
    "SIMULATOR_TTS_VOICE",
    "SIMULATOR_LLM_MODEL",
    "SIMULATOR_STT_MODEL",
    "SIMULATOR_TTS_MODEL",
    "HARNESS_OUTCOME",
    "ACCEPTANCE_LIVEKIT_URL",
)
INSTRUCTION = "You are Noor calling to book a ride to the office."
TESTS = "passes when the pickup and dropoff are confirmed and the booking is made"
DOC = {
    "persona": PERSONA,
    "instruction": INSTRUCTION,
    "tests": TESTS,
    "fixture": {"origin": "seed", "phone": "+14155550107"},
    "scenario_key": "noor-books-a-ride",
    "scenario_id": "scn_noor",
}
SHARED_ENV = {
    "CARTESIA_API_KEY": "test-cartesia-key",
    "DEEPGRAM_API_KEY": "test-deepgram-key",
    "LIVEKIT_URL": "wss://example.livekit.cloud",
    "LIVEKIT_TARGET_AGENT_NAME": "agent-w0",
    "LIVEKIT_TARGET_SYSTEM_PROMPT": INSTRUCTION,
    "HARNESS_PERSONA": json.dumps(PERSONA),
    "HARNESS_SCENARIO": "noor-books-a-ride",
    "HARNESS_INSTRUCTION": INSTRUCTION,
    "HARNESS_FIXTURE": json.dumps({"origin": "seed", "phone": "+14155550107"}),
}


def _dimensions(spec) -> dict:
    simulator = spec.environment.config["simulator"]
    persona = spec.scenario.dataset[0]
    return {
        "llm": simulator["llm"],
        "stt": simulator["stt"],
        "tts": simulator["tts"],
        "instructions": simulator["instructions"],
        "persona_voice": persona.persona.get("voice"),
        "persona_metadata": persona.persona.get("metadata"),
        "outcome": persona.outcome,
    }


@pytest.fixture()
def both_specs(tmp_path: Path, monkeypatch):
    for name in OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    with mock.patch.dict(os.environ, SHARED_ENV, clear=False):
        local = local_build_spec("run-equivalence")
        hosted = _build_spec(
            run_id="run-equivalence",
            room_name="harness-run-equivalence",
            agent_name="agent-w0",
            doc=DOC,
            livekit_url=SHARED_ENV["LIVEKIT_URL"],
            call_timeout_seconds=300.0,
            run_seconds=300.0,
            recordings_root=tmp_path,
            simulator_config={},
            environ=SHARED_ENV,
        )
    return local, hosted


def test_both_lanes_build_the_same_caller(both_specs):
    local, hosted = both_specs
    assert _dimensions(hosted) == _dimensions(local)


def test_neither_lane_hands_the_caller_the_pass_criteria(both_specs):
    local, hosted = both_specs
    for spec in (local, hosted):
        assert spec.scenario.dataset[0].outcome == ""
    assert TESTS not in json.dumps(_dimensions(hosted), default=str)


def test_a_cartesia_voice_is_not_the_cartesia_model(both_specs):
    _local, hosted = both_specs
    tts = hosted.environment.config["simulator"]["tts"]
    if tts.get("provider") == "cartesia":
        assert tts["voice"] != tts["model"]


def test_the_caller_carries_the_scenario_phone():
    scenario = caller_scenario(
        name=DOC["scenario_key"],
        persona=DOC["persona"],
        situation=DOC["instruction"],
        fixture=DOC["fixture"],
        tts_provider="cartesia",
    )
    assert scenario.dataset[0].persona["metadata"]["caller_phone"] == "+14155550107"


def test_only_the_callers_own_number_is_followed_into_a_nested_fixture():
    # A phone under a driver, a business or a support line belongs to someone else. Handing it
    # over sends the call in as the wrong rider, which every identity-dependent check then reads.
    assert fixture_caller_phone(
        {"support_line": {"phone": "+18005551234"}, "rider": {"caller_phone": "+14155550102"}}
    ) == "+14155550102"
    # A lone number is the caller's, whoever the fixture filed it under. Only a competing
    # caller-scoped key outranks it.
    assert fixture_caller_phone({"rider": {"phone": "+14155550102"}}) == "+14155550102"


def test_the_caller_phone_is_found_however_the_fixture_nests_it():
    # A fixture that nests the number, or names it with any accepted alias, must still reach the
    # target. Missing it sends every persona in as the demo rider.
    assert fixture_caller_phone({"phone": "+14155550107"}) == "+14155550107"
    assert fixture_caller_phone({"ani": "+14155550102"}) == "+14155550102"
    assert fixture_caller_phone({"rider": {"caller_phone": "+14155550103"}}) == "+14155550103"
    assert fixture_caller_phone({"origin": "seed"}) == ""


def test_a_nested_fixture_phone_reaches_the_persona_metadata():
    scenario = caller_scenario(
        name="nested",
        persona={"name": "Noor"},
        situation="book a ride",
        fixture={"origin": "seed", "rider": {"caller_ani": "+14155550109"}},
        tts_provider="cartesia",
    )
    assert scenario.dataset[0].persona["metadata"]["caller_phone"] == "+14155550109"
