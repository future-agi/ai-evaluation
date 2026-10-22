"""A run plays the platform's current edits over the sealed scenario document."""

from fi.alk.harness.background_noise import place_of, scenario_source
from fi.alk.harness.call_runner import _with_scenario_edits
from fi.alk.harness.chat_call_runner import _conversation_scenario

SEALED = {
    "scenario_key": "book-ride",
    "name": "book_ride",
    "instruction": "book a ride",
    "tests": "books the ride",
    "background_noise": "office",
    "max_turns": 10,
    "persona": {"name": "Asha", "accent": "Indian", "personality": "calm"},
}

EDITS = {
    "scenario_edits": {
        "book-ride": {
            "background_noise": "street",
            "max_turns": 4,
            "tests": "unused by a call",
            "instruction": "ignored",
            "persona": {
                "personality": "impatient",
                "languages": ["English", "Hindi"],
                "name": "ignored",
            },
        }
    }
}


def test_edits_replace_only_what_a_call_plays():
    merged = _with_scenario_edits(SEALED, EDITS, "book-ride")

    assert merged["background_noise"] == "street"
    assert merged["persona"] == {
        "name": "Asha",
        "accent": "Indian",
        "personality": "impatient",
        "languages": ["English", "Hindi"],
    }
    assert merged["max_turns"] == 4
    assert merged["tests"] == "books the ride"
    assert merged["instruction"] == "book a ride"
    assert SEALED["background_noise"] == "office"


def test_a_voice_call_plays_the_edited_place():
    merged = _with_scenario_edits(SEALED, EDITS, "book-ride")

    source = scenario_source(merged["background_noise"], merged.get("fixture"), seed="book_ride|run")

    assert place_of(source) == "street"


def test_a_chat_caller_is_the_edited_persona():
    merged = _with_scenario_edits(SEALED, EDITS, "book-ride")

    scenario = _conversation_scenario(merged)

    assert scenario.persona.personality == "impatient"
    assert scenario.max_turns == 4


def test_no_edits_leave_the_sealed_document():
    for metadata in (None, {}, {"scenario_edits": {}}, {"scenario_edits": {"other": {}}}):
        assert _with_scenario_edits(SEALED, metadata, "book-ride") == SEALED
