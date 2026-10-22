from fi.alk.harness.simulator_voice import SIMULATOR_INSTRUCTIONS


def test_simulator_waits_for_final_confirmation_before_closing() -> None:
    assert "Wait for the complete question" in SIMULATOR_INSTRUCTIONS
    assert "booking summary is not a completed outcome" in SIMULATOR_INSTRUCTIONS
    assert "Do not use goodbye" in SIMULATOR_INSTRUCTIONS
    assert "Follow sequence words literally" in SIMULATOR_INSTRUCTIONS
    assert "do not reveal or request the later action" in SIMULATOR_INSTRUCTIONS


def test_simulator_speaks_opaque_identifiers_character_by_character() -> None:
    assert "R E S, J K nine zero" in SIMULATOR_INSTRUCTIONS
    assert "one character at a time" in SIMULATOR_INSTRUCTIONS


def test_the_caller_may_not_claim_an_action_it_cannot_perform():
    """The deadlock this closes: the caller says it tapped a link, the world never changed, and the
    agent waits for a state change that cannot arrive."""
    from fi.alk.harness.simulator_voice import (
        SIMULATOR_INSTRUCTIONS,
        simulator_instructions,
    )

    assert "tapped a link" in SIMULATOR_INSTRUCTIONS
    assert "nothing has arrived" in SIMULATOR_INSTRUCTIONS
    # Carried by every framing a person answers under, and absent from the mailbox, which has no
    # rules of its own.
    assert "tapped a link" in simulator_instructions()
    assert "tapped a link" in simulator_instructions("outbound", "unaware")
    assert "tapped a link" not in simulator_instructions(
        "outbound", "unaware", "voicemail"
    )


def test_the_caller_asks_again_for_the_part_of_its_question_left_unanswered():
    assert "Hold every answer against what you asked" in SIMULATOR_INSTRUCTIONS
    assert "ask for that part once" in SIMULATOR_INSTRUCTIONS


def test_the_closing_turn_answers_the_agents_last_question():
    assert "the answer goes in that closing turn too" in SIMULATOR_INSTRUCTIONS


def test_a_caller_heard_through_a_bed_knows_where_it_is():
    """A caller told nothing about its bed answered an agent asking about the noise with 'I am in
    a completely quiet room', which the recording contradicts."""
    from fi.alk.harness.simulator_voice import simulator_instructions

    assert "an office" in simulator_instructions(place="office")
    assert "never claim to be somewhere quiet" in simulator_instructions("outbound", "unaware", place="street")
    assert "even if your situation says you were somewhere else" in simulator_instructions(place="home")
    assert "WHERE YOU ARE" not in simulator_instructions()
    assert "WHERE YOU ARE" not in simulator_instructions(place="nowhere")
