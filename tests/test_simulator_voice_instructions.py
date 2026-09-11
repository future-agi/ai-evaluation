from fi.alk.harness.simulator_voice import SIMULATOR_INSTRUCTIONS


def test_simulator_waits_for_final_confirmation_before_closing() -> None:
    assert "Wait for the complete question" in SIMULATOR_INSTRUCTIONS
    assert "booking summary is not a completed outcome" in SIMULATOR_INSTRUCTIONS
    assert "Do not use goodbye" in SIMULATOR_INSTRUCTIONS
    assert "Follow sequence words literally" in SIMULATOR_INSTRUCTIONS
    assert "do not reveal or request the later action" in SIMULATOR_INSTRUCTIONS


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
