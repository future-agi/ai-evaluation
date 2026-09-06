"""A second voice in the room, and the caller never claiming an action it cannot perform."""

from test_harness import _built_environment


def _scenario(name: str, **overrides):
    """A scenario as a writer would submit one, against the cart world's catalogue."""
    from fi.alk.harness.scenario import Persona, Scenario

    payload = {
        "name": name,
        "instruction": "Order the usual and pay with the saved card.",
        "persona": Persona(
            name=name,
            personality="patient",
            communication_style="brief",
            initial_message="Hi, I would like to order.",
            languages=["English"],
            accent="American",
            keywords=["order"],
        ),
        "fixture": {"origin": "seed"},
        "solution": [
            {"tool": "lst", "arguments": {}},
            {"tool": "add", "arguments": {"item_id": "big_mac"}},
        ],
        "sub_goals": ["item-added"],
    }
    payload.update(overrides)
    return Scenario(**payload)


def test_a_bystander_is_one_line_and_not_across_a_mailbox():
    from fi.alk.harness.scenario import LONGEST_BYSTANDER, bystander_problems

    assert bystander_problems(_scenario("quiet")) == []
    assert bystander_problems(_scenario("child", bystander="Mum, are we nearly there")) == []

    mailbox = bystander_problems(
        _scenario(
            "over-a-recording",
            bystander="Mum, are we nearly there",
            answered_by="voicemail",
            call_direction="outbound",
        )
    )
    assert mailbox and "no room for somebody to speak across" in mailbox[0]

    long = bystander_problems(_scenario("speech", bystander="x" * (LONGEST_BYSTANDER + 1)))
    assert long and "not a second conversation" in long[0]


def test_validation_reports_a_bystander_that_could_not_be_there(tmp_path):
    """The rule reaches a writer through the same list every other problem does."""
    from fi.alk.harness.scenario import validate_scenario

    _root, _contract, catalogue = _built_environment(tmp_path)
    said = " ".join(
        validate_scenario(
            _scenario(
                "over-a-recording",
                bystander="Mum, are we nearly there",
                answered_by="voicemail",
                call_direction="outbound",
            ),
            catalogue,
            {},
        )
    )
    assert "no room for somebody to speak across" in said


def test_a_bystander_stays_a_minority_of_a_suite():
    from fi.alk.harness.scenario import suite_diversity_problems

    over = [_scenario(f"noisy-{index}", bystander=f"Line {index}") for index in range(3)] + [
        _scenario(f"quiet-{index}") for index in range(3)
    ]
    assert "have a bystander speaking" in " ".join(suite_diversity_problems(over))

    within = [_scenario("noisy-only", bystander="Mum, are we nearly there")] + [
        _scenario(f"quiet-{index}") for index in range(11)
    ]
    assert "have a bystander speaking" not in " ".join(suite_diversity_problems(within))


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
    assert "tapped a link" not in simulator_instructions("outbound", "unaware", "voicemail")
