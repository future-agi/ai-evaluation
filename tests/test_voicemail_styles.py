"""Which kind of mailbox answered, and the tone that follows its greeting."""

from test_voicemail_scenarios import _mailbox, _person


def _problems(scenario):
    from fi.alk.harness.scenario import voicemail_style_problems

    return voicemail_style_problems(scenario)


def test_every_named_style_is_accepted_and_an_unknown_one_is_refused():
    from fi.alk.harness.scenario import VOICEMAIL_STYLES

    for style in VOICEMAIL_STYLES:
        assert _problems(_mailbox("m", "Hi.", voicemail_style=style)) == []

    said = _problems(_mailbox("m", "Hi.", voicemail_style="android"))
    assert said and "voicemail_style must be" in said[0]
    assert "personal" in said[0] and "full" in said[0]


def test_a_style_without_a_mailbox_is_refused_because_nothing_plays_it():
    said = _problems(_person(1).model_copy(update={"voicemail_style": "carrier"}))
    assert said and "answered_by is not 'voicemail'" in said[0]


def test_no_style_is_the_ordinary_case():
    assert _problems(_mailbox("m", "Hi.")) == []


def test_the_greeting_wording_follows_the_style():
    from fi.alk.harness.simulator_voice import simulator_instructions

    personal = simulator_instructions("outbound", "unaware", "voicemail", "personal")
    carrier = simulator_instructions("outbound", "unaware", "voicemail", "carrier")
    operator = simulator_instructions("outbound", "unaware", "voicemail", "operator")
    full = simulator_instructions("outbound", "unaware", "voicemail", "full")

    # Every style keeps the rules that make a mailbox a mailbox.
    for text in (personal, carrier, operator, full):
        assert "VOICEMAIL SYSTEM" in text
        assert "Never end the call" in text

    assert "your own words" in personal
    assert "names nobody" in carrier
    assert "automated voice messaging system" in operator
    # The one that must never invite a message it cannot take.
    assert "FULL" in full
    assert "never mention a tone" in full.lower()
    assert len({personal, carrier, operator, full}) == 4


def test_an_unset_style_speaks_the_personal_greeting():
    from fi.alk.harness.simulator_voice import simulator_instructions

    assert simulator_instructions(
        "outbound", "unaware", "voicemail"
    ) == simulator_instructions("outbound", "unaware", "voicemail", "personal")


def test_a_person_answering_is_untouched_by_a_style():
    from fi.alk.harness.simulator_voice import (
        SIMULATOR_INSTRUCTIONS,
        simulator_instructions,
    )

    assert (
        simulator_instructions("inbound", "expecting", "", "carrier")
        == SIMULATOR_INSTRUCTIONS
    )


def test_three_mailboxes_of_one_style_are_reported_and_two_styles_are_not():
    from fi.alk.harness.scenario import suite_diversity_problems

    same = [
        _mailbox(f"m-{index}", f"Greeting {index}.", voicemail_style="personal")
        for index in range(3)
    ] + [_person(index) for index in range(15)]
    said = " ".join(suite_diversity_problems(same))
    assert "voicemail_style 'personal'" in said

    varied = [
        _mailbox("m-0", "Greeting 0.", voicemail_style="personal"),
        _mailbox("m-1", "Greeting 1.", voicemail_style="carrier"),
        _mailbox("m-2", "Greeting 2.", voicemail_style="full"),
    ] + [_person(index) for index in range(15)]
    assert "voicemail_style" not in " ".join(suite_diversity_problems(varied))
