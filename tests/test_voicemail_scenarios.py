"""A mailbox answering an outbound call, which tests the agent and not the caller."""

from test_harness import _built_environment


def _mailbox(name: str, greeting: str, **overrides):
    """A voicemail scenario as a writer would submit one, against the cart world's catalogue."""
    from fi.alk.harness.scenario import Persona, Scenario

    payload = {
        "name": name,
        "instruction": "Play the greeting once and say nothing else for the rest of the call.",
        "persona": Persona(
            name=name,
            personality="patient",
            communication_style="brief",
            initial_message=greeting,
            languages=["English"],
            accent="American",
            keywords=["mailbox"],
        ),
        "fixture": {"origin": "seed"},
        "solution": [
            {"tool": "lst", "arguments": {}},
            {"tool": "add", "arguments": {"item_id": "big_mac"}},
        ],
        "sub_goals": ["item-added"],
        "call_direction": "outbound",
        "answered_by": "voicemail",
    }
    payload.update(overrides)
    return Scenario(**payload)


def _person(index: int):
    return _mailbox(
        f"person-{index}",
        f"Hello, this is caller {index}.",
        answered_by="",
        call_direction="",
    )


def test_a_mailbox_can_only_answer_a_call_the_agent_placed():
    """`call_direction` empty defers to the contract, so a voicemail scenario that does not state
    outbound itself is one the run may legally make inbound."""
    from fi.alk.harness.scenario import answered_by_problems

    assert answered_by_problems(_mailbox("named", "Hi, this is Dana.")) == []
    assert answered_by_problems(_person(1)) == []

    silent = answered_by_problems(_mailbox("undeclared", "Hi.", call_direction=""))
    assert silent and "must state call_direction 'outbound'" in silent[0]

    inbound = answered_by_problems(
        _mailbox("dialled-in", "Hi.", call_direction="inbound")
    )
    assert inbound and "only happens on a call the agent placed" in inbound[0]

    unknown = answered_by_problems(
        _mailbox("odd", "Hi.", answered_by="answering machine")
    )
    assert unknown and "answered_by must be" in unknown[0]


def test_validation_reports_a_mailbox_that_could_not_have_answered(tmp_path):
    """The rule reaches a writer through the same list every other problem does."""
    from fi.alk.harness.scenario import validate_scenario

    root, _contract, catalogue = _built_environment(tmp_path)
    said = " ".join(
        validate_scenario(
            _mailbox("dialled-in", "Hi.", call_direction="inbound"), catalogue, {}
        )
    )
    assert "only happens on a call the agent placed" in said


def test_voicemail_stays_a_minority_of_a_suite():
    """One narrow test, worth a few scenarios and never a theme."""
    from fi.alk.harness.scenario import VOICEMAIL, suite_diversity_problems

    over = [_mailbox(f"box-{index}", f"Mailbox {index}.") for index in range(3)] + [
        _person(index) for index in range(3)
    ]
    assert f"are answered_by {VOICEMAIL!r}" in " ".join(suite_diversity_problems(over))

    within = [_mailbox("box-only", "You have reached Dana.")] + [
        _person(index) for index in range(11)
    ]
    assert f"are answered_by {VOICEMAIL!r}" not in " ".join(
        suite_diversity_problems(within)
    )


def test_the_rare_condition_ceiling_is_one_in_twenty():
    """The share is a product judgement, so the number it produces is pinned here rather than left
    to be inferred. A ceiling with no floor: none at all is a legitimate suite."""
    from fi.alk.harness.scenario import RARE_CONDITION_SHARE, rare_event_ceiling

    assert RARE_CONDITION_SHARE == 0.05
    # Rounded up, so a short suite is allowed one rather than none.
    assert rare_event_ceiling(4) == 1
    assert rare_event_ceiling(10) == 1
    assert rare_event_ceiling(20) == 1
    assert rare_event_ceiling(21) == 2
    assert rare_event_ceiling(50) == 3
    assert rare_event_ceiling(200) == 10


def test_several_mailboxes_have_to_be_different_mailboxes():
    from fi.alk.harness.scenario import suite_diversity_problems

    same = [_mailbox(f"same-{index}", "Leave a message.") for index in range(3)] + [
        _person(index) for index in range(17)
    ]
    assert "use the same greeting" in " ".join(suite_diversity_problems(same))

    varied = [
        _mailbox("named", "You have reached Dana Whitfield."),
        _mailbox("carrier", "The person you called is not available."),
        _mailbox("full", "This mailbox is full."),
    ] + [_person(index) for index in range(17)]
    assert "use the same greeting" not in " ".join(suite_diversity_problems(varied))


def test_the_mailbox_replaces_the_callers_rules_rather_than_adding_to_them():
    from fi.alk.harness.simulator_voice import (
        SIMULATOR_INSTRUCTIONS,
        simulator_instructions,
    )

    mailbox = simulator_instructions("outbound", "unaware", "voicemail")
    assert "VOICEMAIL SYSTEM" in mailbox
    assert "Never end the call" in mailbox
    assert SIMULATOR_INSTRUCTIONS not in mailbox

    assert simulator_instructions() == SIMULATOR_INSTRUCTIONS
    assert simulator_instructions("inbound", "expecting") == SIMULATOR_INSTRUCTIONS
    assert "THIS CALL WAS PLACED TO YOU" in simulator_instructions(
        "outbound", "unaware"
    )


def test_the_simulator_definition_reads_the_mailbox_from_its_lane():
    from fi.alk.harness.simulator_voice import simulator_definition

    settings = {
        "SIMULATOR_LLM_PROVIDER": "google",
        "HARNESS_CALL_DIRECTION": "outbound",
        "HARNESS_ANSWERED_BY": "voicemail",
    }
    made = simulator_definition(lambda name: settings.get(name, ""))
    assert "VOICEMAIL SYSTEM" in made.instructions

    settings.pop("HARNESS_ANSWERED_BY")
    person = simulator_definition(lambda name: settings.get(name, ""))
    assert "VOICEMAIL SYSTEM" not in person.instructions


def test_a_judged_only_mailbox_is_refused_where_the_world_can_be_read(tmp_path):
    """The judged-only path in `prove` is for a world with nothing readable. A real world has rows,
    so a mailbox scenario there still has to name a sub-goal settled in code. Recorded rather than
    worked around."""
    from fi.alk.harness.prove import prove

    root, _contract, catalogue = _built_environment(tmp_path)

    judged = _mailbox("judged-mailbox", "You have reached Dana.", sub_goals=["polite"])
    proof = prove(judged, catalogue, root)
    assert not proof.holds and not proof.judged_only
    assert "is in code" in " ".join(proof.broken)

    settled = _mailbox("settled-mailbox", "You have reached Dana.")
    assert prove(settled, catalogue, root).holds


def _catalogue_with(check: str):
    """A catalogue holding one sub-goal whose check is the text under test."""
    from fi.alk.harness.catalogue import Catalogue, SubGoal

    return Catalogue(
        sub_goals=[SubGoal(name="needs-a-call", what="a tool ran", check=check)]
    )


def test_a_mailbox_scenario_may_not_require_a_tool_call():
    """Six measured mailbox calls failed on a tool the agent only reaches once somebody speaks, and on
    a mailbox nobody ever does. The scenario was wrong, not the agent."""
    from fi.alk.harness.scenario import voicemail_sub_goal_problems

    scenario = _mailbox("carrier", "The person you called is not available.")
    scenario.sub_goals = ["needs-a-call"]
    catalogue = _catalogue_with(
        "def check(world, calls):\n"
        "    if not any(one['name'] == 'get_booking_status' for one in calls):\n"
        "        return 'get_booking_status was not called'\n"
    )

    problems = voicemail_sub_goal_problems(scenario, catalogue)
    assert len(problems) == 1
    assert "needs-a-call" in problems[0]
    assert "nobody on the line" in problems[0]


def test_a_mailbox_sub_goal_about_not_calling_is_kept():
    """Talking to a machine is where an agent should stop calling things, so a check that fails when a
    call *was* made is exactly what a mailbox scenario should ask for."""
    from fi.alk.harness.scenario import voicemail_sub_goal_problems

    scenario = _mailbox("carrier", "The person you called is not available.")
    scenario.sub_goals = ["needs-a-call"]
    catalogue = _catalogue_with(
        "def check(world, calls):\n"
        "    if any(one['name'] == 'book_ride' for one in calls):\n"
        "        return 'booked a ride into a mailbox'\n"
    )

    assert voicemail_sub_goal_problems(scenario, catalogue) == []


def test_a_person_answering_keeps_every_sub_goal():
    """The rule is about mailboxes only; an ordinary call is untouched."""
    from fi.alk.harness.scenario import voicemail_sub_goal_problems

    scenario = _mailbox("person", "Hello?")
    scenario.answered_by = "person"
    scenario.sub_goals = ["needs-a-call"]
    catalogue = _catalogue_with(
        "def check(world, calls):\n"
        "    if not any(one['name'] == 'get_booking_status' for one in calls):\n"
        "        return 'get_booking_status was not called'\n"
    )

    assert voicemail_sub_goal_problems(scenario, catalogue) == []


def test_the_shape_a_real_writer_produced_is_caught():
    """Verbatim from run 574d31ec, where two mailbox scenarios failed on it after an earlier version of
    this validator matched neither `not any(` nor `if not calls` and let them through."""
    from fi.alk.harness.scenario import voicemail_sub_goal_problems

    scenario = _mailbox("carrier", "The person you called is not available.")
    scenario.sub_goals = ["needs-a-call"]
    catalogue = _catalogue_with(
        "def check(world, calls):\n"
        '    checks = [c for c in calls if c.name == "get_booking_status" and c.ok]\n'
        "    if not checks:\n"
        '        return "Booking status was not retrieved"\n'
    )

    problems = voicemail_sub_goal_problems(scenario, catalogue)
    assert len(problems) == 1
    assert "nobody on the line" in problems[0]


def test_a_positive_test_on_the_same_shape_is_still_kept():
    """The exemption has to survive the widening: failing when a call WAS made is a mailbox sub-goal."""
    from fi.alk.harness.scenario import voicemail_sub_goal_problems

    scenario = _mailbox("carrier", "The person you called is not available.")
    scenario.sub_goals = ["needs-a-call"]
    catalogue = _catalogue_with(
        "def check(world, calls):\n"
        '    booked = [c for c in calls if c.name == "book_ride" and c.ok]\n'
        "    if booked:\n"
        '        return "booked a ride into a mailbox"\n'
    )

    assert voicemail_sub_goal_problems(scenario, catalogue) == []


def test_the_switch_refuses_a_mailbox_scenario(monkeypatch) -> None:
    """Refused at validation as well as withheld from the schema, because a model that saw the field
    on an earlier turn can still ask for it."""
    from fi.alk.harness.scenario import answered_by_problems

    monkeypatch.setenv("ALK_VOICEMAIL_SCENARIOS", "0")
    said = " ".join(answered_by_problems(_mailbox("box", "You have reached Dana.")))
    assert "turned off for this run" in said

    monkeypatch.setenv("ALK_VOICEMAIL_SCENARIOS", "1")
    assert "turned off for this run" not in " ".join(
        answered_by_problems(_mailbox("box", "You have reached Dana."))
    )


def test_the_switch_withholds_the_fields_it_would_be_asked_through(monkeypatch) -> None:
    from fi.alk.harness.scenario_tools import _mailbox_fields

    monkeypatch.setenv("ALK_VOICEMAIL_SCENARIOS", "0")
    assert _mailbox_fields() == {}

    monkeypatch.setenv("ALK_VOICEMAIL_SCENARIOS", "1")
    assert set(_mailbox_fields()) == {"answered_by", "voicemail_style"}


def test_the_switch_stops_the_writer_being_told_mailboxes_exist(monkeypatch) -> None:
    """Gated through `applies_to`, the same seam that gates the voice skill itself, so turning it off
    removes the instructions rather than leaving them to be read and then refused."""
    from fi.alk.harness.config import discovered_skills

    on = discovered_skills(modality="voice", voicemail="on")
    off = discovered_skills(modality="voice", voicemail="off")

    assert "mailbox" in on.lower()
    assert "mailbox" not in off.lower()
    assert "voicemail" not in off.lower()
    # The rest of the voice skill is untouched by the switch.
    assert "background_noise" in off
    assert "An attempted transfer is not a completed one" in off


def test_the_switch_survives_every_allowlist_between_here_and_the_writer() -> None:
    """It has to cross four of them, and three silently dropping a name is how this went wrong for
    background noise. Pinned here so adding a fifth is a failing test rather than a quiet run."""
    from fi.alk.harness.hosted_authoring_entrypoint import _PASSTHROUGH
    from fi.alk.harness.hosted_entrypoint import _SIMULATOR_SECRET_ALIASES
    from fi.alk.harness.scenario import VOICEMAIL_SWITCH

    assert VOICEMAIL_SWITCH in _SIMULATOR_SECRET_ALIASES, "the call lane cannot see it"
    assert VOICEMAIL_SWITCH in _PASSTHROUGH, (
        "authoring cannot see it, so the writer cannot"
    )


def test_two_mailboxes_already_have_to_differ():
    """Three was reachable when the ceiling was a sixth of the suite. At a twentieth it needed forty
    one scenarios, which put the rule beyond every suite we run."""
    from fi.alk.harness.scenario import suite_diversity_problems

    same = [
        _mailbox("first", "Leave a message.", voicemail_style="personal"),
        _mailbox("second", "Leave a message.", voicemail_style="personal"),
    ] + [_person(index) for index in range(38)]
    said = " ".join(suite_diversity_problems(same))
    assert "use the same greeting" in said
    assert "voicemail_style" in said

    differing = [
        _mailbox(
            "named", "You have reached Dana Whitfield.", voicemail_style="personal"
        ),
        _mailbox(
            "network",
            "The person you called is not available.",
            voicemail_style="carrier",
        ),
    ] + [_person(index) for index in range(38)]
    said = " ".join(suite_diversity_problems(differing))
    assert "use the same greeting" not in said
    assert "voicemail_style" not in said
