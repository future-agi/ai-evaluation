from fi.simulate.simulation.models import Persona
from fi.simulate.simulation.voice_prompt import build_voice_simulator_prompt


def _persona() -> Persona:
    return Persona(
        persona={
            "name": "Priya",
            "occupation": "Nurse",
            "age_group": "35-44",
            "location": "Mumbai",
            "gender": "female",
            "personality": "anxious",
            "communication_style": "questioning",
            "language": "en",
            "accent": "Indian",
            "metadata": {"appointment_id": "APT-123"},
        },
        situation="Your specialist appointment was cancelled without notice.",
        outcome="Get a new appointment time and confirm the clinic location.",
    )


def test_voice_prompt_preserves_complete_platform_persona_rules() -> None:
    prompt = build_voice_simulator_prompt(
        _persona(),
        call_type="inbound",
        agent_name="healthcare assistant",
    )

    for section in (
        "# YOUR IDENTITY",
        "# YOUR CURRENT SITUATION",
        "# YOUR PERSONALITY & COMMUNICATION",
        "# LANGUAGE & SPEECH PATTERNS",
        "# CONTEXTUAL AWARENESS",
        "# ADDITIONAL CHARACTERISTICS",
        "# HOW TO BE THIS PERSON",
        "# CONVERSATION EXECUTION RULES",
        "## Voice-Natural Formatting",
        "## Embody Your Situation",
    ):
        assert section in prompt
    assert "Your specialist appointment was cancelled without notice." in prompt
    assert "Get a new appointment time and confirm the clinic location." in prompt
    assert "Never Break Character" in prompt
    assert "endCall tool" in prompt
    assert "Let the situation guide your behavior, not your narration" in prompt


def test_simulator_instructions_supplement_scenario_prompt() -> None:
    prompt = build_voice_simulator_prompt(
        _persona(),
        call_type="inbound",
        additional_instructions="Ask for an escalation if a same-day appointment is unavailable.",
    )

    assert "# ADDITIONAL SIMULATOR INSTRUCTIONS" in prompt
    assert "Ask for an escalation" in prompt
    assert "Your specialist appointment was cancelled without notice." in prompt
    assert "Get a new appointment time and confirm the clinic location." in prompt
    # The call's own instructions come after the general rules, and the objective closes the
    # prompt: what lands last is what survives a long conversation.
    assert prompt.index("# ADDITIONAL SIMULATOR INSTRUCTIONS") > prompt.index(
        "# CONVERSATION EXECUTION RULES"
    )
    assert prompt.rstrip().endswith("applies at turn twenty exactly as it applied at turn one.")


def test_prompt_closes_by_naming_who_the_caller_is() -> None:
    """A drifting caller answered as the agent and addressed itself by its own name, which reads
    as the agent talking to itself. The identity is restated last, where it survives a long call."""
    prompt = build_voice_simulator_prompt(_persona(), call_type="inbound")

    tail = prompt.rsplit("---", 1)[-1]
    assert "You are Priya" in tail
    assert "never address Priya" in tail
    assert tail.index("You are Priya") < tail.index("What you came for:")
