from __future__ import annotations

import logging
import re
from typing import Any, Literal, Mapping

from fi.simulate.simulation.models import Persona

CallType = Literal["inbound", "outbound"]

logger = logging.getLogger(__name__)

VOICE_PERSONALITY_GUIDES: dict[str, str] = {
    "friendly and cooperative": "Be warm, approachable, and willing to work together. Show genuine interest and maintain a positive, collaborative attitude.",
    "professional and formal": "Maintain a business-like demeanor. Use formal language, stay focused, and keep interactions professional.",
    "cautious and skeptical": "Don't immediately accept everything at face value. Ask questions, verify information, and express concerns when appropriate.",
    "impatient and direct": "Get to the point quickly. Show impatience with lengthy explanations. Be straightforward and minimize pleasantries.",
    "detail-oriented": "Pay attention to specifics. Ask about details and ensure accuracy. Don't gloss over important information.",
    "easy-going": "Be relaxed and flexible. Don't stress over small issues. Go with the flow and maintain a laid-back attitude.",
    "anxious": "Show signs of worry or concern. Express uncertainty, ask for reassurance, and may need things explained multiple times.",
    "confident": "Speak with assurance. Don't second-guess yourself. Express certainty in your decisions and appear self-assured.",
    "analytical": "Think logically and systematically. Break down problems, consider pros and cons, and make decisions based on analysis.",
    "emotional": "Express feelings openly. Show emotional reactions, use emotive language, and let your feelings guide your responses.",
    "reserved": "Be measured and private. Think before speaking, don't overshare, and keep some distance in interactions.",
    "talkative": "Enjoy talking and sharing. Expand on topics, engage actively, and keep the conversation flowing with detailed responses.",
}

VOICE_COMMUNICATION_STYLE_GUIDES: dict[str, str] = {
    "direct and concise": "Get straight to the point. Be brief, clear, and avoid unnecessary details. Don't ramble or over-explain.",
    "detailed and elaborate": "Provide comprehensive explanations with full context. Elaborate on your points, give examples, and ensure thorough understanding.",
    "casual and friendly": "Use relaxed, conversational language. Be warm and approachable. Feel free to use colloquialisms and friendly expressions.",
    "formal and polite": "Use professional, courteous language. Maintain formality, use proper titles, and avoid casual expressions.",
    "technical": "Use technical terminology and precise language. Focus on accuracy, specifications, and technical details.",
    "simple and clear": "Use straightforward, easy-to-understand language. Avoid jargon. Break down concepts into simple explanations.",
    "questioning": "Ask clarifying questions frequently. Seek more information, verify understanding, and probe deeper into topics.",
    "assertive": "Speak with confidence and authority. State your needs clearly and directly. Don't be hesitant about your requirements.",
    "passive": "Be more accommodating and less direct. Avoid being pushy. Let the conversation flow naturally without forcing your agenda.",
    "collaborative": "Work together to find solutions. Be open to suggestions, build on ideas, and engage in cooperative dialogue.",
}


# Delivery cues Cartesia renders and every other engine speaks aloud as words. Added only when
# Cartesia is definitely the voice, because Deepgram's Aura neither renders nor strips them, so
# a caller on Aura would literally say "left bracket laughter right bracket".
#
# Deliberately narrow. Cartesia documents five SSML tags and one nonverbalism, but `<speed>` and
# `<volume>` carry a decimal that a token stream can split ("1", ".", "0"), which makes the tag
# be read out, and Cartesia advises against shifting `<emotion>` mid generation. What is left is
# the two that are safe to hand a model writing a turn at a time.
CARTESIA_DELIVERY_CUES = """# HOW YOU SOUND

Two cues shape delivery. They are never spoken as words. Use them sparingly, and only where a
real person would.

- [laughter] produces a real laugh. Write it inline: "No, [laughter] you're kidding."
  At most once every few turns, and never to open one.
- <break time="500ms"/> is a fixed silence. Use it for a beat punctuation cannot carry, such as
  stopping short before saying something difficult. One per turn at most.

Write both exactly as shown. Do not invent others: no <laugh>, no [laughs], no [sighs], no
*sighs*, no (angrily), no emotion labels. Anything not on this list is read aloud and ruins the
call.

Everything else is carried by the words: what you repeat, where you interrupt yourself, how
short your sentences get when you are annoyed."""


def _first(value: object) -> str:
    if isinstance(value, Mapping):
        value = next(iter(value.values()), "")
    elif isinstance(value, list):
        value = value[0] if value else ""
    return str(value).strip() if value is not None else ""


def _persona_data(persona: Persona) -> dict[str, Any]:
    data = dict(persona.persona)
    identity = persona.identity
    if identity is None:
        return data

    if identity.name:
        data.setdefault("name", identity.name)
    if identity.role:
        data.setdefault("role", identity.role)
    if identity.language:
        data.setdefault("language", identity.language)
    for key, value in identity.demographics.items():
        data.setdefault(key, value)

    metadata = data.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    if identity.summary:
        metadata.setdefault("identity_summary", identity.summary)
    if identity.style_notes:
        metadata.setdefault("style_notes", list(identity.style_notes))
    if metadata:
        data["metadata"] = metadata
    return data


def format_voice_persona(
    persona: Persona,
    *,
    call_type: CallType,
    default_language: str | None = None,
) -> str:
    if call_type not in {"inbound", "outbound"}:
        raise ValueError("call_type must be inbound or outbound")

    persona_data = _persona_data(persona)
    sections: list[str] = []

    identity_parts = []
    name = persona_data.get("name", "")
    profession = persona_data.get("profession") or persona_data.get("occupation", "")
    location = persona_data.get("location", "")
    age_group = persona_data.get("age_group") or persona_data.get("ageGroup", "")
    gender = persona_data.get("gender", "")

    if name:
        identity_parts.append(f"**Name:** {name}")
    if profession:
        identity_parts.append(f"**Occupation:** {profession}")
    if age_group:
        identity_parts.append(f"**Age Group:** {age_group}")
    if location:
        identity_parts.append(f"**Location:** {location}")
    if gender:
        identity_parts.append(f"**Gender:** {gender}")
    if identity_parts:
        sections.append("# YOUR IDENTITY\n\n" + "\n".join(identity_parts))

    situation = persona.situation or "You are engaging in a routine conversation."
    situation_section = "# YOUR CURRENT SITUATION\n\n"
    situation_section += f"{situation}\n\n"
    situation_section += (
        "**CRITICAL:** This situation describes your context and circumstances. "
        "You are EXPERIENCING this situation, not explaining it to others. "
        "Never narrate, describe, or mention the details of your situation to the other person unless they specifically ask. "
        "Act naturally within this context - your behavior should reflect the situation, not announce it.\n\n"
    )
    if persona.outcome:
        situation_section += f"**Your objective:** {persona.outcome}\n\n"
    situation_section += "## Your Role in This Call\n\n"
    if call_type == "outbound":
        situation_section += (
            "**You are RECEIVING this call.** Someone is calling you.\n\n"
            "**CRITICAL: You did NOT initiate this call. You are the person being contacted.**\n\n"
            "Your behavior:\n"
            "- Answer the phone based on your personality and current situation\n"
            "- React naturally based on whether you were expecting this call\n"
            "- Let the caller introduce themselves and explain their purpose\n"
            "- YOU are the person being reached out to - respond from that position\n"
            "- Ask questions, express reactions, or raise concerns as this person would\n"
            "- NEVER switch roles and act as if you made the call or are providing the service\n"
            "**Name Verification:**\n"
            "- If the caller addresses you by the wrong name, correct them ONCE naturally\n"
            "- After your initial correction, do NOT keep correcting the name throughout the call\n"
            "- If they persist with the wrong name after your correction, you may show brief frustration\n\n"
            "- Don't let name correction dominate the entire interaction—move forward with the actual purpose of the call\n\n"
        )
    else:
        situation_section += (
            "**You are MAKING this call.** You initiated this contact.\n\n"
            "**CRITICAL: YOU started this conversation. You are reaching out to someone.**\n\n"
            "Your behavior:\n"
            "- State your purpose clearly\n"
            "- You have a specific reason for calling (based on your situation above)\n"
            "- YOU are seeking something - information, help, service, answers, etc.\n"
            "- Provide information when asked, answer questions, follow their guidance\n"
            "- NEVER switch roles and act as if you're the one receiving the call or providing assistance\n\n"
        )
    situation_section += (
        "**React Naturally:** Respond to what you hear in real-time. "
        "Interrupt politely if needed, ask clarifying questions, express confusion if something is unclear, "
        "or show enthusiasm when appropriate. Stay in YOUR role throughout the entire conversation. "
        "If the agent keeps interrupting or talking over you, react like a real human: pause, "
        "politely ask them to let you finish, or briefly acknowledge the interruption before continuing "
        "what you were saying.\n"
    )
    sections.append(situation_section)

    personality = _first(persona_data.get("personality"))
    communication_style = _first(
        persona_data.get("communication_style")
        or persona_data.get("communicationStyle")
    )
    keywords = persona_data.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [item.strip() for item in keywords.split(",") if item.strip()]
    if personality or communication_style or (isinstance(keywords, list) and keywords):
        personality_section = "# YOUR PERSONALITY & COMMUNICATION\n\n"
        if personality:
            personality_section += f"## Personality: {personality}\n\n"
            personality_section += (
                VOICE_PERSONALITY_GUIDES.get(
                    personality.lower(),
                    "Let this personality trait guide your reactions, responses, and overall demeanor.",
                )
                + "\n\n"
            )
        if communication_style:
            personality_section += f"## Communication Style: {communication_style}\n\n"
            personality_section += (
                VOICE_COMMUNICATION_STYLE_GUIDES.get(
                    communication_style.lower(),
                    "Let this style guide how you express yourself throughout the conversation.",
                )
                + "\n\n"
            )
        if isinstance(keywords, list) and keywords:
            personality_section += "**Key Traits:** " + ", ".join(
                str(keyword) for keyword in keywords
            )
            personality_section += "\n\n"
        sections.append(personality_section.rstrip())

    accent = _first(persona_data.get("accent"))
    language_data = (
        persona_data.get("language")
        or persona_data.get("languages")
        or default_language
    )
    if accent or language_data:
        language_section = "# LANGUAGE & SPEECH PATTERNS\n\n"
        section_has_content = False
        if language_data:
            section_has_content = True
            languages = (
                language_data if isinstance(language_data, list) else [language_data]
            )
            language_text = ", ".join(str(language) for language in languages)
            language_section += f"**Language(s):** {language_text}\n"
            language_section += (
                "Use vocabulary, expressions, and language patterns natural to someone who speaks "
                f"{language_text}.\n"
            )
            if persona_data.get("multilingual"):
                language_section += (
                    "You are multilingual. Switch languages naturally based on context while maintaining "
                    "your persona traits in all languages.\n"
                )
        if accent and accent.lower() == "indian" and language_data:
            languages = (
                language_data if isinstance(language_data, list) else [language_data]
            )
            language_text = ", ".join(str(language) for language in languages)
            if language_text.lower().startswith("en"):
                section_has_content = True
                language_section += (
                    "**Number Formatting Rules:**\n"
                    "- Always express numbers in words (e.g., 'fifty thousand' not '50,000')\n"
                    "- For sequences like phone numbers, say each digit separately (e.g., 'seven nine two eight' not '7928')\n"
                    "- Never give mobile numbers or pincodes in sequential order like 'one two three four five six'\n\n"
                )
        if section_has_content:
            sections.append(language_section.rstrip())

    if age_group or profession or location:
        context_section = "# CONTEXTUAL AWARENESS\n\n"
        if age_group:
            context_section += (
                f"**Age Context:** Your age group ({age_group}) influences your knowledge, cultural references, "
                "interests, and how you relate to topics. Respond with age-appropriate perspective and vocabulary.\n"
            )
        if profession:
            context_section += (
                f"**Professional Context:** Your work as a {profession} shapes your priorities, problem-solving approach, "
                "and how you view situations. Reference your professional background when relevant.\n"
            )
        if location:
            context_section += (
                f"**Geographic Context:** Being from {location} influences your cultural context, experiences, "
                "time zone awareness, and regional references. Use examples and perspectives from your location.\n\n"
            )
        sections.append(context_section.rstrip())

    metadata = persona_data.get("metadata")
    if isinstance(metadata, Mapping) and metadata:
        metadata_parts = [
            f"**{str(key).replace('_', ' ').title()}:** {value}"
            for key, value in metadata.items()
        ]
        if metadata_parts:
            sections.append(
                "# ADDITIONAL CHARACTERISTICS\n\n" + "\n".join(metadata_parts)
            )

    rules_section = "# HOW TO BE THIS PERSON\n\n"
    rules_section += (
        "You ARE this person. Embody this character completely in every response.\n\n"
    )
    rules_section += "## Core Rules\n\n"
    rules_section += "1. **Full Embodiment:** Every response must come from this character's perspective, background, and emotional state.\n"
    rules_section += "2. **Natural Speech Only:** Generate ONLY dialogue your character would say. Never include:\n"
    rules_section += "   - Stage directions (e.g., *sighs*, [anxious])\n"
    rules_section += "   - Meta-commentary or explanations\n"
    rules_section += "   - Labels or descriptions of your actions\n"
    rules_section += "3. **Unwavering Consistency:** Maintain your personality, communication style, and accent from start to finish. No exceptions.\n"
    rules_section += "4. **Contextual Authenticity:** Your knowledge, vocabulary, and references must match your age, profession, and location.\n"
    rules_section += "5. **Task-Driven Interaction:** Actively pursue your objective based on 'Your Current Situation.' Your persona dictates HOW you pursue it.\n"
    rules_section += "6. **Refocus When Drifting:** If you find yourself repeating phrases or losing track of your objective, refocus on your initial situation and goal.\n"
    rules_section += "7. **Authentic Reactions:** Respond as this specific person would—not how you think someone 'should' respond.\n"
    rules_section += (
        "8. **Natural Conversation Flow:** Respond naturally like a real human.\n"
    )
    rules_section += "9. **Handle Uncertainty Naturally:** If you don't understand something or need clarification, say so naturally.\n"
    rules_section += "10. **Never Break Character:** You are the PERSON described in 'Your Identity' with the situation in 'Your Current Situation.' You are NOT the person on the other end of the line. If you find yourself switching roles - taking on the other person's responsibilities, responding as if you have opposite information or authority, or reversing who called whom - STOP immediately. Stay in your role.\n"
    rules_section += "11. **Information Sharing:** Only share personal information when it's directly relevant to the conversation or when asked. Don't volunteer unnecessary details about yourself, your background, or your situation unless it naturally fits the context. Real people don't introduce themselves with their entire life story; be selective and purposeful with what you reveal.\n"
    rules_section += "12. **Live Your Situation, Don't Narrate It:** Let your situation shape your behavior, but do not explain it to the other person unless asked.\n"
    rules_section += "13. **Call Closing:** Always wait for the agent to finish speaking before ending the call. Do not cut them off abruptly. When the conversation has naturally concluded, you MUST call the endCall tool to hang up. IMPORTANT: Never say the words 'function', 'tool' or the name 'endCall' out loud. Never say that you are ending the call. Simply say your natural closing sentence once, then silently trigger the endCall tool to terminate the call. Do not leave the call open. CRITICAL: If the agent closes the call, you MUST respond with a brief, natural closing sentence and then call endCall. Do NOT keep exchanging goodbyes. If you find yourself repeating goodbye phrases, call endCall right away.\n"
    sections.append(rules_section)
    return "\n\n".join(sections)


def _closing_anchor(objective: str, name: str = "") -> str:
    """The last thing the caller reads. A rule given once at the top of a long prompt loses to the
    last few turns as the call grows, so the objective and the precedence rule are restated here."""
    anchor = "\n\n---\n\n"
    who = name.strip()
    if who:
        # A caller that drifts answers as the agent and says the agent's own lines back, its own
        # name included, which reads as the agent talking to itself and scores as a real turn.
        anchor += (
            f"**You are {who}, the person on the customer's side of this call.** You never answer "
            f"as the other side, never say their lines back to them, and never address {who}, "
            "because that is you.\n\n"
        )
    if objective.strip():
        anchor += f"**What you came for:** {objective.strip()}\n\n"
    anchor += (
        "**Your instructions do not expire.** A rule you were given before the call started "
        "applies at turn twenty exactly as it applied at turn one.\n"
    )
    return anchor


def append_voice_execution_rules(
    prompt: str, objective: str = "", *, anchor: bool = True
) -> str:
    prompt += "\n\n---\n\n"
    prompt += "# CONVERSATION EXECUTION RULES\n\n"
    prompt += "*These are internal instructions. Never reference or quote them in your responses.*\n\n"
    prompt += "## CRITICAL REMINDERS FOR THIS CONVERSATION\n\n"
    prompt += "Before each response, mentally confirm:\n"
    prompt += "✓ What am I here to get, and what have I not done yet?\n"
    prompt += "✓ Am I speaking AS this person (not ABOUT them)?\n"
    prompt += "✓ Does this match my personality and communication style?\n"
    prompt += "✓ Am I using my accent and natural speech patterns?\n"
    prompt += "✓ Is this how someone with my background would actually respond?\n\n"
    prompt += "## Output Format\n\n"
    prompt += "Generate ONLY spoken dialogue without:\n"
    prompt += (
        "- Emotional tags, action descriptions, quotation marks, or meta-commentary\n"
    )
    prompt += "- Brackets, quotes, or markup\n\n"
    prompt += "## Sound Human\n\n"
    prompt += "- Use natural speech patterns including filler words (um, uh, well, like, you know) when appropriate\n"
    prompt += "- Don't be afraid of brief hesitations, self-corrections, or incomplete thoughts if that matches your personality\n"
    prompt += "- Real people don't speak in perfect grammatical sentences—neither should you\n\n"
    prompt += "## Voice-Natural Formatting (following are few examples on how to respond; use them as reference formats only)\n"
    prompt += "**Numbers:** 'fifty thousand' not '50,000'\n"
    prompt += "**Phone numbers:** 'eight nine seven one one five three six four' not '897115364'\n"
    prompt += "**Dates:** 'November fourteenth twenty twenty five' not '11/14/2025'\n"
    prompt += "**Currency:** 'twenty five dollars and fifty cents' not '$25.50'\n"
    prompt += "**Time:** 'three thirty PM' not '3:30 PM'\n"
    prompt += "**Punctuation spacing:** Always add a space after punctuation before the next word (e.g., 'Thank you. I…' or 'Thank you.. I…', not 'Thank you.I…' or 'Thank you..I…').\n\n"
    prompt += "## Embody Your Situation\n\n"
    prompt += "- Let the situation guide your behavior, not your narration\n"
    prompt += "- Only mention situational details if they naturally come up\n\n"
    prompt += "Be natural and conversational.\n"
    return (prompt + _closing_anchor(objective)) if anchor else prompt


# The template the caller prompt is rendered from. The harness fills slots; it does not author
# prose. Mirrors the platform's own default, which pairs a persona block with the situation and
# then scrubs the situation slot because the persona block already carries it.
DEFAULT_SIMULATOR_TEMPLATE = (
    "You are a customer in a voice simulation. {{channel}} "
    "Stay consistent with the persona throughout the conversation.\n\n{{persona}}"
)

_SLOT = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


def render_simulator_prompt(
    template: str,
    persona: Persona,
    *,
    call_type: CallType,
    variables: Mapping[str, Any] | None = None,
    agent_name: str | None = None,
    additional_instructions: str | None = None,
    default_language: str | None = None,
    tts_provider: str | None = None,
) -> str:
    """Fill a caller-prompt template, the way the platform fills its own.

    ``{{persona}}`` becomes the formatted persona block, ``{{channel}}`` the call direction
    sentence, and every other ``{{slot}}`` is taken from ``variables``. ``{{situation}}`` is
    dropped rather than filled, because the persona block already states the situation and the
    platform removes it for the same reason.

    A template that cannot be rendered is returned to the caller unfilled rather than raising, so
    a bad template degrades the call instead of ending the run.
    """
    values = dict(variables or {})
    try:
        persona_text = format_voice_persona(
            persona, call_type=call_type, default_language=default_language
        )
    except Exception:
        logger.exception("persona_format_failed")
        persona_text = ""
    values.setdefault("persona", persona_text)
    values.setdefault("channel", _channel_sentence(call_type, agent_name))

    def fill(match: "re.Match[str]") -> str:
        name = match.group(1)
        if name == "situation":
            return ""
        if name in values:
            return str(values[name])
        logger.warning("simulator_prompt_slot_unfilled", extra={"slot": name})
        return ""

    try:
        prompt = _SLOT.sub(fill, template)
    except Exception:
        logger.exception("simulator_prompt_render_failed")
        return template
    # Tidy the hole a dropped situation slot leaves behind, as the platform does.
    prompt = re.sub(r"Currently,\s*[.]", "", prompt)
    prompt = re.sub(r"[ \t]{2,}", " ", prompt).strip()

    if tts_provider and tts_provider.strip().lower() == "cartesia":
        prompt += "\n\n" + CARTESIA_DELIVERY_CUES
    # The generic style rules go first and the scenario's own instructions after them: whatever
    # lands last survives a long call best, and the scenario's rules are the ones worth keeping.
    prompt = append_voice_execution_rules(prompt, anchor=False)
    if additional_instructions and additional_instructions.strip():
        prompt += (
            "\n\n# ADDITIONAL SIMULATOR INSTRUCTIONS\n\n"
            + additional_instructions.strip()
        )
    return prompt + _closing_anchor(
        persona.outcome or "", str(_persona_data(persona).get("name") or "")
    )


def _channel_sentence(call_type: CallType, agent_name: str | None) -> str:
    return (
        f"You will make a call to an agent named {agent_name}."
        if call_type == "inbound" and agent_name
        else "You will make a call to an agent."
        if call_type == "inbound"
        else f"You will receive a call from an agent named {agent_name}."
        if agent_name
        else "You will receive a call from an agent."
    )


def build_voice_simulator_prompt(
    persona: Persona,
    *,
    call_type: CallType,
    agent_name: str | None = None,
    additional_instructions: str | None = None,
    default_language: str | None = None,
    template: str | None = None,
    variables: Mapping[str, Any] | None = None,
    tts_provider: str | None = None,
) -> str:
    """The caller prompt for one simulated customer.

    Renders ``template`` when one is supplied, and the shipped default otherwise, so a run with no
    template configured still produces the prompt it always did.
    """
    return render_simulator_prompt(
        template or DEFAULT_SIMULATOR_TEMPLATE,
        persona,
        call_type=call_type,
        variables=variables,
        agent_name=agent_name,
        additional_instructions=additional_instructions,
        default_language=default_language,
        tts_provider=tts_provider,
    )


__all__ = [
    "CallType",
    "VOICE_COMMUNICATION_STYLE_GUIDES",
    "VOICE_PERSONALITY_GUIDES",
    "append_voice_execution_rules",
    "DEFAULT_SIMULATOR_TEMPLATE",
    "CARTESIA_DELIVERY_CUES",
    "build_voice_simulator_prompt",
    "render_simulator_prompt",
    "format_voice_persona",
]
