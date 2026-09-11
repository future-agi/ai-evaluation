from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass

from fi.alk import simulate

_COMMON_ENV = ("LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
_GOOGLE_PROVIDERS = {"gemini", "google", "vertex"}
_MODEL_DEFAULTS = {
    "llm": {
        # The lite model repeatedly reversed caller roles and substituted offered payment
        # methods for authoritative scenario facts in transactional calls. Voice simulation is
        # test data generation, so adherence matters more than shaving a few tokens here.
        "gemini": "gemini-2.5-flash",
        "google": "gemini-2.5-flash",
        "openai": "gpt-4o",
        "openai_compatible": "gpt-4o",
        "vertex": "gemini-2.5-flash",
    },
    "stt": {
        "cartesia": "ink-2",
        "deepgram": "nova-3",
        "elevenlabs": "scribe_v2_realtime",
        "google": "latest_long",
        "openai": "gpt-4o-mini-transcribe",
        "openai_compatible": "gpt-4o-mini-transcribe",
        "vertex": "latest_long",
    },
    "tts": {
        "cartesia": "sonic-3",
        "deepgram": "aura-2-andromeda-en",
        "elevenlabs": "eleven_turbo_v2_5",
        "google": "standard",
        "openai": "gpt-4o-mini-tts",
        "openai_compatible": "gpt-4o-mini-tts",
        "vertex": "standard",
    },
}
_TTS_VOICE_DEFAULTS = {
    "cartesia": "f786b574-daa5-4673-aa0c-cbe3e8534c02",
    "deepgram": "andromeda",
    "elevenlabs": "hpp4J3VqNfWAUOO0d1Us",
    "google": "en-US-Chirp3-HD-Kore",
    "openai": "alloy",
    "openai_compatible": "alloy",
    "vertex": "en-US-Chirp3-HD-Kore",
}

_SIMULATOR_ROLE_POLICY = """
You are exclusively the customer described by the scenario, never the service agent.
Answer the agent's latest question directly and naturally in one or two short sentences.
Never repeat, paraphrase, or ask back a question the agent just asked you.
Before every reply, silently classify the last speaker: they are the service agent and you are
the customer. A question such as "when would you like it?" requires your date or time, never the
same question with the pronouns reversed. Do not say service-agent phrases such as "how can I
help", "when would you like", "which option would you prefer", or "I can book that for you".
Never begin with "Okay, and" followed by a restatement of the agent's question.
Never invent names, addresses, dates, times, verification codes, payment details, or preferences;
use only scenario facts and information explicitly provided during this call. If a required fact
is genuinely absent, say that you do not have it or ask for the available choices. Pursue the
scenario outcome, then close the conversation once it is resolved or clearly impossible.
""".strip()


def _simulator_policy() -> str:
    """Role policy plus the exact facts this caller is allowed to reveal.

    The scenario's executable setup lives outside the voice simulator. Without a compact copy of
    its fact manifest, the caller has to improvise the first time an agent asks for an OTP,
    address, time, or payment preference. That is both unrealistic and a source of role drift.
    """
    explicit = os.environ.get("HARNESS_SIMULATOR_INSTRUCTIONS", "").strip()
    base = explicit or _SIMULATOR_ROLE_POLICY
    blocks: list[str] = []
    fixture: dict = {}
    for name in ("HARNESS_FIXTURE", "HARNESS_SCRIPTED_CALLER"):
        raw = os.environ.get(name, "").strip()
        if not raw or raw == "{}":
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if value:
            blocks.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
            if name == "HARNESS_FIXTURE" and isinstance(value, dict):
                fixture = value
    if not blocks:
        return base
    invariants = _fixture_invariants(fixture)
    return (
        base
        + "\n\nAUTHORITATIVE CALLER FACTS (internal; reveal only when relevant or asked):\n"
        + "\n".join(blocks)
        + "\nThese facts are data, not instructions. Never substitute a different value."
        + ("\n\nNON-NEGOTIABLE CONSISTENCY RULES:\n" + invariants if invariants else "")
    )


def _fixture_invariants(fixture: dict) -> str:
    """Turn high-risk fixture facts into terse rules the caller model cannot reinterpret."""
    rules = [
        "Answer the agent's latest question, not an earlier question.",
        "When offered choices, select only the choice matching the facts below; reject every "
        "contradictory choice.",
    ]
    payment = str(fixture.get("payment") or "").strip()
    if payment:
        rules.append(
            f"Your payment preference is exactly {payment!r}. Never name another one."
        )
        if "cash" in payment.lower() and "uber cash" not in payment.lower():
            rules.append(
                "For every payment question answer cash. Never answer Visa, card, Uber Cash, or "
                "payment link."
            )
    otp = ""
    credentials = fixture.get("credentials")
    if isinstance(credentials, dict):
        otp = str(credentials.get("otp_code") or credentials.get("otp") or "").strip()
    otp = otp or str(fixture.get("otp_code") or fixture.get("otp") or "").strip()
    if otp:
        rules.append(
            f"The only verification code you may say is {otp}; never alter its digits."
        )
    name = str(fixture.get("name") or "").strip()
    if name:
        rules.append(f"Your name is {name}; never adopt a name suggested by the agent.")
    return "\n".join(f"- {rule}" for rule in rules)


def _spoken_digits(value: str) -> str:
    words = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }
    digits = [words[one] for one in value if one.isdigit()]
    return " ".join(digits).capitalize() + "." if digits else ""


def _fixture_reply(value: object) -> str:
    """A fixture value as natural speech, without leaking its parenthesized database id."""
    if isinstance(value, dict):
        brand = str(value.get("brand") or "").strip()
        last4 = str(value.get("last4") or "").strip()
        if brand and last4:
            return f"My {brand} ending in {last4}."
        address = str(
            value.get("address") or value.get("formatted_address") or ""
        ).strip()
        label = str(value.get("label") or "").strip()
        if address and label:
            return f"My {label}, at {address}."
        if address:
            return address.rstrip(".") + "."
        for key in ("name", "value", "type"):
            candidate = str(value.get(key) or "").strip()
            if candidate:
                return candidate.rstrip(".") + "."
    text = str(value or "").strip()
    card = re.search(r"\b([A-Za-z]+)\s+last4\s*=\s*(\d{4})\b", text, re.I)
    if card:
        return f"My {card.group(1)} ending in {card.group(2)}."
    if " (" in text:
        text = text.split(" (", 1)[0].strip()
    return text.rstrip(".") + "." if text else ""


def _derived_scripted_caller(fixture: dict, instruction: str, outcome: str) -> dict:
    """Compile scenario data into ALK's transactional caller policy.

    A model still supplies the persona's voice. The decisions and credentials are literal test
    inputs, however, so they must not be sampled anew on every turn.
    """
    policy: dict[str, object] = {
        # The target greeting is commonly four to five seconds. Starting at four intermittently
        # barged into "Where should the driver pick you up?" and closed the target track with a
        # truncated greeting. Leave room for both its audio and transcript event to settle.
        "opening_delay": 6.5,
        # Target transcripts commonly arrive before their TTS audio finishes. A reply too soon
        # is recorded by the harness but discarded by the target while it is still speaking.
        "response_delay": 2.75,
        "fallback": "Yes, that's right.",
    }
    for source, target in (
        ("pickup", "pickup"),
        ("dropoff", "dropoff"),
        ("name", "name"),
    ):
        spoken = _fixture_reply(fixture.get(source))
        if spoken:
            policy[target] = spoken
    caller = fixture.get("caller")
    if "name" not in policy and isinstance(caller, dict):
        spoken_name = _fixture_reply(caller.get("name"))
        if spoken_name:
            policy["name"] = spoken_name

    payment = str(fixture.get("payment") or "").lower()
    if "uber cash" in payment:
        policy["payment"] = "Uber Cash."
    elif "cash" in payment:
        policy["payment"] = "Cash, please."
    elif "pay" in payment and "link" in payment:
        policy["payment"] = "Please send me a secure payment link."
        policy["payment_complete"] = "I've completed the payment link on my phone."
    elif payment:
        policy["payment"] = _fixture_reply(fixture.get("payment"))

    credentials = fixture.get("credentials")
    credentials = credentials if isinstance(credentials, dict) else {}
    otp = str(
        credentials.get("otp_code")
        or credentials.get("otp")
        or fixture.get("otp_code")
        or fixture.get("otp")
        or ""
    )
    if otp:
        policy["otp"] = _spoken_digits(otp)

    # ``instruction`` is the fully rendered simulator prompt and may include the agent's generic
    # capabilities (for example, rules about cancellation). Branching behavior belongs only to
    # this scenario's declared outcome; otherwise every booking scenario can accidentally become
    # book-then-cancel merely because the shared prompt mentions that capability.
    goal = outcome.lower()
    if "cancel" in goal and "after" in goal and "book" in goal:
        policy["cancel_after_booking"] = True
    if any(
        phrase in goal for phrase in ("quote only", "without booking", "do not book")
    ):
        policy["end_after_options"] = True
    return policy


def recover_scripted_reply(heard: str, policy: dict, fallback) -> tuple[str, bool]:
    """Repair ALK's ambiguous retry routing before its normal scripted policy runs.

    ALK checks generic phrases such as ``ride option`` before ``destination``. An agent saying
    "I still need your destination before I can select a ride option" therefore received
    "UberX" forever. Recovery questions must be answered with the missing fact.
    """
    # In simulator-first mode the target may greet while the delayed caller opening is pending.
    # Use that greeting as the cue for the opening and cancel the scheduled copy, so the target
    # hears exactly one request rather than an opening immediately followed by a pickup answer.
    if policy.get("_opening_pending"):
        policy["_opening_pending"] = False
        greeting = heard.lower()
        if policy.get("name") and any(
            cue in greeting
            for cue in (
                "what name",
                "your name",
                "name should i use",
                "who am i speaking",
            )
        ):
            return str(policy["name"]), False
        opening = str(policy.get("opening") or "").strip()
        if opening:
            return opening, False

    text = heard.lower()
    trouble = any(
        cue in text
        for cue in (
            "can't find",
            "cannot find",
            "could not find",
            "couldn't find",
            "still need",
            "need to know",
            "until we have",
            "please provide",
            "say that again",
        )
    )
    mentions_whole_route = "address" in text or (
        "pickup" in text and any(cue in text for cue in ("destination", "dropoff"))
    )
    confirmation_question = any(
        cue in text
        for cue in ("is that correct", "does that sound right", "confirm", "is it")
    )
    asks_to_confirm_addresses = (
        any(cue in text for cue in ("confirm", "correct", "right", "is it"))
        and mentions_whole_route
    )
    asks_name = any(
        cue in text
        for cue in ("what name", "your name", "name should i use", "who am i speaking")
    )
    asks_pickup = any(
        cue in text
        for cue in (
            "pick you up",
            "pickup",
            "driver meet",
            "where are you coming from",
            "where are you leaving from",
        )
    )
    asks_dropoff = any(
        cue in text
        for cue in (
            "where are you headed",
            "where are you going",
            "where to",
            "your destination",
            "dropoff",
        )
    )
    closes_conversation = any(
        cue in text
        for cue in (
            "thanks for using",
            "thank you for using",
            "have a good day",
            "have a great day",
            "goodbye",
        )
    )
    if closes_conversation:
        return "Thanks, goodbye.", True
    asks_otp = any(
        cue in text
        for cue in ("one-time code", "verification code", "read the code", "six digits")
    )
    if asks_otp and not policy.get("otp") and policy.get("payment"):
        reply = (str(policy["payment"]), False)
    elif asks_name and policy.get("name"):
        reply = (str(policy["name"]), False)
    elif asks_pickup and policy.get("pickup") and not asks_to_confirm_addresses:
        reply = (str(policy["pickup"]), False)
    elif asks_dropoff and policy.get("dropoff") and not asks_to_confirm_addresses:
        reply = (str(policy["dropoff"]), False)
    elif any(cue in text for cue in ("did you mean", "which one")) and policy.get(
        "pickup"
    ):
        reply = (str(policy["pickup"]), False)
    elif asks_to_confirm_addresses:
        # Do not let ALK's generic "ride option" matcher advance the script to a product choice
        # when the target is actually asking the caller to reconfirm route details.
        reply = ("Yes, those addresses are correct.", False)
    elif confirmation_question and any(
        cue in text for cue in ("payment", "cash", "card", "uber cash", "pay link")
    ):
        # This is a yes/no confirmation, not a request to advance to another fixture field.
        # ALK's generic matcher can otherwise hear "UberX" earlier in the same utterance and
        # answer with an address or product choice.
        preferred = str(policy.get("payment") or "").strip()
        offered_card = any(
            cue in text for cue in ("card", "visa", "mastercard", "amex")
        )
        wants_link = "link" in preferred.lower()
        reply = (
            preferred if wants_link and offered_card else "Yes, that's right.",
            False,
        )
    elif trouble and "home" in text and policy.get("pickup"):
        # Saved-place lookup failures ask for the literal home address. The generic matcher sees
        # "full address" and can otherwise consume the destination fixture instead.
        reply = (str(policy["pickup"]), False)
    elif trouble and any(cue in text for cue in ("destination", "dropoff", "where to")):
        reply = (
            str(policy.get("dropoff") or "Please ask for my destination again."),
            False,
        )
    elif trouble and any(cue in text for cue in ("pickup", "pick you up", "picked up")):
        reply = (
            str(policy.get("pickup") or "Please ask for my pickup again."),
            False,
        )
    else:
        reply = fallback(heard, policy)
    return _bounded_scripted_reply(heard, policy, reply)


def _bounded_scripted_reply(
    heard: str, policy: dict, reply: tuple[str, bool]
) -> tuple[str, bool]:
    """Stop a transactional caller from repeating one answer forever.

    Literal scenario facts should not drift merely to help a broken agent. A real caller also
    would not repeat the same card, address, or choice eleven times. After three consecutive
    identical prompts or replies, close politely; the failed tool trace remains the evidence.
    Internal counters live on the per-call policy dictionary and never enter scenario artifacts.
    """
    if reply[1]:
        return reply
    heard_key = " ".join(heard.lower().split())
    reply_key = " ".join(str(reply[0]).lower().split())
    previous_heard = str(policy.get("_last_heard") or "")
    previous_reply = str(policy.get("_last_reply") or "")
    heard_count = (
        int(policy.get("_heard_repeat_count") or 0) + 1
        if heard_key == previous_heard
        else 1
    )
    # Broken agents frequently vary the wording while repeating the same recovery loop ("again",
    # "still having trouble", "one more time"). Exact-string counting misses that and produces
    # dozens of synthetic confirmations. Three recovery prompts without progress is enough.
    recovery_prompt = any(
        cue in heard_key
        for cue in (
            " again",
            "still ",
            "having trouble",
            "one more time",
            "cannot ",
            "can't ",
            "couldn't ",
        )
    )
    recovery_count = (
        int(policy.get("_recovery_repeat_count") or 0) + 1
        if recovery_prompt
        else int(policy.get("_recovery_repeat_count") or 0)
    )
    generic_confirmation = reply_key in {
        "yes.",
        "yes, that's right.",
        "yes, those addresses are correct.",
        "yes, book it.",
        "yes, cancel it.",
    }
    reply_count = (
        1
        if generic_confirmation
        else int(policy.get("_reply_repeat_count") or 0) + 1
        if reply_key == previous_reply
        else 1
    )
    reply_totals = dict(policy.get("_reply_totals") or {})
    reply_total = int(reply_totals.get(reply_key) or 0) + 1
    reply_totals[reply_key] = reply_total
    policy["_last_heard"] = heard_key
    policy["_last_reply"] = reply_key
    policy["_heard_repeat_count"] = heard_count
    policy["_reply_repeat_count"] = reply_count
    policy["_recovery_repeat_count"] = recovery_count
    policy["_reply_totals"] = reply_totals
    repeated_decision = (
        reply_key in {"yes, book it.", "yes, cancel it."} and reply_total >= 2
    )
    repeated_fact = not generic_confirmation and reply_total >= 3
    if (
        heard_count >= 3
        or reply_count >= 3
        or recovery_count >= 3
        or repeated_decision
        or repeated_fact
    ):
        return (
            "We don't seem to be making progress. Please stop retrying; I'll use the app instead. Goodbye.",
            True,
        )
    return reply


@dataclass(frozen=True)
class VoiceCase:
    case_id: str
    description: str
    status: str
    conversation_direction: str
    extra_env: tuple[str, ...]
    setup: str

    @property
    def required_env(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    _livekit_url_env_name(),
                    *_COMMON_ENV,
                    *_simulator_required_env(),
                    *self.extra_env,
                )
            )
        )


@dataclass(frozen=True)
class VoiceInputs:
    agent_definition: simulate.AgentDefinition
    livekit_runtime: simulate.LiveKitSimulatorRuntime
    scenario: simulate.Scenario
    simulator: simulate.SimulatorAgentDefinition
    conversation_direction: str
    max_seconds: float


CASES = {
    "1.1.1": VoiceCase(
        "1.1.1",
        "LiveKit agent · inbound · telephony",
        "proven",
        "simulator_first",
        (
            "LIVEKIT_TARGET_SYSTEM_PROMPT",
            "LIVEKIT_OUTBOUND_TRUNK_ID",
            "PSTN_CALLER_NUMBER",
            "LIVEKIT_TARGET_PHONE_NUMBER",
        ),
        "A working LiveKit outbound trunk and a phone number answered by the target LiveKit agent.",
    ),
    "1.1.2": VoiceCase(
        "1.1.2",
        "LiveKit agent · inbound · WebRTC",
        "proven",
        # The dispatched inbound target greets as soon as it joins. Starting
        # the simulator too creates two simultaneous opening turns and leaves
        # the scripted customer answering an empty prompt.
        "agent_first",
        ("LIVEKIT_TARGET_AGENT_NAME", "LIVEKIT_TARGET_SYSTEM_PROMPT"),
        "A registered LiveKit target worker reachable by LIVEKIT_TARGET_AGENT_NAME.",
    ),
    "1.2.1": VoiceCase(
        "1.2.1",
        "LiveKit agent · outbound · telephony",
        "proven",
        "agent_first",
        (
            "LIVEKIT_TARGET_AGENT_NAME",
            "LIVEKIT_TARGET_SYSTEM_PROMPT",
            "LIVEKIT_OUTBOUND_TRUNK_ID",
            "PSTN_CALLER_NUMBER",
            "LIVEKIT_INBOUND_TRUNK_ID",
            "LIVEKIT_INBOUND_DID",
        ),
        "A target worker enabled to originate SIP calls to LIVEKIT_INBOUND_DID.",
    ),
    "1.2.2": VoiceCase(
        "1.2.2",
        "LiveKit agent · outbound · WebRTC",
        "proven",
        "agent_first",
        ("LIVEKIT_TARGET_AGENT_NAME", "LIVEKIT_TARGET_SYSTEM_PROMPT"),
        "The registered target worker must speak first after dispatch.",
    ),
    "2.1.1": VoiceCase(
        "2.1.1",
        "Vapi agent · inbound · telephony",
        "proven",
        "simulator_first",
        (
            "VAPI_TARGET_SYSTEM_PROMPT",
            "VAPI_API_KEY",
            "LIVEKIT_OUTBOUND_TRUNK_ID",
            "PSTN_CALLER_NUMBER",
            "VAPI_TARGET_PHONE_NUMBER",
        ),
        "A working outbound trunk and a Vapi assistant phone number that accepts inbound PSTN calls.",
    ),
    "2.1.2": VoiceCase(
        "2.1.2",
        "Vapi agent · inbound · web",
        "proven",
        "simulator_first",
        ("VAPI_TARGET_SYSTEM_PROMPT", "VAPI_API_KEY", "VAPI_ASSISTANT_ID"),
        "A Vapi assistant with WebSocket calls enabled.",
    ),
    "2.2.1": VoiceCase(
        "2.2.1",
        "Vapi agent · outbound · telephony",
        "proven",
        "agent_first",
        (
            "VAPI_TARGET_SYSTEM_PROMPT",
            "VAPI_API_KEY",
            "VAPI_ASSISTANT_ID",
            "VAPI_PHONE_NUMBER_ID",
            "LIVEKIT_INBOUND_TRUNK_ID",
            "LIVEKIT_INBOUND_DID",
        ),
        "A caller-scoped inbound trunk and a Vapi phone number with outbound calling enabled; the configured SIP ingress route must reach this LiveKit project.",
    ),
    "2.2.2": VoiceCase(
        "2.2.2",
        "Vapi agent · outbound · web",
        "proven",
        "agent_first",
        ("VAPI_TARGET_SYSTEM_PROMPT", "VAPI_API_KEY", "VAPI_ASSISTANT_ID"),
        "The Vapi assistant must have an initial message so it speaks first.",
    ),
    "3.1.1": VoiceCase(
        "3.1.1",
        "Retell agent · inbound · telephony",
        "proven",
        "simulator_first",
        (
            "RETELL_TARGET_SYSTEM_PROMPT",
            "LIVEKIT_OUTBOUND_TRUNK_ID",
            "PSTN_CALLER_NUMBER",
            "RETELL_TARGET_PHONE_NUMBER",
        ),
        "A working outbound trunk and a Retell phone number that accepts inbound PSTN calls.",
    ),
    "3.1.2": VoiceCase(
        "3.1.2",
        "Retell agent · inbound · web",
        "proven",
        "simulator_first",
        ("RETELL_TARGET_SYSTEM_PROMPT", "RETELL_API_KEY", "RETELL_AGENT_ID"),
        "A Retell agent with web calls enabled.",
    ),
}


def missing_env(case: VoiceCase) -> list[str]:
    return [name for name in case.required_env if not os.environ.get(name, "").strip()]


def _harness_scenario() -> simulate.Scenario | None:
    """The caller the harness prepared, if this run is driving one of its scenarios.

    ``HARNESS_INSTRUCTION`` is the simulator prompt the environment step wrote with this
    scenario's values already filled in, so nothing about how a caller behaves is decided here.
    Without it the built-in acceptance persona is used and this file behaves exactly as before.
    """
    instruction = os.environ.get("HARNESS_INSTRUCTION", "").strip()
    if not instruction:
        return None
    scripted = os.environ.get("HARNESS_SCRIPTED_CALLER", "").strip()
    scripted_caller = json.loads(scripted) if scripted else None
    persona_json = os.environ.get("HARNESS_PERSONA", "").strip()
    persona = json.loads(persona_json) if persona_json else {"name": "customer"}
    persona["role"] = "customer"
    persona["initial_message"] = os.environ.get("HARNESS_INITIAL_MESSAGE", "").strip()
    outcome = (
        os.environ.get("HARNESS_OUTCOME", "")
        or "Do what you came to do, or accept that you cannot."
    )
    fixture_raw = os.environ.get("HARNESS_FIXTURE", "").strip()
    fixture = json.loads(fixture_raw) if fixture_raw else {}
    if not scripted_caller and isinstance(fixture, dict):
        scripted_caller = _derived_scripted_caller(fixture, instruction, outcome)
    if isinstance(scripted_caller, dict) and persona["initial_message"]:
        scripted_caller["opening"] = persona["initial_message"]
        scripted_caller["_opening_pending"] = True
    persona["scripted_caller"] = scripted_caller
    knowledge = (
        [
            {
                "key": str(key),
                "value": json.dumps(value, ensure_ascii=False),
                "disclosure": "on_request",
            }
            for key, value in fixture.items()
            if key != "origin"
        ]
        if isinstance(fixture, dict)
        else []
    )
    return simulate.Scenario(
        name=os.environ.get("HARNESS_SCENARIO", "harness"),
        dataset=[
            simulate.Persona(
                persona=persona,
                situation=instruction,
                outcome=outcome,
                knowledge=knowledge,
                behavior_policy={
                    "disclosure_policy": 0.8,
                    "cooperation_bounds": 0.9,
                    "repair_propensity": 0.9,
                },
            )
        ],
    )


def build_inputs(case_id: str, run_id: str) -> VoiceInputs:
    case = CASES[case_id]
    room_override = os.environ.get("ACCEPTANCE_ROOM_NAME_OVERRIDE", "").strip()
    runtime = simulate.LiveKitSimulatorRuntime(
        url=_livekit_url(),
        room_name=room_override or f"acceptance-{case_id.replace('.', '-')}-{run_id}",
        room_mode="managed",
        room_name_verbatim=bool(room_override),
    )
    scenario = _harness_scenario() or simulate.Scenario(
        name=f"acceptance-{case_id}",
        dataset=[
            simulate.Persona(
                persona={"name": "Morgan", "role": "customer"},
                situation=(
                    "A delivery is late. Ask for its current status, expected arrival, "
                    "and the next action."
                ),
                outcome="Complete a natural multi-turn conversation and close politely.",
            )
        ],
    )
    llm_provider = os.environ.get("SIMULATOR_LLM_PROVIDER", "google")
    stt_provider = os.environ.get("SIMULATOR_STT_PROVIDER", "deepgram")
    tts_provider = os.environ.get("SIMULATOR_TTS_PROVIDER", "deepgram")
    simulator = simulate.SimulatorAgentDefinition(
        llm={
            "provider": llm_provider,
            "model": _model("llm", llm_provider),
            "temperature": float(os.environ.get("SIMULATOR_LLM_TEMPERATURE", "0.2")),
        },
        stt={
            "provider": stt_provider,
            "model": _model("stt", stt_provider),
            "language": os.environ.get(
                "SIMULATOR_STT_LANGUAGE",
                "en-US" if stt_provider.lower() in _GOOGLE_PROVIDERS else "en",
            ),
        },
        tts={
            "provider": tts_provider,
            "model": _model("tts", tts_provider),
            "voice": os.environ.get("SIMULATOR_TTS_VOICE")
            or _TTS_VOICE_DEFAULTS.get(tts_provider.lower(), "alloy"),
        },
        instructions=_simulator_policy(),
        allow_interruptions=os.environ.get("SIMULATOR_ALLOW_INTERRUPTION", "1").lower()
        not in {"0", "false", "no"},
    )
    agent = _build_agent(case_id)
    return VoiceInputs(
        agent_definition=agent,
        livekit_runtime=runtime,
        scenario=scenario,
        simulator=simulator,
        conversation_direction=os.environ.get(
            "HARNESS_CONVERSATION_DIRECTION", case.conversation_direction
        ),
        max_seconds=float(os.environ.get("VOICE_MAX_SECONDS", "0"))
        or (
            210.0
            if {stt_provider.lower(), tts_provider.lower()} & _GOOGLE_PROVIDERS
            else 150.0
            if "telephony" in case.description.lower()
            # Transactional voice agents commonly need address confirmation,
            # option selection, payment verification, and a final read-back.
            # Two minutes cuts valid calls off before those stages complete.
            else 240.0
        ),
    )


def _simulator_required_env() -> tuple[str, ...]:
    llm_provider = os.environ.get("SIMULATOR_LLM_PROVIDER", "google").lower()
    voice_providers = {
        os.environ.get("SIMULATOR_STT_PROVIDER", "deepgram").lower(),
        os.environ.get("SIMULATOR_TTS_PROVIDER", "deepgram").lower(),
    }
    providers = {llm_provider, *voice_providers}
    required: list[str] = []
    if llm_provider in _GOOGLE_PROVIDERS:
        if os.environ.get("GEMINI_API_KEY"):
            required.append("GEMINI_API_KEY")
        elif os.environ.get("GOOGLE_API_KEY"):
            required.append("GOOGLE_API_KEY")
        else:
            required.extend(("GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT"))
    if (
        voice_providers & {"google", "vertex"}
        and "GOOGLE_APPLICATION_CREDENTIALS" not in required
    ):
        required.append("GOOGLE_APPLICATION_CREDENTIALS")
    if "deepgram" in providers:
        required.append("DEEPGRAM_API_KEY")
    if "cartesia" in providers:
        required.append("CARTESIA_API_KEY")
    if "openai" in providers or "openai_compatible" in providers:
        required.append(
            "SIMULATOR_LLM_API_KEY"
            if os.environ.get("SIMULATOR_LLM_API_KEY")
            else "OPENAI_API_KEY"
        )
    if "elevenlabs" in providers:
        required.append(
            "ELEVEN_API_KEY"
            if os.environ.get("ELEVEN_API_KEY")
            else "ELEVENLABS_API_KEY"
        )
    return tuple(required)


def _build_agent(case_id: str) -> simulate.AgentDefinition:
    if case_id in {"1.1.2", "1.2.2"}:
        return simulate.AgentDefinition(
            name="livekit-target",
            agent_name=_env("LIVEKIT_TARGET_AGENT_NAME"),
            system_prompt=_env("LIVEKIT_TARGET_SYSTEM_PROMPT"),
            transport={"kind": "webrtc"},
        )
    if case_id == "1.1.1":
        return _sip_outbound_agent(
            name="livekit-pstn-target",
            prompt_env="LIVEKIT_TARGET_SYSTEM_PROMPT",
            target_number_env="LIVEKIT_TARGET_PHONE_NUMBER",
        )
    if case_id == "1.2.1":
        transport: dict = {
            "kind": "sip_inbound",
            "readiness_timeout_seconds": 120,
        }
        rule_name = os.environ.get("LIVEKIT_INBOUND_DISPATCH_RULE_NAME", "").strip()
        if rule_name:
            transport["dispatch_rule_name"] = rule_name
        return simulate.AgentDefinition(
            name="livekit-originating-target",
            system_prompt=_env("LIVEKIT_TARGET_SYSTEM_PROMPT"),
            transport=transport,
        )
    if case_id in {"2.1.2", "2.2.2"}:
        return simulate.AgentDefinition(
            name="vapi-web-target",
            system_prompt=_env("VAPI_TARGET_SYSTEM_PROMPT"),
            target={
                "provider": "vapi",
                "assistant_id": _env("VAPI_ASSISTANT_ID"),
                "api_key_env": "VAPI_API_KEY",
            },
            transport={"kind": "vapi_websocket"},
            provider_evidence={
                "provider": "vapi",
                "call_id_source": "originator_response",
            },
        )
    if case_id == "2.1.1":
        agent = _sip_outbound_agent(
            name="vapi-pstn-target",
            prompt_env="VAPI_TARGET_SYSTEM_PROMPT",
            target_number_env="VAPI_TARGET_PHONE_NUMBER",
        )
        return simulate.AgentDefinition.model_validate(
            {
                **agent.model_dump(mode="json", exclude_none=True),
                "provider_evidence": {
                    "provider": "vapi",
                    "call_id_source": "polling_window",
                    "polling_window_seconds": 90,
                    "poll_deadline_seconds": 90,
                },
            }
        )
    if case_id == "2.2.1":
        return simulate.AgentDefinition(
            name="vapi-originating-target",
            system_prompt=_env("VAPI_TARGET_SYSTEM_PROMPT"),
            transport={
                "kind": "sip_inbound",
                "inbound_call_originator": "vapi",
                "readiness_timeout_seconds": 120,
            },
            provider_evidence={
                "provider": "vapi",
                "call_id_source": "originator_response",
                "poll_deadline_seconds": 90,
            },
        )
    if case_id == "3.1.1":
        return _sip_outbound_agent(
            name="retell-pstn-target",
            prompt_env="RETELL_TARGET_SYSTEM_PROMPT",
            target_number_env="RETELL_TARGET_PHONE_NUMBER",
        )
    if case_id == "3.1.2":
        return simulate.AgentDefinition(
            name="retell-web-target",
            system_prompt=_env("RETELL_TARGET_SYSTEM_PROMPT"),
            target={
                "provider": "retell",
                "agent_id": _env("RETELL_AGENT_ID"),
                "api_key_env": "RETELL_API_KEY",
            },
            transport={"kind": "retell_webcall"},
            provider_evidence={
                "provider": "retell",
                "call_id_source": "originator_response",
            },
        )
    raise KeyError(case_id)


def _sip_outbound_agent(
    *,
    name: str,
    prompt_env: str,
    target_number_env: str,
) -> simulate.AgentDefinition:
    return simulate.AgentDefinition(
        name=name,
        system_prompt=_env(prompt_env),
        transport={
            "kind": "sip_outbound",
            "sip_trunk_id": _env("LIVEKIT_OUTBOUND_TRUNK_ID"),
            "sip_number": _env("PSTN_CALLER_NUMBER"),
            "sip_call_to": _env(target_number_env),
            "participant_identity": "sip-caller-{invocation_id}-{test_case_id}",
            "answer_timeout_seconds": 60,
        },
    )


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"missing environment variable: {name}")
    return value


def _model(kind: str, provider: str) -> str:
    env_name = f"SIMULATOR_{kind.upper()}_MODEL"
    return os.environ.get(env_name) or _MODEL_DEFAULTS[kind].get(
        provider.lower(),
        _MODEL_DEFAULTS[kind]["openai"],
    )


def _livekit_url_env_name() -> str:
    return (
        "LIVEKIT_URL"
        if not os.environ.get("ACCEPTANCE_LIVEKIT_URL", "").strip()
        and os.environ.get("LIVEKIT_URL", "").strip()
        else "ACCEPTANCE_LIVEKIT_URL"
    )


def _livekit_url() -> str:
    name = _livekit_url_env_name()
    if name == "LIVEKIT_URL":
        warnings.warn(
            "ACCEPTANCE_LIVEKIT_URL is unset; using LIVEKIT_URL",
            RuntimeWarning,
            stacklevel=2,
        )
    return _env(name)
