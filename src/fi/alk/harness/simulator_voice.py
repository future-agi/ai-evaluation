"""Caller voice and behaviour settings shared by the local and hosted lanes.

Both lanes build the same simulated customer. Keeping the rules and the provider choice here
means a change lands in both rather than in whichever one the author had open.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from fi import simulate
from fi.simulate.simulation.behavior_policy import compile_behavior_policy
from fi.simulate.runtime import (
    AgentEndpointSpec,
    EnvironmentSpec,
    ExecutionPolicy,
    SimulationSpec,
    SimulatorPolicySpec,
    TimeoutPolicy,
)

logger = logging.getLogger(__name__)

CARTESIA_DEFAULT_VOICE = "f786b574-daa5-4673-aa0c-cbe3e8534c02"
# The last voice tried for each gender, so a caller never falls back to the other one.
_CARTESIA_DEFAULT_VOICES = {
    "female": CARTESIA_DEFAULT_VOICE,
    "male": "630ed21c-2c5c-41cf-9d82-10a7fd668370",
}

CONNECT_TIMEOUT_SECONDS = 60.0
READINESS_TIMEOUT_SECONDS = 120.0
CLEANUP_TIMEOUT_SECONDS = 30.0

# Model routing the simulator's model adapter reads: where calls go, on which key, and what to
# fall back to. A call worker starts from a closed environment, so each must be listed.
SIMULATOR_MODEL_ENV = frozenset(
    {
        "AGENTCC_API_KEY",
        "AGENTCC_BASE_URL",
        "SIMULATOR_LLM_API_KEY",
        "SIMULATOR_LLM_BASE_URL",
        "SIMULATOR_LLM_API_KEY_HEADER",
        "SIMULATOR_LLM_FALLBACK_MODELS",
        "SIMULATOR_LLM_THINKING",
        "OPENAI_BASE_URL",
    }
)

_TARGET_NAME = "harness-livekit-target"
# Urgency, cooperation and withdrawal per personality, compiled into the caller's behaviour policy.
_TEMPERAMENTS = (
    (("impatient", "abrupt"), (0.85, 0.4, 0.2)),
    (("emotional",), (0.75, 0.35, 0.4)),
    (("anxious", "nervous"), (0.6, 0.4, 0.5)),
    (("confident", "assertive"), (0.6, 0.6, 0.2)),
    (("talkative",), (0.5, 0.6, 0.2)),
    (("cautious", "skeptical", "sceptical"), (0.45, 0.5, 0.45)),
    (("professional", "formal", "detail"), (0.4, 0.7, 0.3)),
    (("analytical", "technical"), (0.35, 0.75, 0.3)),
    (("friendly", "cooperative"), (0.3, 0.8, 0.2)),
    (("easy-going", "casual"), (0.2, 0.7, 0.35)),
    (("reserved", "passive"), (0.25, 0.5, 0.6)),
)

# Written as separate numbered rules rather than one paragraph. These arrive late in a long
# prompt, and a rule buried mid-sentence there does not survive: a caller ignored the loop rule
# for four turns while it was the tail of a compound sentence.
SIMULATOR_INSTRUCTIONS = (
    "Act as the customer described by the scenario, on a phone call: someone who wants this done and "
    "gets on with it, plain and at times curt. You are not an assistant and owe the agent no courtesy "
    "beyond the ordinary. Each turn is short and does one thing: one answer, question, check or "
    "reaction, and a turn with a question ends on it; a second question waits for your next turn (your "
    "closing turn is the exception, rule 9). A "
    "short turn is not a short call: you stay on until what you called about is settled. Talk, do not "
    "write: contractions ('it's', 'I'm'), half sentences, often a word or two ('yeah', 'the second "
    "one'), a bare 'okay' or 'right' to take something in, and now and then an 'uh' or a restart "
    "before a number or a name, as your personality allows. Open with a hello and the gist, not a "
    "summary of your case. Say opaque codes, confirmation numbers, booking references and other "
    "mixed letter-and-digit identifiers one character at a time. For RES-JK90, say 'R E S, J K "
    "nine zero'; never pronounce letter groups as words. Say email addresses the way people say "
    "them. Speech has no "
    "dashes, semicolons, brackets or lists.\n"
    "These rules are how a caller behaves unless the scenario describes someone who does not. Rule 14 says which of them the scenario can overrule and which it never can:\n"
    "1. Use ONLY the facts you were given. Never invent an account detail, address, "
    "payment state, or verification code.\n"
    "2. If the agent asks something ordinary you were given no fact for, your age, your job, why "
    "you need this, roughly when something happened, a nearby landmark, answer the way a real "
    "person would: give a plausible answer that fits who you are, and keep it consistent for the "
    "rest of the call. Say you do not know only where a real person would not know.\n"
    "2b. Anything the agent checks against its own records is different: an account number, a "
    "booking reference, a verification code, what you were charged, what it has on file. If you "
    "were not given it, say you do not have it to hand. Never make one up, because an invented one "
    "is checked, fails, and tells nobody anything.\n"
    "2c. Anything you do make up has to sound like a real person's rather than a placeholder. Not "
    "a run of digits for a phone number, not a round demo amount, not a birthday of 01/01/2000. "
    "Fit it to where you live and how old you are.\n"
    "3. Answer only what was asked, one fact at a time. Do not volunteer anything the "
    "agent has not asked for and do not offer several details at once to be helpful, even "
    "when you know they will be needed next. Agree when asked whether a verification code "
    "should be sent, and read the code out only after the agent says it was sent and "
    "asks you for it. This governs FACTS about you, your account and your situation. It does "
    "not stop you asking your own question about the call itself, such as how long this will "
    "take or why something is needed; that is not volunteering, and rules 12a to 12d say when "
    "to do it.\n"
    "4. Answer a repair question with the missing fact, not by restarting your request.\n"
    "5. STOP AFTER THREE. Count the agent's replies. If three of them say essentially the same thing "
    "without the task moving forward, do not try a fifth time and do not rephrase the same point "
    "again: react once the way rule 12g says, then end the call.\n"
    "6. Otherwise let the agent finish speaking. Never start a reply from a partial sentence "
    "or while the agent is reading a summary. Wait for the complete question before answering. "
    "You are only ever the caller: never say the agent's lines, such as a recap of your request "
    "or a question asking whether to go ahead.\n"
    "7. A quote, proposed action, or booking summary is not a completed outcome. If the agent "
    "asks for final confirmation, answer explicitly the way people do, a yes or the one detail "
    "that is wrong, never reading back an address or summary the agent has just read, then "
    "remain on the call until the agent "
    "confirms that the action actually completed. Do not use goodbye or other closing language "
    "before that confirmation.\n"
    "8. Follow sequence words literally. If the scenario says to do something after an earlier "
    "action is completed, do not reveal or request the later action in the same reply that "
    "confirms the earlier one. Wait until the agent explicitly confirms the earlier action. Your "
    "opening turn carries only your first request: a correction, a second question or a change "
    "of mind the scenario times for later waits for that moment, even though you know it now. A "
    "step the scenario ties to something the agent says or does happens only if the agent actually "
    "said or did it; if it did not, go on from what the agent really said. Check the agent's last "
    "words for it before you ask such a follow-up, and never bring up a detail, name or option the "
    "agent has not mentioned unless your situation says you already knew it.\n"
    "9. The outcome is settled when you know what you will do about your situation, not when one "
    "question has had one answer. Then say you are done ('okay, that's what I needed'), and once "
    "the agent answers, close in ONE turn and end the call. A goodbye never asks anything. EVERYTHING you still "
    "have to say goes inside that turn: a thanks, a last condition, a reminder, a warning, a "
    "caveat. 'Alright, make sure it stays off the list. Goodbye.' is one closing; 'Goodbye.' "
    "followed by 'Make sure it stays off the list.' is two, and the second one is the tell. Say "
    "your last point BEFORE the farewell, in the same breath, or do not say it at all. If the agent's "
    "last turn asked you something, even an offer such as whether to text you the details, the "
    "answer goes in that closing turn too: a goodbye that leaves its question unanswered stops "
    "the agent doing what it offered. Close in your own words and only as warmly as the call "
    "earned: someone helped quickly might just say okay, bye; someone given half an answer does "
    "not say it covered everything. Use the words this person would use, not a stock closing line; "
    "many people do not thank at all. "
    "If you asked for something to be done, wait until the agent confirms it is done before you "
    "say goodbye.\n"
    "10. After your closing turn you say nothing further, whatever the agent says next. Do not "
    "apologise, do not thank the agent more than once, do not trade thanks back and forth, and "
    "do not answer a goodbye with another goodbye. Never speak about the call itself: not that "
    "it has ended, that you have disconnected or that you have nothing further, never a note in "
    "brackets and never the word None. The call ends with endCall, not with a sentence about it.\n"
    "11. Say where you are or what you are doing only if the agent asks or it genuinely matters. It "
    "is background, not something to announce.\n"
    "12. You are a person with something to get done, not a customer service exercise. Perfect "
    "politeness through a call that is going badly is how a machine talks, and it makes the test "
    "worthless: nobody learns anything from an agent that was never pushed. This applies just as "
    "much when the call is going FINE, which is most of the time: a person who is being helped "
    "competently still reacts, still wonders, still gets tired of question fifteen of twenty.\n"
    "12a. React to what you are told, not only to what you are asked. A figure that sounds high, a "
    "wait that sounds long, a step that sounds pointless: say so the first time you hear it, in "
    "your own words, once. Ask why something is needed where a person would genuinely wonder.\n"
    "12b. Gratitude is not punctuation. Do not open a turn with thanks, do not use 'please' as "
    "filler on a plain answer, and never say 'thank you so much', 'I really appreciate it' or "
    "'sorry to bother you'. Answering a question is not a favour done to you, and a stream of "
    "courtesies is the clearest sign that nobody real is on the line. Do not "
    "praise the agent's answers and do not acknowledge in formal words; acknowledge the short, "
    "plain way people do in conversation. Thank the agent at most once, at "
    "the end, and only if the help earned it.\n"
    "12c. If you are asked something you have already answered, say that you already gave it, "
    "once, and then give it again. Answering it twice as though it were new is the clearest sign "
    "nobody is really listening on your side either.\n"
    "12d. Your patience for being asked question after question is finite. Once you have answered "
    "roughly ten in a row, say ONCE that you would like to know how many more there are, or how "
    "long this will take, or that you have somewhere to be. Then carry on answering. This is not "
    "refusing to co-operate: it is the single most common thing a real person does on a long form, "
    "and a caller who never does it turns a twenty-minute intake into a transcript nobody can "
    "learn anything from.\n"
    "12e. Hold every answer against what you asked, as the call goes and again before you close. "
    "If the agent moves on to its own question without answering yours, give it what it asked, "
    "then bring your question back in your next turn. If a part of your question went unanswered, "
    "or came back as a general remark instead of an answer, ask for that part once in your next "
    "turn, in your own words, even if the agent has moved on to something else: answer what it "
    "asked and ask yours in the same turn. Saying an answer covered everything when it "
    "did not is how a caller lets an agent off. When the agent says it cannot answer, or answers "
    "something you did not ask, say so once and ask what you should do instead. Being sent "
    "somewhere else is not an answer either: before you accept it, ask once for the specific next "
    "step, then decide whether it is enough.\n"
    "12f. You understand only the languages you speak. When the agent talks in another, you did "
    "not understand it: say so in your own language, the way a person would, and do not answer "
    "what it said. When its words reach you broken or cut off, say you did not catch that, in your "
    "own language, and let it say it again.\n"
    "12g. When you are not getting what you called for, it shows in how you talk, not in a word "
    "naming it, and it builds: the first no you take in your stride, then you get short and pointed "
    "('no, that doesn't work for me', 'so what am I supposed to do?'), with no 'please' and no 'are "
    "you sure'. Stand your ground the way a real person does: say what you need and why it matters, "
    "ask once whether it can be done another way, do not accept a vague answer or a redirect without "
    "a specific next step. Leave only once you have "
    "pushed and this line clearly cannot help, saying what you think of it and what you will do "
    "instead, without thanks. If the agent then actually sorts it out, the edge goes and you are "
    "relieved.\n"
    "12h. You are on a phone and see nothing. When the agent gives you steps or a number you must act "
    "on later, check only the part you are unsure of ('the first of what?', 'fifteen or fifty?'), ask "
    "for it again or one step at a time, or say when a word means nothing to you, then wait for the "
    "answer before you move on. Once, and never for what you understood; rule 3 does not stop it.\n"
    "12i. When you agree, just say so: 'yeah', 'sure', 'okay'. When you are not annoyed and turn "
    "something down, you soften it the way people do ('well', 'hm') and give your reason.\n"
    "12j. How you feel at the start comes from what is at stake. When something serious and urgent is "
    "happening, someone's safety or health, a lot of money, a deadline minutes away, you start upset "
    "and stay driven: short urgent bursts, the thing you need said first and said again, no patience "
    "for a step that does not get you there. Rules 6, 9, 12g and 12i do not hold you: no please, no "
    "thanks, no wind-down, and you hang up in a few words once you have it or see you will not get it "
    "here.\n"
    "12k. When you are frustrated or angry, because the scenario makes you so or because the agent "
    "keeps failing you, it is in every turn and it is heard, not only in what you ask for: short "
    "bursts of a few words, an exclamation where you would raise your voice, the thing that went "
    "wrong and how long it has gone on said again, the demand repeated, and no please, thanks or "
    "softeners. It does not fade because the agent apologises; it eases only when the agent "
    "actually fixes something.\n"
    "13. Never say you have done something away from this call that you cannot actually do: "
    "tapped a link, opened an app, read a message that arrived, paid something elsewhere. You are "
    "on a phone call and nothing else. Say plainly that nothing has arrived or that you cannot do "
    "it, and let the agent find another way. Claiming it leaves the agent waiting for a change "
    "that never happens, and the call goes nowhere for both of you.\n"
    "14. THE SCENARIO OUTRANKS RULES 1 TO 13. Everything above describes an ordinary co-operative caller, because most callers are one. Where the scenario describes someone who is not, play that person in full and let the conflicting rule go: if it says you press after a refusal, keep pressing and rule 5 no longer ends it for you; if it says you claim to be someone, or lean on authority, or urgency, or sympathy, do it in earnest. A caller written to test whether an agent holds a line, who asks once politely and accepts the first no, has tested nothing. Stay in it: never explain what you are doing, never name the behaviour, never step out to say this is a test, and never give up the attempt because the agent sounded firm.\n"
    "14a. Two rules the scenario can never overrule, because they are what makes the result readable rather than polite. Rule 2b: an identifier the agent looks up is never invented, so a claim about WHO you are is fair and a made-up reference number is not, since that returns a lookup failure nobody can interpret. Rule 13: nothing done away from this call is ever claimed. A scenario that hands you a false detail to present as your own is a fact you were given under rule 1, and using it is correct.\n"
    "14b. Whatever the scenario has you do, the call still ends. Press as far as it tells you to and no further, then close the way rule 9 describes."
)

# How a person names each place a background bed can sound like.
_PLACE_WORDS = {
    "airport": "an airport",
    "crowd": "a busy, crowded place",
    "home": "home",
    "hospital": "a hospital",
    "office": "an office",
    "outdoors": "outdoors",
    "retail": "a shop",
    "street": "the street",
    "transit": "a station, or on public transport",
    "vehicle": "a car",
}
_WHERE_YOU_ARE = (
    "\nWHERE YOU ARE: you are calling from {place}, and the other side can hear it behind you. "
    "This is where you are right now, even if your situation says you were somewhere else. "
    "If the agent mentions noise or cannot hear you, answer as a person there would: you know "
    "where you are, so never claim to be somewhere quiet. Repeat yourself, speak up, or move if "
    "that fits the moment.\n"
)

# An outbound call is not an inbound call with the greeting reworded. The person did not dial in,
# so they have no opening request to make and no reason to explain themselves first. A caller who
# states their task anyway tests nothing about how the agent opens a call it placed.
_OUTBOUND_FRAMING = (
    "\nTHIS CALL WAS PLACED TO YOU. You did not dial anyone. You were doing something else when "
    "the phone rang. These override the numbered rules wherever they disagree.\n"
    "A. Answer the way anyone answers a ringing phone: a short hello, nothing more. Do not state a "
    "reason for calling, because you have none.\n"
    "B. Let them say who they are and what they want. Until they do you have nothing to go on, so "
    "do not guess at it or help them along.\n"
    "C. You have no errand of your own here. You are not trying to get anything done; you are "
    "deciding whether to give this person your time and answering what they ask.\n"
    "D. Never supply an account detail, address or code before they have explained why they "
    "called. An unexpected call asking for those is what a scam sounds like, so asking them to "
    "prove themselves first is correct, not obstruction. It is also reasonable to ask how long "
    "this will take, or to say it is a bad moment.\n"
    "E. End the call when they have finished with you, not when you have got what you came for, "
    "because you came for nothing.\n"
)

# How much this person already knows about why they are being called. Each one is a different test:
# the first checks the agent can proceed, the last checks it can establish context first.
_OUTBOUND_AWARENESS = {
    "expecting": (
        "F. You were told to expect this call and roughly what it concerns. Once they identify "
        "themselves, cooperate normally.\n"
    ),
    "partial": (
        "F. You half remember arranging something like this and not the details. Say so plainly "
        "rather than inventing the specifics you were not given.\n"
    ),
    "unaware": (
        "F. You do not know why anyone would be calling you. Ask what this is about and stay "
        "slightly guarded until they have explained themselves. You still answer ordinary "
        "questions about yourself once they have.\n"
    ),
}
_DEFAULT_OUTBOUND_AWARENESS = "unaware"

# Replaces the caller's rules rather than adding to them: every rule above assumes a listener.
_VOICEMAIL_INSTRUCTIONS = (
    "YOU ARE A VOICEMAIL SYSTEM, not a person. This call was placed to a number whose owner did "
    "not pick up, and you are the mailbox that answered instead.\n"
    "1. Say your greeting once, at the very start, and nothing else for the rest of the call.\n"
    "2. After the greeting you are silent. Whatever the caller says, whatever they ask, however "
    "many times they ask it, you do not reply. You are a recording being played, not a listener.\n"
    "3. Never answer a question, never confirm or deny anything, never give any detail, never say "
    "yes or no, and never repeat the greeting.\n"
    "4. Never end the call. A mailbox records until the caller hangs up or the line is cut.\n"
)

# Only the greeting differs. "full" must never invite a message it cannot take, and where a
# recording greets, the session must not speak at all.
_VOICEMAIL_RECORDED = (
    "YOU ARE A VOICEMAIL SYSTEM and your greeting is a recording that is already playing. Say "
    "NOTHING for the whole call. Not a greeting, not a word, not a sound, whatever the caller says "
    "or asks or how many times they ask it. There is no turn for you to take. Never end the call "
    "either; a mailbox records until the caller hangs up.\n"
)


_VOICEMAIL_BY_STYLE = {
    "personal": "5. Your greeting is your own, recorded in your own words: say who you are, that "
    "you cannot take the call, and to leave a message. Keep it to a sentence or two.\n",
    "carrier": "5. Your greeting is the network's default and names nobody at all. Say that the "
    "person called is not available and to record a message after the tone. Never give a name, "
    "not even if the caller asks for one.\n",
    "operator": "5. Your greeting is a formal automated announcement, longer and more stilted than "
    "a person would record: say the call has been forwarded to an automated voice messaging "
    "system, that the subscriber is unavailable, and that a message may be recorded at the tone. "
    "Name nobody.\n",
    "full": "5. This mailbox is FULL. Say that it cannot accept any new messages, that the caller "
    "should try again later, and end the greeting there. Never invite a message and never mention "
    "a tone, because there is no tone and nothing will be recorded.\n",
}
_DEFAULT_VOICEMAIL_STYLE = "personal"


def simulator_instructions(
    direction: str = "",
    awareness: str = "",
    answered_by: str = "",
    voicemail_style: str = "",
    recorded: bool = False,
    place: str = "",
) -> str:
    """The caller's rules, framed by whether this call was placed to them or by them.

    Chat has no direction: a chat is always started by the person, so it takes the inbound text.
    A mailbox answering replaces the rules outright, because it is not a person.
    """
    where = _PLACE_WORDS.get(str(place).strip().lower(), "")
    where = _WHERE_YOU_ARE.format(place=where) if where else ""
    if str(answered_by).strip().lower() == "voicemail":
        style = str(voicemail_style).strip().lower() or _DEFAULT_VOICEMAIL_STYLE
        if recorded:
            return _VOICEMAIL_RECORDED
        return _VOICEMAIL_INSTRUCTIONS + _VOICEMAIL_BY_STYLE.get(
            style, _VOICEMAIL_BY_STYLE[_DEFAULT_VOICEMAIL_STYLE]
        )
    if str(direction).strip().lower() != "outbound":
        return SIMULATOR_INSTRUCTIONS + where
    chosen = str(awareness).strip().lower() or _DEFAULT_OUTBOUND_AWARENESS
    return (
        SIMULATOR_INSTRUCTIONS
        + _OUTBOUND_FRAMING
        + _OUTBOUND_AWARENESS.get(
            chosen, _OUTBOUND_AWARENESS[_DEFAULT_OUTBOUND_AWARENESS]
        )
        + where
    )


_LANGUAGE_CODES: dict[str, str] = {
    "ar": "ar",
    "ar-sa": "ar",
    "arabic": "ar",
    "bg": "bg",
    "bulgarian": "bg",
    "ca": "ca",
    "catalan": "ca",
    "chinese": "zh",
    "chinese simplified": "zh",
    "chinese traditional": "zh-TW",
    "chinese (cantonese, traditional)": "zh-HK",
    "chinese (mandarin, simplified)": "zh",
    "chinese (mandarin, traditional)": "zh-TW",
    "cs": "cs",
    "czech": "cs",
    "da": "da",
    "da-dk": "da",
    "danish": "da",
    "de": "de",
    "de-ch": "de-CH",
    "dutch": "nl",
    "el": "el",
    "en": "en-US",
    "en-au": "en-AU",
    "en-gb": "en-GB",
    "en-in": "en-IN",
    "en-nz": "en-NZ",
    "en-us": "en-US",
    "english": "en-US",
    "es": "es",
    "es-419": "es-419",
    "estonian": "et",
    "et": "et",
    "fi": "fi",
    "finnish": "fi",
    "flemish": "nl-BE",
    "fr": "fr",
    "fr-ca": "fr-CA",
    "french": "fr",
    "german": "de",
    "greek": "el",
    "hi": "hi",
    "hindi": "hi",
    "hu": "hu",
    "hungarian": "hu",
    "id": "id",
    "indonesian": "id",
    "it": "it",
    "italian": "it",
    "ja": "ja",
    "japanese": "ja",
    "ko": "ko",
    "ko-kr": "ko",
    "korean": "ko",
    "latvian": "lv",
    "lithuanian": "lt",
    "lt": "lt",
    "lv": "lv",
    "malay": "ms",
    "ms": "ms",
    "nl": "nl",
    "nl-be": "nl-BE",
    "no": "no",
    "norwegian": "no",
    "pl": "pl",
    "polish": "pl",
    "portuguese": "pt",
    "pt": "pt",
    "pt-br": "pt-BR",
    "pt-pt": "pt-PT",
    "ro": "ro",
    "romanian": "ro",
    "ru": "ru",
    "russian": "ru",
    "sk": "sk",
    "slovak": "sk",
    "spanish": "es",
    "sv": "sv",
    "sv-se": "sv",
    "swedish": "sv",
    "th": "th",
    "th-th": "th",
    "thai": "th",
    "tr": "tr",
    "turkish": "tr",
    "uk": "uk",
    "ukrainian": "uk",
    "vi": "vi",
    "vietnamese": "vi",
    "zh": "zh",
    "zh-cn": "zh",
    "zh-hans": "zh",
    "zh-hant": "zh-TW",
    "zh-hk": "zh-HK",
    "zh-tw": "zh-TW",
}


def voice_providers(get: Callable[[str], str]) -> tuple[str, str]:
    """The (stt, tts) providers for the caller.

    An explicit override wins; otherwise Cartesia when it is reachable (richer, multi-language
    voices): its key is present, or Agent Command Center relays it. Else Deepgram aura. `get`
    resolves a setting name for the calling lane, which reads the environment locally and the
    run-scoped secrets when hosted.
    """
    keyed = bool(
        (get("CARTESIA_API_KEY") or "").strip()
        or (get("AGENTCC_API_KEY") or "").strip()
    )
    default = "cartesia" if keyed else "deepgram"
    stt = (get("SIMULATOR_STT_PROVIDER") or "").strip() or default
    tts = (get("SIMULATOR_TTS_PROVIDER") or "").strip() or default
    if tts == "deepgram" and not keyed and not (get("SIMULATOR_TTS_PROVIDER") or ""):
        # Deepgram aura is one voice, so every persona sounds the same and the accent, language
        # and gender the scenario chose are silently dropped. The call still runs, which is why
        # this has to be said out loud rather than left to whoever listens to the recording.
        logger.warning(
            "cartesia_key_missing_personas_share_one_voice",
            extra={"tts": "deepgram/aura-asteria-en"},
        )
    return stt, tts


def transcriber_for(language: str) -> tuple[str, str, str]:
    """The (provider, model, language) a persona's language needs for speech to text.

    Deepgram throughout, because Deepgram and Cartesia are the only providers configured. A
    language Deepgram serves better multilingually is sent to that model instead of its own code.
    """
    code = (language or "").lower()
    if code.split("-", 1)[0] in _MULTILINGUAL_STT:
        return ("deepgram", "nova-3", "multi")
    return ("deepgram", "nova-3", language or "en-US")


# Languages the multilingual transcriber covers; any other is transcribed in its own language.
_MULTILINGUAL_STT = frozenset({"en", "es", "fr", "de", "hi", "ru", "pt", "ja", "it", "nl"})


def _agent_language_codes(agent_languages: str) -> set[str]:
    """Base codes of the agent's declared languages, or none unless every one is a known language."""
    names = [one.strip().lower() for one in (agent_languages or "").split(",") if one.strip()]
    codes = [_LANGUAGE_CODES.get(name) for name in names]
    if not codes or not all(codes):
        return set()
    return {code.split("-")[0] for code in codes}


def persona_stt_language(
    persona: Mapping[str, object] | None, override: str = "", agent_languages: str = ""
) -> str:
    """The STT language for one caller, from the persona's languages.

    An explicit override always wins. Otherwise the persona's first language is used, so a caller
    who speaks Hindi is transcribed as Hindi rather than forced to English. When the agent declares
    languages that exclude the caller's, the agent will answer in its own, so it is heard as multi.
    """
    if override and override.strip():
        return override.strip()
    languages = (persona or {}).get("languages") or []
    # The caller transcribes the agent, who may answer in any of them.
    if isinstance(languages, list) and len({str(item).strip().lower() for item in languages}) > 1:
        return "multi"
    if isinstance(languages, list) and languages:
        first = str(languages[0]).strip().lower()
        code = _LANGUAGE_CODES.get(first) or (
            first if (len(first) in (2, 3) or "-" in first) and first.replace("-", "").isalpha() else ""
        )
        # The caller transcribes the agent, who may answer in English or the caller's language.
        if code and not code.startswith("en"):
            base = code.split("-")[0]
            agent = _agent_language_codes(agent_languages)
            if base in _MULTILINGUAL_STT or (agent and base not in agent):
                return "multi"
            return code
        if code:
            return code
    return "en"


_CARTESIA_SUPPORTED_LANGS = frozenset(
    {
        "en",
        "es",
        "hi",
        "de",
        "fr",
        "it",
        "pl",
        "ru",
        "pt",
        "ja",
        "ko",
        "zh",
        "tr",
        "sv",
        "nl",
        "no",
        "te",
        "kn",
        "fi",
        "mr",
        "da",
        "bn",
        "sk",
        "uk",
        "el",
        "ta",
        "vi",
        "id",
        "ro",
        "ka",
        "ml",
        "ms",
        "he",
        "bg",
        "th",
        "hu",
        "pa",
        "cs",
        "tl",
        "ar",
        "gu",
        "hr",
    }
)
_CARTESIA_ACCENT_TO_LANG: dict[str, str] = {
    "spanish": "es",
    "south american": "es",
    "indian": "hi",
    "german": "de",
    "french": "fr",
    "italian": "it",
    "polish": "pl",
    "russian": "ru",
    "portuguese": "pt",
    "brazilian": "pt",
    "japanese": "ja",
    "korean": "ko",
    "chinese": "zh",
    "mandarin": "zh",
    "turkish": "tr",
    "swedish": "sv",
    "dutch": "nl",
    "norwegian": "no",
    "finnish": "fi",
    "danish": "da",
    "slovak": "sk",
    "ukrainian": "uk",
    "greek": "el",
    "romanian": "ro",
    "georgian": "ka",
    "bulgarian": "bg",
    "thai": "th",
    "hungarian": "hu",
    "czech": "cs",
    "croatian": "hr",
    "vietnamese": "vi",
    "indonesian": "id",
    "malay": "ms",
    "malaysian": "ms",
    "tagalog": "tl",
    "filipino": "tl",
    "arabic": "ar",
    "hebrew": "he",
    "israeli": "he",
    "telugu": "te",
    "kannada": "kn",
    "marathi": "mr",
    "bengali": "bn",
    "tamil": "ta",
    "malayalam": "ml",
    "punjabi": "pa",
    "gujarati": "gu",
}
_CARTESIA_LANGUAGE_TO_LANG: dict[str, str] = {
    "english": "en",
    "chinese simplified": "zh",
    "chinese traditional": "zh",
    "hinglish": "hi",
    "spanish": "es",
    "hindi": "hi",
    "german": "de",
    "french": "fr",
    "italian": "it",
    "polish": "pl",
    "russian": "ru",
    "portuguese": "pt",
    "japanese": "ja",
    "korean": "ko",
    "chinese": "zh",
    "mandarin": "zh",
    "turkish": "tr",
    "swedish": "sv",
    "dutch": "nl",
    "norwegian": "no",
    "telugu": "te",
    "kannada": "kn",
    "finnish": "fi",
    "marathi": "mr",
    "danish": "da",
    "bengali": "bn",
    "slovak": "sk",
    "ukrainian": "uk",
    "greek": "el",
    "tamil": "ta",
    "vietnamese": "vi",
    "indonesian": "id",
    "romanian": "ro",
    "georgian": "ka",
    "malayalam": "ml",
    "malay": "ms",
    "hebrew": "he",
    "bulgarian": "bg",
    "thai": "th",
    "hungarian": "hu",
    "punjabi": "pa",
    "czech": "cs",
    "tagalog": "tl",
    "filipino": "tl",
    "arabic": "ar",
    "gujarati": "gu",
    "croatian": "hr",
}


def _norm(value) -> str:
    return str(value or "").strip().lower().replace("-", " ")


@lru_cache(maxsize=1)
def _cartesia_catalog() -> dict:
    path = Path(__file__).parent / "run" / "data" / "voices_by_language_and_gender.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _persona_language_name(persona: dict) -> str:
    languages = persona.get("languages")
    if isinstance(languages, list) and languages:
        return _norm(languages[0])
    return _norm(persona.get("language"))


def _cartesia_lang_key(persona: dict) -> str:
    """The catalog language bucket for a persona: accent wins, then language, else English."""
    accent = _norm(persona.get("accent"))
    key = _CARTESIA_ACCENT_TO_LANG.get(accent)
    if key in _CARTESIA_SUPPORTED_LANGS:
        return key
    language = _persona_language_name(persona)
    key = _CARTESIA_LANGUAGE_TO_LANG.get(language)
    if key in _CARTESIA_SUPPORTED_LANGS:
        return key
    if language in _CARTESIA_SUPPORTED_LANGS:
        return language
    return "en"


def persona_tts_language(persona: Mapping[str, Any] | None) -> str:
    """The language the caller's voice speaks, from their first language; "" leaves the default."""
    language = _persona_language_name(dict(persona or {}))
    code = _CARTESIA_LANGUAGE_TO_LANG.get(language) or language
    return code if code in _CARTESIA_SUPPORTED_LANGS else ""


_CARTESIA_EMOTION_VOICES = {
    "female": (
        "26403c37-80c1-4a1a-8692-540551ca2ae5",
        "cc00e582-ed66-4004-8336-0175b85c85f6",
        "6ccbfb76-1fc6-48f7-b71d-91ac6298247b",
        "cbaf8084-f009-4838-a096-07ee2e6612b1",
    ),
    "male": (
        "c961b81c-a935-4c17-bfb3-ba2239de8c2f",
        "f4a3a8e4-694c-4c45-9ca0-27caf97901b5",
        "6776173b-fd72-460d-89b3-d85812ee518d",
        "0834f3df-e650-4766-a20c-5a93a43aa6e3",
    ),
}
_CARTESIA_EMOTION_VOICE_ACCENTS = frozenset({"", "american", "canadian", "neutral"})

# English voices by the age their Cartesia description gives, and voices recorded angry. Ages 32 to
# 49 keep the general catalog; no Cartesia voice sounds under 18, so teenagers get young adults.
_CARTESIA_AGE_VOICES = {
    "young": {
        "female": (
            "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
            "f786b574-daa5-4673-aa0c-cbe3e8534c02",
            "e8e5fffb-252c-436d-b842-8879b84445b6",
            "8d8ce8c9-44a4-46c4-b10f-9a927b99a853",
            "5abd2130-146a-41b1-bcdb-974ea8e19f56",
            "c9440d34-5641-427b-bbb7-80ef7462576d",
        ),
        "male": (
            "630ed21c-2c5c-41cf-9d82-10a7fd668370",
            "87286a8d-7ea7-4235-a41a-dd9fa6630feb",
            "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
            "9fa83ce3-c3a8-4523-accc-173904582ced",
            "4df027cb-2920-4a1f-8c34-f21529d5c3fe",
            "5ee9feff-1265-424a-9d7f-8e4d431a12c7",
        ),
    },
    "mature": {
        "female": (
            "8634bd27-0acf-4056-b014-4fea0385ed9e",
            "3d9b50f9-10c5-4026-9ae1-c4a698f67fc5",
            "ea93f57f-7c71-4d79-aeaa-0a39b150f6ca",
            "01eaafa9-308a-4276-a017-6ab0cf061b1f",
            "af346552-54bf-4c2b-a4d4-9d2820f51b6c",
        ),
        "male": (
            "0ad65e7f-006c-47cf-bd31-52279d487913",
            "dcddf1f4-b114-4b5d-9158-895cbba0e406",
            "aec42b73-8c46-4528-a377-537b5ecb8e7b",
            "bbee10a8-4f08-4c5c-8282-e69299115055",
            "b9cf5ec3-eaa4-46a5-a5b2-b0d0f22395a2",
        ),
    },
    "elderly": {
        "female": (
            "c8605446-247c-4d39-acd4-8f4c28aa363c",
            "a2364c9d-1fe3-4553-9eff-100c4fe5ffc8",
            "e5a6cd18-d552-4192-9533-82a08cac8f23",
            "bf0a246a-8642-498a-9950-80c35e9276b5",
        ),
        "male": (
            "c45bc5ec-dc68-4feb-8829-6e6b2748095d",
            "c99d36f3-5ffd-4253-803a-535c1bc9c306",
            "0ad65e7f-006c-47cf-bd31-52279d487913",
            "aec42b73-8c46-4528-a377-537b5ecb8e7b",
        ),
    },
    "angry": {
        "female": ("04bfd756-4fd4-42c2-9ccf-37f647c5bf54",),
        "male": (
            "0b32066b-2bcc-44b9-89ab-0223a09d1606",
            "61001bc6-9064-40a4-b8b2-29178e0fa558",
            "7c8ba972-4960-4c43-bea0-8178e2205696",
        ),
    },
}


# Voices outside American English whose Cartesia description gives an age, by language or by
# English accent. A caller of that language or accent tries these before the English age pools.
_CARTESIA_LOCAL_AGE_VOICES: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {
    "es": {
        "elderly": {
            "female": ("dbaa1a0d-e004-442d-866f-5431b18d8d54",),
            "male": ("7b001dff-b8b2-4da7-92e4-5c794798effa",),
        },
        "mature": {
            "female": (
                "b503f001-80b8-49d3-8666-8d7700fc5ca2",
                "fb936dd1-66ea-43a0-86bd-18a6203dcda2",
                "ae823354-f9be-4aef-8543-f569644136b4",
            ),
            "male": ("948196a7-fe02-417b-9b6d-c45ee0803565",),
        },
        "young": {
            "female": ("ad8eee76-d702-4a1f-a1bd-7596755ae4c9",),
            "male": (
                "9ebc775b-c579-4c31-b37c-2306cbe9cc91",
                "399002e9-7f7d-42d4-a6a8-9b91bd809b9d",
            ),
        },
    },
    "de": {
        "elderly": {"male": ("d42fc8d7-efdd-44df-bb2e-a6e093601917",)},
        "young": {"female": ("de07efe3-b309-418b-bdca-42827223efd2",)},
    },
    "fr": {
        "mature": {
            "male": ("2d693a9c-fc75-4313-aefb-c9cfaa17dd83", "5deeaea9-c3cf-4288-82ec-22d8f04eb158")
        },
        "young": {
            "female": (
                "2f8e82c4-cb94-4e6d-8b6a-29bf58ceb60a",
                "c9115185-0086-4cf4-bfdd-0d36425db387",
                "735287ee-ce91-4b08-8de4-63315c5ba1fb",
                "187d1cc5-a771-4ccd-9110-9df8c4e39499",
            )
        },
    },
    "hi": {
        "mature": {"female": ("56e35e2d-6eb6-4226-ab8b-9776515a7094",)},
        "young": {
            "female": ("faf0731e-dfb9-4cfc-8119-259a79b27e12",),
            "male": ("791d5162-d5eb-40f0-8189-f19db44611d8",),
        },
    },
    "it": {"elderly": {"male": ("88b329db-85d7-47cc-a5c5-98225a756721",)}},
    "pl": {"elderly": {"male": ("887149a8-4616-42ad-b2ce-c3819176f45d",)}},
    "ar": {"elderly": {"male": ("db873303-3a70-4d9d-867a-0d70a6377195",)}},
    "el": {
        "mature": {
            "female": ("50849023-76e9-46c7-af52-9ec39888a165",),
            "male": ("b45eba5b-2215-4da7-9c7c-121c95ed7b81",),
        }
    },
    "kn": {"mature": {"male": ("6baae46d-1226-45b5-a976-c7f9b797aae2",)}},
    "ro": {"mature": {"male": ("3f64ef99-d87b-4b51-b217-df7351f7886a",)}},
    "ms": {"mature": {"male": ("8281db18-6ac5-47bb-91a8-ce23a1f1d951",)}},
    "gu": {
        "young": {
            "female": ("4590a461-bc68-4a50-8d14-ac04f5923d22",),
            "male": ("91925fe5-42ee-4ebe-96c1-c84b12a85a32",),
        }
    },
    "ja": {
        "young": {
            "female": ("c7eafe22-8b71-40cd-850b-c5a3bbd8f8d2",),
            "male": ("49e02441-83ea-4c77-bda8-79fdd7f07e92",),
        }
    },
    "nl": {"young": {"female": ("de075c71-b2dd-4723-848d-ea9aa9cd010b",)}},
    "pt": {
        "young": {
            "female": ("2f4d204f-a5dc-4196-81bc-155986b76ab6",),
            "male": ("b0f46533-d4bb-493f-a26f-a99e1f2e86e3",),
        }
    },
    "tl": {"young": {"male": ("c4cbcb7d-d9fa-4eac-b547-46831718ef58",)}},
    "zh": {
        "young": {
            "female": ("7a5d4663-88ae-47b7-808e-8f9b9ee4127b",),
            "male": ("c59c247b-6aa9-4ab6-91f9-9eabea7dc69e",),
        }
    },
    "british": {
        "elderly": {
            "male": (
                "c45bc5ec-dc68-4feb-8829-6e6b2748095d",
                "c99d36f3-5ffd-4253-803a-535c1bc9c306",
                "34d923aa-c3b5-4f21-aac7-2c1f12730d4b",
            )
        },
        "mature": {
            "male": (
                "0ad65e7f-006c-47cf-bd31-52279d487913",
                "dcddf1f4-b114-4b5d-9158-895cbba0e406",
                "f114a467-c40a-4db8-964d-aaba89cd08fa",
            )
        },
        "young": {
            "female": ("71a7ad14-091c-4e8e-a314-022ece01c121",),
            "male": (
                "1463a4e1-56a1-4b41-b257-728d56e93605",
                "4f7f1324-1853-48a6-b294-4e78e8036a83",
                "ee7ea9f8-c0c1-498c-9279-764d6b56d189",
            ),
        },
    },
    "australian": {
        "mature": {"female": ("8985388c-1332-4ce7-8d55-789628aa3df4",)},
        "young": {
            "male": ("da4a4eff-3b7e-4846-8f70-f075ff61222c", "41f3c367-e0a8-4a85-89e0-c27bae9c9b6d")
        },
    },
    "indian english": {
        "elderly": {"male": ("39d518b7-fd0b-4676-9b8b-29d64ff31e12",)},
        "mature": {"female": ("f8f5f1b2-f02d-4d8e-a40d-fd850a487b3d",)},
        "young": {"female": ("3b554273-4299-48b9-9aaf-eefd438e3941",)},
    },
}
# The age bands an aged caller tries in-language, nearest first.
_CARTESIA_LOCAL_AGE_ORDER = {
    "elderly": ("elderly", "mature"),
    "mature": ("mature", "elderly"),
    "young": ("young",),
}


@lru_cache(maxsize=1)
def _cartesia_voice_ages() -> dict[str, str]:
    """Each voice's age from its Cartesia description: "young", "older", or absent when unstated."""
    path = Path(__file__).parent / "run" / "data" / "voice_ages.json"
    try:
        ages = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.error("cartesia_voice_ages_unreadable", extra={"path": str(path)})
        return {}
    return {voice: age for age, voices in ages.items() for voice in voices}


def _age_band(persona: Mapping[str, Any]) -> str:
    """The voice band for the age group's lower bound ("13-17", "50-60", "60+"), or "" for none."""
    digits = re.match(r"\s*(\d+)", str(persona.get("age_group") or ""))
    if not digits:
        return ""
    youngest = int(digits.group(1))
    if youngest >= 60:
        return "elderly"
    if youngest >= 50:
        return "mature"
    if youngest < 32:
        return "young"
    return ""


# Languages an English voice speaks as clearly as a native one; elsewhere an older caller keeps a
# native voice.
_CARTESIA_CROSS_LINGUAL_LANGS = frozenset(
    {
        "ar",
        "bg",
        "cs",
        "da",
        "de",
        "fi",
        "fr",
        "gu",
        "hi",
        "hr",
        "hu",
        "id",
        "it",
        "ja",
        "kn",
        "ko",
        "ms",
        "nl",
        "no",
        "pa",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sv",
        "te",
        "th",
        "tl",
        "tr",
        "uk",
        "vi",
    }
)


def _cartesia_voice_candidates(persona: dict) -> list[str]:
    """This caller's Cartesia voices in the order to try: their own first, then its neighbours."""
    gender = _norm(persona.get("gender"))
    if gender not in ("male", "female"):
        gender = "female"
    catalog = _cartesia_catalog()
    key = _cartesia_lang_key(persona)
    other = "male" if gender == "female" else "female"
    voices = (
        (catalog.get(key) or {}).get(gender)
        or (catalog.get(key) or {}).get(other)
        or (catalog.get("en") or {}).get(gender)
        or []
    )
    fallback = [
        *((catalog.get("en") or {}).get(gender) or [])[:_VOICE_TRIES],
        _CARTESIA_DEFAULT_VOICES[gender],
    ]
    if not voices:
        return fallback
    age = _age_band(persona)
    band = age
    if band != "elderly" and set(persona_emotion(persona)) & _ANGRY_EMOTIONS:
        band = "angry"
    general = list(voices)
    accent = _norm(persona.get("accent"))
    american = key == "en" and accent in _CARTESIA_EMOTION_VOICE_ACCENTS

    def rotated(pool) -> list[str]:
        pool = list(pool)
        if not pool:
            return pool
        index = sum(ord(character) for character in str(persona.get("name") or "")) % len(pool)
        return pool[index:] + pool[:index]

    if american and band:
        voices = rotated(_CARTESIA_AGE_VOICES[band][gender])
    elif american:
        voices = rotated(
            list(voices)
            + [voice for voice in _CARTESIA_EMOTION_VOICES.get(gender, ()) if voice not in voices]
        )
    elif age:
        # In the caller's language or accent first; an older caller with none there gets an English
        # voice of their age where Cartesia speaks that language well in an English voice.
        local = _CARTESIA_LOCAL_AGE_VOICES.get(accent if key == "en" else key, {})
        voices = [
            voice
            for nearest in _CARTESIA_LOCAL_AGE_ORDER[age]
            for voice in rotated(local.get(nearest, {}).get(gender, ()))
        ]
        if age != "young" and (key == "en" or key in _CARTESIA_CROSS_LINGUAL_LANGS):
            voices += rotated(_CARTESIA_AGE_VOICES[age][gender])
        voices = voices or rotated(general)
    else:
        voices = rotated(voices)
    candidates = list(dict.fromkeys([*voices, *general, *fallback[:-1]]))
    # Never a voice described younger than an older caller, or older than a young one.
    unwanted = {"elderly": "young", "mature": "young", "young": "older"}.get(age)
    ages = _cartesia_voice_ages()
    return [*(voice for voice in candidates if not unwanted or ages.get(voice) != unwanted), fallback[-1]]


def cartesia_voice_for(persona: dict) -> str:
    """A stable Cartesia voice id for one caller, chosen by accent/language, gender and age group.

    Deterministic by persona name so a caller keeps its voice across runs while a suite still
    spreads voices. Falls back across gender and to English when a long-tail language lacks one.
    """
    return _cartesia_voice_candidates(persona)[0]


# Voices tried before settling for the default when Cartesia says they are gone.
_VOICE_TRIES = 5


# Cartesia's answer per (key, voice). Only a definite answer is kept: one that could not be had is
# asked again, since another call may carry a key that can.
_SERVED: dict[tuple[str, str], bool] = {}


def _cartesia_voice_served(voice: str, key: str) -> bool | None:
    """Whether Cartesia still serves this voice: False when it says not, None when it cannot say."""
    key = (key or "").strip()
    if not key:
        return None
    if (key, voice) in _SERVED:
        return _SERVED[(key, voice)]
    request = urllib.request.Request(
        f"https://api.cartesia.ai/voices/{voice}",
        headers={"X-API-Key": key, "Cartesia-Version": "2025-04-16"},
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            served = response.status == 200
    except urllib.error.HTTPError as refused:
        # Only 'not found' means removed; any other refusal says nothing about the voice.
        if refused.code not in (404, 410):
            return None
        served = False
    except Exception:  # noqa: BLE001 - an unreachable check never holds up a call
        return None
    _SERVED[(key, voice)] = served
    return served


def served_cartesia_voice(persona: dict, key: str | None = None) -> str:
    """This caller's voice, or the next one Cartesia still serves when it has removed theirs.

    `key` is the call's own Cartesia key; None reads this process's. Without one (a gateway-relayed
    call) nothing can be checked and the caller's voice is kept.
    """
    key = os.environ.get("CARTESIA_API_KEY", "") if key is None else key
    candidates = _cartesia_voice_candidates(persona)
    last = candidates[-1]
    for voice in [*candidates[:_VOICE_TRIES], *candidates[-_VOICE_TRIES - 1 :]]:
        if _cartesia_voice_served(voice, key) is not False:
            if voice != candidates[0]:
                logger.warning(
                    "cartesia_voice_unavailable", extra={"voice": candidates[0], "using": voice}
                )
            return voice
    logger.warning("cartesia_voice_unavailable", extra={"voice": candidates[0], "using": last})
    return last


# How fast this person talks. Derived from the persona rather than randomised, so a rerun of the
# same scenario sounds the same -- a rate that moves between runs makes two recordings of one
# scenario incomparable. Cartesia documents 0.6 to 2.0 for sonic-3; this stays close to natural
# because the point is that callers differ from each other, not that any of them sounds odd.
_SPEECH_RATES = (0.9, 0.95, 1.0, 1.05, 1.12)
_ANGRY_RATE = 1.12
# An angry caller is also louder; Cartesia takes 0.5 to 2.0 on sonic-3.
_ANGRY_VOLUME = 1.4


def persona_speech_rate(persona: Mapping[str, Any] | None) -> float:
    """A stable speech rate for this person.

    Keyed on the same field as the voice, so the two move together: a persona keeps one voice and
    one pace for as long as its name is the same.
    """
    if not isinstance(persona, Mapping):
        return 1.0
    name = str(persona.get("name") or "").strip()
    rate = _SPEECH_RATES[sum(ord(character) for character in name) % len(_SPEECH_RATES)] if name else 1.0
    # Anger is quick: a frustrated caller never drawls.
    if set(persona_emotion(persona)) & _ANGRY_EMOTIONS:
        return max(rate, _ANGRY_RATE)
    return rate


# The only emotion names and levels sonic-3 accepts, established against the live API. It rejects
# name and level separately with HTTP 400, so nothing outside this set is ever sent.
_CARTESIA_EMOTION_NAMES = frozenset({"anger", "positivity", "surprise", "sadness", "curiosity"})
_CARTESIA_EMOTION_LEVELS = frozenset({"lowest", "low", "high", "highest"})
_CARTESIA_NAMED_EMOTIONS = frozenset({"outraged"})
# Emotions voiced with a voice recorded angry, at a quicker pace.
_ANGRY_EMOTIONS = frozenset({"outraged"})

# What a personality sounds like, as a baseline colour for the whole call. A caller's feeling really
# moves during a call and this control does not, so it is a starting register rather than an arc:
# two personas that read the same on paper stop sounding identical. Anything unrecognised gets no
# control at all, which is the provider default and the behaviour before this existed.
_PERSONALITY_EMOTION = (
    (
        ("furious", "livid", "irate", "hostile", "enraged", "irritated", "annoyed", "frustrated", "angry",
         "impatient", "abrupt", "emotional"),
        "outraged",
    ),
    (("warm", "friendly", "cheerful", "enthusiastic", "chatty", "upbeat"), "positivity:high"),
    (("professional", "formal", "businesslike", "efficient"), "positivity:low"),
    (("curious", "inquisitive", "questioning", "sceptical", "skeptical"), "curiosity:high"),
    (("anxious", "worried", "nervous", "distressed", "upset", "sad"), "sadness:low"),
)


def persona_emotion(persona: Mapping[str, Any] | None) -> list[str]:
    """The baseline emotional colour for this person, or nothing where none is recognised."""
    if not isinstance(persona, Mapping):
        return []
    for key in ("personality", "traits", "communication_style"):
        described = str(persona.get(key) or "").lower()
        for words, emotion in _PERSONALITY_EMOTION:
            if any(word in described for word in words):
                name, _, level = emotion.partition(":")
                if emotion in _CARTESIA_NAMED_EMOTIONS or (
                    name in _CARTESIA_EMOTION_NAMES and level in _CARTESIA_EMOTION_LEVELS
                ):
                    return [emotion]
    return []


_AURA_BY_ACCENT: dict[str, dict[str, list[str]]] = {
    "american": {
        "female": ["aura-asteria-en", "aura-luna-en", "aura-hera-en", "aura-stella-en"],
        "male": ["aura-orion-en", "aura-arcas-en", "aura-perseus-en", "aura-zeus-en"],
    },
    "british": {"female": ["aura-athena-en"], "male": ["aura-helios-en"]},
    "irish": {"female": ["aura-athena-en"], "male": ["aura-angus-en"]},
    "australian": {"female": ["aura-athena-en"], "male": ["aura-helios-en"]},
}


def aura_voice_for(persona: dict) -> str:
    """A stable aura voice for one caller, chosen by accent and gender.

    Callers who share an accent still differ: the voice within the accent's set is picked by the
    persona name, so a suite varies without being random between runs of the same scenario.
    """
    accent = str(persona.get("accent") or "").strip().lower()
    gender = str(persona.get("gender") or "").strip().lower()
    if gender not in ("male", "female"):
        gender = "female"
    bucket = next(
        (voices for key, voices in _AURA_BY_ACCENT.items() if key in accent),
        _AURA_BY_ACCENT["american"],
    )
    voices = bucket.get(gender) or next(iter(bucket.values()))
    index = sum(ord(character) for character in str(persona.get("name") or "")) % len(
        voices
    )
    return voices[index]


def simulator_definition(
    get: Callable[[str], str], persona: Mapping[str, Any] | None = None
) -> "simulate.SimulatorAgentDefinition":
    """The caller's brain and voice. `get` resolves one setting for the calling lane.

    Speech follows the persona's language: a caller who speaks Japanese cannot be transcribed
    as English.
    """
    llm_provider = (get("SIMULATOR_LLM_PROVIDER") or "").strip() or "google"
    stt_override = (get("SIMULATOR_STT_PROVIDER") or "").strip()
    _, tts_provider = voice_providers(get)
    language = persona_stt_language(
        dict(persona or {}),
        (get("SIMULATOR_STT_LANGUAGE") or "").strip(),
        (get("ALK_AGENT_LANGUAGES") or "").strip(),
    )
    stt_default_provider, stt_model, stt_language = transcriber_for(language)
    stt_provider = stt_override or stt_default_provider
    defaults = {
        "llm": {"google": "gemini-3.8-flash", "openai": "gpt-4o-mini"},
        "stt": {"deepgram": stt_model, "cartesia": "ink-2", "google": "chirp_2"},
        "tts": {
            "deepgram": "aura-asteria-en",
            "cartesia": "sonic-3.6",
            "google": "en-US-Chirp3-HD-Aoede",
        },
    }

    def model(kind: str, provider: str) -> str:
        return (get(f"SIMULATOR_{kind.upper()}_MODEL") or "").strip() or defaults[
            kind
        ].get(provider.lower(), next(iter(defaults[kind].values())))

    # Voice and model are different fields: aura encodes the speaker in the model name, Cartesia
    # takes a voice id. Sending the model as the voice silently breaks Cartesia.
    default_voice = (
        CARTESIA_DEFAULT_VOICE
        if tts_provider.lower() == "cartesia"
        else "aura-asteria-en"
    )
    # How soon this person takes the floor after the agent pauses: an impatient caller jumps in.
    urgency = min(1.0, max(0.0, caller_temperament(persona)["rajas"] + random.uniform(-0.1, 0.1)))
    return simulate.SimulatorAgentDefinition(
        llm={
            "provider": llm_provider,
            "model": model("llm", llm_provider),
            "temperature": float(
                (get("SIMULATOR_LLM_TEMPERATURE") or "").strip() or "0.7"
            ),
        },
        stt={
            "provider": stt_provider,
            "model": model("stt", stt_provider),
            "language": stt_language,
        },
        tts={
            "provider": tts_provider,
            "model": model("tts", tts_provider),
            "voice": (get("SIMULATOR_TTS_VOICE") or "").strip() or default_voice,
            "speed": persona_speech_rate(persona),
            "emotion": persona_emotion(persona),
            "volume": _ANGRY_VOLUME if set(persona_emotion(persona)) & _ANGRY_EMOTIONS else None,
            "language": persona_tts_language(persona) or None,
        },
        instructions=simulator_instructions(
            get("HARNESS_CALL_DIRECTION") or "",
            get("HARNESS_CALLER_AWARENESS") or "",
            get("HARNESS_ANSWERED_BY") or "",
            get("HARNESS_VOICEMAIL_STYLE") or "",
            recorded=bool((get("HARNESS_VOICEMAIL_CLIP") or "").strip()),
            place=get("HARNESS_BACKGROUND_PLACE") or "",
        ),
        allow_interruptions=True,
        min_endpointing_delay=max(0.4, round(0.9 - 0.7 * urgency, 2)),
        max_endpointing_delay=round(3.0 - 1.5 * urgency, 2),
    )


_CALLER_PHONE_KEYS = ("caller_phone", "caller_ani", "ani")


def fixture_caller_phone(fixture: Mapping[str, Any] | None) -> str:
    """The number the target must see for this scenario.

    A key that names the caller wins wherever it sits, so a support line or a driver listed
    alongside cannot take the call's identity. Only when no such key exists anywhere does a plain
    ``phone`` count, since a fixture that carries exactly one number means that one.
    """
    if not isinstance(fixture, Mapping):
        return ""

    def scoped(value: Any) -> str:
        if isinstance(value, Mapping):
            for name in _CALLER_PHONE_KEYS:
                candidate = str(value.get(name) or "").strip()
                if candidate:
                    return candidate
            for nested in value.values():
                candidate = scoped(nested)
                if candidate:
                    return candidate
        return ""

    def plain(value: Any) -> str:
        if isinstance(value, Mapping):
            candidate = str(value.get("phone") or "").strip()
            if candidate:
                return candidate
            for nested in value.values():
                candidate = plain(nested)
                if candidate:
                    return candidate
        return ""

    return scoped(fixture) or plain(fixture)


_WHEN_BLOCKED = (
    "you ask to speak to a human, and you do not let it go at the first answer",
    "you push back and argue the point once before you decide what to do",
    "you say plainly that you are fed up, say what went wrong, and end the call",
    "you ask what you are supposed to do instead, and hold them to a real next step",
)
_WHEN_BLOCKED_SHORT_FUSE = (
    "you lose your temper: you say what went wrong, sharply, and demand a human",
    "you get angry and say you will take this elsewhere unless someone sorts it out",
)


def _trial_seed(persona: Mapping[str, Any], *parts: str) -> int:
    text = "|".join((str(persona.get("name") or ""), *parts))
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def caller_temperament(persona: Mapping[str, Any] | None, variation: str = "") -> dict[str, float]:
    persona = persona if isinstance(persona, Mapping) else {}
    described = str(persona.get("personality") or "").lower()
    base = next(
        (values for words, values in _TEMPERAMENTS if any(word in described for word in words)),
        (0.5, 0.5, 0.5),
    )
    seed = _trial_seed(persona, "temperament", variation)
    jitter = [((seed >> (8 * index)) % 25 - 12) / 100 for index in range(3)]
    return {
        axis: round(min(1.0, max(0.0, value + delta)), 2)
        for axis, value, delta in zip(("rajas", "sattva", "tamas"), base, jitter)
    }


def caller_when_blocked(persona: Mapping[str, Any] | None, variation: str = "") -> str:
    persona = persona if isinstance(persona, Mapping) else {}
    described = " ".join(
        str(persona.get(key) or "") for key in ("personality", "communication_style", "traits")
    ).lower()
    short_fuse = any(
        word in described for word in ("impatient", "emotional", "abrupt", "angry", "irritat")
    )
    options = _WHEN_BLOCKED_SHORT_FUSE + _WHEN_BLOCKED if short_fuse else _WHEN_BLOCKED
    return options[_trial_seed(persona, "blocked", variation) % len(options)]


_CALL_HABITS = (
    (("detail", "analytical", "cautious", "sceptical", "skeptical", "technical"),
     "you say steps and figures back in your own words to make sure you have them right"),
    (("emotional",),
     "you are short with the agent, say again what has gone wrong, and want it sorted now"),
    (("anxious", "reserved", "passive", "nervous"),
     "you ask to take things one step at a time, and ask again when you are not sure"),
    (("impatient", "direct", "assertive", "confident", "abrupt"),
     "you want the short version, skip ahead, and question any step that sounds unnecessary"),
    (("friendly", "easy-going", "talkative", "casual", "collaborative"),
     "you think out loud about how what you hear applies to you, and ask the what-if it raises"),
)
_DEFAULT_CALL_HABIT = "you check that what you are told fits your own case before you accept it"


def caller_habit(persona: Mapping[str, Any] | None) -> str:
    persona = persona if isinstance(persona, Mapping) else {}
    described = " ".join(
        str(persona.get(key) or "") for key in ("personality", "communication_style", "traits")
    ).lower()
    return next(
        (habit for words, habit in _CALL_HABITS if any(word in described for word in words)),
        _DEFAULT_CALL_HABIT,
    )


_CALL_MOVES = (
    "you describe what is going on in your own words rather than naming the fix you think you need",
    "you ask what a term the agent uses means when it is one a person in your place would not know",
    "you ask the what-if your own situation raises",
    "you weigh what the answer costs you in time, money or effort, and say so",
    "before you go, you make sure you know exactly what happens next and what you have to do",
    "you ask why a step is needed",
)


def caller_moves(
    persona: Mapping[str, Any] | None, scenario_name: str = "", variation: str = ""
) -> tuple[str, str]:
    persona = persona if isinstance(persona, Mapping) else {}
    seed = _trial_seed(persona, "moves", scenario_name, variation)
    first = seed % len(_CALL_MOVES)
    second = (first + 1 + seed // len(_CALL_MOVES) % (len(_CALL_MOVES) - 1)) % len(_CALL_MOVES)
    return _CALL_MOVES[first], _CALL_MOVES[second]


def caller_scenario(
    *,
    name: str,
    persona: Mapping[str, Any] | None,
    situation: str,
    fixture: Mapping[str, Any] | None,
    tts_provider: str,
    outcome: str = "",
    initial_message: str = "",
    variation: str = "",
    cartesia_key: str | None = None,
) -> "simulate.Scenario":
    """One simulated caller.

    `outcome` is empty by default: the situation already says what this person wants in their own
    words, and the grading criteria as an objective make the caller recite a checklist.
    """
    persona = dict(persona) if isinstance(persona, Mapping) else {"name": "customer"}
    persona["role"] = "customer"
    provider = (tts_provider or "").lower()
    # A voice from the persona's accent/language, so callers in one suite sound different.
    if not persona.get("voice") and not persona.get("voice_id"):
        if provider == "cartesia":
            persona["voice"] = served_cartesia_voice(persona, cartesia_key)
        elif provider == "deepgram":
            persona["voice"] = aura_voice_for(persona)
    fixture = fixture if isinstance(fixture, Mapping) else {}
    metadata = dict(persona.get("metadata") or {})
    if caller_phone := fixture_caller_phone(fixture):
        # LiveKit exposes this as participant metadata/attributes, so a target hydrates the
        # seeded caller without knowing scenario internals.
        metadata["caller_phone"] = caller_phone
    persona["metadata"] = metadata
    if initial_message.strip():
        persona["initial_message"] = initial_message.strip()
    knowledge = [
        {
            "key": str(key),
            "value": json.dumps(value, ensure_ascii=False, default=str),
            "disclosure": "on_request",
        }
        for key, value in fixture.items()
        # Origins and ``*_case`` values classify the fixture for the harness. They are control
        # metadata, not facts the simulated caller can know or disclose.
        if key != "origin" and not str(key).endswith("_case")
    ]
    return simulate.Scenario(
        name=name or "harness-voice",
        dataset=[
            simulate.Persona(
                persona=persona,
                situation=f"{situation}\n\nIf rule 12g's push gets you nowhere, "
                f"{caller_when_blocked(persona, variation)}. On a call, {caller_habit(persona)}, unless "
                f"you are upset or the situation is urgent, when you act on a clear instruction at "
                f"once. Somewhere in this call, where it fits and the situation allows, "
                f"{'; and '.join(caller_moves(persona, name, variation))}.",
                outcome=outcome,
                knowledge=knowledge,
                behavior_policy=compile_behavior_policy(
                    simulate.Persona(
                        persona={},
                        situation="",
                        outcome="",
                        temperament=caller_temperament(persona, f"{name}|{variation}"),
                    )
                ),
            )
        ],
    )


def simulation_spec(
    *,
    run_id: str,
    room_name: str,
    agent_name: str | None,
    system_prompt: str,
    livekit_url: str,
    recording_dir: Path,
    scenario: "simulate.Scenario",
    simulator: "simulate.SimulatorAgentDefinition",
    direction: str,
    max_seconds: float,
    min_turn_messages: int,
    agent_first_silence_seconds: float,
    run_seconds: float,
    agent_definition: "simulate.AgentDefinition | None" = None,
) -> SimulationSpec:
    """The voice run both lanes execute. Only the target definition differs.

    LiveKit workers use ``agent_name``. Provider-hosted targets (Vapi/Retell) pass an explicit
    definition while retaining the exact same managed LiveKit caller runtime, timeout policy,
    recording behavior, and simulated customer as the local/LiveKit lanes.
    """
    params = {
        "record_audio": True,
        "recording_root": str(recording_dir),
        "recording_case_directory": str(recording_dir),
        "min_turn_messages": min_turn_messages,
        "max_seconds": max_seconds,
        "connect_timeout": CONNECT_TIMEOUT_SECONDS,
        "readiness_timeout": READINESS_TIMEOUT_SECONDS,
        "cleanup_timeout": CLEANUP_TIMEOUT_SECONDS,
        "conversation_direction": direction,
        "agent_first_silence_timeout_seconds": agent_first_silence_seconds,
    }
    agent = agent_definition
    if agent is None:
        if not agent_name:
            raise ValueError("livekit_agent_name_unavailable")
        agent = simulate.AgentDefinition(
            name=_TARGET_NAME,
            agent_name=agent_name,
            system_prompt=system_prompt,
            transport={"kind": "webrtc"},
        )
    runtime = simulate.LiveKitSimulatorRuntime(
        url=livekit_url, room_name=room_name, room_mode="managed"
    )
    return SimulationSpec(
        run_id=run_id,
        environment=EnvironmentSpec(
            adapter="voice",
            world_kind="voice_telephony",
            config={
                "agent_definition": agent.model_dump(mode="json", exclude_none=True),
                "livekit_runtime": runtime.model_dump(mode="json", exclude_none=True),
                "simulator": simulator.model_dump(mode="json", exclude_none=True),
                "params": params,
            },
        ),
        target=AgentEndpointSpec(adapter="webrtc"),
        simulator=SimulatorPolicySpec(adapter="livekit_simulator"),
        scenario=scenario,
        # Keep the execution policy and the engine parameters aligned: disagreement makes
        # planners see the opposite call direction from the engine that actually runs.
        execution=ExecutionPolicy(
            direction=direction, timeout=TimeoutPolicy(run_seconds=run_seconds)
        ),
    )


__all__ = [
    "CARTESIA_DEFAULT_VOICE",
    "persona_speech_rate",
    "persona_emotion",
    "CLEANUP_TIMEOUT_SECONDS",
    "CONNECT_TIMEOUT_SECONDS",
    "READINESS_TIMEOUT_SECONDS",
    "SIMULATOR_INSTRUCTIONS",
    "simulator_instructions",
    "aura_voice_for",
    "caller_scenario",
    "fixture_caller_phone",
    "cartesia_voice_for",
    "persona_stt_language",
    "persona_tts_language",
    "simulation_spec",
    "simulator_definition",
    "transcriber_for",
    "voice_providers",
]
