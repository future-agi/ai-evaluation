"""Caller voice and behaviour settings shared by the local and hosted lanes.

Both lanes build the same simulated customer. Keeping the rules and the provider choice here
means a change lands in both rather than in whichever one the author had open.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from fi import simulate
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

CONNECT_TIMEOUT_SECONDS = 60.0
READINESS_TIMEOUT_SECONDS = 120.0
CLEANUP_TIMEOUT_SECONDS = 30.0

_TARGET_NAME = "harness-livekit-target"
_BEHAVIOR_POLICY = {
    "disclosure_policy": 0.72,
    "cooperation_bounds": 0.9,
    "repair_propensity": 0.85,
}

# Languages transcribed with Deepgram's multilingual model rather than a single language code.
_MULTILINGUAL_STT = ("ar", "es")

# Written as separate numbered rules rather than one paragraph. These arrive late in a long
# prompt, and a rule buried mid-sentence there does not survive: a caller ignored the loop rule
# for four turns while it was the tail of a compound sentence.
SIMULATOR_INSTRUCTIONS = (
    "Act as the customer described by the scenario. Speak naturally and briefly.\n"
    "These rules override anything else when they conflict:\n"
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
    "5. STOP AFTER THREE. Count the agent's replies. If three of them say essentially "
    "the same thing without the task moving forward, do not try a fifth time and do not "
    "rephrase the same point again. Say once that this is not working and you will try "
    "later, then end the call.\n"
    "6. Otherwise let the agent finish speaking. Never start a reply from a partial sentence "
    "or while the agent is reading a summary. Wait for the complete question before answering.\n"
    "7. A quote, proposed action, or booking summary is not a completed outcome. If the agent "
    "asks for final confirmation, answer explicitly, then remain on the call until the agent "
    "confirms that the action actually completed. Do not use goodbye or other closing language "
    "before that confirmation.\n"
    "8. Follow sequence words literally. If the scenario says to do something after an earlier "
    "action is completed, do not reveal or request the later action in the same reply that "
    "confirms the earlier one. Wait until the agent explicitly confirms the earlier action.\n"
    "9. Once the outcome is confirmed, close in ONE turn and end the call. EVERYTHING you still "
    "have to say goes inside that turn: a thanks, a last condition, a reminder, a warning, a "
    "caveat. 'Alright, make sure it stays off the list. Goodbye.' is one closing; 'Goodbye.' "
    "followed by 'Make sure it stays off the list.' is two, and the second one is the tell. Say "
    "your last point BEFORE the farewell, in the same breath, or do not say it at all.\n"
    "10. After your closing turn you say nothing further, whatever the agent says next. Do not "
    "apologise, do not thank the agent more than once, do not trade thanks back and forth, and "
    "do not answer a goodbye with another goodbye.\n"
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
    "courtesies is the clearest sign in a transcript that nobody real was on the line.\n"
    "12c. If you are asked something you have already answered, say that you already gave it, "
    "once, and then give it again. Answering it twice as though it were new is the clearest sign "
    "nobody is really listening on your side either.\n"
    "12d. Your patience for being asked question after question is finite. Once you have answered "
    "roughly ten in a row, say ONCE that you would like to know how many more there are, or how "
    "long this will take, or that you have somewhere to be. Then carry on answering. This is not "
    "refusing to co-operate: it is the single most common thing a real person does on a long form, "
    "and a caller who never does it turns a twenty-minute intake into a transcript nobody can "
    "learn anything from.\n"
    "13. Never say you have done something away from this call that you cannot actually do: "
    "tapped a link, opened an app, read a message that arrived, paid something elsewhere. You are "
    "on a phone call and nothing else. Say plainly that nothing has arrived or that you cannot do "
    "it, and let the agent find another way. Claiming it leaves the agent waiting for a change "
    "that never happens, and the call goes nowhere for both of you."
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
) -> str:
    """The caller's rules, framed by whether this call was placed to them or by them.

    Chat has no direction: a chat is always started by the person, so it takes the inbound text.
    A mailbox answering replaces the rules outright, because it is not a person.
    """
    if str(answered_by).strip().lower() == "voicemail":
        style = str(voicemail_style).strip().lower() or _DEFAULT_VOICEMAIL_STYLE
        if recorded:
            return _VOICEMAIL_RECORDED
        return _VOICEMAIL_INSTRUCTIONS + _VOICEMAIL_BY_STYLE.get(
            style, _VOICEMAIL_BY_STYLE[_DEFAULT_VOICEMAIL_STYLE]
        )
    if str(direction).strip().lower() != "outbound":
        return SIMULATOR_INSTRUCTIONS
    chosen = str(awareness).strip().lower() or _DEFAULT_OUTBOUND_AWARENESS
    return (
        SIMULATOR_INSTRUCTIONS
        + _OUTBOUND_FRAMING
        + _OUTBOUND_AWARENESS.get(chosen, _OUTBOUND_AWARENESS[_DEFAULT_OUTBOUND_AWARENESS])
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

    An explicit override wins; otherwise Cartesia when its key is present (richer, multi-language
    voices), else Deepgram aura. `get` resolves a setting name for the calling lane, which reads
    the environment locally and the run-scoped secrets when hosted.
    """
    keyed = bool((get("CARTESIA_API_KEY") or "").strip())
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


def persona_stt_language(
    persona: Mapping[str, object] | None, override: str = ""
) -> str:
    """The STT language for one caller, from the persona's languages.

    An explicit override always wins. Otherwise the persona's first language is used, so a caller
    who speaks Hindi is transcribed as Hindi rather than forced to English.
    """
    if override and override.strip():
        return override.strip()
    languages = (persona or {}).get("languages") or []
    if isinstance(languages, list) and languages:
        first = str(languages[0]).strip().lower()
        if first in _LANGUAGE_CODES:
            return _LANGUAGE_CODES[first]
        if 2 <= len(first) <= 5 and first.replace("-", "").isalpha():
            return first
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


def cartesia_voice_for(persona: dict) -> str:
    """A stable Cartesia voice id for one caller, chosen by accent/language and gender.

    Deterministic by persona name so a caller keeps its voice across runs while a suite still
    spreads voices. Falls back across gender and to English when a long-tail language lacks one.
    """
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
    if not voices:
        return CARTESIA_DEFAULT_VOICE
    index = sum(ord(character) for character in str(persona.get("name") or "")) % len(
        voices
    )
    return voices[index]


# How fast this person talks. Derived from the persona rather than randomised, so a rerun of the
# same scenario sounds the same -- a rate that moves between runs makes two recordings of one
# scenario incomparable. Cartesia documents 0.6 to 2.0 for sonic-3; this stays close to natural
# because the point is that callers differ from each other, not that any of them sounds odd.
_SPEECH_RATES = (0.9, 0.95, 1.0, 1.05, 1.12)


def persona_speech_rate(persona: Mapping[str, Any] | None) -> float:
    """A stable speech rate for this person.

    Keyed on the same field as the voice, so the two move together: a persona keeps one voice and
    one pace for as long as its name is the same.
    """
    if not isinstance(persona, Mapping):
        return 1.0
    name = str(persona.get("name") or "").strip()
    if not name:
        return 1.0
    return _SPEECH_RATES[sum(ord(character) for character in name) % len(_SPEECH_RATES)]


# The only emotion names and levels sonic-3 accepts, established against the live API. It rejects
# name and level separately with HTTP 400, so nothing outside this set is ever sent.
_CARTESIA_EMOTION_NAMES = frozenset({"anger", "positivity", "surprise", "sadness", "curiosity"})
_CARTESIA_EMOTION_LEVELS = frozenset({"lowest", "low", "high", "highest"})

# What a personality sounds like, as a baseline colour for the whole call. A caller's feeling really
# moves during a call and this control does not, so it is a starting register rather than an arc:
# two personas that read the same on paper stop sounding identical. Anything unrecognised gets no
# control at all, which is the provider default and the behaviour before this existed.
_PERSONALITY_EMOTION = (
    (("warm", "friendly", "cheerful", "enthusiastic", "chatty", "upbeat"), "positivity:high"),
    (("professional", "formal", "businesslike", "direct", "efficient"), "positivity:low"),
    (("irritated", "annoyed", "frustrated", "angry", "impatient", "abrupt"), "anger:low"),
    (("curious", "inquisitive", "questioning", "sceptical", "skeptical"), "curiosity:high"),
    (("anxious", "worried", "nervous", "distressed", "upset", "sad"), "sadness:low"),
)


def persona_emotion(persona: Mapping[str, Any] | None) -> list[str]:
    """The baseline emotional colour for this person, or nothing where none is recognised."""
    if not isinstance(persona, Mapping):
        return []
    described = " ".join(
        str(persona.get(key) or "") for key in ("personality", "communication_style", "traits")
    ).lower()
    for words, emotion in _PERSONALITY_EMOTION:
        if any(word in described for word in words):
            name, _, level = emotion.partition(":")
            if name in _CARTESIA_EMOTION_NAMES and level in _CARTESIA_EMOTION_LEVELS:
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
        dict(persona or {}), (get("SIMULATOR_STT_LANGUAGE") or "").strip()
    )
    stt_default_provider, stt_model, stt_language = transcriber_for(language)
    stt_provider = stt_override or stt_default_provider
    defaults = {
        "llm": {"google": "gemini-2.5-flash", "openai": "gpt-4o-mini"},
        "stt": {"deepgram": stt_model, "cartesia": "ink-2", "google": "chirp_2"},
        "tts": {
            "deepgram": "aura-asteria-en",
            "cartesia": "sonic-3.5",
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
    return simulate.SimulatorAgentDefinition(
        llm={
            "provider": llm_provider,
            "model": model("llm", llm_provider),
            "temperature": float(
                (get("SIMULATOR_LLM_TEMPERATURE") or "").strip() or "0.35"
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
        },
        instructions=simulator_instructions(
            get("HARNESS_CALL_DIRECTION") or "",
            get("HARNESS_CALLER_AWARENESS") or "",
            get("HARNESS_ANSWERED_BY") or "",
            get("HARNESS_VOICEMAIL_STYLE") or "",
            recorded=bool((get("HARNESS_VOICEMAIL_CLIP") or "").strip()),
        ),
        allow_interruptions=True,
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


def caller_scenario(
    *,
    name: str,
    persona: Mapping[str, Any] | None,
    situation: str,
    fixture: Mapping[str, Any] | None,
    tts_provider: str,
    outcome: str = "",
    initial_message: str = "",
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
            persona["voice"] = cartesia_voice_for(persona)
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
        if key != "origin"
    ]
    return simulate.Scenario(
        name=name or "harness-voice",
        dataset=[
            simulate.Persona(
                persona=persona,
                situation=situation,
                outcome=outcome,
                knowledge=knowledge,
                behavior_policy=dict(_BEHAVIOR_POLICY),
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
    "simulation_spec",
    "simulator_definition",
    "transcriber_for",
    "voice_providers",
]
