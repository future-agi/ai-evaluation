from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

import aiohttp

try:
    from livekit.agents import llm as livekit_llm
    from livekit.agents import stt as livekit_stt
    from livekit.agents import tts as livekit_tts
except ImportError as exc:
    raise ImportError(
        "LiveKit model construction requires the 'livekit' optional dependency"
    ) from exc

from fi.simulate.agent.definition import LLMConfig, STTConfig, TTSConfig


@dataclass
class LiveKitModels:
    stt: livekit_stt.STT
    llm: livekit_llm.LLM
    tts: livekit_tts.TTS
    http_session: aiohttp.ClientSession | None = None

    async def aclose(self) -> None:
        if self.http_session is not None and not self.http_session.closed:
            await self.http_session.close()


STTFactory = Callable[[STTConfig, aiohttp.ClientSession | None], livekit_stt.STT]
LLMFactory = Callable[[LLMConfig], livekit_llm.LLM]
TTSFactory = Callable[[TTSConfig, aiohttp.ClientSession | None], livekit_tts.TTS]


def _import_plugin(name: str) -> ModuleType:
    try:
        import importlib

        return importlib.import_module(f"livekit.plugins.{name}")
    except ImportError:
        raise ImportError(
            f"livekit-plugins-{name} is not installed. "
            f"Install it with: pip install livekit-plugins-{name}"
        ) from None


def _openai_llm(config: LLMConfig) -> livekit_llm.LLM:
    openai = _import_plugin("openai")
    kwargs: dict[str, object] = {
        "model": config.model,
        "temperature": config.temperature,
    }
    api_key = os.environ.get("SIMULATOR_LLM_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    base_url = os.environ.get("SIMULATOR_LLM_BASE_URL") or os.environ.get(
        "OPENAI_BASE_URL"
    )
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")
    header_name = os.environ.get("SIMULATOR_LLM_API_KEY_HEADER")
    if api_key and header_name:
        kwargs["extra_headers"] = {header_name: api_key}
    return openai.LLM(**kwargs)


def _openai_stt(
    config: STTConfig,
    _http_session: aiohttp.ClientSession | None,
) -> livekit_stt.STT:
    openai = _import_plugin("openai")
    return openai.STT(
        model=config.model,
        language=config.language or "en",
    )


def _elevenlabs_stt(
    config: STTConfig,
    http_session: aiohttp.ClientSession | None,
) -> livekit_stt.STT:
    elevenlabs = _import_plugin("elevenlabs")
    return elevenlabs.STT(
        api_key=_required_env("ELEVEN_API_KEY", "ELEVENLABS_API_KEY"),
        http_session=http_session,
        model_id=_provider_model(
            config.model,
            default="gpt-4o-mini-transcribe",
            replacement="scribe_v2_realtime",
        ),
        server_vad={
            "vad_silence_threshold_secs": 0.8,
            "vad_threshold": 0.4,
            "min_speech_duration_ms": 100,
            "min_silence_duration_ms": 500,
        },
    )


def _deepgram_stt(
    config: STTConfig,
    http_session: aiohttp.ClientSession | None,
) -> livekit_stt.STT:
    deepgram = _import_plugin("deepgram")
    return deepgram.STT(
        api_key=_required_env("DEEPGRAM_API_KEY"),
        http_session=http_session,
        model=_provider_model(
            config.model,
            default="gpt-4o-mini-transcribe",
            replacement="nova-3",
        ),
        language=config.language or "en-US",
    )


def _cartesia_stt(
    config: STTConfig,
    http_session: aiohttp.ClientSession | None,
) -> livekit_stt.STT:
    cartesia = _import_plugin("cartesia")
    return cartesia.STT(
        api_key=_required_env("CARTESIA_API_KEY"),
        http_session=http_session,
        model=_provider_model(
            config.model,
            default="gpt-4o-mini-transcribe",
            replacement="ink-2",
        ),
        language=config.language or "en",
    )


def _openai_tts(
    config: TTSConfig,
    _http_session: aiohttp.ClientSession | None,
) -> livekit_tts.TTS:
    openai = _import_plugin("openai")
    return openai.TTS(model=config.model, voice=config.voice)


def _elevenlabs_tts(
    config: TTSConfig,
    http_session: aiohttp.ClientSession | None,
) -> livekit_tts.TTS:
    elevenlabs = _import_plugin("elevenlabs")
    return elevenlabs.TTS(
        api_key=_required_env("ELEVEN_API_KEY", "ELEVENLABS_API_KEY"),
        http_session=http_session,
        model=_provider_model(
            config.model,
            default="gpt-4o-mini-tts",
            replacement="eleven_flash_v2_5",
        ),
        voice_id=config.voice,
    )


def _deepgram_tts(
    config: TTSConfig,
    http_session: aiohttp.ClientSession | None,
) -> livekit_tts.TTS:
    deepgram = _import_plugin("deepgram")
    return deepgram.TTS(
        api_key=_required_env("DEEPGRAM_API_KEY"),
        http_session=http_session,
        model=_provider_model(
            config.model,
            default="gpt-4o-mini-tts",
            replacement="aura-2-andromeda-en",
        ),
    )


def _cartesia_tts(
    config: TTSConfig,
    http_session: aiohttp.ClientSession | None,
) -> livekit_tts.TTS:
    cartesia = _import_plugin("cartesia")
    voice = (
        config.voice
        if config.voice not in {"alloy", ""}
        else "f786b574-daa5-4673-aa0c-cbe3e8534c02"
    )
    # Both controls ride on Cartesia's ``__experimental_controls``. Verified against the live API
    # on sonic-3: it validates the emotion NAME and the LEVEL separately and rejects either being
    # wrong with HTTP 400, so a bad value fails the call rather than being ignored.
    #
    # Note the livekit plugin's own ``TTSVoiceEmotion`` type lists a different vocabulary
    # ("Neutral", "Frustrated", "Tired"), and every one of those is rejected by this API version.
    # Only the "<name>:<level>" form works, which is why the values are built here from a
    # validated set rather than taken from the plugin's types.
    speed = getattr(config, "speed", None)
    emotion = getattr(config, "emotion", None)
    return cartesia.TTS(
        api_key=_required_env("CARTESIA_API_KEY"),
        http_session=http_session,
        model=_provider_model(
            config.model,
            default="gpt-4o-mini-tts",
            replacement="sonic-3",
        ),
        voice=voice,
        **({"speed": float(speed)} if isinstance(speed, (int, float)) else {}),
        **({"emotion": list(emotion)} if emotion else {}),
    )


def _google_credentials_kwargs() -> dict[str, object]:
    """Pick Vertex AI vs Gemini API from env — Vertex when possible.

    Vertex is preferred: it has higher throughput and uses ADC so the
    key never lives in the SDK process. Falls back to the direct Gemini
    API when only ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) is set.
    """

    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("VERTEX_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get(
        "VERTEX_LOCATION",
        "us-central1",
    )
    credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if project and credentials:
        return {"vertexai": True, "project": project, "location": location}
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return {"vertexai": False, "api_key": api_key}
    raise ValueError(
        "google_credentials_missing: set GOOGLE_APPLICATION_CREDENTIALS + "
        "GOOGLE_CLOUD_PROJECT for Vertex or GEMINI_API_KEY for Gemini API"
    )


def _google_speech_credentials_kwargs() -> dict[str, object]:
    credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials:
        return {"credentials_file": credentials}
    raise ValueError(
        "google_cloud_speech_credentials_missing: set "
        "GOOGLE_APPLICATION_CREDENTIALS for Google STT/TTS"
    )


def _google_llm(config: LLMConfig) -> livekit_llm.LLM:
    google = _import_plugin("google")
    kwargs = _google_credentials_kwargs()
    model = _provider_model(
        config.model,
        default="gpt-4o",
        replacement="gemini-2.5-flash-lite",
    )
    if (
        kwargs.get("vertexai") is True
        and model.startswith("gemini-3")
        and not os.environ.get("GOOGLE_CLOUD_LOCATION")
        and not os.environ.get("VERTEX_LOCATION")
    ):
        kwargs["location"] = "global"
    thinking = _simulator_thinking(model)
    if thinking is not None:
        kwargs["thinking_config"] = thinking
    return google.LLM(model=model, temperature=config.temperature, **kwargs)


# Which control a model accepts for deliberation is a provider fact and cannot be inferred from a
# version inside its name. Gemini 2.5 and earlier take a token budget and reject a level outright;
# Gemini 3 takes a level, and the LiveKit plugin answers a budget it cannot use by substituting its
# own "minimal", which Vertex rejects. A model in neither list is left alone deliberately: running
# at the provider's own default costs some latency, where sending a control it rejects costs every
# inference and leaves the caller silent for the whole call with nothing in the transcript to say
# why. So a model nobody has characterised yet still holds a conversation.
_THINKING_BY_BUDGET = ("gemini-1.5", "gemini-2.0", "gemini-2.5")
_THINKING_BY_LEVEL = ("gemini-3",)
# The least a level-taking model will accept. "minimal" exists in the plugin but Vertex refuses it.
_LEAST_THINKING_LEVEL = "low"


def _simulator_thinking(model: str) -> dict[str, object] | None:
    """How much the simulated caller deliberates before answering.

    A person on a phone call answers from what they already know, so thinking buys nothing here and
    is charged twice: once in latency the target hears as an unnatural pause, and again in a call
    whose duration no longer reflects how the conversation actually went. As near off as the model
    allows. `SIMULATOR_LLM_THINKING` takes a thinking level, or a token budget as a number, and an
    explicit request is honoured even on a model this module does not recognise.
    """
    asked = os.environ.get("SIMULATOR_LLM_THINKING", "").strip().lower()
    off = asked in ("", "off", "0", "none", "false")
    if model.startswith(_THINKING_BY_LEVEL):
        return {"thinking_level": _LEAST_THINKING_LEVEL if off or asked.isdigit() else asked}
    if model.startswith(_THINKING_BY_BUDGET):
        if off:
            return {"thinking_budget": 0}
        return {"thinking_budget": int(asked)} if asked.isdigit() else {"thinking_level": asked}
    if off:
        return None
    return {"thinking_budget": int(asked)} if asked.isdigit() else {"thinking_level": asked}


def _google_stt(
    config: STTConfig,
    _http_session: aiohttp.ClientSession | None,
) -> livekit_stt.STT:
    google = _import_plugin("google")
    kwargs = _google_speech_credentials_kwargs()
    languages = [
        language.strip()
        for language in (config.language or "en-US").split(",")
        if language.strip()
    ]
    # Non-streaming mode lets AgentSession's VAD-backed StreamAdapter defer
    # the Google request until speech exists. This avoids Google's 400
    # "Long duration elapsed without audio" during agent-first quiet gaps.
    return google.STT(
        languages=languages or ["en-US"],
        use_streaming=False,
        **kwargs,
    )


def _google_tts(
    config: TTSConfig,
    _http_session: aiohttp.ClientSession | None,
) -> livekit_tts.TTS:
    from google.cloud import texttospeech

    google = _import_plugin("google")
    kwargs = _google_speech_credentials_kwargs()
    voice = (
        config.voice if config.voice not in {"alloy", ""} else "en-US-Chirp3-HD-Kore"
    )
    language = "-".join(voice.split("-")[:2]) if "-" in voice else "en-US"
    return google.TTS(
        voice_name=voice,
        language=language,
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        **kwargs,
    )


def _gemini_tts(
    config: TTSConfig,
    _http_session: aiohttp.ClientSession | None,
) -> livekit_tts.TTS:
    """Streaming Gemini TTS over the genai/Vertex endpoint (multilingual).

    Cloud TTS (``_google_tts``) needs the Text-to-Speech API the Vertex SA
    lacks; the genai speech endpoint works with the same ADC and streams audio
    for gemini-3.1, keeping time-to-first-byte low.
    """
    from fi.simulate.simulation.gemini_tts_stream import StreamingGeminiTTS

    kwargs = _google_credentials_kwargs()
    model = _provider_model(
        config.model,
        default="gpt-4o-mini-tts",
        replacement="gemini-3.1-flash-tts-preview",
    )
    if not model.startswith("gemini-"):
        model = "gemini-3.1-flash-tts-preview"
    voice = config.voice if config.voice not in {"alloy", ""} else "Kore"
    if kwargs.get("vertexai") is True:
        location = kwargs.get("location")
        if model.startswith("gemini-3"):
            location = "global"
        return StreamingGeminiTTS(
            model=model,
            voice_name=voice,
            vertexai=True,
            project=kwargs.get("project"),
            location=location,
        )
    return StreamingGeminiTTS(model=model, voice_name=voice, api_key=kwargs["api_key"])


_LLM_FACTORIES: dict[str, LLMFactory] = {
    "openai": _openai_llm,
    "openai_compatible": _openai_llm,
    "google": _google_llm,
    "vertex": _google_llm,
    "gemini": _google_llm,
}
_STT_FACTORIES: dict[str, STTFactory] = {
    "openai": _openai_stt,
    "elevenlabs": _elevenlabs_stt,
    "deepgram": _deepgram_stt,
    "cartesia": _cartesia_stt,
    "google": _google_stt,
    "vertex": _google_stt,
}
_TTS_FACTORIES: dict[str, TTSFactory] = {
    "openai": _openai_tts,
    "elevenlabs": _elevenlabs_tts,
    "deepgram": _deepgram_tts,
    "cartesia": _cartesia_tts,
    "google": _google_tts,
    "vertex": _google_tts,
    "gemini": _gemini_tts,
    "gemini_tts": _gemini_tts,
}
_HTTP_PROVIDERS = {"cartesia", "deepgram", "elevenlabs"}


def build_livekit_llm(config: LLMConfig) -> livekit_llm.LLM:
    provider = config.provider.lower()
    factory = _factory(_LLM_FACTORIES, provider, "LLM")
    return factory(config)


async def build_livekit_models(
    *,
    llm_config: LLMConfig,
    stt_config: STTConfig,
    tts_config: TTSConfig,
) -> LiveKitModels:
    stt_provider = stt_config.provider.lower()
    tts_provider = tts_config.provider.lower()
    stt_factory = _factory(_STT_FACTORIES, stt_provider, "STT")
    tts_factory = _factory(_TTS_FACTORIES, tts_provider, "TTS")
    http_session = (
        aiohttp.ClientSession()
        if {stt_provider, tts_provider} & _HTTP_PROVIDERS
        else None
    )
    try:
        return LiveKitModels(
            stt=stt_factory(stt_config, http_session),
            llm=build_livekit_llm(llm_config),
            tts=tts_factory(tts_config, http_session),
            http_session=http_session,
        )
    except Exception:
        if http_session is not None:
            await http_session.close()
        raise


def _factory(factories: dict[str, Callable], provider: str, model_type: str):
    factory = factories.get(provider)
    if factory is None:
        supported = ", ".join(sorted(factories))
        raise ValueError(
            f"Unsupported LiveKit {model_type} provider: {provider!r}. "
            f"Supported: {supported}"
        )
    return factory


def _required_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    raise ValueError(f"Missing provider credential: {' or '.join(names)}")


def _provider_model(configured: str, *, default: str, replacement: str) -> str:
    return replacement if configured == default else configured
