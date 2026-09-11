from __future__ import annotations

from google.genai.errors import APIError, ClientError, ServerError
from livekit.agents import APIConnectionError, APIStatusError, tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.plugins.google.beta.gemini_tts import TTS as _BaseGeminiTTS

from google.genai import types


class _StreamingChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: "StreamingGeminiTTS",
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        opts = self._tts._opts
        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=opts.voice_name,
                    )
                )
            ),
        )
        input_text = self._input_text
        if opts.instructions is not None:
            input_text = f'{opts.instructions}:\n"{input_text}"'

        try:
            stream = await self._tts._client.aio.models.generate_content_stream(
                model=opts.model,
                contents=input_text,
                config=config,
            )
            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=self._tts.sample_rate,
                num_channels=self._tts.num_channels,
                mime_type="audio/pcm",
            )
            got_audio = False
            async for chunk in stream:
                for candidate in chunk.candidates or []:
                    content = candidate.content
                    if not content or not content.parts:
                        continue
                    for part in content.parts:
                        inline = part.inline_data
                        if (
                            inline
                            and inline.data
                            and inline.mime_type
                            and inline.mime_type.startswith("audio/")
                        ):
                            output_emitter.push(inline.data)
                            got_audio = True
            if not got_audio:
                raise APIStatusError("gemini tts: no audio content generated")
        except ClientError as e:
            raise APIStatusError(
                "gemini tts: client error",
                status_code=e.code,
                body=f"{e.message} {e.status}",
                retryable=e.code in {429, 499},
            ) from e
        except ServerError as e:
            raise APIStatusError(
                "gemini tts: server error",
                status_code=e.code,
                body=f"{e.message} {e.status}",
                retryable=True,
            ) from e
        except APIError as e:
            raise APIStatusError(
                "gemini tts: api error",
                status_code=e.code,
                body=f"{e.message} {e.status}",
                retryable=True,
            ) from e
        except Exception as e:
            raise APIConnectionError(f"gemini tts: {e}", retryable=True) from e


class StreamingGeminiTTS(_BaseGeminiTTS):
    """Gemini TTS that streams audio as it is generated (low time-to-first-byte).

    Reuses the beta plugin's Vertex/genai client and options, but consumes
    ``generate_content_stream`` and pushes frames as they arrive instead of
    buffering the whole utterance. Matches what livekit-plugins-google 1.6.x
    does internally, without the whole-stack bump.
    """

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return _StreamingChunkedStream(
            tts=self, input_text=text, conn_options=conn_options
        )
