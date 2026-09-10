import re
from typing import Annotated, Literal, Optional

from pydantic import AnyHttpUrl, AnyUrl, BaseModel, Field, model_validator


# Always .fullmatch, never .match: "$" also matches immediately before a
# trailing newline, so .match alone would let "+14155551234\n" through.
_E164 = re.compile(r"^\+[1-9]\d{6,14}$")


class ProviderEvidenceConfig(BaseModel):
    """Post-call provider-side evidence collection (Vapi/Retell).

    After the phone leg completes, the SDK optionally queries the
    provider's API to enrich the canonical report with their own
    transcripts, recordings, tool calls, and latency. Read the
    provider's call ID either from a LiveKit SIP participant attribute
    (``participant_attribute``) or from a bounded polling window over
    the provider's list-calls endpoint (``polling_window``).
    """

    provider: Literal["vapi", "retell"] = Field(
        ...,
        description="Provider whose evidence to collect after the call.",
    )
    call_id_source: Literal[
        "participant_attribute", "polling_window", "originator_response"
    ] = Field(
        "participant_attribute",
        description="How to key the provider's call ID.",
    )
    participant_attribute: Optional[str] = Field(
        None,
        description="LiveKit participant attribute name (e.g. sip.callID).",
    )
    polling_window_seconds: Optional[float] = Field(
        None,
        gt=0,
        description="Window before + after case start for list-calls matching.",
    )
    poll_interval_seconds: float = Field(
        3.0,
        gt=0,
        description="Interval between provider status polls.",
    )
    poll_deadline_seconds: float = Field(
        60.0,
        gt=0,
        description="Total time to poll the provider before giving up.",
    )

    @model_validator(mode="after")
    def _check_source(self) -> "ProviderEvidenceConfig":
        if self.call_id_source == "participant_attribute":
            if not self.participant_attribute or not self.participant_attribute.strip():
                raise ValueError(
                    "participant_attribute source requires a non-empty attribute name"
                )
        elif self.call_id_source == "polling_window":
            if not self.polling_window_seconds:
                raise ValueError(
                    "polling_window source requires polling_window_seconds"
                )
        return self


class TelephonyTransport(BaseModel):
    """Optional telephony transport for a LiveKit-backed target.

    Default (or omitted): WebRTC — the SDK connects to the room over WS
    and the target is dispatched as a registered agent, unchanged.

    ``sip_outbound``: the SDK creates the per-case room and dials
    ``sip_call_to`` through ``sip_trunk_id``; the target answers on the
    phone. Requires ``sip_trunk_id`` and E.164 ``sip_call_to``.

    ``sip_inbound``: the SDK does not dial; a dispatch rule routes an
    incoming call into the per-case room. If ``dispatch_rule_name`` is
    set the SDK verifies and reuses that rule; otherwise it provisions
    a per-run rule and tears it down on cleanup.
    """

    kind: Literal[
        "webrtc", "sip_outbound", "sip_inbound", "vapi_websocket", "retell_webcall"
    ] = Field(
        "webrtc",
        description="Transport used to reach the target participant.",
    )
    sip_trunk_id: Optional[str] = Field(
        None,
        description="LiveKit outbound SIP trunk ID (sip_outbound only).",
    )
    sip_call_to: Optional[str] = Field(
        None,
        description="E.164 phone number to dial (sip_outbound only).",
    )
    sip_number: Optional[str] = Field(
        None,
        description="E.164 originating caller ID (sip_outbound only).",
    )
    participant_identity: Optional[str] = Field(
        None,
        description=(
            "Template for the SIP participant identity. May contain {test_case_id}, "
            "{run_id}, or {invocation_id}. The default includes invocation and case IDs."
        ),
    )
    dispatch_rule_name: Optional[str] = Field(
        None,
        description="Dispatch rule that routes the inbound call (sip_inbound only).",
    )
    readiness_timeout_seconds: Optional[float] = Field(
        None,
        gt=0,
        description="Seconds to wait for the inbound SIP participant to appear.",
    )
    answer_timeout_seconds: Optional[float] = Field(
        None,
        gt=0,
        description="Seconds to wait for an outbound SIP call to be answered.",
    )
    inbound_call_originator: Literal["vapi", "retell"] | None = Field(
        None,
        description="Provider that originates an inbound SIP call after room readiness.",
    )
    originator_agent_id: Optional[str] = Field(
        None,
        description=(
            "Provider agent/assistant ID to run for the originated call "
            "(sip_inbound with the retell originator only)."
        ),
    )
    originator_from_number: Optional[str] = Field(
        None,
        description=(
            "E.164 number the originator dials from "
            "(sip_inbound with the retell originator only)."
        ),
    )

    @model_validator(mode="after")
    def _check_kind_fields(self) -> "TelephonyTransport":
        if self.kind == "sip_outbound":
            if self.inbound_call_originator is not None:
                raise ValueError("sip_outbound cannot set inbound_call_originator")
            if (
                self.originator_agent_id is not None
                or self.originator_from_number is not None
            ):
                raise ValueError("sip_outbound cannot set originator fields")
            if not self.sip_trunk_id or not self.sip_trunk_id.strip():
                raise ValueError("sip_outbound requires sip_trunk_id")
            if not self.sip_call_to or not _E164.fullmatch(self.sip_call_to):
                raise ValueError(
                    "sip_outbound requires E.164 sip_call_to (e.g. +14155551234)"
                )
            if not self.sip_number or not _E164.fullmatch(self.sip_number):
                raise ValueError(
                    "sip_outbound requires E.164 sip_number (e.g. +14155551234)"
                )
        elif self.kind == "sip_inbound":
            if (
                self.dispatch_rule_name is not None
                and not self.dispatch_rule_name.strip()
            ):
                raise ValueError(
                    "sip_inbound dispatch_rule_name must be non-empty when set"
                )
            if (
                self.originator_agent_id is not None
                or self.originator_from_number is not None
            ) and self.inbound_call_originator is None:
                raise ValueError("originator fields require inbound_call_originator")
            # C1: the two originator fields are Retell-only; a Vapi originator reads
            # its config from env and would otherwise silently ignore them.
            if (
                self.inbound_call_originator is not None
                and self.inbound_call_originator != "retell"
                and (
                    self.originator_agent_id is not None
                    or self.originator_from_number is not None
                )
            ):
                raise ValueError(
                    f"{self.inbound_call_originator}_originator_does_not_take_originator_fields"
                )
            # Emptiness is only meaningful once an originator is set (None stays forward-compatible).
            if (
                self.inbound_call_originator is not None
                and self.originator_agent_id is not None
                and not self.originator_agent_id.strip()
            ):
                raise ValueError("originator_agent_id must be non-empty when set")
            if self.originator_from_number is not None and not _E164.fullmatch(
                self.originator_from_number
            ):
                raise ValueError(
                    "originator_from_number must be E.164 (e.g. +14155551234)"
                )
        elif self.kind in {"webrtc", "vapi_websocket", "retell_webcall"}:
            if any(
                [
                    self.sip_trunk_id,
                    self.sip_call_to,
                    self.sip_number,
                    self.dispatch_rule_name,
                    self.inbound_call_originator,
                    self.originator_agent_id,
                    self.originator_from_number,
                ]
            ):
                raise ValueError(f"{self.kind} transport cannot set SIP fields")
        return self


class VapiTargetConfig(BaseModel):
    """Non-secret configuration for a Vapi assistant under test."""

    provider: Literal["vapi"] = "vapi"
    assistant_id: str = Field(..., min_length=1)
    api_base_url: AnyHttpUrl = Field(
        "https://api.vapi.ai",
        validate_default=True,
    )
    api_key_env: str = Field(
        "VAPI_API_KEY",
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )


class RetellTargetConfig(BaseModel):
    """Non-secret configuration for a Retell agent under test."""

    provider: Literal["retell"] = "retell"
    agent_id: str = Field(..., min_length=1)
    api_url: AnyHttpUrl = Field(
        "https://api.retellai.com/v2/create-web-call",
        validate_default=True,
    )
    livekit_url: AnyUrl = Field(
        "wss://retell-ai-4ihahnq7.livekit.cloud",
        validate_default=True,
    )
    api_key_env: str = Field(
        "RETELL_API_KEY",
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )

    @model_validator(mode="after")
    def _check_livekit_url(self) -> "RetellTargetConfig":
        if self.livekit_url.scheme not in {"ws", "wss"}:
            raise ValueError("retell_livekit_url_invalid: URL must use ws:// or wss://")
        return self


VoiceProviderTarget = Annotated[
    VapiTargetConfig | RetellTargetConfig,
    Field(discriminator="provider"),
]


class LiveKitSimulatorRuntime(BaseModel):
    """FutureAGI-owned LiveKit runtime for the simulator and bridge."""

    url: AnyUrl = Field(..., description="FutureAGI LiveKit WebSocket URL.")
    room_name: str = Field(..., min_length=1)
    room_mode: Literal["external", "managed"] = "managed"
    room_name_verbatim: bool = Field(
        False,
        description=(
            "When True, use room_name exactly as provided (no invocation/case "
            "suffix). Required when routing to a pre-existing dispatch rule that "
            "binds to a fixed room. A multi-persona run is allowed when cases run "
            "one at a time (max_concurrency 1; always the case for SIP transports)."
        ),
    )
    api_key_env: str = Field(
        "LIVEKIT_API_KEY",
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )
    api_secret_env: str = Field(
        "LIVEKIT_API_SECRET",
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )

    @model_validator(mode="after")
    def _check_url(self) -> "LiveKitSimulatorRuntime":
        if self.url.scheme not in {"ws", "wss"}:
            raise ValueError("livekit_url_invalid: URL must use ws:// or wss://")
        return self


class LLMConfig(BaseModel):
    """Configuration for the simulator language model."""

    provider: str = Field("openai", description="The LiveKit LLM provider.")
    model: str = Field("gpt-4o", description="The language model to use.")
    temperature: float = Field(
        0.7, ge=0.0, le=2.0, description="Controls randomness in the LLM's output."
    )


class TTSConfig(BaseModel):
    """Configuration for simulator text-to-speech."""

    provider: str = Field("openai", description="The LiveKit TTS provider.")
    model: str = Field("gpt-4o-mini-tts", description="The TTS model to use.")
    voice: str = Field("alloy", description="The voice or voice ID to use.")


class STTConfig(BaseModel):
    """Configuration for simulator speech-to-text."""

    provider: str = Field("openai", description="The LiveKit STT provider.")
    model: str = Field(
        "gpt-4o-mini-transcribe",
        description="The STT model to use.",
    )
    language: Optional[str] = Field("en", description="The transcription language.")


class VADConfig(BaseModel):
    """Configuration for Voice Activity Detection (VAD)."""

    provider: str = Field(
        "silero", description="The VAD provider to use. 'silero' is recommended."
    )
    min_silence_duration: float = Field(
        0.1,
        description="Minimum duration of silence to consider as the end of a speech segment.",
    )
    speech_pad_ms: int = Field(
        200,
        description="Additional padding in milliseconds to add to the end of a speech segment.",
    )


class AgentDefinition(BaseModel):
    """
    The core configuration for a voice AI agent.
    """

    name: str = Field(..., description="A unique name for the target agent.")
    description: Optional[str] = Field(
        None,
        description="A safe description of the target agent's purpose and capabilities.",
    )
    system_prompt: str = Field(
        ...,
        description="Current system prompt or instructions of the target agent.",
    )
    target: VoiceProviderTarget | None = Field(
        None,
        description="Non-secret provider configuration for a direct target agent.",
    )
    agent_name: Optional[str] = Field(
        None,
        description="Exact registered LiveKit target agent name used for managed dispatch.",
    )
    dispatch_metadata: Optional[dict] = Field(
        None,
        description=(
            "Metadata sent to the target agent's LiveKit dispatch. Default None "
            "sends EMPTY metadata: agents built from LiveKit templates treat any "
            "job metadata as an outbound/no-greet call and stop publishing audio, "
            "so injecting simulation context makes readiness time out. Set this "
            "only for agents that are built to consume dispatch metadata."
        ),
    )
    target_participant_identity: Optional[str] = Field(
        None,
        description="Exact target participant identity when it is known in advance.",
    )
    transport: Optional[TelephonyTransport] = Field(
        None,
        description="Transport used to reach the target agent.",
    )
    provider_evidence: Optional[ProviderEvidenceConfig] = Field(
        None,
        description=(
            "Optional post-call provider evidence collection "
            "(Vapi/Retell). None = SDK-observed evidence only."
        ),
    )
    url: AnyUrl | None = Field(
        None,
        description=(
            "Legacy FutureAGI LiveKit URL. Use LiveKitSimulatorRuntime for new "
            "voice simulations."
        ),
    )
    room_name: str | None = Field(
        None,
        description=(
            "Legacy FutureAGI LiveKit room template. Use LiveKitSimulatorRuntime "
            "for new voice simulations."
        ),
    )
    room_mode: Literal["external", "managed"] = Field(
        "external",
        description=(
            "Legacy FutureAGI LiveKit room lifecycle setting. Use "
            "LiveKitSimulatorRuntime for new voice simulations."
        ),
    )

    @model_validator(mode="after")
    def _check_transport(self) -> "AgentDefinition":
        if self.url is not None and self.url.scheme not in {"ws", "wss"}:
            raise ValueError("livekit_url_invalid: URL must use ws:// or wss://")
        transport = self.transport
        if (
            self.url is not None
            and transport is not None
            and transport.kind != "webrtc"
            and self.room_mode != "managed"
        ):
            raise ValueError("managed_transport_requires_managed_room")
        expected_target_provider = (
            {
                "vapi_websocket": "vapi",
                "retell_webcall": "retell",
            }.get(transport.kind)
            if transport is not None
            else None
        )
        if (
            expected_target_provider is not None
            and self.target is not None
            and self.target.provider != expected_target_provider
        ):
            raise ValueError(
                f"{transport.kind}_requires_{expected_target_provider}_target"
            )
        if self.target is not None:
            expected_transport = {
                "vapi": "vapi_websocket",
                "retell": "retell_webcall",
            }[self.target.provider]
            if transport is None or transport.kind != expected_transport:
                raise ValueError(
                    f"{self.target.provider}_target_requires_{expected_transport}"
                )
        evidence = self.provider_evidence
        originator = (
            transport.inbound_call_originator if transport is not None else None
        )
        if originator is not None:
            # Templated on the originator name so Vapi's strings stay byte-identical.
            if transport.kind != "sip_inbound":
                raise ValueError(f"{originator}_originator_requires_sip_inbound")
            if evidence is None or evidence.provider != originator:
                raise ValueError(
                    f"{originator}_originator_requires_{originator}_evidence"
                )
            if evidence.call_id_source != "originator_response":
                raise ValueError(
                    f"{originator}_originator_requires_originator_response"
                )
        if evidence is not None and transport is not None:
            if evidence.provider == "retell" and transport.kind == "sip_outbound":
                raise ValueError(
                    "retell_pstn_outbound_unsupported: Retell has no outbound "
                    "phone API; use sip_inbound or a different provider"
                )
            web_provider = {
                "vapi_websocket": "vapi",
                "retell_webcall": "retell",
            }.get(transport.kind)
            if web_provider is not None:
                if evidence.provider != web_provider:
                    raise ValueError(
                        f"{transport.kind}_requires_{web_provider}_evidence"
                    )
                if evidence.call_id_source != "originator_response":
                    raise ValueError(f"{transport.kind}_requires_originator_response")
        return self

    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    initial_message: str = Field(
        "Hello! How can I help you today?",
        description="The first message the agent speaks to start the conversation.",
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "name": "vapi-support-agent",
                "description": "Customer-support voice assistant.",
                "system_prompt": "Copy the current target-agent prompt here.",
                "target": {
                    "provider": "vapi",
                    "assistant_id": "assistant-id",
                    "api_key_env": "VAPI_API_KEY",
                },
                "transport": {"kind": "vapi_websocket"},
            }
        }


class SimulatorAgentDefinition(BaseModel):
    """
    Configuration for the simulated customer persona agent used by the TestRunner.

    This is intentionally separate from the deployed AgentDefinition so tests can
    run with lightweight/cheaper models and different voice/transcription settings.
    """

    name: Optional[str] = Field(
        None, description="Optional label for the simulator agent"
    )
    instructions: Optional[str] = Field(
        None,
        description=(
            "Optional policy appended to the scenario-derived simulator prompt. "
            "It never replaces persona, situation, or outcome instructions."
        ),
    )

    llm: LLMConfig = Field(
        default_factory=lambda: LLMConfig(model="gpt-4o-mini", temperature=0.6)
    )
    tts: TTSConfig = Field(default_factory=TTSConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    vad: VADConfig = Field(default_factory=VADConfig)

    allow_interruptions: Optional[bool] = Field(
        None,
        description="Whether the simulator agent allows interruptions during TTS.",
    )
    min_endpointing_delay: Optional[float] = Field(
        None,
        description="Minimum endpointing delay (s) to declare end of user turn.",
    )
    max_endpointing_delay: Optional[float] = Field(
        None,
        description="Maximum endpointing delay (s) to force end of user turn.",
    )
    use_tts_aligned_transcript: Optional[bool] = Field(
        None,
        description="Whether to use TTS-aligned transcript as transcription source.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "simulator-customer",
                "instructions": "You are a concise customer. Ask clarifying questions and confirm resolution.",
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "temperature": 0.6,
                },
                "tts": {
                    "provider": "openai",
                    "model": "gpt-4o-mini-tts",
                    "voice": "alloy",
                },
                "stt": {
                    "provider": "openai",
                    "model": "gpt-4o-mini-transcribe",
                    "language": "en",
                },
                "vad": {"provider": "silero"},
                "allow_interruptions": True,
                "min_endpointing_delay": 0.3,
                "max_endpointing_delay": 4.0,
                "use_tts_aligned_transcript": False,
            }
        }
