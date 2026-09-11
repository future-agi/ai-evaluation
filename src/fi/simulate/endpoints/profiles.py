"""Target-endpoint profiles (canonical plan §4.1) — the factory that retires the
hardcoded ``transport.kind`` branches.

A profile is the single declarative record for one target adapter: its capability
manifest plus the *decisions* the voice engine used to make with ``if
transport.kind == ...`` chains — whether the leg is SIP, whether it places an
outbound call, whether it needs a web audio bridge, which provider evidence to
collect, and which connector to build. The engine keeps its session/audio
*mechanics*; it just asks the profile instead of branching on strings.

Adding a provider = registering one profile here (or via the
``fi.simulate.endpoints`` entry-point group) with **zero engine edits**.

Import weight: connector / ``livekit.rtc`` imports are deferred into
``build_connector`` so the planner can read ``profile.manifest.capabilities``
without pulling the optional voice stack.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from fi.simulate.endpoints.actor_sources import (
    resolve_factory,
    resolve_framework,
    resolve_http,
    resolve_import_object,
    resolve_python_callable,
    resolve_system_prompt,
)
from fi.simulate.endpoints.base import AgentEndpointManifest
from fi.simulate.registry import register_endpoint
from fi.simulate.runtime.capabilities import EndpointCapabilities


class EndpointProfile:
    """One target adapter's manifest + provider decisions."""

    def __init__(
        self,
        manifest: AgentEndpointManifest,
        *,
        is_sip: bool = False,
        places_outbound_call: bool = False,
        receives_inbound_call: bool = False,
        uses_web_audio_bridge: bool = False,
        bridge_provider: Optional[str] = None,
        evidence_provider: Optional[str] = None,
        required_env_rule: Optional[Callable[[Any], list[str]]] = None,
        connector_builder: Optional[Callable[..., Any]] = None,
        target_resolver: Optional[Callable[..., Any]] = None,
        runs_caller_code: bool = True,
    ) -> None:
        self.manifest = manifest
        self.is_sip = is_sip
        self.places_outbound_call = places_outbound_call
        self.receives_inbound_call = receives_inbound_call
        self.uses_web_audio_bridge = uses_web_audio_bridge
        self.bridge_provider = bridge_provider
        self.evidence_provider = evidence_provider
        self._required_env_rule = required_env_rule
        self._connector_builder = connector_builder
        self._target_resolver = target_resolver
        # Deny-by-default: a turn-based target executes caller-supplied Python
        # unless it explicitly declares otherwise (http/system_prompt/deployed
        # endpoints). An undeclared third-party kind is treated as unsafe. The
        # hosted gate (resolve_chat_target) reads this, not a module denylist.
        self.runs_caller_code = runs_caller_code

    @property
    def is_turn_based_target(self) -> bool:
        """This adapter resolves to a turn-based agent (chat) rather than a
        realtime/voice endpoint."""
        return self._target_resolver is not None

    def resolve_target(
        self, config: Any = None, secret_refs: Any = None, *, hosted: bool = False
    ) -> Any:
        """Resolve this actor-source to a runnable turn-based agent (plan §4.1).

        Raises for realtime/voice adapters, which are reached through
        ``build_connector`` + the engine instead. ``hosted`` restricts env reads
        to job-provisioned secrets (see resolvers)."""
        if self._target_resolver is None:
            raise ValueError(
                f"target_adapter_not_turn_based: {self.manifest.name!r}"
            )
        return self._target_resolver(config or {}, secret_refs, hosted=hosted)

    @property
    def joins_as_sip_participant(self) -> bool:
        """The target joins the room as a SIP participant — true for real SIP
        legs and for the web providers reached through the LiveKit audio bridge."""
        return self.is_sip or self.uses_web_audio_bridge

    @property
    def uses_external_room(self) -> bool:
        """Only plain WebRTC targets run in an external (non-managed) room."""
        return not self.joins_as_sip_participant

    @property
    def name(self) -> str:
        return self.manifest.name

    def required_env(self, agent_definition: Any) -> list[str]:
        """Transport-specific env var *names* this adapter needs (the branch that
        used to live in ``voice._voice_required_env``). Base adapters need none."""
        if self._required_env_rule is None:
            return []
        return list(self._required_env_rule(agent_definition))

    def build_connector(
        self, provider_target: Any, *, conversation_direction: str
    ) -> Any:
        """Provider connector for web-bridged targets; ``None`` for webrtc/sip."""
        if self._connector_builder is None:
            return None
        return self._connector_builder(
            provider_target, conversation_direction=conversation_direction
        )


# --------------------------------------------------------------------------- #
# required-env rules (parity with the old voice._voice_required_env branches)
# --------------------------------------------------------------------------- #
def _vapi_web_required_env(agent_definition: Any) -> list[str]:
    if agent_definition.target is None:
        return ["VAPI_API_KEY", "VAPI_ASSISTANT_ID"]
    return []


def _retell_web_required_env(agent_definition: Any) -> list[str]:
    if agent_definition.target is None:
        return ["RETELL_API_KEY", "RETELL_AGENT_ID"]
    return []


def _sip_inbound_required_env(agent_definition: Any) -> list[str]:
    transport = agent_definition.transport
    target = agent_definition.target
    names: list[str] = []
    if transport is not None and not transport.dispatch_rule_name:
        names.append("LIVEKIT_INBOUND_TRUNK_ID")
    if transport is not None and transport.inbound_call_originator == "vapi":
        # Same conditional-append idiom as the Retell branch below; behaviour
        # identical (the consumer already dropped blanks from the old tuple).
        names.append(
            target.api_key_env
            if target is not None and target.provider == "vapi"
            else "VAPI_API_KEY"
        )
        if target is None or target.provider != "vapi":
            names.append("VAPI_ASSISTANT_ID")
        names.append("VAPI_PHONE_NUMBER_ID")
        names.append("LIVEKIT_INBOUND_DID")
    if transport is not None and transport.inbound_call_originator == "retell":
        # Parity-only rule (CLI/manifest path); the job carries these fields directly.
        names.append("RETELL_API_KEY")
        if not transport.originator_agent_id:
            names.append("RETELL_AGENT_ID")
        if not transport.originator_from_number:
            names.append("RETELL_FROM_NUMBER")
        names.append("LIVEKIT_INBOUND_DID")
    return names


# --------------------------------------------------------------------------- #
# connector builders (lazy imports keep the voice stack off the planner path)
# --------------------------------------------------------------------------- #
def _build_vapi_connector(provider_target: Any, *, conversation_direction: str) -> Any:
    from fi.simulate.agent.definition import VapiTargetConfig
    from fi.simulate.simulation.bridge import VapiWebSocketConnector

    if isinstance(provider_target, VapiTargetConfig):
        return VapiWebSocketConnector.from_target(
            provider_target,
            first_message_mode=(
                "assistant-waits-for-user"
                if conversation_direction == "simulator_first"
                else "assistant-speaks-first"
            ),
        )
    return VapiWebSocketConnector.from_env()


def _build_retell_connector(
    provider_target: Any, *, conversation_direction: str
) -> Any:
    from fi.simulate.agent.definition import RetellTargetConfig
    from fi.simulate.simulation.bridge import RetellWebCallConnector

    if isinstance(provider_target, RetellTargetConfig):
        return RetellWebCallConnector.from_target(provider_target)
    return RetellWebCallConnector.from_env()


# --------------------------------------------------------------------------- #
# built-in profiles
# --------------------------------------------------------------------------- #
_CHAT_CAPS = EndpointCapabilities(text=True, transcript_events=True, tool_events=True)
_WEBRTC_CAPS = EndpointCapabilities(
    audio=True,
    streaming=True,
    interruption=True,
    recording=True,
    transcript_events=True,
    web_rtc=True,
)
_SIP_CAPS = EndpointCapabilities(
    audio=True,
    streaming=True,
    interruption=True,
    recording=True,
    transcript_events=True,
    sip=True,
)

_PROFILES: list[EndpointProfile] = [
    # chat-era + actor-source target adapters ("drop in any agent", plan §4.1).
    # config keys mirror the manifest agent: block (target/factory/args/kwargs/
    # method/input_mode/system_prompt/...).
    EndpointProfile(
        AgentEndpointManifest(
            name="callable", provider="callable",
            world_kinds=["chat", "text"], capabilities=_CHAT_CAPS,
        ),
        target_resolver=resolve_python_callable,
    ),
    EndpointProfile(
        AgentEndpointManifest(
            name="python_callable", provider="callable",
            world_kinds=["chat", "text"], capabilities=_CHAT_CAPS,
        ),
        target_resolver=resolve_python_callable,
    ),
    EndpointProfile(
        AgentEndpointManifest(
            name="import_object", provider="python",
            world_kinds=["chat", "text"], capabilities=_CHAT_CAPS,
        ),
        target_resolver=resolve_import_object,
    ),
    EndpointProfile(
        AgentEndpointManifest(
            name="factory", provider="python",
            world_kinds=["chat", "text"], capabilities=_CHAT_CAPS,
        ),
        target_resolver=resolve_factory,
    ),
    EndpointProfile(
        AgentEndpointManifest(
            name="framework", provider="framework",
            world_kinds=["chat", "text"], capabilities=_CHAT_CAPS,
        ),
        target_resolver=resolve_framework,
    ),
    EndpointProfile(
        AgentEndpointManifest(
            name="system_prompt", provider="llm",
            world_kinds=["chat", "text"], capabilities=_CHAT_CAPS,
        ),
        target_resolver=resolve_system_prompt,
        runs_caller_code=False,
    ),
    EndpointProfile(
        AgentEndpointManifest(
            name="http", provider="http",
            world_kinds=["chat", "text"], capabilities=_CHAT_CAPS,
        ),
        target_resolver=resolve_http,
        runs_caller_code=False,
    ),
    EndpointProfile(
        AgentEndpointManifest(
            name="websocket", provider="websocket",
            world_kinds=["chat", "text"],
            capabilities=EndpointCapabilities(
                text=True, streaming=True, transcript_events=True, tool_events=True
            ),
        )
    ),
    # voice target adapters (keyed by transport.kind)
    EndpointProfile(
        AgentEndpointManifest(
            name="webrtc", provider="livekit",
            world_kinds=["voice"], capabilities=_WEBRTC_CAPS,
        )
    ),
    EndpointProfile(
        AgentEndpointManifest(
            name="livekit", provider="livekit",
            world_kinds=["voice"], capabilities=_WEBRTC_CAPS,
        )
    ),
    EndpointProfile(
        AgentEndpointManifest(
            name="vapi_websocket", provider="vapi",
            world_kinds=["voice"], capabilities=_WEBRTC_CAPS,
        ),
        uses_web_audio_bridge=True,
        bridge_provider="vapi",
        evidence_provider="vapi",
        required_env_rule=_vapi_web_required_env,
        connector_builder=_build_vapi_connector,
    ),
    EndpointProfile(
        AgentEndpointManifest(
            name="retell_webcall", provider="retell",
            world_kinds=["voice"], capabilities=_WEBRTC_CAPS,
        ),
        uses_web_audio_bridge=True,
        bridge_provider="retell",
        evidence_provider="retell",
        required_env_rule=_retell_web_required_env,
        connector_builder=_build_retell_connector,
    ),
    EndpointProfile(
        AgentEndpointManifest(
            name="sip_outbound", provider="sip",
            world_kinds=["voice"], capabilities=_SIP_CAPS,
        ),
        is_sip=True,
        places_outbound_call=True,
    ),
    EndpointProfile(
        AgentEndpointManifest(
            name="sip_inbound", provider="sip",
            world_kinds=["voice"], capabilities=_SIP_CAPS,
        ),
        is_sip=True,
        receives_inbound_call=True,
        required_env_rule=_sip_inbound_required_env,
    ),
]

for _profile in _PROFILES:
    register_endpoint(_profile.name, _profile)


def get_profile(name: str) -> Optional[EndpointProfile]:
    """Return the registered profile for a target adapter name, or ``None``."""
    from fi.simulate.registry import endpoint_registry

    value = endpoint_registry.get_or_none(name)
    return value if isinstance(value, EndpointProfile) else None


__all__ = ["EndpointProfile", "get_profile"]
