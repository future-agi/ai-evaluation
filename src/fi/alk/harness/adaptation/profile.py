"""The agent being adapted: its inspected profile, per-tool admission and binding identity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..provider_import import (
    ProviderImportError,
    ToolAdmission,
    _safe_profile_value,
    admit_provider_tools,
    admitted_tool_definitions,
    inspect_provider_target,
    provider_profile_digest,
)
from ..provider_lifecycle import ProviderType
from ..releases import AgentRef, BindingMode, ToolDefinition, seal_agent_binding
from .errors import AdaptationFailure

ADAPTABLE_CONNECTORS = ("retell_chat", "retell", "vapi")
_TARGET_CREDENTIALS = {
    ProviderType.RETELL: "RETELL_API_KEY",
    ProviderType.VAPI: "VAPI_API_KEY",
}


@dataclass(frozen=True)
class AgentProfile:
    connector: str
    provider: ProviderType
    target: dict[str, Any]
    admission: list[ToolAdmission]
    agent_tools: list[dict[str, Any]]
    agent: AgentRef

    def document(self) -> dict[str, Any]:
        return {
            "agent": self.agent.model_dump(mode="json"),
            "connector": self.connector,
            "profile": self.target,
            "admission": [one.model_dump(mode="json") for one in self.admission],
            "agent_tools": self.agent_tools,
        }

    @property
    def admitted(self) -> list[ToolAdmission]:
        return [one for one in self.admission if one.action in {"routed", "replaced"}]


def provider_of(connector: str) -> ProviderType:
    if connector not in ADAPTABLE_CONNECTORS:
        raise AdaptationFailure(
            "compose_unsupported",
            f"adaptation supports {', '.join(ADAPTABLE_CONNECTORS)} agents, not {connector}",
        )
    return ProviderType.VAPI if connector == "vapi" else ProviderType.RETELL


def build_profile(
    target: Mapping[str, Any], *, connector: str, agent: Mapping[str, Any] | None
) -> AgentProfile:
    """Admission over a (re-redacted) provider profile; `agent` is the platform's AgentRef.

    Without one (a local run from a profile file) the agent is identified by the profile itself:
    its configuration digest is the canonical digest of the redacted profile.
    """
    provider = provider_of(connector)
    redacted = _safe_profile_value(dict(target))
    try:
        admission = admit_provider_tools(provider, redacted)
        definitions = admitted_tool_definitions(provider, redacted)
    except ProviderImportError as exc:
        code = str(exc).split(":", 1)[0]
        raise AdaptationFailure(code or "provider_import_failed", str(exc)) from exc
    target_id = str(redacted.get("source_target_id") or "").strip()
    if agent is None:
        local = f"local:{target_id or 'agent'}"
        digest = provider_profile_digest(redacted)
        agent = {
            "agent_definition_id": local,
            "agent_version_id": local,
            "configuration_digest": digest,
            "agent_config_digest": digest,
            "connector": connector,
        }
    try:
        reference = AgentRef.model_validate(
            {
                **dict(agent),
                "provider_target_id": target_id,
                "provider_digest": provider_profile_digest(redacted),
            }
        )
    except ValueError as exc:
        raise AdaptationFailure(
            "adaptation_job_invalid", f"agent reference: {exc}"
        ) from exc
    if reference.connector != connector:
        raise AdaptationFailure(
            "adaptation_job_invalid",
            f"agent reference connector {reference.connector} differs from {connector}",
        )
    return AgentProfile(
        connector=connector,
        provider=provider,
        target=redacted,
        admission=admission,
        agent_tools=definitions,
        agent=reference,
    )


def profile_from_file(body: Mapping[str, Any]) -> AgentProfile:
    """A saved `inspect_provider_target` result, optionally carrying `agent` (an AgentRef)."""
    target = dict(body)
    agent = target.pop("agent", None)
    provider = str(target.get("provider") or "")
    if provider == "vapi":
        connector = "vapi"
    elif provider == "retell":
        connector = "retell_chat" if target.get("modality") == "chat" else "retell"
    else:
        raise AdaptationFailure(
            "adaptation_profile_invalid",
            f"profile provider {provider!r} is not retell or vapi",
        )
    return build_profile(target, connector=connector, agent=agent)


def profile_from_job(
    job: Any, *, credentials: Mapping[str, str], request: Any = None
) -> AgentProfile:
    """Inspect the job's provider target read-only, with the target's own API key."""
    connector = job.agent.connector.strip().lower()
    provider = provider_of(connector)
    adaptation = (job.metadata or {}).get("adaptation") or {}
    agent = adaptation.get("agent") if isinstance(adaptation, Mapping) else None
    if not isinstance(agent, Mapping):
        raise AdaptationFailure(
            "adaptation_job_invalid", "job metadata.adaptation.agent is missing"
        )
    target_key = "assistant_id" if provider is ProviderType.VAPI else "agent_id"
    target_id = str(job.agent.config.get(target_key) or "").strip()
    try:
        target = inspect_provider_target(
            provider,
            source_target_id=target_id,
            api_key=credentials.get(_TARGET_CREDENTIALS[provider], ""),
            target_modality="chat" if connector == "retell_chat" else "voice",
            request=request,
        )
    except ProviderImportError as exc:
        raise AdaptationFailure("provider_inspect_failed", str(exc)) from exc
    return build_profile(target, connector=connector, agent=agent)


def select_mode(
    profile: AgentProfile, environment_tools: Sequence[ToolDefinition]
) -> BindingMode:
    """§7.5 step 2: canonical when every admitted tool is an environment tool the agent can call
    as-is, same name and compatible arguments; anything else needs a routed binding."""
    admitted = {one.name for one in profile.admitted}
    if not admitted:
        return BindingMode.CONVERSATION_ONLY
    environment = {tool.name: tool.parameters for tool in environment_tools}
    schemas = {
        str(one.get("name")): one.get("parameters") or {} for one in profile.agent_tools
    }
    if all(
        name in environment
        and not argument_mismatches(schemas.get(name) or {}, environment[name])
        for name in admitted
    ):
        return BindingMode.CANONICAL
    return BindingMode.TOOL_ROUTED


def argument_mismatches(
    agent: Mapping[str, Any], environment: Mapping[str, Any]
) -> list[str]:
    """Why calls the agent makes with this schema would not be valid environment calls.

    Every argument the agent sends must be one the environment takes, with the same JSON type,
    and only values the environment accepts; every argument the environment requires must be
    one the agent can send.
    """
    sent = agent.get("properties") or {}
    taken = environment.get("properties") or {}
    problems = [
        f"{name} is not an environment argument" for name in sent if name not in taken
    ]
    problems += [
        f"{name} is required but the agent never sends it"
        for name in environment.get("required") or []
        if name not in sent
    ]
    for name, schema in sent.items():
        target = taken.get(name)
        if not isinstance(target, Mapping) or not isinstance(schema, Mapping):
            continue
        if (
            schema.get("type")
            and target.get("type")
            and schema["type"] != target["type"]
        ):
            problems.append(
                f"{name} is {schema['type']}, the environment takes {target['type']}"
            )
        allowed = target.get("enum")
        offered = schema.get("enum")
        if allowed and (not offered or set(map(str, offered)) - set(map(str, allowed))):
            problems.append(f"{name} may carry values the environment does not accept")
    return problems


def binding_fields(
    profile: AgentProfile, mode: BindingMode, mapping: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """A complete binding document minus what the sealer sets; mapping only for tool_routed."""
    mapping = mapping or {}
    routed = mode is BindingMode.TOOL_ROUTED
    return {
        "agent": profile.agent.model_dump(mode="json"),
        "mode": mode.value,
        "agent_tools": profile.agent_tools if routed else [],
        "tools": list(mapping.get("tools") or []) if routed else [],
        "fixtures": dict(mapping.get("fixtures") or {}) if routed else {},
        "aliases": dict(mapping.get("aliases") or {}) if routed else {},
        "rendering": dict(mapping.get("rendering") or {}),
        "prompt_changes": [],
    }


def seal_binding(
    profile: AgentProfile,
    mode: BindingMode,
    mapping: Mapping[str, Any] | None,
    *,
    environment_root: Any,
) -> Any:
    return seal_agent_binding(
        binding_fields(profile, mode, mapping),
        environment_root=environment_root,
        agent_tools=[one.name for one in profile.admitted],
    )
