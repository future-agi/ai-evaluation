from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from fi.simulate.agent.definition import AgentDefinition
from fi.simulate.simulation.models import Scenario

from ._download import (
    _config,
    _field,
    _headers,
    _rows,
    ScenarioDownloadError,
    fetch_dataset_rows,
    hydrate_platform_scenario,
    validate_download,
)
from ._scan import DownloadRejected, scan_content

_AGENT_LIST_PATH = "/simulate/agent-definitions/"
_AGENT_CREATE_PATH = "/simulate/agent-definitions/create/"
_AGENT_VERSION_CREATE_PATH = "/simulate/agent-definitions/{agent_id}/versions/create/"
_SCENARIO_LIST_PATH = "/simulate/scenarios/"
_SCENARIO_CREATE_PATH = "/simulate/scenarios/create/"
_SCENARIO_DETAIL_PATH = "/simulate/scenarios/{scenario_id}/"
_TERMINAL_FAILURE_STATUSES = {"failed", "error", "cancelled"}


class ScenarioGenerationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        scenario_id: str | None = None,
        status: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.scenario_id = scenario_id
        self.status = status
        self.retryable = retryable


@dataclass(frozen=True)
class PlatformAgentReference:
    agent_definition_id: str
    agent_version_id: str
    configuration_hash: str
    reused: bool


@dataclass(frozen=True)
class PlatformScenarioRequest:
    name: str
    agent_definition: AgentDefinition | None = None
    platform_agent_definition_id: str | None = None
    platform_agent_version_id: str | None = None
    description: str | None = None
    custom_instruction: str | None = None
    kind: Literal["graph"] = "graph"
    no_of_rows: int = 10
    poll_interval_seconds: float = 2.0
    timeout_seconds: float = 900.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("scenario name must be non-empty")
        if self.kind != "graph":
            raise ValueError("only graph scenario generation is supported")
        if not 10 <= self.no_of_rows <= 20_000:
            raise ValueError("no_of_rows must be between 10 and 20000")
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than zero")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        has_local = self.agent_definition is not None
        has_platform = bool(self.platform_agent_definition_id)
        if has_local == has_platform:
            raise ValueError(
                "provide exactly one of agent_definition or platform_agent_definition_id"
            )
        if self.platform_agent_version_id and not has_platform:
            raise ValueError(
                "platform_agent_version_id requires platform_agent_definition_id"
            )


@dataclass(frozen=True)
class GeneratedScenario:
    scenario: Scenario
    platform_agent_definition_id: str
    platform_agent_version_id: str | None
    platform_scenario_id: str
    platform_dataset_id: str
    platform_status: str
    polling_duration_seconds: float
    checksum_sha256: str


def _request_json(
    url: str,
    headers: Mapping[str, str],
    *,
    method: str = "GET",
    payload: Mapping[str, Any] | None = None,
    timeout: float = 30.0,
) -> Any:
    body = None
    request_headers = dict(headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url,
        data=body,
        headers=request_headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ScenarioGenerationError(
            f"Future AGI API request failed with HTTP {exc.code}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ScenarioGenerationError(
            "Future AGI API request could not be completed",
            retryable=True,
        ) from exc


def _required_config(config: Any | None) -> Any:
    resolved = _config(config)
    if not resolved.api_key or not resolved.secret_key:
        raise ScenarioGenerationError(
            "Future AGI API and secret keys are required for scenario generation"
        )
    return resolved


def _safe_livekit_url(value: object) -> str:
    parsed = urllib.parse.urlsplit(str(value))
    if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
        raise ScenarioGenerationError("LiveKit agent URL must use ws:// or wss://")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ScenarioGenerationError(
            "LiveKit agent URL must not contain credentials, query parameters, or fragments"
        )
    host = parsed.hostname
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urllib.parse.urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def _agent_payload(agent_definition: AgentDefinition) -> tuple[dict[str, Any], str]:
    transport = agent_definition.transport
    transport_kind = transport.kind if transport else "webrtc"
    inbound = transport_kind != "sip_inbound"
    description = agent_definition.system_prompt
    scan = scan_content({"description": description})
    if scan["status"] == "flagged":
        raise ScenarioGenerationError(
            "agent description was rejected by the local content and secret scan"
        )

    safe_configuration: dict[str, Any] = {
        "name": agent_definition.name,
        "description": description,
        "transport": transport_kind,
        "inbound": inbound,
        "model": agent_definition.llm.model,
        "model_provider": agent_definition.llm.provider,
        "language": agent_definition.stt.language,
    }
    identity_configuration: dict[str, Any] = {"transport": transport_kind}
    payload: dict[str, Any] = {
        "agent_type": "voice",
        "commit_message": "Created by Agent Learning Kit for scenario generation",
        "description": description,
        "inbound": inbound,
        "language": agent_definition.stt.language,
        "languages": (
            [agent_definition.stt.language] if agent_definition.stt.language else None
        ),
        "model": agent_definition.llm.model,
        "model_details": {
            "provider": agent_definition.llm.provider,
            "temperature": agent_definition.llm.temperature,
        },
    }
    target = agent_definition.target
    if target is not None:
        target_id = (
            target.assistant_id if target.provider == "vapi" else target.agent_id
        )
        target_url = (
            target.api_base_url if target.provider == "vapi" else target.api_url
        )
        payload.update(
            {
                "provider": target.provider,
                "assistant_id": target_id,
            }
        )
        safe_configuration.update(
            {
                "provider": target.provider,
                "assistant_id": target_id,
                "provider_api_url": str(target_url),
            }
        )
        identity_configuration.update(
            {"provider": target.provider, "assistant_id": target_id}
        )
    elif transport_kind in {"sip_outbound", "sip_inbound"}:
        if transport is None or not transport.sip_call_to:
            raise ScenarioGenerationError(
                "SIP platform agent creation requires a target contact number"
            )
        payload.update(
            {
                "provider": "others",
                "contact_number": transport.sip_call_to,
            }
        )
        safe_configuration["contact_number"] = transport.sip_call_to
        identity_configuration.update(
            {"provider": "others", "contact_number": transport.sip_call_to}
        )
    else:
        if agent_definition.url is None:
            raise ScenarioGenerationError(
                "LiveKit target creation requires an AgentDefinition url"
            )
        livekit_url = _safe_livekit_url(agent_definition.url)
        livekit_agent_name = agent_definition.agent_name or agent_definition.name
        payload.update(
            {
                "provider": "livekit",
                "livekit_url": livekit_url,
                "livekit_agent_name": livekit_agent_name,
            }
        )
        safe_configuration.update(
            {
                "livekit_url": livekit_url,
                "livekit_agent_name": livekit_agent_name,
            }
        )
        identity_configuration.update(
            {
                "provider": "livekit",
                "livekit_url": livekit_url,
                "livekit_agent_name": livekit_agent_name,
            }
        )

    configuration_hash = hashlib.sha256(
        json.dumps(
            safe_configuration,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    identity_hash = hashlib.sha256(
        json.dumps(
            identity_configuration,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    base_name = "-".join(agent_definition.name.strip().split()) or "agent"
    payload["agent_name"] = f"{base_name[:220]}-alk-{identity_hash[:12]}"
    payload["commit_message"] = _configuration_commit(configuration_hash)
    return {
        key: value for key, value in payload.items() if value is not None
    }, configuration_hash


def _configuration_commit(configuration_hash: str) -> str:
    return f"Agent Learning Kit configuration {configuration_hash}"


def _active_version(detail: Mapping[str, Any]) -> Mapping[str, Any] | None:
    active = detail.get("active_version")
    if isinstance(active, Mapping) and active.get("id"):
        return active
    versions = detail.get("versions")
    if isinstance(versions, list):
        return next(
            (
                version
                for version in versions
                if isinstance(version, Mapping) and version.get("id")
            ),
            None,
        )
    return None


def _active_version_id(detail: Mapping[str, Any]) -> str | None:
    active = _active_version(detail)
    if active is not None:
        return str(active["id"])
    latest = detail.get("latest_version_id")
    return str(latest) if latest else None


def _logical_agent_match(
    item: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> bool:
    if str(_field(item, "agent_name") or "") == str(payload["agent_name"]):
        return True
    provider = str(payload.get("provider") or "")
    assistant_id = payload.get("assistant_id")
    if assistant_id:
        return bool(
            str(_field(item, "provider") or "") == provider
            and str(_field(item, "assistant_id") or "") == str(assistant_id)
        )
    contact_number = payload.get("contact_number")
    return bool(
        contact_number
        and str(_field(item, "provider") or "") == provider
        and str(_field(item, "contact_number") or "") == str(contact_number)
    )


def ensure_platform_agent(
    agent_definition: AgentDefinition,
    *,
    config: Any | None = None,
) -> PlatformAgentReference:
    cfg = _required_config(config)
    headers = _headers(cfg)
    base = str(cfg.api_url).rstrip("/")
    payload, configuration_hash = _agent_payload(agent_definition)
    search = payload.get("assistant_id") or agent_definition.name
    query = urllib.parse.urlencode(
        {"search": search, "agent_type": "voice", "limit": 100}
    )
    listing = _request_json(f"{base}{_AGENT_LIST_PATH}?{query}", headers)
    existing = next(
        (
            item
            for item in _rows(listing)
            if isinstance(item, Mapping) and _logical_agent_match(item, payload)
        ),
        None,
    )
    if existing is not None:
        agent_id = str(_field(existing, "id") or "")
        if not agent_id:
            raise ScenarioGenerationError(
                "reused platform Agent Definition has no agent id"
            )
        detail = _request_json(f"{base}{_AGENT_LIST_PATH}{agent_id}/", headers)
        active = _active_version(detail) if isinstance(detail, Mapping) else None
        if active is None:
            raise ScenarioGenerationError(
                "reused platform Agent Definition has no active version"
            )
        version_id = str(active["id"])
        if str(active.get("commit_message") or "") == _configuration_commit(
            configuration_hash
        ):
            return PlatformAgentReference(
                agent_definition_id=agent_id,
                agent_version_id=version_id,
                configuration_hash=configuration_hash,
                reused=True,
            )
        created_version = _request_json(
            f"{base}{_AGENT_VERSION_CREATE_PATH.format(agent_id=agent_id)}",
            headers,
            method="POST",
            payload=payload,
        )
        version = (
            created_version.get("version")
            if isinstance(created_version, Mapping)
            else None
        )
        version_id = (
            str(_field(version, "id") or "")
            if isinstance(version, Mapping)
            else ""
        )
        if not version_id:
            raise ScenarioGenerationError(
                "platform Agent Version response did not include a version id"
            )
        return PlatformAgentReference(
            agent_definition_id=agent_id,
            agent_version_id=version_id,
            configuration_hash=configuration_hash,
            reused=False,
        )

    created = _request_json(
        f"{base}{_AGENT_CREATE_PATH}",
        headers,
        method="POST",
        payload=payload,
    )
    agent = created.get("agent") if isinstance(created, Mapping) else None
    agent_id = str(_field(agent, "id") or "") if isinstance(agent, Mapping) else ""
    if not agent_id:
        raise ScenarioGenerationError(
            "platform Agent Definition response did not include an agent id"
        )
    detail = _request_json(f"{base}{_AGENT_LIST_PATH}{agent_id}/", headers)
    version_id = _active_version_id(detail) if isinstance(detail, Mapping) else None
    if not version_id:
        raise ScenarioGenerationError(
            "created platform Agent Definition has no active version"
        )
    return PlatformAgentReference(
        agent_definition_id=agent_id,
        agent_version_id=version_id,
        configuration_hash=configuration_hash,
        reused=False,
    )


def _extract_scenario(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    scenario = payload.get("scenario")
    if isinstance(scenario, Mapping):
        return scenario
    result = payload.get("result")
    if isinstance(result, Mapping):
        nested = result.get("scenario")
        return nested if isinstance(nested, Mapping) else result
    return payload


def _completed_scenario(
    detail: Mapping[str, Any],
    *,
    agent_definition_id: str,
    agent_version_id: str | None,
    polling_duration_seconds: float,
    base: str,
    headers: Mapping[str, str],
) -> GeneratedScenario:
    scenario_id = str(_field(detail, "id") or "")
    dataset_id = str(_field(detail, "dataset_id") or _field(detail, "dataset") or "")
    if not dataset_id:
        raise ScenarioGenerationError(
            "completed platform Scenario has no dataset id",
            scenario_id=scenario_id,
            status=str(_field(detail, "status") or "Completed"),
        )
    try:
        rows = fetch_dataset_rows(base, headers, dataset_id)
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        ScenarioDownloadError,
    ) as exc:
        raise ScenarioGenerationError(
            "completed platform Scenario dataset could not be retrieved",
            scenario_id=scenario_id,
            status=str(_field(detail, "status") or "Completed"),
            retryable=True,
        ) from exc
    artifact = {
        "id": scenario_id,
        "updated_at": _field(detail, "updated_at"),
        "scenario": dict(detail),
        "rows": rows,
    }
    try:
        pin = validate_download(
            artifact,
            source=urllib.parse.urlsplit(base).netloc or base,
        )
    except DownloadRejected as exc:
        raise ScenarioGenerationError(
            "generated platform Scenario was rejected by the local content scan",
            scenario_id=scenario_id,
            status=str(_field(detail, "status") or "Completed"),
        ) from exc
    scenario = hydrate_platform_scenario(detail, rows, pin=pin)
    return GeneratedScenario(
        scenario=scenario,
        platform_agent_definition_id=agent_definition_id,
        platform_agent_version_id=agent_version_id,
        platform_scenario_id=scenario_id,
        platform_dataset_id=dataset_id,
        platform_status=str(_field(detail, "status") or "Completed"),
        polling_duration_seconds=polling_duration_seconds,
        checksum_sha256=str(pin["checksum_sha256"]),
    )


def fetch_scenario(
    scenario_id: str,
    *,
    platform_agent_definition_id: str = "",
    platform_agent_version_id: str | None = None,
    poll_interval_seconds: float = 2.0,
    timeout_seconds: float = 900.0,
    config: Any | None = None,
) -> GeneratedScenario:
    if not scenario_id.strip():
        raise ValueError("scenario_id must be non-empty")
    if poll_interval_seconds <= 0 or timeout_seconds <= 0:
        raise ValueError("poll interval and timeout must be greater than zero")
    cfg = _required_config(config)
    headers = _headers(cfg)
    base = str(cfg.api_url).rstrip("/")
    started = time.monotonic()
    while True:
        raw = _request_json(
            f"{base}{_SCENARIO_DETAIL_PATH.format(scenario_id=scenario_id)}",
            headers,
        )
        detail = _extract_scenario(raw)
        status = str(_field(detail, "status") or "").strip()
        normalized_status = status.lower()
        elapsed = time.monotonic() - started
        if normalized_status == "completed":
            return _completed_scenario(
                detail,
                agent_definition_id=platform_agent_definition_id,
                agent_version_id=platform_agent_version_id,
                polling_duration_seconds=elapsed,
                base=base,
                headers=headers,
            )
        if normalized_status in _TERMINAL_FAILURE_STATUSES:
            raise ScenarioGenerationError(
                f"platform Scenario ended with status {status}",
                scenario_id=scenario_id,
                status=status,
            )
        if elapsed >= timeout_seconds:
            raise ScenarioGenerationError(
                "platform Scenario generation timed out; resume with fetch_scenario",
                scenario_id=scenario_id,
                status=status or "Processing",
                retryable=True,
            )
        time.sleep(min(poll_interval_seconds, timeout_seconds - elapsed))


def generate_scenario(
    request: PlatformScenarioRequest,
    *,
    config: Any | None = None,
) -> GeneratedScenario:
    cfg = _required_config(config)
    if request.agent_definition is not None:
        agent = ensure_platform_agent(request.agent_definition, config=cfg)
        agent_definition_id = agent.agent_definition_id
        agent_version_id: str | None = agent.agent_version_id
    else:
        agent_definition_id = str(request.platform_agent_definition_id)
        agent_version_id = request.platform_agent_version_id

    payload: dict[str, Any] = {
        "name": request.name.strip(),
        "kind": request.kind,
        "generate_graph": True,
        "agent_definition_id": agent_definition_id,
        "no_of_rows": request.no_of_rows,
    }
    if agent_version_id:
        payload["agent_definition_version_id"] = agent_version_id
    if request.description is not None:
        payload["description"] = request.description
    if request.custom_instruction is not None:
        payload["custom_instruction"] = request.custom_instruction
    if scan_content(payload)["status"] == "flagged":
        raise ScenarioGenerationError(
            "scenario generation request was rejected by the local content and secret scan"
        )

    headers = _headers(cfg)
    base = str(cfg.api_url).rstrip("/")
    query = urllib.parse.urlencode(
        {
            "search": request.name.strip(),
            "agent_definition_id": agent_definition_id,
            "limit": 100,
        }
    )
    listing = _request_json(f"{base}{_SCENARIO_LIST_PATH}?{query}", headers)
    existing = next(
        (
            item
            for item in _rows(listing)
            if str(_field(item, "name") or "") == request.name.strip()
        ),
        None,
    )
    if existing is not None:
        scenario_id = str(_field(existing, "id") or "")
        if scenario_id:
            return fetch_scenario(
                scenario_id,
                platform_agent_definition_id=agent_definition_id,
                platform_agent_version_id=agent_version_id,
                poll_interval_seconds=request.poll_interval_seconds,
                timeout_seconds=request.timeout_seconds,
                config=cfg,
            )
    created = _request_json(
        f"{base}{_SCENARIO_CREATE_PATH}",
        headers,
        method="POST",
        payload=payload,
    )
    scenario = _extract_scenario(created)
    scenario_id = str(_field(scenario, "id") or "")
    if not scenario_id:
        raise ScenarioGenerationError(
            "platform Scenario response did not include a scenario id"
        )
    return fetch_scenario(
        scenario_id,
        platform_agent_definition_id=agent_definition_id,
        platform_agent_version_id=agent_version_id,
        poll_interval_seconds=request.poll_interval_seconds,
        timeout_seconds=request.timeout_seconds,
        config=cfg,
    )


__all__ = [
    "GeneratedScenario",
    "PlatformAgentReference",
    "PlatformScenarioRequest",
    "ScenarioGenerationError",
    "ensure_platform_agent",
    "fetch_scenario",
    "generate_scenario",
]
