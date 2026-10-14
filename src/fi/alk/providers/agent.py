from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .setup_hash import compute_setup_hash

AGENT_SCHEMA_VERSION = "1.0"
MAIN_UNIT_ID = "main"

_REDACTION_MARKERS = (
    "secret",
    "token",
    "api_key",
    "apikey",
    "credential",
    "authorization",
    "password",
    "passphrase",
    "cookie",
    "api-key",
    "private_key",
)
_REDACTED = "[redacted]"
_DIRECTION_ORDER = ("inbound", "outbound")
_AUTHORED_TYPES = ("voice", "chat", "browser")
_AUTHORED_TOOL_KINDS = {"service": "http", "import": "builtin", "construct": "builtin"}
_NAME_MAX = 255

Direction = Literal["inbound", "outbound"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderRef(_Strict):
    provider: str
    id: str = Field(min_length=1, max_length=255)
    version: str | None = None


class ModelConfig(_Strict):
    provider: str | None = None
    name: str
    temperature: float | None = None
    max_tokens: int | None = None


class VoiceConfig(_Strict):
    provider: str | None = None
    voice_id: str
    language: str | None = None
    speed: float | None = None


class Endpoint(_Strict):
    url: str


class Tool(_Strict):
    name: str
    description: str | None = None
    kind: Literal["builtin", "http", "mcp", "unknown"]
    parameters: dict[str, Any] | None = None
    endpoint: Endpoint | None = None


class BehaviourUnit(_Strict):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
    shape: Literal["single_prompt", "state", "flow_node", "external"]
    instructions: str | None = None
    tools: list[str] = []
    transitions: list[str] = []


class Telephony(_Strict):
    number: str
    provider: str | None = None
    direction_supported: list[Direction] = []


class Agent(_Strict):
    schema_version: Literal["1.0"]
    type: Literal["voice", "chat", "coding", "cua", "browser", "crawler"]
    kind: Literal["hosted", "source", "endpoint"]
    name: str
    provider_ref: ProviderRef | None
    behaviour: list[BehaviourUnit] = Field(min_length=1)
    tools: list[Tool]
    model: ModelConfig | None
    voice: VoiceConfig | None
    telephony: list[Telephony]
    supports: list[Direction]
    direction_answer: Direction | None
    gaps: list[
        Literal["states_not_described", "flow_not_described", "engine_not_readable"]
    ]
    instrumentation: Literal["traceai", "opentelemetry", "none"]
    extension: dict[str, Any]
    setup_hash: str

    @model_validator(mode="after")
    def _consistent(self) -> Agent:
        if self.kind == "hosted" and self.provider_ref is None:
            raise ValueError("provider_ref_required")
        if self.kind != "hosted" and self.provider_ref is not None:
            raise ValueError("provider_ref_not_allowed")
        unit_ids = [unit.id for unit in self.behaviour]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("unit_id_repeated")
        tool_names = {tool.name for tool in self.tools}
        if any(
            name not in tool_names for unit in self.behaviour for name in unit.tools
        ):
            raise ValueError("unit_tool_unknown")
        directions = [
            self.supports,
            *(entry.direction_supported for entry in self.telephony),
        ]
        if any(not _sorted_directions(listed) for listed in directions):
            raise ValueError("direction_list_unsorted")
        if self.setup_hash != compute_setup_hash(self.model_dump(mode="json")):
            raise ValueError("setup_hash_mismatch")
        return self


class AgentBuildError(ValueError):
    def __init__(
        self,
        code: Literal[
            "provider_unsupported",
            "document_missing",
            "retell_response_engine_unsupported",
            "agent_invalid",
        ],
        detail: str = "",
    ) -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail


def build_authored_agent(document: Mapping[str, Any]) -> dict[str, Any]:
    contract = _redact(
        {key: value for key, value in document.items() if key != "base_environment"}
    )
    modality = contract.get("modality")
    if modality not in _AUTHORED_TYPES:
        raise AgentBuildError("agent_invalid", "modality")
    tool_specs = contract.get("tools") or []
    tools = [
        _authored_tool(spec, contract.get("tool_entrypoints") or [])
        for spec in tool_specs
    ]
    agent = {
        "schema_version": AGENT_SCHEMA_VERSION,
        "type": modality,
        "kind": "source",
        "name": str(contract.get("agent") or "").strip()[:_NAME_MAX],
        "provider_ref": None,
        "behaviour": [
            {
                "id": MAIN_UNIT_ID,
                "shape": "external",
                "instructions": contract.get("system_prompt_excerpt") or None,
                "tools": [tool["name"] for tool in tools],
                "transitions": [],
            }
        ],
        "tools": tools,
        "model": None,
        "voice": None,
        "telephony": [],
        "supports": [],
        "direction_answer": None,
        "gaps": ["flow_not_described"],
        "instrumentation": contract.get("instrumentation") or "none",
        "extension": {"contract": contract},
    }
    agent["setup_hash"] = compute_setup_hash(agent)
    try:
        return Agent.model_validate(agent).model_dump(mode="json")
    except ValidationError as error:
        raise AgentBuildError("agent_invalid", _error_path(error)) from None


def _authored_tool(
    spec: Mapping[str, Any], entrypoints: list[Mapping[str, Any]]
) -> dict[str, Any]:
    name = spec["name"]
    mode = next(
        (entry.get("mode") for entry in entrypoints if entry.get("tool") == name), ""
    )
    args = spec.get("args") or []
    arg_values = spec.get("arg_values") or {}
    parameters = None
    if args:
        parameters = {
            "type": "object",
            "properties": {
                arg: {"enum": arg_values[arg]}
                if isinstance(arg_values.get(arg), list)
                else {}
                for arg in args
            },
            "required": list(args),
        }
    return {
        "name": name,
        "description": spec.get("description") or None,
        "kind": _AUTHORED_TOOL_KINDS.get(str(mode), "unknown"),
        "parameters": parameters,
        "endpoint": None,
    }


def _redact(value: Any, key: str = "") -> Any:
    lowered = key.lower()
    if any(marker in lowered for marker in _REDACTION_MARKERS):
        return _REDACTED
    if lowered == "headers":
        if isinstance(value, Mapping):
            return {str(item_key): _REDACTED for item_key in value}
        return _REDACTED
    if isinstance(value, Mapping):
        return {
            str(item_key): _redact(item, str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item, key) for item in value]
    if isinstance(value, str) and "://" in value:
        parsed = urlsplit(value)
        if parsed.scheme and parsed.netloc:
            hostname = parsed.hostname or ""
            port = f":{parsed.port}" if parsed.port else ""
            return urlunsplit((parsed.scheme, hostname + port, parsed.path, "", ""))
    return value


def _sorted_directions(listed: Sequence[str]) -> bool:
    positions = [_DIRECTION_ORDER.index(direction) for direction in listed]
    return positions == sorted(set(positions))


def _error_path(error: ValidationError) -> str:
    first = error.errors()[0]
    return "/".join(str(part) for part in first["loc"])
