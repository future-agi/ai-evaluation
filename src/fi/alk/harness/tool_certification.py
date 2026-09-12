"""Deterministic certification of the submitted agent tool inventory.

This layer never invents or executes agent behaviour.  It proves that every tool in the
contract has one unambiguous implementation seam in the authored bundle, and records when
behaviour can only be observed safely during a real call.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contract import AgentContract

TOOL_CERTIFICATION_SCHEMA_VERSION = "futureagi.tool-certification.v1"


class ToolProbeMode(str, Enum):
    READ_ONLY = "read_only"
    DISPOSABLE_WORLD = "disposable_world"
    PROVIDER_SANDBOX = "provider_sandbox"
    RUNTIME_ONLY = "runtime_only"


class ToolAvailability(str, Enum):
    CERTIFIED = "certified"
    RUNTIME_ONLY = "runtime_only"
    REJECTED = "rejected"


class ToolCertification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str
    entrypoint_mode: str
    probe_mode: ToolProbeMode
    availability: ToolAvailability
    schema_validated: bool
    implementation_validated: bool
    reason: str


class ToolCertificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = TOOL_CERTIFICATION_SCHEMA_VERSION
    tools: tuple[ToolCertification, ...]
    certified_or_runtime_only: int = Field(ge=0)
    total: int = Field(ge=0)
    fingerprint: str

    @classmethod
    def create(cls, tools: list[ToolCertification]) -> "ToolCertificationReport":
        ordered = tuple(sorted(tools, key=lambda item: item.tool))
        raw: dict[str, Any] = {
            "schema_version": TOOL_CERTIFICATION_SCHEMA_VERSION,
            "tools": ordered,
            "certified_or_runtime_only": sum(
                item.availability is not ToolAvailability.REJECTED for item in ordered
            ),
            "total": len(ordered),
        }
        raw["fingerprint"] = _fingerprint(raw)
        return cls.model_validate(raw)

    @model_validator(mode="after")
    def _canonical(self) -> "ToolCertificationReport":
        if self.schema_version != TOOL_CERTIFICATION_SCHEMA_VERSION:
            raise ValueError("tool_certification_schema_version_unsupported")
        if self.tools != tuple(sorted(self.tools, key=lambda item: item.tool)):
            raise ValueError("tool_certification_not_canonical")
        expected = _fingerprint(self.model_dump(mode="python", exclude={"fingerprint"}))
        if self.fingerprint != expected:
            raise ValueError("tool_certification_fingerprint_mismatch")
        return self


def _fingerprint(raw: dict[str, Any]) -> str:
    def jsonable(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, tuple):
            return [jsonable(item) for item in value]
        if isinstance(value, dict):
            return {str(key): jsonable(value[key]) for key in sorted(value)}
        return value

    encoded = json.dumps(
        jsonable(raw), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def certify_tool_inventory(
    contract: AgentContract,
    bundle_dir: Path,
    *,
    external_provider: bool,
) -> ToolCertificationReport:
    """Classify every declared tool, rejecting missing or ambiguous implementation seams."""

    entries: dict[str, list[Any]] = {}
    for entry in contract.tool_entrypoints:
        entries.setdefault(entry.tool, []).append(entry)

    results: list[ToolCertification] = []
    for tool in contract.tools:
        matches = entries.get(tool.name, [])
        if external_provider:
            results.append(
                ToolCertification(
                    tool=tool.name,
                    entrypoint_mode="provider",
                    probe_mode=ToolProbeMode.RUNTIME_ONLY,
                    availability=ToolAvailability.RUNTIME_ONLY,
                    schema_validated=True,
                    implementation_validated=False,
                    reason="implementation is owned by the connected provider agent",
                )
            )
            continue
        if len(matches) != 1:
            results.append(
                ToolCertification(
                    tool=tool.name,
                    entrypoint_mode="missing" if not matches else "ambiguous",
                    probe_mode=ToolProbeMode.RUNTIME_ONLY,
                    availability=ToolAvailability.REJECTED,
                    schema_validated=True,
                    implementation_validated=False,
                    reason=(
                        "no implementation entrypoint was discovered"
                        if not matches
                        else "multiple implementation entrypoints were discovered"
                    ),
                )
            )
            continue

        entry = matches[0]
        if entry.mode in {"import", "construct"}:
            handler = bundle_dir / "handlers" / f"{tool.name}.py"
            reachable = handler.is_file() and not handler.is_symlink()
            results.append(
                ToolCertification(
                    tool=tool.name,
                    entrypoint_mode=entry.mode,
                    probe_mode=ToolProbeMode.DISPOSABLE_WORLD,
                    availability=(
                        ToolAvailability.CERTIFIED
                        if reachable
                        else ToolAvailability.REJECTED
                    ),
                    schema_validated=True,
                    implementation_validated=reachable,
                    reason=(
                        "compiled source handler is present; effects run in a disposable world"
                        if reachable
                        else "compiled source handler is absent from the bundle"
                    ),
                )
            )
        elif entry.mode == "service" and entry.endpoint and entry.method:
            results.append(
                ToolCertification(
                    tool=tool.name,
                    entrypoint_mode=entry.mode,
                    probe_mode=ToolProbeMode.RUNTIME_ONLY,
                    availability=ToolAvailability.RUNTIME_ONLY,
                    schema_validated=True,
                    implementation_validated=True,
                    reason="service seam is declared; semantic effects are captured during calls",
                )
            )
        else:
            results.append(
                ToolCertification(
                    tool=tool.name,
                    entrypoint_mode=entry.mode or "unreachable",
                    probe_mode=ToolProbeMode.RUNTIME_ONLY,
                    availability=ToolAvailability.REJECTED,
                    schema_validated=True,
                    implementation_validated=False,
                    reason="tool has no runnable implementation seam",
                )
            )

    # Entrypoints not grounded in a declared tool are also a contract mismatch.
    declared = {tool.name for tool in contract.tools}
    for name in sorted(set(entries) - declared):
        results.append(
            ToolCertification(
                tool=name,
                entrypoint_mode="orphan",
                probe_mode=ToolProbeMode.RUNTIME_ONLY,
                availability=ToolAvailability.REJECTED,
                schema_validated=False,
                implementation_validated=False,
                reason="implementation entrypoint has no declared tool contract",
            )
        )
    return ToolCertificationReport.create(results)


__all__ = [
    "TOOL_CERTIFICATION_SCHEMA_VERSION",
    "ToolAvailability",
    "ToolCertification",
    "ToolCertificationReport",
    "ToolProbeMode",
    "certify_tool_inventory",
]
