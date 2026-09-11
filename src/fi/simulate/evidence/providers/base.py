from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from fi.simulate.artifacts.manifest import ArtifactManifestEntry
from fi.simulate.evidence.base import EvidenceSourceSummary


class ProviderConfigError(ValueError):
    """The configured provider evidence adapter cannot run for this call."""


@dataclass(frozen=True)
class EvidenceContext:
    """Context passed by the LiveKit engine to a provider evidence adapter."""

    run_id: str
    test_case_id: str
    case_directory: Path
    started_at: datetime
    call_id_hint: str | None = None
    caller_phone: str | None = None
    callee_phone: str | None = None
    termination_source: str | None = None


@dataclass
class ProviderFetchResult:
    """Return value of ``AgentEvidenceSource.fetch_final``."""

    summary: EvidenceSourceSummary
    artifacts: list[ArtifactManifestEntry] = field(default_factory=list)


def checksum_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def redact_phone(phone: str | None) -> str | None:
    if not phone:
        return None
    digits = "".join(ch for ch in phone if ch.isdigit())
    if len(digits) < 4:
        return "***"
    return "***" + digits[-4:]


def coerce_json(value: Any) -> Any:
    """Downcast provider payload to JSON-safe primitives for report metadata."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [coerce_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): coerce_json(item) for key, item in value.items()}
    return str(value)
