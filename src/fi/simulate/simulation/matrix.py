"""Matrix runner: sweep a scenario across provider/channel legs.

Composes an SDK simulation manifest with per-leg overrides so the same
scenario can be executed against ``{Vapi, Retell, LiveKit}`` in
``{webrtc, sip_outbound, sip_inbound}`` shapes. Legs that cannot run
against a provider (e.g. Retell over PSTN outbound) are declared with
``skip_reason`` and reported as ``UNSUPPORTED`` in the canonical output
instead of being silently omitted.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, Field

from fi.simulate.runtime import (
    FailureStage,
    SimulationFailure,
    TestCaseStatus,
)

logger = logging.getLogger(__name__)

Channel = Literal["webrtc", "sip_outbound", "sip_inbound"]


class MatrixLeg(BaseModel):
    """One column of the provider × channel matrix."""

    provider: str
    channel: Channel
    agent_definition_overrides: dict[str, Any] = Field(default_factory=dict)
    provider_evidence_overrides: dict[str, Any] | None = None
    skip_reason: str | None = None
    max_concurrent_calls: int | None = None

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.channel}"


@dataclass
class MatrixLegResult:
    leg: MatrixLeg
    manifest: dict[str, Any]
    started_at: datetime
    ended_at: datetime | None = None
    report: Any = None
    error: str | None = None
    skipped: bool = False
    skipped_cases: list[dict[str, Any]] = field(default_factory=list)


def _leg_manifest(base: Mapping[str, Any], leg: MatrixLeg) -> dict[str, Any]:
    manifest = copy.deepcopy(dict(base))
    agent_definition = dict(manifest.get("agent_definition") or {})
    agent_definition.update(leg.agent_definition_overrides)
    transport = dict(agent_definition.get("transport") or {})
    transport["kind"] = leg.channel
    if leg.channel == "webrtc":
        for legacy in ("sip_trunk_id", "sip_number", "sip_call_to", "dispatch_rule_name"):
            transport.pop(legacy, None)
    agent_definition["transport"] = transport
    if leg.provider_evidence_overrides is not None:
        agent_definition["provider_evidence"] = leg.provider_evidence_overrides
    manifest["agent_definition"] = agent_definition
    simulation = dict(manifest.get("simulation") or {})
    simulation["engine"] = simulation.get("engine", "livekit")
    manifest["simulation"] = simulation
    manifest["name"] = f"{manifest.get('name', 'matrix-run')}:{leg.label}"
    return manifest


def build_skipped_report(
    leg: MatrixLeg,
    base: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> MatrixLegResult:
    manifest = _leg_manifest(base, leg)
    started = now or datetime.now(timezone.utc)
    scenario = manifest.get("scenario") or {}
    dataset = scenario.get("dataset") or []
    skipped_cases = [
        {
            "index": index,
            "persona": (row.get("persona") if isinstance(row, dict) else None),
            "status": TestCaseStatus.UNSUPPORTED.value,
            "failure": SimulationFailure(
                stage=FailureStage.PREPARING,
                code="provider_channel_unsupported",
                message=leg.skip_reason or "provider_channel_unsupported",
                retryable=False,
                provider=leg.provider,
                details={"channel": leg.channel},
            ).model_dump(mode="json", exclude_none=True),
        }
        for index, row in enumerate(dataset)
    ]
    return MatrixLegResult(
        leg=leg,
        manifest=manifest,
        started_at=started,
        ended_at=started,
        skipped=True,
        skipped_cases=skipped_cases,
    )


async def run_matrix(
    base_manifest: Mapping[str, Any],
    legs: list[MatrixLeg],
    *,
    manifest_path: Path,
    max_concurrent_calls: int = 1,
    run_leg,
) -> list[MatrixLegResult]:
    """Sweep ``legs`` against ``base_manifest``.

    ``run_leg`` is an async callable ``(manifest, manifest_path, leg)``
    that executes one leg and returns the SDK report. It is injected
    rather than imported here so this module remains framework-lean and
    testable without a live LiveKit backend.
    """

    results: list[MatrixLegResult] = []
    for leg in legs:
        if leg.skip_reason:
            skipped = build_skipped_report(leg, base_manifest)
            logger.info(
                "matrix_leg_skipped",
                extra={"leg": leg.label, "reason": leg.skip_reason},
            )
            results.append(skipped)
            continue
        started = datetime.now(timezone.utc)
        manifest = _leg_manifest(base_manifest, leg)
        result = MatrixLegResult(leg=leg, manifest=manifest, started_at=started)
        try:
            result.report = await run_leg(
                manifest=manifest,
                manifest_path=manifest_path,
                leg=leg,
                max_concurrent_calls=leg.max_concurrent_calls or max_concurrent_calls,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "matrix_leg_error",
                extra={"leg": leg.label, "error": type(exc).__name__},
            )
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            result.ended_at = datetime.now(timezone.utc)
        results.append(result)
    return results


__all__ = [
    "Channel",
    "MatrixLeg",
    "MatrixLegResult",
    "build_skipped_report",
    "run_matrix",
]
