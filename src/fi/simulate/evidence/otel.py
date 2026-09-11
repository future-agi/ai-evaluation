"""OpenTelemetryEvidenceSource skeleton (plan §6.3).

Consumes OTLP spans/metrics that customer-instrumented agents emit.
Real span aggregation lands with the observability wiring in Stage 7.
"""

from __future__ import annotations

import uuid

from .base import EvidenceCapabilities, EvidenceClass, EvidenceSourceSummary
from .providers.base import EvidenceContext, ProviderFetchResult


class OpenTelemetryEvidenceSource:
    capabilities = EvidenceCapabilities(
        transcript=False,
        audio=False,
        tool_calls=True,
        tool_results=True,
        usage=True,
        internal_latency=True,
        configuration_snapshot=True,
    )

    def __init__(self, *, endpoint: str | None = None) -> None:
        self._source_id = f"otel:{uuid.uuid4().hex[:12]}"
        self._endpoint = endpoint
        self._context: EvidenceContext | None = None

    async def connect(self, context: EvidenceContext) -> None:
        self._context = context

    async def fetch_final(self) -> ProviderFetchResult:
        summary = EvidenceSourceSummary(
            source_id=self._source_id,
            adapter="otel",
            evidence_class=EvidenceClass.AGENT_INSTRUMENTED,
            capabilities=self.capabilities,
            available=False,
            metadata={"reason": "not_implemented", "endpoint": self._endpoint},
        )
        return ProviderFetchResult(summary=summary, artifacts=[])

    async def close(self) -> None:
        return None


__all__ = ["OpenTelemetryEvidenceSource"]
