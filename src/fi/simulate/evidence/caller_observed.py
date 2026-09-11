"""CallerObservedEvidenceSource skeleton (plan §6).

Reports whatever the SDK's own recorder + transcript captured for the
simulator leg (audio, transcript, timing) without consulting provider
APIs or agent instrumentation. Real per-track summary derivation lands
alongside the media router in §5.
"""

from __future__ import annotations

import uuid

from .base import EvidenceCapabilities, EvidenceClass, EvidenceSourceSummary
from .providers.base import EvidenceContext, ProviderFetchResult


class CallerObservedEvidenceSource:
    capabilities = EvidenceCapabilities(
        transcript=True,
        audio=True,
        tool_calls=False,
        tool_results=False,
        usage=False,
        internal_latency=False,
        configuration_snapshot=False,
    )

    def __init__(self) -> None:
        self._source_id = f"caller_observed:{uuid.uuid4().hex[:12]}"
        self._context: EvidenceContext | None = None

    async def connect(self, context: EvidenceContext) -> None:
        self._context = context

    async def fetch_final(self) -> ProviderFetchResult:
        summary = EvidenceSourceSummary(
            source_id=self._source_id,
            adapter="caller_observed",
            evidence_class=EvidenceClass.CALLER_OBSERVED,
            capabilities=self.capabilities,
            available=False,
            metadata={"reason": "not_implemented"},
        )
        return ProviderFetchResult(summary=summary, artifacts=[])

    async def close(self) -> None:
        return None


__all__ = ["CallerObservedEvidenceSource"]
