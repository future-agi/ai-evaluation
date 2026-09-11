"""LiveKitAgentInstrumentationSource skeleton (plan §6.6).

Consumes canonical events emitted by
``fi.simulate.instrumentation.livekit.FutureAGIObserver`` and surfaces
them as agent-instrumented evidence. Bridging the observer's in-process
event stream into a case-scoped ``EvidenceSourceSummary`` lands with
Stage 7.
"""

from __future__ import annotations

import uuid

from .base import EvidenceCapabilities, EvidenceClass, EvidenceSourceSummary
from .providers.base import EvidenceContext, ProviderFetchResult


class LiveKitAgentInstrumentationSource:
    capabilities = EvidenceCapabilities(
        transcript=True,
        audio=False,
        tool_calls=True,
        tool_results=True,
        usage=True,
        internal_latency=True,
        configuration_snapshot=True,
    )

    def __init__(self) -> None:
        self._source_id = f"livekit_instrumentation:{uuid.uuid4().hex[:12]}"
        self._context: EvidenceContext | None = None

    async def connect(self, context: EvidenceContext) -> None:
        self._context = context

    async def fetch_final(self) -> ProviderFetchResult:
        summary = EvidenceSourceSummary(
            source_id=self._source_id,
            adapter="livekit_instrumentation",
            evidence_class=EvidenceClass.AGENT_INSTRUMENTED,
            capabilities=self.capabilities,
            available=False,
            metadata={"reason": "not_implemented"},
        )
        return ProviderFetchResult(summary=summary, artifacts=[])

    async def close(self) -> None:
        return None


__all__ = ["LiveKitAgentInstrumentationSource"]
