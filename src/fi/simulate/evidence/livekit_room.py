"""LiveKitRoomDataEvidenceSource skeleton (plan §6.5).

Streams participant events, tracks, and data-channel messages via the
LiveKit room API. This skeleton establishes the seam for Stage 7 and
declares the capabilities LiveKit-native evidence carries even before
the collector is wired up.
"""

from __future__ import annotations

import uuid

from .base import EvidenceCapabilities, EvidenceClass, EvidenceSourceSummary
from .providers.base import EvidenceContext, ProviderFetchResult


class LiveKitRoomDataEvidenceSource:
    capabilities = EvidenceCapabilities(
        transcript=False,
        audio=True,
        tool_calls=False,
        tool_results=False,
        usage=False,
        internal_latency=False,
        configuration_snapshot=True,
    )

    def __init__(self) -> None:
        self._source_id = f"livekit_room:{uuid.uuid4().hex[:12]}"
        self._context: EvidenceContext | None = None

    async def connect(self, context: EvidenceContext) -> None:
        self._context = context

    async def fetch_final(self) -> ProviderFetchResult:
        summary = EvidenceSourceSummary(
            source_id=self._source_id,
            adapter="livekit_room",
            evidence_class=EvidenceClass.PLATFORM_VERIFIED,
            capabilities=self.capabilities,
            available=False,
            metadata={"reason": "not_implemented"},
        )
        return ProviderFetchResult(summary=summary, artifacts=[])

    async def close(self) -> None:
        return None


__all__ = ["LiveKitRoomDataEvidenceSource"]
