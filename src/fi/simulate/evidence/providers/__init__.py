"""Post-call provider evidence adapters (Vapi, Retell).

Each adapter implements the ``AgentEvidenceSource`` Protocol from
``fi.simulate.evidence.base``: after the phone leg finishes the SDK asks
the adapter to fetch the provider's own transcript, recording, tool
calls, and latency, and to hand back an ``EvidenceSourceSummary`` and a
list of ``ArtifactManifestEntry`` rows. Adapters live SDK-side and are
the only place we contact the provider APIs — they never import from
``futureagi/``.
"""

from __future__ import annotations

from .base import EvidenceContext, ProviderConfigError, ProviderFetchResult
from .retell import RetellEvidenceSource
from .vapi import VapiEvidenceSource

__all__ = [
    "EvidenceContext",
    "ProviderConfigError",
    "ProviderFetchResult",
    "RetellEvidenceSource",
    "VapiEvidenceSource",
]
