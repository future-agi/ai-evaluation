"""Agent Command Center (AgentCC): the gateway every model call goes through when configured.

``AGENTCC_API_KEY`` turns it on and ``AGENTCC_BASE_URL`` names the gateway. Each client keeps
speaking its own vendor's protocol and only changes where it sends it, because the gateway serves
the Gemini, Anthropic and OpenAI APIs natively: a request is relayed as written instead of being
translated through a common format that drops what it has no field for (Gemini thinking levels,
thought signatures).

Only ALK's own clients are pointed here, explicitly. The SDKs' base-URL environment variables are
left alone, because an agent under test running in the same process reads them too and would
otherwise send its traffic, on our key, to our gateway.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.genai import types

DEFAULT_BASE_URL = "https://gateway.futureagi.com"


@dataclass(frozen=True)
class Gateway:
    base_url: str
    api_key: str

    def genai_http_options(self) -> types.HttpOptions:
        """Point a google-genai client at the gateway's Gemini API routes (``/v1beta/models``).

        The client must also be in Gemini API mode with this key (``vertexai=False``,
        ``api_key=api_key``); the gateway reaches Vertex itself.
        """
        from google.genai import types

        return types.HttpOptions(base_url=self.base_url, api_version="v1beta")

    @property
    def openai_base_url(self) -> str:
        """The gateway's OpenAI-compatible root, for clients that append ``/chat/completions``."""
        return f"{self.base_url}/v1"


def gateway() -> Gateway | None:
    """The configured gateway, or None when model calls go straight to each vendor."""
    api_key = os.environ.get("AGENTCC_API_KEY", "").strip()
    if not api_key:
        return None
    base_url = os.environ.get("AGENTCC_BASE_URL", "").strip() or DEFAULT_BASE_URL
    return Gateway(base_url=base_url.rstrip("/"), api_key=api_key)


def vendor_model(model: str) -> str:
    """The vendor's own name for a gateway model: ``vertex_ai/gemini-3.7-flash`` is ``gemini-3.7-flash``.

    The gateway reads the prefix to choose the upstream provider, so it stays on the wire; what a
    model can do, and a call made straight to the vendor, go by the name without it.
    """
    return model.rsplit("/", 1)[-1]


def vertex_model(model: str) -> str:
    """The gateway's name for a model served by Vertex: ``gemini-3.7-flash`` is ``vertex_ai/gemini-3.7-flash``.

    A bare name is the Gemini API's, so a client configured for Vertex adds the prefix; a name that
    already names its provider is left alone.
    """
    return model if "/" in model else f"vertex_ai/{model}"
