from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


class ProviderConnector(ABC):
    @abstractmethod
    async def connect(self) -> None:
        """Create the provider call and establish its media connection."""

    @abstractmethod
    async def send_audio(self, data: bytes, sample_rate: int) -> None:
        """Send PCM s16le mono audio to the provider."""

    @abstractmethod
    async def recv_audio(self) -> AsyncIterator[tuple[bytes, int]]:
        """Yield provider PCM audio frames and their sample rate."""
        yield b"", 0  # pragma: no cover

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the provider media connection."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the provider media connection is alive."""

    @property
    def is_agent_ready(self) -> bool:
        return self.is_connected

    @property
    def call_id(self) -> str | None:
        return None


@dataclass(frozen=True)
class ConnectorConfig:
    api_key: str
    assistant_id: str
    api_url: str
    livekit_url: str = ""
    first_message_mode: str | None = None
