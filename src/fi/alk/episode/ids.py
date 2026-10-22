"""Derived ids of an Episode and its events.

Every id is a uuid5 over fixed parts, so a re-read of the same call gives the
same ids and ingestion can upsert instead of duplicating.
"""

from __future__ import annotations

from uuid import UUID, uuid5

# Equal to the simulation runtime's event-id namespace and separator, so an
# Episode event id is the id today's derive_event_id gives for the same parts.
# Copied rather than imported: this package must not load the simulation stack.
EPISODE_ID_NAMESPACE = UUID("79626429-64cd-4ec6-88e8-97b61bb649f3")
_SEPARATOR = "\x1f"


def _stable_id(prefix: str, *parts: str) -> str:
    value = _SEPARATOR.join((prefix, *parts))
    return f"{prefix}_{uuid5(EPISODE_ID_NAMESPACE, value).hex}"


def derive_episode_id(tenant_id: str, provider: str, provider_call_id: str) -> str:
    return _stable_id("episode", tenant_id, provider, provider_call_id)


def derive_episode_event_id(episode_id: str, source: str, seq: int) -> str:
    return _stable_id("event", episode_id, source, str(seq))
