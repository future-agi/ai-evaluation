"""When a bound sub-actor enters a run, and which channels can carry one.

Shared by the build (which refuses a binding no run could honour) and every engine (which decides,
turn by turn, whether a sub-actor is due). The agent under test never sees any of this.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

# Every channel, the agent modalities it reaches, whether this runtime can deliver it yet, and what
# it is, in the words the build is shown.
CHANNELS: dict[str, dict[str, Any]] = {
    "audio_mix": {
        "modalities": {"voice"},
        "delivered": True,
        "about": "a voice near the person, mixed under them, that the agent may hear",
    },
    "audio_handoff": {
        "modalities": {"voice"},
        "delivered": True,
        "about": "someone who takes the line from the person for up to `max_lines` turns",
    },
    "transfer_leg": {
        "modalities": {"voice"},
        "delivered": True,
        "about": "whoever answers when the agent hands the call over",
    },
    "ivr": {
        "modalities": {"voice"},
        "delivered": True,
        "about": "an automated menu that answers before the person does",
    },
    "tool_response": {
        "modalities": {"voice", "chat", "browser"},
        "delivered": True,
        "about": "a service behind the agent's tools whose responses fail as its `faults` say",
    },
    "thread_message": {
        "modalities": {"chat"},
        "delivered": True,
        "about": "a second participant in the conversation, seen by the agent and the person",
    },
    "inbound_message": {
        "modalities": {"voice", "chat"},
        "delivered": True,
        "about": "a record that arrives in the world mid-task: `inbound.collection` and `inbound.record`",
    },
    "ui_event": {"modalities": {"computer_use"}, "delivered": False, "about": "an event on the screen"},
    "page_content": {"modalities": {"browser"}, "delivered": False, "about": "content on a page"},
    "repo_change": {"modalities": {"coding"}, "delivered": False, "about": "a change in the repository"},
    "peer_agent": {"modalities": {"multi_agent"}, "delivered": False, "about": "another agent"},
}

# Channels that put words into the conversation, so the sub-actor needs a behaviour to speak from.
SPEAKING = frozenset(
    {"audio_mix", "audio_handoff", "transfer_leg", "ivr", "thread_message", "inbound_message"}
)

# When a bound sub-actor enters, and what each trigger needs.
TRIGGER_ABOUT: dict[str, str] = {
    "always": "after the agent's first turn",
    "mid_journey": "after the agent's turn `turn`",
    "on_tool": "on call `on_call` of the agent's tool `tool`",
    "on_keyword": "after a turn containing one of `words`",
    "on_escalation": "when the agent hands over through its tool `tool`",
    "probability": "with chance `p`, drawn once per run; only with a sub-goal whose `sub_actor` names it",
}
TRIGGERS = tuple(TRIGGER_ABOUT)


def delivered_channels() -> list[str]:
    return [name for name, entry in CHANNELS.items() if entry["delivered"]]


def described(entries: dict[str, str]) -> str:
    return "; ".join(f"`{name}`: {about}" for name, about in entries.items())


@dataclass
class Moment:
    """What a run knows at the end of an agent turn: enough to decide whether a sub-actor is due."""

    agent_turns: int
    last_text: str = ""
    tools_called: list[str] = field(default_factory=list)


def drawn(seed: str, sub_actor_id: str, p: float) -> bool:
    """A probability trigger's one seeded draw for this run, the same however often it is asked."""
    digest = hashlib.sha256(f"{seed}|{sub_actor_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < max(0.0, min(1.0, p))


def due(trigger: dict[str, Any], moment: Moment, *, seed: str = "", sub_actor_id: str = "") -> bool:
    """Whether a sub-actor with this trigger enters now. Each trigger fires at most once per run;
    the caller remembers that it fired."""
    kind = str(trigger.get("kind") or "always")
    if kind == "always":
        return moment.agent_turns >= 1
    if kind == "mid_journey":
        return moment.agent_turns == int(trigger.get("turn") or 0)
    if kind in ("on_tool", "on_escalation"):
        tool = str(trigger.get("tool") or "")
        wanted = int(trigger.get("on_call") or 1) if kind == "on_tool" else 1
        return bool(tool) and moment.tools_called.count(tool) >= wanted
    if kind == "on_keyword":
        words = [str(one).strip().lower() for one in trigger.get("words") or [] if str(one).strip()]
        said = moment.last_text.lower()
        return any(re.search(rf"\b{re.escape(word)}\b", said) for word in words)
    if kind == "probability":
        return moment.agent_turns >= max(1, int(trigger.get("turn") or 1)) and drawn(
            seed, sub_actor_id, float(trigger.get("p") or 0)
        )
    return False


class ChannelAdapter(Protocol):
    """One way a sub-actor reaches a run. An agent type lists the channels its adapters carry."""

    def supports(self, modality: str) -> bool: ...

    async def prepare(self, actor: dict[str, Any], scenario: Any) -> None: ...

    async def deliver(self, actor: dict[str, Any], moment: Moment) -> None: ...

    def events(self) -> list[dict[str, Any]]: ...


def supports(channel: str, modality: str) -> bool:
    entry = CHANNELS.get(channel)
    return bool(entry) and modality in entry["modalities"]
