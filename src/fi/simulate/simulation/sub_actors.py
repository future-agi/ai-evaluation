"""When a bound sub-actor enters a run, and which channels can carry one.

Shared by the build (which refuses a binding no run could honour) and every runner (which decides,
turn by turn, whether a sub-actor is due). The agent under test never sees any of this.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

# Every channel: the agent kinds it reaches, whether a runner can deliver it yet, what it is in the
# words the build is shown, how a runner carries it (`enters`), and the one line the simulated person
# is told about it, if any.
#   enters: mixed  a voice under the person      line   holds the line instead of the person
#           thread a second participant in chat
CHANNELS: dict[str, dict[str, Any]] = {
    "audio_mix": {
        "modalities": {"voice"},
        "delivered": True,
        "enters": "mixed",
        "about": "a voice near the person, mixed under them, that the agent may hear",
        "tells_person": "With you, and may speak during the call: {who}.",
        "setting": "You are near someone who is on a phone call with an automated agent. You are not on "
        "the call yourself; the agent may still hear you.",
    },
    "audio_handoff": {
        "modalities": {"voice"},
        "delivered": True,
        "enters": "line",
        "about": "someone who takes the line from the person for up to `max_lines` turns",
        "tells_person": "{who} may take the phone from you for a few turns. While they have it, say "
        "nothing; carry on once they hand it back.",
        "setting": "You have just taken the phone from the person who was talking to an automated agent. "
        "You do not know their codes, numbers, account details or what they have chosen: never give one "
        "or choose for them; when the agent asks for one, say you will hand the phone back.",
    },
    "ivr": {
        "modalities": {"voice"},
        "delivered": True,
        "enters": "line",
        "first": True,
        "about": "an automated menu that answers before the person does",
        "tells_person": "{who} answers first. Say nothing until it has put the call through to you.",
        "setting": "You are the automated phone menu that answers this line before a person does.",
    },
    "thread_message": {
        "modalities": {"chat"},
        "delivered": True,
        "enters": "thread",
        "about": "a second participant in the conversation, seen by the agent and the person",
        "tells_person": "{who} is in this chat with you and may write to the agent too. Give them a turn "
        "before you end the conversation.",
        "setting": "You are a second person in a chat between someone and an automated agent. Never claim "
        "to be the other person.",
    },
    "ui_event": {"modalities": {"computer_use"}, "delivered": False, "enters": "screen", "about": "an event on the screen"},
    "page_content": {"modalities": {"browser"}, "delivered": False, "enters": "page", "about": "content on a page"},
    "repo_change": {"modalities": {"coding"}, "delivered": False, "enters": "repo", "about": "a change in the repository"},
    "peer_agent": {"modalities": {"multi_agent"}, "delivered": False, "enters": "peer", "about": "another agent"},
}

# When a bound sub-actor enters, and what each trigger needs.
TRIGGER_ABOUT: dict[str, str] = {
    "always": "after the agent's first turn (a menu that answers first enters before anyone speaks)",
    "mid_journey": "after the agent's turn `turn`",
    "on_tool": "on call `on_call` of the agent's tool `tool`",
    "on_keyword": "after a turn containing one of `words`",
    "probability": "with chance `p`, drawn once per run; only with a sub-goal whose `sub_actor` names it",
}
TRIGGERS = tuple(TRIGGER_ABOUT)
# Triggers decided from the tools the agent has called.
NEEDS_TOOLS = frozenset({"on_tool"})


def enters(channel: str) -> str:
    return str((CHANNELS.get(channel) or {}).get("enters") or "")


def delivered_channels() -> list[str]:
    return [name for name, entry in CHANNELS.items() if entry["delivered"]]


def described(entries: dict[str, str]) -> str:
    return "; ".join(f"`{name}`: {about}" for name, about in entries.items())


def speaking_as(actor: dict[str, Any]) -> str:
    """Who a speaking sub-actor is and where it is: the opening of every line it writes, on any channel."""
    role = f", {actor['role']}" if actor.get("role") else ""
    about = actor.get("about") or (actor.get("behaviour") or {}).get("prompt") or ""
    setting = (CHANNELS.get(str(actor.get("channel") or "")) or {}).get("setting") or ""
    return f"You are {actor.get('name') or 'someone'}{role}. {setting}\n\n{about}".strip()


def speaking_on(actor: dict[str, Any], said: list[str]) -> str:
    """The opening of a sub-actor's next line: who it is, and the last of what it has said, so it stays itself."""
    heard = " | ".join(said[-3:])
    return speaking_as(actor) + (f"\n\nWhat you have said so far: {heard}" if heard else "")


def tells_person(channel: str, who: str) -> str:
    template = (CHANNELS.get(channel) or {}).get("tells_person") or ""
    return template.format(who=who) if template else ""


def with_others(instruction: str, participants: list[dict[str, Any]]) -> str:
    """The person's situation, saying who else is there and what the person does then."""
    lines = [
        tells_person(
            str(one.get("channel") or ""),
            f"{one['name']} ({one['role']})" if one.get("role") else str(one["name"]),
        )
        for one in participants
        if isinstance(one, dict) and one.get("name")
    ]
    lines = [line for line in lines if line]
    return instruction + "\n\n" + "\n".join(lines) if lines else instruction


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
    if kind in NEEDS_TOOLS:
        tool = str(trigger.get("tool") or "")
        wanted = int(trigger.get("on_call") or 1)
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


class Entering:
    """Whether one bound sub-actor acts on this agent turn, on any channel: it enters once, when its trigger
    is due, and one that speaks then keeps the line for the rest of its `max_lines`."""

    def __init__(self, actor: dict[str, Any], seed: str = "") -> None:
        self.actor = actor
        self.seed = seed
        self.trigger = dict(actor.get("trigger") or {"kind": "always"})
        self.lines = max(1, int((actor.get("behaviour") or {}).get("max_lines") or 1))
        self.entered = False
        self.turns_left = 0

    @property
    def kind(self) -> str:
        return str(self.trigger.get("kind") or "always")

    def due(self, moment: Moment) -> bool:
        return not self.entered and due(self.trigger, moment, seed=self.seed, sub_actor_id=str(self.actor.get("id") or ""))

    def enter(self) -> None:
        self.entered = True
        speaks = enters(str(self.actor.get("channel") or "")) in ("line", "thread")
        self.turns_left = self.lines - 1 if speaks else 0

    def holding(self) -> bool:
        """Whether it still has the line this turn, counting the turn."""
        if self.turns_left <= 0:
            return False
        self.turns_left -= 1
        return True

