"""A second person heard on the caller's side of a call, entering when its scenario says."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_LINES_INSTRUCTIONS = (
    "You are {name}{role}, near someone who is on a phone call with an automated agent. You are not on "
    "the call yourself; the agent may still hear you. Who you are and what you want: {prompt}\n\n"
    "Write at most {count} short things you would say out loud during that call, one per line, each "
    "under 15 words, in your own voice. Never claim to be the caller and never address the agent by a "
    "name. Output only the lines."
)


class SubActorVoice:
    """One bound sub-actor that speaks: its lines are written and voiced before it is needed."""

    def __init__(
        self,
        *,
        actor: dict[str, Any],
        model: Any,
        tts: Any,
        play: Any,
        clock: Any = time.time,
    ):
        self.actor = actor
        self.model = model
        self.tts = tts
        # Plays a list of frames at a volume and returns when they have been heard.
        self.play = play
        self.clock = clock
        behaviour = actor.get("behaviour") or {}
        self.max_lines = max(1, int(behaviour.get("max_lines") or 1))
        self.volume = 10 ** (float(behaviour.get("loudness_db", -12.0)) / 20)
        trigger = actor.get("trigger") or {}
        self.kind = str(trigger.get("kind") or "always")
        self.turn = int(trigger.get("turn") or 0)
        self._lines: list[tuple[str, list[Any]]] = []
        self._preparation: asyncio.Task[None] | None = None
        self._delivery: asyncio.Task[None] | None = None
        self._delivered = False
        self._events: list[dict[str, Any]] = []

    def start(self) -> None:
        self._preparation = asyncio.create_task(self.prepare())

    async def prepare(self) -> None:
        behaviour = self.actor.get("behaviour") or {}
        role = self.actor.get("role")
        try:
            from livekit.agents import llm as livekit_llm

            context = livekit_llm.ChatContext.empty()
            context.add_message(
                role="user",
                content=_LINES_INSTRUCTIONS.format(
                    name=self.actor.get("name") or "someone",
                    role=f", {role}" if role else "",
                    prompt=behaviour.get("prompt") or "",
                    count=self.max_lines,
                ),
            )
            parts: list[str] = []
            async with self.model.chat(chat_ctx=context, tools=[], tool_choice="none") as stream:
                async for chunk in stream:
                    content = getattr(getattr(chunk, "delta", None), "content", None)
                    if isinstance(content, str):
                        parts.append(content)
            lines = [
                line.strip().strip('"“”-* ')
                for line in "".join(parts).splitlines()
                if line.strip()
            ][: self.max_lines]
            for line in lines:
                async with self.tts.synthesize(line) as stream:
                    frames = [item.frame async for item in stream]
                if frames:
                    self._lines.append((line, frames))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # A quiet sub-actor is a weaker test, never a failed call.
            logger.warning("sub-actor %s not prepared: %s", self.actor.get("id"), type(exc).__name__)

    def due(self, agent_turns: int) -> bool:
        """Whether the agent has now finished the turn this sub-actor enters after."""
        if self._delivered:
            return False
        if self.kind == "always":
            return agent_turns >= 1
        if self.kind == "mid_journey":
            return agent_turns == self.turn
        return False

    def agent_turn_ended(self, agent_turns: int) -> None:
        if self.due(agent_turns):
            self._delivered = True
            self._delivery = asyncio.create_task(self.deliver())

    async def deliver(self) -> None:
        if self._preparation is not None:
            await asyncio.gather(self._preparation, return_exceptions=True)
        for text, frames in self._lines:
            started = self.clock()
            try:
                await self.play(frames, self.volume)
            except Exception as exc:
                logger.warning("sub-actor line not played: %s", type(exc).__name__)
                continue
            self._events.append(
                {
                    "sub_actor": self.actor.get("id"),
                    "name": self.actor.get("name"),
                    "kind": "spoke",
                    "text": text,
                    "at": started,
                    "ended_at": self.clock(),
                }
            )

    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    async def close(self) -> None:
        for task in (self._preparation, self._delivery):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (self._preparation, self._delivery) if task is not None),
            return_exceptions=True,
        )


def sub_actor_messages(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report messages for what the sub-actors said, in the transcript's own shape."""
    return [
        {
            "role": "sub_actor",
            "name": event.get("name") or event.get("sub_actor"),
            "sub_actor": event.get("sub_actor"),
            "content": event["text"],
            "created_at": event["at"],
            "started_speaking_at": event["at"],
            "stopped_speaking_at": event.get("ended_at"),
            "interrupted": False,
            "e2e_latency": None,
        }
        for event in events
        if event.get("kind") == "spoke" and event.get("text")
    ]


def with_sub_actor_turns(
    messages: list[dict[str, Any]], events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The call's messages with the sub-actors' lines placed where they were heard."""
    spoken = sub_actor_messages(events)
    if not spoken:
        return messages

    def when(message: dict[str, Any]) -> float:
        value = message.get("started_speaking_at") or message.get("created_at")
        return float(value) if isinstance(value, (int, float)) else float("inf")

    ordered = list(messages)
    for line in spoken:
        at = when(line)
        index = next((i for i, one in enumerate(ordered) if when(one) > at), len(ordered))
        ordered.insert(index, line)
    return ordered
