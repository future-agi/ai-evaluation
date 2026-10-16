"""Sub-actors on a voice call: a voice near the caller, someone who takes the line, a menu that answers
first, a supervisor on a transfer, or something arriving in the world mid-call."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fi.simulate.simulation.sub_actors import CHANNELS, NEEDS_TOOLS, Moment, due, enters

logger = logging.getLogger(__name__)

_BACKGROUND = (
    "You are {name}{role}, near someone who is on a phone call with an automated agent. You are not on "
    "the call yourself; the agent may still hear you. Who you are and what you want: {prompt}\n\n"
    "Write at most {count} short things you would say out loud during that call, one per line, each "
    "under 15 words, in your own voice. Never claim to be the caller and never address the agent by a "
    "name. Output only the lines."
)
_FOREGROUND = {
    "audio_handoff": "You are {name}{role}. You have just taken the phone from the person who was "
    "talking to an automated agent. Who you are and what you want: {prompt}",
    "transfer_leg": "You are {name}{role}, and a call has just been transferred to you by an automated "
    "agent. Who you are and what you know: {prompt}",
    "ivr": "You are {name}{role}, the automated phone menu that answers this line before a person does. "
    "What the menu offers and how it routes: {prompt}",
}
_NEXT_LINE = (
    "\n\nThe agent's last words: {said}\n\nSay your next line out loud, under 30 words, in your own "
    "voice. Output only the line."
)


def _who(actor: dict[str, Any]) -> dict[str, str]:
    role = actor.get("role")
    return {"name": actor.get("name") or "someone", "role": f", {role}" if role else ""}


class SubActorVoice:
    """One bound sub-actor on a call: decides when it is due, then enters on its channel."""

    def __init__(
        self,
        *,
        actor: dict[str, Any],
        model: Any,
        tts: Any,
        play: Any,
        say: Any = None,
        tools_called: Any = None,
        write: Any = None,
        seed: str = "",
        clock: Any = time.time,
    ):
        self.actor = actor
        self.model = model
        self.tts = tts
        # Plays frames under the caller at a volume, returning when they have been heard.
        self.play = play
        # Speaks frames as the call's main speaker, returning when they have been heard, and leaves
        # the line in the person's own context so they know what was said.
        self.say = say
        # The names of the tools the agent has called so far, read from the run's live trace.
        self.tools_called = tools_called
        # Adds a record to the world: (collection, record).
        self.write = write
        self.seed = seed
        self.clock = clock
        self.channel = str(actor.get("channel") or "audio_mix")
        behaviour = actor.get("behaviour") or {}
        self.max_lines = max(1, int(behaviour.get("max_lines") or 1))
        self.volume = 10 ** (float(behaviour.get("loudness_db", -12.0)) / 20)
        self.trigger = dict(actor.get("trigger") or {"kind": "always"})
        self._lines: list[tuple[str, list[Any]]] = []
        self._preparation: asyncio.Task[None] | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._delivered = False
        self._turns_left = 0
        self._events: list[dict[str, Any]] = []

    # Kept for the background voice, whose lines do not depend on the conversation.
    @property
    def kind(self) -> str:
        return str(self.trigger.get("kind") or "always")

    def start(self) -> None:
        if self.channel == "audio_mix":
            self._preparation = asyncio.create_task(self.prepare())

    async def _write_lines(self, prompt: str) -> list[str]:
        from livekit.agents import llm as livekit_llm

        context = livekit_llm.ChatContext.empty()
        context.add_message(role="user", content=prompt)
        parts: list[str] = []
        async with self.model.chat(chat_ctx=context, tools=[], tool_choice="none") as stream:
            async for chunk in stream:
                content = getattr(getattr(chunk, "delta", None), "content", None)
                if isinstance(content, str):
                    parts.append(content)
        return [
            line.strip().strip('"“”-* ')
            for line in "".join(parts).splitlines()
            if line.strip()
        ]

    async def _voiced(self, line: str) -> list[Any]:
        async with self.tts.synthesize(line) as stream:
            return [item.frame async for item in stream]

    async def prepare(self) -> None:
        behaviour = self.actor.get("behaviour") or {}
        try:
            lines = await self._write_lines(
                _BACKGROUND.format(**_who(self.actor), prompt=behaviour.get("prompt") or "", count=self.max_lines)
            )
            for line in lines[: self.max_lines]:
                frames = await self._voiced(line)
                if frames:
                    self._lines.append((line, frames))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # A quiet sub-actor is a weaker test, never a failed call.
            logger.warning("sub-actor %s not prepared: %s", self.actor.get("id"), type(exc).__name__)

    def due(self, agent_turns: int, last_text: str = "", tools_called: list[str] | None = None) -> bool:
        """Whether the agent has now reached the moment this sub-actor enters at."""
        if self._delivered:
            return False
        moment = Moment(agent_turns=agent_turns, last_text=last_text, tools_called=list(tools_called or []))
        return due(self.trigger, moment, seed=self.seed, sub_actor_id=str(self.actor.get("id") or ""))

    @property
    def answers_first(self) -> bool:
        """A menu bound to enter `always` answers the call before the person can."""
        return bool((CHANNELS.get(self.channel) or {}).get("first")) and self.kind == "always"

    async def enter_first(self) -> None:
        self._delivered = True
        self._turns_left = self.max_lines - 1
        await self._speak_next("(the call has just been answered)")

    async def on_agent_turn(self, agent_turns: int, last_text: str = "") -> bool:
        """React to one committed agent turn. True while this sub-actor holds the line."""
        try:
            if self._turns_left > 0:
                self._turns_left -= 1
                await self._speak_next(last_text)
                return True
            if self.answers_first:
                return False
            tools: list[str] = []
            if self.kind in NEEDS_TOOLS and self.tools_called is not None:
                tools = list(await asyncio.to_thread(self.tools_called))
            if not self.due(agent_turns, last_text, tools):
                return False
            self._delivered = True
            return await self.deliver(last_text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("sub-actor %s not delivered: %s", self.actor.get("id"), type(exc).__name__)
            return False

    async def deliver(self, last_text: str = "") -> bool:
        way = enters(self.channel)
        if way == "world":
            await self._arrive()
            return False
        if way == "line":
            self._turns_left = self.max_lines - 1
            return await self._speak_next(last_text)
        self._tasks.append(asyncio.create_task(self._play_prepared()))
        return False

    async def _play_prepared(self) -> None:
        if self._preparation is not None:
            await asyncio.gather(self._preparation, return_exceptions=True)
        for text, frames in self._lines:
            started = self.clock()
            try:
                await self.play(frames, self.volume)
            except Exception as exc:
                logger.warning("sub-actor line not played: %s", type(exc).__name__)
                continue
            self._record("spoke", text, started)

    async def _speak_next(self, last_text: str) -> bool:
        if self.say is None:
            return False
        behaviour = self.actor.get("behaviour") or {}
        opening = _FOREGROUND[self.channel].format(**_who(self.actor), prompt=behaviour.get("prompt") or "")
        heard = " | ".join(event["text"] for event in self._events[-3:])
        prompt = opening + (f"\n\nWhat you have said so far: {heard}" if heard else "")
        lines = await self._write_lines(prompt + _NEXT_LINE.format(said=last_text or "(nothing yet)"))
        if not lines:
            return False
        frames = await self._voiced(lines[0])
        if not frames:
            return False
        started = self.clock()
        await self.say(frames, lines[0], str(self.actor.get("name") or "someone"))
        self._record("spoke", lines[0], started)
        return True

    async def _arrive(self) -> None:
        inbound = self.actor.get("inbound") or {}
        if self.write is None or not inbound.get("collection"):
            return
        await asyncio.to_thread(self.write, str(inbound["collection"]), dict(inbound.get("record") or {}))
        self._record(
            "arrived", f"(a new {inbound['collection']} record arrived)", self.clock(), record=inbound.get("record")
        )

    def _record(self, kind: str, text: str, started: float, **extra: Any) -> None:
        self._events.append(
            {
                "sub_actor": self.actor.get("id"),
                "name": self.actor.get("name"),
                "channel": self.channel,
                "kind": kind,
                "text": text,
                "at": started,
                "ended_at": self.clock(),
                **extra,
            }
        )

    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    async def close(self) -> None:
        pending = [task for task in (self._preparation, *self._tasks) if task is not None]
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


class SubActorsOnCall:
    """Every sub-actor on one call, told about each committed agent turn exactly once."""

    def __init__(self, voices: list[SubActorVoice]) -> None:
        self.voices = voices
        self.agent_turns = 0

    def __bool__(self) -> bool:
        return bool(self.voices)

    def answers_first(self) -> bool:
        return any(voice.answers_first for voice in self.voices)

    async def open_line(self) -> None:
        for voice in self.voices:
            if voice.answers_first:
                await voice.enter_first()

    async def agent_turn(self, text: str) -> bool:
        """One committed agent turn. True when a sub-actor holds the line, so the person stays silent."""
        self.agent_turns += 1
        holding = False
        for voice in self.voices:
            holding = await voice.on_agent_turn(self.agent_turns, text) or holding
        return holding

    def events(self) -> list[dict[str, Any]]:
        return [event for voice in self.voices for event in voice.events()]

    async def close(self) -> None:
        for voice in self.voices:
            await voice.close()


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
        if event.get("kind") in ("spoke", "arrived") and event.get("text")
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


def world_access(dsn: str) -> tuple[Any, Any]:
    """Read the run's live tool trace and add records to its world, both through the world's database."""
    if not dsn:
        return None, None

    def tools_called() -> list[str]:
        import psycopg

        with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as connection:
            return [row[0] for row in connection.execute("SELECT name FROM _alk_tool_trace ORDER BY at")]

    def write(collection: str, record: dict[str, Any]) -> None:
        import json

        import psycopg
        from psycopg import sql

        columns = list(record)
        statement = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(collection),
            sql.SQL(", ").join(map(sql.Identifier, columns)),
            sql.SQL(", ").join(sql.Placeholder() * len(columns)),
        )
        values = [json.dumps(value) if isinstance(value, (dict, list)) else value for value in record.values()]
        with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as connection:
            connection.execute(statement, values)

    return tools_called, write
