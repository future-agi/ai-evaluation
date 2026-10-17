"""Sub-actors on a voice call: a voice near the caller, someone who takes the line from them, or the
caller's own menu that answers first."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
from typing import Any

from fi.simulate.simulation.engines.caller_barge_in import complete_text, synthesize_frames
from fi.simulate.simulation.sub_actors import CHANNELS, NEEDS_TOOLS, Entering, Moment, enters, speaking_as, speaking_on

logger = logging.getLogger(__name__)

_BACKGROUND = (
    "\n\nWrite at most {count} short things you would say out loud during that call, one per line, each "
    "under 15 words, in your own voice and language. Never claim to be the caller and never address the "
    "agent by a name. Output only the lines."
)
# The longest the person waits on a sub-actor's turn: writing, voicing and playing one line.
_TURN_SECONDS = 30.0
# Ends a line whose speaker gives the line back, so the person speaks next.
_BACK = "[BACK]"
_NEXT_LINE = (
    "\n\nThe agent's last words: {said}\n\nSay your next line out loud, under 30 words, in your own "
    f"voice. Output only the line. If you give the line back with it, end it with {_BACK}."
)



class SubActorVoice:
    """One bound sub-actor on a call: decides when it is due, then enters on its channel."""

    def __init__(
        self,
        *,
        actor: dict[str, Any],
        model: Any,
        tts: Any,
        play: Any,
        fallback_tts: Any = None,
        say: Any = None,
        tools_called: Any = None,
        seed: str = "",
        clock: Any = time.time,
    ):
        self.actor = actor
        self.model = model
        self.tts = tts
        # A second voice of its own, used when its first cannot speak; never the caller's.
        self.fallback_tts = fallback_tts
        # Plays frames under the caller at a volume, returning when they have been heard.
        self.play = play
        # Speaks frames as the call's main speaker, returning when they have been heard, and leaves
        # the line in the person's own context so they know what was said.
        self.say = say
        # The names of the tools the agent has called so far, read from the run's live trace.
        self.tools_called = tools_called
        self.seed = seed
        self.clock = clock
        self.channel = str(actor.get("channel") or "audio_mix")
        behaviour = actor.get("behaviour") or {}
        self.volume = 10 ** (float(behaviour.get("loudness_db", -12.0)) / 20)
        self.entering = Entering(actor, seed)
        self.max_lines = self.entering.lines
        self._lines: list[tuple[str, list[Any]]] = []
        self._preparation: asyncio.Task[None] | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._events: list[dict[str, Any]] = []
        # A background voice says one line at a time, a moment into the person's reply, and not on every
        # turn, so its lines fall through the call as someone nearby would say them.
        self._random = random.Random(
            int.from_bytes(hashlib.sha256(f"{seed}|{actor.get('id')}".encode()).digest()[:8], "big")
        )
        self.pause = (0.5, 2.5)
        self.chance = 0.5

    # Kept for the background voice, whose lines do not depend on the conversation.
    @property
    def kind(self) -> str:
        return self.entering.kind

    def start(self) -> None:
        if self.channel == "audio_mix":
            self._preparation = asyncio.create_task(self.prepare())

    async def _write_lines(self, prompt: str) -> list[str]:
        from livekit.agents import llm as livekit_llm

        context = livekit_llm.ChatContext.empty()
        context.add_message(role="user", content=prompt)
        written = await complete_text(self.model, context)
        return [line.strip().strip('"“”-* ') for line in written.splitlines() if line.strip()]

    async def _voiced(self, line: str) -> list[Any]:
        try:
            return await synthesize_frames(self.tts, line)
        except Exception as exc:
            if self.fallback_tts is None or self.fallback_tts is self.tts:
                raise
            logger.warning("sub-actor %s voice failed (%s); using its second voice", self.actor.get("id"), type(exc).__name__)
            self.tts = self.fallback_tts
            return await synthesize_frames(self.tts, line)

    async def prepare(self) -> None:
        try:
            lines = await self._write_lines(speaking_as(self.actor) + _BACKGROUND.format(count=self.max_lines))
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
        return self.entering.due(Moment(agent_turns=agent_turns, last_text=last_text, tools_called=list(tools_called or [])))

    @property
    def answers_first(self) -> bool:
        """A menu bound to enter `always` answers the call before the person can."""
        return bool((CHANNELS.get(self.channel) or {}).get("first")) and self.kind == "always"

    async def enter_first(self) -> None:
        self.entering.enter()
        await self._speak_next("(the call has just been answered)")

    async def on_agent_turn(self, agent_turns: int, last_text: str = "") -> bool:
        """React to one committed agent turn. True while this sub-actor holds the line."""
        try:
            if self.entering.holding():
                return await self._speak_next(last_text)
            if self.answers_first:
                return False
            if self.entering.entered and enters(self.channel) == "mixed":
                if self._random.random() < self.chance:
                    self._tasks.append(asyncio.create_task(self._play_next()))
                return False
            tools: list[str] = []
            if self.kind in NEEDS_TOOLS and self.tools_called is not None:
                tools = list(await asyncio.to_thread(self.tools_called))
            if not self.due(agent_turns, last_text, tools):
                return False
            self.entering.enter()
            return await self.deliver(last_text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("sub-actor %s not delivered: %s", self.actor.get("id"), type(exc).__name__)
            return False

    async def deliver(self, last_text: str = "") -> bool:
        if enters(self.channel) == "line":
            return await self._speak_next(last_text)
        self._tasks.append(asyncio.create_task(self._play_next()))
        return False

    async def _play_next(self) -> None:
        if self._preparation is not None:
            await asyncio.gather(self._preparation, return_exceptions=True)
        if not self._lines:
            return
        text, frames = self._lines.pop(0)
        await asyncio.sleep(self._random.uniform(*self.pause))
        started = self.clock()
        try:
            await self.play(frames, self.volume)
        except Exception as exc:
            logger.warning("sub-actor line not played: %s", type(exc).__name__)
            return
        self._record("spoke", text, started)

    async def _speak_next(self, last_text: str) -> bool:
        if self.say is None:
            return False
        prompt = speaking_on(self.actor, [event["text"] for event in self._events])
        lines = await self._write_lines(prompt + _NEXT_LINE.format(said=last_text or "(nothing yet)"))
        line = lines[0].removesuffix(_BACK).strip() if lines else ""
        frames = await self._voiced(line) if line else []
        if not frames:
            # A line that cannot be heard gives the line back, so the person is never left silent.
            self.entering.turns_left = 0
            return False
        started = self.clock()
        await self.say(frames, line, str(self.actor.get("name") or "someone"))
        self._record("spoke", line, started)
        if lines[0].endswith(_BACK):
            self.entering.turns_left = 0
        return True

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
            try:
                holding = await asyncio.wait_for(voice.on_agent_turn(self.agent_turns, text), _TURN_SECONDS) or holding
            except asyncio.TimeoutError:
                # A sub-actor too slow to speak gives the line back rather than leave the person silent.
                logger.warning("sub-actor %s took too long and gave the line back", voice.actor.get("id"))
                voice.entering.turns_left = 0
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


def world_access(dsn: str) -> Any:
    """Read the run's live tool trace through the world's database: the tools the agent has called."""
    if not dsn:
        return None
    from fi.alk.harness.call_runner import read_tool_trace

    def tools_called() -> list[str]:
        return [str(record.get("name")) for record in read_tool_trace(dsn)]

    return tools_called
