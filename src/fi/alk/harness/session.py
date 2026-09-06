"""A stage as a live conversation, emitting what happened as it happens.

The operator experiences one continuous session: point at an agent, watch a contract appear,
correct something, move on. Underneath, each stage is its own session so context stays small and
any stage can be re-entered without redoing the ones before it.

A stage stays open across turns, so a correction is the next thing said rather than a re-run,
and it yields typed events rather than a wall of text. A terminal renders those events as lines;
a browser renders the same events as a transcript on one side and the artifact on the other.
Neither is privileged, which is the point.

Which loop actually runs the conversation is a backend, selected by ``ALK_HARNESS`` through
``backends.resolve``. A stage describes what it needs in a ``SessionSpec``; the backend supplies
the session and translates its provider's stream into the small reply vocabulary this module
renders. Nothing above this line knows a vendor's name.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from . import spend
from .backends import (
    Call,
    HarnessBackend,
    HarnessSession,
    ModelReply,
    Say,
    SessionOpened,
    SessionSpec,
    StageDone,
    ToolReturned,
    ToolServer,
    resolve,
)

TEXT = "text"
TOOL = "tool"
RESULT = "result"
ARTIFACT = "artifact"
DONE = "done"

# Provider streams normally emit a message or tool event every few seconds.  A subprocess can
# remain alive forever after a dropped upstream stream, though, which previously left a hosted
# job looking healthy while making no progress.  Bound *inactivity*, not total stage duration:
# long scenario suites remain valid as long as they keep producing observable work.
DEFAULT_STAGE_IDLE_TIMEOUT_SECONDS = 600.0
STAGE_IDLE_TIMEOUT_SECONDS = float(
    os.getenv("ALK_STAGE_IDLE_TIMEOUT_SECONDS", str(DEFAULT_STAGE_IDLE_TIMEOUT_SECONDS))
)
STAGE_IDLE_RETRIES = int(os.getenv("ALK_STAGE_IDLE_RETRIES", "1"))


class StageIdleTimeout(TimeoutError):
    """The provider stream stayed open without producing any observable event."""


@dataclass
class Event:
    """One observable thing the stage did.

    ``detail`` carries the data behind what is being shown, not just a label for it: which stage
    emitted this, and for a tool call the arguments it was made with. A terminal renders a line
    and ignores the rest; anything richer needs the data, and re-parsing a rendered line to get
    it back is how a second front end becomes a rewrite.
    """

    kind: str
    text: str = ""
    tool: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        """A terminal-friendly rendering."""
        if self.kind == TEXT:
            return self.text
        if self.kind == TOOL:
            target = self.detail.get("target") or ""
            return f"  [{self.tool}{' ' + target if target else ''}]"
        if self.kind == RESULT:
            marker = "!" if self.detail.get("is_error") else ">"
            body = "\n".join(
                f"  {marker} {row}" for row in self.text.splitlines() if row
            )
            return body or f"  {marker} (no output)"
        if self.kind == ARTIFACT:
            return f"  [saved {self.detail.get('path', '')}]"
        if self.kind == DONE:
            cost = self.detail.get("cost_usd")
            spent = f" ${cost:.4f}" if isinstance(cost, float) else ""
            failure = self.detail.get("error")
            wrong = self.detail.get("unexpected_model") or []
            return (
                f"  [{self.detail.get('outcome', '')} "
                f"turns={self.detail.get('turns', 0)}{spent}]"
                + (f"\n  !! {failure}" if failure else "")
                + (
                    f"\n  !! billed to {', '.join(wrong)}, which is not what was asked for"
                    if wrong
                    else ""
                )
            )
        return self.text


@dataclass
class Turn:
    """What one exchange produced."""

    text: str = ""
    events: list[Event] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    outcome: str = ""
    turns: int = 0
    cost_usd: float | None = None
    error: str = ""


_TARGET_KEYS = (
    "file_path",
    "path",
    "pattern",
    "agent",
    "tool",
    "tool_name",
    "table",
    "name",
)


def _why_it_failed(done: StageDone) -> str:
    """What actually went wrong, said in terms somebody can act on."""
    said = "; ".join(str(error) for error in done.errors)[:400]
    if "invalid_rapt" in said or "invalid_grant" in said:
        return (
            "the provider rejected the credentials. GOOGLE_APPLICATION_CREDENTIALS is probably "
            "not set in this shell, so it fell back to your gcloud login. Load the env file "
            "first: set -a; . ./.env.acceptance; set +a"
        )
    status = done.api_error_status
    return f"the model call failed{f' ({status})' if status else ''}: {said or 'no detail given'}"


def readable(tool_name: str) -> str:
    """A tool's name as somebody reading along would say it.

    ``mcp__scenarios__try_calls`` is how the model addresses it and is noise to anybody else.
    """
    bare = tool_name.rsplit("__", 1)[-1]
    return bare.replace("_", " ")


def _target(payload: Any) -> str:
    """A short label for what a tool call was aimed at, for display only."""
    if not isinstance(payload, dict):
        return ""
    for key in _TARGET_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value if len(value) <= 80 else value[:77] + "..."
    return ""


def _shown(text: str, limit: int = 600) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _saved_path(text: str) -> str:
    """The path a tool reports having written, if it wrote one.

    Only when the tool actually says it saved something. Matching any path-shaped token in any
    result meant that reading a file announced it as an artifact — the stage looks like it is
    producing output while it is still only looking around, and a front end reloads its panes on
    every read.
    """
    said = text.lower()
    if not any(verb in said for verb in ("saved", "wrote", "written")):
        return ""
    for token in text.split():
        # Trimmed before the check, not after. A tool that ends its sentence — "saved to
        # out/contract.json." — produces a token ending in the full stop, so testing the
        # suffix first missed every real save and matched only bare paths, which is what a
        # file *read* returns. The event fired on exactly the wrong occasions.
        cleaned = token.strip(".,;:!?)\"'")
        if cleaned.endswith((".json", ".py", ".sqlite")):
            return cleaned
    return ""


class Stage:
    """One stage of the harness, held open so it can be talked to."""

    def __init__(
        self,
        spec: SessionSpec,
        *,
        name: str = "",
        backend: HarnessBackend | None = None,
    ) -> None:
        self._spec = spec
        self._backend = backend
        self._session: HarnessSession | None = None
        self.name = name
        self.session_id: str | None = None
        self.history: list[Turn] = []
        # What actually got billed, read back rather than assumed. Asking for a model is not the
        # same as getting one: a request that quietly does not take shows up only on the
        # invoice, weeks later, as a number nobody can explain.
        self.models_used: set[str] = set()

    @property
    def spec(self) -> SessionSpec:
        return self._spec

    def grant(
        self, server_name: str, server: ToolServer, tool_names: list[str], ask: Any = None
    ) -> None:
        """Give this stage one more tool server, before it opens.

        The backend builds its permission surface from the spec when the session opens, so a
        grant is a spec change and must land before then. ``tool_names`` is accepted for
        compatibility with existing callers; the server's own tool list is authoritative.
        ``ask`` replaces the operator callback when given, as it always has.
        """
        if self._session is not None:
            raise RuntimeError(
                "grant before the stage opens; the session is already running"
            )
        del tool_names
        self._spec.grant(server_name, server)
        if ask is not None:
            self._spec.ask = ask

    async def __aenter__(self) -> "Stage":
        if self._backend is None:
            self._backend = resolve()
        self._session = self._backend.create(self._spec)
        await self._session.start()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        if self._session is not None:
            await self._session.stop()
            self._session = None

    async def stream(self, message: str) -> AsyncIterator[Event]:
        """Send a message and yield events as they arrive."""
        if self._session is None:
            raise RuntimeError("stage is not open; use it as an async context manager")
        await self._session.send(message)
        turn = Turn()
        replies = self._session.replies().__aiter__()
        # A stage may ask for a longer silence than the default, because for some stages silence
        # is the work: a session whose turn is one tool call that fans out to other sessions
        # emits nothing until that call returns, and killing it then throws away everything the
        # delegates proved.
        idle_bound = self._spec.idle_timeout_seconds or STAGE_IDLE_TIMEOUT_SECONDS
        while True:
            try:
                received = await asyncio.wait_for(
                    replies.__anext__(), timeout=idle_bound
                )
            except StopAsyncIteration:
                break
            except TimeoutError as exc:
                raise StageIdleTimeout(
                    f"{self.name or 'model'} produced no event for {idle_bound:g}s"
                ) from exc
            for event in self._events(received, turn):
                # Which stage this came from, stamped once here rather than by every caller,
                # so a front end showing several stages can tell them apart.
                event.detail.setdefault("stage", self.name)
                turn.events.append(event)
                yield event
        self.history.append(turn)

    def _events(self, received: Any, turn: Turn) -> list[Event]:
        if isinstance(received, SessionOpened):
            self.session_id = received.session_id or self.session_id
            return []
        if isinstance(received, ModelReply):
            events: list[Event] = []
            for part in received.parts:
                if isinstance(part, Say):
                    turn.text += part.text
                    events.append(Event(TEXT, text=part.text))
                elif isinstance(part, Call):
                    turn.tools_used.append(part.name)
                    events.append(
                        Event(
                            TOOL,
                            tool=part.name,
                            detail={
                                "target": _target(part.arguments),
                                "arguments": part.arguments,
                                "label": readable(part.name),
                            },
                        )
                    )
            return events
        if isinstance(received, ToolReturned):
            # What a tool said back is the only view a caller has of whether the work is
            # going well. Dropping it leaves a run that can only be diagnosed by guessing.
            events = [
                Event(
                    RESULT,
                    text=_shown(received.text),
                    detail={"is_error": received.is_error},
                )
            ]
            path = _saved_path(received.text)
            if path:
                turn.artifacts.append(path)
                events.append(Event(ARTIFACT, detail={"path": path}))
            return events
        if isinstance(received, StageDone):
            # The reported outcome alone is not the outcome. A call that failed upstream can
            # still arrive saying "success", so reporting it verbatim tells somebody their
            # stage worked when nothing happened at all.
            failed = bool(received.is_error or received.api_error_status)
            turn.outcome = "failed" if failed else received.outcome
            turn.turns = received.turns
            turn.cost_usd = received.cost_usd
            # Every harness model call passes here, so the spend ledger is fed once rather than
            # per stage: a writer added later is counted without anybody remembering to.
            spend.record(
                self.name, received.cost_usd, received.turns, received.models
            )
            turn.error = _why_it_failed(received) if failed else ""
            self.session_id = received.session_id or self.session_id
            self.models_used |= received.models
            unexpected = self.unexpected_models()
            return [
                Event(
                    DONE,
                    detail={
                        "outcome": turn.outcome,
                        "turns": received.turns,
                        "cost_usd": received.cost_usd,
                        "error": turn.error,
                        "models": sorted(received.models),
                        "unexpected_model": sorted(unexpected),
                    },
                )
            ]
        return []

    async def say(
        self, message: str, *, on_event: Callable[[Event], None] | None = None
    ) -> Turn:
        """Send a message and wait for the whole reply."""
        for attempt in range(STAGE_IDLE_RETRIES + 1):
            try:
                async for event in self.stream(message):
                    if on_event:
                        on_event(event)
                return self.history[-1]
            except StageIdleTimeout:
                if attempt >= STAGE_IDLE_RETRIES:
                    raise
                # A timed-out receive has been cancelled and the provider session may still be
                # waiting on a dead stream.  Reusing it can only reproduce the dead stream.
                # Start a clean session and replay the same stage instruction.  Harness writes
                # are named/idempotent and remain protected by their validation gates.
                if self._session is not None:
                    await self._session.stop()
                assert self._backend is not None
                self._session = self._backend.create(self._spec)
                await self._session.start()
        raise AssertionError("unreachable")

    def unexpected_models(self) -> set[str]:
        """Models that were billed but not the one asked for."""
        asked = self._spec.model
        if not asked:
            return set()
        return {used for used in self.models_used if asked.split("-2")[0] not in used}

    @property
    def spent_usd(self) -> float:
        return sum(turn.cost_usd or 0.0 for turn in self.history)
