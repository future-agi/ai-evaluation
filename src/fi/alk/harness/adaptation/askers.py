"""Who answers adaptation's questions and proposals: a script, a console, or the platform chat.

The conversation asker speaks the §7.5 event table over the existing conversation channel
(`hosted_conversation.ConversationClient`), the same way `hosted_chat_entrypoint._ask` does:
an item is emitted with a message id, the answer is the command whose `reply_to` is that id,
and consumed commands are acknowledged on the next emitted event.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, TextIO

from .errors import AdaptationFailure


@dataclass(frozen=True)
class Option:
    value: Any
    label: str


@dataclass(frozen=True)
class Question:
    id: str
    kind: str  # parameter or agent_rule
    name: str
    default: Any
    prompt: str
    options: tuple[Option, ...]

    def event_payload(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "options": [
                {"value": one.value, "label": one.label} for one in self.options
            ],
            "question": {
                "id": self.id,
                "kind": self.kind,
                "name": self.name,
                "default": self.default,
            },
        }

    def resolve(self, answer: Any) -> Any:
        """The option an answer names, by value or label; None when it names none."""
        for option in self.options:
            if answer == option.value and type(answer) is type(option.value):
                return option.value
        text = (
            str(answer).strip().lower()
            if not isinstance(answer, str)
            else answer.strip().lower()
        )
        for option in self.options:
            if text in {
                json.dumps(option.value).lower(),
                str(option.value).lower(),
                option.label.lower(),
            }:
                return option.value
        return None

    def restated(self) -> str:
        choices = ", ".join(one.label for one in self.options)
        return f"{self.prompt} (options: {choices})"


@dataclass(frozen=True)
class Proposal:
    id: str
    action: str  # drop, specialize or add
    scenario_key: str
    reason: str
    preview: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        verb = {
            "drop": "Drop scenario",
            "specialize": "Use a version of scenario",
            "add": "Add scenario",
        }[self.action]
        return f"{verb} {self.scenario_key}? {self.reason}"

    def event_payload(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "proposal": {
                "id": self.id,
                "action": self.action,
                "scenario_key": self.scenario_key,
                "reason": self.reason,
                "preview": {
                    "name": self.preview.get("name", ""),
                    "instruction": self.preview.get("instruction", ""),
                    "persona": self.preview.get("persona") or {},
                },
            },
        }


_YES = {"yes", "y", "true", "approve", "approved", "ok"}
_NO = {"no", "n", "false", "reject", "rejected"}


def _yes_no(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return True if text in _YES else False if text in _NO else None


class Asker:
    """Base: progress is reported, questions and proposals are answered."""

    async def stage(self, previous: str | None, current: str) -> None:
        return None

    async def say(self, content: str) -> None:
        return None

    async def ask(self, question: Question) -> Any:
        raise NotImplementedError

    async def confirm(self, proposal: Proposal) -> bool:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class ScriptedAsker(Asker):
    """Answers from a file: question id -> value, proposal id -> bool; absent means default.

    A proposal's default is approval: it is what adaptation recommends.
    """

    def __init__(
        self, answers: Mapping[str, Any] | None = None, *, log: TextIO = sys.stderr
    ):
        self.answers = dict(answers or {})
        self.log = log

    async def stage(self, previous: str | None, current: str) -> None:
        print(f"[adaptation] stage {previous or '-'} -> {current}", file=self.log)

    async def say(self, content: str) -> None:
        print(f"[adaptation] {content}", file=self.log)

    async def ask(self, question: Question) -> Any:
        if question.id not in self.answers:
            return question.default
        value = question.resolve(self.answers[question.id])
        if value is None:
            raise AdaptationFailure(
                "answers_invalid",
                f"{question.id}: {self.answers[question.id]!r} is not one of "
                f"{[one.value for one in question.options]}",
            )
        return value

    async def confirm(self, proposal: Proposal) -> bool:
        if proposal.id not in self.answers:
            return True
        approved = _yes_no(self.answers[proposal.id])
        if approved is None:
            raise AdaptationFailure(
                "answers_invalid",
                f"{proposal.id}: {self.answers[proposal.id]!r} is not a bool",
            )
        return approved


class ConsoleAsker(Asker):
    def __init__(self, read: Callable[[str], str] = input, log: TextIO = sys.stderr):
        self.read = read
        self.log = log

    async def stage(self, previous: str | None, current: str) -> None:
        print(f"== {current}", file=self.log)

    async def say(self, content: str) -> None:
        print(content, file=self.log)

    async def ask(self, question: Question) -> Any:
        print(question.prompt, file=self.log)
        for index, option in enumerate(question.options, start=1):
            print(f"  {index}. {option.label}", file=self.log)
        while True:
            answer = (await asyncio.to_thread(self.read, "> ")).strip()
            if not answer:
                return question.default
            if answer.isdigit() and 1 <= int(answer) <= len(question.options):
                return question.options[int(answer) - 1].value
            if (value := question.resolve(answer)) is not None:
                return value
            print("Pick one of the options.", file=self.log)

    async def confirm(self, proposal: Proposal) -> bool:
        print(proposal.prompt, file=self.log)
        while True:
            answer = (await asyncio.to_thread(self.read, "approve? [Y/n] ")).strip()
            if not answer:
                return True
            if (approved := _yes_no(answer)) is not None:
                return approved


class ConversationAsker(Asker):
    """The platform chat as the asker (§7.5 conversation events)."""

    POLL_SECONDS = 0.5
    KEEPALIVE_SECONDS = 20.0

    def __init__(self, client: Any) -> None:
        self.client = client
        self.current = ""
        self.acknowledged = int(client.command_watermark)
        self._keepalive: asyncio.Task | None = None
        self._asking = False

    def _message_id(self, item_id: str) -> str:
        # Stable per conversation and item, so a restarted process re-emits the same question
        # and still finds an answer already given to it.
        conversation = self.client.capabilities.conversation_id
        return str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"alk-adaptation:{conversation}:{item_id}")
        )

    async def _emit(
        self, kind: str, payload: dict[str, Any], message_id: str | None
    ) -> None:
        await self.client.emit(
            kind,
            payload=payload,
            message_id=message_id,
            stage=self.current,
            acknowledge_through=self.acknowledged,
        )

    def _handled(self, sequence: int) -> None:
        # Commands arrive in sequence order and each is handled before the next is looked at,
        # so everything up to the last handled one may be acknowledged. Sequences are shared
        # with assistant messages, so they are not contiguous.
        self.acknowledged = max(self.acknowledged, sequence)

    async def _keep_alive(self) -> None:
        # Listing commands is what keeps the conversation lease warm while proofs run.
        while True:
            await asyncio.sleep(self.KEEPALIVE_SECONDS)
            if not self._asking:
                try:
                    await self.client.commands()
                except Exception:  # noqa: BLE001 - a missed keepalive is retried next tick
                    pass

    async def stage(self, previous: str | None, current: str) -> None:
        if self._keepalive is None:
            self._keepalive = asyncio.create_task(self._keep_alive())
        self.current = current
        await self._emit("stage_changed", {"from": previous, "to": current}, None)

    async def say(self, content: str) -> None:
        # The contract names `content`; the platform's existing message ingest reads `text`.
        await self._emit(
            "assistant_message",
            {"content": content, "text": content},
            str(uuid.uuid4()),
        )

    async def _wait(
        self,
        kind: str,
        message_id: str,
        payload: dict[str, Any],
        restate: str,
        accept: Any,
    ) -> Any:
        self._asking = True
        try:
            await self._emit(kind, payload, message_id)
            while True:
                commands = sorted(
                    await self.client.commands(), key=lambda one: int(one["sequence"])
                )
                for command in commands:
                    sequence = int(command["sequence"])
                    if sequence <= self.acknowledged:
                        continue
                    body = command.get("payload") or {}
                    self._handled(sequence)
                    if str(body.get("reply_to") or "") == message_id:
                        found, value, problem = accept(command.get("kind"), body)
                        if found:
                            return value
                        if problem:
                            await self.say(f"{problem} {restate}")
                        continue
                    if command.get("kind") == "user_message":
                        await self.say(restate)
                await asyncio.sleep(self.POLL_SECONDS)
        finally:
            self._asking = False

    async def ask(self, question: Question) -> Any:
        def accept(kind: str, body: Mapping[str, Any]) -> tuple[bool, Any, str]:
            if kind != "user_response":
                return False, None, ""
            answer = body["value"] if "value" in body else body.get("content")
            value = question.resolve(answer)
            if value is None:
                return False, None, f"{answer!r} is not one of the options."
            return True, value, ""

        return await self._wait(
            "question_requested",
            self._message_id(question.id),
            question.event_payload(),
            question.restated(),
            accept,
        )

    async def confirm(self, proposal: Proposal) -> bool:
        def accept(kind: str, body: Mapping[str, Any]) -> tuple[bool, Any, str]:
            if kind != "approval":
                return False, None, ""
            approved = _yes_no(
                body["approved"] if "approved" in body else body.get("content")
            )
            if approved is None:
                return False, None, "Please approve or reject."
            return True, approved, ""

        return await self._wait(
            "confirmation_requested",
            self._message_id(proposal.id),
            proposal.event_payload(),
            proposal.prompt,
            accept,
        )

    async def close(self) -> None:
        if self._keepalive is not None:
            self._keepalive.cancel()
            try:
                await self._keepalive
            except asyncio.CancelledError:
                pass
            self._keepalive = None
