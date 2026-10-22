"""The world's cast: the primary actor the agent serves and the sub-actors around it.

A sub-actor is someone on the person's side who is not the person: a voice near them, someone who
takes the phone from them, or a second participant in their chat. Never anyone on the agent's side.
It lives in the world, so every scenario sees the same one; a scenario binds the ones it needs and
says when each enters.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_serializer, model_validator

from fi.simulate.simulation.sub_actors import CHANNELS as REGISTRY
from fi.simulate.simulation.sub_actors import TRIGGERS, enters

CAST_FILE = "cast.json"
SWITCH = "HARNESS_SUB_ACTORS"


def sub_actors_enabled() -> bool:
    """Whether this sandbox may build or run sub-actors. Off unless the platform turns it on."""
    return os.environ.get(SWITCH, "").strip().lower() in {"1", "true", "yes", "on"}

# Which agent modalities each channel can reach a sub-actor through.
CHANNELS: dict[str, set[str]] = {name: set(entry["modalities"]) for name, entry in REGISTRY.items()}

ORIGINS = ("derived", "user-added")


class Trigger(BaseModel):
    """When a bound sub-actor enters. Deterministic, so a graded scenario has one right outcome."""

    kind: str = "always"
    # mid_journey: after the agent's Nth turn.
    turn: int = 0
    # on_tool: the Nth call of this tool.
    tool: str = ""
    on_call: int = 1
    # on_keyword: any of these words in the last turn.
    words: list[str] = Field(default_factory=list)
    # probability: the chance it enters at all, drawn once per run from the run's seed.
    p: float = 0.0

    @model_serializer(mode="wrap")
    def _only_what_it_uses(self, handler: Any) -> dict[str, Any]:
        body = handler(self)
        unused = {"turn": 0, "tool": "", "on_call": 1, "words": [], "p": 0.0}
        return {key: value for key, value in body.items() if key == "kind" or unused.get(key, object()) != value}


class Behaviour(BaseModel):
    """What a speaking sub-actor is like and how it sounds."""

    prompt: str = ""
    voice: str = ""
    loudness_db: float = -12.0
    max_lines: int = 2


class SubActor(BaseModel):
    id: str
    name: str
    role: str = ""
    channel: str
    origin: str = "derived"
    # Who a speaking sub-actor is, in the caller persona's fields; how it sounds is chosen from it.
    persona: dict[str, Any] = Field(default_factory=dict)
    behaviour: Behaviour | None = None

    @model_validator(mode="after")
    def _a_persona_the_caller_could_have(self) -> "SubActor":
        if self.persona:
            from .scenario import Persona

            self.persona = Persona.model_validate(self.persona).model_dump(exclude_defaults=True)
        return self


class PrimaryActor(BaseModel):
    role: str = ""
    description: str = ""


class Binding(BaseModel):
    """A sub-actor a scenario uses, and when it enters in that scenario."""

    id: str
    trigger: Trigger = Field(default_factory=Trigger)

    @model_validator(mode="before")
    @classmethod
    def _bare_id(cls, data: Any) -> Any:
        return {"id": data} if isinstance(data, str) else data


class Cast(BaseModel):
    primary: PrimaryActor = Field(default_factory=PrimaryActor)
    sub_actors: list[SubActor] = Field(default_factory=list)

    def get(self, sub_actor_id: str) -> SubActor | None:
        return next((one for one in self.sub_actors if one.id == sub_actor_id), None)

    def summary(self) -> str:
        """One line per sub-actor, for a writer's or a judge's brief."""
        lines = [
            f"- {one.id}: {one.name}" + (f" ({one.role})" if one.role else "") + f", {one.channel}"
            for one in self.sub_actors
        ]
        return "\n".join(lines)


def load_cast(root: Path) -> Cast:
    """The cast saved beside the catalogue; an environment without one has none."""
    path = Path(root) / CAST_FILE
    if not path.exists():
        return Cast()
    return Cast.model_validate(json.loads(path.read_text(encoding="utf-8")))


def save_cast(root: Path, cast: Cast) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / CAST_FILE).write_text(
        cast.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8"
    )


def sub_actor_problems(
    actor: SubActor,
    *,
    modality: str,
) -> list[str]:
    """Why this sub-actor cannot be part of this agent's world, or nothing."""
    problems = []
    if actor.channel not in CHANNELS:
        problems.append(
            f"channel {actor.channel!r} is not one of {sorted(CHANNELS)}"
        )
    elif modality and modality not in CHANNELS[actor.channel]:
        problems.append(
            f"a {modality} agent cannot be reached through {actor.channel}; it reaches "
            + ", ".join(sorted(m for m in CHANNELS[actor.channel]))
        )
    elif not REGISTRY[actor.channel]["delivered"]:
        problems.append(
            f"no run in this environment can deliver {actor.channel} yet: it needs the runner "
            "for that kind of agent"
        )
    if actor.origin not in ORIGINS:
        problems.append(f"origin {actor.origin!r} is not one of {list(ORIGINS)}")
    if not actor.id.strip() or not actor.name.strip():
        problems.append("a sub-actor needs an id and a name")
    if not (actor.behaviour and actor.behaviour.prompt.strip()):
        problems.append("a sub-actor needs a behaviour with a prompt: what they want and what they know")
    return problems


def binding_problems(
    bindings: list[Binding],
    cast: Cast,
    *,
    modality: str,
    max_turns: int,
    tools: set[str] | None = None,
) -> list[str]:
    """Why a scenario's bindings cannot run, or nothing."""
    problems = []
    seen: set[str] = set()
    for binding in bindings:
        actor = cast.get(binding.id)
        if actor is None:
            known = ", ".join(one.id for one in cast.sub_actors) or "none"
            problems.append(f"{binding.id} is not in the cast (it has: {known})")
            continue
        if binding.id in seen:
            problems.append(f"{binding.id} is bound twice")
        seen.add(binding.id)
        trigger = binding.trigger
        if trigger.kind not in TRIGGERS:
            problems.append(
                f"{binding.id}: trigger {trigger.kind!r} is not one of {list(TRIGGERS)}"
            )
        if trigger.kind == "mid_journey" and not 1 <= trigger.turn <= max_turns:
            problems.append(
                f"{binding.id}: mid_journey turn {trigger.turn} is outside 1..{max_turns}"
            )
        if trigger.kind == "on_tool":
            if not trigger.tool.strip():
                problems.append(f"{binding.id}: {trigger.kind} needs the `tool` it waits for")
            elif tools is not None and trigger.tool not in tools:
                problems.append(f"{binding.id}: {trigger.kind} names {trigger.tool!r}, which is not one of the agent's tools")
        if trigger.kind == "on_keyword" and not [one for one in trigger.words if one.strip()]:
            problems.append(f"{binding.id}: on_keyword needs the words that bring it in")
        if trigger.kind == "probability" and not 0 < trigger.p < 1:
            problems.append(f"{binding.id}: probability needs p between 0 and 1")
        allowed, why = (REGISTRY.get(actor.channel) or {}).get("enters_on") or ((), "")
        if allowed and trigger.kind not in allowed:
            problems.append(f"{binding.id}: {why}")
        if modality and actor.channel in CHANNELS and modality not in CHANNELS[actor.channel]:
            problems.append(f"{binding.id} arrives through {actor.channel}, which a {modality} scenario cannot carry")
    return problems


def bound_participants(bindings: list[Binding], cast: Cast) -> list[dict[str, Any]]:
    """The sub-actors that enter a conversation, with their channel and when each enters."""
    entering = []
    for binding in bindings:
        actor = cast.get(binding.id)
        if actor is None or enters(actor.channel) not in ("mixed", "line", "thread"):
            continue
        entering.append(
            {
                "id": actor.id,
                "name": actor.name,
                "role": actor.role,
                "channel": actor.channel,
                "behaviour": actor.behaviour.model_dump() if actor.behaviour else {},
                "persona": {**actor.persona, "name": actor.persona.get("name") or actor.name},
                "about": _described(actor),
                "trigger": binding.trigger.model_dump(),
            }
        )
    return entering


def _described(actor: SubActor) -> str:
    """Who a sub-actor is and what it wants, the way the caller's persona is put to the simulator."""
    from .scenario import Persona

    who = Persona.model_validate({**actor.persona, "name": actor.persona.get("name") or actor.name})
    wants = actor.behaviour.prompt if actor.behaviour else ""
    return "\n\n".join(part for part in (who.format_persona(), wants) if part)


def bound_voices(bindings: list[Binding], cast: Cast) -> list[dict[str, Any]]:
    """The entering sub-actors a voice call carries."""
    return [one for one in bound_participants(bindings, cast) if "voice" in CHANNELS[one["channel"]]]


def absent(sub_actor_id: str, messages: Any) -> str:
    """Why a sub-goal about this sub-actor does not apply to a call it never entered, or nothing.

    A sub-actor that was never heard and never wrote never entered, so nothing the agent did could
    have been a response to it.
    """
    if not sub_actor_id:
        return ""
    heard = any(
        isinstance(message, dict)
        and message.get("role") == "sub_actor"
        and message.get("sub_actor") == sub_actor_id
        for message in messages or ()
    )
    if heard:
        return ""
    from .judge import verdict_of

    explanation = f"{sub_actor_id} did not enter this call, so there was nothing to respond to."
    return verdict_of({"verdict": "not_applicable", "explanation": explanation})[1]


def sub_goal_subject_problems(sub_actor_id: str, root: Path) -> list[str]:
    """Why a sub-goal cannot be about this sub-actor, or nothing."""
    if not sub_actor_id:
        return []
    cast = load_cast(root)
    if cast.get(sub_actor_id) is None:
        known = ", ".join(one.id for one in cast.sub_actors) or "none"
        return [f"sub_actor {sub_actor_id!r} is not in the cast (it has: {known})"]
    return []
