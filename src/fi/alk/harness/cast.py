"""The world's cast: the primary actor the agent serves and the sub-actors around it.

A sub-actor is any participant that is not the agent and not the person it serves: someone speaking
in the background, a third-party service whose responses fail, and later a supervisor or a second
participant in a thread. It lives in the world, so every scenario sees the same one; a scenario binds
the ones it needs and says when each enters.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

CAST_FILE = "cast.json"
SWITCH = "HARNESS_SUB_ACTORS"


def sub_actors_enabled() -> bool:
    """Whether this sandbox may build or run sub-actors. Off unless the platform turns it on."""
    return os.environ.get(SWITCH, "").strip().lower() in {"1", "true", "yes", "on"}

# Which agent modalities each channel can reach a sub-actor through.
CHANNELS: dict[str, set[str]] = {
    "audio_mix": {"voice"},
    "tool_response": {"voice", "chat", "browser"},
}

KINDS = ("behavioral", "system", "hybrid")
ORIGINS = ("derived", "user-added")
TRIGGERS = ("always", "mid_journey", "on_tool")
FAULTS = ("timeout", "error", "rate_limit", "malformed", "latency", "content")


class Trigger(BaseModel):
    """When a bound sub-actor enters. Deterministic, so a graded scenario has one right outcome."""

    kind: str = "always"
    # mid_journey: after the agent's Nth turn.
    turn: int = 0
    # on_tool: the Nth call of this tool.
    tool: str = ""
    on_call: int = 1


class Behaviour(BaseModel):
    """What a speaking sub-actor is like and how it sounds."""

    prompt: str = ""
    voice: str = ""
    loudness_db: float = -12.0
    max_lines: int = 2


class FaultProfile(BaseModel):
    """How one tool's response goes wrong when a system sub-actor is bound."""

    tool: str
    kind: str
    on_call: int = 1
    latency_ms: int = 0
    content: str = ""


class SubActor(BaseModel):
    id: str
    name: str
    role: str = ""
    kind: str = "behavioral"
    channel: str
    origin: str = "derived"
    depends_on: list[str] = Field(default_factory=list)
    behaviour: Behaviour | None = None
    faults: list[FaultProfile] = Field(default_factory=list)


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
            f"- {one.id}: {one.name} ({one.role or one.kind}), {one.channel}"
            + (f", tools {', '.join(one.depends_on)}" if one.depends_on else "")
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
    tools: set[str],
    faults_reach: bool,
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
    if actor.kind not in KINDS:
        problems.append(f"kind {actor.kind!r} is not one of {list(KINDS)}")
    if actor.origin not in ORIGINS:
        problems.append(f"origin {actor.origin!r} is not one of {list(ORIGINS)}")
    if not actor.id.strip() or not actor.name.strip():
        problems.append("a sub-actor needs an id and a name")
    speaks = actor.kind in ("behavioral", "hybrid")
    fails = actor.kind in ("system", "hybrid")
    if speaks and not (actor.behaviour and actor.behaviour.prompt.strip()):
        problems.append(f"a {actor.kind} sub-actor needs a behaviour with a prompt")
    if fails and not actor.faults:
        problems.append(f"a {actor.kind} sub-actor needs at least one fault")
    if actor.channel == "audio_mix" and not speaks:
        problems.append("audio_mix carries a voice, so the sub-actor must have a behaviour")
    if actor.channel == "tool_response" and not fails:
        problems.append("tool_response carries faults, so the sub-actor must have them")
    unknown = sorted({*actor.depends_on, *(f.tool for f in actor.faults)} - tools)
    if unknown:
        problems.append("names tools the agent does not have: " + ", ".join(unknown))
    for fault in actor.faults:
        if fault.kind not in FAULTS:
            problems.append(f"fault {fault.kind!r} on {fault.tool} is not one of {list(FAULTS)}")
        if fault.on_call < 1:
            problems.append(f"on_call for {fault.tool} counts from 1")
        if fault.kind == "latency" and fault.latency_ms <= 0:
            problems.append(f"a latency fault on {fault.tool} needs latency_ms")
        if fault.kind == "content" and not fault.content.strip():
            problems.append(f"a content fault on {fault.tool} needs the content to add")
    if actor.faults and not faults_reach:
        problems.append(
            "this agent's tool calls do not pass through anything the run controls, so a "
            "fault could never be applied"
        )
    return problems


def binding_problems(
    bindings: list[Binding],
    cast: Cast,
    *,
    modality: str,
    max_turns: int,
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
            profiled = {fault.tool for fault in actor.faults}
            if trigger.tool not in profiled:
                problems.append(
                    f"{binding.id}: on_tool names {trigger.tool!r}, which it has no fault for"
                )
        if actor.channel == "audio_mix" and modality and modality != "voice":
            problems.append(f"{binding.id} speaks over audio, which a {modality} scenario has none of")
    return problems


def bound_faults(bindings: list[Binding], cast: Cast) -> list[dict[str, Any]]:
    """The fault rows a call must apply: one per profile of every bound system sub-actor."""
    rows = []
    for binding in bindings:
        actor = cast.get(binding.id)
        if actor is None:
            continue
        for fault in actor.faults:
            on_call = fault.on_call
            if binding.trigger.kind == "on_tool" and binding.trigger.tool == fault.tool:
                on_call = binding.trigger.on_call
            rows.append(
                {
                    "tool": fault.tool,
                    "kind": fault.kind,
                    "on_call": on_call,
                    "latency_ms": fault.latency_ms,
                    "content": fault.content,
                    "sub_actor": actor.id,
                }
            )
    return rows


def bound_voices(bindings: list[Binding], cast: Cast) -> list[dict[str, Any]]:
    """The speaking sub-actors a call must carry, with when each enters."""
    voices = []
    for binding in bindings:
        actor = cast.get(binding.id)
        if actor is None or actor.channel != "audio_mix" or actor.behaviour is None:
            continue
        voices.append(
            {
                "id": actor.id,
                "name": actor.name,
                "role": actor.role,
                "behaviour": actor.behaviour.model_dump(),
                "trigger": binding.trigger.model_dump(),
            }
        )
    return voices
