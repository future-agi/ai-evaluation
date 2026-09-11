"""What is being tested, and how the harness talks to it.

The rest of the run does not care what the agent under test is. It says something and gets a
reply back, and whatever tool calls happened in between landed in the world. That is the entire
interface, and keeping it that narrow is what lets the same scenarios, the same world and the
same grading run against an agent hosted anywhere.

Two things are supplied per target: how to say something to it, and how its tool calls reach the
world. ``LocalAgent`` is only for contract-only specs with no submitted implementation.
``RepositoryChatTarget`` starts the submitted runtime and reaches its existing HTTP/WebSocket
interface. A hosted target uses the same narrow protocol with the transport swapped: its tool
calls arrive over a webhook or in a turn response, and the same world answers them. The world,
scenarios and grading do not change.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import socket
import time
from typing import Any, Callable, Protocol, runtime_checkable
from urllib.parse import urljoin

from ..backends import SessionSpec, resolve as resolve_backend, tool, tool_server

from ..config import chosen_model
from ..contract import AgentContract
from ..session import Stage
from ..world.runtime import GeneratedWorld

AGENT_SERVER = "agent"

_TYPES: dict[str, type] = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
    "list": list,
    "dict": dict,
}


def _python_type(declared: str) -> type:
    """The type a tool's argument is declared with, as something a schema can carry."""
    lowered = (declared or "").strip().lower()
    if lowered.startswith(("list", "sequence", "array")):
        return list
    if lowered.startswith(("dict", "mapping", "object")):
        return dict
    return _TYPES.get(lowered, str)


def describe(spec: Any, contract: AgentContract) -> str:
    """What the agent is told a tool takes, including the values it accepts.

    The values matter more than they look. An agent whose real schema enumerates its menu knows
    that a Big Mac combo is ``big_mac_combo``; the same agent without them guesses, gets refused,
    and reads as broken when what is broken is the harness that withheld them. Anything the
    contract recorded as permitted, the agent under test is told.
    """
    parts = [spec.description or f"{spec.name} for {contract.agent}"]
    for arg in spec.args:
        values = spec.arg_values.get(arg)
        if isinstance(values, (list, tuple)) and values:
            rendered = ", ".join(str(value) for value in values)
            parts.append(f"  {arg} accepts: {rendered}")
        elif arg in spec.arg_types:
            parts.append(f"  {arg}: {spec.arg_types[arg]}")
    return "\n".join(parts)


def agent_tools(contract: AgentContract, world: GeneratedWorld) -> Any:
    """The agent's own tools, wired to the world so a call really happens.

    Every call goes through ``world.call``, so a refusal comes back as a refusal the agent can
    read and recover from, rather than as a success it will happily build on.
    """

    def bind(spec: Any) -> Any:
        schema = {
            arg: _python_type(spec.arg_types.get(arg, "str")) for arg in spec.args
        }

        @tool(spec.name, describe(spec, contract), schema)
        async def call_tool(
            args: dict[str, Any], _name: str = spec.name
        ) -> dict[str, Any]:
            # Through handle_tool_call, not straight to world.call. That method is the interface
            # ALK's own runners drive an environment by, so going around it would leave the
            # claim that a generated world plugs into them untested — and free to drift.
            done = world.handle_tool_call({"name": _name, "arguments": args})
            if done is None:
                return {
                    "content": [{"type": "text", "text": f"no such tool {_name}"}],
                    "is_error": True,
                }
            return {
                "content": [{"type": "text", "text": done.content or ""}],
                **({} if done.success else {"is_error": True}),
            }

        return call_tool

    return tool_server(
        name=AGENT_SERVER,
        version="0.1.0",
        tools=[bind(spec) for spec in contract.tools],
    )


def agent_prompt(contract: AgentContract) -> str:
    """The agent under test, as its contract describes it.

    Only what the contract records, because anything added here is a difference between the agent
    being graded and the agent that exists.
    """
    parts = [
        f"You are {contract.agent}: {contract.one_liner}".strip(),
        contract.system_prompt_excerpt.strip(),
    ]
    if contract.hard_constraints:
        parts.append(
            "Rules you must follow:\n  - " + "\n  - ".join(contract.hard_constraints)
        )
    if contract.modality == "voice":
        parts.append(
            "You are speaking out loud. Keep replies to what a person would actually say: "
            "short, no lists, no markdown."
        )
    parts.append(
        "Use your tools to do anything real. Never tell the customer something is done unless a "
        "tool confirmed it, and if a tool refuses, say so plainly and offer what is possible."
    )
    return "\n\n".join(part for part in parts if part)


@runtime_checkable
class Target(Protocol):
    """An agent under test, reachable by saying something to it."""

    key: str

    async def open(self) -> None: ...
    async def say(self, utterance: str) -> str: ...
    async def close(self) -> None: ...
    @property
    def spent_usd(self) -> float: ...


def _drivable(model: str | None) -> None:
    """Refuse a model the selected backend cannot actually run, before a suite is graded on it.

    Handed a model it cannot reach, a backend does not fail: it produces a session that answers
    nothing, which arrives as a scenario with no turns and no calls and every check red. That
    reads exactly like an agent that ignored the person, and the whole suite is wrong in a way
    nobody would think to question.
    """
    named = (model or "").strip().lower()
    if not named:
        return
    backend = resolve_backend()
    if backend.can_drive(named):
        return
    raise RuntimeError(
        f"this target cannot run {model!r}. The selected harness backend "
        f"({backend.name}) does not drive that model. Pick a model that backend serves, "
        "select the backend that serves it through ALK_HARNESS, or point the spec's target "
        "at one of ALK's own endpoint adapters rather than at this one."
    )


class LocalAgent:
    """The agent run here, from its contract, with its tools bound to the world."""

    key = "local"

    def __init__(
        self,
        contract: AgentContract,
        world: GeneratedWorld,
        *,
        model: str | None = None,
        max_turns: int = 12,
    ) -> None:
        self.contract = contract
        self.world = world
        if contract.runtime or contract.tool_entrypoints or contract.implementation:
            raise RuntimeError(
                "the local contract target is disabled for repository-backed agents because it "
                "reconstructs the agent from its prompt. Register a target that starts the "
                "agent's shipped runtime and applies the provisioned endpoint overrides."
            )
        _drivable(model)
        # The agent under test gets its own tools and nothing else. A target that can reach a
        # file or a shell is not the agent anybody deployed.
        spec = SessionSpec(
            system_prompt=agent_prompt(contract),
            servers={AGENT_SERVER: agent_tools(contract, world)},
            max_turns=max_turns,
            model=chosen_model(model),
        )
        self._stage = Stage(spec, name="target")

    async def open(self) -> None:
        await self._stage.__aenter__()

    async def say(self, utterance: str) -> str:
        turn = await self._stage.say(utterance)
        return turn.text.strip()

    async def close(self) -> None:
        await self._stage.__aexit__(None, None, None)

    @property
    def spent_usd(self) -> float:
        return self._stage.spent_usd


class RepositoryChatTarget:
    """The submitted chat runtime, reached over its existing turn ingress.

    Lifecycle mirrors the repository-backed voice path: bind the scenario's generated world,
    start the isolated submitted runtime with only endpoint substitutions, wait for its real
    ingress, converse, then remove only that runtime. The source prompt and tools are never
    reconstructed in this process.
    """

    key = "repository"

    def __init__(
        self,
        contract: AgentContract,
        world: GeneratedWorld,
        *,
        world_root: str | Path,
        trace_path: str | Path | None = None,
        scenario_name: str = "scenario",
    ) -> None:
        runtime = contract.runtime
        interface = runtime.interface if runtime is not None else None
        if interface is None:
            raise RuntimeError(
                "the submitted chat runtime has no recorded conversational interface. "
                "Record its existing HTTP port, path and protocol during understanding; the "
                "harness will not reconstruct the repository agent from its prompt."
            )
        if interface.kind not in {"http", "websocket"}:
            raise RuntimeError(
                f"submitted chat interface {interface.kind!r} is not wired for hosted repository "
                "execution yet; supported now: HTTP and WebSocket"
            )
        if interface.port is None:
            raise RuntimeError("submitted HTTP chat interface has no container port")
        self.contract = contract
        self.world = world
        self.world_root = Path(world_root)
        self.trace_path = Path(trace_path) if trace_path is not None else None
        self.scenario_name = scenario_name
        self.interface = interface
        self._webhook: Any | None = None
        self._wrapper: Any | None = None
        self._messages: list[dict[str, Any]] = []
        self._turn = 0

    async def open(self) -> None:
        from ..provision import (
            connect_runner_network,
            runtime_endpoint,
            start_runtime,
        )
        from .voice import WorldWebhook

        webhook = WorldWebhook().start()
        webhook.bind(self.world)
        self._webhook = webhook
        try:
            private_host = await asyncio.to_thread(
                connect_runner_network, self.world_root
            )
            tool_url = (
                f"http://{private_host}:{webhook.port}"
                if private_host
                else f"http://host.docker.internal:{webhook.port}"
            )
            await asyncio.to_thread(
                start_runtime,
                self.world_root,
                overrides={"TOOLS_API_URL": tool_url},
                trace_path=self.trace_path,
                publish_ports=[self.interface.port],
                stable_seconds=0.5,
            )
            # A runtime-only Compose project creates its network at start. This second call is
            # the same idempotent attach used by voice and makes its private container address
            # reachable from a hosted runner container.
            await asyncio.to_thread(connect_runner_network, self.world_root)
            scheme = "ws" if self.interface.kind == "websocket" else "http"
            base = await asyncio.to_thread(
                runtime_endpoint,
                self.world_root,
                self.interface.port,
                scheme=scheme,
            )
            await asyncio.to_thread(self._wait_ready, base)
            if self.interface.kind == "websocket":
                from fi.simulate.agent.wrappers.websocket import WebSocketAgentWrapper

                wrapper = WebSocketAgentWrapper
            else:
                from fi.simulate.agent.wrappers.http import HTTPAgentWrapper

                wrapper = HTTPAgentWrapper
            self._wrapper = wrapper(
                endpoint=urljoin(
                    base.rstrip("/") + "/", self.interface.path.lstrip("/")
                ),
                protocol=self.interface.protocol,
                include_tools=self.interface.include_tools,
                timeout=30.0,
                metadata={
                    "target": "submitted_repository_runtime",
                    "scenario": self.scenario_name,
                },
            )
        except Exception:
            await self.close()
            raise

    def _wait_ready(self, base: str) -> None:
        from urllib import error as urllib_error
        from urllib import request as urllib_request
        from urllib.parse import urlsplit

        deadline = time.monotonic() + 60.0
        health = self.interface.health_path
        last = "not reachable"
        while time.monotonic() < deadline:
            try:
                if health and self.interface.kind == "http":
                    url = urljoin(base.rstrip("/") + "/", health.lstrip("/"))
                    with urllib_request.urlopen(url, timeout=2) as response:
                        if int(getattr(response, "status", 200)) < 500:
                            return
                else:
                    parsed = urlsplit(base)
                    with socket.create_connection(
                        (str(parsed.hostname), int(parsed.port or 80)), timeout=2
                    ):
                        return
            except (OSError, urllib_error.URLError) as exc:
                last = f"{type(exc).__name__}: {exc}"
                time.sleep(0.25)
        raise RuntimeError(
            f"submitted chat runtime did not become ready on port {self.interface.port}: {last}"
        )

    async def say(self, utterance: str) -> str:
        if self._wrapper is None:
            raise RuntimeError("submitted chat runtime is not open")
        from fi.simulate.agent.wrapper import AgentInput

        self._messages.append({"role": "user", "content": utterance})
        for _step in range(8):
            request = AgentInput(
                thread_id=self.scenario_name,
                execution_id=self.scenario_name,
                turn_index=self._turn,
                scenario_name=self.scenario_name,
                modality="text",
                messages=list(self._messages),
                new_message=dict(self._messages[-1]),
                tools=self._tools() if self.interface.include_tools else [],
            )
            response = await self._wrapper.call(request)
            trace = dict((response.metadata or {}).get("external_agent") or {})
            if trace and not trace.get("success", False):
                raise RuntimeError(
                    str(trace.get("error") or "submitted chat endpoint request failed")
                )
            calls = list(response.tool_calls or [])
            if calls:
                self._messages.append(
                    {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": calls,
                    }
                )
                for index, call in enumerate(calls, start=1):
                    name, arguments, call_id = self._tool_call(call, index)
                    result = self.world.handle_tool_call(
                        {"id": call_id, "name": name, "arguments": arguments}
                    )
                    self._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": (
                                result.content
                                if result is not None
                                else f"there is no tool called {name}"
                            ),
                        }
                    )
                continue
            said = response.content.strip()
            self._messages.append({"role": "assistant", "content": said})
            self._turn += 1
            return said
        raise RuntimeError(
            "submitted chat agent exceeded 8 tool continuations in one turn"
        )

    @staticmethod
    def _tool_call(call: dict[str, Any], index: int) -> tuple[str, dict[str, Any], str]:
        function = (
            call.get("function") if isinstance(call.get("function"), dict) else {}
        )
        name = str(call.get("name") or function.get("name") or "")
        raw = call.get("arguments", function.get("arguments", {}))
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = {"_raw": raw}
        else:
            parsed = dict(raw or {}) if isinstance(raw, dict) else {}
        return name, parsed, str(call.get("id") or f"call_{index}")

    def _tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for spec in self.contract.tools:
            properties = {
                arg: {"type": self._json_type(spec.arg_types.get(arg, "string"))}
                for arg in spec.args
            }
            tools.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": list(spec.args),
                    },
                }
            )
        return tools

    @staticmethod
    def _json_type(declared: str) -> str:
        normalized = str(declared or "").lower()
        if any(mark in normalized for mark in ("int", "float", "number")):
            return "number"
        if "bool" in normalized:
            return "boolean"
        if any(mark in normalized for mark in ("list", "array", "sequence")):
            return "array"
        if any(mark in normalized for mark in ("dict", "map", "object")):
            return "object"
        return "string"

    async def close(self) -> None:
        from ..provision import stop_runtime

        try:
            await asyncio.to_thread(stop_runtime, self.world_root)
        finally:
            if self._webhook is not None:
                self._webhook.stop()
                self._webhook = None
            self._wrapper = None

    @property
    def spent_usd(self) -> float:
        # The submitted target owns its model/provider accounting. Provider evidence may add it
        # later; the harness must not fabricate a cost from HTTP traffic.
        return 0.0


_REGISTRY: dict[str, Callable[..., Target]] = {
    LocalAgent.key: LocalAgent,
    RepositoryChatTarget.key: RepositoryChatTarget,
}


def register_target(key: str, factory: Callable[..., Target]) -> None:
    """Add a way of reaching an agent. A hosted runtime is a class and this line."""
    _REGISTRY[key] = factory


def resolve(key: str) -> Callable[..., Target]:
    if key not in _REGISTRY:
        raise NotImplementedError(
            f"no target {key!r}; registered targets are {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[key]


def supported() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
