"""One scenario, against the real hosted agent, in the environment the harness built.

The harness wires the whole thing rather than leaving it to be assembled by hand:

1. restore the world and apply the scenario's setup
2. stand the webhook up and bind that world to it
3. expose it publicly, because a hosted agent has to reach it
4. point the assistant's **own** tools at that address — nothing about the agent is redefined
5. run ALK's voice case with the scenario's instruction driving the simulated caller
6. grade from the world afterwards and the calls the webhook recorded

Steps 1, 2, 4 and 6 are the whole difference from what existed before: the agent's tool calls now
land in a database that can refuse, instead of in canned responses that always succeed.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ..simulator_voice import fixture_caller_phone
from ..catalogue import load_catalogue
from ..checks import Outcome, run_check
from ..folder import apply_setup, check_ready
from ..scenario import Scenario
from ..simulator import fill, load_simulator_prompt
from ..world.runtime import GeneratedWorld
from ..world.snapshot import restore
from .voice import WorldWebhook, repoint_assistant


@dataclass
class LiveRun:
    """What a live call left behind."""

    scenario: str
    settled: list[Outcome] = field(default_factory=list)
    judged: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    ended: str = ""
    problems: list[str] = field(default_factory=list)

    @property
    def met(self) -> int:
        return sum(1 for one in self.settled if one.held)

    def line(self) -> str:
        mark = "PASS" if self.settled and self.met == len(self.settled) else "FAIL"
        if self.problems:
            mark = "VOID"
        return f"{mark}  {self.scenario}  {self.met}/{len(self.settled)} sub-goals settled by code"


def scoped_agent_name(base: str, scenario: str) -> str:
    """A dispatch name owned by one worker lifetime, never a stale registration."""
    clean_base = re.sub(r"[^a-zA-Z0-9_-]+", "-", base).strip("-") or "agent"
    clean_case = re.sub(r"[^a-zA-Z0-9_-]+", "-", scenario).strip("-") or "case"
    return f"{clean_base[:28]}-{clean_case[:18]}-{uuid.uuid4().hex[:8]}"


def public_url(
    port: int, *, wait: float = 30.0, tries: int = 3
) -> tuple[str, subprocess.Popen | None]:
    """A publicly reachable address for the webhook, and the process holding it open.

    A hosted agent runs on somebody else's infrastructure, so a loopback address is unreachable
    to it. ``cloudflared`` is what the previous runs used; anything giving a public URL works, and
    ``HARNESS_WEBHOOK_URL`` skips this entirely when a tunnel is already running.

    Retried, because a free tunnel is the least reliable thing in the whole path and it fails
    before anything interesting has happened. One slow handshake should not read as a scenario
    the agent failed, and on a suite of forty it would not fail once.
    """
    named = os.environ.get("HARNESS_WEBHOOK_URL", "").strip()
    if named:
        return named, None
    if not shutil.which("cloudflared"):
        raise RuntimeError(
            "no way to expose the webhook publicly. Either install cloudflared "
            "(brew install cloudflared) or set HARNESS_WEBHOOK_URL to a tunnel you already have."
        )
    for attempt in range(max(1, tries)):
        found, process = _tunnel(port, wait)
        if found:
            return found, process
        if process is not None:
            process.terminate()
        if attempt + 1 < tries:
            time.sleep(2.0)
    raise RuntimeError(
        f"cloudflared did not report a public URL in {tries} attempts. The tunnel is the "
        "flakiest part of this path; set HARNESS_WEBHOOK_URL to one you control to skip it."
    )


def _tunnel(port: int, wait: float) -> tuple[str, subprocess.Popen | None]:
    """One attempt at a tunnel: the URL if it came up, and the process either way."""
    process = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.time() + wait
    while time.time() < deadline:
        line = process.stdout.readline() if process.stdout else ""
        if not line and process.poll() is not None:
            return "", process
        if "trycloudflare.com" in line:
            for word in line.split():
                if word.startswith("https://") and "trycloudflare.com" in word:
                    return word.strip(), process
    return "", process


def prepare(scenario: Scenario, world_root: Path) -> tuple[GeneratedWorld, str]:
    """The world this scenario runs in, and what the simulated caller is told.

    The instruction is the scenario's values filled into the simulator prompt the environment
    step wrote. Nothing about how a caller behaves is decided here; that belongs to the prompt.
    """
    world = restore(world_root)
    world.reset()
    applied = apply_setup(scenario, world)
    if not applied.ok:
        raise RuntimeError(f"the scenario's setup did not run: {applied.said}")
    ready = check_ready(scenario, world)
    if not ready.ok:
        raise RuntimeError(
            f"the world is not ready for this scenario: {ready.said}. Running it would test us "
            "rather than the agent."
        )
    # The setup's own calls are not the agent's.
    world.calls = []

    written = load_simulator_prompt(world_root)
    if not written:
        return world, scenario.instruction
    filled, missing = fill(written, scenario.slots())
    if missing:
        raise RuntimeError(
            f"the simulator prompt asks for {', '.join(missing)}, which {scenario.name} does "
            "not supply. An unfilled slot reaches the caller verbatim."
        )
    return world, filled


def grade(scenario: Scenario, world: GeneratedWorld, world_root: Path) -> LiveRun:
    """The same sub-goal checks every other run uses, against what the call left behind."""
    catalogue = load_catalogue(world_root)
    run = LiveRun(scenario=scenario.name)
    for name in scenario.sub_goals:
        sub_goal = catalogue.named(name)
        if sub_goal is None:
            run.problems.append(f"{name} is not in the catalogue")
        elif sub_goal.deterministic():
            run.settled.append(run_check(sub_goal.check, world, world.calls, name=name))
        else:
            run.judged.append(name)
    run.calls = [
        f"{call.name}({call.arguments}) -> "
        + ("refused: " + call.error if call.refused else "ok" if call.ok else "crashed")
        for call in world.calls
    ]
    return run


def instruction_for(scenario: Scenario, world_root: Path) -> str:
    """What the simulated person is told, from the prompt the environment step wrote."""
    written = load_simulator_prompt(world_root)
    if not written:
        return scenario.instruction
    filled, missing = fill(written, scenario.slots())
    if missing:
        raise RuntimeError(
            f"the simulator prompt asks for {', '.join(missing)}, which {scenario.name} does "
            "not supply. An unfilled slot reaches the caller verbatim."
        )
    return filled


def fixture_phone(scenario: Scenario) -> str:
    """The caller identity the submitted voice runtime must see for this scenario."""
    return fixture_caller_phone(scenario.fixture)


def wire(
    scenario: Scenario,
    world_root: Path,
    *,
    assistant_id: str = "",
    api_key: str = "",
    world: GeneratedWorld | None = None,
    trace_path: str | Path | None = None,
):
    """Everything up to placing the call: world, webhook, tunnel, assistant.

    Returns the bound world, the caller's instruction, the webhook and the tunnel, so whoever
    places the call decides how — ALK's voice case, a phone leg, or a web call.

    ``world`` is taken when the caller has already prepared one. The suite runner sets a
    scenario's world up once and grades what that same world is left holding, so preparing a
    second one here would answer the agent's calls in a world nobody afterwards looks at.
    """
    # How the agent is reached decides what has to be arranged here. A hosted assistant lives
    # somewhere we do not control, so its tools have to be repointed at a URL it can reach from
    # outside. An agent we run ourselves already reads where its tools are from its own
    # environment and shares a network with us, so there is nothing to repoint and nothing to
    # expose -- and doing either would fail for want of credentials we have no reason to hold.
    reachable = os.environ.get("HARNESS_WEBHOOK_URL", "").strip()
    source_environment = (Path(world_root) / "environment.json").exists()
    ours = bool(reachable) or source_environment

    if not ours:
        assistant_id = assistant_id or os.environ.get("VAPI_ASSISTANT_ID", "")
        api_key = api_key or os.environ.get("VAPI_API_KEY", "")
        if not assistant_id or not api_key:
            raise RuntimeError(
                "VAPI_ASSISTANT_ID and VAPI_API_KEY have to be set, or HARNESS_WEBHOOK_URL "
                "given for an agent that already knows where to find its tools."
            )

    if world is None:
        world, instruction = prepare(scenario, world_root)
    else:
        instruction = instruction_for(scenario, world_root)
    webhook = WorldWebhook().start()
    webhook.bind(world)
    try:
        if source_environment:
            from ..provision import (
                connect_runner_network,
                infer_livekit_agent_name,
                start_runtime,
            )

            # The submitted worker runs in Docker while the webhook runs in this harness
            # process. The host-gateway name is injected by start_runtime and keeps the source
            # network private; only this one URL is substituted.
            private_host = connect_runner_network(world_root)
            url = os.environ.get("HARNESS_RUNTIME_WEBHOOK_URL", "").strip() or (
                f"http://{private_host}:{webhook.port}"
                if private_host
                else f"http://host.docker.internal:{webhook.port}"
            )
            # Reusing a registered LiveKit agent name across rapid container restarts lets a new
            # room dispatch to the just-removed worker during server-side deregistration grace.
            # Give every worker lifetime its own name and point this call at that exact worker.
            base_agent_name = infer_livekit_agent_name(world_root) or os.environ.get(
                "LIVEKIT_TARGET_AGENT_NAME", "harness-agent"
            )
            agent_name = scoped_agent_name(base_agent_name, scenario.name)
            runtime_overrides = {
                "TOOLS_API_URL": url,
                "LIVEKIT_AGENT_NAME": agent_name,
            }
            # The submitted agent picks its own model, and a tier that cannot emit a valid
            # function call fails every scenario the moment it reaches for a tool. Allow an
            # operator to pin it for a run without editing the submitted repository.
            agent_model = os.environ.get("ALK_SUBMITTED_AGENT_MODEL", "").strip()
            if agent_model:
                runtime_overrides["AGENT_LLM_MODEL"] = agent_model
            os.environ["LIVEKIT_TARGET_AGENT_NAME"] = agent_name
            caller_phone = fixture_phone(scenario)
            if caller_phone:
                runtime_overrides["DEMO_CALLER_ANI"] = caller_phone
            start_runtime(
                world_root,
                overrides=runtime_overrides,
                trace_path=trace_path,
            )
            # Runtime-only projects have no dependency container (and therefore no Compose
            # network) until the worker starts. The first call above reserves the alias; this
            # second idempotent call joins a hosted runner to the newly created network.
            connect_runner_network(world_root)
            tunnel, moved = None, []
        elif ours:
            # The agent was started pointing here, so this is where its tools already go.
            url, tunnel, moved = reachable, None, []
        else:
            url, tunnel = public_url(webhook.port)
            moved = repoint_assistant(assistant_id, api_key, url)
    except Exception:
        webhook.stop()
        if source_environment:
            from ..provision import stop_runtime

            stop_runtime(world_root)
        world.close()
        raise
    return world, instruction, webhook, tunnel, url, moved
