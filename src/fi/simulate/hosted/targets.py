"""Resolve the agent-under-test target from a job's ``SimulationSpec.target``.

The runner never runs the target agent itself — the target is the customer's
deployed (or supplied) agent. For chat runs the target is a turn-based surface,
resolved through the one endpoint registry: ``spec.target.adapter`` names the
actor-source kind (``callable`` / ``python_callable`` / ``import_object`` /
``factory`` / ``framework`` / ``system_prompt`` / ``http`` / …) and each
registered ``EndpointProfile`` carries the resolver. Adding a target kind is one
profile entry, no edits here (plan §4.1).

This is a HOSTED execution path (the runner runs it on our infra), so target
kinds that execute caller-supplied Python in-process (``callable`` /
``python_callable`` / ``import_object`` / ``factory`` / ``framework``) are
**rejected here** — deny-by-default via ``EndpointProfile.runs_caller_code``.
Hosted runs must reach the agent as a deployed endpoint (``http`` / ``websocket``)
or through the sandboxed runtime. The only in-process escape is a trusted
operator-configured default target, opted in explicitly with
``ALK_UNSAFE_INPROCESS_CODE_ACTORS`` — never set in prod for untrusted jobs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fi.simulate.agent.wrapper import AgentWrapper
from fi.simulate.runtime.spec import SimulationSpec


def resolve_chat_target(spec: SimulationSpec) -> Callable[..., Any] | AgentWrapper:
    from fi.simulate.endpoints.actor_sources import (
        ActorSourceError,
        inprocess_code_allowed,
    )
    from fi.simulate.endpoints.profiles import get_profile

    adapter = (spec.target.adapter or "").lower()
    profile = get_profile(adapter)
    if profile is None or not profile.is_turn_based_target:
        raise ValueError(f"unsupported_chat_target_adapter: {spec.target.adapter}")
    if profile.runs_caller_code and not inprocess_code_allowed():
        raise ActorSourceError(
            f"code_actor_denied_in_hosted: target {adapter!r} would run "
            f"caller-supplied code in the runner process. Hosted runs must use a "
            f"deployed endpoint (http/websocket) or the sandboxed runtime; "
            f"in-process code is developer/local only."
        )
    return profile.resolve_target(
        dict(spec.target.config or {}), spec.target.secret_refs, hosted=True
    )


__all__ = ["resolve_chat_target"]
