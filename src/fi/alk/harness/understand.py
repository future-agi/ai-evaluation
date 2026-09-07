"""Stage one: read an agent and produce its contract.

The stage is the same whatever the agent is. What changes between a repository, a provider
connection and a pasted definition is where the truth lives, and that comes from the source.

It stays open after the first answer, because a contract is usually right on the second look and
not the first. Correcting it is the next thing said, not a re-run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from .config import artifact_dir, load_skill, read_only_session
from .contract import AgentContract
from .session import Stage
from .sources import AgentSource
from .tools import CONTRACT_SERVER, contract_tools

SKILL = "understand-agent"
PROVIDER_IMPORT_PROFILE_PATH_ENV = "ALK_PROVIDER_IMPORT_PROFILE_PATH"


def _provider_import_briefing() -> str:
    """Load the sanitized external-provider definition prepared by the control process."""
    configured = os.environ.get(PROVIDER_IMPORT_PROFILE_PATH_ENV, "").strip()
    if not configured:
        return ""
    path = Path(configured)
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return (
        "\n\n## Imported provider target (authoritative, read-only)\n\n"
        "The submitted repository implements this target's environment-facing webhooks. "
        "The externally hosted assistant definition below is equally authoritative for its "
        "conversation, prompt, model, voice, and tool schemas. Reconcile both sources. Do not "
        "invent behavior or tool inputs that conflict with the provider definition.\n\n"
        f"```json\n{json.dumps(profile, indent=2, sort_keys=True)}\n```"
    )


def _eval_catalogue_briefing(available_evals: list[dict[str, Any]] | None) -> str:
    """The eval catalogue grouped by modality, since the contract has not claimed one yet."""
    grouped: dict[str, list[str]] = {}
    for one in available_evals or []:
        if not isinstance(one, dict):
            continue
        name = str(one.get("name") or "").strip()
        if not name:
            continue
        keys = ", ".join(str(key) for key in one.get("required_keys") or [])
        grouped.setdefault(str(one.get("modality") or "any").strip().lower(), []).append(
            "- {name}: {description}{keys}".format(
                name=name,
                description=str(one.get("description") or "").strip()[:240],
                keys=f" [needs: {keys}]" if keys else "",
            )
        )
    if not grouped:
        return ""
    sections = [
        f"### Applies to {kind}\n\n" + "\n".join(sorted(lines))
        for kind, lines in sorted(grouped.items())
    ]
    return (
        "\n\n## Evals this platform can run on the finished calls\n\n"
        "Record the ones this agent should be judged by in `chosen_evals`, by exact name. Each one "
        "runs as a judge on every call of every scenario, so choose the few that tell you "
        "something the scenarios' own deterministic checks cannot, and choose none rather than "
        "padding.\n\n"
        "Choose only from the section matching the `modality` you are about to record. The "
        "platform refuses an eval belonging to another modality, so an eval for spoken calls "
        "recorded against a chat agent costs the whole submission.\n\n"
        "Two ways to choose wrongly, both common. An eval whose subject only exists in speech, "
        "dead air, voicemail detection, voicemail handling, is meaningless for a chat agent and "
        "worth having for a voice one. An eval named for a domain, misselling, advice authority, "
        "lead qualification, claim intake, intake field accuracy, is only worth choosing when this "
        "agent's own tools and constraints show it doing that work; choose it from the evidence in "
        "front of you, never because its name sounds close to the agent's industry.\n\n"
        + "\n\n".join(sections)
    )


def open_stage(
    source: AgentSource,
    *,
    out: Path | None = None,
    ask: Callable[..., Any] | None = None,
    max_turns: int = 70,
    available_evals: list[dict[str, Any]] | None = None,
) -> tuple[Stage, Path]:
    """A live understand-the-agent stage, and where it will write."""
    destination = out or artifact_dir(source.name)
    spec = read_only_session(
        system_prompt=(
            f"{load_skill(SKILL)}\n\n## This agent\n\n{source.briefing()}"
            f"{_provider_import_briefing()}"
            f"{_eval_catalogue_briefing(available_evals)}"
        ),
        cwd=source.workdir(),
        servers={
            **source.servers(),
            CONTRACT_SERVER: contract_tools(destination, available_evals),
        },
        extra_builtins=source.builtin_tools(),
        max_turns=max_turns,
    )
    if ask is not None:
        spec.permission_override = ask
    return Stage(spec, name=SKILL), destination


def opening(source: AgentSource) -> str:
    # The name is only a label for the artifact folder, and saying so matters: told to "read
    # the agent named verify_fix", a model went hunting the whole workspace for something
    # called verify_fix instead of reading the path it was given.
    return (
        "Read this agent and produce its contract. Where it lives is in your briefing; "
        f"{source.name!r} is only the label its artifacts are filed under, not something to "
        "search for.\n\n"
        "Work through the tools, their exact argument names and types, the constrained argument "
        "values, the rules it enforces, and its data. Ask me if the source genuinely does not "
        "settle something that changes what gets built. Call submit_contract when you are done."
    )


def load(destination: Path) -> AgentContract | None:
    """The contract on disk, if the stage produced one."""
    path = Path(destination) / "contract.json"
    if not path.exists():
        return None
    return AgentContract.model_validate(json.loads(path.read_text(encoding="utf-8")))


async def understand(
    source: AgentSource,
    *,
    out: Path | None = None,
    follow_ups: list[str] | None = None,
    on_event: Callable[..., Any] | None = None,
    ask: Callable[..., Any] | None = None,
    max_turns: int = 70,
) -> AgentContract | None:
    """Run the stage start to finish and return the contract.

    ``follow_ups`` are corrections applied in the same session, the scripted equivalent of an
    operator typing them. ``ask`` handles clarifying questions; without it the model records what
    it could not resolve in ``open_questions`` instead of blocking.
    """
    stage, destination = open_stage(source, out=out, ask=ask, max_turns=max_turns)
    async with stage:
        await stage.say(opening(source), on_event=on_event)
        for follow_up in follow_ups or []:
            await stage.say(follow_up, on_event=on_event)
    return load(destination)
