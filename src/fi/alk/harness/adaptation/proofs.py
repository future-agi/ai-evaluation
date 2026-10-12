"""Proving candidate scenarios for adaptation, through the release sealer only.

The sealer proves a whole suite or refuses it, so a scenario is proved on its own by sealing a
scratch suite that holds just that scenario. Nothing here re-implements a gate: every verdict is
`seal_suite_release` (SQLite and PostgreSQL) or `prove_binding`.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..catalogue import Catalogue, save_catalogue
from ..releases import (
    PolicyProfile,
    ReleaseError,
    ScenarioOrigin,
    SuiteScenario,
    load_suite_release,
    seal_suite_release,
)

SIMULATOR_PROMPT = "simulator_prompt.md"


@dataclass(frozen=True)
class Candidate:
    """A scenario folder that may go into a suite: where its files are and what it would be."""

    entry: SuiteScenario
    source: Path

    @property
    def key(self) -> str:
        return self.entry.key

    @property
    def folder(self) -> str:
        return Path(self.entry.path).name


def write_suite_tree(
    destination: Path,
    candidates: Sequence[Candidate],
    catalogue: Catalogue,
    simulator_prompt: str,
) -> None:
    """Lay out an unsealed suite: the scenario folders, the sub-goals they use, the prompt."""
    destination.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    for candidate in candidates:
        shutil.copytree(candidate.source, destination / candidate.entry.path)
        used.update(_sub_goal_names(candidate.source))
    save_catalogue(
        catalogue.model_copy(
            update={
                "sub_goals": [goal for goal in catalogue.sub_goals if goal.name in used]
            }
        ),
        destination,
    )
    (destination / SIMULATOR_PROMPT).write_text(simulator_prompt, encoding="utf-8")


def _sub_goal_names(folder: Path) -> list[str]:
    body = json.loads((folder / "scenario.json").read_text(encoding="utf-8"))
    return [str(name) for name in body.get("sub_goals") or []]


def _seal_scratch(
    workdir: Path,
    candidates: Sequence[Candidate],
    catalogue: Catalogue,
    simulator_prompt: str,
    *,
    environment_root: Path,
    profile: PolicyProfile,
    postgres_dsn: str,
) -> Path:
    root = Path(tempfile.mkdtemp(prefix="scratch-suite-", dir=workdir))
    write_suite_tree(root, candidates, catalogue, simulator_prompt)
    seal_suite_release(
        root,
        name="adaptation_scratch",
        environment_root=environment_root,
        scenarios=[
            candidate.entry.model_copy(
                update={"origin": ScenarioOrigin.BASE, "derived_from": None}
            )
            for candidate in candidates
        ],
        policy_profile=profile,
        postgres_dsn=postgres_dsn,
    )
    return root


def prove_candidates(
    workdir: Path,
    candidates: Sequence[Candidate],
    catalogue: Catalogue,
    simulator_prompt: str,
    *,
    environment_root: Path,
    profile: PolicyProfile,
    postgres_dsn: str,
) -> tuple[dict[str, str], Path | None]:
    """Why each candidate does not prove under `profile`, and a sealed suite of those that do.

    The whole set is tried first; only a refusal costs one scratch seal per scenario.
    """
    if not candidates:
        return {}, None
    options = {
        "environment_root": environment_root,
        "profile": profile,
        "postgres_dsn": postgres_dsn,
    }
    try:
        return {}, _seal_scratch(
            workdir, candidates, catalogue, simulator_prompt, **options
        )
    except ReleaseError as exc:
        if exc.code != "suite_scenario_unproved":
            raise
    failures: dict[str, str] = {}
    held: list[Candidate] = []
    for candidate in candidates:
        try:
            _seal_scratch(workdir, [candidate], catalogue, simulator_prompt, **options)
        except ReleaseError as exc:
            if exc.code != "suite_scenario_unproved":
                raise
            failures[candidate.key] = exc.detail
            continue
        held.append(candidate)
    if not held:
        return failures, None
    return failures, _seal_scratch(
        workdir, held, catalogue, simulator_prompt, **options
    )


@dataclass(frozen=True)
class BindingVerdict:
    status: str  # proved, failed or unscorable
    reason: str


def binding_verdicts(proof: Any) -> dict[str, BindingVerdict]:
    """Each scenario's standing under a binding proof; a variant that held fails its scenario."""
    verdicts: dict[str, BindingVerdict] = {}
    document = proof.model_dump(mode="json")
    for key, scenario in (document.get("scenarios") or {}).items():
        status = str(scenario.get("status"))
        reason = str(scenario.get("reason") or "")
        held = sorted(
            variant
            for variant, outcome in (scenario.get("adversarial") or {}).items()
            if outcome == "held"
        )
        if status == "proved" and held:
            status = "failed"
            reason = (
                "the checks still hold when the trajectory is altered "
                f"({', '.join(held)}), so they do not depend on the agent's calls"
            )
        verdicts[key] = BindingVerdict(status=status, reason=reason)
    return verdicts


def prove_binding_on(
    suite_root: Path,
    binding: Any,
    trajectories: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    environment_root: Path,
    postgres_dsn: str,
) -> Any:
    from ..binding_proof import prove_binding

    keys = _suite_keys(suite_root)
    return prove_binding(
        environment_root,
        suite_root,
        binding,
        {key: list(calls) for key, calls in trajectories.items() if key in keys},
        postgres_dsn=postgres_dsn,
    )


def _suite_keys(suite_root: Path) -> set[str]:
    return {entry.key for entry in load_suite_release(suite_root).scenarios}
