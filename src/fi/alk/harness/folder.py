"""A scenario as a folder of files, and running the code inside it.

A scenario used to be a row in one big JSON file, and its setup was a list of rows to insert.
That was enough while every world was a database. It stopped being enough the moment a world
could hold a service as well as a table: "the weather service starts returning errors" is not
expressible as rows, and neither is "the file is missing" or "the queue is backed up".

So a scenario owns a folder, and the parts that are logic are files:

    scenarios/<name>/
        scenario.json     what it is: instruction, solution, which sub-goals
        setup.py          def setup(world)  — the changes this scenario makes
        ready.py          def ready(world)  — is the world ready for this scenario
        checks/<goal>.py  def check(world, calls) — one per deterministic sub-goal

The files are the artifact, not a rendering of one. Each is executable on its own, so a check
can be run by hand against what a run left behind and answer exactly what it answers inside the
harness. That is the whole point of them being files: something you can open, read and run is
something you can argue with.
"""

from __future__ import annotations

import ast
import json
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalogue import Catalogue, SubGoal
from .scenario import Scenario
from .world.runtime import GeneratedWorld

SCENARIOS = "scenarios"
INDEX = "scenarios.json"

# Appended to every check file the harness writes. The model writes only ``check(world, calls)``;
# this is what makes that same file runnable by a person, so nobody has to keep two versions of
# one truth in step.
_RUNNABLE = """

if __name__ == "__main__":
    # Run this check by hand against what a run left behind. The first argument is anything
    # inside the saved world's folder, because not every world has a database to name:
    #     python <this file> <world folder>/manifest.json [calls.json]
    import json as _json
    import sys as _sys
    from pathlib import Path as _Path

    from fi.alk.harness.world.runtime import Call as _Call
    from fi.alk.harness.world.snapshot import restore as _restore

    _world = _restore(_Path(_sys.argv[1]).parent) if len(_sys.argv) > 1 else None
    _calls = []
    if len(_sys.argv) > 2:
        _calls = [_Call(**_one) for _one in _json.loads(_Path(_sys.argv[2]).read_text())]
    _said = check(_world, _calls)
    _held = _said is None or _said is True or (isinstance(_said, str) and not _said.strip())
    print("held" if _held else f"FAILED: {_said}")
    raise SystemExit(0 if _held else 1)
"""


@dataclass
class Outcome:
    """What one piece of a scenario's own code did."""

    ok: bool
    said: str = ""
    broken: bool = False


def _run(source: str, name: str, entry: str, *args: Any) -> Outcome:
    """Execute one function out of a scenario's own code.

    A file that will not compile, or that raises, is **broken** rather than failing: it is our
    mistake, and scoring it as though the world were wrong would send somebody looking in the
    wrong place.
    """
    if not source.strip():
        return Outcome(True)
    namespace: dict[str, Any] = {}
    try:
        exec(compile(source, f"<{name}>", "exec"), namespace)
    except Exception as failed:
        return Outcome(False, f"{name} would not compile: {failed}", broken=True)

    function = namespace.get(entry)
    if not callable(function):
        return Outcome(False, f"{name} defines no {entry}()", broken=True)
    try:
        said = function(*args)
    except Exception as failed:
        return Outcome(
            False, f"{name} raised {type(failed).__name__}: {failed}", broken=True
        )
    # The convention is that a complaint is a sentence, and anything else means it held. An empty
    # string is the case worth naming: it reads as "no complaint" to whoever wrote it, and taking
    # it as a failure produces a rejection with no reason attached, which cannot be acted on and
    # sends the author hunting for a problem that is not there.
    if said is None or said is True or (isinstance(said, str) and not said.strip()):
        return Outcome(True)
    if said is False:
        # Bare False from ready() names no precondition, so a scenario that hits it cannot be
        # told apart from one whose ready.py is simply wrong — that is our mistake, not a
        # generation-time precondition failure, so ready() alone reports it broken.
        return Outcome(
            False,
            f"{name} returned False without saying what is wrong. Return the sentence instead, "
            "or None if it holds.",
            broken=(entry == "ready"),
        )
    if entry == "ready" and not isinstance(said, str):
        # Same reasoning as bare False, widened: ready() has no way to turn a non-string value
        # into a precondition sentence, so any of them is our mistake rather than the world's.
        return Outcome(
            False,
            f"{name} returned {type(said).__name__} {repr(said)[:200]}. Return the sentence "
            "naming what is missing, or None if it holds.",
            broken=True,
        )
    return Outcome(False, str(said))


def apply_setup(scenario: Scenario, world: GeneratedWorld) -> Outcome:
    """Make this scenario's changes to the world."""
    return _run(scenario.setup_code, f"{scenario.name}/setup.py", "setup", world)


def check_ready(scenario: Scenario, world: GeneratedWorld) -> Outcome:
    """Whether the world now holds what this scenario presumes."""
    return _run(scenario.ready_code, f"{scenario.name}/ready.py", "ready", world)


def folder_for(destination: Path, name: str) -> Path:
    return Path(destination) / SCENARIOS / name


def document_for(scenario: Scenario) -> dict:
    """One scenario as JSON: everything about it except the code, which lives in its own files."""
    body = scenario.model_dump()
    body.pop("setup_code", None)
    body.pop("ready_code", None)
    if not body.get("sub_actors"):
        body.pop("sub_actors", None)
    return body


def write_folder(scenario: Scenario, catalogue: Catalogue, destination: Path) -> Path:
    """Write one scenario out as its own folder of files."""
    root = folder_for(destination, scenario.name)
    (root / "checks").mkdir(parents=True, exist_ok=True)

    body = document_for(scenario)
    (root / "scenario.json").write_text(
        json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    (root / "setup.py").write_text(
        scenario.setup_code
        or 'def setup(world):\n    """This scenario runs on the base world unchanged."""\n',
        encoding="utf-8",
    )
    (root / "ready.py").write_text(
        scenario.ready_code
        or 'def ready(world):\n    """Nothing beyond the base world is presumed."""\n',
        encoding="utf-8",
    )

    for name in scenario.sub_goals:
        sub_goal = catalogue.named(name)
        if sub_goal is None or not sub_goal.deterministic():
            continue
        (root / "checks" / f"{name}.py").write_text(
            sub_goal.check.rstrip() + "\n" + _RUNNABLE, encoding="utf-8"
        )

    # Drop checks for sub-goals a rewrite removed.
    wanted = {f"{name}.py" for name in scenario.sub_goals}
    for stale in (root / "checks").glob("*.py"):
        if stale.name not in wanted:
            stale.unlink()
    return root


def refresh_check(destination: Path, sub_goal: SubGoal) -> list[str]:
    """Bring every folder that names this sub-goal into step with its new definition."""
    root = Path(destination) / SCENARIOS
    if not root.is_dir():
        return []
    touched: list[str] = []
    for folder in sorted(one for one in root.iterdir() if one.is_dir()):
        body = folder / "scenario.json"
        if not body.is_file():
            continue
        try:
            named = json.loads(body.read_text(encoding="utf-8")).get("sub_goals") or []
        except (OSError, ValueError):
            continue
        if sub_goal.name not in named:
            continue
        path = folder / "checks" / f"{sub_goal.name}.py"
        if sub_goal.deterministic():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                sub_goal.check.rstrip() + "\n" + _RUNNABLE, encoding="utf-8"
            )
        elif path.is_file():
            path.unlink()
        else:
            continue
        touched.append(folder.name)
    return touched


def read_folder(destination: Path, name: str) -> Scenario | None:
    """One scenario, reassembled from its folder."""
    root = folder_for(destination, name)
    body = root / "scenario.json"
    if not body.exists():
        return None
    payload = json.loads(body.read_text(encoding="utf-8"))
    for field, filename in (("setup_code", "setup.py"), ("ready_code", "ready.py")):
        path = root / filename
        payload[field] = path.read_text(encoding="utf-8") if path.exists() else ""
    return Scenario.model_validate(payload)


def write_index(scenarios: list[Scenario], destination: Path) -> Path:
    """The whole suite, over the folders.

    Regenerated from the folders rather than maintained alongside them, so it can never disagree
    with what is actually on disk.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / INDEX
    path.write_text(
        json.dumps(
            [
                {
                    **document_for(one),
                    "steps": len(one.solution),
                    "folder": f"{SCENARIOS}/{one.name}",
                }
                for one in scenarios
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def read_all(destination: Path) -> list[Scenario]:
    """Every scenario on disk, read from the folders."""
    root = Path(destination) / SCENARIOS
    if not root.exists():
        return []
    found: list[Scenario] = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        try:
            scenario = read_folder(destination, folder.name)
        except Exception:
            # A folder we cannot read is skipped rather than crashing the stage: the rest of the
            # suite is still usable, and the gap shows up as a missing scenario.
            continue
        if scenario is not None:
            found.append(scenario)
    return found


def _tools_selected(body: str) -> set[str]:
    """Every tool name this check narrows the calls to."""
    try:
        tree = ast.parse(textwrap.dedent(body))
    except SyntaxError:
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or not node.ops:
            continue
        if not isinstance(node.ops[0], ast.Eq):
            continue
        left, right = node.left, node.comparators[0]
        if isinstance(right, ast.Attribute) and right.attr == "name":
            left, right = right, left
        if not (isinstance(left, ast.Attribute) and left.attr == "name"):
            continue
        if isinstance(right, ast.Constant) and isinstance(right.value, str):
            found.add(right.value)
    return found


def unchecked_sub_goals(folder: Path, catalogue: Catalogue) -> list[str]:
    """Scenarios naming a sub-goal the catalogue settles in code with no check file to settle it."""
    settled = {one.name for one in catalogue.sub_goals if one.deterministic()}
    if not settled:
        return []
    problems: list[str] = []
    for scenario_dir in sorted(
        one for one in (Path(folder) / SCENARIOS).glob("*") if one.is_dir()
    ):
        body = scenario_dir / "scenario.json"
        if not body.is_file():
            continue
        try:
            named = json.loads(body.read_text(encoding="utf-8")).get("sub_goals") or []
        except (OSError, ValueError):
            continue
        missing = [
            name
            for name in named
            if name in settled
            and not (scenario_dir / "checks" / f"{name}.py").is_file()
        ]
        if missing:
            problems.append(
                f"{scenario_dir.name}: {', '.join(sorted(missing))} settled in code by the "
                "catalogue but no check file was written, so nothing would measure it. Define "
                "it again with add_sub_goal and save"
            )
    return problems


def unasserted_behaviour(scenarios: list[Scenario], folder: Path) -> list[str]:
    """The claimed last call of a reference solution that no check for that scenario reads."""
    problems: list[str] = []
    for scenario in scenarios:
        used = [step.tool for step in (scenario.solution or []) if step.tool]
        if not used:
            continue
        checks = folder_for(folder, scenario.name) / "checks"
        if not checks.is_dir():
            continue
        asserted = ""
        for check in sorted(checks.glob("*.py")):
            asserted += check.read_text(encoding="utf-8", errors="replace").split("if __name__")[0]
        outcome = used[-1]
        claimed = f"{scenario.name} {scenario.tests or ''}".lower()
        spoken = [word for word in re.split(r"[^a-z]+", outcome.lower()) if len(word) > 3]
        # Adjacent words only: matching them separately flagged unrelated tests lines.
        together = re.search(r"\W+".join(spoken), claimed) if spoken else None
        if outcome not in asserted and together:
            problems.append(
                f"{scenario.name}: it says it tests {outcome}, its reference solution ends there, "
                "and no check mentions it, so an agent that stops short of it passes anyway"
            )
    return problems


def check_problems(folder: Path) -> list[str]:
    """Checks that touch neither world nor arguments, or that read the same tool call twice."""
    problems: list[str] = []
    plumbing = 0
    for scenario_dir in sorted(one for one in (folder / SCENARIOS).glob("*") if one.is_dir()):
        by_tool: dict[str, list[str]] = defaultdict(list)
        for check in sorted(scenario_dir.glob("checks/*.py")):
            body = check.read_text(encoding="utf-8").split("if __name__")[0]
            inside = body.replace("def check(world, calls):", "")
            reads_world = "world" in inside
            reads_arguments = "arguments" in inside
            if not reads_world and not reads_arguments:
                plumbing += 1
            if not reads_world:
                for tool in _tools_selected(inside):
                    by_tool[tool].append(check.stem)
        for tool, names in sorted(by_tool.items()):
            if len(names) > 1:
                problems.append(
                    f"{scenario_dir.name}: {', '.join(sorted(names))} all read the same {tool} call "
                    "and none of them reads the world, so they are one claim written more than once"
                )
    if plumbing:
        problems.append(
            f"{plumbing} checks assert only that a tool was called, touching neither its arguments "
            "nor the world. Every agent that reaches the tool passes them."
        )
    return problems
