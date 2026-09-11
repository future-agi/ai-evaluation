"""Replay an external benchmark's hand-written trajectories against a world we generated.

Every gate in this harness so far is one we wrote, checking work we produced. That is worth
something, but it cannot answer the question that actually matters about a generated
environment: **is it faithful to the agent it was built from?**

An independent benchmark answers it. Sierra's tau-bench ships hand-written tasks, each with the
exact tool calls a correct agent should make. Those trajectories were written by people who had
never seen this harness, against the real implementation. Replaying them through a world the
harness built automatically from the same source is therefore an external check: if the world is
faithful, the trajectories run clean; where they do not, the difference is a real defect in the
world and it is pointed at directly.

    .venv/bin/python scripts/replay_ground_truth.py \
        --world artifacts/environments/tau_retail \
        --tasks .../tau-bench/tau_bench/envs/retail/tasks_test.py

What it reports, per trajectory: every call accepted, or the first one the world refused or
crashed on. A refusal is the interesting case — either the trajectory relies on data our sample
does not have, or a handler is stricter than the real tool.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fi.alk.harness.world.snapshot import restore  # noqa: E402


@dataclass
class Trajectory:
    """One hand-written task: what the user wanted, and the calls a correct agent makes."""

    instruction: str
    actions: list[tuple[str, dict]] = field(default_factory=list)


def read_tasks(path: Path) -> list[Trajectory]:
    """The trajectories, read from the benchmark's own Python without importing it.

    Parsed rather than imported: importing would pull in the benchmark's package and its
    dependencies, and all that is wanted here is literal data it already states plainly.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[Trajectory] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Task"):
            continue
        instruction, actions = "", []
        for keyword in node.keywords:
            if keyword.arg == "instruction" and isinstance(keyword.value, ast.Constant):
                instruction = str(keyword.value.value)
            if keyword.arg == "actions" and isinstance(keyword.value, ast.List):
                for entry in keyword.value.elts:
                    if not (
                        isinstance(entry, ast.Call)
                        and getattr(entry.func, "id", "") == "Action"
                    ):
                        continue
                    name, kwargs = "", {}
                    for field_ in entry.keywords:
                        if field_.arg == "name" and isinstance(field_.value, ast.Constant):
                            name = str(field_.value.value)
                        if field_.arg == "kwargs":
                            try:
                                kwargs = ast.literal_eval(field_.value)
                            except ValueError:
                                kwargs = {}
                    if name:
                        actions.append((name, kwargs))
        if actions:
            found.append(Trajectory(instruction=instruction, actions=actions))
    return found


@dataclass
class Replay:
    index: int
    steps: int = 0
    accepted: int = 0
    stopped_at: str = ""
    why: str = ""
    crashed: bool = False

    @property
    def clean(self) -> bool:
        return not self.stopped_at


def replay(trajectory: Trajectory, world_root: Path, index: int) -> Replay:
    """One trajectory against its own fresh copy of the world."""
    result = Replay(index=index, steps=len(trajectory.actions))
    world = restore(world_root)
    try:
        world.reset()
        for name, arguments in trajectory.actions:
            call = world.call(name, arguments)
            if call.ok:
                result.accepted += 1
                continue
            result.stopped_at = f"{name}({json.dumps(arguments, default=str)[:120]})"
            result.why = call.error
            result.crashed = not call.refused
            break
    finally:
        world.close()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", required=True, help="a built environment")
    parser.add_argument("--tasks", required=True, help="the benchmark's tasks file")
    parser.add_argument("--limit", type=int, default=0, help="only the first N trajectories")
    parser.add_argument("--show", type=int, default=12, help="how many failures to detail")
    args = parser.parse_args(argv)

    world_root = Path(args.world)
    if not (world_root / "world.sqlite").exists():
        print(f"no world at {world_root}. Run `build` first.", file=sys.stderr)
        return 1

    trajectories = read_tasks(Path(args.tasks))
    if args.limit:
        trajectories = trajectories[: args.limit]
    if not trajectories:
        print("no trajectories found in that file", file=sys.stderr)
        return 1

    results = [replay(one, world_root, index) for index, one in enumerate(trajectories)]
    clean = [one for one in results if one.clean]
    crashed = [one for one in results if one.crashed]
    calls = sum(one.steps for one in results)
    accepted = sum(one.accepted for one in results)

    print(f"world:        {world_root}")
    print(f"trajectories: {len(results)} hand-written, from {Path(args.tasks).name}")
    print(f"replayed:     {len(clean)}/{len(results)} clean")
    print(f"calls:        {accepted}/{calls} accepted by the world")
    if crashed:
        print(f"crashes:      {len(crashed)} — these are defects in the world, not refusals")

    failed = [one for one in results if not one.clean]
    if failed:
        print("\nwhere they stopped:")
        for one in failed[: args.show]:
            mark = "CRASH" if one.crashed else "refused"
            print(f"  [{one.index}] {mark} after {one.accepted}/{one.steps}: {one.stopped_at}")
            print(f"        {one.why[:160]}")
        if len(failed) > args.show:
            print(f"  … and {len(failed) - args.show} more")

    # The tools a real suite actually exercises, which is what our own coverage is measured
    # against: a generated suite that never reaches the write tools has not tested the agent.
    used: dict[str, int] = {}
    for one in trajectories:
        for name, _ in one.actions:
            used[name] = used.get(name, 0) + 1
    print("\nwhat the hand-written trajectories exercise:")
    for name, count in sorted(used.items(), key=lambda pair: -pair[1]):
        print(f"  {count:4}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
