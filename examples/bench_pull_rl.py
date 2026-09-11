"""Pull / RL benchmark — the AGENT drives a live environment via reset/step.

The push lane has the harness drive the agent; ``artifact_in`` scores a submitted
artifact. **Pull** inverts control: the agent is a policy ``obs -> action`` that
steps a deterministic in-process environment until done, and the score is the
environment's cumulative reward. This is the Gym / OpenEnv env *shape*, run live
(reset/step), not replayed.

Loads the shipped ``pull_starter`` bench suite (two envs: ``reach_target`` and
``guess_number``) and runs it under ``control_mode="pull"`` with the env's own
reference policy (``{"type": "reference"}``), so the example is deterministic and
credential-free. Swap in a callable ``obs -> action`` to drive your own policy.

Run it::

    python examples/bench_pull_rl.py artifacts/bench-pull-rl.json

A live external env server (an HTTP reset/step endpoint) is the same contract
with a network transport; it plugs in as another ``Environment`` without changing
the driver or the unified Result, and is deferred to owner infra.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import bench

SUITE_PATH = Path(__file__).parent / "bench_suites" / "pull_starter.json"
OUTPUT_KIND = "agent-learning.bench-pull-rl.v1"


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    # agent = the policy. {"type": "reference"} uses each env's optimal policy
    # (proves the suite is solvable, deterministically). Pass a callable
    # obs -> action to drive your own policy instead.
    # emit_telemetry=False keeps the docs/release fresh-lane silent (no ledger).
    result = bench.run_bench(
        SUITE_PATH,
        agent={"type": "reference"},
        control_mode="pull",
        evidence_class="captured_fixture",
        emit_telemetry=False,
    )
    payload: dict[str, Any] = {
        "kind": OUTPUT_KIND,
        "suite_name": result["dataset_name"],
        "suite_version": result["dataset_version"],
        "modalities": result["modalities"],
        "aggregate": result["aggregate"],
        "per_task": result["per_task"],
    }
    if output_path is not None:
        out = Path(output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    # Never print inside run(): the docs fresh-lane exec-loads this module and the
    # release-check asserts empty stdout. Printing is __main__-only.
    return payload


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    destination = args[0] if args else None
    print(json.dumps(run(destination), indent=2, sort_keys=True, default=str))
