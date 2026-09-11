"""Coding benchmark quickstart — score candidate code against a held-out oracle.

Loads the shipped ``coding_starter`` bench suite and scores it in ``artifact_in``
mode through the unified bench harness. Each task ships a held-out ``checks``
oracle (executed against the candidate, never importable by it) plus a gold
``reference_solution``. We score the gold reference here so the example is
deterministic and credential-free; swap in ``{task_id: your_agent_source}`` to
grade your own agent's output instead.

Run it::

    python examples/bench_coding_quickstart.py artifacts/bench-coding.json

The per-task verdict is all-or-nothing: every held-out check must pass. The
candidate code really executes in a scrubbed subprocess sandbox with a hard
timeout. For untrusted agent output, pass ``sandbox="docker"`` for OS-level
isolation (see ``docs/eval/benchmark-sandboxes.md``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import bench

SUITE_PATH = Path(__file__).parent / "bench_suites" / "coding_starter.json"
OUTPUT_KIND = "agent-learning.bench-coding-quickstart.v1"


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    suite = bench.load_coding_suite(SUITE_PATH)
    # The gold reference is the candidate here (deterministic, credential-free).
    # Replace this mapping with {task_id: your_agent_source} to grade an agent.
    submission = bench.reference_submission(suite)
    # emit_telemetry=False keeps the docs/release fresh-lane silent (no ledger).
    result = bench.run_bench(
        SUITE_PATH,
        control_mode="artifact_in",
        submission=submission,
        sandbox="subprocess",
        evidence_class="local_gate",
        emit_telemetry=False,
    )
    payload: dict[str, Any] = {
        "kind": OUTPUT_KIND,
        "suite_name": result["dataset_name"],
        "suite_version": result["dataset_version"],
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
