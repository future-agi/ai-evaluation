"""Command-graded coding benchmark — score files against a held-out grader.

Loads the shipped ``coding_command_starter`` bench suite and scores it in
``artifact_in`` mode through the unified bench harness. This is the *hardened*
coding tier: instead of importing ``check_*`` functions in-process, each task
hands the candidate a working directory, the candidate produces files/output,
and a **held-out grader** runs AFTERWARD — its verdict is the grader's exit code
plus a grader-owned ``reward.json``, never anything the candidate prints.

Two properties fall out of the temporal + path separation:

* **No verdict forgery** — the candidate's stdout is never parsed for a verdict;
  authority is the grader's exit code (0 = pass) and the grader-written reward
  file in ``$GRADER_DIR``, a directory the candidate phase never saw.
* **No oracle read** — the grader's held-out files (expected cases, tests) are
  materialised ONLY after the candidate command has finished and its processes
  are killed, so the candidate cannot peek at the expected values mid-run.

It is also multi-language for free: the candidate ``build`` and the ``grader_cmd``
are arbitrary shell, so the same lane grades the Python and bash tasks shipped
here side by side.

Run it::

    python examples/bench_command_graded.py artifacts/bench-command-graded.json

The per-task verdict is all-or-nothing: the grader's exit code is 0 only when
every held-out case passes. The candidate really executes in a scrubbed
subprocess sandbox with a hard timeout. For untrusted agent output, pass
``sandbox="docker"`` for OS-level isolation where the grader files are owned by a
different user the candidate uid cannot read (see ``docs/eval/benchmark-sandboxes.md``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import bench

SUITE_PATH = Path(__file__).parent / "bench_suites" / "coding_command_starter.json"
OUTPUT_KIND = "agent-learning.bench-command-graded.v1"


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    suite = bench.load_coding_suite(SUITE_PATH)
    # The gold reference files are the candidate here (deterministic, credential-
    # free). For command-graded tasks the reference is a {path: content} file map,
    # not a source string. Replace this with {task_id: {path: content}} to grade
    # your own agent's output instead.
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
