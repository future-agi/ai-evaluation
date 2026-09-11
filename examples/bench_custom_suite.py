"""Author your own coding bench suite — build, validate, and score it in-memory.

Most pages load a *shipped* suite. This one BUILDS a brand-new
``agent-learning.bench-suite.v1`` suite from scratch (a one-task ``is_even``
suite), validates it with ``bench.load_coding_suite(obj)``, and scores its own
gold reference through the unified harness in ``artifact_in`` mode. Use it as the
template for writing the suite that grades *your* agent's coding output.

The bench-suite shape, per the validator in ``bench._coding``:

* top level: ``kind`` (exactly ``agent-learning.bench-suite.v1``), ``name``,
  ``version``, ``language`` (``python`` today), ``modality`` (``coding``), and a
  non-empty ``tasks`` list of unique-``id`` tasks.
* each ``checks``-graded task carries: ``id``, ``instruction``, ``checks`` (the
  held-out oracle — ``check_*`` functions that ``import solution`` and assert),
  ``reference_solution`` (the gold the release gate scores to prove the verifier
  accepts a correct answer), and a ``guards`` block with
  ``min_guard_count >= 1`` — the mandatory anti-gaming contract.

The candidate code never sees the ``checks`` file, so it cannot reflect the
expected answers; it has to actually solve the task. The verdict is
all-or-nothing: a task is resolved only when every held-out check passes.

Run it::

    python examples/bench_custom_suite.py artifacts/bench-custom-suite.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import bench

OUTPUT_KIND = "agent-learning.bench-custom-suite.v1"


def build_suite() -> dict[str, Any]:
    """Hand-author a tiny one-task coding bench suite (the schema, in full)."""

    return {
        "kind": "agent-learning.bench-suite.v1",
        "name": "is_even_suite",
        "version": "1",
        "language": "python",
        "modality": "coding",
        "description": "A hand-authored one-task coding suite for the write-a-suite page.",
        "tasks": [
            {
                "id": "is_even",
                "instruction": (
                    "Implement is_even(n): return True if the integer n is even, "
                    "otherwise False."
                ),
                # Held-out oracle: lives in its own module the candidate never
                # imports. Each check_* imports the candidate as `solution` and
                # asserts behaviour the instruction implies but does not spell out
                # (zero, negatives) so a partial guess cannot resolve the task.
                "checks": (
                    "import solution\n\n"
                    "def check_even():\n"
                    "    assert solution.is_even(4) is True\n\n"
                    "def check_odd():\n"
                    "    assert solution.is_even(7) is False\n\n"
                    "def check_zero():\n"
                    "    assert solution.is_even(0) is True\n\n"
                    "def check_negative():\n"
                    "    assert solution.is_even(-3) is False\n"
                ),
                # Gold reference: the release gate scores this to prove the oracle
                # ACCEPTS a correct answer (not just rejects wrong ones).
                "reference_solution": (
                    "def is_even(n):\n    return n % 2 == 0\n"
                ),
                # Mandatory anti-gaming contract. min_guard_count >= 1 or
                # load_coding_suite raises CodingSuiteError. The held-out oracle is
                # the real defence; the guards block makes the suite DECLARE it.
                "guards": {
                    "min_guard_count": 1,
                    "oracle_held_out": True,
                    "sentinel": (
                        "a no-op or fake-success candidate must fail the held-out checks"
                    ),
                },
                "difficulty": "easy",
                "timeout_s": 10,
            }
        ],
    }


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    suite = build_suite()
    # 1) Validate the suite BEFORE scoring — enforces the schema + the
    # guards.min_guard_count >= 1 contract. Raises CodingSuiteError if malformed.
    validated = bench.load_coding_suite(suite)
    # 2) The gold reference is the candidate here (deterministic, credential-free).
    # Replace this mapping with {task_id: your_agent_source} to grade an agent.
    submission = bench.reference_submission(validated)
    # 3) Score the in-memory suite through the unified harness. A Mapping suite is
    # compiled in place — no file needed. emit_telemetry=False keeps the docs /
    # release fresh-lane silent (no ledger row written).
    result = bench.run_bench(
        validated,
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
