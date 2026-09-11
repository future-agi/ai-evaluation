"""Bench CI gate — turn a bench result into a single pass/block decision.

Scores the shipped ``coding_starter`` suite in ``artifact_in`` mode (against its
own gold references, so the run is deterministic and credential-free) and then
applies a CI gate over the honest aggregate:

  * ``aggregate.pass_rate`` (computed over SCORED tasks only — ``void`` rows from a
    missing submission or an infra failure are excluded, never read as "0% passed")
    must be ``>=`` a threshold, and
  * ``aggregate.honesty.any_overclaim`` must be ``False`` — the overclaim tripwire
    fires when a non-live ``execution_class`` carries a live ``evidence_class``, so
    a gamed honesty stamp blocks the gate even at a perfect pass_rate.

The harness keeps each modality honest the same way: ``execution_class`` is
DERIVED from the substrate (``executable`` / ``typed_only`` / ``fixture``, never
asserted above it), and ``evidence_class`` records HOW the run was witnessed
(``local_gate`` / ``live_lane`` / ``live_stressed`` / ``captured_fixture``). In
the coding ``artifact_in`` lane the anti-gaming defence is the held-out oracle:
the verdict is all-or-nothing (``pass`` only when every held-out check passes), so
a candidate cannot game a scorer it cannot read. Infra failures are ``void``, not
``fail``.

Run it::

    python examples/bench_ci_gate.py artifacts/in-ci.json

The artifact records the gate decision (``gate.passed``) plus the inputs that
produced it, so CI can ``assert`` on the file and a human can read why.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import bench

SUITE_PATH = Path(__file__).parent / "bench_suites" / "coding_starter.json"
OUTPUT_KIND = "agent-learning.bench-ci-gate.v1"

#: The CI gate thresholds. A reader tunes these to their bar; the suite's gold
#: references resolve every task, so this gate passes here at 1.0.
MIN_PASS_RATE = 1.0


def evaluate_gate(aggregate: dict[str, Any], *, min_pass_rate: float) -> dict[str, Any]:
    """Reduce a bench aggregate to one CI decision plus its reasons.

    The gate blocks unless the pass_rate (over scored tasks) clears the bar AND
    the honesty rollup reports no overclaim. Both reasons are recorded so the
    artifact explains a block, never just asserts one.
    """

    honesty = aggregate.get("honesty") or {}
    pass_rate = float(aggregate.get("pass_rate", 0.0))
    any_overclaim = bool(honesty.get("any_overclaim", False))
    pass_rate_ok = pass_rate >= min_pass_rate
    reasons: list[str] = []
    if not pass_rate_ok:
        reasons.append(
            f"pass_rate {pass_rate} < threshold {min_pass_rate} "
            f"(over {aggregate.get('scored')} scored of {aggregate.get('count')} tasks)"
        )
    if any_overclaim:
        reasons.append("honesty.any_overclaim is True (a live evidence stamp on a non-live row)")
    return {
        "passed": pass_rate_ok and not any_overclaim,
        "min_pass_rate": min_pass_rate,
        "pass_rate": pass_rate,
        "pass_rate_ok": pass_rate_ok,
        "any_overclaim": any_overclaim,
        "scored": aggregate.get("scored"),
        "void": aggregate.get("void"),
        "count": aggregate.get("count"),
        "reasons": reasons,
    }


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    suite = bench.load_coding_suite(SUITE_PATH)
    # The gold reference is the candidate here (deterministic, credential-free).
    # Replace this mapping with {task_id: your_agent_source} to gate an agent.
    submission = bench.reference_submission(suite)
    # evidence_class=local_gate is the honest stamp for an offline self-check; the
    # subprocess sandbox really executes the candidate against its held-out oracle,
    # and the verdict is all-or-nothing, so a no-op candidate fails the gate.
    result = bench.run_bench(
        SUITE_PATH,
        control_mode="artifact_in",
        submission=submission,
        sandbox="subprocess",
        evidence_class="local_gate",
        emit_telemetry=False,  # keep the docs/release fresh-lane silent (no ledger).
    )
    aggregate = result["aggregate"]
    gate = evaluate_gate(aggregate, min_pass_rate=MIN_PASS_RATE)
    payload: dict[str, Any] = {
        "kind": OUTPUT_KIND,
        "suite_name": result["dataset_name"],
        "suite_version": result["dataset_version"],
        "gate": gate,
        "aggregate": aggregate,
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
    payload = run(destination)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    # A real CI gate exits non-zero to block the pipeline when the gate fails.
    sys.exit(0 if payload["gate"]["passed"] else 1)
