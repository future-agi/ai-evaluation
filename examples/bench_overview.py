"""Unified bench harness overview — one Task<->Verifier contract, three control
modes, one Result shape across modalities.

The benchmark harness is a single contract with one constant (the fixed
Task<->Verifier coupling — a suite carries its own oracle) and two dimensions:
the *modality* (coding / rl / voice / text / tool / ...) and the *control mode*:

  * ``push``        — the harness drives the agent over a task dataset
                      (``tasks.run_benchmark``); shown as a code snippet below
                      because it needs a live agent, not exercised here.
  * ``artifact_in`` — submit a candidate and score it against a held-out oracle;
                      used here for the coding lane and the voice lane.
  * ``pull``        — the agent drives a live environment via reset/step; used
                      here for the rl lane with the env's reference policy.

This example runs the three credential-free lanes end to end and assembles ONE
combined artifact that proves the headline: many modes, many modalities, but
every per-task verdict projects into the same unified
``Result`` ``{scalar, components, pass_fail, explanation}``.

Run it::

    python examples/bench_overview.py artifacts/bench-overview.json

Each lane is scored against its own gold reference (the coding suite's
``reference_submission``, the rl env's ``{"type": "reference"}`` optimal policy,
and the voice suite's ``reference_dialogue``) so the run is deterministic and
credential-free. Swap any reference for your own agent's output / policy to grade
the real thing. No network, no API keys, no Docker container is launched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import bench

SUITE_DIR = Path(__file__).parent / "bench_suites"
CODING_SUITE = SUITE_DIR / "coding_starter.json"
PULL_SUITE = SUITE_DIR / "pull_starter.json"
VOICE_SUITE = SUITE_DIR / "voice_starter.json"
OUTPUT_KIND = "agent-learning.bench-overview.v1"

# The unified Result keys every modality's verdict projects into. We assert the
# combined artifact carries exactly this shape on a row from each lane, so the
# "one Result across modalities" claim is checkable, not asserted.
RESULT_KEYS = ("scalar", "components", "pass_fail", "explanation")


def _lane(result: dict[str, Any]) -> dict[str, Any]:
    """Distil one bench result into the per-lane record we store in the artifact."""

    rows = result["per_task"]
    return {
        "control_mode": result["control_mode"],
        "modalities": result["modalities"],
        "aggregate": result["aggregate"],
        "verdicts": [{"task_id": r["task_id"], "verdict": r["verdict"]} for r in rows],
        # one sample row's unified-Result keys — proof the shape is shared.
        "result_keys": sorted(rows[0]["result"].keys()),
    }


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    # --- artifact_in / coding: candidate code vs a held-out check oracle ---
    coding_suite = bench.load_coding_suite(CODING_SUITE)
    coding = bench.run_bench(
        CODING_SUITE,
        control_mode="artifact_in",
        submission=bench.reference_submission(coding_suite),
        sandbox="subprocess",
        evidence_class="local_gate",
        emit_telemetry=False,
    )

    # --- pull / rl: the agent drives a live env; here the env's optimal policy ---
    pull = bench.run_bench(
        PULL_SUITE,
        {"type": "reference"},
        control_mode="pull",
        evidence_class="local_gate",
        emit_telemetry=False,
    )

    # --- artifact_in / voice: score a voice episode transcript on its budgets ---
    voice_suite = json.loads(VOICE_SUITE.read_text("utf-8"))
    voice_task = voice_suite["tasks"][0]
    voice = bench.run_bench(
        VOICE_SUITE,
        control_mode="artifact_in",
        submission={voice_task["id"]: voice_task["reference_dialogue"]},
        evidence_class="local_gate",
        emit_telemetry=False,
    )

    lanes = {"coding": _lane(coding), "rl": _lane(pull), "voice": _lane(voice)}
    payload: dict[str, Any] = {
        "kind": OUTPUT_KIND,
        # The headline assertions, pre-computed so the postcondition is a one-liner.
        "control_modes": sorted({lane["control_mode"] for lane in lanes.values()}),
        "modalities": sorted(
            m for lane in lanes.values() for m in lane["modalities"]
        ),
        "unified_result_keys": list(RESULT_KEYS),
        # Every lane's Result row exposes the same key set -> one Result shape.
        "result_shape_consistent": all(
            lane["result_keys"] == sorted(RESULT_KEYS) for lane in lanes.values()
        ),
        "lanes": lanes,
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
