"""Voice benchmark quickstart — score a voice episode transcript on a temporal contract.

Loads the shipped ``voice_starter`` bench suite and scores it in ``artifact_in``
mode through the unified bench harness. Voice is the modality where the verifier
is *temporal*, not an exit code: each task's transcript is graded on four
sub-scores — **latency** (the agent answers within the per-task budget after the
caller stops), **turn-taking** (no harmful overlap outside a legitimate
barge-in), **barge_in** (the agent yields promptly when interrupted), and
**content** (the agent's words cover the task's ``required_content``). A task is
a pass only when *every* sub-score meets the floor.

Run it::

    python examples/bench_voice.py artifacts/bench-voice.json

A voice suite runs ``control_mode="artifact_in"`` with
``submission={task_id: dialogue}``, where a dialogue is the interleaved
caller/agent turn list (each turn carries ``speaker``, ``start_ms``, ``end_ms``,
``text``, and an optional ``interrupt`` flag on caller turns). Here we submit each
task's own shipped ``reference_dialogue`` so the run is deterministic and
credential-free; swap in ``{task_id: your_captured_transcript}`` to grade a real
episode. The same verifier consumes a transcript captured from a live
audio/SIP/WebRTC call plus ASR — it plugs in unchanged by producing the same
turn shape.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import bench

SUITE_PATH = Path(__file__).parent / "bench_suites" / "voice_starter.json"
OUTPUT_KIND = "agent-learning.bench-voice-quickstart.v1"


def _reference_submission(suite: dict[str, Any]) -> dict[str, Any]:
    """Map each task to its shipped ``reference_dialogue`` (the gold transcript)."""

    return {
        str(task["id"]): task["reference_dialogue"]
        for task in suite.get("tasks", [])
        if task.get("reference_dialogue") is not None
    }


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    suite = json.loads(SUITE_PATH.read_text("utf-8"))
    # The gold reference_dialogue is the submitted transcript here (deterministic,
    # credential-free). Replace this mapping with {task_id: your_transcript} —
    # a list of {speaker, start_ms, end_ms, text[, interrupt]} turns — to grade a
    # real captured episode against the same temporal contract.
    submission = _reference_submission(suite)
    # emit_telemetry=False keeps the docs/release fresh-lane silent (no ledger).
    result = bench.run_bench(
        SUITE_PATH,
        control_mode="artifact_in",
        submission=submission,
        evidence_class="local_gate",
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
