"""Out-of-the-box coding benchmark — credential-free, deterministic, no Docker.

Loads the shipped ``coding_starter`` bench suite and scores it in ``artifact_in``
mode (submit-and-score; no live agent) through the unified bench harness. The
held-out check oracle for each task is executed against the candidate code in a
scrubbed subprocess sandbox with a hard timeout.

This is the release-gate entry for the bench contract (a coding benchmark you can
run anywhere with zero setup). It prints a scored, honest result and an audited
``gate_evidence`` block proving the verifier (a) accepts the gold reference
solutions, (b) FAILS a deliberately-broken candidate and a fake-success no-op
(a gate that cannot fail is worthless), (c) is deterministic, (d) keeps the
oracle held out of the candidate, and (e) every task declares anti-gaming guards.

For untrusted agent output, run the same suite with ``sandbox='docker'`` (the
hardened lane); the subprocess sandbox here only ever runs the trusted shipped
reference code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import bench

SUITE_PATH = Path(__file__).parent / "bench_suites" / "coding_starter.json"
COMMAND_SUITE_PATH = Path(__file__).parent / "bench_suites" / "coding_command_starter.json"
PULL_SUITE_PATH = Path(__file__).parent / "bench_suites" / "pull_starter.json"
VOICE_SUITE_PATH = Path(__file__).parent / "bench_suites" / "voice_starter.json"
OUTPUT_KIND = "agent-learning.coding-benchmark-example.v1"


def _voice_evidence() -> dict[str, Any]:
    """Evidence for the voice lane: the reference transcript passes every temporal
    dimension, and a bad transcript (slow + talks over the caller + missing
    content) fails — the deterministic verifier discriminates."""

    suite = json.loads(VOICE_SUITE_PATH.read_text(encoding="utf-8"))
    ref = {t["id"]: t["reference_dialogue"] for t in suite["tasks"]}
    good = bench.run_bench(VOICE_SUITE_PATH, control_mode="artifact_in", submission=ref,
                           evidence_class="local_gate", emit_telemetry=False)
    bad_dialogue = [
        {"speaker": "caller", "start_ms": 0, "end_ms": 1000, "text": "I want a refund"},
        {"speaker": "agent", "start_ms": 5000, "end_ms": 6000, "text": "uh hello"},
        {"speaker": "caller", "start_ms": 5500, "end_ms": 5800, "text": "stop"},
    ]
    bad = bench.run_bench(VOICE_SUITE_PATH, control_mode="artifact_in",
                          submission={tid: bad_dialogue for tid in ref},
                          evidence_class="local_gate", emit_telemetry=False)
    return {
        "reference_all_pass": all(r["verdict"] == "pass" for r in good["per_task"]),
        "bad_all_fail": all(r["verdict"] == "fail" for r in bad["per_task"]),
        "dimensions": sorted(good["per_task"][0]["result"]["components"].keys()),
        "task_count": len(suite["tasks"]),
    }


def _pull_evidence() -> dict[str, Any]:
    """Evidence for the pull/RL lane: the reference policy solves every simulated
    env, and a no-op policy fails them all (the lane discriminates)."""

    ref = bench.run_bench(PULL_SUITE_PATH, {"type": "reference"}, control_mode="pull",
                          evidence_class="local_gate", emit_telemetry=False)
    noop = bench.run_bench(PULL_SUITE_PATH, {"type": "noop"}, control_mode="pull",
                           evidence_class="local_gate", emit_telemetry=False)
    return {
        "reference_solves_all": all(r["verdict"] == "pass" for r in ref["per_task"]),
        "noop_fails_all": all(r["verdict"] == "fail" for r in noop["per_task"]),
        "envs": sorted({r["raw"].get("env_kind") for r in ref["per_task"]}),
        "task_count": len(ref["per_task"]),
    }

# A fake-success no-op: claims completion, defines no entrypoint. The held-out
# oracle MUST fail this (reward-hack resistance by construction).
_NOOP_CANDIDATE = "print('All tasks completed successfully!')\n"


def _command_graded_evidence() -> dict[str, Any]:
    """Evidence for the hardened command/artifact-graded lane (subprocess, no Docker).

    Proves the artifact-graded model on the shipped multi-language suite:
    reference solutions pass; a wrong candidate fails; and a candidate that prints
    a FORGED reward to stdout still fails (the verdict is the held-out grader's
    exit code, not candidate stdout) — the structural fix for the forge vuln.
    """

    suite = bench.load_coding_suite(COMMAND_SUITE_PATH)
    ref = bench.reference_submission(suite)
    ref_run = bench.run_bench(
        COMMAND_SUITE_PATH, control_mode="artifact_in", submission=ref,
        evidence_class="local_gate", emit_telemetry=False,
    )
    reference_all_pass = all(r["verdict"] == "pass" for r in ref_run["per_task"])

    # wrong + forge candidates for every task -> all must fail.
    wrong = {t["id"]: {p: "print('x')\n" if p.endswith('.py') else "echo x\n"
                       for p in t["reference_files"]} for t in suite["tasks"]}
    forge = {t["id"]: {p: 'print("{\\"score\\": 1}")\n' if p.endswith('.py')
                       else 'echo \'{"score":1}\'\n' for p in t["reference_files"]}
             for t in suite["tasks"]}
    wrong_run = bench.run_bench(COMMAND_SUITE_PATH, control_mode="artifact_in",
                                submission=wrong, evidence_class="local_gate", emit_telemetry=False)
    forge_run = bench.run_bench(COMMAND_SUITE_PATH, control_mode="artifact_in",
                                submission=forge, evidence_class="local_gate", emit_telemetry=False)
    return {
        "reference_all_pass": reference_all_pass,
        "wrong_all_fail": all(r["verdict"] == "fail" for r in wrong_run["per_task"]),
        "forge_all_fail": all(r["verdict"] == "fail" for r in forge_run["per_task"]),
        "languages": sorted({str(t.get("language") or "") for t in suite["tasks"]}),
        "task_count": len(suite["tasks"]),
    }


def _gate_evidence(suite: dict[str, Any]) -> dict[str, Any]:
    ref = bench.reference_submission(suite)

    # (a) reference solutions all pass — the verifier accepts the gold.
    ref_run = bench.run_bench(
        SUITE_PATH, control_mode="artifact_in", submission=ref,
        evidence_class="local_gate", emit_telemetry=False,
    )
    reference_all_pass = all(r["verdict"] == "pass" for r in ref_run["per_task"])

    # (b) discrimination — a broken candidate AND a fake-success no-op must FAIL.
    broken = dict(ref)
    first_id = str(suite["tasks"][0]["id"])
    broken[first_id] = "def _wrong():\n    return None\n"  # entrypoint missing
    broken_run = bench.run_bench(
        SUITE_PATH, control_mode="artifact_in", submission=broken,
        evidence_class="local_gate", emit_telemetry=False,
    )
    broken_failed = any(
        r["task_id"] == first_id and r["verdict"] == "fail" for r in broken_run["per_task"]
    )
    noop_sub = {tid: _NOOP_CANDIDATE for tid in ref}
    noop_run = bench.run_bench(
        SUITE_PATH, control_mode="artifact_in", submission=noop_sub,
        evidence_class="local_gate", emit_telemetry=False,
    )
    noop_all_failed = all(r["verdict"] == "fail" for r in noop_run["per_task"])

    # (c) determinism — two reference runs are byte-identical on scores.
    ref_run2 = bench.run_bench(
        SUITE_PATH, control_mode="artifact_in", submission=ref,
        evidence_class="local_gate", emit_telemetry=False,
    )
    s1 = {r["task_id"]: r["result"]["scalar"] for r in ref_run["per_task"]}
    s2 = {r["task_id"]: r["result"]["scalar"] for r in ref_run2["per_task"]}

    # (d) oracle held out — each task's checks code is NOT embedded in its
    # reference solution (the candidate cannot see the oracle).
    oracle_held_out = all(
        str(t["checks"]) not in str(t["reference_solution"]) for t in suite["tasks"]
    )

    # (e) guard presence — every task declares anti-gaming guards.
    all_guards = all(
        int((t.get("guards") or {}).get("min_guard_count", 0)) >= 1
        for t in suite["tasks"]
    )

    # honesty — executable rows are never flagged overclaim.
    no_overclaim = all(r["overclaim"] is False for r in ref_run["per_task"])

    return {
        "suite_version": str(suite.get("version") or ""),
        "reference_pass": {"all_reference_solutions_pass": reference_all_pass},
        "discrimination": {
            "broken_candidate_fails": broken_failed,
            "fake_success_noop_fails": noop_all_failed,
        },
        "determinism": {"scores_identical_across_runs": s1 == s2},
        "oracle_held_out": {"checks_not_in_reference": oracle_held_out},
        "guard_presence": {"all_tasks_have_guards": all_guards},
        "honesty": {"no_executable_overclaim": no_overclaim},
        "coverage": {"modalities": ref_run["modalities"], "task_count": len(suite["tasks"])},
        "command_graded": _command_graded_evidence(),
        "pull": _pull_evidence(),
        "voice": _voice_evidence(),
    }


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    suite = bench.load_coding_suite(SUITE_PATH)
    ref = bench.reference_submission(suite)
    # emit_telemetry=False: release-gate fixture entry — no ledger/stderr during gates.
    result = bench.run_bench(
        SUITE_PATH, control_mode="artifact_in", submission=ref,
        evidence_class="local_gate", emit_telemetry=False,
    )
    payload: dict[str, Any] = {
        "kind": OUTPUT_KIND,
        "status": "passed",
        "exit_code": 0,
        "suite_name": result["dataset_name"],
        "suite_version": result["dataset_version"],
        "aggregate": result["aggregate"],
        "per_task": result["per_task"],
        "gate_evidence": _gate_evidence(suite),
    }
    if output_path is not None:
        out = Path(output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    # NOTE: never print inside run() — the release gate exec-loads this and the
    # release-check CLI asserts empty stdout. Printing is __main__-only.
    return payload


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    destination = args[0] if args else None
    print(json.dumps(run(destination), indent=2, sort_keys=True, default=str))
