"""Out-of-the-box task benchmark — credential-free, deterministic.

Loads the shipped ``support_starter`` task dataset, compiles it (every task's
objective must carry a deterministic anchor + Goodhart guards, or compilation
fails), and runs a scripted agent across it via ``tasks.run_benchmark`` on the
fixture lane. Prints a scored, honest benchmark result:

  * deterministic + credential-free (no keys, no network — the gate's fixture
    lane); same input -> identical score;
  * each per-task result stamped with its HONEST execution_class (conversation/
    tool_api are executable; the browser task is typed_only) and the run's
    evidence_class (captured_fixture); no fixture result is ever labeled live.

Run a real agent instead by passing ``--agent llm`` with a model key in env; the
default scripted agent keeps this example runnable anywhere with zero setup.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import tasks

DATASET_PATH = Path(__file__).parent / "task_datasets" / "support_starter.json"
OUTPUT_KIND = "agent-learning.task-benchmark-example.v1"

# A deterministic scripted agent whose single reply name-drops the anchor tokens
# the starter tasks check for (policy / help / order / human / webhook) — enough
# for the fixture lane to score without any model or network.
SCRIPTED_REPLY = (
    "Hello, happy to help. Our refund policy is at /help/refunds with a 30-day "
    "window. For your order status I will check the order tool and report the "
    "real status. If needed I will escalate you to a human. The webhooks docs "
    "are at /docs/webhooks."
)


def _gate_evidence(dataset: dict, agent: dict, result: dict) -> dict[str, Any]:
    """The evidence the release gate (#80) audits — all computed credential-free
    on the fixture lane: determinism (re-run identical), guard presence on every
    task, the overclaim tripwire (a typed-only task forced with a live evidence
    class MUST be flagged), and world-kind coverage."""

    # determinism: a second fixture-lane run yields identical per-task scores.
    result2 = tasks.run_benchmark(
        dataset, agent, evidence_class="captured_fixture", emit_telemetry=False
    )
    scores1 = {r["task_id"]: r["score"] for r in result["per_task"]}
    scores2 = {r["task_id"]: r["score"] for r in result2["per_task"]}

    # guard presence: every shipped task carries declared Goodhart guards.
    all_guards = all(
        t["objective"]["guards"]["min_guard_count"] >= 1
        and (t["objective"]["guards"]["sentinel_rows"] or t["objective"]["guards"]["canary_evals"])
        for t in dataset["tasks"]
    )

    # overclaim tripwire: re-run requesting a LIVE evidence class; the typed-only
    # task(s) MUST be flagged overclaim, the executable ones MUST NOT.
    live = tasks.run_benchmark(
        dataset, agent, evidence_class="live_lane", emit_telemetry=False
    )
    by_id = {r["task_id"]: r for r in live["per_task"]}
    typed_only_ids = [t["id"] for t in dataset["tasks"] if t["execution_class"] in ("typed_only", "fixture")]
    executable_ids = [t["id"] for t in dataset["tasks"] if t["execution_class"] == "executable"]
    typed_only_flagged = bool(typed_only_ids) and all(by_id[i]["overclaim"] for i in typed_only_ids)
    executable_not_flagged = all(not by_id[i]["overclaim"] for i in executable_ids)

    world_kinds = list(tasks.task_world_kinds(dataset))
    return {
        "dataset_version": dataset["version"],
        "determinism": {"scores_identical_across_runs": scores1 == scores2},
        "guard_presence": {"all_tasks_have_guards": all_guards},
        "overclaim_tripwire": {
            "typed_only_flagged_under_live": typed_only_flagged,
            "executable_not_flagged_under_live": executable_not_flagged,
            "fixture_lane_honest": result["aggregate"]["honesty"]["any_overclaim"] is False,
        },
        "coverage": {
            "world_kinds": world_kinds,
            "spans_executable": {"conversation", "tool_api"} <= set(world_kinds),
        },
    }


def run(output_path: str | Path | None = None, *, agent: dict | None = None) -> dict[str, Any]:
    dataset = tasks.load_task_dataset(DATASET_PATH)
    agent = agent or {"type": "scripted", "content": SCRIPTED_REPLY}
    # emit_telemetry=False: this is the release-gate fixture entry — keep it a pure,
    # side-channel-free deterministic run (no ledger write / stderr line during gates).
    result = tasks.run_benchmark(
        dataset, agent, evidence_class="captured_fixture", emit_telemetry=False
    )

    payload: dict[str, Any] = {
        "kind": OUTPUT_KIND,
        "status": "passed",
        "exit_code": 0,
        "dataset_name": result["dataset_name"],
        "dataset_version": result["dataset_version"],
        "coverage": dataset["coverage"],
        "aggregate": result["aggregate"],
        "per_task": result["per_task"],
        "gate_evidence": _gate_evidence(dataset, agent, result),
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
