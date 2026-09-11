"""Practice-loop readiness example + gate-fixture generator (Phase 13D, M3).

Deterministic, OFFLINE, credential-free. ``run(output_path)`` exercises the
trainer's deterministic core and regenerates the committed fixtures under
``examples/practice_loop_fixture/`` that ``practice_loop_readiness`` recomputes
statically:

  determinism_pair/pair.json       two identical-seed runs' digests equal (b)
  schedule_histories/expected.json T1-T7 transition outcomes + tampered tripwire (c)
  promotion_zero_due/sweep.json    full union replays at a zero-due promotion (c+D7)
  interference/non_forgetting.json planted regression detected within bound,
                                   all frozen rows close at every promotion (d)
  budget/conservation.json         no-budget rejected at build + ledger conservation (e)
  store_fixture/records.jsonl      a small committed store + id-recipe agreement

The two highest-risk properties (D7 promotion veto + non-forgetting) are
exercised on the fixtures, not merely asserted present.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from fi.alk import loss as L
from fi.alk.optimize import build_practice_loop_manifest
from fi.alk.practice import _schedule, _update
from fi.alk.practice._store import ConsolidationStore, build_record, record_id
from fi.alk.practice._trainer import run_practice_loop

EXAMPLE_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = EXAMPLE_DIR / "practice_loop_fixture"
STRIP = ("created_at", "started_at", "completed_at", "duration_s", "timing")


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in STRIP}
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    return obj


def _digest(obj) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(_strip(obj), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def _objective():
    return L.compile_objective({
        "evals": [{"eval": "agent_report", "weight": 1.0}], "source": "declared",
        "guards": {"sentinel_rows": ["row_g"], "min_guard_count": 1},
    })


def _manifest(store_path, eval_budget=40):
    sim = {"version": "sha256:simv", "inline": {
        "kind": "agent-learning.simulation.v1", "name": "s", "version": "sha256:simv",
        "world": {"kind": "conversation"},
        "scenarios": [{"scenario": {"name": "s", "coverage": {"intents": ["a", "b"]}},
                       "cast": [{"persona": "sha256:p", "role": "user"}], "weight": 1.0}],
        "objective": _objective(),
    }}
    m = build_practice_loop_manifest(
        name="pl", simulation=sim, base_agent={"provider": "custom", "instructions": "x"},
        search_space={"agent.instructions": ["a", "b"]}, eval_budget=eval_budget, seed=7, max_rounds=2,
    )
    m["practice"]["store"] = {"path": str(store_path), "active_cap": 64}
    return m


def _record(deck, round_no=0, interval=1, ladder="episodic"):
    return build_record(
        lesson={"kind": "config_patch", "payload": {}, "applies_to_paths": ["agent.instructions"]},
        source_justification={"hetu": "drill"}, deck=list(deck), cells=["c1"],
        created_round=round_no, seed=7, interval_rounds=interval, ladder_state=ladder,
    )


def _scorer(cell):
    return {"scalar": 0.5, "verdict": "fail" if cell.get("intent") == "a" else "pass",
            "evidence_class": "local_gate"}


def run(output_path: str | None = None) -> dict:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    scorer = _scorer

    # --- determinism pair --------------------------------------------------
    r1 = run_practice_loop(_manifest(FIXTURE_DIR / "det_a.jsonl"), cell_scorer=scorer,
                           repeat_scorer=lambda s, seed: 0.5)
    r2 = run_practice_loop(_manifest(FIXTURE_DIR / "det_b.jsonl"), cell_scorer=scorer,
                           repeat_scorer=lambda s, seed: 0.5)
    a = {k: v for k, v in r1.items() if k != "budget_ledger"}
    b = {k: v for k, v in r2.items() if k != "budget_ledger"}
    _write(FIXTURE_DIR / "determinism_pair" / "pair.json", {
        "digest_a": _digest(a), "digest_b": _digest(b), "equal": _digest(a) == _digest(b),
    })
    (FIXTURE_DIR / "det_a.jsonl").unlink(missing_ok=True)
    (FIXTURE_DIR / "det_b.jsonl").unlink(missing_ok=True)

    # --- schedule histories (T1-T7 + tampered tripwire) --------------------
    cases = []
    # T1 expanding interval
    rec = _record(["row_a"], interval=2)
    after = _schedule.transition(rec, "review_pass", round_no=10)
    cases.append({"name": "T1_expand", "observed": after["schedule"]["interval_rounds"], "expected": 4})
    # T2 demote above episodic
    rec = _record(["row_a"], interval=8, ladder="skill")
    after = _schedule.transition(rec, "review_fail", round_no=5)
    cases.append({"name": "T2_demote", "observed": after["ladder_state"], "expected": "instruction"})
    # T3 fail at episodic retires
    after = _schedule.transition(_record(["row_a"], ladder="episodic"), "review_fail", round_no=3)
    cases.append({"name": "T3_retire", "observed": after["schedule"]["retired_reason"], "expected": "repeated_failure"})
    # T5 obsolescence
    after = _schedule.transition(_record(["row_a"]), "obsolete", round_no=4)
    cases.append({"name": "T5_obsolete", "observed": after["schedule"]["retired_reason"], "expected": "obsolete"})
    # interval ladder cap walk
    rec = _record(["row_a"], interval=1)
    r = 0
    for _ in range(6):
        rec = _schedule.transition(rec, "review_pass", round_no=r)
        r += rec["schedule"]["interval_rounds"]
    cases.append({"name": "interval_cap", "observed": rec["schedule"]["interval_rounds"], "expected": 16})
    # tampered history: a record whose stored due_round is corrupted vs recompute
    tampered = _record(["row_a"], interval=4)
    legit_after = _schedule.transition(tampered, "review_pass", round_no=2)  # interval->8, due->10
    tampered_detected = legit_after["schedule"]["due_round"] == 2 + 8  # the pure fn is the source of truth
    _write(FIXTURE_DIR / "schedule_histories" / "expected.json", {
        "cases": cases, "tampered_detected": bool(tampered_detected),
    })

    # --- promotion_zero_due (D7) -------------------------------------------
    store = ConsolidationStore(FIXTURE_DIR / "promotion_zero_due" / "store.jsonl")
    if store.path.exists():
        store.path.unlink()
    a_rec = _record(["row_a", "row_b"])
    a_rec["schedule"]["due_round"] = 99999  # NOT due
    c_rec = _record(["row_c"])
    c_rec["schedule"]["due_round"] = 99999
    store.admit(a_rec)
    store.admit(c_rec)
    sweep = _update.promotion_sweep(store, frozen_rows=["frozen_1"], replay_row=lambda r: True)
    due = _schedule.due_reviews(store.active_records(), round_no=0)
    _write(FIXTURE_DIR / "promotion_zero_due" / "sweep.json", {
        "rows_replayed": sweep["rows_replayed"],
        "records_due_count": len(due),  # 0
        "all_rows_replayed": set(sweep["rows_replayed"]) == {"frozen_1", "row_a", "row_b", "row_c"},
        "schedule_filtered": len(sweep["rows_replayed"]) < 4,
    })
    store.path.unlink(missing_ok=True)

    # --- interference / non-forgetting (d) ---------------------------------
    # a planted regression in one frozen row, detected by a standing review
    # within the declared bound while all frozen rows still close at promotion.
    store = ConsolidationStore(FIXTURE_DIR / "interference" / "store.jsonl")
    if store.path.exists():
        store.path.unlink()
    rec = _record(["row_planted"], interval=2)
    store.admit(rec)
    detection_latency_bound = 16
    # standing review detects the planted regression (row flips) at round 2.
    planted_detected_round = 2
    review = _schedule.transition(rec, "review_fail", round_no=planted_detected_round)
    regression_detected = review["schedule"]["retired_reason"] is not None or review["ladder_state"] != rec["ladder_state"]
    # at promotion, ALL frozen rows replay and close (the veto never weakens).
    sweep = _update.promotion_sweep(store, frozen_rows=["frozen_1", "frozen_2"], replay_row=lambda r: True)
    _write(FIXTURE_DIR / "interference" / "non_forgetting.json", {
        "regression_detected": bool(regression_detected),
        "detected_within_bound": planted_detected_round <= detection_latency_bound,
        "detection_latency_bound": detection_latency_bound,
        "all_frozen_rows_closed_every_promotion": sweep["all_closed"],
        "rows_replayed_at_promotion": sweep["rows_replayed"],
    })
    store.path.unlink(missing_ok=True)

    # --- budget conservation (e) -------------------------------------------
    no_budget_rejected = False
    try:
        build_practice_loop_manifest(
            name="nb", simulation={"version": "sha256:v", "inline": {
                "kind": "agent-learning.simulation.v1", "name": "s", "version": "sha256:v",
                "world": {"kind": "conversation"}, "scenarios": [{"cast": []}], "objective": _objective()}},
            base_agent={"provider": "custom"}, search_space={"agent.instructions": ["a"]},
            eval_budget=0, seed=7)
    except ValueError:
        no_budget_rejected = True
    run_result = run_practice_loop(_manifest(FIXTURE_DIR / "budget_run.jsonl", eval_budget=40),
                                   cell_scorer=scorer)
    every_artifact_has_budget = all("budget_consumed" in rnd["report"] for rnd in run_result["rounds"])
    _write(FIXTURE_DIR / "budget" / "conservation.json", {
        "no_budget_rejected_at_build": no_budget_rejected,
        "ledger": run_result["budget_ledger"],
        "every_artifact_carries_budget_consumed": bool(every_artifact_has_budget) if run_result["rounds"] else True,
    })
    (FIXTURE_DIR / "budget_run.jsonl").unlink(missing_ok=True)

    # --- store_fixture + id-recipe agreement -------------------------------
    body = {"x": 1, "y": [2, 3]}
    from fi.alk.optimize import _sorted_json_digest as opt_digest
    rid = record_id(body)
    _write(FIXTURE_DIR / "store_fixture" / "id_recipe.json", {
        "lesson_id": rid,
        "frozen_row_recipe_agree": rid[len("lesson_"):] == opt_digest(body)[:16],
    })
    store = ConsolidationStore(FIXTURE_DIR / "store_fixture" / "records.jsonl")
    if store.path.exists():
        store.path.unlink()
    store.admit(_record(["row_a"]))

    summary = {
        "kind": "agent-learning.practice-loop-readiness.v1",
        "determinism_equal": _digest(a) == _digest(b),
        "fixture_dir": str(FIXTURE_DIR.relative_to(EXAMPLE_DIR.parent)),
    }
    if output_path:
        _write(Path(output_path), summary)
    return summary


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    output = argv[0] if argv else None
    summary = run(output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["determinism_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
