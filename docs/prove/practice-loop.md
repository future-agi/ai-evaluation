---
kind: agent-learning.docs-page.v1
track: prove
objective: reliability
stage: prove
backing:
  - examples/sdk_practice_loop.py
artifact_kinds:
  - agent-learning.practice-loop.v1
  - agent-learning.practice-result.v1
  - agent-learning.practice-report.v1
  - agent-learning.consolidated-lesson.v1
commands:
  - python examples/sdk_practice_loop.py artifacts/practice-loop.json
postcondition: python -c "import json; p=json.load(open('artifacts/practice-loop.json')); assert p['kind']=='agent-learning.practice-loop-readiness.v1', p; assert p['determinism_equal'] is True, p; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Practice loop (search-backed): spaced regression replay that never weakens the veto

> **Twin:** [`examples/sdk_practice_loop.py`](../../examples/sdk_practice_loop.py)
> · emits `agent-learning.practice-loop.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

The practice loop (search-backed) runs a six-phase round — assess, diagnose,
drill, update, consolidate, calibrate — over the simulation contract's declared
objective. Drills target measured ZPD; updates are layer-scoped; consolidated
lessons land in an append-only store with a spaced-replay schedule. The
load-bearing invariant: spacing governs between-promotion standing health, but
at every candidate promotion the full frozen-row union replays regardless of
schedule state — the promotion veto is never weakened.

The failure classes this page targets: a non-deterministic loop, a schedule
state machine that drifts, a promotion that skips schedule-quiet rows, and a
planted regression that goes undetected.

## 2. Run it

Generate the committed fixtures (the determinism pair, the schedule-history
matrix, the zero-due promotion sweep, the non-forgetting interference run, and
the budget-conservation ledger):

```bash
python examples/sdk_practice_loop.py artifacts/practice-loop.json
```

## 3. What you built

Run the postcondition verbatim:

```bash
python -c "import json; p=json.load(open('artifacts/practice-loop.json')); assert p['kind']=='agent-learning.practice-loop-readiness.v1', p; assert p['determinism_equal'] is True, p; print('ok')"
```

The artifact records that two identical-seed runs produce byte-identical phase
artifacts after the envelope strip.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| build refused with `budget_undeclared` | the manifest has no `eval_budget` | `missing_engine_modules` |
| build refused with `objective_guards_missing` | the simulation objective has no guards | `missing_engine_modules` |
| a promotion replayed fewer rows than the full union | a schedule filter leaked into the sweep | `public_boundary_passed` |

## 5. Prove it / keep it

The `practice_loop_readiness` release gate recomputes these committed fixtures
on every `release-check`: determinism, the schedule transition table, the
zero-due promotion-veto sweep, the non-forgetting interference run, budget
conservation, and the claims-lint all gate the release.
