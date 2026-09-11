---
kind: agent-learning.docs-page.v1
track: eval
objective: capability
stage: evaluate
backing:
  - examples/bench_coding_quickstart.py
artifact_kinds: []
commands:
  - python examples/bench_coding_quickstart.py artifacts/bench-coding.json
  - agent-learn bench examples/bench_suites/coding_starter.json --mode artifact_in --reference
postcondition: python -c "import json; p=json.load(open('artifacts/bench-coding.json')); assert p['aggregate']['pass_rate']==1.0, p; assert p['aggregate']['scored']==3, p; print('ok')"
claims: []
doctor_checks:
  - missing_public_modules
opt_in_lane: false
---

# Coding benchmark: score candidate code against a held-out oracle

> **Twin:** [`examples/bench_coding_quickstart.py`](../../examples/bench_coding_quickstart.py)
> · `artifact_in` control mode · offline, no credentials, no Docker.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

The unified bench harness scores a coding agent the way a real benchmark does:
the agent produces candidate code, and a **held-out oracle** — a set of
`check_*` functions that ship with the task, executed by a harness-written
runner and never imported by the candidate — decides the verdict. The candidate
is not given the oracle, so it cannot simply reflect the expected answers; it
has to actually solve the task. (The subprocess lane shares a tempdir with the
candidate and so holds only against accidental gaming; run untrusted agent
output under `sandbox="docker"` — see section 5.)

A coding bench suite (`agent-learning.bench-suite.v1`) carries, per task, an
`instruction`, the held-out `checks`, a gold `reference_solution`, and a
`guards` block declaring the anti-gaming contract. You score it through one
call — `run_bench(suite, control_mode="artifact_in", submission=...)` — and get
one unified `Result` per task plus an honest aggregate.

The verdict is **all-or-nothing**: a task is resolved only when every held-out
check passes. The failure classes this page targets: an oracle the candidate
can peek at, a "benchmark" that a no-op candidate still passes, and a runner
that reports a missing sandbox as an agent failure instead of a `void`.

## 2. Run it

Score the shipped `coding_starter` suite against its own gold references (so the
run is deterministic and credential-free), and write the artifact:

```bash
python examples/bench_coding_quickstart.py artifacts/bench-coding.json
```

The same scoring from the CLI, against the suite's reference solutions:

```bash
agent-learn bench examples/bench_suites/coding_starter.json --mode artifact_in --reference
```

To grade **your** agent instead of the gold reference, pass a submission map of
`task_id -> candidate source`:

```python
from fi.alk import bench

suite = bench.load_coding_suite("examples/bench_suites/coding_starter.json")
submission = {"fibonacci": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n"}
result = bench.run_bench(
    "examples/bench_suites/coding_starter.json",
    control_mode="artifact_in",
    submission=submission,
)
print(result["aggregate"]["pass_rate"], result["aggregate"]["scored"])
```

A task with no submission is recorded `void` (never silently passed); a task
whose sandbox could not run is `void` too — neither counts against `pass_rate`,
which is computed over *scored* tasks only.

## 3. What you built

Run the postcondition verbatim:

```bash
python -c "import json; p=json.load(open('artifacts/bench-coding.json')); assert p['aggregate']['pass_rate']==1.0, p; assert p['aggregate']['scored']==3, p; print('ok')"
```

The artifact records the suite name and version, the aggregate (`count`,
`scored`, `void`, `passed`, `pass_rate`, `mean_score`, plus the `by_modality` /
`by_world_kind` / `by_execution_class` rollups and the `honesty` block), and one
row per task carrying the unified `result`, the `verdict`, and the honesty fields
(`execution_class`, `evidence_class`, `overclaim`, `sandbox`).

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| a task you submitted shows `verdict: void` | no submission for that `task_id`, or the sandbox could not start | `missing_public_modules` |
| `pass_rate` lower than expected on correct code | a held-out check is stricter than the instruction implies — read the task's `checks` | `missing_public_modules` |
| `CodingSuiteError: ... missing required field` | the suite is malformed — every task needs `id` / `instruction` / `checks` / `reference_solution` and a `guards.min_guard_count >= 1` | `missing_public_modules` |
| a no-op candidate "passes" | the held-out oracle was bypassed — file a bug; never ship | `missing_public_modules` |

## 5. Prove it / keep it

For untrusted agent output, run the same suite under OS-level isolation with
`sandbox="docker"` — see [benchmark-sandboxes](./benchmark-sandboxes.md). For the
forge- and oracle-read-resistant grading model (a held-out grader that runs
after the candidate is killed), see
[benchmark-command-graded](./benchmark-command-graded.md). The harness, its three
control modes, and the unified `Result` are covered in
[benchmark-overview](./benchmark-overview.md).
