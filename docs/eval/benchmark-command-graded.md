---
kind: agent-learning.docs-page.v1
track: eval
objective: reliability
stage: evaluate
backing:
  - examples/bench_command_graded.py
artifact_kinds: []
commands:
  - python examples/bench_command_graded.py artifacts/bench-command-graded.json
  - agent-learn bench examples/bench_suites/coding_command_starter.json --mode artifact_in --reference
postcondition: python -c "import json; p=json.load(open('artifacts/bench-command-graded.json')); a=p['aggregate']; assert a['pass_rate']==1.0 and a['scored']==2 and a['void']==0, a; rows=p['per_task']; assert all(r['raw']['grading']=='command' and r['raw']['grader_exit']==0 for r in rows), rows; print('ok')"
claims: []
doctor_checks:
  - missing_public_modules
opt_in_lane: false
---

# Command-graded benchmark: a held-out grader runs after the candidate is killed

> **Twin:** [`examples/bench_command_graded.py`](../../examples/bench_command_graded.py)
> · `artifact_in` control mode · offline, no credentials, no Docker.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

The convenience coding tier imports the held-out `check_*` functions in the same
process as the candidate — fine for trusted, shipped code, but a candidate that
actively attacks the harness could read the oracle or print a forged result. The
**command-graded** tier closes both holes by changing the grading *model*, not by
bolting on isolation:

1. The task hands the candidate a working directory and a way to **run** it. The
   candidate produces files / output. No grader is present yet.
2. The candidate command finishes and its processes are killed.
3. **Only then** is a **held-out grader** materialised — in its own directory
   (`$GRADER_DIR`) the candidate phase never saw — and run.
4. The verdict is the grader's **exit code** (0 = pass) plus an optional
   grader-written `reward.json`. The candidate's stdout is never parsed for a
   verdict.

This temporal + path separation is what defeats two distinct failure classes:

- **Output forgery** is structurally impossible: the grader's exit code (and its
  own reward file) is the only verdict authority, so a candidate that prints a
  fake `PASS` line changes nothing. The reward file lives in `$GRADER_DIR`, which
  the candidate phase never knew about.
- **Oracle-read** is structurally impossible *in time*: the expected cases and
  tests are written to disk only **after** the candidate has finished and been
  killed, so there is no moment at which the candidate co-runs with the oracle and
  could reflect its expected values. In the Docker lane the grader files are also
  owned by a different user the candidate uid cannot read — separation in *space*
  on top of separation in *time*.

It is multi-language for free: the candidate `build` and the `grader_cmd` are
arbitrary shell. The shipped suite grades a Python task and a bash task in the
same run.

## 2. Run it

Score the shipped `coding_command_starter` suite against its own gold reference
files (so the run is deterministic and credential-free), and write the artifact:

```bash
python examples/bench_command_graded.py artifacts/bench-command-graded.json
```

The same scoring from the CLI, against the suite's reference files:

```bash
agent-learn bench examples/bench_suites/coding_command_starter.json --mode artifact_in --reference
```

To grade **your** agent instead of the gold reference, pass a submission map of
`task_id -> {path: content}`. A command-graded candidate is a **file map** (the
files the candidate wrote), not a source string:

```python
from fi.alk import bench

suite = bench.load_coding_suite("examples/bench_suites/coding_command_starter.json")
submission = {
    "sum-stdin-python": {
        "solution.py": "import sys\na, b = map(int, sys.stdin.read().split())\nprint(a + b)\n"
    }
}
result = bench.run_bench(
    "examples/bench_suites/coding_command_starter.json",
    control_mode="artifact_in",
    submission=submission,
    sandbox="subprocess",
)
print(result["aggregate"]["pass_rate"], result["aggregate"]["scored"])
```

A task with no submission is recorded `void` (never silently passed); a task whose
sandbox could not start at all (no Docker daemon, missing `grader_cmd`, unknown
sandbox — anything tagged `raw.infra_error`) is `void` too — neither counts against
`pass_rate`, which is computed over *scored* tasks only. A grader that actually
runs but exits non-zero (a real failing check, or the grader itself crashing) is a
`fail`, not a `void` — the lane ran, so its verdict counts.

## 3. What you built

Run the postcondition verbatim:

```bash
python -c "import json; p=json.load(open('artifacts/bench-command-graded.json')); a=p['aggregate']; assert a['pass_rate']==1.0 and a['scored']==2 and a['void']==0, a; rows=p['per_task']; assert all(r['raw']['grading']=='command' and r['raw']['grader_exit']==0 for r in rows), rows; print('ok')"
```

The artifact records the suite name and version, the aggregate (`count`, `scored`,
`void`, `passed`, `pass_rate`, `mean_score`, plus the `by_modality` /
`by_world_kind` / `by_execution_class` rollups and the `honesty` block), and one
row per task. Each command-graded row carries the unified `result` whose `scalar`
and `pass_fail` come from the grader (`grader_exit_ok` in `components`, the
grader's per-case `checks` in `pass_fail`), the honesty fields
(`execution_class`, `evidence_class`, `overclaim`, `sandbox`), and a `raw` block
proving the grading path: `raw.grading == "command"` and `raw.grader_exit == 0`.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| a task you submitted shows `verdict: void` | no submission for that `task_id`, or the sandbox could not start at all (no Docker daemon, missing `grader_cmd`); the row's `error` field carries the reason (an `infra:` prefix marks a lane that never ran) | `missing_public_modules` |
| `verdict: fail` with `raw.grader_exit` non-zero on code you believe correct | the held-out grader is stricter than the instruction implies — read the task's `grader_files` (`spec.json`) for the exact expected cases | `missing_public_modules` |
| `CodingSuiteError: ... missing required field` | a command-graded task needs `id` / `instruction` / `grader_cmd` / `grader_files` / `reference_files` and a `guards.min_guard_count >= 1` | `missing_public_modules` |
| candidate prints `score=1` yet the row is `fail` | working as designed — the verdict is the grader's exit code, never candidate stdout; forged output cannot pass | `missing_public_modules` |
| `verdict: fail` with `raw.timed_out: true` | the candidate `build` or the grader exceeded the task `timeout_s` — the lane ran out of wall-clock, so it scores `0.0` (`fail`); raise the task `timeout_s` if the work is legitimately slow | `missing_public_modules` |

## 5. Prove it / keep it

For untrusted agent output, run the same suite under OS-level isolation with
`sandbox="docker"`: a per-task, network-off, capped, ephemeral container where the
candidate runs as an unprivileged uid and the grader files land in a root-owned
directory the candidate cannot read — separation in space on top of separation in
time. See [benchmark-sandboxes](./benchmark-sandboxes.md). For the convenience
`check_*` tier (in-process held-out oracle, trusted code only), see
[benchmark-coding](./benchmark-coding.md). The harness, its three control modes,
and the unified `Result` are covered in [benchmark-overview](./benchmark-overview.md).
