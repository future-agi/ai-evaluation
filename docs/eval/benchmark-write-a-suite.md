---
kind: agent-learning.docs-page.v1
track: eval
objective: capability
stage: evaluate
backing:
  - examples/bench_custom_suite.py
artifact_kinds: []
commands:
  - python examples/bench_custom_suite.py artifacts/bench-custom-suite.json
  - agent-learn bench examples/bench_suites/coding_starter.json --mode artifact_in --reference
postcondition: python -c "import json; p=json.load(open('artifacts/bench-custom-suite.json')); assert p['aggregate']['pass_rate']==1.0, p; assert p['aggregate']['scored']==1, p; assert p['suite_name']=='is_even_suite', p; print('ok')"
claims: []
doctor_checks:
  - missing_public_modules
opt_in_lane: false
---

# Write a bench suite: author your own coding tasks with a held-out oracle

> **Twin:** [`examples/bench_custom_suite.py`](../../examples/bench_custom_suite.py)
> · builds a suite in-memory · `artifact_in` control mode · offline, no
> credentials, no Docker. A coding agent can complete this page from the
> frontmatter alone.

## 1. What you are testing

Every other coding-bench page loads a *shipped* suite. This page teaches you to
author your **own** — the `agent-learning.bench-suite.v1` shape — so you can grade
your agent on tasks you control.

A coding bench suite is a JSON object (or an in-memory mapping) with this shape:

- **top level:** `kind` (exactly `agent-learning.bench-suite.v1`), `name`,
  `version`, `language` (`python` today), `modality` (`coding`), and a non-empty
  `tasks` list of unique-`id` tasks.
- **each `checks`-graded task:** `id`, `instruction`, `checks` (the held-out
  oracle), `reference_solution` (the gold), and a `guards` block with
  `min_guard_count >= 1`.

The `checks` value is the heart of it: a Python source string defining one or
more `check_*` functions that `import solution` (the candidate, written by the
harness to a module named `solution`) and `assert` the expected behaviour. The
candidate **never imports the checks file**, so it cannot read the expected
answers — it has to actually solve the task. `bench.load_coding_suite(obj)`
validates the shape and raises `CodingSuiteError` if any field is missing.

This page builds a one-task `is_even` suite, validates it, scores its own gold
reference, and writes the artifact. The verdict is **all-or-nothing**: a task is
resolved only when every held-out check passes.

## 2. Run it

Build, validate, and score the hand-authored `is_even` suite in-memory, and
write the artifact:

```bash
python examples/bench_custom_suite.py artifacts/bench-custom-suite.json
```

For comparison, the same `artifact_in` scoring against a shipped suite's
reference solutions, from the CLI:

```bash
agent-learn bench examples/bench_suites/coding_starter.json --mode artifact_in --reference
```

The minimum a task must declare — `instruction`, the held-out `checks`, a
`reference_solution`, and `guards.min_guard_count >= 1` — then validate and score
it. A Mapping suite is compiled in place; no file is required:

```python
from fi.alk import bench

suite = {
    "kind": "agent-learning.bench-suite.v1",
    "name": "is_even_suite",
    "version": "1",
    "language": "python",
    "modality": "coding",
    "tasks": [
        {
            "id": "is_even",
            "instruction": "Implement is_even(n): True if n is even, else False.",
            "checks": (
                "import solution\n"
                "def check_even():\n    assert solution.is_even(4) is True\n"
                "def check_odd():\n    assert solution.is_even(7) is False\n"
            ),
            "reference_solution": "def is_even(n):\n    return n % 2 == 0\n",
            "guards": {"min_guard_count": 1, "oracle_held_out": True},
        }
    ],
}
validated = bench.load_coding_suite(suite)          # raises CodingSuiteError if malformed
submission = bench.reference_submission(validated)  # {task_id: gold source}
result = bench.run_bench(validated, control_mode="artifact_in", submission=submission)
print(result["aggregate"]["pass_rate"], result["aggregate"]["scored"])
```

To grade **your** agent instead of the gold reference, pass a submission map of
`task_id -> candidate source` rather than `reference_submission(...)`. A task
with no submission is recorded `void` (never silently passed); a task whose
sandbox could not run is `void` too — neither counts against `pass_rate`, which
is computed over *scored* tasks only.

### Two ways to grade — and why guards are mandatory

A task is graded one of two ways. The `checks` tier (used above) is the
convenience tier: the held-out `check_*` functions import the candidate
in-process. It catches *accidental* gaming — a no-op, a fake "success" print, a
wrong answer, a missing entrypoint all fail deterministically — but it is not a
boundary against an *adversarial* candidate that knows the runner's protocol.
For that, author `command`-graded tasks (`grader_cmd` + `grader_files` +
`reference_files`): a held-out grader runs **after** the candidate is killed and
emits its verdict via exit code — covered in
[benchmark-command-graded](./benchmark-command-graded.md).

Either way, `guards.min_guard_count >= 1` is **required**: `load_coding_suite`
rejects any task that does not declare at least one guard. The held-out oracle is
the real defence; the `guards` block forces the suite to *say so* — the
anti-gaming contract is explicit, not implied. A `sentinel` string ("a no-op or
fake-success candidate must fail the held-out checks") documents the intent for
the next author.

## 3. What you built

Run the postcondition verbatim:

```bash
python -c "import json; p=json.load(open('artifacts/bench-custom-suite.json')); assert p['aggregate']['pass_rate']==1.0, p; assert p['aggregate']['scored']==1, p; assert p['suite_name']=='is_even_suite', p; print('ok')"
```

The artifact records the suite name and version, the aggregate (`count`,
`scored`, `void`, `passed`, `pass_rate`, `mean_score`, plus the `by_modality` /
`by_world_kind` / `by_execution_class` rollups and the `honesty` block), and one
row per task carrying the unified `result`, the `verdict`, and the honesty fields
(`execution_class`, `evidence_class`, `overclaim`, `sandbox`). Scoring the gold
reference proves the oracle *accepts* a correct answer — the same self-check the
release gate runs over every shipped suite.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `CodingSuiteError: ... missing required field` | a task is missing `id` / `instruction` / `checks` / `reference_solution` | `missing_public_modules` |
| `CodingSuiteError: ... must declare guards.min_guard_count >= 1` | the `guards` block is absent or its count is `0` — add the anti-gaming contract | `missing_public_modules` |
| `CodingSuiteError: duplicate task id` | two tasks share an `id`; ids must be unique within a suite | `missing_public_modules` |
| `CodingSuiteError: not a ... suite` | the top-level `kind` is not exactly `agent-learning.bench-suite.v1` | `missing_public_modules` |
| your reference shows `verdict: void` | the sandbox could not start (read the row's `error`); `void` never counts against `pass_rate` | `missing_public_modules` |
| a wrong candidate "passes" | a `check_*` is weaker than the instruction — add cases for edges the instruction implies (zero, negatives, empties) | `missing_public_modules` |

## 5. Prove it / keep it

Once your suite scores its own reference at `pass_rate == 1.0`, run it repeatedly
against your agent's output and drill the tasks it fails. For untrusted agent
output, run the same suite under OS-level isolation with `sandbox="docker"` — see
[benchmark-sandboxes](./benchmark-sandboxes.md). For the forge- and
oracle-read-resistant grading model (a held-out grader that runs after the
candidate is killed), author `command`-graded tasks per
[benchmark-command-graded](./benchmark-command-graded.md). The harness, its three
control modes, and the unified `Result` are covered in
[benchmark-overview](./benchmark-overview.md); the shipped-suite walkthrough is
[benchmark-coding](./benchmark-coding.md).
