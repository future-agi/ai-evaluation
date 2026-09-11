---
kind: agent-learning.docs-page.v1
track: eval
objective: capability
stage: evaluate
backing:
  - examples/bench_overview.py
artifact_kinds: []
commands:
  - python examples/bench_overview.py artifacts/bench-overview.json
  - agent-learn bench examples/bench_suites/coding_starter.json --mode artifact_in --reference
  - agent-learn bench examples/bench_suites/pull_starter.json --mode pull --agent '{"type":"reference"}'
postcondition: python -c "import json; p=json.load(open('artifacts/bench-overview.json')); L=p['lanes']; assert set(L)=={'coding','rl','voice'}, L; assert all(l['aggregate']['pass_rate']==1.0 and l['aggregate']['scored']==l['aggregate']['count'] for l in L.values()), L; assert p['result_shape_consistent'] is True, p; assert set(p['modalities'])=={'coding','rl','voice'}, p; print('ok')"
claims: []
doctor_checks:
  - missing_public_modules
opt_in_lane: false
---

# Benchmark harness: one Task↔Verifier contract, three control modes, one Result

> **Twin:** [`examples/bench_overview.py`](../../examples/bench_overview.py)
> · `artifact_in` + `pull` lanes exercised, `push` documented · offline, no
> credentials, no Docker. A coding agent can complete this page from the
> frontmatter alone.

## 1. What you are testing

The unified bench harness is a single contract with **one constant** and **two
dimensions**. The constant is the fixed **Task↔Verifier coupling**: a bench
suite carries (or references) its own oracle, so the thing that decides the
verdict never drifts from the thing being asked. The two dimensions are the
**modality** (coding / rl / voice / text / tool / …) and the **control mode** —
*who drives whom*:

| Control mode | Who drives | What you give it | This page's lane |
| --- | --- | --- | --- |
| `push` | the harness drives the agent over a task dataset | an `agent` spec | documented (needs an agent) |
| `artifact_in` | nobody live — you submit and score against a held-out oracle | a `submission` map | coding + voice |
| `pull` | the agent drives a live environment via `reset`/`step` | a policy / `{"type": "reference"}` | rl |

Across every mode and every modality, each per-task verdict projects into **one
unified `Result`** — `{scalar, components, pass_fail, explanation}`. The
modality decides what `pass_fail` *means* (`{"verdict": …}` for push,
`{check_name: …}` for coding, `{"goal_reached": …}` for rl, `{"voice": …}` for
voice), but the row-level `verdict` plus `result.scalar` is the portable signal
you read the same way everywhere.

The failure classes this page targets: a harness that forks per modality so
"one number" stops meaning the same thing; a runner that reports a missing
sandbox as an agent failure instead of a `void`; and a "benchmark" whose oracle
the candidate can peek at.

## 2. Run it

Run the three credential-free lanes end to end and assemble one combined
artifact that proves many modes, many modalities, one `Result` shape:

```bash
python examples/bench_overview.py artifacts/bench-overview.json
```

Each lane individually from the CLI — the coding `artifact_in` self-check
against the suite's reference solutions, and the rl `pull` lane driven by the
environment's reference policy:

```bash
agent-learn bench examples/bench_suites/coding_starter.json --mode artifact_in --reference
agent-learn bench examples/bench_suites/pull_starter.json --mode pull --agent '{"type":"reference"}'
```

The harness is one call per lane; only the control mode and the modality of the
suite change:

```python
from fi.alk import bench

# artifact_in / coding — score a candidate against a held-out check oracle.
coding_suite = bench.load_coding_suite("examples/bench_suites/coding_starter.json")
coding = bench.run_bench(
    "examples/bench_suites/coding_starter.json",
    control_mode="artifact_in",
    submission=bench.reference_submission(coding_suite),  # swap for {task_id: source}
)

# pull / rl — the agent (a policy obs->action, or a reference spec) drives the env.
rl = bench.run_bench(
    "examples/bench_suites/pull_starter.json",
    {"type": "reference"},
    control_mode="pull",
)

for res in (coding, rl):
    agg = res["aggregate"]
    print(res["control_mode"], res["modalities"], agg["pass_rate"], agg["scored"])
```

The third mode, `push`, drives a **live agent** over a task dataset, so it needs
a real agent rather than a held-out artifact. The shape is identical — only the
arguments differ:

```python
from fi.alk import bench

# push / text — the HARNESS drives the agent across a task dataset.
result = bench.run_bench(
    "examples/task_datasets/support_starter.json",
    {"type": "scripted", "content": "Our refund policy is at /help/refunds (30-day window)."},
    control_mode="push",          # the default; the harness calls the agent per task
    evidence_class="captured_fixture",
)
print(result["control_mode"], result["aggregate"]["pass_rate"])
```

A task with no submission (or a `pull` env that could not start, or a sandbox
that could not run) is recorded `void` — never silently passed. `pass_rate` is
computed over *scored* tasks only, so an infra failure that voids every row does
**not** read as "0% passed".

## 3. What you built

Run the postcondition verbatim:

```bash
python -c "import json; p=json.load(open('artifacts/bench-overview.json')); L=p['lanes']; assert set(L)=={'coding','rl','voice'}, L; assert all(l['aggregate']['pass_rate']==1.0 and l['aggregate']['scored']==l['aggregate']['count'] for l in L.values()), L; assert p['result_shape_consistent'] is True, p; assert set(p['modalities'])=={'coding','rl','voice'}, p; print('ok')"
```

The artifact carries a `lanes` block with one record per modality (`coding`,
`rl`, `voice`), each holding its `control_mode`, `modalities`, the full
`aggregate` (`count`, `scored`, `void`, `passed`, `pass_rate`, `mean_score`, the
`by_modality` / `by_world_kind` / `by_execution_class` rollups, and the
`honesty` block), and its per-task `verdicts`. The top-level
`result_shape_consistent: true` is the headline proof: a sample `Result` row
from every lane exposes the **same** key set
(`scalar` / `components` / `pass_fail` / `explanation`) — one Result across
three modalities and two control modes.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `BenchError: unknown control_mode '…'` | mode must be one of `push` / `artifact_in` / `pull` | `missing_public_modules` |
| `BenchError: artifact_in currently requires a coding bench suite` | you used `artifact_in` on a task dataset — that mode scores coding/voice suites; use `push` for a dataset | `missing_public_modules` |
| `BenchError: pull bench suites run under control_mode='pull'` | the suite declares `"control": "pull"` but you passed a different `--mode` | `missing_public_modules` |
| a lane's row shows `verdict: void` | no submission for that `task_id`, an unknown `pull` env kind, or a sandbox that could not start — never counted against `pass_rate` | `missing_public_modules` |
| `result_shape_consistent: false` | a modality projected a Result missing a key — file a bug; the unified shape is the invariant | `missing_public_modules` |

## 5. Prove it / keep it

Each modality has its own page that goes deep on its verifier and anti-gaming
contract:

- the coding `artifact_in` lane and its held-out check oracle —
  [benchmark-coding](./benchmark-coding.md);
- the hardened command/artifact-graded coding tier (held-out grader runs after
  the candidate is killed) — [benchmark-command-graded](./benchmark-command-graded.md);
- OS-level isolation for untrusted candidate output (`sandbox="docker"`) —
  [benchmark-sandboxes](./benchmark-sandboxes.md);
- the `pull` rl lane (the agent drives a live env via `reset`/`step`) —
  [benchmark-pull-rl](./benchmark-pull-rl.md);
- the voice episode verifier (latency / turn-taking / barge-in / content) —
  [benchmark-voice](./benchmark-voice.md).
