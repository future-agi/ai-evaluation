---
kind: agent-learning.docs-page.v1
track: prove
objective: reliability
stage: prove
backing:
  - examples/bench_ci_gate.py
artifact_kinds: []
commands:
  - python examples/bench_ci_gate.py artifacts/in-ci.json
  - agent-learn bench examples/bench_suites/coding_starter.json --mode artifact_in --reference --evidence-class local_gate --no-telemetry -o artifacts/bench-ci.json --quiet
postcondition: python -c "import json; p=json.load(open('artifacts/in-ci.json')); assert p['gate']['passed'] is True, p['gate']; assert p['aggregate']['honesty']['any_overclaim'] is False, p['aggregate']['honesty']; assert p['aggregate']['scored']==3, p['aggregate']; print('ok')"
claims: []
doctor_checks:
  - missing_public_modules
opt_in_lane: false
---

# Benchmark in CI: gate a merge on an honest pass_rate

> **Twin:** [`examples/bench_ci_gate.py`](../../examples/bench_ci_gate.py)
> · `artifact_in` control mode · offline, no credentials, no Docker.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

A bench result is only useful in CI if its headline number cannot be inflated by
the things that usually inflate it: a missing sandbox read as a failure, a no-op
candidate read as a pass, or a fixture run wearing a live badge. This page wires a
bench run into a **single merge gate** over the harness's honesty model, so the
gate blocks a real regression and only a real regression.

Every per-task row carries two honest stamps the harness never lets the caller
inflate:

- **`execution_class`** is DERIVED from the substrate, never asserted above it:
  `executable` (the code really ran), `typed_only` (the world is typed but not yet
  executed), or `fixture` (a committed replay). A coding task that executes in a
  sandbox is stamped `executable` and nothing else can claim it.
- **`evidence_class`** records HOW the run was witnessed: `local_gate` (an offline
  self-check), `live_lane` / `live_stressed` (a real, possibly perturbed lane), or
  `captured_fixture` (a replayed capture).

The **overclaim tripwire** fires when a non-live `execution_class` carries a live
`evidence_class` — a typed-only or fixture row claiming `live_lane`. That row is
flagged `overclaim: True`, and the aggregate's `honesty.any_overclaim` rolls it
up. The CI gate reads that flag: a perfect `pass_rate` with an overclaim still
blocks.

Two more honesty rules the gate depends on:

- **`void` rows are excluded from `pass_rate`.** A task with no submission, or one
  whose sandbox could not start (no Docker daemon, image pull failure), is `void`
  — never a `fail`. `pass_rate` is computed over **scored** tasks only
  (`scored = count - void`), so an infra outage that voids every row reads as
  "nothing scored", not "0% passed".
- **The held-out oracle decides the verdict, and the verdict is all-or-nothing.**
  Each coding task ships a `checks` oracle the candidate cannot import, so it
  cannot reflect the expected answers. A task is `pass` only when *every* held-out
  check passes (`result.scalar >= 1.0`); a no-op candidate fails because the checks
  fail. That is the anti-gaming defence this lane relies on — a candidate cannot
  game a scorer it cannot read.

The failure classes this page targets: a green CI run that hid a missing sandbox,
a gate that a no-op candidate slipped through, and a pass_rate diluted by tasks
that never ran.

## 2. Run it

Score the shipped `coding_starter` suite against its own gold references (so the
run is deterministic and credential-free), apply the gate, and write the artifact:

```bash
python examples/bench_ci_gate.py artifacts/in-ci.json
```

The example exits non-zero when the gate fails, so it drops straight into a CI
step. The same scoring from the CLI — usable verbatim in a pipeline — emits the
unified bench result, and the postcondition below gates on the file:

```bash
agent-learn bench examples/bench_suites/coding_starter.json --mode artifact_in --reference --evidence-class local_gate --no-telemetry -o artifacts/bench-ci.json --quiet
```

The gate itself is one reduction over the aggregate — clear the pass_rate bar AND
carry no overclaim:

```python
from fi.alk import bench

suite = bench.load_coding_suite("examples/bench_suites/coding_starter.json")
result = bench.run_bench(
    "examples/bench_suites/coding_starter.json",
    control_mode="artifact_in",
    submission=bench.reference_submission(suite),  # swap in {task_id: your_source}
    evidence_class="local_gate",
    emit_telemetry=False,
)
agg = result["aggregate"]
gate_ok = agg["pass_rate"] >= 1.0 and not agg["honesty"]["any_overclaim"]
print(gate_ok, agg["scored"], agg["void"], agg["honesty"]["any_overclaim"])
```

To gate **your** agent, pass a submission map of `task_id -> candidate source`
instead of the gold reference, and tune `MIN_PASS_RATE` to your bar.

## 3. What you built

Run the postcondition verbatim:

```bash
python -c "import json; p=json.load(open('artifacts/in-ci.json')); assert p['gate']['passed'] is True, p['gate']; assert p['aggregate']['honesty']['any_overclaim'] is False, p['aggregate']['honesty']; assert p['aggregate']['scored']==3, p['aggregate']; print('ok')"
```

The artifact records the `gate` decision (`passed`, the `min_pass_rate` bar, the
observed `pass_rate`, `pass_rate_ok`, `any_overclaim`, `scored` / `void` / `count`,
and a `reasons` list that explains a block), the full `aggregate` (`count`,
`scored`, `void`, `passed`, `pass_rate`, `mean_score`, the `by_modality` /
`by_world_kind` / `by_execution_class` rollups, and the `honesty` block), and one
row per task carrying the unified `result`, the `verdict`, and the honesty fields.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| gate `passed: False` with a `pass_rate` reason | real regression OR the bar is set above what the suite resolves — read `gate.reasons` and the per-task `verdict`s | `missing_public_modules` |
| gate `passed: False` with `any_overclaim` | a non-live row was stamped a live `evidence_class` — drop to `local_gate` / `captured_fixture` for an offline run | `missing_public_modules` |
| `pass_rate` is `1.0` but `scored` is below `count` | tasks went `void` (no submission, or the sandbox could not start) — they are excluded, not failed; check `void` and each row's `error` | `missing_public_modules` |
| a candidate you expected to fail shows `verdict: pass` | the held-out oracle is too weak, or it was bypassed — read the task's `checks` and the row's `result.pass_fail` (every check must be `True` for a `pass`) | `missing_public_modules` |
| `BenchError: artifact_in requires submission=...` | a coding suite was scored in `artifact_in` with no submission — pass `--reference` / `--submission-file` (CLI) or `submission={task_id: source}` (API). A non-coding suite raises a different `BenchError` (`artifact_in currently requires a coding bench suite`) | `missing_public_modules` |

## 5. Prove it / keep it

The coding lane this gate reads — the held-out oracle, the all-or-nothing verdict,
and the subprocess sandbox — is covered in
[benchmark-coding](../eval/benchmark-coding.md). To wire the same gate into a full
pipeline alongside the simulation, red-team, and release checks, see
[release-check-in-your-ci](./release-check-in-your-ci.md). For the run ledger that
records each gated run over time, see [run-ledger](./run-ledger.md).
