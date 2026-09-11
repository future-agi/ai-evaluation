---
kind: agent-learning.docs-page.v1
track: eval
objective: capability
stage: evaluate
backing:
  - examples/sdk_task_evaluation.py
artifact_kinds:
  - agent-learning.eval.v1
  - agent-learning.artifact-evaluation.v1
commands:
  - agent-learn eval examples/eval_suite.json --output artifacts/eval.json
  - agent-learn eval-task examples/task_evidence.json --config examples/task_evidence_eval_config.json --threshold 0.85 --output artifacts/task-eval.json
postcondition: python -c "import json; p=json.load(open('artifacts/eval.json')); assert p['kind']=='agent-learning.eval.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
opt_in_lane: false
---

# Evaluate any task

> **Twin:** [`examples/sdk_task_evaluation.py`](../../examples/sdk_task_evaluation.py)
> · emits `agent-learning.eval.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

Whether your agent did the task — not whether one string matched. The eval
track has two on-ramps, and this guide shows both. The first is the suite
on-ramp: a JSON file of providers, prompts, and tests with assertions
(`examples/eval_suite.json` is the complete shape — an `echo` provider, one
templated prompt, `contains` assertions). It answers "does this prompt and
provider combination still produce acceptable output?" and it is the format
the rest of the track reuses for thresholds and CI wiring.

The second on-ramp evaluates *evidence of work*: a transcript with messages,
tool calls, and final environment state. The twin for this page builds exactly
that — a refund-approval task with `approve_refund` and `write_safe_memory`
tool calls — and scores it against a task config (description, expected
result, success criteria, required tools, forbidden patterns). This is the
failure class string assertions miss: an agent that says the right words but
never called the verification tool, or completed the task while leaking a
canary value into memory.

Everything here runs offline. The suite uses a deterministic echo provider;
the evidence evaluation scores a recorded transcript with local agent-report
metrics. No model keys are involved, so a red CI run means your agent or your
rubric changed — not a provider hiccup.

## 2. Run it

CLI — suite first, then evidence:

```bash
agent-learn eval examples/eval_suite.json --output artifacts/eval.json
agent-learn eval-task examples/task_evidence.json \
  --config examples/task_evidence_eval_config.json \
  --threshold 0.85 --output artifacts/task-eval.json
```

SDK — the same two operations:

```python
from fi.alk import evals

suite_result = evals.run_eval_suite_file("examples/eval_suite.json")
task_result = evals.evaluate_task_evidence_file(
    "examples/task_evidence.json",
    config=evals.load_artifact_file("examples/task_evidence_eval_config.json"),
    threshold=0.85,
)
```

The twin, `examples/sdk_task_evaluation.py`, builds the evidence and config in
code via `evals.build_task_evaluation_config(...)` instead of loading files —
read it when your evidence comes from your own runtime rather than disk.

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/eval.json')); assert p['kind']=='agent-learning.eval.v1', p['kind']; print('ok')"
```

`artifacts/eval.json` is an `agent-learning.eval.v1` artifact: per-test
results, assertion outcomes, a score, and an exit code CI can consume.
`artifacts/task-eval.json` is `agent-learning.artifact-evaluation.v1`: metric
averages (task completion, tool selection accuracy), findings, and a
pass/fail verdict against your threshold.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| suite rejected before any test runs | config fault | the `agent-learn eval` error line names the bad field |
| score 0.0 with all criteria missing | evidence fault | your transcript lacks the messages/tool_calls the config expects |

## 5. Prove it / keep it

Each surface in this track gets its own page: [eval suites](eval-suites.md)
for the assertion format, [artifact evals](artifact-evals.md) for scoring
saved run artifacts, [task evidence](task-evidence.md) for the evidence
format and config synthesis, [judge reliability](judge-reliability.md) for
checking that your scorer itself is stable, and
[eval hooks](eval-hooks.md) for plugging in your own judge endpoint. When an
eval matters, wire it into a suite job (see
`examples/task_evidence_suite.json`) and let `agent-learn release-check`
re-run it on every cut — the page you just completed stays a living check,
not a one-off demo.
