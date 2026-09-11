---
kind: agent-learning.docs-page.v1
track: eval
objective: reliability
stage: evaluate
backing:
  - examples/sdk_task_evaluation.py
artifact_kinds:
  - agent-learning.artifact-evaluation.v1
commands:
  - agent-learn eval-artifact examples/fixtures/task_artifacts/refund_task_run.json --config examples/artifact_task_eval_config.json --threshold 0.85 --output artifacts/artifact-eval.json
postcondition: python -c "import json; p=json.load(open('artifacts/artifact-eval.json')); assert p['kind']=='agent-learning.artifact-evaluation.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
opt_in_lane: false
---

# Artifact evals

> **Twin:** [`examples/sdk_task_evaluation.py`](../../examples/sdk_task_evaluation.py)
> · emits `agent-learning.artifact-evaluation.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

A run you already executed. Simulations, red-team campaigns, and
optimizations all leave saved artifacts behind; an artifact eval re-scores
one of those files with agent-report metrics, without re-running anything.
The input here is `examples/fixtures/task_artifacts/refund_task_run.json` —
an `agent-learning.run.v1` artifact whose report carries messages, metrics
(`task_completion`, `tool_selection_accuracy`, `world_contract_quality`),
and environment state including a `task_evidence` block.

The failure class is post-hoc: you discover a new rule after the run
happened. The config file (`examples/artifact_task_eval_config.json`)
declares the task description, expected result, success criteria, and
`forbidden_patterns` — including a canary marker (`canary_exfiltrated":
true`) and credential words. Evaluating the saved artifact against that
config answers "did any recorded run violate the rule we just wrote?"
across your whole artifact history, at file-read speed.

Because the input is a file and the metrics are local, the result is
reproducible: the same artifact and config produce the same verdict on every
machine. That is what makes artifact evals safe to put in CI gates — the
covering gate for this page's twin does exactly this evaluation, against
exactly these files, on every release-check.

## 2. Run it

CLI:

```bash
agent-learn eval-artifact examples/fixtures/task_artifacts/refund_task_run.json \
  --config examples/artifact_task_eval_config.json \
  --threshold 0.85 --output artifacts/artifact-eval.json
```

SDK — the same operation:

```python
from fi.alk import evals

config = evals.load_artifact_file("examples/artifact_task_eval_config.json")
result = evals.evaluate_artifact_file(
    "examples/fixtures/task_artifacts/refund_task_run.json",
    config=config,
    threshold=0.85,
    name="refund-artifact-evaluation",
)
```

`eval-artifact` accepts any saved Agent Learning artifact — run, red-team,
or optimization output — and locates the report inside it. The related suite
form, `examples/artifact_task_eval_suite.json`, uses the `artifact` provider
type to pull individual fields (scores, evidence flags, framework name) out
of the artifact by path and assert on them with the standard test grammar.

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/artifact-eval.json')); assert p['kind']=='agent-learning.artifact-evaluation.v1', p['kind']; print('ok')"
```

The artifact carries a `summary` with the score against your threshold, case
counts, finding counts, the source artifact's kind and status, and the
environment-state keys the report exposed — enough to audit *why* the
verdict came out the way it did without opening the source artifact.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `artifact root must be an object` | input fault | the file is not a saved Agent Learning artifact |
| score drops with `forbidden_patterns` findings | real catch | the recorded run contains a pattern your config bans |

## 5. Prove it / keep it

An artifact eval that caught something becomes a permanent rule: keep the
config file in your repo next to the artifact fixtures it polices, and add
the evaluation as a suite job so every release re-checks it. When new runs
land, evaluate them with the same config before promoting them to baselines
— `agent-learn baseline` then `agent-learn compare` closes the loop, so a
run that violates a rule you learned the hard way can never become the
reference run again.
