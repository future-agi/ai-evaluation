---
kind: agent-learning.docs-page.v1
track: eval
objective: capability
stage: evaluate
backing:
  - examples/sdk_task_evaluation_synthesis.py
artifact_kinds:
  - agent-learning.artifact-evaluation.v1
  - agent-learning.task-evidence.v1
commands:
  - agent-learn eval-task examples/task_evidence.json --config examples/task_evidence_eval_config.json --threshold 0.85 --output artifacts/task-evidence-eval.json
postcondition: python -c "import json; p=json.load(open('artifacts/task-evidence-eval.json')); assert p['kind']=='agent-learning.artifact-evaluation.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
opt_in_lane: false
---

# Task evidence

> **Twin:** [`examples/sdk_task_evaluation_synthesis.py`](../../examples/sdk_task_evaluation_synthesis.py)
> · emits `agent-learning.artifact-evaluation.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

The record of what an agent actually did: its input, its messages, the tool
calls it made with their results, and the final output. That record is task
evidence, and `examples/task_evidence.json` is the reference instance — a
refund-approval transcript where the assistant calls `approve_refund` and
`write_safe_memory`, each answered by a tool message, before declaring the
task complete.

Evidence evaluation scores that record against a config
(`examples/task_evidence_eval_config.json`): a task description, expected
result, success criteria, `required_tools` versus `available_tools`, and
forbidden patterns. The failure classes this separates cleanly: an agent
that *claimed* completion without calling a required tool scores low on
tool selection; one that called the right tools but leaked a banned pattern
fails on safety; one that did both right passes — and you can see which
metric moved.

Writing configs by hand does not scale to arbitrary tasks, so the twin for
this page demonstrates synthesis: `evaluate_task_evidence_auto` derives the
evaluation config *from the evidence itself* — task description, expected
result, and tool expectations are read out of the record — and then scores
against it. That is the path for heterogeneous task streams where each run
has its own contract.

## 2. Run it

CLI — evaluate the reference evidence with the hand-written config:

```bash
agent-learn eval-task examples/task_evidence.json \
  --config examples/task_evidence_eval_config.json \
  --threshold 0.85 --output artifacts/task-evidence-eval.json
```

SDK — the synthesis path the twin takes (no config file at all):

```python
from fi.alk import evals

evidence = evals.load_artifact_file("examples/task_evidence.json")
config = evals.synthesize_task_evaluation_config(evidence)
result = evals.evaluate_task_evidence_auto(evidence, threshold=0.9)
```

To persist evidence as a first-class artifact for later evaluation, use
`evals.write_task_evidence_file(evidence, path)` — it normalizes the record
into an `agent-learning.task-evidence.v1` file that `eval-task` accepts
directly.

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/task-evidence-eval.json')); assert p['kind']=='agent-learning.artifact-evaluation.v1', p['kind']; print('ok')"
```

The evaluation artifact reports metric averages (task completion, tool
selection accuracy, world contract quality), per-criterion findings, and a
verdict against the threshold. The evidence file itself, when normalized,
carries the `agent-learning.task-evidence.v1` kind — two artifacts, one for
the record and one for the judgment, so each can evolve independently.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| every criterion missing, score near zero | evidence fault | messages/tool_calls absent from the record you passed |
| required tool reported unused | real catch | the transcript shows the agent skipped a verification step |

## 5. Prove it / keep it

`examples/task_evidence_suite.json` shows the keep-it form: a suite job with
`command: eval-task`, the evidence and config paths, and a 0.85 threshold,
plus `required_capabilities` that name the metrics and environment-state
keys the job depends on — so a runner missing those capabilities fails
loudly instead of passing vacuously. Run it with `agent-learn suite
examples/task_evidence_suite.json`, then let `release-check` carry it
forward on every cut. Evidence that exposed a gap belongs in the suite
permanently; that is how a one-time investigation becomes a standing check.
