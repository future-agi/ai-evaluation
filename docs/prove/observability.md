---
kind: agent-learning.docs-page.v1
track: prove
objective: reliability
stage: prove
backing:
  - examples/sdk_framework_adapter_trace_export.py
artifact_kinds:
  - agent-learning.optimization.v1
  - agent-learning.report.v1
  - agent-learning.run.v1
commands:
  - AGENT_LEARNING_WORKSPACE_OBSERVABILITY_OPT_EXAMPLE_KEY=local-offline agent-learn optimize examples/workspace_observability_optimization.json --output artifacts/workspace-observability.json
  - agent-learn report artifacts/workspace-observability.json --output workspace-observability.report.json
  - python examples/sdk_framework_adapter_trace_export.py artifacts/trace-export.json
postcondition: python -c "import json; p=json.load(open('artifacts/workspace-observability.json')); r=json.load(open('artifacts/workspace-observability.report.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; assert r['kind']=='agent-learning.report.v1', r['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Observability: evidence that makes artifacts replayable

> **Twin:** [`examples/sdk_framework_adapter_trace_export.py`](../../examples/sdk_framework_adapter_trace_export.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

An artifact you cannot replay is a screenshot, not evidence. What makes a kit
artifact auditable is the observability layer underneath it: which repository
and commit the agent came from, which commands ran, which spans the framework
emitted, which rows failed and can be re-executed. The failure class this page
targets is rollout evidence that looks complete but is not — a workspace run
that reports green while its command logs, red-team evidence, or replayable
failure rows are missing.

`examples/workspace_observability_optimization.json` scores exactly that. Its
scenario checks out an agent repository (provenance pinned by
`repository_url` and `commit_sha`), runs simulations and evals, replays
failed observability rows, red-teams the agent, and the optimizer scores
whether each evidence surface — repository provenance, command logs,
artifacts, red-team evidence, replay failures, credentials, security gates —
is actually visible in the result. The backing twin works one level lower:
a local framework adapter emits an OTLP-shaped trace export (resource spans,
attributes, span ids), and the kit normalizes those spans into the run
artifact's framework trace evidence — the raw material every replay and
report is built from.

## 2. Run it

Score the workspace's observability surfaces, then render the artifact as a
report:

```bash
AGENT_LEARNING_WORKSPACE_OBSERVABILITY_OPT_EXAMPLE_KEY=local-offline \
agent-learn optimize examples/workspace_observability_optimization.json \
  --output artifacts/workspace-observability.json

agent-learn report artifacts/workspace-observability.json \
  --output workspace-observability.report.json

python examples/sdk_framework_adapter_trace_export.py artifacts/trace-export.json
```

The optimize `--output` resolves against your current working directory, and
`report` writes its relative `--output` next to the input artifact, so all
three artifacts land under `artifacts/`.

The same operations from the SDK:

```python
import os

from fi.alk import optimize

os.environ.setdefault(
    "AGENT_LEARNING_WORKSPACE_OBSERVABILITY_OPT_EXAMPLE_KEY", "local-offline"
)
result = optimize.optimize_manifest_file(
    "examples/workspace_observability_optimization.json"
)
```

## 3. What you built

Postcondition (machine-checkable — same shape the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/workspace-observability.json')); r=json.load(open('artifacts/workspace-observability.report.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; assert r['kind']=='agent-learning.report.v1', r['kind']; print('ok')"
```

The optimization artifact records the workspace run's evidence surfaces and
the optimizer's scoring of each gap; the report artifact
(`agent-learning.report.v1`) is the same evidence rendered with sections and
cards — the `agent-learn report` command works on any saved kit artifact.
The trace-export run artifact carries the normalized framework spans in its
trace state, which is what `agent-learn replay` consumes when it re-executes
recorded sessions.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `missing required environment variable(s)` | config fault | export the manifest's `required_env` key with any placeholder value |
| manifest rejected | manifest fault | `summary.public_boundary_passed` + the manifest error line |
| report has empty sections | source artifact lacks evidence | inspect the optimization artifact's evidence blocks before re-rendering |

## 5. Prove it / keep it

Treat observability evidence as a gated surface, not a nice-to-have: keep
this optimization in your lane so a workspace run that loses its command logs
or replay rows fails the lane by name. The artifacts it produces feed the
rest of the prove track — embed them in a suite run
([trinity-suite](trinity-suite.md)), list their follow-up operations with
[actions](actions.md), and let `agent-learn replay` re-execute the recorded
sessions whenever you need the evidence re-earned rather than re-read.
