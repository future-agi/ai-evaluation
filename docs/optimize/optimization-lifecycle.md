---
kind: agent-learning.docs-page.v1
track: optimize
objective: reliability
stage: optimize
backing:
  - examples/sdk_task_world_optimization.py
artifact_kinds:
  - agent-learning.optimization.v1
  - agent-learning.regression-promotion.v1
  - agent-learning.replay.v1
commands:
  - agent-learn optimize examples/optimization_manifest.json --output artifacts/lifecycle-optimization.json
  - agent-learn report artifacts/lifecycle-optimization.json --markdown artifacts/lifecycle-report.md
  - agent-learn promote-to-regression artifacts/lifecycle-optimization.json --min-level note --output artifacts/lifecycle-promotion.json --manifest artifacts/regression-manifest.json
postcondition: python -c "import json; p=json.load(open('artifacts/lifecycle-optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# The Optimization Lifecycle

> **Twin:** [`examples/sdk_task_world_optimization.py`](../../examples/sdk_task_world_optimization.py)
> · emits `agent-learning.optimization.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

A single optimization run answers "which candidate is best today". The
lifecycle answers the harder question: "does the improvement survive
tomorrow". The kit's lifecycle is optimize → report → promote → replay: the
optimization result is rendered into a human report, its findings are
promoted into a runnable regression manifest, and that manifest is replayed
on every future change. The failure class is regression by drift — an agent
that was fixed once and quietly un-fixed by a later edit, because the fix was
never converted into a repeatable check.

| Archetype | Dharma (what it may change) | Constraint (what it must preserve) |
| --- | --- | --- |
| Transformer (proposer) | candidate config: harness, memory, tooling | the task contract |
| Critic | scores + objections | evidence admissibility |
| Mediator | candidate retention/merge | lineage continuity |
| Steward (preserver) | rollback / veto | governance + the regression baseline |

The lifecycle is the steward's home ground: promotion and replay are how the
regression baseline in the constraint column actually gets built and
preserved. The twin example constructs the refund world contract whose weak
agent (inspects the refund, never applies the transition) and strong agent
the lifecycle distinguishes — [`examples/sdk_optimization_lifecycle.py`](../../examples/sdk_optimization_lifecycle.py)
builds its workspace manifest directly from that module.

## 2. Run it

CLI — the three lifecycle steps as separate commands:

```bash
agent-learn optimize examples/optimization_manifest.json \
  --output artifacts/lifecycle-optimization.json

agent-learn report artifacts/lifecycle-optimization.json \
  --markdown artifacts/lifecycle-report.md

agent-learn promote-to-regression artifacts/lifecycle-optimization.json \
  --min-level note \
  --output artifacts/lifecycle-promotion.json \
  --manifest artifacts/regression-manifest.json
```

SDK — the whole journey as one call, exactly as the lifecycle example does:

```python
from fi.alk import suite

result = suite.run_optimization_lifecycle_file(
    "manifests/optimize.json",
    workspace_dir="workspace",
    name="sdk-optimization-lifecycle",
)
```

`run_optimization_lifecycle_file` runs optimize, renders both reports,
promotes findings, writes the regression manifest, and replays it — emitting
the JSON, JUnit, SARIF, and Markdown bundles for each step.

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/lifecycle-optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
```

After the third command you also hold `artifacts/regression-manifest.json` —
a runnable manifest derived from the optimization findings. That file, not
the score, is the durable output of the lifecycle.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| manifest rejected | config fault | `summary.public_boundary_passed` + the manifest error line |
| promotion writes no manifest | no findings at `--min-level` | rerun with `--min-level note` |

## 5. Prove it / keep it

Replay the promoted manifest on every change:

```bash
agent-learn replay artifacts/regression-manifest.json \
  --output artifacts/replay.json
```

Wire that replay into CI next to `agent-learn release-check` — see the prove
track's release-check page. The same promote-and-replay spine, applied to
red-team findings, is `../redteam/promote-to-regression.md`.
