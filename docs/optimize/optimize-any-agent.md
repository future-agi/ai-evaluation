---
kind: agent-learning.docs-page.v1
track: optimize
objective: behavior
stage: optimize
backing:
  - examples/sdk_target_optimization.py
artifact_kinds:
  - agent-learning.optimization.v1
commands:
  - agent-learn optimize examples/optimization_manifest.json --output artifacts/optimization.json
postcondition: python -c "import json; p=json.load(open('artifacts/optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Optimize Any Agent

> **Twin:** [`examples/sdk_target_optimization.py`](../../examples/sdk_target_optimization.py)
> · emits `agent-learning.optimization.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

Optimization in this kit operates on the whole agent, never just the prompt. A
target declares a `base_config` (the agent, its simulation, its world
contract) plus explicit candidate values at named paths — `target_candidates`
keyed by dotted paths such as `simulation.environments.0.data.transitions`.
The optimizer swaps candidates in, simulates each resulting agent, scores it
against the evaluation config, and keeps only candidates whose score clears
the manifest threshold. The failure class this catches: an agent whose harness,
tooling, or world wiring silently degrades the task, in a way no prompt edit
can repair.

Every optimizer page in this track shares one design lineage, expressed as
four archetypes:

| Archetype | Dharma (what it may change) | Constraint (what it must preserve) |
| --- | --- | --- |
| Transformer (proposer) | candidate config: harness, memory, tooling | the task contract |
| Critic | scores + objections | evidence admissibility |
| Mediator | candidate retention/merge | lineage continuity |
| Steward (preserver) | rollback / veto | governance + the regression baseline |

The generic target optimizer is the smallest complete assembly of these
roles: the proposer enumerates `target_candidates`, the simulation evaluator
acts as critic, beam retention mediates between partial winners, and the
threshold plus `include_seed` comparison stewards the baseline.

## 2. Run it

CLI, against the manifest the README quickstart uses:

```bash
agent-learn optimize examples/optimization_manifest.json \
  --output artifacts/optimization.json
```

SDK, the same operation over explicit target paths (condensed from the twin):

```python
from fi.alk import optimize

result = optimize.optimize_target(
    name="sdk-target-optimization",
    base_config=base_config,             # agent + simulation + world contract
    evaluation_config=evaluation_config, # task, tools, success criteria
    target_candidates={
        "simulation.environments.0.data.transitions": [[], [approve_transition]],
    },
    layers=["world", "environment", "evaluator"],
)
```

Both paths run scripted agents in a local world contract; no provider key is
used and nothing leaves the machine.

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
```

The artifact records every candidate with its score, the selected best
candidate, a candidate-lineage block (which proposal produced which config),
and a governance block describing how acceptance was decided. It is a
replayable record, not a one-off score.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| manifest rejected | config fault | `summary.public_boundary_passed` + the manifest error line |
| best score below threshold | candidate fault | inspect `candidates[*].score` in the artifact |

## 5. Prove it / keep it

The optimization result is the entry point of a longer spine. Render it,
promote its findings into a regression manifest, and replay that manifest in
CI so the improvement cannot silently regress — the full journey is
[`optimization-lifecycle.md`](./optimization-lifecycle.md). When acceptance
itself needs audit and veto rules, continue to
[`governance.md`](./governance.md).
