---
kind: agent-learning.docs-page.v1
track: optimize
objective: capability
stage: optimize
backing:
  - examples/sdk_world_hooks_optimization.py
artifact_kinds:
  - agent-learning.optimization.v1
commands:
  - agent-learn optimize examples/world_model_optimization.json --output artifacts/world-model-optimization.json
postcondition: python -c "import json; p=json.load(open('artifacts/world-model-optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# World-Model Optimization

> **Twin:** [`examples/sdk_world_hooks_optimization.py`](../../examples/sdk_world_hooks_optimization.py)
> · emits `agent-learning.optimization.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

An agent's competence is bounded by the world it is tested in. A toy world —
no state, no preconditions, no hostile inputs — passes agents that fail in
production. World-model optimization turns the world itself into the
optimization surface: the committed manifest
([`examples/world_model_optimization.json`](../../examples/world_model_optimization.json))
runs a stateful tool world whose attack surfaces include an injected tool
return (`indirect_prompt_injection`) and a persistent note
(`stored_prompt_injection`), and searches candidates under
`simulation.environments` for the configuration in which the agent
authenticates, applies policy-safe state deltas, and quarantines the injected
tool output. The threshold is 0.95 — stricter than the track default,
because a world model that mostly holds is a world model that leaks.

| Archetype | Dharma (what it may change) | Constraint (what it must preserve) |
| --- | --- | --- |
| Transformer (proposer) | candidate config: harness, memory, tooling | the task contract |
| Critic | scores + objections | evidence admissibility |
| Mediator | candidate retention/merge | lineage continuity |
| Steward (preserver) | rollback / veto | governance + the regression baseline |

Here the proposer's dharma extends to the world contract itself —
transitions, invariants, attack surfaces — while the task contract in the
constraint column is exactly what keeps a "better" world from being a world
that merely flatters the agent. The twin example grounds the same machinery
in world hooks, citing the research it implements (offline agent evaluation
with world-model rollouts, verified stateful execution environments) in its
target metadata.

## 2. Run it

CLI:

```bash
agent-learn optimize examples/world_model_optimization.json \
  --output artifacts/world-model-optimization.json
```

SDK, the same operation as the world-model example
([`examples/sdk_world_model_optimization.py`](../../examples/sdk_world_model_optimization.py))
runs it:

```python
from fi.alk import optimize

result = optimize.optimize_world_model(
    name="sdk-world-model-optimization",
    target_metadata={"cookbook": "sdk-world-model-optimization"},
)
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/world-model-optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
```

The artifact records each world candidate with its score under the manifest's
scoring block, plus lineage and governance. The winning candidate is a world
definition you can reuse as the simulation environment for every other page
in this track.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| manifest rejected | config fault | `summary.public_boundary_passed` + the manifest error line |
| no candidate reaches 0.95 | world fault | inspect which invariant or quarantine step scored low |

## 5. Prove it / keep it

A hardened world is regression material: promote the result
(`agent-learn promote-to-regression`) so the injected-tool-return and
stored-note channels stay closed — the cross-session attack class itself is
taught in `../redteam/stored-prompt-injection.md`. The simulate-track
companion for world state and hooks is `../simulate/worlds-and-hooks.md`.
