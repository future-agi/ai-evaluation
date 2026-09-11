---
kind: agent-learning.docs-page.v1
track: optimize
objective: behavior
stage: optimize
backing:
  - examples/sdk_multi_agent_target_optimization.py
artifact_kinds:
  - agent-learning.optimization.v1
commands:
  - AGENT_LEARNING_SDK_MULTI_AGENT_TARGET_OPTIMIZATION_KEY=local-dev-key python examples/sdk_multi_agent_target_optimization.py artifacts/multi-agent-target-optimization.json
postcondition: python -c "import json; p=json.load(open('artifacts/multi-agent-target-optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - api_key_configured
opt_in_lane: false
---

# Multi-Agent Targets: optimizing the room, not the agent

> **Twin:** [`examples/sdk_multi_agent_target_optimization.py`](../../examples/sdk_multi_agent_target_optimization.py)
> · emits `agent-learning.optimization.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

In a multi-agent system, the composition of the room is config: which roles
participate, how they hand off, what each may touch. This page optimizes
that composition as an explicit target path —
`simulation.environments.0.data.participants`. The twin builds two
candidates from
[`examples/sdk_multi_agent_optimization.py`](../../examples/sdk_multi_agent_optimization.py):
the weak one is the strong room with the critic participant removed; the
strong one restores it. The optimizer must detect, by score alone, that a
room without a critic completes the task worse. The failure class is quiet
role erosion — a reviewer or checker dropped during a refactor, with no test
that notices the room got more agreeable and less correct.

| Archetype | Dharma (what it may change) | Constraint (what it must preserve) |
| --- | --- | --- |
| Transformer (proposer) | candidate config: harness, memory, tooling | the task contract |
| Critic | scores + objections | evidence admissibility |
| Mediator | candidate retention/merge | lineage continuity |
| Steward (preserver) | rollback / veto | governance + the regression baseline |

This page is the archetype table made literal: the candidate under test IS
the presence of the critic. The same structural claim the sabha makes about
optimization — remove the objecting role and quality drops
([`society-of-agents.md`](./society-of-agents.md)) — is here measured on the
optimized system itself.

## 2. Run it

CLI — the env value is a local placeholder (scripted participants, nothing
leaves the machine):

```bash
AGENT_LEARNING_SDK_MULTI_AGENT_TARGET_OPTIMIZATION_KEY=local-dev-key \
  python examples/sdk_multi_agent_target_optimization.py \
  artifacts/multi-agent-target-optimization.json
```

SDK, the same operation in the explicit-target form:

```python
from fi.alk import optimize

result = optimize.optimize_target(
    name="sdk-multi-agent-target-optimization",
    base_config=base_config,  # room with the critic removed
    target_candidates={
        "simulation.environments.0.data.participants": [
            missing_critic_participants,
            full_participants,
        ],
    },
)
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/multi-agent-target-optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
```

The artifact contains both room candidates with scores, a multi-agent
coordination proof block, and lineage showing the full-participants candidate
as the survivor.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `Set AGENT_LEARNING_SDK_MULTI_AGENT_TARGET_OPTIMIZATION_KEY...` | missing placeholder env | `agent-learn doctor` → `summary.api_key_configured` |
| `vendored import failed` | infra | `summary.missing_engine_modules` |
| both rooms score the same | evaluator fault | check the coordination checks in the evaluation config |

## 5. Prove it / keep it

Promote the winning room into a regression manifest so the critic cannot be
dropped again without a failing replay
([`optimization-lifecycle.md`](./optimization-lifecycle.md)). Behavioral
diversity and collaboration quality inside the room — rather than its
composition — are optimized in
[`behavior-and-collaboration.md`](./behavior-and-collaboration.md);
multi-agent rooms under plain simulation are `../simulate/multi-agent.md`.
