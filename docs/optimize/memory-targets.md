---
kind: agent-learning.docs-page.v1
track: optimize
objective: capability
stage: optimize
backing:
  - examples/sdk_memory_target_optimization.py
  - examples/sdk_memory_layer_probe_optimization.py
artifact_kinds:
  - agent-learning.optimization.v1
commands:
  - python examples/sdk_memory_layer_probe_optimization.py artifacts/memory-layer-probe.json
  - AGENT_LEARNING_SDK_MEMORY_TARGET_OPTIMIZATION_KEY=local-dev-key python examples/sdk_memory_target_optimization.py artifacts/memory-target-optimization.json
postcondition: python -c "import json; p=json.load(open('artifacts/memory-layer-probe.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - api_key_configured
opt_in_lane: false
---

# Memory Targets: optimizing what the agent retains

> **Twin:** [`examples/sdk_memory_layer_probe_optimization.py`](../../examples/sdk_memory_layer_probe_optimization.py)
> · emits `agent-learning.optimization.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

Memory is agent config like any other: which operations run (read, write,
recall), what gets retained, and whether retained context actually reaches
the next turn. This page optimizes that layer two ways. The probe twin scores
weak and strong memory candidates directly against cases that require the
`read`, `write`, and `recall` operations — no full simulation, just the
memory layer under interrogation. The target example then embeds the same
candidates into a full run manifest and optimizes the explicit path
`simulation.environments.1.data.operations`, with the weak candidate being an
empty operations list — an agent that remembers nothing. The failure class:
memory that exists in the architecture diagram but never fires in the run.

| Archetype | Dharma (what it may change) | Constraint (what it must preserve) |
| --- | --- | --- |
| Transformer (proposer) | candidate config: harness, memory, tooling | the task contract |
| Critic | scores + objections | evidence admissibility |
| Mediator | candidate retention/merge | lineage continuity |
| Steward (preserver) | rollback / veto | governance + the regression baseline |

Memory sits explicitly in the proposer's dharma column — and in the society
role graph it has a dedicated specialist (`smriti`, path prefixes `memory`,
`retrieval`, `retriever`; see
[`society-of-agents.md`](./society-of-agents.md)). The constraint that
matters here is evidence admissibility: a memory candidate is accepted only
when the probe shows the required operations actually executed.

## 2. Run it

CLI — the probe needs no env at all; the target run takes a local
placeholder key (scripted agents, nothing leaves the machine):

```bash
python examples/sdk_memory_layer_probe_optimization.py \
  artifacts/memory-layer-probe.json

AGENT_LEARNING_SDK_MEMORY_TARGET_OPTIMIZATION_KEY=local-dev-key \
  python examples/sdk_memory_target_optimization.py \
  artifacts/memory-target-optimization.json
```

SDK, the probe operation:

```python
from fi.alk import optimize

result = optimize.optimize_memory_layer_probe(
    name="sdk-memory-layer-probe-optimization",
    memory_candidates=[weak_candidate, strong_candidate],
    cases=[{
        "id": "refund-memory",
        "input": "Recall the current refund policy memory.",
        "required_operations": ["read", "write", "recall"],
    }],
)
```

Both candidates come from
[`examples/sdk_memory_optimization.py`](../../examples/sdk_memory_optimization.py),
which defines the weak/strong memory lineages the probe distinguishes.

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/memory-layer-probe.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
```

The probe artifact carries a memory-layer proof block alongside candidate
lineage and governance; the target artifact records which operations list
won at the optimized path.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `Set AGENT_LEARNING_SDK_MEMORY_TARGET_OPTIMIZATION_KEY...` | missing placeholder env | `summary.api_key_configured` |
| required operation never observed | memory wiring fault | inspect the probe proof block per case |

## 5. Prove it / keep it

Promote the winning memory configuration into a regression manifest so a
refactor cannot quietly drop the `recall` path
([`optimization-lifecycle.md`](./optimization-lifecycle.md)). Memory under
simulation (rather than optimization) is `../simulate/memory.md`; memory as
a red-team persistence channel is `../redteam/stored-prompt-injection.md`.
