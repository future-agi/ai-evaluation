---
kind: agent-learning.docs-page.v1
track: simulate
objective: reliability
stage: simulate
backing:
  - examples/sdk_world_hooks_optimization.py
  - examples/sdk_task_world_optimization.py
artifact_kinds:
  - agent-learning.optimization.v1
commands:
  - AGENT_LEARNING_SDK_WORLD_HOOKS_KEY=offline-demo-key python examples/sdk_world_hooks_optimization.py artifacts/world-hooks.json
  - AGENT_LEARNING_SDK_TASK_WORLD_EXAMPLE_KEY=offline-demo-key python examples/sdk_task_world_optimization.py artifacts/task-world.json
postcondition: python -c "import json; p=json.load(open('artifacts/world-hooks.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - api_key_configured
opt_in_lane: false
---

# Worlds and Hooks: simulate against executable state

> **Twin:** [`examples/sdk_world_hooks_optimization.py`](../../examples/sdk_world_hooks_optimization.py)
> · emits `agent-learning.optimization.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

A transcript tells you what an agent said. A world tells you what the agent
actually changed. In the kit, a world is an executable state machine declared
in the manifest: an initial state (`refund.status: pending`), transitions
with preconditions and effects (`approve_refund` requires
`refund.status: pending` and sets it to `approved`), invariants that must
hold throughout (`policy.can_refund: true`), and success conditions that
define the terminal state. The agent does not merely talk about approving a
refund — it must call `apply_world_transition` and drive the world to
`status: success`.

The failure class is plausible-but-inert agents: responses that read
correctly while no required transition ever fires. The
`world_contract_quality` block in `examples/sdk_task_world_optimization.py`
encodes that distinction as checkable facts — required transitions,
`min_completed_transitions`, zero invariant violations, an `expected_state`
the world must end in. The example's weak candidate "inspected the refund
request but did not complete the world transition"; the strong candidate
applies the transition. The optimizer must tell them apart on world
evidence, not prose.

Hooks are the second half: in-process lifecycle interception points around
world execution, so contracts, adversarial pressure, memory provenance, and
replay evidence are captured natively rather than through an out-of-process
adapter. `optimize.optimize_world_hooks` searches complete world-candidate
configurations and emits the selection as one artifact.

## 2. Run it

CLI:

```bash
AGENT_LEARNING_SDK_WORLD_HOOKS_KEY=offline-demo-key \
  python examples/sdk_world_hooks_optimization.py artifacts/world-hooks.json

AGENT_LEARNING_SDK_TASK_WORLD_EXAMPLE_KEY=offline-demo-key \
  python examples/sdk_task_world_optimization.py artifacts/task-world.json
```

SDK (same operations the examples perform):

```python
from fi.alk import optimize

result = optimize.optimize_world_hooks(
    name="world-hooks-optimization",
    required_env=["AGENT_LEARNING_SDK_WORLD_HOOKS_KEY"],
)
# Task-world variant: build_task_optimization_manifest -> optimize_manifest
```

The placeholder env values are CI wiring metadata; both examples run on
local deterministic engines.

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/world-hooks.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
```

The artifact records the candidate search: which world/agent configuration
won, the world-contract metrics that decided it
(`world_contract_quality`, `world_contract_coverage`,
`tool_selection_accuracy`, `task_completion` with explicit weights), and the
transition/invariant evidence behind each score.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `Set AGENT_LEARNING_SDK_..._KEY before running` | env not exported | `summary.api_key_configured` is about the kit key; the example key is set inline as shown in §2 |
| world ends non-terminal / transitions missing | agent never called `apply_world_transition` | inspect the world contract evidence in the artifact's case records |

## 5. Prove it / keep it

Both backing examples are re-executed on every `agent-learn release-check`
by the `world_hooks_readiness` and `task_world_optimizer_readiness` gates,
so the world surface this page teaches cannot silently rot. To keep your own
world-backed run honest over time, freeze its artifact with
`agent-learn simulate baseline` and follow
[`regression-lifecycle.md`](regression-lifecycle.md); to put adversarial
pressure on a world, the red-team track's autonomous task-world page picks
up exactly this manifest shape.
