---
kind: agent-learning.docs-page.v1
track: simulate
objective: reliability
stage: simulate
backing:
  - examples/sdk_framework_adapter_memory_trace.py
  - examples/sdk_memory_target_optimization.py
artifact_kinds:
  - agent-learning.run.v1
  - agent-learning.optimization.v1
commands:
  - python examples/sdk_framework_adapter_memory_trace.py artifacts/memory-trace.json
  - AGENT_LEARNING_SDK_MEMORY_TARGET_OPTIMIZATION_KEY=offline-demo-key python examples/sdk_memory_target_optimization.py artifacts/memory-target-optimization.json
postcondition: python -c "import json; p=json.load(open('artifacts/memory-trace.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Memory: simulate the layer that persists between sessions

> **Twin:** [`examples/sdk_framework_adapter_memory_trace.py`](../../examples/sdk_framework_adapter_memory_trace.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

Memory is the part of an agent that outlives the conversation, which makes
it the part where failures compound silently: stale policy recalled as
current, writes without provenance, one tenant's namespace bleeding into
another's, poisoned entries that survive retention. None of that shows up in
a single transcript — it shows up in the memory *trace*.

The twin example simulates a LangGraph/Mem0-style memory adapter
(`LocalFrameworkMemoryGraph`) whose export is the full governance surface
the kit's memory environments check: `memory_operations` (write/read with
key, namespace, `trace_id`, `thread_id`, `source_ids`, and a
`policy_decision`), `checkpoints` (saved state keys per thread),
`memory_records` with source lineage, `memory_searches` with
`freshness_checked` retrievals, plus explicit `poison_tests`,
`isolation_tests`, and `retention_tests`. Its weak path returns an answer
"without checkpoint or memory lineage evidence"; the strong path approves a
refund "with current policy recall and governed memory lineage". The
simulation must score the difference on lineage evidence, not on the prose.

The second backing example moves from observing memory to selecting it:
`sdk_memory_target_optimization.py` points the target optimizer at one path
inside the manifest — `simulation.environments.1.data.operations` — and
searches candidate memory-operation sets (an empty, lineage-free set versus
a governed one) against a memory-layer run manifest with a 0.98 threshold.

## 2. Run it

CLI:

```bash
python examples/sdk_framework_adapter_memory_trace.py artifacts/memory-trace.json

AGENT_LEARNING_SDK_MEMORY_TARGET_OPTIMIZATION_KEY=offline-demo-key \
  python examples/sdk_memory_target_optimization.py artifacts/memory-target-optimization.json
```

SDK (the same operations the examples perform):

```python
import asyncio
from fi.alk import optimize, simulate

manifest = optimize.build_framework_run_manifest_from_local_adapter(
    target="examples/sdk_framework_adapter_memory_trace.py:LocalFrameworkMemoryGraph",
)
simulate.write_manifest_file(manifest, "memory-trace.manifest.json")
result = asyncio.run(simulate.run_manifest_file("memory-trace.manifest.json"))
```

The first command needs no env at all; the second's placeholder key is CI
wiring metadata for a local deterministic engine.

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/memory-trace.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
```

The run artifact carries the adapter's memory trace as environment evidence:
every operation with its namespace, thread, policy decision, and source ids;
every checkpoint; the poison/isolation/retention test outcomes. The
optimization artifact (second command) records which operation set won and
the lineage metrics that decided it.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| adapter target rejected | config fault | `summary.public_boundary_passed` + the manifest error line |
| memory case scores low | lineage evidence missing (no checkpoints, no `source_ids`, stale retrievals) | read the memory operations in the artifact's case record |

## 5. Prove it / keep it

Both backing examples are re-executed on every `agent-learn release-check`
(`stateful_framework_adapter_readiness` and
`memory_target_optimizer_readiness` gates). For your own agent: export its
memory layer through an adapter with this trace shape, run it here, then
baseline the passing artifact and follow
[`regression-lifecycle.md`](regression-lifecycle.md). Memory is also the
channel for cross-session injection — when you are ready to attack it, the
red-team track's stored-prompt-injection page starts from the same
persisted-state surface.
