---
kind: agent-learning.docs-page.v1
track: optimize
objective: capability
stage: optimize
backing:
  - examples/sdk_workflow_target_profile_matrix.py
  - examples/sdk_workflow_target_optimization.py
artifact_kinds:
  - agent-learning.optimization.v1
commands:
  - AGENT_LEARNING_SDK_WORKFLOW_TARGET_OPTIMIZATION_KEY=local-dev-key python examples/sdk_workflow_target_optimization.py artifacts/workflow-target-optimization.json
  - AGENT_LEARNING_SDK_WORKFLOW_TARGET_PROFILE_MATRIX_KEY=local-dev-key python examples/sdk_workflow_target_profile_matrix.py artifacts/workflow-profile-matrix.json
postcondition: python -c "import json; p=json.load(open('artifacts/workflow-target-optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - api_key_configured
opt_in_lane: false
---

# Workflow Profile Matrix: one workflow target, six frameworks

> **Twin:** [`examples/sdk_workflow_target_profile_matrix.py`](../../examples/sdk_workflow_target_profile_matrix.py)
> · emits `agent-learning.optimization.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

A workflow trace is the skeleton of an agent run: nodes, edges, steps,
checkpoints, route decisions. The workflow target optimizer treats that trace
as the optimization surface (`simulation.environments.0.data.trace`): the
weak candidate is a trace with a single `intake` node, no edges, and no
checkpoints — a workflow that technically ran but recorded nothing usable.
Optimization must select the trace that satisfies the required metrics
(`workflow_trace_coverage`, `workflow_graph_quality`,
`tool_selection_accuracy`, `artifact_coverage`, `task_completion`) and the
required structural counts (4 nodes, 3 edges, 4 steps, 2 checkpoints, 1
route decision).

The matrix twin then repeats that exact optimization across six framework
profiles — langgraph, crewai, llamaindex, langchain, pipecat, livekit — each
with its native export type (`langgraph_checkpoint_graph`,
`crewai_flow_route_state`, `llamaindex_workflow_events`, and so on). The
failure class is framework-shaped blindness: a workflow target that holds for
the framework you developed against and degrades for the one you deploy on.

| Archetype | Dharma (what it may change) | Constraint (what it must preserve) |
| --- | --- | --- |
| Transformer (proposer) | candidate config: harness, memory, tooling | the task contract |
| Critic | scores + objections | evidence admissibility |
| Mediator | candidate retention/merge | lineage continuity |
| Steward (preserver) | rollback / veto | governance + the regression baseline |

The matrix is the mediator's row at corpus scale: per-profile optimizations
propose and score independently, and the matrix merges them under one
verdict while keeping each profile's lineage separate and inspectable.

## 2. Run it

CLI — env values are local placeholders (scripted runs, nothing leaves the
machine):

```bash
AGENT_LEARNING_SDK_WORKFLOW_TARGET_OPTIMIZATION_KEY=local-dev-key \
  python examples/sdk_workflow_target_optimization.py \
  artifacts/workflow-target-optimization.json

AGENT_LEARNING_SDK_WORKFLOW_TARGET_PROFILE_MATRIX_KEY=local-dev-key \
  python examples/sdk_workflow_target_profile_matrix.py \
  artifacts/workflow-profile-matrix.json
```

SDK, the single-profile operation both examples build on:

```python
from fi.alk import optimize

result = optimize.optimize_target(
    name="sdk-workflow-target-optimization",
    base_config=base_config,  # run manifest with the weak workflow trace
    target_candidates={
        "simulation.environments.0.data.trace": [weak_trace, strong_trace],
    },
)
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/workflow-target-optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
```

The single-profile artifact is a standard optimization payload. The matrix
artifact aggregates one such optimization per framework profile: its
`profiles` array holds the per-framework summaries and its summary block
reports `passed_profile_count` and `failed_profiles` — an empty
`failed_profiles` list is the matrix verdict you want.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `Set AGENT_LEARNING_SDK_WORKFLOW_TARGET_PROFILE_MATRIX_KEY...` | missing placeholder env | `agent-learn doctor` → `summary.api_key_configured` |
| `vendored import failed` | infra | `summary.missing_engine_modules` |
| one profile in `failed_profiles` | framework-specific trace fault | open that profile's summary; compare its export type counts |

## 5. Prove it / keep it

Promote the single-profile result into a regression manifest and re-run the
matrix when adding a framework — the matrix is exactly the artifact to attach
when claiming cross-framework workflow support. The per-framework cookbook
narratives live in the frameworks track (e.g. `../frameworks/langgraph.md`);
the promotion spine is [`optimization-lifecycle.md`](./optimization-lifecycle.md).
