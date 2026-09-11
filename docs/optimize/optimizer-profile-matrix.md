---
kind: agent-learning.docs-page.v1
track: optimize
objective: capability
stage: optimize
backing:
  - examples/sdk_optimizer_profile_matrix.py
artifact_kinds:
  - agent-learning.optimization.v1
  - agent-learning.apply-plan.v1
commands:
  - AGENT_LEARNING_SDK_OPTIMIZER_PROFILE_MATRIX_KEY=local-dev-key python examples/sdk_optimizer_profile_matrix.py artifacts/optimizer-profile-matrix.json
postcondition: python -c "import json; p=json.load(open('artifacts/optimizer-profile-matrix.json')); assert p['kind']=='agent-learning.optimizer-profile-matrix.v1', p['kind']; assert p['summary']['cell_count']==33, p['summary']; assert p['apply_plans'] and all(plan['kind']=='agent-learning.apply-plan.v1' for plan in p['apply_plans']); print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - api_key_configured
opt_in_lane: false
---

# Optimizer Profile Matrix: 33 declared cells, per-cell winners only

> **Twin:** [`examples/sdk_optimizer_profile_matrix.py`](../../examples/sdk_optimizer_profile_matrix.py)
> · emits `agent-learning.optimization.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

"Which optimizer backend should run for this target?" is usually answered by
a static default — folklore, not evidence. The optimizer profile matrix
replaces the folklore with a declared 3-axis evidence corpus: framework
profile (langgraph, crewai, llamaindex, langchain, pipecat, livekit) ×
target kind (`prompt`, `whole_agent`, `memory_ops`, `multi_agent_roster`,
`workflow_trace`, `orchestration_spans`, `framework_method`) × backend token
(`gepa`, `tpe`, `evolution_elo`, `bandit`, `society`, `regression_replay`).
The launch subset is exactly 33 declared coordinates — not a cartesian
product — and the release gate asserts exactly that set, so growing coverage
is a visible constant-plus-example diff.

Each cell runs a real optimization under a declared setting and a declared
evaluation budget (at most 24 evaluations per cell), records its winner, its
selected patch paths, and its trajectory fitness profile. Winners are
per-cell only: orderings invert across settings, so the payload schema has
no global best-backend key and the gate fails the release if one appears.

The `whole_agent` cells exercise the staged whole-agent contract — stage
`component_text` (instructions, first message, per-node prompts), then
`structural_config` (model, voice, tools, memory policy, topology), then
`global_repolish` — with samiti generation and sabha deliberation seated in
every stage. Each whole-agent cell emits an `agent-learning.apply-plan.v1`
artifact: ordered field-level ops, read-back checks, an abort mismatch
policy, and the frozen-profile and nirnaya references. The kit never
applies; the platform executes the plan and re-fetches the provider agent to
evaluate every read-back check.

| Archetype | Dharma (what it may change) | Constraint (what it must preserve) |
| --- | --- | --- |
| Transformer (proposer) | candidate config: harness, memory, tooling | the task contract |
| Critic | scores + objections | evidence admissibility |
| Mediator | candidate retention/merge | lineage continuity |
| Steward (preserver) | rollback / veto | governance + the regression baseline |

## 2. Run it

CLI — the env value is a local placeholder (scripted runs, nothing leaves
the machine):

```bash
AGENT_LEARNING_SDK_OPTIMIZER_PROFILE_MATRIX_KEY=local-dev-key \
  python examples/sdk_optimizer_profile_matrix.py \
  artifacts/optimizer-profile-matrix.json
```

SDK, the same operation:

```python
from fi.alk import optimize

manifests = optimize.build_optimizer_profile_matrix_manifests()
payload = optimize.run_optimizer_profile_matrix(
    manifests,
    output_path="artifacts/optimizer-profile-matrix.json",
)
```

A single whole-agent contract outside the matrix:

```python
manifest = optimize.build_whole_agent_optimization_manifest(
    name="my-whole-agent",
    base_agent={"provider": "livekit", "model": "base", "voice": "base",
                "first_message": "Hello.", "instructions": "Answer briefly.",
                "responses": [{"content": "weak"}], "type": "scripted"},
    search_space={"model": ["base", "tuned"], "voice": ["base", "warm"]},
    evaluation_config={"task_description": "t", "expected_result": "strong"},
    eval_budget=12,
)
result = optimize.optimize_manifest(manifest)
plan = result["apply_plan"]  # agent-learning.apply-plan.v1
```

## 3. What you built

```bash
python -c "import json; p=json.load(open('artifacts/optimizer-profile-matrix.json')); assert p['kind']=='agent-learning.optimizer-profile-matrix.v1', p['kind']; assert p['summary']['cell_count']==33, p['summary']; assert p['apply_plans'] and all(plan['kind']=='agent-learning.apply-plan.v1' for plan in p['apply_plans']); print('ok')"
```

The artifact carries `cells[]` (one record per declared coordinate: setting,
declared budget, budget actuals, winner, selected patch paths, trajectory
profile), `summary.per_axis_coverage`, the regenerated `routing_table`
(byte-compared against `examples/optimizer_routing_table.json` by the
release gate), and `apply_plans[]` for every whole-agent cell.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `Set AGENT_LEARNING_SDK_OPTIMIZER_PROFILE_MATRIX_KEY...` | missing local placeholder env | `api_key_configured` |
| `ModuleNotFoundError: fi.opt` | optimizer engine not installed | `missing_engine_modules` |
| `failed_cells` non-empty in summary | a declared cell no longer closes its native proof | `missing_engine_modules` |
| `budget_exceeded` on a cell | actual evaluations exceeded the declared budget | `missing_engine_modules` |

## 5. Prove it / keep it

`agent-learn release-check --project-root .` executes every declared cell in
the `optimizer_profile_matrix_readiness` gate and byte-compares the
regenerated routing table against the committed copy. Backend routing built
on these cells is the next page: [Backend Routing](backend-routing.md).
Freeze the capabilities a winner must not regress with
[Capability-Profile Freezing](capability-profile-freezing.md).
