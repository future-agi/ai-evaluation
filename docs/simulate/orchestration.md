---
kind: agent-learning.docs-page.v1
track: simulate
objective: reliability
stage: simulate
backing:
  - examples/sdk_framework_adapter_orchestration_trace.py
  - examples/sdk_orchestration_stack_probe_optimization.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - python examples/sdk_framework_adapter_orchestration_trace.py artifacts/orchestration-trace.json
  - python examples/sdk_orchestration_stack_probe_optimization.py artifacts/orchestration-stack-probe.json
postcondition: python -c "import json; p=json.load(open('artifacts/orchestration-trace.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Orchestration: simulate the graph, not just the agents

> **Twin:** [`examples/sdk_framework_adapter_orchestration_trace.py`](../../examples/sdk_framework_adapter_orchestration_trace.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

In a supervisor-style system, most production incidents are not bad answers —
they are bad routing: a delegation that never fired, a retry loop that ate
the budget, a critic that was bypassed on the path to the final answer. To
test that, the simulation has to see the orchestration graph itself.

The twin example exports exactly that from a LangGraph-style supervisor
(`LocalLangGraphOrchestrationAgent`): `orchestration_nodes` (supervisor,
policy agent, critic, finalizer — each typed and signal-tagged),
`orchestration_edges` (delegate / handoff / route, each with the condition
that licenses it, e.g. `policy_review_required`, `critic_approved`), and
`orchestration_steps` — the actual execution log with `route_from`/`route_to`,
`attempt` counts, `recoverable`/`recovered` flags, per-step `latency_ms` and
cost, and the state carried across the hop. The scripted strong path
includes a policy retry that *recovers*; the weak path returns a plain
answer "without multi-agent orchestration evidence". The evaluator scores
graph facts: did delegation happen, did the retry recover, did the critic
vote precede the stop.

The second backing example treats orchestration as a search problem:
`optimize.optimize_orchestration_stack_probe` compares weak/strong stack and
agent candidates, then promotes the winner into a run manifest and executes
it — the probe-then-promote pattern used across the kit.

## 2. Run it

CLI (no env required — both engines are local and deterministic):

```bash
python examples/sdk_framework_adapter_orchestration_trace.py artifacts/orchestration-trace.json

python examples/sdk_orchestration_stack_probe_optimization.py artifacts/orchestration-stack-probe.json
```

SDK (the operation the twin performs):

```python
import asyncio
from fi.alk import optimize, simulate

manifest = optimize.build_framework_run_manifest_from_local_adapter(
    target="examples/sdk_framework_adapter_orchestration_trace.py:LocalLangGraphOrchestrationAgent",
)
simulate.write_manifest_file(manifest, "orchestration.manifest.json")
result = asyncio.run(simulate.run_manifest_file("orchestration.manifest.json"))
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/orchestration-trace.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
```

The artifact's case evidence contains the full graph export: every node,
every conditioned edge, and the step log with attempts, recoveries,
latencies, and costs — enough to answer "which path did the request take
and what did each hop cost" from the artifact alone.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| adapter target / manifest rejected | config fault | `summary.public_boundary_passed` + the manifest error line |
| run passes but orchestration score is low | steps missing routing/recovery evidence | inspect `orchestration_steps` in the artifact's case record |

## 5. Prove it / keep it

Both backing examples are re-proven on every `agent-learn release-check`
(`stateful_framework_adapter_readiness` and
`orchestration_stack_probe_readiness` gates). When your own supervisor's
trace passes here, freeze the artifact as a baseline and follow
[`regression-lifecycle.md`](regression-lifecycle.md) — routing regressions
(a dropped edge condition, a retry that stops recovering) then surface as
compare findings instead of production incidents. For the team-coordination
view of the same systems, see [`multi-agent.md`](multi-agent.md).
