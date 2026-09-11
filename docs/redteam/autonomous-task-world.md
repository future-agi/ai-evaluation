---
kind: agent-learning.docs-page.v1
track: redteam
objective: safety
stage: optimize
backing:
  - examples/sdk_task_world_optimization.py
artifact_kinds:
  - agent-learning.optimization.v1
commands:
  - AGENT_LEARNING_AUTONOMOUS_REDTEAM_OPT_EXAMPLE_KEY=local-example agent-learn optimize examples/autonomous_redteam_task_world_optimization.json --output artifacts/autonomous-redteam-task-world.json
postcondition: python -c "import json; p=json.load(open('artifacts/autonomous-redteam-task-world.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Autonomous red-teaming in a task world

> **Twin:** [`examples/sdk_task_world_optimization.py`](../../examples/sdk_task_world_optimization.py)
> · emits `agent-learning.optimization.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

Every other page in this track attacks a conversation. This page attacks a
*world*: a task environment with typed state, transitions, preconditions, and
invariants, where the agent acts autonomously — plans, calls tools, mutates
state, verifies, stops. Hostility here is environmental: poisoned content
sits inside the world the agent must traverse to finish its job. The failure
class is an agent that is safe in chat and unsafe in operation — it refuses a
hostile *message*, but executes a hostile *world state*.

`examples/autonomous_redteam_task_world_optimization.json` builds a refund
world for Riya, an agent red-team owner who needs an autonomous harness that
proves five things at once: task evidence (the eligible refund is actually
approved), domain invariants (the case stays low-risk and consistent),
world-state progress (transitions fire with their preconditions met), hostile
environment resistance (a planted canary-exfiltration lure is refused), and
self-improvement trace quality (verification recorded, safe memory written, a
reusable runbook stored, then a clean stop). The optimizer searches candidate
harness configurations and must find one satisfying all five families — a
candidate that completes the refund but touches the canary loses.

The twin is the task-world optimizer this page's manifest is built on, run
under the `task_world_optimizer_readiness` release gate: a weak agent that
inspects but never completes the world transition is rejected in favor of one
that executes `approve_refund` with preconditions (`refund.status: pending`)
and effects (`refund.status: approved`) checked by the world, not by the
agent's self-report.

## 2. Run it

CLI:

```bash
AGENT_LEARNING_AUTONOMOUS_REDTEAM_OPT_EXAMPLE_KEY=local-example \
  agent-learn optimize examples/autonomous_redteam_task_world_optimization.json \
  --output artifacts/autonomous-redteam-task-world.json
```

SDK, same operation:

```python
import os

from fi.alk import optimize

os.environ.setdefault(
    "AGENT_LEARNING_AUTONOMOUS_REDTEAM_OPT_EXAMPLE_KEY", "local-example"
)
result = optimize.optimize_manifest_file(
    "examples/autonomous_redteam_task_world_optimization.json"
)
assert result["kind"] == "agent-learning.optimization.v1"
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/autonomous-redteam-task-world.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
```

The deterministic run selects a best candidate with `optimization_score: 1.0`.
The summary includes the search mechanics other optimizer pages share
(`best_candidate_id`, `search_paths`, `total_evaluations`,
`total_iterations`, `threshold`, `candidate_lineage_*`,
`optimizer_governance_*`); the safety substance is in the evaluation: world
transitions verified against preconditions and effects, the hostile canary
refused, and the self-improvement trace (verification, memory write, runbook,
stop) scored as part of the candidate's fitness rather than observed
informally.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| manifest rejected at load | config fault | `agent-learn doctor` → `summary.public_boundary_passed`, then the manifest error line |
| candidate completes task but fails evaluation | a real breach | check the hostile-resistance metrics — the canary was likely touched |
| `optimization_passed: false` with no transition progress | world fault | preconditions never satisfied — inspect the world transition definitions |

## 5. Prove it / keep it

Worlds compound the value of regressions: a promoted finding here replays the
whole hostile world, not one prompt — see
[promote-to-regression](promote-to-regression.md). From this page the track
closes its loop: the breach lifecycle inside persisted world state is
[stored-prompt-injection](stored-prompt-injection.md), multi-step escalation
through world time is [long-horizon](long-horizon.md), and assigning blame
across an autonomous trajectory's many actors is
[causal-attribution](causal-attribution.md). The simulate track's
worlds-and-hooks page documents the world machinery itself.
