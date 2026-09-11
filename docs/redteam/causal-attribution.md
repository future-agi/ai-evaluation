---
kind: agent-learning.docs-page.v1
track: redteam
objective: safety
stage: optimize
backing:
  - examples/sdk_redteam_causal_attribution_optimization.py
artifact_kinds:
  - agent-learning.optimization.v1
commands:
  - AGENT_LEARNING_REDTEAM_CAUSAL_ATTRIBUTION_OPT_EXAMPLE_KEY=local-example agent-learn optimize examples/redteam_causal_attribution_optimization.json --output artifacts/redteam-causal-attribution.json
postcondition: python -c "import json; p=json.load(open('artifacts/redteam-causal-attribution.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Causal attribution for red-team findings

> **Twin:** [`examples/sdk_redteam_causal_attribution_optimization.py`](../../examples/sdk_redteam_causal_attribution_optimization.py)
> · emits `agent-learning.optimization.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

A finding tells you *that* the agent broke; attribution tells you *what*
broke it. In compositional attacks the distinction is the whole problem: a
breach assembled from `intent_hijacking`, `task_injection`, `objective_drift`,
and `tool_chaining` steps has many touched components — planner, tools,
memory, handoffs — and only some of them are causally responsible. The failure
class this page closes is mitigation by guesswork: patching the loudest
component in the trace while the actual causal step ships to production
unchanged.

`examples/redteam_causal_attribution_optimization.json` is an auto-generated
campaign optimization whose taxonomies extend past the OWASP lists into
`compositional_orchestration_attacks` — the class where attribution is hardest
— and whose evaluation requires attribution evidence, not just breach
evidence. Candidates are scored on whether responsibility for each finding is
assigned to specific steps with supporting checks, so a candidate that detects
breaches but cannot localize them does not win.

The twin runs this exact manifest through the public
`optimize.optimize_redteam_causal_attribution(...)` entry point and is
executed by the `redteam_society_causal_readiness` release gate, which also
covers the society-of-attackers variant — deliberation and attribution are two
halves of the same evidence standard.

## 2. Run it

CLI:

```bash
AGENT_LEARNING_REDTEAM_CAUSAL_ATTRIBUTION_OPT_EXAMPLE_KEY=local-example \
  agent-learn optimize examples/redteam_causal_attribution_optimization.json \
  --output artifacts/redteam-causal-attribution.json
```

SDK, same operation:

```python
import os

from fi.alk import optimize

os.environ.setdefault(
    "AGENT_LEARNING_REDTEAM_CAUSAL_ATTRIBUTION_OPT_EXAMPLE_KEY", "local-example"
)
result = optimize.optimize_manifest_file(
    "examples/redteam_causal_attribution_optimization.json"
)
assert result["kind"] == "agent-learning.optimization.v1"
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/redteam-causal-attribution.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
```

The deterministic run selects a best candidate at `optimization_score` ≈
0.967 with `evaluation_score: 1.0`. Beyond the standard optimizer fields
(`best_candidate_id`, `candidate_lineage_*`, `optimizer_governance_*`), the
summary carries the multi-agent coordination proof block
(`multi_agent_coordination_proof_passed`, check counts, assurance level) —
attribution across orchestrated components is only credible if the
coordination evidence itself checks out — and `summary.redteam` records the
compositional campaign (attack types, surfaces, signals, taxonomies) the
attribution was earned against.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| manifest rejected at load | config fault | `agent-learn doctor` → `summary.public_boundary_passed`, then the manifest error line |
| coordination proof checks failing | evidence fault | inspect `multi_agent_coordination_proof_*` before trusting any attribution |
| `optimization_passed: false` | real gap | no candidate produced localizable findings — your trace lacks the signals attribution needs |

## 5. Prove it / keep it

Attribution converts a long trace into a short list of responsible steps; pin
each of them. The minimal reproduction path is
[attack-evolution-shrink](attack-evolution-shrink.md) — shrink the attributed
counterexample to its essential trigger — and the permanence path is
[promote-to-regression](promote-to-regression.md). For the trajectories that
make attribution necessary in the first place, work backwards from
[long-horizon](long-horizon.md); for attribution across cooperating attacker
roles, the optimizer track's society-of-agents page extends the same gate.
