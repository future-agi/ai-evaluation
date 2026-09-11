---
kind: agent-learning.docs-page.v1
track: redteam
objective: safety
stage: optimize
backing:
  - examples/sdk_redteam_society_optimization.py
artifact_kinds:
  - agent-learning.optimization.v1
commands:
  - AGENT_LEARNING_REDTEAM_OPT_EXAMPLE_KEY=local-example agent-learn optimize examples/redteam_campaign_optimization.json --output artifacts/redteam-campaign-optimization.json
postcondition: python -c "import json; p=json.load(open('artifacts/redteam-campaign-optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Campaign optimization: searching over the attack matrix

> **Twin:** [`examples/sdk_redteam_society_optimization.py`](../../examples/sdk_redteam_society_optimization.py)
> · emits `agent-learning.optimization.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

A hand-written campaign covers what its author thought of. Campaign
optimization treats the campaign itself as the candidate: the optimizer
proposes campaign variants from a declared search space, runs each one, scores
the resulting coverage and resilience metrics, and selects the candidate that
closes the most matrix cells. The failure class is a campaign frozen at its
weakest draft — tool-only, single-attack — while the agent's actual surface
keeps growing.

`examples/redteam_campaign_optimization.json` starts from the same two-attack,
tool-plus-memory matrix as [first-campaign](first-campaign.md) and declares a
search space over campaign candidates. The optimizer's verdict comes back with
governance attached: the artifact reports `optimizer_governance_passed` with
its check counts, and `redteam_campaign_proof_passed` with an assurance level
— so the selected campaign carries evidence that selection followed the rules,
not just that a score went up.

The twin on this page is the society variant of the same operation: multiple
attacker roles propose and critique campaign candidates before selection. It
runs under the `redteam_society_causal_readiness` release gate, and its
deliberation framing is developed further in the optimizer track's
society-of-agents page.

## 2. Run it

CLI:

```bash
AGENT_LEARNING_REDTEAM_OPT_EXAMPLE_KEY=local-example \
  agent-learn optimize examples/redteam_campaign_optimization.json \
  --output artifacts/redteam-campaign-optimization.json
```

SDK, same operation:

```python
import os

from fi.alk import optimize

os.environ.setdefault("AGENT_LEARNING_REDTEAM_OPT_EXAMPLE_KEY", "local-example")
result = optimize.optimize_manifest_file("examples/redteam_campaign_optimization.json")
assert result["kind"] == "agent-learning.optimization.v1"
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/redteam-campaign-optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
```

The artifact's `summary` names `best_candidate_id` (a content-addressed
candidate hash), `optimization_score`, and `evaluation_score`, plus the
lineage block — `candidate_lineage_count` and
`candidate_lineage_selected_score_delta` — that records how the winner was
reached. The governance fields (`optimizer_governance_*`,
`redteam_campaign_proof_*`) are the difference between "the optimizer says so"
and an auditable selection.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| manifest rejected at load | config fault | `agent-learn doctor` → `summary.public_boundary_passed`, then the manifest error line |
| `optimization_passed: false` | real gap | the search space contains no candidate clearing the threshold — widen it |
| governance checks failing | selection fault | inspect `optimizer_governance_*` counts before trusting `best_candidate_id` |

## 5. Prove it / keep it

The optimized campaign is a manifest like any other: run it as your standing
gate, and re-optimize when the agent gains surfaces. Two escalations build on
this page: [adaptive-loop](adaptive-loop.md) regenerates the search from a
failed campaign result instead of a static manifest, and
[attack-evolution-shrink](attack-evolution-shrink.md) descends below the
campaign level to mutate individual attacks. Findings from any of them land in
[promote-to-regression](promote-to-regression.md).
