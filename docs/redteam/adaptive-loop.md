---
kind: agent-learning.docs-page.v1
track: redteam
objective: safety
stage: optimize
backing:
  - examples/sdk_redteam_adaptive_loop_optimization.py
artifact_kinds:
  - agent-learning.optimization.v1
commands:
  - AGENT_LEARNING_SDK_REDTEAM_ADAPTIVE_LOOP_KEY=local-example python examples/sdk_redteam_adaptive_loop_optimization.py artifacts/redteam-adaptive-loop.json
postcondition: python -c "import json; p=json.load(open('artifacts/redteam-adaptive-loop.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# The adaptive red-team loop

> **Twin:** [`examples/sdk_redteam_adaptive_loop_optimization.py`](../../examples/sdk_redteam_adaptive_loop_optimization.py)
> · emits `agent-learning.optimization.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

[Campaign optimization](campaign-optimization.md) searches over a search space
you wrote down. The adaptive loop closes the circuit: the input is a previous
campaign **result**, and the next campaign is derived from what that result
says — failed cells stay, passed cells widen, uncovered surfaces enter. The
failure class is a red-team program that never escalates: the same campaign
re-run weekly, green forever, while every adjacent surface goes untested.

The twin makes the loop concrete and deterministic. It embeds a failed
`agent-learning.redteam.v1` source result — a campaign that only ever ran
`prompt_injection` through the `tool` surface — and builds an optimization
whose candidates must expand to four attacks (`prompt_injection`,
`indirect_prompt_injection`, `credential_exfiltration`, `memory_poisoning`)
across four surfaces (`tool`, `memory`, `retrieval`, `multi_agent_handoff`)
under four taxonomies including `owasp_mcp_top_10` and `agentic_security`. The
optimizer scores candidates on campaign quality and resilience metrics and
selects the expansion, with full lineage recorded.

This is the loop a standing red-team program runs on every artifact: result in,
broader campaign out, repeat. Each iteration is itself an
`agent-learning.optimization.v1` artifact, so escalation decisions are as
auditable as the campaigns they produce.

## 2. Run it

CLI (the example is the runnable unit; it writes the artifact to the path you
give it):

```bash
AGENT_LEARNING_SDK_REDTEAM_ADAPTIVE_LOOP_KEY=local-example \
  python examples/sdk_redteam_adaptive_loop_optimization.py artifacts/redteam-adaptive-loop.json
```

SDK, same operation (the example builds the expanded campaign manifest from
its embedded source result, then calls `optimize.optimize_manifest` — the same
exec-load mechanism the release gate uses):

```python
import importlib.util
import os

os.environ.setdefault("AGENT_LEARNING_SDK_REDTEAM_ADAPTIVE_LOOP_KEY", "local-example")
spec = importlib.util.spec_from_file_location(
    "adaptive_loop", "examples/sdk_redteam_adaptive_loop_optimization.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.run("artifacts/redteam-adaptive-loop.json")
assert result["kind"] == "agent-learning.optimization.v1"
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/redteam-adaptive-loop.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
```

The artifact pairs the optimizer verdict (`best_candidate_id`,
`optimization_score` — 0.959 on the deterministic run — and
`candidate_lineage_*`) with the campaign evidence for the selected candidate:
`summary.redteam` lists the expanded `attack_types`, `surfaces`, and
`taxonomies`, and `redteam_campaign_proof_*` reports the proof checks with an
assurance level. Compare it to the embedded weak source result and the
escalation is legible cell by cell.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `Set AGENT_LEARNING_SDK_REDTEAM_ADAPTIVE_LOOP_KEY ...` | config fault | export the env var; any local value works offline |
| `optimization_passed: false` | real gap | no candidate covered the required attack/surface expansion |
| campaign proof checks failing | evidence fault | `agent-learn doctor` → `summary.public_boundary_passed`, then the `redteam_campaign_proof_*` counts |

## 5. Prove it / keep it

Run the loop on a schedule, not on inspiration: every campaign artifact —
green or red — is a valid input, and the loop's own artifacts form the audit
trail of how your coverage grew. When an expanded campaign produces a finding,
two pages take over: [attack-evolution-shrink](attack-evolution-shrink.md)
minimizes it to its essential trigger, and
[promote-to-regression](promote-to-regression.md) pins it as a permanent test.
The same release gate that covers this twin (`redteam_attack_evolution_readiness`)
verifies both halves on every `agent-learn release-check`.
