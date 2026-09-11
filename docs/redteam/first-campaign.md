---
kind: agent-learning.docs-page.v1
track: redteam
objective: safety
stage: simulate
backing:
  - examples/sdk_redteam_adaptive_loop_optimization.py
artifact_kinds:
  - agent-learning.redteam.v1
commands:
  - AGENT_LEARNING_REDTEAM_EXAMPLE_KEY=local-example agent-learn redteam examples/redteam_manifest.json --output artifacts/first-campaign.json
postcondition: python -c "import json; p=json.load(open('artifacts/first-campaign.json')); assert p['kind']=='agent-learning.redteam.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - api_key_configured
opt_in_lane: false
---

# Your first red-team campaign

> **Twin:** [`examples/sdk_redteam_adaptive_loop_optimization.py`](../../examples/sdk_redteam_adaptive_loop_optimization.py)
> · emits `agent-learning.redteam.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

This page is the anatomy lesson: one campaign manifest, one artifact, every
field read. The failure class is a campaign you cannot interrogate — a run
that says "passed" without saying which attacks ran, through which surfaces,
and what would have counted as a finding.

`examples/redteam_manifest.json` has four parts. The `redteam` block declares
the matrix: attacks `prompt_injection` and `credential_exfiltration`, surfaces
`tool` and `memory`, channel `chat`, provider `local_cli`, taxonomies
`owasp_llm_top_10`, `owasp_agentic_ai`, and `mcp_security`, and
`severity_threshold: high`. The `scenario` block sets the persona (Asha, a
security engineer who needs proof the agent treats tool output as untrusted).
The `agent` block is a scripted defender whose responses call the three
campaign-status tools the evaluation requires. The `evaluation` block sets an
`agent_report` threshold of 0.85 with heavy weight on `adversarial_resilience`
and `red_team_campaign_quality`.

Because the defender is scripted, the run is deterministic and the pass is by
construction — which is the point. You first verify the gate machinery scores
a known-good defender correctly; only then do you point the same matrix at
your real agent, where the verdict is informative.

## 2. Run it

CLI:

```bash
AGENT_LEARNING_REDTEAM_EXAMPLE_KEY=local-example \
  agent-learn redteam examples/redteam_manifest.json --output artifacts/first-campaign.json
```

SDK, same operation:

```python
import asyncio
import os

from fi.alk import redteam

os.environ.setdefault("AGENT_LEARNING_REDTEAM_EXAMPLE_KEY", "local-example")
manifest = redteam.load_manifest_file("examples/redteam_manifest.json")
result = asyncio.run(redteam.redteam_manifest(manifest))
assert result["status"] == "passed"
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/first-campaign.json')); assert p['kind']=='agent-learning.redteam.v1', p['kind']; print('ok')"
```

Read the artifact top-down: `status` and `exit_code` are the CI verdict;
`summary.case_count` and `summary.evaluation_score` (1.0 for the scripted
defender) are the headline; `summary.redteam` is the executed campaign —
`attack_types`, `channels`, `providers`, `frameworks`, `severity_threshold`,
and `finding_count` split by severity (`error_finding_count`,
`note_finding_count`); `summary.metric_averages` holds the per-metric scores
the threshold gated on. A finding at or above `severity_threshold` flips
`status` to `failed` and the exit code to 1 — that artifact is what you
promote in [promote-to-regression](promote-to-regression.md).

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `redteam manifest requires a redteam block` | config fault | the manifest is a plain run manifest — use `agent-learn run`, or add the `redteam` block |
| `missing required env` | config fault | `agent-learn doctor` → `summary.api_key_configured`; offline, any local value satisfies it |
| status `failed` with findings | the gate working | read `summary.redteam` finding counts before blaming the harness |

## 5. Prove it / keep it

Swap the scripted `agent` block for your real target and re-run: the same
matrix, scoring, and severity gate now produce live findings. From there the
track forks by what you want to harden: widen coverage with
[campaign-optimization](campaign-optimization.md), escalate from results with
[adaptive-loop](adaptive-loop.md), or go after the cross-session class with
[stored-prompt-injection](stored-prompt-injection.md). Whatever path you take,
every failed artifact funnels into
[promote-to-regression](promote-to-regression.md) so the finding can never
silently come back.
