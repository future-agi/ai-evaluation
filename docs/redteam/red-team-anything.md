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
  - AGENT_LEARNING_REDTEAM_EXAMPLE_KEY=local-example agent-learn redteam examples/redteam_manifest.json --output artifacts/redteam.json
  - AGENT_LEARNING_REDTEAM_EXAMPLE_KEY=local-example agent-learn redteam examples/redteam_manifest.json --dry-run
postcondition: python -c "import json; p=json.load(open('artifacts/redteam.json')); assert p['kind']=='agent-learning.redteam.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - api_key_configured
opt_in_lane: false
---

# Red-Team Anything

> **Twin:** [`examples/sdk_redteam_adaptive_loop_optimization.py`](../../examples/sdk_redteam_adaptive_loop_optimization.py)
> · emits `agent-learning.redteam.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

A red-team campaign is a declared matrix, not a pile of prompts. Every campaign
in this track names its attacks (`prompt_injection`, `credential_exfiltration`,
`memory_poisoning`, ...), the surfaces they enter through (`tool`, `memory`,
`retrieval`, `multi_agent_handoff`), the channels they ride (`chat`, `memory`,
`file`), and the taxonomies they map to (`owasp_llm_top_10`,
`owasp_agentic_ai`, `mcp_security`). The artifact records which cells of that
matrix were covered, which were executed, and which produced findings — so a
gap is a named cell, not a feeling.

`examples/redteam_manifest.json` is the track's smallest complete campaign:
two attacks across the `tool` and `memory` surfaces, a `high` severity
threshold, and a scripted defender that treats tool output as untrusted and
calls the three campaign-status tools the evaluation requires
(`adversarial_pack_status`, `red_team_campaign_status`,
`list_red_team_campaign_gaps`). The persona is Asha, a security engineer who
needs evidence, and the `agent_report` threshold is 0.85. Everything runs
offline against the `local_cli` provider; the env var in the command is CI
metadata, not a credential.

The rest of the track deepens one axis at a time: benchmark corpora as
campaign evidence ([corpus](corpus.md)), reading your first artifact
([first-campaign](first-campaign.md)), optimizing campaign coverage
([campaign-optimization](campaign-optimization.md)), result-driven escalation
([adaptive-loop](adaptive-loop.md)), cross-session persistence attacks
([stored-prompt-injection](stored-prompt-injection.md)), multi-phase campaigns
([long-horizon](long-horizon.md)), mutation and minimization
([attack-evolution-shrink](attack-evolution-shrink.md)), blame assignment
([causal-attribution](causal-attribution.md)), hostile task worlds
([autonomous-task-world](autonomous-task-world.md)), and the step that makes
all of it permanent ([promote-to-regression](promote-to-regression.md)).

## 2. Run it

CLI:

```bash
AGENT_LEARNING_REDTEAM_EXAMPLE_KEY=local-example \
  agent-learn redteam examples/redteam_manifest.json --output artifacts/redteam.json
```

SDK, same operation:

```python
import asyncio
import os

from fi.alk import redteam

os.environ.setdefault("AGENT_LEARNING_REDTEAM_EXAMPLE_KEY", "local-example")
result = asyncio.run(redteam.redteam_manifest_file("examples/redteam_manifest.json"))
assert result["kind"] == "agent-learning.redteam.v1"
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/redteam.json')); assert p['kind']=='agent-learning.redteam.v1', p['kind']; print('ok')"
```

`artifacts/redteam.json` carries `summary.redteam` — the campaign block with
`attack_types`, `channels`, `providers`, `frameworks`, `severity_threshold`,
and per-severity finding counts — plus `summary.metric_averages` with the
`adversarial_resilience` and `red_team_campaign_quality` scores the 0.85
threshold gates on. The scripted defender passes with `evaluation_score: 1.0`;
swap in your agent and the same matrix becomes an honest gate.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `missing required env` on the manifest | config fault | set the manifest's `required_env` (any local value works offline) |
| key errors on a platform-connected lane | config fault | `agent-learn doctor` → `summary.api_key_configured` |

## 5. Prove it / keep it

The campaign artifact is the unit everything downstream consumes: the
[adaptive loop](adaptive-loop.md) feeds a failed campaign back into an
optimizer (the twin on this page embeds exactly such an
`agent-learning.redteam.v1` source result and expands tool-only coverage to
four surfaces), and any finding promotes into a replayable regression with
`agent-learn promote-to-regression` — see
[promote-to-regression](promote-to-regression.md). Start with
[first-campaign](first-campaign.md) to read every field of the artifact you
just produced.
