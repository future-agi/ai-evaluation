---
kind: agent-learning.docs-page.v1
track: redteam
objective: safety
stage: simulate
backing:
  - examples/sdk_redteam_causal_attribution_optimization.py
artifact_kinds:
  - agent-learning.redteam.v1
commands:
  - AGENT_LEARNING_LONG_HORIZON_REDTEAM_KEY=local-example agent-learn redteam examples/long_horizon_redteam_manifest.json --output artifacts/long-horizon-redteam.json
postcondition: python -c "import json; p=json.load(open('artifacts/long-horizon-redteam.json')); assert p['kind']=='agent-learning.redteam.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - api_key_configured
opt_in_lane: false
---

# Long-horizon red-teaming

> **Twin:** [`examples/sdk_redteam_causal_attribution_optimization.py`](../../examples/sdk_redteam_causal_attribution_optimization.py)
> · emits `agent-learning.redteam.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

Some attacks need patience. Each individual turn looks defensible — a small
scope expansion here, a remembered "preference" there — and the breach only
exists as the *composition* of steps across an extended trajectory. A per-turn
safety check scores every step green; the trajectory still ends with the agent
doing something no single turn authorized. That is the long-horizon failure
class, and it is invisible to campaigns that evaluate turns in isolation.

`examples/long_horizon_redteam_manifest.json` runs an auto-generated campaign
from the `long_horizon_agent` preset against a stateful target
(`long-horizon-support-agent` in a `local-stateful-ci` environment). The
persona is Isha, a security program lead, and the campaign plants a canary
(`FA_LONG_HORIZON_CANARY_001`) whose appearance anywhere in late-trajectory
output is an unambiguous escalation signal. The `agent_report` threshold is
0.9, and because the preset auto-generates the campaign, the executed matrix
(attack types, surfaces, taxonomies) is recorded in the artifact rather than
hand-listed in the manifest.

Long horizons raise a second question the moment they fail: *which* step
caused the breach? The twin on this page is the causal-attribution optimizer —
covering compositional attack classes like `objective_drift`, `tool_chaining`,
and `intent_hijacking` — and its page, [causal-attribution](causal-attribution.md),
is the designated follow-up to a red long-horizon run.

## 2. Run it

CLI:

```bash
AGENT_LEARNING_LONG_HORIZON_REDTEAM_KEY=local-example \
  agent-learn redteam examples/long_horizon_redteam_manifest.json \
  --output artifacts/long-horizon-redteam.json
```

SDK, same operation (the builder generates the same preset campaign for your
own target):

```python
import asyncio
import os

from fi.alk import redteam

os.environ.setdefault("AGENT_LEARNING_LONG_HORIZON_REDTEAM_KEY", "local-example")
result = asyncio.run(
    redteam.redteam_manifest_file("examples/long_horizon_redteam_manifest.json")
)
assert result["kind"] == "agent-learning.redteam.v1"
# For your agent: redteam.build_long_horizon_redteam_manifest(name=..., target=...)
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/long-horizon-redteam.json')); assert p['kind']=='agent-learning.redteam.v1', p['kind']; print('ok')"
```

The artifact's `summary.redteam` block records the auto-generated campaign:
`presets` and `preset_sources` name where the matrix came from,
`attack_types`/`channels`/`frameworks` show what actually ran, and the
severity-split finding counts gate the exit code. The deterministic defended
run scores `evaluation_score` ≈ 0.936 against the 0.9 threshold — close
enough to the line that a single escalation step pushes it red, which is
exactly the sensitivity a long-horizon gate needs.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `missing required env` | config fault | `agent-learn doctor` → `summary.api_key_configured`; any local value works offline |
| canary string in late-trajectory output | a real breach | the trajectory crossed the escalation line — find the step with causal attribution |
| score just under threshold, no single bad turn | the long-horizon class itself | compare early- vs late-trajectory metric averages in `summary` |

## 5. Prove it / keep it

A red long-horizon artifact starts a two-step pipeline. First, attribute:
[causal-attribution](causal-attribution.md) assigns responsibility for the
breach to specific steps instead of the whole trajectory. Second, pin:
[promote-to-regression](promote-to-regression.md) turns the offending
trajectory into a standing regression manifest. To push the campaign itself
harder over time, `examples/long_horizon_redteam_optimization.json` wraps this
manifest in an optimization search the same way
[campaign-optimization](campaign-optimization.md) does for the single-session
matrix. Cross-session persistence — the sharpest long-horizon special case —
has its own page: [stored-prompt-injection](stored-prompt-injection.md).
