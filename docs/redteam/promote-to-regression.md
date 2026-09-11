---
kind: agent-learning.docs-page.v1
track: redteam
objective: safety
stage: promote
backing:
  - examples/sdk_redteam_readiness_certification_optimization.py
  - examples/sdk_regression_artifact_suite.py
artifact_kinds:
  - agent-learning.regression-promotion.v1
commands:
  - mkdir -p artifacts && cp examples/regression_artifacts/redteam-finding.json artifacts/redteam-finding.json
  - agent-learn promote-to-regression artifacts/redteam-finding.json --output regression-promotion.json --manifest promoted-regression.json
postcondition: python -c "import json; p=json.load(open('artifacts/regression-promotion.json')); assert p['kind']=='agent-learning.regression-promotion.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Promote findings to regressions

> **Twin:** [`examples/sdk_redteam_readiness_certification_optimization.py`](../../examples/sdk_redteam_readiness_certification_optimization.py)
> · emits `agent-learning.regression-promotion.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

A red-team finding has a half-life. The campaign that produced it moves on,
the agent gets retrained or reconfigured, and six months later nobody can say
whether the hole is still closed — because nothing re-checks it.
`agent-learn promote-to-regression` ends that decay: it reads a failed
artifact, extracts its findings at or above a severity floor, and emits a
*runnable* red-team regression manifest that replays exactly those findings
as a standing test. The failure class is regression by amnesia — re-shipping
a vulnerability that was found, fixed, and forgotten.

`examples/regression_artifacts/redteam-finding.json` is the committed source
fixture: a failed campaign artifact (`status: failed`, `policy_score: 0.0`)
recording a `prompt_injection` breach through the `system_prompt` surface on
the `chat` channel. Promotion turns it into two files — the promotion record
(what was promoted, from where, at what level) and the regression manifest
(an `agent-learning.run.v1` manifest you check into your repo and replay
forever). `--min-level` (default `warning`) and `--max-findings` (default 25)
control the promotion gate.

The two twins cover both ends of the contract: the readiness-certification
optimizer proves campaign evidence can be certified for promotion across
framework targets, and the regression artifact suite drives the full
baseline → compare → replay → promote → report journey under the
`regression_artifact_readiness` gate.

## 2. Run it

CLI (work on a copy so the committed fixture stays pristine; outputs land
beside the source artifact in `artifacts/`):

```bash
mkdir -p artifacts && cp examples/regression_artifacts/redteam-finding.json artifacts/redteam-finding.json

agent-learn promote-to-regression artifacts/redteam-finding.json \
  --output regression-promotion.json \
  --manifest promoted-regression.json
```

SDK, same operation:

```python
from fi.alk import simulate

promotion = simulate.promote_to_regression_file(
    "artifacts/redteam-finding.json",
    min_level="warning",
)
assert promotion["kind"] == "agent-learning.regression-promotion.v1"
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/regression-promotion.json')); assert p['kind']=='agent-learning.regression-promotion.v1', p['kind']; print('ok')"
```

`artifacts/regression-promotion.json` is the audit record: source artifact,
promotion level, and the findings carried over.
`artifacts/promoted-regression.json` is the living test — an
`agent-learning.run.v1` manifest named after its source
(`redteam-finding-source-regression`) that reconstructs the breached scenario
so the original failure must stay fixed to pass. The manifest is plain JSON:
commit it next to your other regression manifests and treat edits to it like
edits to a test.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| no findings promoted from a green artifact | expected | promotion consumes failures; a passing run has nothing to pin |
| `promote-to-regression requires --max-findings greater than 0` | config fault | raise `--max-findings`, or tighten `--min-level` instead |
| promoted manifest fails immediately on replay | the gate working | the regression caught drift — fix the agent, not the manifest |

## 5. Prove it / keep it

Replay is the keep-it step: `agent-learn replay artifacts/promoted-regression.json
--output regression-replay.json` runs the promoted manifest (and any glob of
its siblings) and aggregates a CI verdict — the simulate track's
regression-lifecycle page documents the full baseline/compare/replay loop.
Every page in this track funnels here: campaign findings
([first-campaign](first-campaign.md)), shrunken counterexamples
([attack-evolution-shrink](attack-evolution-shrink.md)), cross-session
breaches ([stored-prompt-injection](stored-prompt-injection.md)), and world
breaches ([autonomous-task-world](autonomous-task-world.md)) all become
permanent the same way. That funnel is what makes a red-team program
cumulative: the attack surface you have closed only ever grows.
