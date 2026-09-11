---
kind: agent-learning.docs-page.v1
track: redteam
objective: safety
stage: optimize
backing:
  - examples/sdk_voice_redteam_campaign.py
artifact_kinds:
  - agent-learning.optimization.v1
  - agent-learning.run.v1
commands:
  - agent-learn redteam examples/voice_redteam/composed_ab_manifest.json --ab-harness --output artifacts/voice-composed-ab.json
postcondition: python -c "import json; p=json.load(open('artifacts/voice-composed-ab.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Voice composed campaigns: persona × signal search and the A/B harness

> **Twin:** [`examples/sdk_voice_redteam_campaign.py`](../../examples/sdk_voice_redteam_campaign.py)
> · emits `agent-learning.optimization.v1` (the A/B result rides an
> `ab_harness` block) · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

> **Authorization boundary.** Voice red-team campaigns run only against agents
> the user owns or is explicitly authorized to test. Voice attacks never target
> third parties; telephony rungs are owner-keyed; all examples and fixtures run
> against kit-owned local agents.

## 1. What you are testing

The composed search is ONE optimizer target: a single search space whose
dimensions are persona dials (the `rajas` / `sattva` / `tamas` temperament axes
and the searchable behavior axes such as `interruption_propensity` and
`escalation_schedule`) crossed with signal parameters (the rung-1 operator, its
rate, and seed). It rides the Phase-4 task-optimization manifest contract, runs
with a declared `eval_budget`, and ranks by attack-success-at-fidelity using
external-verification ranking (`ranking_source: "evaluation_suite"`).

The A/B harness runs three arms at equal declared budget: `composed` (both dial
families vary), `persona_only` (signal frozen clean), and `signal_only`
(persona frozen at the embedded values). The result is no new artifact kind —
it is an `ab_harness` block embedded in the `agent-learning.optimization.v1`
payload.

## 2. Run it

CLI (the `--ab-harness` flag — one contract, two doors):

```bash
agent-learn redteam examples/voice_redteam/composed_ab_manifest.json \
  --ab-harness --output artifacts/voice-composed-ab.json
```

SDK, same operation:

```python
from fi.alk import redteam

result = redteam.run_composed_voice_attack_ab(
    name="voice-composed-ab",
    persona=persona,            # a typed attack-conditioned Persona
    scenario=scenario,          # an adversarial Scenario with an escalation arc
    persona_space={"temperament.rajas": [0.3, 0.6, 0.9]},
    signal_space={"operator": ["homophone", "code_switch"], "rate": [0.05, 0.15]},
    eval_budget_per_arm=6,
)
assert result["ab_harness"]["ab_verdict"] in {"composed_lift", "no_lift", "inconclusive"}
```

## 3. What you built

Postcondition (machine-checkable — the same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/voice-composed-ab.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
```

The `ab_harness` block records the three arms, their equal per-arm budget, the
per-seed numbers, the per-seed-unanimity `ab_verdict` enum, and the numeric
`lift` as an evidence field beside the verdict. The verdict is re-derivable from
the per-seed numbers — the harness can never hand-assign a lift.

The `lift` obeys the null rules: it renders `null` whenever any arm under-ran
its declared budget (finding `composed_budget_mismatch`, exit 0 with a warning)
or any arm's quarantine rate exceeds one half (finding
`composed_arm_quarantine_epidemic`, exit 1). Quarantine is only ever for
verdict instability and a voided simulator row — never for low fidelity.
Fidelity scales the score through the kit's halving contract: a
character-broken success is halved, never dropped and never excluded.

Every artifact stamps `attack_rung: "transcript_level"` and carries the rung-1
`phone_survival` pin `{"status": "untested", "tier": "research_pinned"}`. The
acoustic rung lands as an increment when the Phase-9A loopback transport ships;
asking for rung-2 acoustic operators today refuses structured-loud with
`voice_rung_unavailable` — see the corpus page
([voice-attack-corpus](voice-attack-corpus.md)) for the opt-in lane it links to.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `voice_rung_unavailable` | expected refusal | run the transcript_level form; the acoustic rung is Phase-9A |
| `voice_target_authorization_missing` | config fault | add `target.authorization` for a non-local target |
| `lift: null` + `composed_budget_mismatch` | real gap | an arm did not complete its declared budget |

## 5. Prove it / keep it

A composed-search run that proves `composed_lift` on the gate-pinned harness is
the capstone evidence for the persona-conditioned acoustic-attack search. Wire
the `--ab-harness` command into CI at a declared budget; archive the artifact;
graduate any real breach into a credential-free regression pack via the capture
flow.
