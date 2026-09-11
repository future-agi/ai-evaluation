---
kind: agent-learning.docs-page.v1
track: redteam
objective: safety
stage: optimize
backing:
  - examples/sdk_redteam_attack_evolution_optimization.py
artifact_kinds:
  - agent-learning.attack-evolution-shrink.v1
  - agent-learning.optimization.v1
commands:
  - AGENT_LEARNING_SDK_REDTEAM_ATTACK_EVOLUTION_KEY=local-example python examples/sdk_redteam_attack_evolution_optimization.py artifacts/attack-evolution.json
  - agent-learn shrink artifacts/attack-evolution.json --output attack-evolution-shrink.json --manifest attack-evolution-regression.json
postcondition: python -c "import json; p=json.load(open('artifacts/attack-evolution-shrink.json')); assert p['kind']=='agent-learning.attack-evolution-shrink.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Attack evolution and shrink

> **Twin:** [`examples/sdk_redteam_attack_evolution_optimization.py`](../../examples/sdk_redteam_attack_evolution_optimization.py)
> · emits `agent-learning.attack-evolution-shrink.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

Attack evolution is search below the campaign level: instead of selecting
between whole campaigns, the optimizer mutates *individual attacks*. Seeds go
in; operators — `semantic_mutation`, `trajectory_splice`, `surface_transfer` —
produce variants; a verifier replays each variant; and coverage is tracked on
a four-axis grid (`attack_type` × `surface` × `operator` × `verifier`). The
failure class is a finding you cannot use: an attack that "worked once" in a
sprawling trajectory, with no minimal reproduction and no way to tell which
mutation mattered.

The twin runs the whole arc deterministically. Its weak candidate is a
seed-only evolution — one `prompt_injection` seed on the `tool` surface, one
proposed mutation, no counterexample replay evidence. Its verified candidate
closes every gate: semantic mutation, trajectory splice, surface transfer,
outcome feedback, counterexample minimization, and a replayable regression
verifier, across three attack types and three surfaces. The optimizer must
select the verified candidate, and the resulting artifact contains verified
counterexamples with minimal-repro records.

`agent-learn shrink` is the second half: it takes that artifact, extracts the
verified counterexample, minimizes it, and writes a runnable regression
manifest. Shrink refuses inputs without attack-evolution evidence and refuses
evidence that is not local-only — a counterexample you cannot replay offline
is not a regression test.

## 2. Run it

CLI (evolve, then shrink — shrink's outputs land beside its source artifact in
`artifacts/`):

```bash
AGENT_LEARNING_SDK_REDTEAM_ATTACK_EVOLUTION_KEY=local-example \
  python examples/sdk_redteam_attack_evolution_optimization.py artifacts/attack-evolution.json

agent-learn shrink artifacts/attack-evolution.json \
  --output attack-evolution-shrink.json \
  --manifest attack-evolution-regression.json
```

SDK, same operation:

```python
from fi.alk import simulate

shrunk = simulate.shrink_attack_evolution_file("artifacts/attack-evolution.json")
assert shrunk["kind"] == "agent-learning.attack-evolution-shrink.v1"
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/attack-evolution-shrink.json')); assert p['kind']=='agent-learning.attack-evolution-shrink.v1', p['kind']; print('ok')"
```

The shrink artifact's `summary` is a checklist of what makes a counterexample
trustworthy: `counterexample_id` and `minimized_replay_id` name the minimal
repro, `reproduces_current_failure` confirms the minimized form still breaks
the current candidate, `fixed_candidate_passes` confirms the fixed candidate
survives it (`non_regression_gate`), `local_only` certifies offline
replayability, and `kept_record_count` vs `lineage_record_count` quantifies
the minimization. Next to it, `artifacts/attack-evolution-regression.json` is
a runnable `agent-learning.run.v1` manifest — the finding as a test.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `shrink requires an artifact with attack-evolution evidence` | wrong input | point shrink at the evolution artifact, not a plain campaign result |
| `shrink requires local-only evidence; external markers: ...` | evidence fault | remove external dependencies from the counterexample path |
| `shrink requires at least one verified counterexample` | real gap | the evolution never verified a breach — there is nothing to minimize |

## 5. Prove it / keep it

Run the regression manifest on every change —
`agent-learn replay artifacts/attack-evolution-regression.json` slots it into
the same replay suite as your other regressions — and promote shrink output
through [promote-to-regression](promote-to-regression.md) when you want the
finding tracked alongside campaign-level promotions. The
`redteam_attack_evolution_readiness` release gate executes this twin on every
`agent-learn release-check`, so the evolution-and-shrink machinery itself is
re-proven each time you cut a release.
