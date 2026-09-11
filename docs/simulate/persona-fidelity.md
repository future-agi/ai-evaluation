---
kind: agent-learning.docs-page.v1
track: simulate
objective: reliability
stage: evaluate
backing:
  - examples/sdk_persona_scenario_studio.py
artifact_kinds:
  - agent-learning.persona-calibration.v1
  - agent-learning.run.v1
commands:
  - python examples/sdk_persona_scenario_studio.py artifacts/persona-fidelity.json
postcondition: python -c "import json; p=json.load(open('artifacts/persona-fidelity.json')); assert p['fidelity']['clean']['verdict']=='pass', p['fidelity']['clean']['verdict']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - api_key_configured
opt_in_lane: false
---

# Persona Fidelity: did the simulated user behave as declared?

> **Twin:** [`examples/sdk_persona_scenario_studio.py`](../../examples/sdk_persona_scenario_studio.py)
> · emits `agent-learning.persona-calibration.v1` and an in-row
> `agent-learning.run.v1` fidelity block · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

Persona fidelity asks whether the simulated user actually behaved the way the
persona declared — measured from the transcript, never from a single
unperturbed judge. The record is the Eval4Sim triple, all two-sided:
**adherence** (every axis matched against its realization metric, with separate
under/over shortfall so over-acting fails like drift), **consistency** (facts,
identity, and style stable across turns), and **naturalness** (a two-sided
caricature/flatness index — the over-acted persona fails just as the inert one
does). Drift is decomposed three ways (prompt→line, line→line, probe) and
tracked as a per-turn trajectory, fastest under pressure.

The verdict is three-valued: floors met → `pass`; floors violated →
`inconclusive` (a broken simulator says nothing about the agent, so the row is
quarantined and excluded from the score matrix but still counted); `fail` is
reserved for measurement impossibility on a typed persona. Above a `0.5`
admission-inconclusive rate the simulator, not the agent, is declared unusable
for the run.

## 2. Run it

```bash
python examples/sdk_persona_scenario_studio.py artifacts/persona-fidelity.json
```

The example scores three committed transcripts against the same typed persona:
the clean transcript passes and is admissible, the drifted transcript is
quarantined as `inconclusive`, and the over-acted transcript is failed by the
two-sided naturalness check (its caricature index is pinned high).

## 3. What you built

A calibration artifact (`agent-learning.persona-calibration.v1`) and, on each
run row, an in-row fidelity block under `agent-learning.run.v1` metadata —
fidelity is never a standalone artifact kind. The admission block marks
quarantined rows so `TestReport.admissible_results()` excludes them from
pass/fail tallies while the report still surfaces an inconclusive count.

## 4. When it fails

`agent-learn doctor` reports `missing_engine_modules` when the engine is not
importable and `api_key_configured` for the keyed lane. An untyped/legacy
persona produces no fidelity record at all — it runs fine but cannot back a
release claim. A typed persona with an empty or garbled trajectory is verdict
`fail` (reason-coded), distinct from the `inconclusive` floor quarantine.

## 5. Prove it / keep it

The `persona_scenario_studio_readiness` release gate executes this example and
asserts the clean→pass, drifted→quarantined, and over-acted→naturalness-failed
admission loop on every release-check. Calibration upgrades are monotone and the
admitted class is stamped into the library index — re-running the manifest
re-runs any quarantined row.
