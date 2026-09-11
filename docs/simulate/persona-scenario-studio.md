---
kind: agent-learning.docs-page.v1
track: simulate
objective: capability
stage: simulate
backing:
  - examples/sdk_persona_scenario_studio.py
artifact_kinds:
  - agent-learning.persona-library.v1
commands:
  - python examples/sdk_persona_scenario_studio.py artifacts/persona-scenario-studio.json
postcondition: python -c "import json; p=json.load(open('artifacts/persona-scenario-studio.json')); assert p['kind']=='agent-learning.persona-scenario-studio-readiness.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - api_key_configured
opt_in_lane: false
---

# Persona & Scenario Studio: typed test cases you can measure

> **Twin:** [`examples/sdk_persona_scenario_studio.py`](../../examples/sdk_persona_scenario_studio.py)
> · emits `agent-learning.persona-library.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

A free-text persona ("an impatient customer") is unmeasurable: you cannot tell
whether the simulated user actually behaved impatiently or the model just said
it would. The studio replaces prose with five typed layers — identity,
temperament, behavior_policy, knowledge, provenance — where the behavior policy
is executable and searchable. The six canon behavior axes (`patience`,
`disclosure`, `interruption`, `escalation`, `cooperation`, `repair`) each ship
paired one-to-one with a transcript-observable realization metric
(`turns_to_escalation`, `info_withholding_rate`, `interruption_count`,
`intensity_trajectory_match`, `compliance_rate`, `repair_turn_fraction`). An
axis with no realization metric does not ship.

The temperament axes (`rajas`, `sattva`, `tamas`) are a scholarly design device
used as deterministic engineering metadata, never a psychometric claim about
simulated users — and they always appear with their realization metrics, which
is what makes them falsifiable rather than decorative. Demographics are
optional, always lint-flagged, and explain about 1.5% of behavioral variance.
No layer ever claims population representativeness; the schema says so
(`representativeness_claim: "none"`), not just the docs.

Legacy embedded-dict personas keep working unchanged: they validate, run, and
auto-upgrade with `provenance=legacy` — they simply stay untyped and produce no
fidelity evidence.

## 2. Run it

```bash
python examples/sdk_persona_scenario_studio.py artifacts/persona-scenario-studio.json
```

The example runs entirely on the committed `examples/persona_library/`
fixtures: it round-trips typed personas, upgrades a legacy row, compiles
behavior policies, writes a content-addressed library, computes obligation
coverage with a budgeted residual estimator, runs the set-level bias lint, and
imports Vapi/Retell personas byte-exact. No network, no API key.

## 3. What you built

A content-addressed persona/scenario library (`agent-learning.persona-library.v1`):
each persona and scenario is stored under its own `sha256` content hash, so a
hand-edit is loud (the re-hash mismatches and the load is refused). Coverage is
reported as obligation coverage per axis plus a residual estimate — never a
global library count; `library_size` and `scenario_count` are forbidden headline
keys. The whole thing is local-first under `.agent-learning/library/`.

## 4. When it fails

`agent-learn doctor` reports `missing_engine_modules` when the simulation engine
is not importable, and `api_key_configured` for the keyed pull lane (not needed
for this page — everything here is local). A persona that declares a behavior
axis without its realization metric is refused by `validate_persona`; a
demographics-bearing persona is flagged and cannot be admitted until the
set-level bias lint passes.

## 5. Prove it / keep it

The `persona_scenario_studio_readiness` release gate executes this exact example
on every release-check and audits the fidelity admission loop, calibration
lifecycle, coverage, bias lint, vendor import parity, and scan refusals. To
prove it live, the keyed account pull (`agent-learn persona pull`) downloads
real personas with full pin + checksum + content-scan provenance — linked here,
never claimed offline.
