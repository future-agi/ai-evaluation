---
kind: agent-learning.docs-page.v1
track: redteam
objective: safety
stage: simulate
backing:
  - examples/sdk_persona_scenario_studio.py
artifact_kinds:
  - agent-learning.redteam.v1
  - agent-learning.persona-library.v1
commands:
  - python examples/sdk_persona_scenario_studio.py artifacts/persona-conditioned-attacks.json
postcondition: python -c "import json; p=json.load(open('artifacts/persona-conditioned-attacks.json')); assert p['persona_conditioned_manifest']['built'] is True; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - api_key_configured
opt_in_lane: false
---

# Persona-Conditioned Attacks: in-character is the attack quality

> **Twin:** [`examples/sdk_persona_scenario_studio.py`](../../examples/sdk_persona_scenario_studio.py)
> · emits `agent-learning.redteam.v1` and `agent-learning.persona-library.v1`
> · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

An adversarial turn is only a realistic attack if the simulated attacker stayed
in character while pressing. Persona-conditioned red-teaming binds a typed
attack persona — its `attack.strategies` and `attack.surfaces` drawn from the
gate-enforced 10×6 taxonomy — to an adversarial scenario carrying a turn-wise
escalation arc (the Crescendo finding). The arc's per-turn pressure feeds the
policy directives so the attacker escalates in character, and feeds the fidelity
trajectory so you can measure whether character held under counter-pressure.

The wedge no vendor measures: in-character fidelity becomes the attack quality.
Each attack row reuses the unit-3 fidelity record verbatim, gets a
`character_held` flag (the verdict held across the arc) and a
`quality = asr_weight × adherence`. A character-broken attack is flagged and
down-weighted — never dropped: a successful but out-of-character attack is still
a finding, just a less realistic one.

## 2. Run it

```bash
python examples/sdk_persona_scenario_studio.py artifacts/persona-conditioned-attacks.json
```

The example builds a persona-conditioned red-team manifest from a typed
attack-conditioned persona and an adversarial scenario whose escalation arc sets
`min_turns = max_turns = len(arc.steps)`, so the Crescendo arc has turns to
escalate across. The embedded typed persona replaces the default red-team-owner
persona in the scenario rows.

## 3. What you built

A red-team manifest (`agent-learning.redteam.v1`) conditioned on a library
persona (`agent-learning.persona-library.v1`). PCAP-style parallel
multi-persona search is simply N manifests from N personas — the existing
campaign machinery runs them; no new runner. The executable persona-conditioned
campaign (with per-attack in-character fidelity) is exercised by the red-team
readiness certification gate.

## 4. When it fails

`agent-learn doctor` reports `missing_engine_modules` when the engine is not
importable and `api_key_configured` for the keyed lane. A persona whose
`attack.strategies` or `attack.surfaces` fall outside the 10×6 taxonomy are
refused by `validate_persona`; an adversarial scenario without its attack_type,
attack_surface, and escalation arc is refused at construction.

## 5. Prove it / keep it

The `redteam_readiness_certification` gate executes the persona-conditioned
campaign and asserts per-attack fidelity records (verdicts in the persona
fidelity vocabulary) with at least one in-character attack, and the
`persona_scenario_studio_readiness` gate executes the manifest build on every
release-check. Bias monitoring runs on attack-conditioned persona sets
identically — the Dark-Triad amplification finding is covered by the same lint.
