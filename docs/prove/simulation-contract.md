---
kind: agent-learning.docs-page.v1
track: prove
objective: reliability
stage: prove
backing:
  - examples/sdk_simulation_contract.py
artifact_kinds:
  - agent-learning.simulation.v1
  - agent-learning.objective.v1
commands:
  - python examples/sdk_simulation_contract.py artifacts/simulation-contract.json
  - agent-learn simulation lift examples/run_manifest.json --output artifacts/lifted-simulation.json
  - agent-learn simulation validate artifacts/lifted-simulation.json
postcondition: python -c "import json; c=json.load(open('artifacts/simulation-contract.json')); l=json.load(open('artifacts/lifted-simulation.json')); assert c['roundtrip_all_equal'] is True, c; assert l['simulation']['kind']=='agent-learning.simulation.v1', l; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Simulation contract: one typed world, every builder lifts into it

> **Twin:** [`examples/sdk_simulation_contract.py`](../../examples/sdk_simulation_contract.py)
> · emits `agent-learning.simulation.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

`agent-learning.simulation.v1` is the generic world definition: it owns its
personas, binds scenarios to a typed cast (`user`/`opponent`/`coworker`/
`counterpart`), declares a typed `world.kind` with first-class tool mocking,
and is content-addressed by the Persona rule (changing a tool mock level
changes the simulation's identity). It sits ABOVE the adapters — no new
step API — so every existing run/optimization builder lifts into it
mechanically and re-derives a run manifest that replays byte-for-byte.

The failure classes this page targets: a typed persona layer silently dropped
on the manifest path, a declared goal that the engine never evaluates, and a
round-trip that diverges from the original run.

## 2. Run it

Generate the committed fixtures (the S1-S8 round-trip census, the typed-persona
and declared-goal fixtures, the world-kind matrix, the tool-mock identity
pair, and the content-hash tripwire), then lift and validate a legacy manifest:

```bash
python examples/sdk_simulation_contract.py artifacts/simulation-contract.json
agent-learn simulation lift examples/run_manifest.json --output artifacts/lifted-simulation.json
agent-learn simulation validate artifacts/lifted-simulation.json
```

The same flow from the SDK:

```python
from fi.alk import simulate

run_manifest = simulate.load_manifest_file("examples/run_manifest.json")
simulation = simulate.derive_simulation_manifest(run_manifest)
rerun = simulate.derive_simulation_run_manifest(simulation, agent=run_manifest["agent"])
```

## 3. What you built

Run the postcondition verbatim:

```bash
python -c "import json; c=json.load(open('artifacts/simulation-contract.json')); l=json.load(open('artifacts/lifted-simulation.json')); assert c['roundtrip_all_equal'] is True, c; assert l['simulation']['kind']=='agent-learning.simulation.v1', l; print('ok')"
```

The artifact records the census size and that every builder's original run and
its re-derived run produce the same envelope-stripped canonical JSON.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `simulation validate` exits 1 with `cast_role_unknown` | a cast member used a role outside the closed set | `missing_engine_modules` |
| `simulation validate` exits 1 with `tool_mock_level_undeclared` | a tool binding has no declared mock level | `missing_engine_modules` |
| `simulation run` exits 1 with `world_kind_refusal` | a typed-only kind was asked to run contract-native | `public_boundary_passed` |

## 5. Prove it / keep it

The `simulation_contract_readiness` release gate recomputes these committed
fixtures on every `release-check`: the round-trip census equality, the G4/G3
repairs, the world-kind executable/typed split, the tool-mock identity flip,
the content-hash tripwire, and the objective schema all gate the release.
