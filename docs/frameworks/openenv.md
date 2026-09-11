---
kind: agent-learning.docs-page.v1
track: frameworks
objective: behavior
stage: simulate
backing:
  - examples/sdk_framework_adapter_openenv_trace.py
  - examples/sdk_openenv_environment_optimization.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - AGENT_LEARNING_OPENENV_EXAMPLE_KEY=local-example agent-learn run examples/framework_openenv_manifest.json --output artifacts/framework-openenv.json
postcondition: python -c "import json; p=json.load(open('artifacts/framework-openenv.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# OpenEnv: compatibility inputs for environment replay

> **Twin:** [`examples/sdk_framework_adapter_openenv_trace.py`](../../examples/sdk_framework_adapter_openenv_trace.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

OpenEnv/Gymnasium shapes are compatibility inputs, not the product center. The kit's
owned surface is environment replay: Agent Learning Kit is the primary runtime and
release contract, and the bar is the executable `environment_10x_robustness`
release gate. OpenEnv/Gymnasium-shaped traces remain compatibility evidence inside
that bar — this page shows how such a trace enters the kit and what the kit does
with it.

The twin, [`examples/sdk_framework_adapter_openenv_trace.py`](../../examples/sdk_framework_adapter_openenv_trace.py),
builds a local `LocalOpenEnvRunner` whose export is an `openenv_trace`: an
in-process runtime with local transport (`requires_external_service: false`),
`deterministic_reset: true`, a discrete action space (`approve_refund`,
`probe_policy_drift`), a dict observation space, and initial/current observations
for a refund scenario. The adapter ingests that shape and carries it into a normal
run artifact, where the same evaluation and regression machinery applies as for
every other lane.

The failure class this addresses is unreplayable environment evidence: an
environment-shaped trace that cannot be reset deterministically or replayed locally
cannot anchor a regression. The second twin,
[`examples/sdk_openenv_environment_optimization.py`](../../examples/sdk_openenv_environment_optimization.py),
runs the environment-replay optimizer over the same compatibility shape; the
simulation-side walkthrough lives in
[`examples/sdk_openenv_environment_simulation.py`](../../examples/sdk_openenv_environment_simulation.py).
The run manifest, [`examples/framework_openenv_manifest.json`](../../examples/framework_openenv_manifest.json),
targets the twin's runner directly (`method: run`, `input_mode: dict`) on the
`local_text` engine: offline, deterministic, no provider keys.

## 2. Run it

CLI (the `required_env` key is CI metadata for this offline manifest — any
placeholder value satisfies it):

```bash
AGENT_LEARNING_OPENENV_EXAMPLE_KEY=local-example \
agent-learn run examples/framework_openenv_manifest.json \
  --output artifacts/framework-openenv.json
```

SDK, same operation (export the same placeholder env first):

```python
import asyncio
from fi.alk import simulate

result = asyncio.run(
    simulate.run_manifest_file("examples/framework_openenv_manifest.json")
)
assert result["kind"] == "agent-learning.run.v1"
```

## 3. What you built

Postcondition (machine-checkable — the same check the docs gate pattern uses):

```bash
python -c "import json; p=json.load(open('artifacts/framework-openenv.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
```

The artifact carries `status`, the simulated transcript, the evaluation report,
and the ingested `openenv_trace` — action space, observation space, reset
determinism flags, and the observation history. The environment shape is recorded
as compatibility evidence inside a kit-native run artifact: replayable, diffable,
and subject to the same gates as every other run.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `missing required environment variable(s)` | config fault | set the placeholder key shown above; `agent-learn doctor` → `summary.public_boundary_passed` confirms the install surface |
| trace lacks `deterministic_reset` or claims an external service | behavior regression | re-run the twin and compare the `openenv_trace` flags — replayability is the contract |

## 5. Prove it / keep it

The trace twin is admitted by the `framework_environment_replay_adapter_readiness`
release gate and the optimizer twin by `environment_replay_optimizer_readiness`;
the lane's overall bar is the executable `environment_10x_robustness` release gate.
Every `agent-learn release-check` re-executes these paths — the page stays true or
the release fails. To keep your own environment evidence honest, promote the run
artifact into a regression baseline with the `baseline` / `promote-to-regression` /
`compare` command family: the compatibility input is the entry point, and the
replayable kit artifact is what you maintain. The reader's job here is maintenance
of a living proof, not a one-off demo.
