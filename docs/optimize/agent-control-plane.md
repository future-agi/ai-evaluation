---
kind: agent-learning.docs-page.v1
track: optimize
objective: safety
stage: optimize
backing:
  - examples/sdk_agent_control_plane_optimization.py
  - examples/sdk_agent_control_plane_simulation.py
artifact_kinds:
  - agent-learning.optimization.v1
commands:
  - agent-learn optimize examples/agent_control_plane_optimization.json --output artifacts/agent-control-plane-optimization.json
postcondition: python -c "import json; p=json.load(open('artifacts/agent-control-plane-optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Agent Control Plane: optimizing trust boundaries and autonomy

> **Twin:** [`examples/sdk_agent_control_plane_optimization.py`](../../examples/sdk_agent_control_plane_optimization.py)
> · emits `agent-learning.optimization.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

The control plane is the part of the agent that decides what the agent may
do: trust boundaries, policy enforcement, autonomy limits. The committed
manifest
([`examples/agent_control_plane_optimization.json`](../../examples/agent_control_plane_optimization.json))
names its target "agent-learning-trust-and-control-plane" and searches
candidates under `simulation.environments` in an `agent_trust_boundary`
environment, across the `security`, `policy`, `autonomy`, and `evaluator`
layers. The failure class: an agent whose capabilities are correct but whose
permission surface is wrong — too much autonomy in the wrong place, or a
trust boundary that exists in documentation and not in the run.

| Archetype | Dharma (what it may change) | Constraint (what it must preserve) |
| --- | --- | --- |
| Transformer (proposer) | candidate config: harness, memory, tooling | the task contract |
| Critic | scores + objections | evidence admissibility |
| Mediator | candidate retention/merge | lineage continuity |
| Steward (preserver) | rollback / veto | governance + the regression baseline |

The control plane is where the steward's column becomes runtime config:
rollback and veto are not properties of the optimization process here — they
are the candidate values being optimized. A control-plane candidate is
accepted only when the trust-boundary environment scores it as both
permitting the task and refusing the overreach.

## 2. Run it

CLI:

```bash
agent-learn optimize examples/agent_control_plane_optimization.json \
  --output artifacts/agent-control-plane-optimization.json
```

SDK — optimization, and the simulation twin to observe the control plane
before optimizing it:

```python
from fi.alk import optimize, simulate

result = optimize.optimize_agent_control_plane(
    name="sdk-agent-control-plane-optimization",
    target_metadata={"cookbook": "sdk-agent-control-plane-optimization"},
)

manifest = simulate.build_agent_control_plane_run_manifest(
    name="sdk-agent-control-plane-simulation",
)
```

Both backing examples are executed by the same release gate
(`agent_control_plane_readiness`), so the surface this page teaches is
re-verified on every release check.

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/agent-control-plane-optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
```

The artifact records each control-plane candidate with its trust-boundary
score, plus the governance and lineage blocks every optimization payload
carries — the permission surface you ship is the one the artifact proves.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| manifest rejected | config fault | `summary.public_boundary_passed` + the manifest error line |
| best candidate still over-permissive | candidate fault | tighten the autonomy candidates under `simulation.environments` |

## 5. Prove it / keep it

A trust boundary that passed once must keep passing: promote the result into
a regression manifest and replay it in CI
([`optimization-lifecycle.md`](./optimization-lifecycle.md)). The
adversarial counterpart — actively attacking the boundary rather than
optimizing it — lives in the red-team track, and the trust-certificate story
for shipping this evidence is in the prove track.
