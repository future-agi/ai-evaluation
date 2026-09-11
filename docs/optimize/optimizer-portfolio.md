---
kind: agent-learning.docs-page.v1
track: optimize
objective: capability
stage: optimize
backing:
  - examples/sdk_optimizer_portfolio_optimization.py
artifact_kinds:
  - agent-learning.optimization.v1
commands:
  - AGENT_LEARNING_SDK_OPTIMIZER_PORTFOLIO_KEY=local-dev-key python examples/sdk_optimizer_portfolio_optimization.py artifacts/optimizer-portfolio.json
postcondition: python -c "import json; p=json.load(open('artifacts/optimizer-portfolio.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - api_key_configured
opt_in_lane: false
---

# Optimizer Portfolio: choosing the optimizer with evidence

> **Twin:** [`examples/sdk_optimizer_portfolio_optimization.py`](../../examples/sdk_optimizer_portfolio_optimization.py)
> · emits `agent-learning.optimization.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

Which optimizer backend should optimize this target? Picking one by habit is
itself an unverified config choice. The portfolio run treats optimizer
backends as candidates: it executes the agent-search, TPE, and bandit
backends against the same `optimizer_backend_portfolio` environment, requires
all of them to complete, demands consensus evidence between them, and records
which backend was selected and why. The failure class is silent
mis-allocation — an expensive search backend assigned to a target a simpler
one already solves, or a backend whose results cannot be reconciled with the
others.

| Archetype | Dharma (what it may change) | Constraint (what it must preserve) |
| --- | --- | --- |
| Transformer (proposer) | candidate config: harness, memory, tooling | the task contract |
| Critic | scores + objections | evidence admissibility |
| Mediator | candidate retention/merge | lineage continuity |
| Steward (preserver) | rollback / veto | governance + the regression baseline |

In the portfolio, the archetypes operate one level up: each backend is a
proposer, the cross-backend consensus check is the critic, the selection rule
mediates between backend results, and the portfolio proof block stewards the
allocation decision into the artifact.

## 2. Run it

CLI — the example is the executable surface; the env value is a local
placeholder (the agents are scripted; nothing leaves the machine):

```bash
AGENT_LEARNING_SDK_OPTIMIZER_PORTFOLIO_KEY=local-dev-key \
  python examples/sdk_optimizer_portfolio_optimization.py \
  artifacts/optimizer-portfolio.json
```

SDK, the same operation:

```python
from fi.alk import optimize

result = optimize.optimize_optimizer_portfolio(
    name="sdk-optimizer-portfolio-optimization",
    target_metadata={"cookbook": "sdk-optimizer-portfolio-optimization"},
)
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/optimizer-portfolio.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
```

The release gate that covers this twin pins the contract the artifact must
satisfy: completed backends `agent`, `tpe`, and `bandit`; a
`backend_consensus` dependency; a recorded `selected_optimizer`; and an
attached portfolio proof
(`agent-learning.optimization.optimizer-portfolio-proof.v1`). The artifact is
the allocation decision plus the evidence for it.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `Set AGENT_LEARNING_SDK_OPTIMIZER_PORTFOLIO_KEY...` | missing placeholder env | `agent-learn doctor` → `summary.api_key_configured` |
| `vendored import failed` | infra | `summary.missing_engine_modules` |
| a backend missing from completed set | engine fault | inspect the portfolio proof block in the artifact |

## 5. Prove it / keep it

The selection only stays trustworthy while the consensus evidence holds:
re-run the portfolio when the target changes shape, and keep the previous
artifact as the comparison point (`agent-learn baseline` / `compare`). The
acceptance rules the portfolio operates under are the subject of
[`governance.md`](./governance.md); the lifecycle that locks the chosen
backend's results into CI is
[`optimization-lifecycle.md`](./optimization-lifecycle.md).
