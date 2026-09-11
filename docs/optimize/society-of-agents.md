---
kind: agent-learning.docs-page.v1
track: optimize
objective: safety
stage: optimize
backing:
  - examples/sdk_redteam_society_optimization.py
artifact_kinds:
  - agent-learning.optimization.v1
commands:
  - agent-learn optimize examples/redteam_society_optimization.json --output artifacts/society-optimization.json
postcondition: python -c "import json; p=json.load(open('artifacts/society-optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Society of Agents: optimization as a sabha

> **Twin:** [`examples/sdk_redteam_society_optimization.py`](../../examples/sdk_redteam_society_optimization.py)
> · emits `agent-learning.optimization.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

A sabha is a deliberative assembly: proposals are made, objected to,
reconciled, and only then ratified. The kit's society optimizer
(`SocietyAgentOptimizer` in `src/fi/opt/optimizers/council.py`, extending
`CouncilAgentOptimizer`) runs optimization the same way — deterministic
multi-round social search over the whole agent: harness, memory, tooling,
orchestration, never just the prompt. The single-optimizer failure class it
addresses is greedy collapse: one proposal stream converging on a local fix
that repairs one path while breaking another.

| Archetype | Dharma (what it may change) | Constraint (what it must preserve) |
| --- | --- | --- |
| Transformer (proposer) | candidate config: harness, memory, tooling | the task contract |
| Critic | scores + objections | evidence admissibility |
| Mediator | candidate retention/merge | lineage continuity |
| Steward (preserver) | rollback / veto | governance + the regression baseline |

The society's default role graph instantiates each archetype as concrete
proposal roles: specialists bundle repairs by path prefix (`sutradhara` for
orchestration and routing, `smriti` for memory and retrieval, `hanuman` for
tools and framework wiring), an explorer (`arjuna`) probes one controllable
path at a time, an adversary (`vidura`) stresses policy and trust-boundary
choices, a phase-two critic (`krishna`) tests one more change against the
strong partial candidates, a synthesis role (`sangha`) merges the best path
representatives, and a `dharma_steward` removes one change at a time, keeping
only metric-proven repairs. The names are a design lineage, used with
respect: the code states that role names and archetypes are inspiration
labels only — candidate acceptance depends entirely on the metric and
evaluator contract.

Why multiple roles instead of one judge? The judge-reliability evidence
([`../eval/judge-reliability.md`](../eval/judge-reliability.md)) shows
single-judge scores shift under formatting, verbosity, and paraphrase
perturbations of the same content. The sabha is the structural answer:
proposal, objection, synthesis, and pruning are separated into distinct
roles, and a change survives only if its score holds across rounds — no
single voice ratifies its own proposal.

The assembly's temperament and procedure are declared data. Each
role carries an optional guna mapping — `{rajas, sattva, tamas}`, each in
[0, 1], derived from the role's archetype when absent — with deterministic
mechanical effects: rajas scales mutation radius and proposal-slot share,
sattva scales synthesis breadth and reconciliation, tamas scales steward
simplification aggressiveness. Roles also seat in one of two chambers,
orthogonal to phases: the samiti generates widely and cheaply, the sabha
deliberates narrowly and promotes, each chamber under its own declared
evaluation budget. The values are psychometrically grounded design devices
(the Triguna research lineage), used as deterministic metadata — candidate
acceptance remains entirely metric-based. Per-round diagnosis pooling is
also declared: the society ledger pools every evaluated candidate's
diagnoses (not just the winner's) and hands them to every role next round,
persisting across campaigns through the social-memory store.

## 2. Run it

CLI, against the committed society manifest (scripted multi-agent room,
search over `simulation.environments`, threshold 0.9, agent-search
algorithm):

```bash
agent-learn optimize examples/redteam_society_optimization.json \
  --output artifacts/society-optimization.json
```

SDK, the same operation as the twin runs it:

```python
from fi.alk import optimize

result = optimize.optimize_redteam_society(
    name="sdk-redteam-society-optimization",
    target_metadata={"cookbook": "sdk-redteam-society-optimization"},
)
```

The manifest's target spans `security`, `multi_agent`, `orchestration`,
`memory`, and `evaluator` layers — the deliberation ranges over the whole
agent stack.

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/society-optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
```

The artifact's candidate-lineage block records which role proposed each
candidate (`proposal_role`, `proposal_round` in candidate metadata), so the
deliberation itself is auditable after the fact.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| manifest rejected | config fault | `summary.public_boundary_passed` + the manifest error line |
| no candidate beats the seed | search-space fault | widen the candidate lists under `simulation.environments` |

## 5. Prove it / keep it

The steward's pruning is only durable if it lands in the regression baseline:
promote the result with `agent-learn promote-to-regression` and replay it per
[`optimization-lifecycle.md`](./optimization-lifecycle.md). For the rules
that govern acceptance, veto, and rollback across any optimizer, continue to
[`governance.md`](./governance.md).
