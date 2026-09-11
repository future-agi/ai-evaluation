---
kind: agent-learning.docs-page.v1
track: optimize
objective: reliability
stage: optimize
backing:
  - examples/sdk_optimizer_governance_optimization.py
artifact_kinds:
  - agent-learning.optimization.v1
commands:
  - agent-learn optimize examples/optimizer_governance_optimization.json --output artifacts/governance-optimization.json
postcondition: python -c "import json; p=json.load(open('artifacts/governance-optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Optimizer Governance: the steward's veto

> **Twin:** [`examples/sdk_optimizer_governance_optimization.py`](../../examples/sdk_optimizer_governance_optimization.py)
> · emits `agent-learning.optimization.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

An optimizer that can change your agent is itself an agent acting on your
system, and it needs the same controls you demand of any actor: an audit
trail of what it changed, a veto for changes that fail policy, and a rollback
path when an accepted candidate later proves wrong. This page tests that the
governance machinery itself holds: every optimization payload the public SDK
emits passes through `with_optimization_governance` and
`with_optimization_candidate_lineage`, so acceptance decisions and candidate
ancestry are recorded in the artifact, not in someone's memory.

| Archetype | Dharma (what it may change) | Constraint (what it must preserve) |
| --- | --- | --- |
| Transformer (proposer) | candidate config: harness, memory, tooling | the task contract |
| Critic | scores + objections | evidence admissibility |
| Mediator | candidate retention/merge | lineage continuity |
| Steward (preserver) | rollback / veto | governance + the regression baseline |

This is the steward's page. Where other pages exercise the proposer and
critic, governance optimization exercises the fourth row directly: rollback,
veto, and candidate freeze are the dharma; the constraint it preserves is the
governance record and the regression baseline that every other page depends
on. The committed manifest optimizes an `optimizer_trace` environment across
`multi_agent`, `orchestration`, `planner`, `security`, and `evaluator`
layers — the candidates are governance configurations, scored on whether the
trace they produce is complete and admissible.

The audited record extends from "who proposed and who selected" to
"why, in checkable form". Every proposal carries a structured five-member
justification in its metadata — claim, reason, rule-with-example,
application, conclusion (the panca-avayava schema, a Nyaya-Sutra syllogism
structure used as an auditable record format; a scholarly design device,
not a doctrinal claim). Every rejection carries one of five closed defect
classes (`savyabhichara`, `viruddha`, `satpratipaksha`, `asiddha`,
`badhita`), and the steward's decision is a recorded nirnaya object — one
selected candidate per round with its justification and classed
rejections, never an average. The trace also declares the run's contract
properties: guna axes and two-chamber budgets, staged conditioning, layer
locality on diagnoses, the declared evaluation budget, evaluation-suite-only
ranking, and the pooled society ledger — twenty trace flags and twelve
governance checks in total, all asserted by the release gate.

## 2. Run it

CLI:

```bash
agent-learn optimize examples/optimizer_governance_optimization.json \
  --output artifacts/governance-optimization.json
```

SDK, the same operation as the twin runs it:

```python
from fi.alk import optimize

result = optimize.optimize_optimizer_governance(
    name="sdk-optimizer-governance-optimization",
    target_metadata={"cookbook": "sdk-optimizer-governance"},
)
```

A simulation-side counterpart exists at
[`examples/sdk_optimizer_governance_simulation.py`](../../examples/sdk_optimizer_governance_simulation.py)
if you want to observe the governed run before optimizing it.

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/governance-optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
```

Inside the artifact, the governance block records how each candidate was
admitted or rejected, and the lineage block ties every surviving config to
the proposal that produced it. Audit means you can answer "why is this the
config" from the artifact alone.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| manifest rejected | config fault | `summary.public_boundary_passed` + the manifest error line |
| governance block missing fields | engine drift | rerun against a clean checkout; compare lineage blocks |

## 5. Prove it / keep it

Freeze the accepted candidate by promoting the result into a regression
manifest (`agent-learn promote-to-regression`) so any future optimizer run
that would silently overturn it fails replay first — the full pattern is
[`optimization-lifecycle.md`](./optimization-lifecycle.md). To see which
optimizer backend should be trusted with a given target at all, continue to
[`optimizer-portfolio.md`](./optimizer-portfolio.md).
