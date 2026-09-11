---
kind: agent-learning.docs-page.v1
track: optimize
objective: reliability
stage: promote
backing:
  - examples/sdk_capability_freeze_regression.py
artifact_kinds:
  - agent-learning.optimization.v1
  - agent-learning.frozen-capability-profile.v1
commands:
  - AGENT_LEARNING_SDK_CAPABILITY_FREEZE_EXAMPLE_KEY=local-dev-key python examples/sdk_capability_freeze_regression.py artifacts/capability-freeze-regression.json
postcondition: python -c "import json; p=json.load(open('artifacts/capability-freeze-regression.json')); assert p['frozen']['kind']=='agent-learning.frozen-capability-profile.v1', p['frozen']['kind']; assert p['replays']['improving_but_breaking']['veto'] is True; assert p['replays']['improving_but_breaking']['hetvabhasa_class']=='badhita'; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - api_key_configured
opt_in_lane: false
---

# Capability-Profile Freezing: the frozen rows a winner must not break

> **Twin:** [`examples/sdk_capability_freeze_regression.py`](../../examples/sdk_capability_freeze_regression.py)
> · emits `agent-learning.optimization.v1` evidence embedding an
> `agent-learning.frozen-capability-profile.v1` contract · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

An optimizer that improves the metric you searched can silently regress a
capability you already shipped on. Capability profiles describe what an
adapter can do today; freezing turns that description into an evidence
contract: rows of `{framework, capability, metric, floor, setting,
security, source}`, each content-addressed (`row_id` is the sha256 of the
sorted JSON of the other fields) under one `contract_digest`. A later
promotion must re-close every frozen row — an improving candidate that
breaks one row is vetoed, and the veto is recorded.

The failure class is silent capability regression. Three rules make the
contract executable: a broken row defeats the win (`badhita` — overridden by
stronger admissible evidence); a win measured under a different declared
setting is recorded as non-admissible and never counts (orderings invert
across settings); and rows with `security: true` — derived from
stored-injection red-team checks — are non-tradable: any candidate touching
context-memory paths must re-pass them at floor, regardless of score.

| Archetype | Dharma (what it may change) | Constraint (what it must preserve) |
| --- | --- | --- |
| Transformer (proposer) | candidate config: harness, memory, tooling | the task contract |
| Critic | scores + objections | evidence admissibility |
| Mediator | candidate retention/merge | lineage continuity |
| Steward (preserver) | rollback / veto | governance + the regression baseline |

Freezing is the steward's row made durable: the veto fires from frozen
evidence, not taste, and lands in the governance record as a nirnaya entry
citing the broken `row_id`s.

## 2. Run it

CLI — the env value is a local placeholder (deterministic fixtures, nothing
leaves the machine):

```bash
AGENT_LEARNING_SDK_CAPABILITY_FREEZE_EXAMPLE_KEY=local-dev-key \
  python examples/sdk_capability_freeze_regression.py \
  artifacts/capability-freeze-regression.json
```

SDK, the freeze → attach → replay loop:

```python
from fi.alk import optimize, simulate

profiles = simulate.framework_adapter_capability_profiles(
    frameworks=["langgraph", "livekit"],
)
frozen = optimize.freeze_capability_profile(
    profiles,
    setting={"engine": "local_text", "driver": "deterministic_scripted"},
    metric_floors={"task_completion": 0.9},
    security_rows=[{"metric": "redteam_pass_rate", "floor": 1.0}],
)
promotion = optimize.attach_frozen_profile(promotion_artifact, frozen)
verdict = optimize.replay_frozen_profile(candidate_result, frozen)
assert verdict["veto"] is False  # every row re-closed
```

The CLI lifecycle gains the veto step with
`build_optimization_lifecycle_plan(frozen_profile_path=...)` — the
`replay_frozen_profile` step runs between promotion and regression replay.

## 3. What you built

```bash
python -c "import json; p=json.load(open('artifacts/capability-freeze-regression.json')); assert p['frozen']['kind']=='agent-learning.frozen-capability-profile.v1', p['frozen']['kind']; assert p['replays']['improving_but_breaking']['veto'] is True; assert p['replays']['improving_but_breaking']['hetvabhasa_class']=='badhita'; print('ok')"
```

The artifact carries the frozen contract (`frozen`), the committed-fixture
match (`fixture.match`), and five replay verdicts: a compliant candidate
(all rows re-closed), the improving-but-row-breaking candidate (vetoed,
`badhita`), an out-of-setting win (non-admissible), a security-row trade
(vetoed regardless of score), and a tampered row (content-address mismatch
detected as `asiddha`).

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `Set AGENT_LEARNING_SDK_CAPABILITY_FREEZE_EXAMPLE_KEY...` | missing local placeholder env | `api_key_configured` |
| `fixture.match` is false | capability profiles drifted from the committed fixture — refreeze deliberately | `missing_engine_modules` |
| compliant replay vetoed | a frozen floor is no longer reachable under the declared setting | `missing_engine_modules` |
| `ModuleNotFoundError: fi.simulate` | simulate engine not installed | `missing_public_modules` |

## 5. Prove it / keep it

`agent-learn release-check --project-root .` proves the loop in the
`capability_profile_freeze_readiness` gate: freeze, replay, veto, recorded
nirnaya, non-admissible out-of-setting wins, non-tradable security rows.
Promotion and replay mechanics live in
[The Optimization Lifecycle](optimization-lifecycle.md); the cells that
produce candidates worth freezing against live in the
[Optimizer Profile Matrix](optimizer-profile-matrix.md).
