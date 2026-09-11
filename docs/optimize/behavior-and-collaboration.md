---
kind: agent-learning.docs-page.v1
track: optimize
objective: behavior
stage: optimize
backing:
  - examples/sdk_multi_agent_room_probe_optimization.py
artifact_kinds:
  - agent-learning.optimization.v1
commands:
  - python examples/sdk_multi_agent_room_probe_optimization.py artifacts/room-probe-optimization.json
  - AGENT_LEARNING_SDK_BEHAVIOR_ENTROPY_KEY=local-dev-key python examples/sdk_behavior_entropy_optimization.py artifacts/behavior-entropy-optimization.json
  - AGENT_LEARNING_SDK_COLLABORATIVE_COMPETENCE_KEY=local-dev-key python examples/sdk_collaborative_competence_optimization.py artifacts/collaborative-competence-optimization.json
postcondition: python -c "import json; p=json.load(open('artifacts/room-probe-optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - api_key_configured
opt_in_lane: false
---

# Behavior and Collaboration: optimizing how agents act together

> **Twin:** [`examples/sdk_multi_agent_room_probe_optimization.py`](../../examples/sdk_multi_agent_room_probe_optimization.py)
> · emits `agent-learning.optimization.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

Two behavioral failure classes that ordinary task metrics miss. First,
behavioral collapse: the weak agent in
[`examples/sdk_behavior_entropy_optimization.py`](../../examples/sdk_behavior_entropy_optimization.py)
calls the same `search_policy` tool with the same `refund` query turn after
turn — every call succeeds, no progress is made. Second, collaborative
incompetence: the weak agent in
[`examples/sdk_collaborative_competence_optimization.py`](../../examples/sdk_collaborative_competence_optimization.py)
declares "I will approve the refund by myself", never models partner intent,
never updates shared state, and ships an unreviewed decision past a room that
contains a planner, a retriever, and a critic. Both agents look busy; both
fail the task as a social act.

The probe twin scores these dynamics directly: it takes weak/strong agent
candidates and weak/strong room candidates and optimizes over both at once,
so the artifact shows whether the deficit lives in the agent's behavior or in
the room's structure.

| Archetype | Dharma (what it may change) | Constraint (what it must preserve) |
| --- | --- | --- |
| Transformer (proposer) | candidate config: harness, memory, tooling | the task contract |
| Critic | scores + objections | evidence admissibility |
| Mediator | candidate retention/merge | lineage continuity |
| Steward (preserver) | rollback / veto | governance + the regression baseline |

Behavioral diversity and collaboration are critic's-row concerns: the
objection "you are repeating yourself" or "you skipped review" is exactly the
score-plus-objection evidence the table licenses, and acceptance still rests
on the metric, not on the role label
([`society-of-agents.md`](./society-of-agents.md)).

## 2. Run it

CLI — the probe needs no env; the two full optimizations take local
placeholder keys (scripted agents, nothing leaves the machine):

```bash
python examples/sdk_multi_agent_room_probe_optimization.py \
  artifacts/room-probe-optimization.json

AGENT_LEARNING_SDK_BEHAVIOR_ENTROPY_KEY=local-dev-key \
  python examples/sdk_behavior_entropy_optimization.py \
  artifacts/behavior-entropy-optimization.json
```

SDK, the probe operation:

```python
from fi.alk import optimize

result = optimize.optimize_multi_agent_room_probe(
    name="sdk-multi-agent-room-probe-optimization",
    participants=participants,                       # planner / retriever / critic
    agent_candidates=[weak_agent, strong_agent],
    room_candidates=[weak_room, strong_room],
)
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/room-probe-optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
```

The probe artifact carries a multi-agent room proof block plus candidate
lineage across both axes (agent x room). The two full-optimization artifacts
record which behavioral profile won and why the repetitive and solo agents
scored low.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `Set AGENT_LEARNING_SDK_BEHAVIOR_ENTROPY_KEY...` | missing placeholder env | `summary.api_key_configured` |
| weak and strong agents tie | evaluator fault | check the behavior checks in the evaluation config |

## 5. Prove it / keep it

Promote the strong agent-room pairing into a regression manifest so a future
"simplification" that removes review or collapses behavior fails replay
([`optimization-lifecycle.md`](./optimization-lifecycle.md)). Room
composition as an explicit target path is
[`multi-agent-targets.md`](./multi-agent-targets.md); the same dynamics under
plain simulation are `../simulate/multi-agent.md`.
