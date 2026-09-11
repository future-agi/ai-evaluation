---
kind: agent-learning.docs-page.v1
track: optimize
objective: reliability
stage: optimize
backing:
  - examples/sdk_cua_improvement.py
artifact_kinds:
  - agent-learning.practice-report.v1
  - agent-learning.run.v1
commands:
  - python examples/sdk_cua_improvement.py artifacts/cua-improvement.json
postcondition: python -c "import json; p=json.load(open('artifacts/cua-improvement.json')); assert p['kind']=='agent-learning.cua-improvement.v1', p['kind']; print('ok')"
claims:
  - phrase: CUA improvement loop
    gate_id: cua_loop_readiness
  - phrase: fake-completion
    gate_id: cua_loop_readiness
  - phrase: trainer
    gate_id: practice_loop_readiness
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: true
---

# CUA improvement loop: the 13D Practice Loop on browser/computer-use

> **Twin:** [`examples/sdk_cua_improvement.py`](../../examples/sdk_cua_improvement.py)
> · emits `agent-learning.practice-report.v1` + `agent-learning.run.v1` · offline,
> no credentials, deterministic. A coding agent can complete this page from the
> frontmatter alone.

This wires the CUA-task evals as a **loss** and runs the generic 13D Practice Loop
on `world.kind=browser` (and `computer_use` for the desktop surface). No new
optimizer is invented — the existing six-phase trainer runs over CUA cells. The
loss is **multi-objective and deterministic-post-state-anchored**: every declared
CUA objective MUST carry at least one deterministic post-state anchor
(`task_success` / `state_match` on browser, `grounding_step_accuracy` on desktop).
A judge-only CUA objective is structurally rejected — there is no judge-only CUA
loss.

**Honesty disclaimer (load-bearing).** The deterministic loss runs credential-free
and stays `local_gate` / `captured_fixture` (a `deterministic_fixture` artifact,
never `live_lane`). The keyed `completion_judge` term, the entire desktop
full-post-state rungs, and the live browser/VM lane are opt-in keyed/infra lanes,
never a release prerequisite.

## 1. What you are tuning

The SCOPED UPDATE optimizes the **whole CUA agent** (`target_kind=whole_agent`):
model, instructions, tool routing — plus the config-only knobs no text optimizer
reaches: **grounding / action policy** (`agent.grounding.*`, the observe→ground
seam), **observation-resolution / escalation policy** (`agent.observe.*` /
`agent.escalation.*`), and **reflection / memory** (`agent.reflection.*` /
`agent.memory.*`). The loss carries a mandatory fake/unsafe-completion Goodhart
guard: a fake-completion sentinel (narrated success while the deterministic
post-state is flat) is score-zeroed on the anchor; an unsafe-completion canary (a
config that "completes" only by touching a prompt-injection surface) is zeroed on
`action_quality` — that is the tell.

## 2. Run it

```bash
python examples/sdk_cua_improvement.py artifacts/cua-improvement.json
```

SDK (the operation the twin performs):

```python
from fi.alk import cua_loop

manifest = cua_loop.build_cua_practice_loop_manifest(
    name="cua-improvement",
    base_agent={"model": "gpt-4o"},
    search_space={
        "agent.grounding.mode": ["element-id", "coordinate", "selector"],
        "agent.observe.channel": ["screenshot", "DOM", "AXTree"],
        "agent.reflection.postmortems": ["on", "off"],
    },
    objective=objective,  # multi-objective, >= 1 deterministic anchor, guarded
    eval_budget=6, seed=1142,
)
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/cua-improvement.json')); assert p['kind']=='agent-learning.cua-improvement.v1', p['kind']; print('ok')"
```

The artifact holds the compiled multi-objective guarded loss, the constructed
judge-only / single-term / missing-anchor rejections, the whole-agent search space
(incl. `agent.grounding.*` + `agent.observe.*`/`agent.escalation.*` +
`agent.reflection.*`/`agent.memory.*`), the loop-vs-no-loop A/B at equal budget
(the held-out-battery capstone with the fake/unsafe-completion canaries holding),
and the CUA-sublayer attribution per weak cell (`perception` / `grounding` /
`action_policy` / `reasoning_memory`).

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `cua_loss_guard_missing` | config fault | a judge-only / single-term / missing-anchor CUA objective |
| `objective_guards_missing` | config fault | a declared loss with no Goodhart guards |
| public boundary error | config fault | `summary.public_boundary_passed` |

## 5. Prove it / keep it

The twin runs fresh on every `agent-learn release-check` via the
`cua_loop_readiness` gate. The A/B capstone is the credential-free proof that the
loop beats no-loop on a held-out CUA battery with the fake/unsafe-completion
canaries holding. For the keyed real-browser/VM lane, the keyed `completion_judge`
term, and the desktop full-post-state rungs, see the roadmap — all are
owner-keyed/infra opt-in lanes, never release prerequisites. To inspect the
deterministic substrate, see [cua-loop](../simulate/cua-loop.md).
