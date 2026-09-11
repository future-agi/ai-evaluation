---
kind: agent-learning.docs-page.v1
track: simulate
objective: capability
stage: simulate
backing:
  - examples/sdk_cua_loop.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - python examples/sdk_cua_loop.py artifacts/cua-loop.json
postcondition: python -c "import json; p=json.load(open('artifacts/cua-loop.json')); assert p['kind']=='agent-learning.cua-loop.v1', p['kind']; print('ok')"
claims:
  - phrase: fake-completion-guard
    gate_id: cua_loop_readiness
  - phrase: fake-completion
    gate_id: cua_loop_readiness
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: true
---

# CUA loop: the deterministic browser/computer-use substrate, credential-free

> **Twin:** [`examples/sdk_cua_loop.py`](../../examples/sdk_cua_loop.py)
> · emits `agent-learning.run.v1` · offline, no credentials, deterministic, no
> real browser, no VM. A coding agent can complete this page from the frontmatter
> alone.

This is the CUA / browser / computer-use loop substrate. It runs **entirely
in-process** on committed synthetic-DOM fixtures (a multi-step form flow, a
selector-drift family, an injected-DOM family, a fake-completion sentinel set, a
desktop episode) — no network, no keys, no model, no real browser, no VM — over
the already-shipped `BrowserEnvironment` and the 7-dimension deterministic
`score_browser_cua_probe_result` verifier. Same seed in, byte-identical loop
trajectory out.

**Honesty disclaimer (load-bearing).** A deterministic in-process fixture is
**NOT a live lane**. Every deterministic artifact carries
`fidelity_tier: "deterministic_fixture"` and an evidence class of `local_gate`
(or `captured_fixture` when stored) — **never `live_lane`**. `live_lane` is
reserved for the keyed real-browser/VM lane. The gate fails any deterministic
fixture that claims `live_lane` (the `cua_fidelity_overclaim` tripwire).

## 1. What you are testing

CUA agents fail on perception, grounding, and action policy, not just on words: a
stale screenshot, a drifted selector, a step loop, a touched injection banner.
The loop exercises exactly those signals deterministically. `browser` and
`computer_use` are already frozen world kinds — 9C flips their **executable-loop**
status through the R4 registry hook, recording the executable-loop evidence
WITHOUT widening the frozen `SIMULATION_WORLD_KINDS` tuple (the byte-pin stays
green). It is "typed → executable": typed the moment it is a frozen member,
executable the moment its rung-1 fixture run is green, never silently claimed.

## 2. Run it

```bash
python examples/sdk_cua_loop.py artifacts/cua-loop.json
```

SDK (the operation the twin performs):

```python
from fi.alk import cua_loop
from fi.simulate.environment import BrowserEnvironment
from fi.simulate.simulation import contract

cua_loop._ensure_cua_world_registered("browser")
assert "browser" in contract.resolved_world_kinds()
env = BrowserEnvironment(url="https://shop.example.test/checkout/step-1")
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/cua-loop.json')); assert p['kind']=='agent-learning.cua-loop.v1', p['kind']; print('ok')"
```

The artifact holds the loop-determinism proof (byte-identical trajectory + the
mutation-pack stressed runs under the pinned seed), the deterministic anchors
(`state_quality` / `action_quality` / `mutation_grounding_quality` reproducible
over the fixtures, plus the desktop `grounding_step_accuracy`), the
fake-completion-guard outcome (a sentinel that **narrates** success while
`state_match` is flat is score-zeroed on the anchor), the unsafe-completion canary
(an injected-DOM-following config is zeroed on `action_quality`), and the
constructed overclaim negatives the gate must catch.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `cua_fixture_missing` | config fault | a committed JSON fixture path is missing/unreadable |
| `cua_fidelity_overclaim` in the gate | overclaim | a deterministic fixture stamped `evidence_class: live_lane` |
| public boundary error | config fault | `summary.public_boundary_passed` |

## 5. Prove it / keep it

The twin runs fresh on every `agent-learn release-check` via the
`cua_loop_readiness` gate (eight evidence arrays, all credential-free). The
fake-completion-guard outcome is honest by computation: it is licensed only while
that gate is green. For the keyed real-browser/VM lane (the only honest
`live_lane`) and the desktop full-post-state rungs, see the roadmap — they are
owner-keyed/infra opt-in lanes, never a release prerequisite. To tune the whole
CUA agent against these signals, see [cua-improvement](../optimize/cua-improvement.md).
