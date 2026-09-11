---
kind: agent-learning.docs-page.v1
track: optimize
objective: reliability
stage: optimize
backing:
  - examples/sdk_voice_improvement.py
artifact_kinds:
  - agent-learning.practice-report.v1
  - agent-learning.run.v1
commands:
  - python examples/sdk_voice_improvement.py artifacts/voice-improvement.json
postcondition: python -c "import json; p=json.load(open('artifacts/voice-improvement.json')); assert p['kind']=='agent-learning.voice-improvement.v1', p['kind']; print('ok')"
claims:
  - phrase: codec-survival
    gate_id: voice_loopback_readiness
  - phrase: trainer
    gate_id: practice_loop_readiness
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: true
---

# Voice improvement loop: the 13D Practice Loop on voice

> **Twin:** [`examples/sdk_voice_improvement.py`](../../examples/sdk_voice_improvement.py)
> · emits the practice-loop manifest · offline, deterministic, no credentials.
> A coding agent can complete this page from the frontmatter alone.

The voice improvement loop is the existing 13D Practice Loop instantiated on
`world.kind = voice_telephony`. No new optimizer is invented: the same
six-phase trainer, the same `base_agent` + `search_space` whole-agent contract,
the same A/B experiment engine. What is new is that **voice-quality evals are
the per-cell loss** and the **whole voice agent is the search space**.

## 1. What you are testing

A text-only loop cannot see a mis-heard tool argument, a barge-in the agent
ignored, or a claim that died through the codec-survival channel. The voice loss
is multi-objective by construction (a single timing term is reward-hackable, so
it is structurally rejected) and carries a mandatory Goodhart guard — the
unedited loss-channel enforcement, with no override. The search space spans the
whole agent: `voice.id`, `voice.tts.rate`, `agent.first_message`,
`voice.endpointing.threshold`, `voice.barge_in.policy`, `agent.instructions`,
`agent.tools.routing` — not prompt-only.

## 2. Run it

```bash
python examples/sdk_voice_improvement.py artifacts/voice-improvement.json
```

SDK (the operation the twin performs):

```python
from fi.alk import voice_loop

manifest = voice_loop.build_voice_practice_loop_manifest(
    name="voice-improvement",
    base_agent={"model": "gpt-4o", "voice": {"id": "alloy"}},
    search_space={"voice.id": ["alloy", "shimmer"], "voice.tts.rate": [0.9, 1.0]},
    objective=objective,   # multi-objective + guard (single-timing is rejected)
    eval_budget=4,
    seed=1142,
)
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/voice-improvement.json')); assert p['kind']=='agent-learning.voice-improvement.v1', p['kind']; print('ok')"
```

The artifact records the compiled multi-objective voice loss, the constructed
single-timing rejection, the whole-agent voice search space, the loop-vs-no-loop
A/B at equal budget, and the voice sub-attribution
(`acoustic_codec`/`asr_mishear`/`llm`/`tts_endpointing`) stamped alongside the
base failure layer on each weak cell.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `voice_loss_guard_missing` | config fault | a single-timing or guardless voice objective |
| `objective_guards_missing` | config fault | the loss has no sentinel/canary guard |
| public boundary error | config fault | `summary.public_boundary_passed` |

## 5. Prove it / keep it

The twin runs fresh on every `agent-learn release-check` via the
`voice_loopback_readiness` gate. The codec-survival term is honest by
computation; the loop-beats-no-loop claim is the equal-budget A/B, not a vibe.
The audio channel this loop optimizes against is the deterministic rung-2
loopback ([voice-loopback](../simulate/voice-loopback.md)) — credential-free and
byte-reproducible.
