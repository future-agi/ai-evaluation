---
kind: agent-learning.docs-page.v1
track: simulate
objective: behavior
stage: simulate
backing:
  - examples/sdk_multi_agent_room_probe_optimization.py
  - examples/sdk_framework_adapter_handoff_transcript.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - python examples/sdk_multi_agent_room_probe_optimization.py artifacts/multi-agent-room-probe.json
  - python examples/sdk_framework_adapter_handoff_transcript.py artifacts/handoff-transcript.json
postcondition: python -c "import json; p=json.load(open('artifacts/multi-agent-room-probe.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Multi-Agent: rooms, handoffs, and coordination evidence

> **Twin:** [`examples/sdk_multi_agent_room_probe_optimization.py`](../../examples/sdk_multi_agent_room_probe_optimization.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

Single-agent simulation asks "did the agent do the right thing". Multi-agent
simulation has to ask a second question: "did the *team* coordinate, or did
one agent quietly do everything while the others decorated the transcript".
The kit makes coordination a first-class, checkable trace rather than a
vibe.

The room probe twin sets up a room with participants and two axes of
candidates — weak/strong agents and weak/strong room configurations — runs
`optimize.optimize_multi_agent_room_probe` over them, then promotes the
winning combination into a run manifest with
`build_multi_agent_run_manifest_from_probe_optimization` and executes it.
The artifact you get is therefore not just a selection record: it is the
promoted room actually running, with the evaluation config applied.

The handoff transcript example covers the other dominant multi-agent shape:
sequential delegation. Its `LocalHandoffTeam` (an `openai_agents`-style
shim) emits a typed transcript — `HandoffMessage` (triage → retrieval →
critic, each with an explicit task), a `ReviewMessage` with a
`review_status`, a `ReconciliationMessage` naming the `accepted_source`,
and a `FinalMessage` — ending in `stop_reason: completed`. The weak path
returns an answer "without coordination evidence". The failure class this
catches: chains that skip review, reconciliations that never name a source,
final answers produced before the handoff chain closed.

## 2. Run it

CLI (no env required — both engines are local and deterministic):

```bash
python examples/sdk_multi_agent_room_probe_optimization.py artifacts/multi-agent-room-probe.json

python examples/sdk_framework_adapter_handoff_transcript.py artifacts/handoff-transcript.json
```

SDK (the probe-then-promote operation the twin performs):

```python
import asyncio
from fi.alk import optimize, simulate

probe = optimize.optimize_multi_agent_room_probe(
    name="room-probe",
    participants=[...],          # who is in the room
    agent_candidates=[...],      # weak/strong agent configs
    room_candidates=[...],       # weak/strong room configs
)
manifest = optimize.build_multi_agent_run_manifest_from_probe_optimization(probe)
simulate.write_manifest_file(manifest, "room-run.manifest.json")
result = asyncio.run(simulate.run_manifest_file("room-run.manifest.json"))
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/multi-agent-room-probe.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
```

The room artifact records which agent/room pairing the probe selected and
the promoted run's case evidence. The handoff artifact carries the full
typed transcript — every handoff, review, and reconciliation message — as
framework trace evidence the evaluator scored.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| adapter target / manifest rejected | config fault | `summary.public_boundary_passed` + the manifest error line |
| run passes but coordination score is low | transcript lacks review/reconciliation messages | read the typed messages in the artifact's case record |

## 5. Prove it / keep it

Both backing examples are re-proven on every `agent-learn release-check`
(`multi_agent_room_probe_readiness` and `framework_adapter_probe_readiness`
gates). Next steps along the spine: orchestration graphs that route between
agents are [`orchestration.md`](orchestration.md); once your team's run
passes, baseline it and wire the
[`regression-lifecycle.md`](regression-lifecycle.md) comparison into CI so a
later prompt edit cannot silently un-coordinate the team.
