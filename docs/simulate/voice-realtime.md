---
kind: agent-learning.docs-page.v1
track: simulate
objective: capability
stage: simulate
backing:
  - examples/sdk_framework_adapter_realtime_trace.py
  - examples/sdk_realtime_stack_probe_optimization.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - python examples/sdk_framework_adapter_realtime_trace.py artifacts/realtime-trace.json
  - AGENT_LEARNING_VOICE_STREAMING_EXAMPLE_KEY=offline-demo-key agent-learn run examples/voice_streaming_realtime_manifest.json --output artifacts/voice-realtime.json
postcondition: python -c "import json; p=json.load(open('artifacts/realtime-trace.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: true
---

# Voice and Realtime: simulate the session before you dial it

> **Twin:** [`examples/sdk_framework_adapter_realtime_trace.py`](../../examples/sdk_framework_adapter_realtime_trace.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

This is an opt-in lane. The first live-lane rung is implemented for LiveKit
and Pipecat behind their opt-in extras; loopback-audio and telephony
transports are planned (see [ROADMAP.md](../../ROADMAP.md)). This page runs
the local deterministic realtime fixture — the same manifests and trace
contracts apply on the live transports, only the transport changes.

## 1. What you are testing

Voice agents fail on timing and sequencing, not just on words: a tool call
fired after the agent already started speaking, a transcript finalized
before the tool result arrived, a session that never reached `closed`. None
of that requires audio hardware to test — it requires the frame and event
sequence, which is exactly what the local fixture replays deterministically.

The twin's `LocalRealtimeVoiceStack` exports both vocabularies of the
realtime world. Pipecat-style frames: `AudioRawFrame` (16 kHz inbound) →
`TranscriptionFrame` (caller asks about a refund) → `FunctionCallFrame`
(`lookup_refund_policy`) → `FunctionCallResultFrame` (eligible, 30-day
policy) → `EndFrame`, each stamped with `timestamp_ms` and direction. And
LiveKit-style session events: `agent_state_changed`
(listening → thinking → speaking), `tool_execution_started`/`completed`,
`transcript_final`, `session_closed`. The case contract then demands
evidence, not vibes: `required_events` covering `realtime_frame`,
`realtime_tool_call`, `realtime_tool_response`, `realtime_transcript`, and
`realtime_lifecycle`, plus `required_tools` and `realtime_trace` state.

The second command runs the manifest form of the same idea:
`examples/voice_streaming_realtime_manifest.json` simulates a scripted
voice agent with `voice` and `streaming_trace` environments — call routing,
transcription, TTS, and streaming token/tool events from one file. The
second backing example (`sdk_realtime_stack_probe_optimization.py`) probes
weak/strong realtime stack candidates and promotes the winner, keeping the
selection surface proven.

## 2. Run it

CLI (the first command needs no env; the second's placeholder is CI wiring
metadata — both engines are local and deterministic):

```bash
python examples/sdk_framework_adapter_realtime_trace.py artifacts/realtime-trace.json

AGENT_LEARNING_VOICE_STREAMING_EXAMPLE_KEY=offline-demo-key \
  agent-learn run examples/voice_streaming_realtime_manifest.json \
  --output artifacts/voice-realtime.json
```

Note: `agent-learn` resolves a relative `--output` against your current
working directory, so the second artifact lands at `artifacts/voice-realtime.json`.

SDK (the operation the twin performs):

```python
import asyncio
from fi.alk import optimize, simulate

manifest = optimize.build_framework_run_manifest_from_local_adapter(
    framework="livekit",
    target="examples/sdk_framework_adapter_realtime_trace.py:LocalRealtimeVoiceStack",
    method_candidates=["respond", "run_session"],
)
simulate.write_manifest_file(manifest, "realtime.manifest.json")
result = asyncio.run(simulate.run_manifest_file("realtime.manifest.json"))
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/realtime-trace.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
```

The artifact's case evidence holds the complete timestamped frame and
session-event sequence, the tool call with its result, and the lifecycle
states the session moved through — enough to assert ordering properties
(tool result before final transcript, session closed last) from the
artifact alone.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| manifest rejected / `required_env` missing | config fault | `summary.public_boundary_passed` + the manifest error line |
| required realtime events missing | adapter export lacks frames or session events | inspect the realtime trace evidence in the artifact's case record |

## 5. Prove it / keep it

The twin runs fresh on every `agent-learn release-check` via the docs gate,
and the probe example is re-proven by the `realtime_stack_probe_readiness`
gate — so the realtime trace contract this page teaches stays executable.
Baseline a passing artifact and follow
[`regression-lifecycle.md`](regression-lifecycle.md) so a timing regression
shows up as a compare finding. For the framework-specific adapters behind
the live lanes, see the LiveKit and Pipecat pages in `docs/frameworks/`
(both flagged opt-in with the same roadmap pointer).
