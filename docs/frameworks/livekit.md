---
kind: agent-learning.docs-page.v1
track: frameworks
objective: behavior
stage: simulate
backing:
  - examples/sdk_framework_adapter_livekit_run_session_promotion.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY=local-example agent-learn run examples/framework_livekit_manifest.json --output artifacts/framework-livekit.json
postcondition: python -c "import json; p=json.load(open('artifacts/framework-livekit.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# LiveKit: offline framework-adapter simulation

> **Twin:** [`examples/sdk_framework_adapter_livekit_run_session_promotion.py`](../../examples/sdk_framework_adapter_livekit_run_session_promotion.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

LiveKit coverage in the kit is probe-promoted, and this page is deliberately
offline: it tests the agent-session adapter, not a live room. The twin,
[`examples/sdk_framework_adapter_livekit_run_session_promotion.py`](../../examples/sdk_framework_adapter_livekit_run_session_promotion.py),
builds a local `LocalLiveKitAgentSession` with two surfaces: a `respond(text)` path
that returns content with no room-session evidence, and an async
`run_session(payload)` path that returns verified evidence only when the adapter
passes `metadata.framework == "livekit"` in the payload dict. Promotion selects the
session entrypoint with evidence; the text path is recorded as weak.

The failure class this catches is session-shape mismatch: a voice agent harness
that drives the text surface can look healthy while never exercising the session
contract a real room would use. Pinning the adapter shape offline means the
behavioral contract is already proven before any live infrastructure enters the
picture.

The run manifest, [`examples/framework_livekit_manifest.json`](../../examples/framework_livekit_manifest.json),
drives the same adapter from the CLI. It targets the factory
`framework_shims.py:build_livekit_agent` with `trace_runtime: true` and replays a
`framework_trace` environment whose span is `agent.respond`. Everything runs on the
`local_text` engine in one turn: offline, deterministic, no provider keys.

## 2. Run it

CLI (the `required_env` key is CI metadata for this offline manifest — any
placeholder value satisfies it):

```bash
AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY=local-example \
agent-learn run examples/framework_livekit_manifest.json \
  --output artifacts/framework-livekit.json
```

SDK, same operation (export the same placeholder env first):

```python
import asyncio
from fi.alk import simulate

result = asyncio.run(
    simulate.run_manifest_file("examples/framework_livekit_manifest.json")
)
assert result["kind"] == "agent-learning.run.v1"
```

## 3. What you built

Postcondition (machine-checkable — the same check the docs gate pattern uses):

```bash
python -c "import json; p=json.load(open('artifacts/framework-livekit.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
```

The artifact carries `status`, a per-turn transcript for the framework-owner
scenario, the evaluation report, and the framework runtime trace evidence the
adapter extracted — including the `agent.respond` span the manifest replays. It is
a replayable record, not a log line: the same file feeds `baseline`, `compare`, and
`replay`.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `missing required environment variable(s)` | config fault | set the placeholder key shown above; `agent-learn doctor` → `summary.public_boundary_passed` confirms the install surface |
| transcript shows the weak `respond(text)` path (no session evidence) | behavior regression | re-run the twin promotion and compare `run_session` evidence against the text fallback |

## 5. Prove it / keep it

The twin is admitted by the `framework_adapter_probe_readiness` release gate, so
every `agent-learn release-check` re-executes this exact session promotion path —
the page stays true or the release fails. To keep your own LiveKit agent honest,
promote the run artifact into a regression baseline with the `baseline` /
`promote-to-regression` / `compare` command family, then wire the manifest into CI.
Live LiveKit sessions (real rooms, real audio) are an opt-in lane (see
ROADMAP.md: voice lane rungs) — this page stays on the offline golden path. The reader's job here is maintenance of a
living proof, not a one-off demo.
