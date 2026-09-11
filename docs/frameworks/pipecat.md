---
kind: agent-learning.docs-page.v1
track: frameworks
objective: behavior
stage: simulate
backing:
  - examples/sdk_framework_adapter_pipecat_process_promotion.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY=local-example agent-learn run examples/framework_pipecat_manifest.json --output artifacts/framework-pipecat.json
postcondition: python -c "import json; p=json.load(open('artifacts/framework-pipecat.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Pipecat: offline framework-adapter simulation

> **Twin:** [`examples/sdk_framework_adapter_pipecat_process_promotion.py`](../../examples/sdk_framework_adapter_pipecat_process_promotion.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

Pipecat coverage in the kit is probe-promoted, and this page is deliberately
offline: it tests the frame-pipeline adapter, not a live audio transport. The twin,
[`examples/sdk_framework_adapter_pipecat_process_promotion.py`](../../examples/sdk_framework_adapter_pipecat_process_promotion.py),
builds a local `LocalPipecatPipeline` with two surfaces: a `run(input)` path that
returns content with no frame trace and no tool evidence, and a `process(payload)`
path that returns verified evidence only when the adapter passes
`metadata.framework == "pipecat"` in the payload dict. Promotion selects the frame
entrypoint with evidence; the text path is recorded as weak.

The failure class this catches is pipeline-shape mismatch: a voice harness that
drives the convenience text surface can look healthy while the frame-processing
contract a real pipeline uses goes untested. Pinning the adapter shape offline
means the behavioral contract is already proven before any live transport enters
the picture.

The run manifest, [`examples/framework_pipecat_manifest.json`](../../examples/framework_pipecat_manifest.json),
drives the same adapter from the CLI. It targets the factory
`framework_shims.py:build_pipecat_pipeline` with `trace_runtime: true` and replays
a `framework_trace` environment whose span is `pipeline.process`. Everything runs
on the `local_text` engine in one turn: offline, deterministic, no provider keys.

## 2. Run it

CLI (the `required_env` key is CI metadata for this offline manifest — any
placeholder value satisfies it):

```bash
AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY=local-example \
agent-learn run examples/framework_pipecat_manifest.json \
  --output artifacts/framework-pipecat.json
```

SDK, same operation (export the same placeholder env first):

```python
import asyncio
from fi.alk import simulate

result = asyncio.run(
    simulate.run_manifest_file("examples/framework_pipecat_manifest.json")
)
assert result["kind"] == "agent-learning.run.v1"
```

## 3. What you built

Postcondition (machine-checkable — the same check the docs gate pattern uses):

```bash
python -c "import json; p=json.load(open('artifacts/framework-pipecat.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
```

The artifact carries `status`, a per-turn transcript for the framework-owner
scenario, the evaluation report, and the framework runtime trace evidence the
adapter extracted — including the `pipeline.process` span the manifest replays. It
is a replayable record, not a log line: the same file feeds `baseline`, `compare`,
and `replay`.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `missing required environment variable(s)` | config fault | set the placeholder key shown above; `agent-learn doctor` → `summary.public_boundary_passed` confirms the install surface |
| transcript shows the weak `run(input)` path (no frame trace, no tool calls) | behavior regression | re-run the twin promotion and compare `process` evidence against the text fallback |

## 5. Prove it / keep it

The twin is admitted by the `framework_adapter_probe_readiness` release gate, so
every `agent-learn release-check` re-executes this exact frame-pipeline promotion
path — the page stays true or the release fails. To keep your own pipeline honest,
promote the run artifact into a regression baseline with the `baseline` /
`promote-to-regression` / `compare` command family, then wire the manifest into CI.
Live Pipecat transports (real audio in and out) are an opt-in lane (see
ROADMAP.md: voice lane rungs) — this page stays on the offline golden path. The reader's job here is maintenance of a
living proof, not a one-off demo.
