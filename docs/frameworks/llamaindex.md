---
kind: agent-learning.docs-page.v1
track: frameworks
objective: behavior
stage: simulate
backing:
  - examples/sdk_framework_adapter_probe.py
  - examples/sdk_framework_adapter_probe_promotion.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY=local-example agent-learn run examples/framework_llamaindex_manifest.json --output artifacts/framework-llamaindex.json
postcondition: python -c "import json; p=json.load(open('artifacts/framework-llamaindex.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# LlamaIndex: offline framework-adapter simulation

> **Twin:** [`examples/sdk_framework_adapter_probe.py`](../../examples/sdk_framework_adapter_probe.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

LlamaIndex coverage in the kit is probe-promoted: before a chat engine is simulated,
the BYO adapter probes its candidate entrypoints and promotes the one that produces
real runtime evidence. The probe mechanics are shown by two twins.
[`examples/sdk_framework_adapter_probe.py`](../../examples/sdk_framework_adapter_probe.py)
calls `simulate.run_framework_adapter_probe` against a local object with one callable
method, declaring `method`, `input_mode`, and probe cases explicitly.
[`examples/sdk_framework_adapter_probe_promotion.py`](../../examples/sdk_framework_adapter_probe_promotion.py)
adds the promotion step: the object also has a weak `run(text)` path with no tool
evidence, and promotion selects the entrypoint whose response carries
`framework_trace` events and tool calls.

The failure class this catches is silent adapter mismatch: a LlamaIndex chat engine
answers through a text-only surface, the harness accepts the string, and nothing
proves the engine's async chat path actually executed. The probe records the weak
path as weak; the promoted adapter is the one with evidence.

The run manifest, [`examples/framework_llamaindex_manifest.json`](../../examples/framework_llamaindex_manifest.json),
drives the LlamaIndex-specific shape from the CLI. It targets the factory
`framework_shims.py:build_llamaindex_chat_engine` with `trace_runtime: true` and
replays a `framework_trace` environment whose span is `chat_engine.achat`.
Everything runs on the `local_text` engine in one turn: offline, deterministic, no
provider keys.

## 2. Run it

CLI (the `required_env` key is CI metadata for this offline manifest — any
placeholder value satisfies it):

```bash
AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY=local-example \
agent-learn run examples/framework_llamaindex_manifest.json \
  --output artifacts/framework-llamaindex.json
```

SDK, same operation (export the same placeholder env first):

```python
import asyncio
from fi.alk import simulate

result = asyncio.run(
    simulate.run_manifest_file("examples/framework_llamaindex_manifest.json")
)
assert result["kind"] == "agent-learning.run.v1"
```

## 3. What you built

Postcondition (machine-checkable — the same check the docs gate pattern uses):

```bash
python -c "import json; p=json.load(open('artifacts/framework-llamaindex.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
```

The artifact carries `status`, a per-turn transcript for the framework-owner
scenario, the evaluation report, and the framework runtime trace evidence the
adapter extracted — including the `chat_engine.achat` span the manifest replays. It
is a replayable record, not a log line: the same file feeds `baseline`, `compare`,
and `replay`.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `missing required environment variable(s)` | config fault | set the placeholder key shown above; `agent-learn doctor` → `summary.public_boundary_passed` confirms the install surface |
| transcript shows a weak text-only path (no tool calls, no trace) | behavior regression | re-run the twin promotion and compare the `achat`-style evidence against the text fallback |

## 5. Prove it / keep it

Both twins are admitted by the `framework_adapter_probe_readiness` release gate, so
every `agent-learn release-check` re-executes this exact probe-and-promote path —
the page stays true or the release fails. To keep your own LlamaIndex engine honest,
promote the run artifact into a regression baseline with the `baseline` /
`promote-to-regression` / `compare` command family, then wire the manifest into CI.
The reader's job here is maintenance of a living proof, not a one-off demo: the
artifact you just wrote is the input to the next regression cycle.
