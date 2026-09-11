---
kind: agent-learning.docs-page.v1
track: frameworks
objective: behavior
stage: simulate
backing:
  - examples/sdk_framework_adapter_langgraph_ainvoke_promotion.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY=local-example agent-learn run examples/framework_langgraph_manifest.json --output artifacts/framework-langgraph.json
postcondition: python -c "import json; p=json.load(open('artifacts/framework-langgraph.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# LangGraph: offline framework-adapter simulation

> **Twin:** [`examples/sdk_framework_adapter_langgraph_ainvoke_promotion.py`](../../examples/sdk_framework_adapter_langgraph_ainvoke_promotion.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

LangGraph coverage in the kit is probe-promoted: before a compiled graph is
simulated, the BYO adapter probes its candidate entrypoints and promotes the one
that produces real runtime evidence. A LangGraph-style app exposes both a synchronous
`invoke(dict)` and an asynchronous `ainvoke(dict)`. The twin,
[`examples/sdk_framework_adapter_langgraph_ainvoke_promotion.py`](../../examples/sdk_framework_adapter_langgraph_ainvoke_promotion.py),
builds a local `LocalLangGraphRunnable` whose sync `invoke` returns content with no
tool calls and no trace, while `ainvoke` returns verified evidence only when the
adapter passes `metadata.framework == "langgraph"` in the input dict.

The failure class this catches is silent adapter mismatch on async graphs: a harness
that calls the sync path gets an answer-shaped response with no proof that graph
nodes executed. The probe records the weak path as weak and promotes the async
entrypoint that carries framework evidence — the distinction is in the artifact, not
in your memory.

The run manifest, [`examples/framework_langgraph_manifest.json`](../../examples/framework_langgraph_manifest.json),
drives the same adapter from the CLI. It targets the factory
`framework_shims.py:build_langgraph_agent` with `trace_runtime: true` and replays a
`framework_trace` environment whose span is `refund_graph.ainvoke`, with required
adapter signals and mappings declared in the manifest. Everything runs on the
`local_text` engine in one turn: offline, deterministic, no provider keys.

## 2. Run it

CLI (the `required_env` key is CI metadata for this offline manifest — any
placeholder value satisfies it):

```bash
AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY=local-example \
agent-learn run examples/framework_langgraph_manifest.json \
  --output artifacts/framework-langgraph.json
```

SDK, same operation (export the same placeholder env first):

```python
import asyncio
from fi.alk import simulate

result = asyncio.run(
    simulate.run_manifest_file("examples/framework_langgraph_manifest.json")
)
assert result["kind"] == "agent-learning.run.v1"
```

## 3. What you built

Postcondition (machine-checkable — the same check the docs gate pattern uses):

```bash
python -c "import json; p=json.load(open('artifacts/framework-langgraph.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
```

The artifact carries `status`, a per-turn transcript for the framework-owner
scenario, the evaluation report, and the framework runtime trace evidence the
adapter extracted — including the `refund_graph.ainvoke` span the manifest replays.
It is a replayable record, not a log line: the same file feeds `baseline`,
`compare`, and `replay`.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `missing required environment variable(s)` | config fault | set the placeholder key shown above; `agent-learn doctor` → `summary.public_boundary_passed` confirms the install surface |
| transcript shows the weak sync `invoke` path (no tool calls, no trace) | behavior regression | re-run the twin promotion and compare `ainvoke` evidence against the sync fallback |

## 5. Prove it / keep it

The twin is admitted by the `framework_adapter_probe_readiness` release gate, so
every `agent-learn release-check` re-executes this exact promotion path — the page
stays true or the release fails. To keep your own LangGraph app honest, promote the
run artifact into a regression baseline with the `baseline` /
`promote-to-regression` / `compare` command family, then wire the manifest into CI.
The reader's job here is maintenance of a living proof, not a one-off demo: the
artifact you just wrote is the input to the next regression cycle.
