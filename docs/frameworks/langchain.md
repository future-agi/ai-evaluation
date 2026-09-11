---
kind: agent-learning.docs-page.v1
track: frameworks
objective: behavior
stage: simulate
backing:
  - examples/sdk_framework_adapter_langchain_invoke_promotion.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY=local-example agent-learn run examples/framework_langchain_manifest.json --output artifacts/framework-langchain.json
postcondition: python -c "import json; p=json.load(open('artifacts/framework-langchain.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# LangChain: offline framework-adapter simulation

> **Twin:** [`examples/sdk_framework_adapter_langchain_invoke_promotion.py`](../../examples/sdk_framework_adapter_langchain_invoke_promotion.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

LangChain coverage in the kit is probe-promoted: before a runnable is simulated, the
BYO adapter probes its candidate entrypoints and promotes the one that produces real
runtime evidence. A LangChain-style object typically exposes more than one callable
surface — a legacy `run(text)` path and the runnable `invoke(dict)` path. The twin,
[`examples/sdk_framework_adapter_langchain_invoke_promotion.py`](../../examples/sdk_framework_adapter_langchain_invoke_promotion.py),
builds a local `LocalLangChainRunnable` whose `run` method returns content with no
tool calls and no trace, while `invoke` returns verified evidence only when the
adapter passes `metadata.framework == "langchain"` in the input dict.

The failure class this catches is silent adapter mismatch: your harness calls the
text-only path, the agent appears to answer, and nothing in the transcript proves the
chain actually executed. The probe makes that distinction explicit — the weak path is
recorded as weak, and promotion selects the entrypoint with framework evidence.

The run manifest, [`examples/framework_langchain_manifest.json`](../../examples/framework_langchain_manifest.json),
drives the same adapter from the CLI. It targets the factory
`framework_shims.py:build_langchain_agent` with `trace_runtime: true` and replays a
`framework_trace` environment whose span is `RunnableSequence.ainvoke` with
`model`/`tool`/`chain` signals. Everything runs on the `local_text` engine in one
turn: offline, deterministic, no provider keys.

## 2. Run it

CLI (the `required_env` key is CI metadata for this offline manifest — any
placeholder value satisfies it):

```bash
AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY=local-example \
agent-learn run examples/framework_langchain_manifest.json \
  --output artifacts/framework-langchain.json
```

SDK, same operation (export the same placeholder env first):

```python
import asyncio
from fi.alk import simulate

result = asyncio.run(
    simulate.run_manifest_file("examples/framework_langchain_manifest.json")
)
assert result["kind"] == "agent-learning.run.v1"
```

## 3. What you built

Postcondition (machine-checkable — the same check the docs gate pattern uses):

```bash
python -c "import json; p=json.load(open('artifacts/framework-langchain.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
```

The artifact carries `status`, a per-turn transcript for the Maya
framework-owner scenario, the evaluation report, and the framework runtime trace
evidence the adapter extracted — including the `RunnableSequence.ainvoke` span and
the adapter signals the manifest requires. It is a replayable record, not a log
line: the same file feeds `baseline`, `compare`, and `replay`.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `missing required environment variable(s)` | config fault | set the placeholder key shown above; `agent-learn doctor` → `summary.public_boundary_passed` confirms the install surface |
| transcript shows the weak `run(text)` path (no tool calls, no trace) | behavior regression | re-run the twin promotion and compare `invoke` evidence against the `run` fallback |

## 5. Prove it / keep it

The twin is admitted by the `framework_adapter_probe_readiness` release gate, so
every `agent-learn release-check` re-executes this exact promotion path — the page
stays true or the release fails. To keep your own LangChain agent honest, promote
the run artifact into a regression baseline with the `baseline` /
`promote-to-regression` / `compare` command family, then wire the manifest into CI.
The reader's job here is maintenance of a living proof, not a one-off demo: the
artifact you just wrote is the input to the next regression cycle.
