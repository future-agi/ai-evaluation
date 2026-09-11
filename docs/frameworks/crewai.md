---
kind: agent-learning.docs-page.v1
track: frameworks
objective: behavior
stage: simulate
backing:
  - examples/sdk_framework_adapter_keyword_inputs.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY=local-example agent-learn run examples/framework_crewai_manifest.json --output artifacts/framework-crewai.json
postcondition: python -c "import json; p=json.load(open('artifacts/framework-crewai.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# CrewAI: offline framework-adapter simulation

> **Twin:** [`examples/sdk_framework_adapter_keyword_inputs.py`](../../examples/sdk_framework_adapter_keyword_inputs.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

CrewAI coverage in the kit is probe-promoted: before a crew is simulated, the BYO
adapter probes its candidate entrypoints and promotes the one that produces real
runtime evidence. CrewAI's distinctive surface is the keyword-only entrypoint:
`kickoff(*, inputs=...)` cannot be called positionally. The twin,
[`examples/sdk_framework_adapter_keyword_inputs.py`](../../examples/sdk_framework_adapter_keyword_inputs.py),
builds a local `LocalCrewOrchestrator` whose `run(text)` path returns content with
no keyword input and no tool evidence, while `kickoff` returns verified evidence —
tool calls plus a `crew_kickoff` framework-trace event — only when the adapter
routes the payload through the keyword `inputs` argument with
`metadata.framework == "crewai"`.

The failure class this catches is entrypoint-shape mismatch: a harness that calls
the text path gets an answer-shaped string while the crew never receives its
structured inputs. The probe records which input key actually carried the payload,
so the promoted adapter is the one with kickoff evidence, not the fallback.

The run manifest, [`examples/framework_crewai_manifest.json`](../../examples/framework_crewai_manifest.json),
drives the same adapter from the CLI. It targets the factory
`framework_shims.py:build_crewai_crew` with `trace_runtime: true` and replays a
`framework_trace` environment whose span is `Crew.kickoff`. Everything runs on the
`local_text` engine in one turn: offline, deterministic, no provider keys.

## 2. Run it

CLI (the `required_env` key is CI metadata for this offline manifest — any
placeholder value satisfies it):

```bash
AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY=local-example \
agent-learn run examples/framework_crewai_manifest.json \
  --output artifacts/framework-crewai.json
```

SDK, same operation (export the same placeholder env first):

```python
import asyncio
from fi.alk import simulate

result = asyncio.run(
    simulate.run_manifest_file("examples/framework_crewai_manifest.json")
)
assert result["kind"] == "agent-learning.run.v1"
```

## 3. What you built

Postcondition (machine-checkable — the same check the docs gate pattern uses):

```bash
python -c "import json; p=json.load(open('artifacts/framework-crewai.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
```

The artifact carries `status`, a per-turn transcript for the framework-owner
scenario, the evaluation report, and the framework runtime trace evidence the
adapter extracted — including the `Crew.kickoff` span the manifest replays. It is a
replayable record, not a log line: the same file feeds `baseline`, `compare`, and
`replay`.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `missing required environment variable(s)` | config fault | set the placeholder key shown above; `agent-learn doctor` → `summary.public_boundary_passed` confirms the install surface |
| transcript shows the weak `run(text)` path (no `inputs` keyword, no tool calls) | behavior regression | re-run the twin promotion and compare `kickoff` evidence against the text fallback |

## 5. Prove it / keep it

The twin is admitted by the `framework_adapter_io_readiness` release gate, so every
`agent-learn release-check` re-executes this exact keyword-input promotion path —
the page stays true or the release fails. To keep your own crew honest, promote the
run artifact into a regression baseline with the `baseline` /
`promote-to-regression` / `compare` command family, then wire the manifest into CI.
The reader's job here is maintenance of a living proof, not a one-off demo: the
artifact you just wrote is the input to the next regression cycle.
