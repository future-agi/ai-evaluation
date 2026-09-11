---
kind: agent-learning.docs-page.v1
track: frameworks
objective: behavior
stage: simulate
backing:
  - examples/sdk_multi_framework_simulation.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY=local-example agent-learn run examples/framework_pydantic_ai_manifest.json --output artifacts/framework-pydantic-ai.json
postcondition: python -c "import json; p=json.load(open('artifacts/framework-pydantic-ai.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# PydanticAI: offline framework-adapter simulation

> **Twin:** [`examples/sdk_multi_framework_simulation.py`](../../examples/sdk_multi_framework_simulation.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

PydanticAI coverage in the kit is runtime-simulated — multi-framework runtime
simulation, in the README's wording — rather than probe-promoted: the PydanticAI
agent shape is exercised inside the kit's multi-framework runtime alongside its
peers, and the evidence is the framework runtime trace each lane produces. The
twin, [`examples/sdk_multi_framework_simulation.py`](../../examples/sdk_multi_framework_simulation.py),
declares one lane per framework — each with a persona, situation, outcome, and a
required trace span — builds the corresponding run manifests against the local
shims in [`examples/framework_shims.py`](../../examples/framework_shims.py), and
simulates them through one shared runtime.

The failure class this catches is structured-output drift: a PydanticAI agent is
defined by typed, validated outputs, and a harness that only checks response text
will not notice when the typed result stops carrying the evidence the trace
requires. The runtime simulation pins the `Agent.run` span and its signals as a
checkable contract.

The run manifest, [`examples/framework_pydantic_ai_manifest.json`](../../examples/framework_pydantic_ai_manifest.json),
drives the single-framework slice of the same lane from the CLI. It targets the
factory `framework_shims.py:build_pydantic_ai_agent` with `trace_runtime: true` and
replays a `framework_trace` environment whose span is `Agent.run`. The full
multi-framework version is the suite manifest,
[`examples/multi_framework_simulation_suite.json`](../../examples/multi_framework_simulation_suite.json).
Everything runs on the `local_text` engine in one turn: offline, deterministic, no
provider keys.

## 2. Run it

CLI (the `required_env` key is CI metadata for this offline manifest — any
placeholder value satisfies it):

```bash
AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY=local-example \
agent-learn run examples/framework_pydantic_ai_manifest.json \
  --output artifacts/framework-pydantic-ai.json
```

SDK, same operation (export the same placeholder env first):

```python
import asyncio
from fi.alk import simulate

result = asyncio.run(
    simulate.run_manifest_file("examples/framework_pydantic_ai_manifest.json")
)
assert result["kind"] == "agent-learning.run.v1"
```

## 3. What you built

Postcondition (machine-checkable — the same check the docs gate pattern uses):

```bash
python -c "import json; p=json.load(open('artifacts/framework-pydantic-ai.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
```

The artifact carries `status`, a per-turn transcript for the framework-owner
scenario, the evaluation report, and the framework runtime trace evidence the
adapter extracted — including the `Agent.run` span the manifest replays. It is a
replayable record, not a log line: the same file feeds `baseline`, `compare`, and
`replay`.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `missing required environment variable(s)` | config fault | set the placeholder key shown above; `agent-learn doctor` → `summary.public_boundary_passed` confirms the install surface |
| trace evidence missing the required `Agent.run` span or signals | behavior regression | re-run the twin and compare the runtime-simulated lane against this single-framework slice |

## 5. Prove it / keep it

The twin is admitted by the `multi_framework_runtime_readiness` release gate, so
every `agent-learn release-check` re-executes this exact runtime simulation — the
page stays true or the release fails. Because this lane is runtime-simulated, the
claim is about runtime trace fidelity, not about a live PydanticAI process. To keep
your own typed agent honest, promote the run artifact into a regression baseline
with the `baseline` / `promote-to-regression` / `compare` command family, and scale
to all frameworks at once with the suite manifest linked above. The reader's job
here is maintenance of a living proof, not a one-off demo.
