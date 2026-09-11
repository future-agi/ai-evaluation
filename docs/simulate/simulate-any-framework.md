---
kind: agent-learning.docs-page.v1
track: simulate
objective: behavior
stage: simulate
backing:
  - examples/sdk_multi_framework_simulation.py
artifact_kinds:
  - agent-learning.suite.v1
  - agent-learning.run.v1
commands:
  - AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY=offline-demo-key agent-learn suite examples/multi_framework_simulation_suite.json --output artifacts/multi-framework-suite.json
postcondition: python -c "import json; p=json.load(open('artifacts/multi-framework-suite.json')); assert p['kind']=='agent-learning.suite.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Simulate Any Framework

> **Twin:** [`examples/sdk_multi_framework_simulation.py`](../../examples/sdk_multi_framework_simulation.py)
> · emits `agent-learning.suite.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

Most agent test harnesses are married to one framework. The kit's simulate
track takes the opposite position: a simulation is a manifest, and the
framework is just one field in it. The same persona, situation, and expected
outcome run against a LangChain runnable, a LangGraph state graph, a CrewAI
crew, a Pipecat pipeline, or your own custom orchestrator — and every run
leaves the same `agent-learning.run.v1` artifact behind.

The failure class this catches is framework-coupling drift: an agent that
passes your bespoke pytest harness, then behaves differently after a port from
one orchestration library to another, because tool-call evidence, trace spans,
or message history were shaped differently and nothing checked them. The
suite manifest used here fans out one `run` job per framework —
`langchain`, `langgraph`, `llamaindex`, `openai_agents`, `autogen`, `crewai`,
`pydantic_ai`, `pipecat`, `livekit`, and a `custom_refund_orchestrator` —
each over its own `examples/framework_*_manifest.json`, each asserting
`framework_trace` environment evidence (span name, input, output, signals).

The placeholder environment variable in the command is CI wiring metadata,
not a provider credential: the engines are local and deterministic, so any
value satisfies the manifest's `required_env` check.

## 2. Run it

CLI:

```bash
AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY=offline-demo-key \
  agent-learn suite examples/multi_framework_simulation_suite.json \
  --output artifacts/multi-framework-suite.json
```

Note: `agent-learn` resolves a relative `--output` against your current
working directory, so the artifact lands at `artifacts/multi-framework-suite.json`.

SDK (same operation):

```python
import os
from fi.alk import suite

os.environ.setdefault("AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY", "offline-demo-key")
result = suite.run_suite_file("examples/multi_framework_simulation_suite.json")
```

The backing example, `examples/sdk_multi_framework_simulation.py`, builds the
same suite programmatically from local framework shims
(`examples/framework_shims.py`) and runs it through `suite.run_suite_file`.

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/multi-framework-suite.json')); assert p['kind']=='agent-learning.suite.v1', p['kind']; print('ok')"
```

The suite artifact contains one child result per framework job, each an
`agent-learning.run.v1` payload with the framework runtime trace that was
captured (span id, span name, signals such as `model`, `tool`, `chain`),
plus a roll-up summary with per-job status and exit code.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| manifest rejected / `required_env` missing | config fault | `summary.public_boundary_passed` + the manifest error line |
| one framework job fails, others pass | framework trace evidence gap | inspect that job's child result inside the suite artifact |

## 5. Prove it / keep it

Each framework in this suite has its own page in `docs/frameworks/` with the
adapter-specific trace contract. Once your own agent's framework runs green
here, capture the artifact as a baseline and wire the comparison into CI —
the full journey (baseline → compare → promote → replay) is
[`regression-lifecycle.md`](regression-lifecycle.md). The suite's verdict is
re-proven on every `agent-learn release-check` by the
`multi_framework_runtime_readiness` gate that admits this page's backing.
