---
kind: agent-learning.docs-page.v1
track: frameworks
objective: behavior
stage: simulate
backing:
  - examples/sdk_framework_adapter_discovery.py
  - examples/sdk_framework_adapter_auto_discovery_optimization.py
  - examples/custom_framework_optimization.json
artifact_kinds:
  - agent-learning.run.v1
  - agent-learning.optimization.v1
commands:
  - AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY=local-example agent-learn run examples/framework_custom_manifest.json --output artifacts/framework-custom.json
  - AGENT_LEARNING_CUSTOM_FRAMEWORK_OPT_EXAMPLE_KEY=local-example agent-learn optimize examples/custom_framework_optimization.json --output artifacts/framework-custom-optimization.json
postcondition: python -c "import json; p=json.load(open('artifacts/framework-custom.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Custom frameworks: offline framework-adapter simulation

> **Twin:** [`examples/sdk_framework_adapter_discovery.py`](../../examples/sdk_framework_adapter_discovery.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

The custom lane is what every other framework page reduces to: any Python object
with a callable method can be probe-promoted, named frameworks are just presets.
The first twin, [`examples/sdk_framework_adapter_discovery.py`](../../examples/sdk_framework_adapter_discovery.py),
calls `simulate.discover_framework_adapter` against a local orchestrator with two
plausible entrypoints — a weak text-only `run(text)` and a structured async
`execute_task(payload)` — and lets discovery choose among
`method_candidates=["run", "execute_task"]` and
`input_mode_candidates=["text", "dict", "agent_input"]`. The second twin,
[`examples/sdk_framework_adapter_auto_discovery_optimization.py`](../../examples/sdk_framework_adapter_auto_discovery_optimization.py),
runs `optimize.optimize_framework_adapter_probe` so the optimizer, not the author,
settles which adapter shape carries framework evidence.

The failure class this catches is hand-wired adapter rot: a custom harness that
hardcodes one method name keeps "working" while the orchestrator grows a better
entrypoint, or while the old one quietly loses its tool evidence. Discovery turns
the adapter choice into recorded, re-checkable output.

The run manifest, [`examples/framework_custom_manifest.json`](../../examples/framework_custom_manifest.json),
pins the discovered shape for the CLI: factory
`framework_shims.py:build_custom_refund_orchestrator`, `method: execute_task`,
`input_mode: dict`, and a `framework_trace` environment whose span is
`CustomRefundOrchestrator.execute_task`. The third twin,
[`examples/custom_framework_optimization.json`](../../examples/custom_framework_optimization.json),
is the optimization manifest for the same custom framework. All of it runs on the
`local_text` engine: offline, deterministic, no provider keys.

## 2. Run it

CLI (the `required_env` keys are CI metadata for these offline manifests — any
placeholder value satisfies them):

```bash
AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY=local-example \
agent-learn run examples/framework_custom_manifest.json \
  --output artifacts/framework-custom.json

AGENT_LEARNING_CUSTOM_FRAMEWORK_OPT_EXAMPLE_KEY=local-example \
agent-learn optimize examples/custom_framework_optimization.json \
  --output artifacts/framework-custom-optimization.json
```

SDK, same first operation (export the same placeholder env first):

```python
import asyncio
from fi.alk import simulate

result = asyncio.run(
    simulate.run_manifest_file("examples/framework_custom_manifest.json")
)
assert result["kind"] == "agent-learning.run.v1"
```

## 3. What you built

Postcondition (machine-checkable — the same check the docs gate pattern uses):

```bash
python -c "import json; p=json.load(open('artifacts/framework-custom.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
```

The run artifact carries `status`, the transcript, the evaluation report, and the
`CustomRefundOrchestrator.execute_task` trace evidence; the optimization artifact
(`agent-learning.optimization.v1`) records candidates and the selected adapter
configuration. Both are replayable records that feed `baseline`, `compare`, and
`replay`.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `missing required environment variable(s)` | config fault | set the placeholder keys shown above; `agent-learn doctor` → `summary.public_boundary_passed` confirms the install surface |
| discovery selects the weak `run(text)` candidate | behavior regression | re-run the auto-discovery optimization twin and compare `execute_task` evidence against the text fallback |

## 5. Prove it / keep it

The discovery twin is admitted by the `framework_adapter_probe_readiness` release
gate and the optimization manifest by `framework_optimizer_readiness`, so every
`agent-learn release-check` re-executes both paths — the page stays true or the
release fails. To keep your own orchestrator honest, pin the discovered adapter in
a manifest like the one above, promote the run artifact into a regression baseline
with the `baseline` / `promote-to-regression` / `compare` command family, and
re-run discovery whenever the orchestrator's surface changes. The reader's job here
is maintenance of a living proof, not a one-off demo.
