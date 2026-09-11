---
kind: agent-learning.docs-page.v1
track: frameworks
objective: behavior
stage: simulate
backing:
  - examples/sdk_framework_adapter_handoff_transcript.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY=local-example agent-learn run examples/framework_openai_agents_manifest.json --output artifacts/framework-openai-agents.json
postcondition: python -c "import json; p=json.load(open('artifacts/framework-openai-agents.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# OpenAI Agents: offline framework-adapter simulation

> **Twin:** [`examples/sdk_framework_adapter_handoff_transcript.py`](../../examples/sdk_framework_adapter_handoff_transcript.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

OpenAI Agents coverage in the kit is runtime-simulated — handoff-transcript
promotion, in the README's wording — rather than probe-promoted: the adapter is
exercised against a simulated runtime that produces the SDK's transcript shapes,
and promotion is decided on that transcript evidence. The twin,
[`examples/sdk_framework_adapter_handoff_transcript.py`](../../examples/sdk_framework_adapter_handoff_transcript.py),
builds a local `LocalHandoffTeam` whose output is a typed handoff transcript:
`HandoffMessage` entries (source, `handoff_to`, task, reason), `ReviewMessage`
entries with a review status, and a `ReconciliationMessage` that records which
agent's answer was accepted. The adapter must turn that multi-agent transcript into
trace evidence, not flatten it to a final string.

The failure class this catches is handoff loss: an agents-SDK run that delegates
between specialized agents can look identical to a single-agent run if the harness
keeps only the last message. The simulated transcript makes delegation, review, and
reconciliation each visible and checkable in the artifact.

The run manifest, [`examples/framework_openai_agents_manifest.json`](../../examples/framework_openai_agents_manifest.json),
drives the same adapter from the CLI. It targets the factory
`framework_shims.py:build_openai_agents_runner` with `trace_runtime: true` and
replays a `framework_trace` environment whose span is `Runner.run`. Everything runs
on the `local_text` engine in one turn: offline, deterministic, no provider keys.

## 2. Run it

CLI (the `required_env` key is CI metadata for this offline manifest — any
placeholder value satisfies it):

```bash
AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY=local-example \
agent-learn run examples/framework_openai_agents_manifest.json \
  --output artifacts/framework-openai-agents.json
```

SDK, same operation (export the same placeholder env first):

```python
import asyncio
from fi.alk import simulate

result = asyncio.run(
    simulate.run_manifest_file("examples/framework_openai_agents_manifest.json")
)
assert result["kind"] == "agent-learning.run.v1"
```

## 3. What you built

Postcondition (machine-checkable — the same check the docs gate pattern uses):

```bash
python -c "import json; p=json.load(open('artifacts/framework-openai-agents.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
```

The artifact carries `status`, a per-turn transcript for the framework-owner
scenario, the evaluation report, and the framework runtime trace evidence the
adapter extracted — including the `Runner.run` span the manifest replays. It is a
replayable record, not a log line: the same file feeds `baseline`, `compare`, and
`replay`.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `missing required environment variable(s)` | config fault | set the placeholder key shown above; `agent-learn doctor` → `summary.public_boundary_passed` confirms the install surface |
| transcript keeps only the final answer (handoff/review entries dropped) | behavior regression | re-run the twin and compare the handoff-transcript evidence against the flattened fallback |

## 5. Prove it / keep it

The twin is admitted by the `framework_adapter_probe_readiness` release gate, so
every `agent-learn release-check` re-executes this exact handoff-transcript path —
the page stays true or the release fails. Because this lane is runtime-simulated,
the promotion claim is about transcript fidelity, not about a live OpenAI Agents
process. To keep your own agents app honest, promote the run artifact into a
regression baseline with the `baseline` / `promote-to-regression` / `compare`
command family, then wire the manifest into CI. The reader's job here is
maintenance of a living proof, not a one-off demo.
