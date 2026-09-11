---
kind: agent-learning.docs-page.v1
track: frameworks
objective: behavior
stage: simulate
backing:
  - examples/sdk_framework_adapter_message_history.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY=local-example agent-learn run examples/framework_autogen_manifest.json --output artifacts/framework-autogen.json
postcondition: python -c "import json; p=json.load(open('artifacts/framework-autogen.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# AutoGen: offline framework-adapter simulation

> **Twin:** [`examples/sdk_framework_adapter_message_history.py`](../../examples/sdk_framework_adapter_message_history.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

AutoGen coverage in the kit is probe-promoted: before an AgentChat team is
simulated, the BYO adapter probes its candidate entrypoints and promotes the one
that produces real runtime evidence. AutoGen's distinctive surface is the message
history: a `TaskResult` whose `messages` list mixes `TextMessage`,
`ToolCallRequestEvent`, and `ToolCallExecutionEvent` entries plus a `stop_reason`.
The twin, [`examples/sdk_framework_adapter_message_history.py`](../../examples/sdk_framework_adapter_message_history.py),
builds a local `LocalAutoGenTeam` that returns exactly that transcript shape, and
the adapter must reconstruct tool calls and events from the message history rather
than from a flat content string.

The failure class this catches is transcript loss: a harness that keeps only the
final message of an AutoGen run silently discards the tool-call evidence that the
team actually did the work. The adapter promotion records whether the message
history round-trips into trace evidence — a weak text-only path is recorded as weak.

The run manifest, [`examples/framework_autogen_manifest.json`](../../examples/framework_autogen_manifest.json),
drives the same adapter from the CLI. It targets the factory
`framework_shims.py:build_autogen_agent` with `trace_runtime: true` and replays a
`framework_trace` environment whose span is `AgentChat.run`. Everything runs on the
`local_text` engine in one turn: offline, deterministic, no provider keys. For the
red-team lane on the same framework, see
[`examples/redteam_autogen_optimization.json`](../../examples/redteam_autogen_optimization.json)
and [`examples/sdk_redteam_autogen_optimization.py`](../../examples/sdk_redteam_autogen_optimization.py)
(that lane requires its own example env key).

## 2. Run it

CLI (the `required_env` key is CI metadata for this offline manifest — any
placeholder value satisfies it):

```bash
AGENT_LEARNING_MULTI_FRAMEWORK_EXAMPLE_KEY=local-example \
agent-learn run examples/framework_autogen_manifest.json \
  --output artifacts/framework-autogen.json
```

SDK, same operation (export the same placeholder env first):

```python
import asyncio
from fi.alk import simulate

result = asyncio.run(
    simulate.run_manifest_file("examples/framework_autogen_manifest.json")
)
assert result["kind"] == "agent-learning.run.v1"
```

## 3. What you built

Postcondition (machine-checkable — the same check the docs gate pattern uses):

```bash
python -c "import json; p=json.load(open('artifacts/framework-autogen.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
```

The artifact carries `status`, a per-turn transcript for the framework-owner
scenario, the evaluation report, and the framework runtime trace evidence the
adapter extracted — including the `AgentChat.run` span the manifest replays. It is
a replayable record, not a log line: the same file feeds `baseline`, `compare`,
and `replay`.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `missing required environment variable(s)` | config fault | set the placeholder key shown above; `agent-learn doctor` → `summary.public_boundary_passed` confirms the install surface |
| transcript keeps only final text (tool-call events dropped) | behavior regression | re-run the twin promotion and compare message-history evidence against the text fallback |

## 5. Prove it / keep it

The twin is admitted by the `framework_adapter_probe_readiness` release gate, so
every `agent-learn release-check` re-executes this exact message-history promotion
path — the page stays true or the release fails. To keep your own AutoGen team
honest, promote the run artifact into a regression baseline with the `baseline` /
`promote-to-regression` / `compare` command family, then wire the manifest into CI.
The reader's job here is maintenance of a living proof, not a one-off demo: the
artifact you just wrote is the input to the next regression cycle.
