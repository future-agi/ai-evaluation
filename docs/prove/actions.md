---
kind: agent-learning.docs-page.v1
track: prove
objective: capability
stage: promote
backing:
  - examples/sdk_agent_integration_optimization.py
artifact_kinds:
  - agent-learning.optimization.v1
  - agent-learning.actions.v1
  - agent-learning.action-run.v1
commands:
  - AGENT_LEARNING_AGENT_INTEGRATION_OPT_EXAMPLE_KEY=local-offline agent-learn optimize examples/agent_integration_optimization.json --output artifacts/agent-integration.json
  - agent-learn actions artifacts/agent-integration.json --output actions.json --markdown actions.md
  - agent-learn action-run artifacts/agent-integration.json --id report_agent_integration_readiness --dry-run --output action-run.json
postcondition: python -c "import json; p=json.load(open('actions.json')); assert p['kind']=='agent-learning.actions.v1', p['kind']; assert 'report_agent_integration_readiness' in p['summary']['action_ids'], p['summary']['action_ids']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
opt_in_lane: false
---

# Artifact Actions: every result knows its next step

> **Twin:** [`examples/sdk_agent_integration_optimization.py`](../../examples/sdk_agent_integration_optimization.py)
> · emits `agent-learning.optimization.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

The expensive part of a test result is rarely producing it — it is deciding
what to do with it. A failed optimization sits in an artifacts directory until
someone reconstructs the rerun command, the report invocation, or the
promotion step by hand, and every reconstruction is a chance to run the wrong
thing. Kit artifacts close that loop by embedding their own follow-up
operations: an `actions` array of typed, parameterized commands the artifact
itself declares valid.

`agent-learn actions <artifact>` lists that catalog as
`agent-learning.actions.v1`. The agent-integration optimization used here
embeds fifteen actions, including `report_agent_integration_readiness`,
`rerun_agent_integration_optimization`, `promote_to_regression`, and
`promote_diagnosed_regression` — reporting, rerun, re-optimization, and
promotion paths, each carrying its exact `command_args` and target layers.
Every artifact also gets a generic `report_artifact` action, so the catalog is
never empty. `agent-learn action-run --id <id>` executes one action;
`--dry-run` resolves the full command without running it, which is the right
first move in CI.

The backing twin builds and optimizes the agent-integration manifest through
the SDK — the provider-matrix scenario (chat, voice, WebRTC, phone, SIP)
whose artifact you interrogate below.

## 2. Run it

Produce an artifact, list its actions, then resolve one without executing:

```bash
AGENT_LEARNING_AGENT_INTEGRATION_OPT_EXAMPLE_KEY=local-offline \
agent-learn optimize examples/agent_integration_optimization.json \
  --output artifacts/agent-integration.json

agent-learn actions artifacts/agent-integration.json \
  --output actions.json --markdown actions.md

agent-learn action-run artifacts/agent-integration.json \
  --id report_agent_integration_readiness --dry-run \
  --output action-run.json
```

Relative outputs resolve against your current working directory, so
`actions.json`, `actions.md`, and `action-run.json` land where you run the
commands. Drop `--dry-run` to execute; add
`--input name=value` for actions whose `requires_input` is true.

The same operations from the SDK:

```python
from fi.alk import actions

artifact = actions.load_artifact_file("artifacts/agent-integration.json")
catalog = actions.action_catalog(artifact, source_path="artifacts/agent-integration.json")
result = actions.run_action(
    artifact,
    "report_agent_integration_readiness",
    source_path="artifacts/agent-integration.json",
    dry_run=True,
)
```

## 3. What you built

Postcondition (machine-checkable — same shape the docs gate enforces):

```bash
python -c "import json; p=json.load(open('actions.json')); assert p['kind']=='agent-learning.actions.v1', p['kind']; assert 'report_agent_integration_readiness' in p['summary']['action_ids'], p['summary']['action_ids']; print('ok')"
```

`actions.json` carries each action's id, label, kind (`cli` or download),
`command_args`, `target_layers`, and `requires_input`, plus a summary with
`action_count` and `source_kind`. The Markdown rendering is a reviewable
table of the same catalog. The action-run artifact
(`agent-learning.action-run.v1`) records the resolved command, declared
outputs and whether they exist, and the exit code — dry-run or real.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| action id not found | catalog mismatch | rerun `agent-learn actions` and copy an id from `summary.action_ids` |
| `missing required environment variable(s)` | config fault | export the manifest's `required_env` key with any placeholder value |
| action-run outputs marked `exists: false` | wrong working directory | set `--cwd` so the action's relative outputs land where you expect |

## 5. Prove it / keep it

The promotion actions are the bridge from one-off run to standing regression:
`promote_to_regression` turns this artifact's winning configuration into a
baseline your CI replays. Wire the loop as artifact → `actions` → `action-run`
in your pipeline so the follow-up command is always the one the artifact
declared, never one reconstructed from memory. Suite artifacts carry action
catalogs too — produce one in [trinity-suite](trinity-suite.md), and gate the
promotion itself with [trust-certificates](trust-certificates.md).
