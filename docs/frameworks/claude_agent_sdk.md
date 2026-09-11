---
kind: agent-learning.docs-page.v1
track: frameworks
objective: behavior
stage: simulate
backing:
  - examples/sdk_framework_adapter_cert_claude_agent_sdk.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - python examples/sdk_framework_adapter_cert_claude_agent_sdk.py artifacts/framework-cert-claude_agent_sdk.json
postcondition: python -c "import json; p=json.load(open('artifacts/framework-cert-claude_agent_sdk.json')); assert p['status']=='passed', p['status']; assert p['method']=='query' and p['input_mode']=='text'; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Claude Agent SDK: offline framework-adapter simulation

> **Twin:** [`examples/sdk_framework_adapter_cert_claude_agent_sdk.py`](../../examples/sdk_framework_adapter_cert_claude_agent_sdk.py)
> · emits the framework-adapter probe evidence · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

Claude Agent SDK ships a preset in `FRAMEWORK_PRESETS`: method `query`, input mode
`text`. This page certifies that preset is real — the twin,
[`examples/sdk_framework_adapter_cert_claude_agent_sdk.py`](../../examples/sdk_framework_adapter_cert_claude_agent_sdk.py), builds a local shim exposing exactly `query` and
wraps it through the same `wrap_framework` / `run_framework_adapter_probe` path a
manifest uses, so the probe exercises the real adapter resolution against the
single-turn text query that yields a transcript. The shim returns contract-shaped synthetic evidence — a
`framework_trace` event and a `framework_trace_status` tool call — and **never
imports the real Claude Agent SDK package and never touches the network**.

The failure class this catches is preset drift: if the `query`/`text` shape no
longer resolves through the adapter, the probe fails and the preset is corrected.
The IO surface this preset binds to is `message_history` (one of the eight existing
framework-adapter IO contracts), classified by adapter shape, not reinvented.

## 2. Run it

CLI — the twin is executable and writes the probe artifact:

```bash
python examples/sdk_framework_adapter_cert_claude_agent_sdk.py artifacts/framework-cert-claude_agent_sdk.json
```

SDK, same operation:

```python
from sdk_framework_adapter_cert_claude_agent_sdk import run  # examples/ on sys.path

result = run("artifacts/framework-cert-claude_agent_sdk.json")
assert result["status"] == "passed"
assert result["method"] == "query" and result["input_mode"] == "text"
```

## 3. What you built

Postcondition (machine-checkable):

```bash
python -c "import json; p=json.load(open('artifacts/framework-cert-claude_agent_sdk.json')); assert p['status']=='passed', p['status']; assert p['method']=='query' and p['input_mode']=='text'; print('ok')"
```

The artifact carries the resolved method/input mode, the framework runtime trace,
and the tool/event evidence the adapter extracted — a replayable record the
release gate re-executes.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` -> `summary.missing_engine_modules` |
| probe `status` is `failed` (method did not resolve) | behavior regression | confirm `query`/`text` against the current Claude Agent SDK SDK; if drifted, add a `V1_FRAMEWORK_PRESET_CORRECTIONS` row and fix the preset |
| public-boundary mismatch | config fault | `agent-learn doctor` -> `summary.public_boundary_passed` |

## 5. Prove it / keep it

The twin is admitted by the `framework_adapter_preset_certification_readiness`
release gate, so every `agent-learn release-check` re-executes this exact probe —
the page stays true or the release fails. To keep your own Claude Agent SDK integration
honest, promote the run artifact into a regression baseline with the `baseline` /
`promote-to-regression` / `compare` command family.
