---
kind: agent-learning.docs-page.v1
track: frameworks
objective: behavior
stage: simulate
backing:
  - examples/sdk_framework_adapter_cert_deepseek.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - python examples/sdk_framework_adapter_cert_deepseek.py artifacts/framework-cert-deepseek.json
postcondition: python -c "import json; p=json.load(open('artifacts/framework-cert-deepseek.json')); assert p['status']=='passed', p['status']; assert p['method']=='chat' and p['input_mode']=='dict'; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# DeepSeek: offline framework-adapter simulation

> **Twin:** [`examples/sdk_framework_adapter_cert_deepseek.py`](../../examples/sdk_framework_adapter_cert_deepseek.py)
> · emits the framework-adapter probe evidence · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

DeepSeek ships a preset in `FRAMEWORK_PRESETS`: method `chat`, input mode
`dict`. This page certifies that preset is real — the twin,
[`examples/sdk_framework_adapter_cert_deepseek.py`](../../examples/sdk_framework_adapter_cert_deepseek.py), builds a local shim exposing exactly `chat` and
wraps it through the same `wrap_framework` / `run_framework_adapter_probe` path a
manifest uses, so the probe exercises the real adapter resolution against the
OpenAI-compatible chat response. The shim returns contract-shaped synthetic evidence — a
`framework_trace` event and a `framework_trace_status` tool call — and **never
imports the real DeepSeek package and never touches the network**.

The failure class this catches is preset drift: if the `chat`/`dict` shape no
longer resolves through the adapter, the probe fails and the preset is corrected.
The IO surface this preset binds to is `provider_response` (one of the eight existing
framework-adapter IO contracts), classified by adapter shape, not reinvented.

## 2. Run it

CLI — the twin is executable and writes the probe artifact:

```bash
python examples/sdk_framework_adapter_cert_deepseek.py artifacts/framework-cert-deepseek.json
```

SDK, same operation:

```python
from sdk_framework_adapter_cert_deepseek import run  # examples/ on sys.path

result = run("artifacts/framework-cert-deepseek.json")
assert result["status"] == "passed"
assert result["method"] == "chat" and result["input_mode"] == "dict"
```

The credential-free probe above is the release bar. An optional `◐` live-validation run proves the real provider:

```bash
DEEPSEEK_API_KEY=... agent-learn probe deepseek --live
```

This is owner-keyed, opt-in, and recorded once in the live-validation lane — it is **never** a release prerequisite.

## 3. What you built

Postcondition (machine-checkable):

```bash
python -c "import json; p=json.load(open('artifacts/framework-cert-deepseek.json')); assert p['status']=='passed', p['status']; assert p['method']=='chat' and p['input_mode']=='dict'; print('ok')"
```

The artifact carries the resolved method/input mode, the framework runtime trace,
and the tool/event evidence the adapter extracted — a replayable record the
release gate re-executes.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` -> `summary.missing_engine_modules` |
| probe `status` is `failed` (method did not resolve) | behavior regression | confirm `chat`/`dict` against the current DeepSeek SDK; if drifted, add a `V1_FRAMEWORK_PRESET_CORRECTIONS` row and fix the preset |
| public-boundary mismatch | config fault | `agent-learn doctor` -> `summary.public_boundary_passed` |

## 5. Prove it / keep it

The twin is admitted by the `framework_adapter_preset_certification_readiness`
release gate, so every `agent-learn release-check` re-executes this exact probe —
the page stays true or the release fails. To keep your own DeepSeek integration
honest, promote the run artifact into a regression baseline with the `baseline` /
`promote-to-regression` / `compare` command family.
