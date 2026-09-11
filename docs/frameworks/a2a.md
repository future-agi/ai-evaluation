---
kind: agent-learning.docs-page.v1
track: frameworks
objective: behavior
stage: simulate
backing:
  - examples/sdk_framework_adapter_cert_a2a.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - python examples/sdk_framework_adapter_cert_a2a.py artifacts/framework-cert-a2a.json
postcondition: python -c "import json; p=json.load(open('artifacts/framework-cert-a2a.json')); assert p['status']=='passed', p['status']; assert p['method']=='send_message' and p['input_mode']=='dict'; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# A2A: offline framework-adapter simulation

> **Twin:** [`examples/sdk_framework_adapter_cert_a2a.py`](../../examples/sdk_framework_adapter_cert_a2a.py)
> · emits the framework-adapter probe evidence · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

Agent2Agent (A2A) ships a preset in `FRAMEWORK_PRESETS`: method `send_message`,
input mode `dict`. This page certifies that preset is real — the twin,
[`examples/sdk_framework_adapter_cert_a2a.py`](../../examples/sdk_framework_adapter_cert_a2a.py),
builds a local `LocalA2ASession` exposing exactly `send_message` (a `message`
kwarg plus a `session` side-kwarg) and wraps it through the same `wrap_framework`
/ `run_framework_adapter_probe` path a manifest uses, so the probe exercises the
real adapter resolution. The shim returns contract-shaped synthetic evidence — a
`framework_trace` event and a `framework_trace_status` tool call — and **never
imports a real a2a-sdk and never touches the network**.

The failure class this catches is preset drift: if the `send_message`/`dict`
shape no longer resolves through the adapter, the probe fails and the preset is
corrected. The IO surface this preset binds to is `side_kwargs` (the message
kwarg + session metadata pattern, like pipecat's frame kwarg) — one of the eight
existing framework-adapter IO contracts, classified by adapter shape.

A2A is deliberately doubly-covered: this certification probe keeps the closed
required set homogeneous, while the deeper A2A protocol surfaces live in the live
lane and the protocol-trace example (see section 5).

## 2. Run it

CLI — the twin is executable and writes the probe artifact:

```bash
python examples/sdk_framework_adapter_cert_a2a.py artifacts/framework-cert-a2a.json
```

SDK, same operation:

```python
from sdk_framework_adapter_cert_a2a import run  # examples/ on sys.path

result = run("artifacts/framework-cert-a2a.json")
assert result["status"] == "passed"
assert result["method"] == "send_message" and result["input_mode"] == "dict"
```

## 3. What you built

Postcondition (machine-checkable):

```bash
python -c "import json; p=json.load(open('artifacts/framework-cert-a2a.json')); assert p['status']=='passed', p['status']; assert p['method']=='send_message' and p['input_mode']=='dict'; print('ok')"
```

The artifact carries the resolved method/input mode, the framework runtime trace,
and the message round-trip evidence the adapter extracted — a replayable record
the release gate re-executes.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` -> `summary.missing_engine_modules` |
| probe `status` is `failed` (method did not resolve) | behavior regression | confirm `send_message`/`dict` against the current A2A SDK; if drifted, add a `V1_FRAMEWORK_PRESET_CORRECTIONS` row and fix the preset |
| public-boundary mismatch | config fault | `agent-learn doctor` -> `summary.public_boundary_passed` |

## 5. Prove it / keep it

The twin is admitted by the `framework_adapter_preset_certification_readiness`
release gate, so every `agent-learn release-check` re-executes this exact probe —
the page stays true or the release fails.

For the deeper A2A protocol surfaces, two existing artifacts go beyond preset
certification: the A2A live lane
([`src/fi/alk/live/a2a_lane.py`](../../src/fi/alk/live/a2a_lane.py))
and the protocol-trace example
([`examples/sdk_framework_adapter_a2a_protocol_trace.py`](../../examples/sdk_framework_adapter_a2a_protocol_trace.py),
admitted by `protocol_adapter_readiness`), which export the agent card, the A2A
event stream, and the per-task lifecycle. To keep your own A2A endpoint honest,
promote the run artifact into a regression baseline with the `baseline` /
`promote-to-regression` / `compare` command family.
