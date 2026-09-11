---
kind: agent-learning.docs-page.v1
track: frameworks
objective: capability
stage: simulate
backing:
  - examples/sdk_framework_adapter_typed_output.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - python examples/sdk_framework_adapter_typed_output.py artifacts/profile-typed_output.json
postcondition: python -c "import json; p=json.load(open('artifacts/profile-typed_output.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Profile: Typed output

> **Twin:** [`examples/sdk_framework_adapter_typed_output.py`](../../../examples/sdk_framework_adapter_typed_output.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

`typed_output` is one of the eight executable framework-adapter IO contracts. It
captures a structured/typed object the adapter coerces from the framework result. Any framework preset whose adapter shape matches
(`execute_task` / `dict`) inherits this contract — the surface binds by adapter
shape, not by framework identity. The twin,
[`examples/sdk_framework_adapter_typed_output.py`](../../../examples/sdk_framework_adapter_typed_output.py), drives a local shim through the real adapter path and
asserts the contract on the emitted evidence.

The failure class this catches is silent capability loss: a framework can return
an acceptable answer while dropping the typed_output evidence the contract is built
around. This page documents what that evidence is and which gate keeps it honest.

## 2. Run it

CLI:

```bash
python examples/sdk_framework_adapter_typed_output.py artifacts/profile-typed_output.json
```

SDK, same operation:

```python
from sdk_framework_adapter_typed_output import run  # examples/ on sys.path

result = run("artifacts/profile-typed_output.json")
assert result["kind"] == "agent-learning.run.v1"
```

## 3. What you built

The contract asserts: `typed_output` state carrying the schema name and the decoded decision (e.g. `schema == RefundDecision`, `decision.verdict == approved`).

Postcondition (machine-checkable):

```bash
python -c "import json; p=json.load(open('artifacts/profile-typed_output.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
```

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` -> `summary.missing_engine_modules` |
| the typed_output evidence is missing or shrunk | behavior regression | re-run the twin and diff the state observations against this contract |
| public-boundary mismatch | config fault | `agent-learn doctor` -> `summary.public_boundary_passed` |

## 5. Prove it / keep it

The twin is admitted by the `framework_adapter_io_readiness` release gate, so every
`agent-learn release-check` re-executes this exact typed_output contract — the page
stays true or the release fails. To document the profile for one of your own
frameworks, point a cookbook page's `backing` at a shim that emits the same
typed_output evidence.
