---
kind: agent-learning.docs-page.v1
track: prove
objective: capability
stage: prove
backing:
  - examples/sdk_framework_adapter_capability_profiles.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - AGENT_LEARNING_RUN_EXAMPLE_KEY=local-offline agent-learn run examples/run_manifest.json --no-eval --output artifacts/run.json
  - agent-learn capabilities artifacts/run.json --require commands=run,redteam,suite --require result_kinds=agent-learning.run.v1 --output capabilities.json --quiet
  - python examples/sdk_framework_adapter_capability_profiles.py artifacts/capability-profiles.json
postcondition: python -c "import json; c=json.load(open('capabilities.json')); p=json.load(open('artifacts/capability-profiles.json')); assert c['summary']['capability_gate_passed'] is True, c['summary']; assert p['passed'] is True and p['framework_count']==5, p; print('ok')"
claims: []
doctor_checks:
  - missing_public_modules
  - missing_engine_modules
opt_in_lane: false
---

# Capability Catalogs: pin what your pipeline assumes

> **Twin:** [`examples/sdk_framework_adapter_capability_profiles.py`](../../examples/sdk_framework_adapter_capability_profiles.py)
> · emits `agent-learning.framework-adapter-capability-profiles.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

Pipelines accumulate silent assumptions: that the installed kit still exposes
the `suite` command, still supports the `pipecat` adapter, still emits
`agent-learning.run.v1`. Nothing checks those assumptions until a minor
upgrade breaks one and the failure surfaces three jobs downstream with an
unrelated error message. `agent-learn capabilities` turns the assumptions into
a gate. It reports the installed kit's static capabilities — commands,
frameworks, providers, channels, environment types, metrics, result kinds —
merges in capabilities observed from any saved artifacts you pass, and fails
(exit code 1) when a `--require key=value` pin is not satisfied.

The backing twin answers the per-framework version of the same question
without importing a single framework: it builds the adapter contract matrix
for langchain, langgraph, openai_agents, livekit, and pipecat and derives a
portable capability profile per framework — which method and input mode the
adapter binds, which capabilities (`tool_calls`, `streaming_trace`,
`voice_frames`) the contract supports, and which simulate/eval/optimize
bindings apply. That bundle is how you decide, in CI, whether a framework
target is even eligible for the lane you are about to run.

## 2. Run it

Produce a run artifact, then gate on required capabilities and derive the
per-framework profiles:

```bash
AGENT_LEARNING_RUN_EXAMPLE_KEY=local-offline \
agent-learn run examples/run_manifest.json --no-eval --output artifacts/run.json

agent-learn capabilities artifacts/run.json \
  --require commands=run,redteam,suite \
  --require result_kinds=agent-learning.run.v1 \
  --output capabilities.json --quiet

python examples/sdk_framework_adapter_capability_profiles.py \
  artifacts/capability-profiles.json
```

Relative outputs resolve against your current working directory: the catalog
lands in `capabilities.json`, the profiles bundle in
`artifacts/capability-profiles.json`.

The same operations from the SDK:

```python
from fi.alk import actions, capabilities, simulate

artifact = actions.load_artifact_file("artifacts/run.json")
catalog = capabilities.capability_catalog(
    [artifact],
    required_capabilities={"commands": ["run", "redteam", "suite"]},
)
matrix = simulate.framework_adapter_contract_matrix(["langchain", "langgraph"])
profiles = simulate.framework_adapter_capability_profiles(matrix=matrix)
```

## 3. What you built

Postcondition (machine-checkable — same shape the docs gate enforces):

```bash
python -c "import json; c=json.load(open('capabilities.json')); p=json.load(open('artifacts/capability-profiles.json')); assert c['summary']['capability_gate_passed'] is True, c['summary']; assert p['passed'] is True and p['framework_count']==5, p; print('ok')"
```

The catalog separates `static_capabilities` (what the installed kit supports)
from `observed_capabilities` (what your artifacts actually exercised), and its
summary records the pins under `required_capabilities` with any
`missing_required_capabilities` named. The profiles bundle holds one profile
per framework with its contract, capability list with categories, evidence
requirements, and library bindings — plus a top-level `passed` and
`framework_count`.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| exit code 1 with `findings[]` populated | a capability pin is unmet | read `summary.missing_required_capabilities` — it names key and value |
| facade import errors in the SDK path | broken install | `agent-learn doctor` → `summary.missing_public_modules` |
| profile `passed: false` for a framework | contract gap | read that profile's `findings` and `evidence_requirements` |

## 5. Prove it / keep it

Commit the `--require` pins next to your pipeline definition and run the
capabilities gate first in every lane: an upgrade that drops a command or a
result kind then fails in seconds with the missing pin named, instead of
failing later inside a job that assumed it. Capability evidence also feeds
the suite's own `required_capabilities` gate
([trinity-suite](trinity-suite.md)) and the kit-level verdict in
[release-check-in-your-ci](release-check-in-your-ci.md).
