---
kind: agent-learning.docs-page.v1
track: prove
objective: reliability
stage: prove
backing:
  - examples/sdk_framework_adapter_trinity_suite.py
artifact_kinds:
  - agent-learning.suite.v1
commands:
  - for key in $(python -c "import json; print(' '.join(json.load(open('examples/agent_learning_suite.json'))['required_env']))"); do export "$key=local-offline"; done
  - agent-learn suite examples/agent_learning_suite.json --output artifacts/suite.json --junit artifacts/suite.junit.xml --markdown artifacts/suite.md
postcondition: python -c "import json; p=json.load(open('artifacts/suite.json')); assert p['kind']=='agent-learning.suite.v1', p['kind']; assert p['trust_certificate']['verdict']=='approved', p['trust_certificate']['verdict']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# The Trinity Suite: one manifest, one verdict

> **Twin:** [`examples/sdk_framework_adapter_trinity_suite.py`](../../examples/sdk_framework_adapter_trinity_suite.py)
> · emits `agent-learning.suite.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

Most agent pipelines run simulation, evaluation, red-teaming, and optimization
as separate jobs in separate tools. Each job can be green while the combined
picture is not: the simulated agent is not the one that was evaluated, the
red-team campaign ran against last week's prompt, the optimizer's winner was
never re-simulated. The failure class is promotion on partial evidence.

A suite manifest closes that gap. `examples/agent_learning_suite.json` declares
24 jobs — simulation, multi-framework adapter runs, promptfoo-style evals,
artifact evals, red-team campaigns, and a dozen optimizers — plus
`required_capabilities` that pin the commands and result kinds the installed
kit must support before any job runs. One command executes everything and
returns one artifact with one exit code and an embedded trust certificate
(`agent-learning.suite.trust-certificate.v1`, verdict `approved` /
`conditional` / `rejected`, `promotion_ready`, `assurance_level`).

The backing twin runs the same machinery end to end on a local framework
adapter: it writes a complete trinity-suite workspace (simulation cases,
red-team attacks, eval criteria) for a scripted orchestrator class, then runs
the suite through `suite.run_suite_file` — in a few seconds, fully offline.

## 2. Run it

The `required_env` keys on example manifests are CI metadata, not credentials;
any placeholder value satisfies them offline:

```bash
for key in $(python -c "import json; print(' '.join(json.load(open('examples/agent_learning_suite.json'))['required_env']))"); do export "$key=local-offline"; done

agent-learn suite examples/agent_learning_suite.json \
  --output artifacts/suite.json \
  --junit artifacts/suite.junit.xml \
  --markdown artifacts/suite.md
```

Relative `--output` paths resolve against your current working directory,
so the artifacts land in `artifacts/`.

The same operation from the SDK:

```python
from fi.alk import suite

result = suite.run_suite_file("examples/agent_learning_suite.json")
print(result["trust_certificate"]["verdict"])
```

## 3. What you built

Postcondition (machine-checkable — same shape the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/suite.json')); assert p['kind']=='agent-learning.suite.v1', p['kind']; assert p['trust_certificate']['verdict']=='approved', p['trust_certificate']['verdict']; print('ok')"
```

`artifacts/suite.json` contains every child result keyed by job id,
a `summary` with `executed_count`, `failed_count`, `capability_gate_passed`,
`evidence_gate_passed`, and `framework_coverage`, and the `trust_certificate`
block. The JUnit file gives your CI one test case per job; the Markdown file
is the human report for the same run.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `missing required environment variable(s)` | config fault | export the listed keys with any placeholder value (step 1 above) |
| `capability_gate_passed: false` in the summary | manifest/kit mismatch | `summary.public_boundary_passed` + the `required_capabilities` block |
| verdict `conditional` instead of `approved` | evidence gap | read `trust_certificate.conditions` and the failing job's child result |

## 5. Prove it / keep it

The suite artifact is the input to the rest of the prove track. Verify its
certificate without re-running anything in
[trust-certificates](trust-certificates.md); list and execute the follow-up
operations embedded in the artifact in [actions](actions.md); wire the same
one-command pattern into your pipeline with `agent-learn init . --preset all`,
whose scaffold README carries the suite command and outputs. Keep the suite in
CI: the manifest is the regression spec, and every run re-earns the verdict
instead of inheriting it.
