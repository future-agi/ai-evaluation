---
kind: agent-learning.docs-page.v1
track: eval
objective: behavior
stage: evaluate
backing:
  - examples/sdk_task_evaluation.py
artifact_kinds:
  - agent-learning.eval.v1
commands:
  - agent-learn eval examples/eval_suite.json --output artifacts/eval-suite.json
  - agent-learn eval examples/eval_suite.json --dry-run
postcondition: python -c "import json; p=json.load(open('artifacts/eval-suite.json')); assert p['kind']=='agent-learning.eval.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
opt_in_lane: false
---

# Eval suites

> **Twin:** [`examples/sdk_task_evaluation.py`](../../examples/sdk_task_evaluation.py)
> · emits `agent-learning.eval.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

That a fixed set of prompts, run through a named provider, still satisfies
your assertions. A suite file has four parts — `providers`, `prompts`,
`tests`, and an optional threshold — and `examples/eval_suite.json` is the
minimal complete instance: one `echo` provider, one `{{question}}` prompt
template, one test with a `contains` assertion. The `version` field is the
artifact kind the run will emit, `agent-learning.eval.v1`.

The failure class is behavioral drift: a prompt edit, a provider swap, or a
template variable change that silently alters what your users see. Suites
catch it because the assertions are declarative — `contains`,
`not_contains` — and the run is reproducible: the echo provider returns its
input deterministically, so the example suite passes or fails for reasons
entirely inside the file.

Suites also scale past string checks. The `artifact` provider type (see
`examples/artifact_task_eval_suite.json`) reads fields out of a saved run
artifact by path, so the same test/assert grammar can interrogate metric
averages and environment state from a real run — that pattern gets its own
treatment in [artifact evals](artifact-evals.md). The covering gate for this
page's twin executes that artifact-backed suite end-to-end on every
release-check, which is what admits this page.

## 2. Run it

CLI — validate the shape first, then execute:

```bash
agent-learn eval examples/eval_suite.json --dry-run
agent-learn eval examples/eval_suite.json --output artifacts/eval-suite.json
```

SDK — the same operation:

```python
from fi.alk import evals

suite = evals.load_eval_suite_file("examples/eval_suite.json")
result = evals.run_eval_suite_file("examples/eval_suite.json")
```

To build a suite in code instead of JSON, use
`evals.build_eval_suite_manifest(name=..., providers=..., prompts=...,
tests=..., threshold=...)` — `examples/sdk_eval_suite.py` is a complete
scripted builder (it expects `AGENT_LEARNING_SDK_EVAL_SUITE_KEY` in the
environment before it will run). `--output` also accepts `.xml` (JUnit) and
`.sarif` paths, and `--markdown` writes a human report — the same artifact,
three renderings.

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/eval-suite.json')); assert p['kind']=='agent-learning.eval.v1', p['kind']; print('ok')"
```

The artifact records every test with its rendered prompt, provider output,
per-assertion verdicts, an aggregate score against the suite threshold, and
an `exit_code` — `agent-learn eval` returns it as the process exit code, so
the suite is CI-ready with no wrapper script.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| suite rejected at load | config fault | run `--dry-run`; the error line names the bad field |
| assertion fails on the echo provider | test fault | the asserted value is not in the rendered prompt itself |

## 5. Prove it / keep it

Wrap the suite in a suite-of-suites job so it runs with the rest of your
checks: `examples/sdk_eval_suite.py` builds exactly that wrapper with
`suite.build_suite_manifest(...)`, declaring `eval` in
`required_capabilities.commands` so the runner refuses to silently skip it.
From there, `agent-learn release-check` re-executes the suite on every cut.
When the suite guards a fix you shipped, pin the passing artifact as a
baseline (`agent-learn baseline`) and compare future runs against it — the
regression lifecycle is the same one the simulate track uses.
