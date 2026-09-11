---
kind: agent-learning.docs-page.v1
track: reference
backing: []
artifact_kinds: []
commands:
  - agent-learn run examples/run_manifest.json --no-eval --output artifacts/run.json
postcondition: python -c "import json; p=json.load(open('artifacts/run.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
opt_in_lane: false
---

# Artifact Reference

> **Twin:** none — reference page (`backing: []`). Every kind below is emitted
> by a command in [reference/cli.md](cli.md) and verified by the postcondition
> pattern shown in section 3.

## 1. What you are testing

Every `agent-learn` command that produces evidence writes a JSON artifact with
a top-level `kind` field. The kind universe is closed: the docs gate rejects a
page that claims to emit a kind outside `V1_DOCS_ALLOWED_ARTIFACT_KINDS`
(`src/fi/alk/trinity.py`), and `agent-learn release-check` asserts the
eleven core kinds in `V1_REQUIRED_SCHEMA_KINDS` are producible. The closed set
is what makes postconditions one-liners — checking `payload["kind"]` is always
sufficient to know what you are holding.

Two values look like kinds but are not artifact kinds:
`agent-learning.cli.v1` is the CLI payload `schema_version` label, and any
vendored `agent-simulate.*` value is rewritten to its public
`agent-learning.*` form by `public_schema_value` /
`normalize_public_payload` in `src/fi/alk/_schema.py` before an
artifact is written.

## 2. Run it

Produce one artifact and inspect its kind (offline, no credentials):

```bash
agent-learn run examples/run_manifest.json --no-eval --output artifacts/run.json
```

```python
import json
payload = json.load(open("artifacts/run.json"))
print(payload["kind"])  # agent-learning.run.v1
```

## 3. What you built

```bash
python -c "import json; p=json.load(open('artifacts/run.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
```

The full catalog. "Core" marks the eleven `V1_REQUIRED_SCHEMA_KINDS` asserted
by the release gate; the remainder come from the public command registry.

| Kind | Core | Emitted by |
| --- | --- | --- |
| `agent-learning.run.v1` | yes | `agent-learn run` — one simulation run: transcript, world/task state, optional eval attachments |
| `agent-learning.eval.v1` | yes | `agent-learn eval` — eval-suite verdicts over prompts and outputs |
| `agent-learning.artifact-evaluation.v1` | yes | `agent-learn eval-artifact` — evaluation computed over an already-saved artifact |
| `agent-learning.task-evidence.v1` | — | `agent-learn eval-task` — synthesized task-evidence record from task artifacts |
| `agent-learning.redteam.v1` | yes | `agent-learn redteam` / `redteam-corpus` — campaign findings and corpus-hook results |
| `agent-learning.optimization.v1` | yes | `agent-learn optimize` — candidate history with content-addressed lineage |
| `agent-learning.eval-optimization.v1` | yes | `agent-learn optimize-eval` — optimization over an eval suite itself |
| `agent-learning.suite.v1` | yes | `agent-learn suite` — combined multi-step suite result |
| `agent-learning.suite-optimization.v1` | yes | `agent-learn optimize-suite` / `action-optimize` — optimization over a suite |
| `agent-learning.actions.v1` | yes | `agent-learn actions` — the available-actions catalog |
| `agent-learning.action-run.v1` | yes | `agent-learn action-run` — one executed action with its result |
| `agent-learning.release-proof.v1` | yes | `agent-learn release-proof` — the seven-check release proof object |
| `agent-learning.baseline.v1` | — | `agent-learn baseline` — pinned regression baseline |
| `agent-learning.compare.v1` | — | `agent-learn compare` — baseline-vs-current comparison verdict |
| `agent-learning.init.v1` | — | `agent-learn init` — scaffold record for a preset |
| `agent-learning.regression-promotion.v1` | — | `agent-learn promote-to-regression` — a finding promoted into the regression set |
| `agent-learning.attack-evolution-shrink.v1` | — | `agent-learn shrink` — minimized counterexample from an evolved attack |
| `agent-learning.replay.v1` | — | `agent-learn replay` — deterministic re-execution verdict for a kept artifact |
| `agent-learning.report.v1` | — | `agent-learn report` — rendered report over saved artifacts |
| `agent-learning.doctor.v1` | — | `agent-learn doctor` — environment and module diagnostics |
| `agent-learning.release-check.v1` | — | `agent-learn release-check` — the full local gate matrix verdict |

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `kind` holds an `agent-simulate.*` value | engine — artifact written without `_schema` normalization | `agent-learn doctor` → `summary.missing_engine_modules` |
| `KeyError: 'kind'` reading an artifact | config fault — file is not an agent-learn artifact (or pre-v1) | re-emit with the current CLI; check the `--output` path |
| docs gate rejects a page's `artifact_kinds` | config fault — value outside the closed kind set | compare against the table above (the gate payload mirrors it) |

## 5. Prove it / keep it

`agent-learn release-check --project-root .` asserts the required kinds and
mirrors the allowed set in its evidence payload
(`docs_allowed_artifact_kinds`), so the catalog above cannot drift silently
from the code. To put that check in your pipeline, continue with
[prove/release-check-in-your-ci.md](../prove/release-check-in-your-ci.md); to
keep a specific artifact as a regression baseline, continue with
[simulate/regression-lifecycle.md](../simulate/regression-lifecycle.md).
