---
kind: agent-learning.docs-page.v1
track: prove
objective: reliability
stage: prove
backing:
  - examples/sdk_trinity_stack_probe_optimization.py
artifact_kinds:
  - agent-learning.release-check.v1
  - agent-learning.init.v1
  - agent-learning.run.v1
  - agent-learning.redteam.v1
commands:
  - agent-learn init ci-workspace --preset ci --quiet
  - agent-learn run ci-workspace/manifests/run.json --output artifacts/run.json --junit artifacts/run.junit.xml --sarif artifacts/run.sarif.json --quiet
  - agent-learn redteam ci-workspace/manifests/redteam.json --output artifacts/redteam.json --junit artifacts/redteam.junit.xml --sarif artifacts/redteam.sarif.json --quiet
  - agent-learn release-check --project-root . --output artifacts/release-check.json --quiet
postcondition: python -c "import json; p=json.load(open('artifacts/release-check.json')); assert p['kind']=='agent-learning.release-check.v1', p['kind']; assert p['summary']['ready'] is True, p['summary']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Release-Check in Your CI: a verdict you can re-run

> **Twin:** [`examples/sdk_trinity_stack_probe_optimization.py`](../../examples/sdk_trinity_stack_probe_optimization.py)
> · emits `agent-learning.optimization.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

When you take a dependency on an agent-testing kit, you inherit its failure
modes. Vendor claims about what a library can do are usually a README; the
question an enterprise reviewer actually asks is "show me the check, and let
me run it myself." `agent-learn release-check` answers that question. It
executes every release gate of the kit — packaging, public-boundary,
framework-adapter, red-team, optimizer, and docs gates — against a source
checkout and emits `agent-learning.release-check.v1`: per-gate `checks[]` with
evidence, milestone rollups, and `summary.ready`. The heavier cut,
`agent-learn release-proof`, re-runs `release_check` plus ruff, pytest, and the
package build, and emits one `agent-learning.release-proof.v1` artifact with
`summary.ready`. Both are ordinary CLI commands: your CI can run them on the
kit checkout you vendor, on a schedule, or before every upgrade — the trust
object is an artifact your pipeline produced, not a statement you read.

The same pattern scales down to your own project. `agent-learn init` scaffolds
a CI lane in any directory: manifests, an artifacts directory, a regressions
directory, and a README carrying the exact next commands. The scaffolded
manifests run offline by default — no env requirement unless you opt in with
`--required-env`. The backing twin shows what a gate is made of: it stands up
a local HTTP agent, probes an orchestration stack, and scores the result —
the same probe-and-score loop `release-check` applies to the kit itself.

## 2. Run it

Scaffold the CI lane, run it with CI-native outputs, then check the kit:

```bash
agent-learn init ci-workspace --preset ci --quiet

agent-learn run ci-workspace/manifests/run.json \
  --output artifacts/run.json \
  --junit artifacts/run.junit.xml \
  --sarif artifacts/run.sarif.json --quiet

agent-learn redteam ci-workspace/manifests/redteam.json \
  --output artifacts/redteam.json \
  --junit artifacts/redteam.junit.xml \
  --sarif artifacts/redteam.sarif.json --quiet

agent-learn release-check --project-root . \
  --output artifacts/release-check.json --quiet
```

JUnit files plug into any CI test reporter; SARIF files plug into
code-scanning surfaces. Relative `--output` paths resolve against your
current working directory, so everything above lands under `artifacts/`.

The same release check from the SDK:

```python
from fi.alk import trinity

payload = trinity.release_status(project_root=".")
assert payload["summary"]["ready"], payload["summary"]["failed_check_count"]
```

## 3. What you built

Postcondition (machine-checkable — same shape the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/release-check.json')); assert p['kind']=='agent-learning.release-check.v1', p['kind']; assert p['summary']['ready'] is True, p['summary']; print('ok')"
```

`artifacts/release-check.json` carries `summary.check_count`,
`summary.passed_check_count`, the full `checks[]` array (each check has an id,
milestone, status, and an `evidence` object you can diff between runs), and
the package name and version the verdict applies to. The scaffolded
`ci-workspace/` is yours to commit: manifests are the spec, `artifacts/` is
gitignored output, `regressions/` is where promoted baselines live.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `summary.ready: false` | a named gate regressed | read the failing entry in `checks[]` — `reason` names the gate, `evidence` shows why |
| run/redteam artifacts not where expected | path resolution | a relative `--output` lands in your current working directory — use absolute paths if your CI step changes directories |
| scaffold files already exist | re-init without `--force` | rerun with `--force`, or init into a fresh directory |

## 5. Prove it / keep it

Pin `agent-learn release-check --project-root <kit checkout>` as a job in your
pipeline and archive `artifacts/release-check.json` with each build: upgrades
to the kit then arrive with a verdict attached, and a regression in the
dependency fails your build with a named gate instead of a mystery. For
release events, run `agent-learn release-proof --project-root .` and keep the
single proof artifact. Your own manifests graduate the same way: promote green
runs into `regressions/` (see [actions](actions.md)) and let the suite page
([trinity-suite](trinity-suite.md)) collapse the whole lane into one verdict.
