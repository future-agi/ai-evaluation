---
kind: agent-learning.docs-page.v1
track: quickstart
backing: []
artifact_kinds:
  - agent-learning.optimization.v1
commands:
  - agent-learn init . --preset optimize
  - agent-learn optimize manifests/optimize.json --dry-run
  - agent-learn optimize manifests/optimize.json --output artifacts/optimization.json --junit artifacts/optimization.junit.xml --sarif artifacts/optimization.sarif.json --markdown artifacts/optimization.md
postcondition: python -c "import json; p=json.load(open('artifacts/optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - api_key_configured
opt_in_lane: false
---

# Golden path: first optimization

> **Twin:** the `agent-learn init --preset optimize` scaffold (`backing: []` —
> scaffold-backed by rule; proven offline by `tests/test_init_golden_paths.py`)
> · emits `agent-learning.optimization.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

Three commands take you from an empty directory to an optimization artifact,
offline. The scaffold is a complete, deliberately small task-world problem: a
weak agent, a world contract it fails, and a search space that contains the
fix. Running it shows the whole optimization mechanic — candidate generation,
simulation against the world, scoring, selection — on a problem where you can
read every moving part.

`manifests/optimize.json` sets up a refund world: a `world_contract` with
actors (`agent`, `customer`), a `refund` resource starting at
`status: pending`, an invariant that policy allows refunds, and a success
condition requiring `refund.status: approved`. The seed agent is `scripted`
and weak by design — it inspects the refund request but never applies the
`approve_refund` transition. The search space offers two axes: whether the
agent issues the `apply_world_transition` tool call, and whether the world
declares the `approve_refund` transition at all. Exactly one combination
reaches terminal success; the optimizer (`algorithm: agent`, up to 5
candidates, seed included) has to find it against an `agent_report` threshold
of 0.95 weighted toward `world_contract_quality` and tool selection.

The `--dry-run` step validates the manifest and previews the candidate plan
without executing simulations — the cheap correctness check you will keep
using on real manifests before paying for full optimization runs.

## 2. Run it

CLI:

```bash
agent-learn init . --preset optimize
agent-learn optimize manifests/optimize.json --dry-run
agent-learn optimize manifests/optimize.json --output artifacts/optimization.json \
  --junit artifacts/optimization.junit.xml --sarif artifacts/optimization.sarif.json \
  --markdown artifacts/optimization.md
```

SDK, same operation:

```python
from fi.alk import optimize

preview = optimize.optimize_manifest_file("manifests/optimize.json", dry_run=True)
result = optimize.optimize_manifest_file("manifests/optimize.json")
assert result["kind"] == "agent-learning.optimization.v1"
```

## 3. What you built

Postcondition (machine-checkable — the same check the scaffold README carries):

```bash
python -c "import json; p=json.load(open('artifacts/optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
```

`artifacts/optimization.json` records every candidate evaluated — its config
delta against the seed, its simulation outcome against the world contract, its
score — plus the selected candidate and its lineage. The same run emits JUnit,
SARIF, and Markdown renderings of the identical evidence, so CI, code-scanning
UIs, and humans read one result in their native formats.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` from any command | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| key/credential errors on a platform-connected lane | config fault | `agent-learn doctor` → `summary.api_key_configured` |
| `init would overwrite existing file(s); use --force` | scaffold state | not a doctor fault — rerun with `--force` or use a clean directory |

The whole path runs with no API key set; `required_env` in the scaffolded
manifest is CI metadata, as the per-preset golden-path test proves.

## 5. Prove it / keep it

An optimization that is not kept is a one-off experiment. The scaffold README
lists the extended lifecycle for exactly this: `agent-learn report` on the
optimization, `agent-learn promote-to-regression` to freeze the winning
candidate as `regressions/optimized-regression.json`, and `agent-learn replay`
on that regression manifest so every future change is checked against the
improvement you just found. The promote → replay loop is covered in
[`../simulate/regression-lifecycle.md`](../simulate/regression-lifecycle.md).

To point this machinery at your own agent — real harness, memory, tooling, and
whole-agent search spaces instead of the refund toy — continue with
[`../optimize/optimize-any-agent.md`](../optimize/optimize-any-agent.md).
