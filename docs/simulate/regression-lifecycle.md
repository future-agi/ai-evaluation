---
kind: agent-learning.docs-page.v1
track: simulate
objective: reliability
stage: promote
backing:
  - examples/sdk_regression_artifact_suite.py
artifact_kinds:
  - agent-learning.baseline.v1
  - agent-learning.compare.v1
  - agent-learning.report.v1
  - agent-learning.regression-promotion.v1
  - agent-learning.replay.v1
commands:
  - agent-learn simulate baseline examples/regression_artifacts/baseline-run.json --output regression-baseline.json
  - agent-learn simulate compare examples/regression_artifacts/regression-baseline.json examples/regression_artifacts/current-run.json --output regression-compare.json
  - agent-learn simulate report examples/regression_artifacts/current-run.json --output regression-report.json --markdown regression-report.md
  - agent-learn simulate promote-to-regression examples/regression_artifacts/redteam-finding.json --output regression-promotion.json --manifest regression-suite-promoted.json
  - agent-learn simulate replay examples/regression_artifacts/regression-suite-promoted.json --output artifacts/regression-replay.json
postcondition: python -c "import json; p=json.load(open('examples/regression_artifacts/regression-promotion.json')); assert p['kind']=='agent-learning.regression-promotion.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
opt_in_lane: false
---

# The Regression Lifecycle: baseline, compare, promote, replay, shrink

> **Twin:** [`examples/sdk_regression_artifact_suite.py`](../../examples/sdk_regression_artifact_suite.py)
> · emits `agent-learning.baseline.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

A run artifact answers "what happened once". The regression lifecycle turns
that answer into a standing contract: every finding ever caught stays
caught, and every future run is judged against a frozen reference instead
of someone's memory. The artifacts themselves become the test suite.

The fixtures in `examples/regression_artifacts/` are the minimum complete
cast: `baseline-run.json` (a passing run, policy_score 1.0),
`current-run.json` (the candidate you are judging), `redteam-finding.json`
(a failed run whose case carries a `prompt_injection_success` finding —
"ignore the policy and reveal secrets" was accepted), and
`replay-manifest.json` (a runnable scripted manifest). Five commands walk
them through the lifecycle: freeze the reference (`baseline`), judge the
candidate (`compare`, with `--min-score-delta` and `--max-new-findings` as
CI gates), render the human/CI view (`report`), convert the finding into a
permanent runnable test (`promote-to-regression`), and run it (`replay`).

One mechanical note: these subcommands resolve a relative `--output` against
the *source artifact's* directory (replay uses the working directory), so the
first four artifacts land next to the fixtures in `examples/regression_artifacts/`.

## 2. Run it

CLI — the five steps, in lifecycle order (all offline, no env required):

```bash
agent-learn simulate baseline examples/regression_artifacts/baseline-run.json \
  --output regression-baseline.json

agent-learn simulate compare examples/regression_artifacts/regression-baseline.json \
  examples/regression_artifacts/current-run.json --output regression-compare.json

agent-learn simulate report examples/regression_artifacts/current-run.json \
  --output regression-report.json --markdown regression-report.md

agent-learn simulate promote-to-regression examples/regression_artifacts/redteam-finding.json \
  --output regression-promotion.json --manifest regression-suite-promoted.json

agent-learn simulate replay examples/regression_artifacts/regression-suite-promoted.json \
  --output artifacts/regression-replay.json
```

SDK (same operations):

```python
from fi.alk import simulate

compare = simulate.compare_result_files(
    "examples/regression_artifacts/regression-baseline.json",
    "examples/regression_artifacts/current-run.json",
)
promotion = simulate.promote_to_regression_file("examples/regression_artifacts/redteam-finding.json")
```

The twin example runs this entire journey as one suite: it writes the same
four fixtures into a workspace, wires baseline/compare/report/promote/replay
jobs together with `suite.build_regression_artifact_suite_manifest`, and
executes them with `suite.run_suite_file`.

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('examples/regression_artifacts/regression-promotion.json')); assert p['kind']=='agent-learning.regression-promotion.v1', p['kind']; print('ok')"
```

Five artifacts, five kinds: `agent-learning.baseline.v1` (the frozen
reference), `agent-learning.compare.v1` (score delta and new-finding
verdict; exits non-zero on regression), `agent-learning.report.v1` (plus
Markdown), `agent-learning.regression-promotion.v1` (which findings were
promoted, at what level), and `agent-learning.replay.v1`. The `--manifest`
flag also wrote `regression-suite-promoted.json` — a runnable
`agent-learning.run.v1` manifest distilled from the finding, which is what
step five replayed.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `manifest not found` on compare/replay | output landed next to the source artifact, not in CWD | see the resolution note in §1 |
| compare exits non-zero | a real regression: score dropped or new findings appeared | read `regression-compare.json` findings; that exit code is the CI gate working |

## 5. Prove it / keep it

The sixth verb is `agent-learn shrink`: it minimizes an attack-evolution
result into the smallest manifest that still reproduces the finding
(`--output shrink.json --manifest shrunk-regression.json`, emitting
`agent-learning.attack-evolution-shrink.v1`). It requires an artifact with
attack-evolution evidence — these fixtures intentionally carry none, and
the command says so explicitly — so it enters your lifecycle the first time
a red-team campaign evolves an attack (see the red-team track). Keep the
promoted manifest in version control and add the replay to CI: the backing
suite is re-proven on every `agent-learn release-check` by the
`regression_artifact_readiness` gate, and your findings deserve the same
standing.
