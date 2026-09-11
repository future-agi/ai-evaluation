---
kind: agent-learning.docs-page.v1
track: optimize
objective: reliability
stage: optimize
backing:
  - examples/sdk_evaluation_hook_probe_optimization.py
artifact_kinds:
  - agent-learning.eval-optimization.v1
commands:
  - agent-learn optimize-eval examples/eval_suite_optimization.json --output artifacts/eval-optimization.json
postcondition: python -c "import json; p=json.load(open('artifacts/eval-optimization.json')); assert p['kind']=='agent-learning.eval-optimization.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Eval-Suite Optimization

> **Twin:** [`examples/sdk_evaluation_hook_probe_optimization.py`](../../examples/sdk_evaluation_hook_probe_optimization.py)
> · emits `agent-learning.optimization.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

The eval track runs a promptfoo-style suite and reports pass/fail. This page
closes the loop: when a suite fails, search the response space for the
configuration that passes it. The committed suite
([`examples/eval_suite_optimization.json`](../../examples/eval_suite_optimization.json))
contains an echo provider, a scripted provider whose response is
"Private credentials only.", and one test asserting the answer mentions the
policy. Its optimization block declares the search space —
`providers.1.response` with two candidates: the original response and a
policy-grounded answer — at threshold 1.0: every assertion must pass, or no
candidate is accepted. The failure class is eval-driven guesswork: editing
the agent by hand until a suite goes green, with no record of what was tried
and why the final answer was chosen.

| Archetype | Dharma (what it may change) | Constraint (what it must preserve) |
| --- | --- | --- |
| Transformer (proposer) | candidate config: harness, memory, tooling | the task contract |
| Critic | scores + objections | evidence admissibility |
| Mediator | candidate retention/merge | lineage continuity |
| Steward (preserver) | rollback / veto | governance + the regression baseline |

In eval-suite optimization the suite's assertions sit in the critic's row —
they are the objections every candidate must answer — and the threshold of
1.0 is the steward's veto stated as a number. The backing twin probes the
adjacent surface: it optimizes evaluation-hook configurations the same way,
confirming the evaluator side of this loop is itself exercised by a release
gate.

## 2. Run it

CLI:

```bash
agent-learn optimize-eval examples/eval_suite_optimization.json \
  --output artifacts/eval-optimization.json
```

SDK, the same operation:

```python
from fi.alk import optimize

result = optimize.optimize_eval_suite_response(
    name="sdk-eval-suite-optimization",
    metadata={"cookbook": "sdk-eval-suite-optimization"},
)
```

`optimize-eval` also accepts `--max-candidates` to cap the search and the
same `--junit` / `--sarif` / `--markdown` outputs as `agent-learn eval`.

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/eval-optimization.json')); assert p['kind']=='agent-learning.eval-optimization.v1', p['kind']; print('ok')"
```

Note the kind: `agent-learning.eval-optimization.v1`, not the plain
optimization kind — the artifact records candidates scored against suite
assertions rather than simulation metrics, with the winning response and the
per-test results that justify it.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| suite rejected | config fault | `summary.public_boundary_passed` + the suite error line |
| no candidate passes at threshold 1.0 | candidate fault | inspect per-test assertion results in the artifact |

## 5. Prove it / keep it

The optimized response is only as good as the suite that selected it — and
suites have their own failure modes. Before trusting an LLM-judged
assertion, read `../eval/judge-reliability.md`. The suite itself, run
without optimization, is `../eval/eval-suites.md`; once the winning
configuration is fixed, keep the suite in CI so the green state is
continuously re-earned.
