---
kind: agent-learning.docs-page.v1
track: eval
objective: reliability
stage: evaluate
backing:
  - examples/sdk_judge_reliability_evaluation.py
artifact_kinds:
  - agent-learning.eval.v1
commands:
  - python examples/sdk_judge_reliability_evaluation.py artifacts/judge-reliability.json
postcondition: python -c "import json; p=json.load(open('artifacts/judge-reliability.json')); assert p['kind']=='agent-learning.eval.v1', p['kind']; assert p['status']=='passed', p['status']; print('ok')"
claims: []
doctor_checks:
  - missing_public_modules
opt_in_lane: false
---

# Judge reliability

> **Twin:** [`examples/sdk_judge_reliability_evaluation.py`](../../examples/sdk_judge_reliability_evaluation.py)
> · emits `agent-learning.eval.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

The judge, not the agent. Every other page in this track trusts a scoring
function; this page asks what that trust is worth. A scorer that moves when
the answer is reformatted, padded, or reworded is measuring presentation,
not quality — and every threshold built on it inherits that noise.

The method is perturbation analysis: hold the content fixed, vary the
surface, and measure how far the score moves. The twin runs a scripted
rubric judge (term coverage with a verbosity penalty — pure Python, fully
deterministic) over three fixed support answers, then perturbs each answer
along three axes: **formatting** (the prose is rewritten as a bulleted list
under a header), **verbosity** (a fixed filler paragraph is appended), and
**paraphrase** (hardcoded rewordings that preserve the rubric facts — no
LLM calls anywhere). For every sample × axis pair it records the score
delta and whether the pass/fail verdict flipped, then asserts the maximum
delta against a tolerance of 0.15 and requires verdict agreement of 1.0.

The reference run is informative in both directions: formatting and
paraphrase deltas come out 0.0 (the judge normalizes case and whitespace,
and the paraphrases keep the rubric anchors), while the verbosity axis
shows a real, bounded delta of 0.05 — the judge's length penalty firing on
the padded variants. A reliability report should look like this: zeros
where the judge is invariant by construction, measured non-zeros where it
is sensitive by design, and an assertion that the sensitivity stays inside
the band you chose.

## 2. Run it

CLI:

```bash
python examples/sdk_judge_reliability_evaluation.py artifacts/judge-reliability.json
```

SDK — the same operation:

```python
from examples.sdk_judge_reliability_evaluation import run

result = run("artifacts/judge-reliability.json")
print(result["summary"]["axes"])
```

To apply the method to your own judge, replace `judge_score` and `SAMPLES`
in a copy of the twin: keep the three perturbation functions and the
agreement bookkeeping, and tighten or loosen `AGREEMENT_DELTA_TOLERANCE` to
the band your thresholds can absorb.

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/judge-reliability.json')); assert p['kind']=='agent-learning.eval.v1', p['kind']; assert p['status']=='passed', p['status']; print('ok')"
```

The artifact's `summary` holds the agreement metrics: per-axis
`mean_score_delta`, `max_score_delta`, and `verdict_agreement`, plus the
overall maximum delta, the tolerance it was asserted against, and the judge
pass threshold. `results` lists every sample × axis comparison with base
score, perturbed score, delta, and the flip flag — the full evidence behind
the verdict, not just the verdict.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `import fi.alk` fails | infra | `agent-learn doctor` → `summary.missing_public_modules` |
| `judge agreement drift exceeded tolerance` | real catch | your judge moved more than the tolerance under a surface change |
| paraphrase deltas large, others zero | rubric fault | rubric anchors are phrasings, not facts — the paraphrase rewords them away |

## 5. Prove it / keep it

This example is executed by the docs gate itself on every
`agent-learn release-check`, so the agreement assertion is already a
standing check for the kit. Do the same for your own judge: keep your copy
of the twin in the repo, run it in CI next to the suites that depend on
that judge, and treat a tolerance breach as a build failure — a judge that
drifted invalidates every score it produced since the last green run. When
you change the judge deliberately, re-run this first, then re-baseline the
suites that consume it.
