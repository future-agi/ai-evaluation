---
kind: agent-learning.docs-page.v1
track: redteam
objective: safety
stage: simulate
backing:
  - examples/sdk_redteam_attack_evolution_optimization.py
artifact_kinds:
  - agent-learning.redteam.v1
commands:
  - agent-learn redteam-corpus --corpus examples/redteam_corpus.json --output artifacts/redteam-corpus.json
postcondition: python -c "import json; p=json.load(open('artifacts/redteam-corpus.json')); assert p['kind']=='agent-learning.redteam.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Red-team corpora: benchmark rows as campaign evidence

> **Twin:** [`examples/sdk_redteam_attack_evolution_optimization.py`](../../examples/sdk_redteam_attack_evolution_optimization.py)
> · emits `agent-learning.redteam.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

Published attack benchmarks are corpora: rows with an `attack_type`, a
`surface`, a `channel`, a `prompt`, and an `expected_behavior`. The failure
class this page closes is corpus rot — a spreadsheet of benchmark prompts that
nobody can map to campaign coverage, so nobody knows which rows the agent has
actually been tested against. `agent-learn redteam-corpus` imports rows and
emits the same campaign-evidence contract every other red-team run uses:
coverage cells, executed cells, findings, and mitigations.

`examples/redteam_corpus.json` carries 12 rows drawn from published benchmark
taxonomies (each row names its `benchmark`, `taxonomy`, and `source` arXiv
link — for example `redbench` rows citing arXiv:2601.03699 and `dtap` rows
citing arXiv:2605.04808), spanning indirect prompt injection through tool,
environment, and memory surfaces. The command needs no credentials at all:
local corpus mode reads the file, builds the campaign matrix, and verifies
every row maps to a covered, executed, mitigated cell. The hook variant
(`--hook <endpoint>`) does the same against an authenticated HTTP corpus
source and records the fetch trace in the artifact.

Corpus rows are also the seed format for attack evolution: the twin on this
page starts from exactly such seed attacks and mutates them across operators —
see [attack-evolution-shrink](attack-evolution-shrink.md).

## 2. Run it

CLI:

```bash
agent-learn redteam-corpus --corpus examples/redteam_corpus.json \
  --output artifacts/redteam-corpus.json
```

SDK, same operation:

```python
import json

from fi.alk import redteam

rows = json.load(open("examples/redteam_corpus.json"))["rows"]
campaign = redteam.build_redteam_corpus_campaign(
    name="redteam-corpus-campaign",
    corpus_rows=rows,
)
assert campaign["summary"]["covered_cell_count"] == len(rows)
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/redteam-corpus.json')); assert p['kind']=='agent-learning.redteam.v1', p['kind']; print('ok')"
```

`artifacts/redteam-corpus.json` reports `row_count: 12`,
`coverage_cell_count: 12`, `covered_cell_count: 12`, `finding_count: 12`, and
`mitigation_count: 12` — one matrix cell, finding, and mitigation per imported
row — plus `blocking_gap_count`, which drives the exit code: any uncovered,
unexecuted, or unmitigated cell fails the run. `summary.source` records the
corpus mode (`local_file` vs `hook`) and provenance, so the evidence says
where every row came from.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `provide exactly one of --corpus/--corpus-file or --hook` | config fault | pick one source mode per invocation |
| status `failed` with `blocking_gap_count > 0` | real gap | inspect `summary` for the missing coverage/mitigation cells |
| hook returns no rows | config fault | `agent-learn doctor` → `summary.public_boundary_passed`, then check the hook trace in the artifact |

## 5. Prove it / keep it

A corpus import that passes today is a baseline, not a conclusion. Wire the
same command into CI so new corpus rows must arrive with coverage and
mitigation evidence, then graduate rows in two directions: evolve them into
stronger variants ([attack-evolution-shrink](attack-evolution-shrink.md)), and
promote any row that produces a real finding into a permanent regression
([promote-to-regression](promote-to-regression.md)).
