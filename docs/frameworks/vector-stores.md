---
kind: agent-learning.docs-page.v1
track: frameworks
objective: behavior
stage: simulate
backing:
  - examples/sdk_retrieval_hook_optimization.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - python examples/sdk_retrieval_hook_optimization.py artifacts/retrieval-hook.json
postcondition: python -c "import json; p=json.load(open('artifacts/retrieval-hook.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Vector stores: offline retrieval-hook simulation

> **Twin:** [`examples/sdk_retrieval_hook_optimization.py`](../../examples/sdk_retrieval_hook_optimization.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

A vector store is the world a retrieving agent reaches into, not an agent to
wrap. It has no turn, no policy, and no tool-selection decision, so it is **not**
a `FRAMEWORK_PRESETS` row and gets no agent-preset probe. Its home is the
`RetrievalHookEnvironment` (`name = "retrieval_hook"`,
[`src/fi/simulate/environment.py`](../../src/fi/simulate/environment.py)), which
normalizes a `.query()` / `.search()` call into `retrieval_memory` trace
evidence. `_normalize_retrieval_response` assigns a `retrieval_rank` and a
`retrieval_score` to every returned document, and
`_probe_retrieval_memory_summary`
([`src/fi/simulate/agent/frameworks.py`](../../src/fi/simulate/agent/frameworks.py))
folds documents / queries / citations / memory writes into a checkable summary.

The failure class this catches is retrieval-shape drift: each vendor returns hits
in its own container (Chroma `documents` + `distances`, Pinecone `matches`,
Qdrant `points`, …), and a harness that reads the wrong key silently retrieves
nothing while the agent still answers. The retrieval hook makes the normalized
document shape — `{id, content, retrieval_rank, retrieval_score, metadata}` — a
checkable field of the artifact, the same for all nine vendors.

The twin runs against a local fixture retrieval endpoint: offline,
deterministic, no provider keys. A connection to a real vendor index is an
optional fidelity check, never required.

## 2. Run it

CLI — the twin is executable and writes both the run artifact and the manifest it
ran:

```bash
python examples/sdk_retrieval_hook_optimization.py artifacts/retrieval-hook.json
```

SDK, same operation:

```python
from sdk_retrieval_hook_optimization import run  # examples/ on sys.path

result = run("artifacts/retrieval-hook.json")
assert result["kind"] == "agent-learning.run.v1"
```

A real-vendor connection is optional (`◐`): point the retrieval hook at a live
index and re-run the same path; the normalization contract is identical, so the
artifact shape does not change.

## 3. What you built

Postcondition (machine-checkable — the same check the docs gate pattern uses):

```bash
python -c "import json; p=json.load(open('artifacts/retrieval-hook.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
```

The artifact carries the retrieval-memory summary — per-document
`retrieval_rank` / `retrieval_score`, namespaces, and doc ids — that the
`RetrievalHookEnvironment` extracted from the synthetic hit list.

### Per-vendor response shape

Each vendor's native hit list normalizes to the same document shape
`{id, content, retrieval_rank, retrieval_score, metadata}`:

| Vendor | Native hit container | Normalized via |
| --- | --- | --- |
| chromadb | `documents` + `distances` (parallel lists) | distance → `retrieval_score`, list index → `retrieval_rank` |
| lancedb | `to_list()` rows with `_distance` | `_distance` → `retrieval_score`, row order → `retrieval_rank` |
| milvus | `hits` with `distance` per hit | `distance` → `retrieval_score`, hit order → `retrieval_rank` |
| mongodb-vector | `$vectorSearch` aggregation rows with `score` | `score` → `retrieval_score`, row order → `retrieval_rank` |
| pgvector | SQL rows with a `distance`/`similarity` column | column → `retrieval_score`, row order → `retrieval_rank` |
| pinecone | `matches[].score` | `score` → `retrieval_score`, match order → `retrieval_rank` |
| qdrant | `points[].score` | `score` → `retrieval_score`, point order → `retrieval_rank` |
| redis-vector | `Documents` with a vector-score field | score field → `retrieval_score`, doc order → `retrieval_rank` |
| weaviate | `objects` with `_additional.distance`/`certainty` | additional metric → `retrieval_score`, object order → `retrieval_rank` |

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `missing required environment variable(s)` | config fault | set the placeholder key; `agent-learn doctor` → `summary.public_boundary_passed` |
| `retrieval_memory` summary empty (wrong hit key) | behavior regression | confirm the vendor row above and re-map the hit container to the normalized shape |

## 5. Prove it / keep it

The twin is admitted by the `retrieval_hook_readiness` release gate, so every
`agent-learn release-check` re-executes this exact retrieval-hook path — the page
stays true or the release fails. The nine vendors are deliberately **excluded**
from `FRAMEWORK_PRESETS`: the
`framework_adapter_preset_certification_readiness` gate positively asserts none
of them is registered as an agent preset, so a maintainer who adds one fails the
release. To keep your own index honest, promote the run artifact into a
regression baseline with the `baseline` / `promote-to-regression` / `compare`
command family.
