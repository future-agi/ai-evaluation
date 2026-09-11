---
kind: agent-learning.docs-page.v1
track: optimize
objective: reliability
stage: optimize
backing:
  - examples/sdk_image_improvement.py
artifact_kinds:
  - agent-learning.practice-report.v1
  - agent-learning.run.v1
commands:
  - python examples/sdk_image_improvement.py artifacts/image-improvement.json
postcondition: python -c "import json; p=json.load(open('artifacts/image-improvement.json')); assert p['kind']=='agent-learning.image-improvement.v1', p['kind']; print('ok')"
claims:
  - phrase: Image improvement loop
    gate_id: image_loop_readiness
  - phrase: perception-bypass
    gate_id: image_loop_readiness
  - phrase: trainer
    gate_id: practice_loop_readiness
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: true
---

# Image improvement loop: the 13D Practice Loop on image

> **Twin:** [`examples/sdk_image_improvement.py`](../../examples/sdk_image_improvement.py)
> · emits `agent-learning.practice-report.v1` + `agent-learning.run.v1` · offline,
> no credentials, deterministic. A coding agent can complete this page from the
> frontmatter alone.

This wires the multimodal-task evals as a **loss** and runs the generic 13D
Practice Loop on `world.kind=image`. No new optimizer is invented — the existing
six-phase trainer runs over image cells. The loss is **multi-objective and
deterministic-anchored**: every declared image objective MUST carry at least one
deterministic ground-truth anchor (`task_success` / `ocr_accuracy` /
`chart_accuracy` / `artifact_grounding`). A judge-only image objective is
structurally rejected — there is no judge-only image loss.

**Honesty disclaimer (load-bearing).** The deterministic loss runs
credential-free and stays `local_gate` / `captured_fixture` (a
`deterministic_fixture` artifact, never `live_lane`). The judge-anchored terms
and the entire `generation` profile are opt-in keyed lanes, never a release
prerequisite.

## 1. What you are tuning

The SCOPED UPDATE optimizes the **whole multimodal agent**
(`target_kind=whole_agent`): model, vision prompt, instructions, tool routing —
plus the config-only knobs no text optimizer reaches: **image
preprocessing/resolution** (`image.preprocess.*`, the Fix-Before-Search knob) and
**multimodal-RAG** config (`mmrag.*`). The loss carries a mandatory
perception-bypass Goodhart guard: perception-bypass sentinels (items answerable
from language priors alone) and perceptual-counterfactual canaries (a minimally
edited twin where the right answer flips). A genuinely-perceiving config DROPS its
score on the counterfactual twin; a perception-bypassing config does not — that is
the tell.

## 2. Run it

```bash
python examples/sdk_image_improvement.py artifacts/image-improvement.json
```

SDK (the operation the twin performs):

```python
from fi.alk import image_loop

manifest = image_loop.build_image_practice_loop_manifest(
    name="image-improvement",
    base_agent={"model": "gpt-4o"},
    search_space={
        "image.preprocess.resolution": [256, 512, 1024],
        "mmrag.retrieve_images": [True, False],
        "agent.vision_prompt": ["describe the scene", "extract every value"],
    },
    objective=objective,  # multi-objective, >= 1 deterministic anchor, guarded
    eval_budget=6, seed=1142,
)
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/image-improvement.json')); assert p['kind']=='agent-learning.image-improvement.v1', p['kind']; print('ok')"
```

The artifact holds the compiled multi-objective guarded loss, the constructed
judge-only / single-term rejections, the whole-agent search space (incl.
`image.preprocess.*` + `mmrag.*`), the loop-vs-no-loop A/B at equal budget (the
held-out-battery capstone with the canary holding), and the image-sublayer
attribution per weak cell (`preprocessing` / `perception` / `reasoning` /
`tool_grounding`).

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `image_loss_guard_missing` | config fault | a judge-only / single-term image objective |
| `objective_guards_missing` | config fault | a declared loss with no Goodhart guards |
| public boundary error | config fault | `summary.public_boundary_passed` |

## 5. Prove it / keep it

The twin runs fresh on every `agent-learn release-check` via the
`image_loop_readiness` gate. The A/B capstone is the credential-free proof that
the loop beats no-loop on a held-out image battery with the perception-bypass
canary holding. For the keyed real-VLM lane and the `generation` profile, see the
roadmap — both are owner-keyed opt-in lanes, never release prerequisites. To
inspect the deterministic substrate, see [image-loop](../simulate/image-loop.md).
