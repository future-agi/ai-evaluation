---
kind: agent-learning.docs-page.v1
track: simulate
objective: capability
stage: simulate
backing:
  - examples/sdk_image_loop.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - python examples/sdk_image_loop.py artifacts/image-loop.json
postcondition: python -c "import json; p=json.load(open('artifacts/image-loop.json')); assert p['kind']=='agent-learning.image-loop.v1', p['kind']; print('ok')"
claims:
  - phrase: image-improvement-loop
    gate_id: image_loop_readiness
  - phrase: perception-bypass-guard
    gate_id: image_loop_readiness
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: true
---

# Image loop: the deterministic multimodal substrate, credential-free

> **Twin:** [`examples/sdk_image_loop.py`](../../examples/sdk_image_loop.py)
> · emits `agent-learning.run.v1` · offline, no credentials, deterministic.
> A coding agent can complete this page from the frontmatter alone.

This is the image / multimodal loop substrate. It runs **entirely in-process**
on committed synthetic PNG fixtures (a rendered chart, a text page, a multi-object
scene, a perceptual-counterfactual pair) — no network, no keys, no model — over
the already-shipped `ImageEnvironment` (`list_images` / `inspect_image`). Same
seed in, byte-identical loop trajectory and perturbation rasters out.

**Honesty disclaimer (load-bearing).** A deterministic in-process fixture is
**NOT a live lane**. Every deterministic artifact carries
`fidelity_tier: "deterministic_fixture"` and an evidence class of `local_gate`
(or `captured_fixture` when stored) — **never `live_lane`**. `live_lane` is
reserved for the keyed real-VLM lane. The gate fails any deterministic fixture
that claims `live_lane` (the `image_fidelity_overclaim` tripwire).

## 1. What you are testing

Multimodal agents fail on perception, not just on words: a chart value misread,
an OCR string dropped under compression, an object misidentified, a tool argument
extracted wrong from the image. The loop exercises exactly those signals
deterministically. `world.kind=image` enters the world-kind space through the R4
registry hook — it is admissible WITHOUT widening the frozen `SIMULATION_WORLD_KINDS`
tuple, so it is "typed → executable": typed the moment it is registered, executable
the moment its rung-1 fixture run is green, never silently claimed.

## 2. Run it

```bash
python examples/sdk_image_loop.py artifacts/image-loop.json
```

SDK (the operation the twin performs):

```python
from fi.alk import image_loop, image_perturb
from fi.simulate.environment import ImageEnvironment

image_loop._ensure_image_world_registered()
env = ImageEnvironment({"chart": "examples/image_loop_fixture/chart_synthetic.png"})
out = image_perturb.apply_image_perturbations(
    raster, operators=["blur", "jpeg_compress", "resolution_drop", "occlusion"],
    seed=1142,
)
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/image-loop.json')); assert p['kind']=='agent-learning.image-loop.v1', p['kind']; print('ok')"
```

The artifact holds the loop-determinism proof (byte-identical perturbation
rasters + an identical stanza under the pinned seed), the deterministic anchors
(exact-match / ANLS / relaxed-accuracy / token-overlap grounding reproducible
over the fixtures), the perception-bypass-guard outcome (the sentinel delta + the
counterfactual control that **drops** the score for a genuinely-perceiving
config), and the constructed overclaim negatives the gate must catch.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `image_fixture_missing` | config fault | a committed PNG/JSON fixture path is missing/unreadable |
| `image_fidelity_overclaim` in the gate | overclaim | a deterministic fixture stamped `evidence_class: live_lane` |
| public boundary error | config fault | `summary.public_boundary_passed` |

## 5. Prove it / keep it

The twin runs fresh on every `agent-learn release-check` via the
`image_loop_readiness` gate (eight evidence arrays, all credential-free). The
perception-bypass-guard outcome is honest by computation: it is licensed only
while that gate is green. For the keyed real-VLM lane (the only honest
`live_lane`), see the roadmap — it is an owner-keyed opt-in lane, never a release
prerequisite. To tune the whole multimodal agent against these signals, see
[image-improvement](../optimize/image-improvement.md).
