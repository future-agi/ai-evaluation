---
kind: agent-learning.docs-page.v1
track: simulate
objective: capability
stage: simulate
backing:
  - examples/sdk_framework_adapter_browser_cua_trace.py
artifact_kinds:
  - agent-learning.run.v1
  - agent-learning.optimization.v1
commands:
  - AGENT_LEARNING_SDK_MULTIMODAL_IMAGE_SIMULATION_KEY=offline-demo-key python examples/sdk_multimodal_image_simulation.py artifacts/multimodal-image.json
  - AGENT_LEARNING_MULTIMODAL_IMAGE_OPT_EXAMPLE_KEY=offline-demo-key agent-learn optimize examples/multimodal_image_optimization.json --output artifacts/multimodal-image-optimization.json
postcondition: python -c "import json; p=json.load(open('artifacts/multimodal-image.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Multimodal Image: simulate grounding, not just looking

> **Twin:** [`examples/sdk_framework_adapter_browser_cua_trace.py`](../../examples/sdk_framework_adapter_browser_cua_trace.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

An agent that "supports images" can still fail in two distinct ways: it
never actually reads the image (answering from the text around it), or it
reads the image and hallucinates contents that are not there. Both pass a
demo; neither survives a grounding check. The kit's `multimodal_image`
environment makes the image a checkable fixture: each image carries a URI,
a description, `labels`, and `ocr_text` ground truth (the example fixture is
a Contoso receipt — total $42.00, paid, refund eligible), and the evaluator
scores whether the agent's claims about the image match that ground truth.

The simulation example builds this with
`simulate.build_multimodal_image_run_manifest` and runs it on the local
deterministic engine, with `auto_execute_tools` so image tools like
`list_images` resolve against the fixture. The optimization manifest goes a
step further: its search space contrasts a weak fixture ("without OCR or
labels") against the fully labeled one, so the optimizer must prefer the
environment in which grounding is verifiable.

This page's gate-covered twin is the kit's other image-bearing surface: the
browser computer-use trace adapter, whose export carries screenshots,
region-anchored actions (selector plus x/y/width/height),
`screenshot_diffs` with per-region scores, and stale-screenshot refresh
metadata. Visual evidence is one evidence family in the kit; this page
exercises its image-fixture form, and the twin keeps the screenshot form
proven on every release-check.

## 2. Run it

CLI (placeholder env values are CI wiring metadata; both engines are local):

```bash
AGENT_LEARNING_SDK_MULTIMODAL_IMAGE_SIMULATION_KEY=offline-demo-key \
  python examples/sdk_multimodal_image_simulation.py artifacts/multimodal-image.json

AGENT_LEARNING_MULTIMODAL_IMAGE_OPT_EXAMPLE_KEY=offline-demo-key \
  agent-learn optimize examples/multimodal_image_optimization.json \
  --output artifacts/multimodal-image-optimization.json
```

Note: `agent-learn` resolves a relative `--output` against your current
working directory, so the second artifact lands at
`artifacts/multimodal-image-optimization.json`.

SDK (same operation as the first command):

```python
import asyncio
import os
from fi.alk import simulate

os.environ.setdefault("AGENT_LEARNING_SDK_MULTIMODAL_IMAGE_SIMULATION_KEY", "offline-demo-key")
manifest = simulate.build_multimodal_image_run_manifest(name="multimodal-image-simulation")
simulate.write_manifest_file(manifest, "multimodal-image.manifest.json")
result = asyncio.run(simulate.run_manifest_file("multimodal-image.manifest.json"))
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/multimodal-image.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
```

The run artifact records the image environment the case ran against and the
grounding evidence the evaluator scored — which labeled facts the agent's
answer matched. The optimization artifact records which environment
candidate won and why the labeled fixture beat the unlabeled one.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| manifest rejected / `images must contain at least one environment` | config fault | `summary.public_boundary_passed` + the manifest error line |
| case scores low | answer not grounded in the fixture's `labels`/`ocr_text` | compare the case response against the image data in the artifact |

## 5. Prove it / keep it

Swap the fixture for your own: pass `images=[{...}]` with your URIs, labels,
and OCR ground truth to `build_multimodal_image_run_manifest`, keep the
threshold, and re-run. A passing artifact then enters the standard spine —
baseline it and wire compare into CI via
[`regression-lifecycle.md`](regression-lifecycle.md). For full
computer-use visual flows (screenshots, region diffs, injected DOM
adversaries), the browser-use page in `docs/frameworks/` builds on the same
twin this page is admitted by.
