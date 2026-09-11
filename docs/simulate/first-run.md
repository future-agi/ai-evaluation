---
kind: agent-learning.docs-page.v1
track: simulate
objective: behavior
stage: simulate
backing:
  - examples/sdk_agent_integration_simulation.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - AGENT_LEARNING_RUN_EXAMPLE_KEY=offline-demo-key agent-learn run examples/run_manifest.json --output artifacts/first-run.json
postcondition: python -c "import json; p=json.load(open('artifacts/first-run.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Your First Simulation Run

> **Twin:** [`examples/sdk_agent_integration_simulation.py`](../../examples/sdk_agent_integration_simulation.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

> **New here?** This page is the manifest/CLI front door. To see the same run
> as the underlying `SimulationSpec` + `SimulationRunner` — the spine every
> simulation flows through — start with [Spec + Runner](spec-and-runner.md).

## 1. What you are testing

A simulation run is the kit's atomic unit: one manifest in, one
`agent-learning.run.v1` artifact out. Everything else on this track —
worlds, memory, multi-agent rooms, regression baselines — is built from this
shape, so it is worth running the smallest possible one first and reading
the artifact end to end.

`examples/run_manifest.json` is that smallest manifest. Its `scenario` block
holds one dataset entry: a persona (Maya, sdk-owner), a situation, and the
outcome the run should reach. Its `agent` block is `type: scripted` — a
deterministic stand-in that answers with fixed content, which is exactly what
you want while validating plumbing rather than model behavior. The
`simulation` block selects the `local_text` engine with a single turn, and
`evaluation.enabled` is `false`, so the run records what happened without
scoring it yet.

The failure class at this stage is plumbing, not behavior: a manifest that
does not validate, an engine module that did not install, a `required_env`
key your CI forgot to declare. Catching those on a one-turn scripted run is
cheap; catching them inside a forty-turn evaluated campaign is not. The
placeholder env value works because the engine is local and deterministic —
the manifest's `required_env` list is CI metadata, not a credential check.

## 2. Run it

CLI:

```bash
AGENT_LEARNING_RUN_EXAMPLE_KEY=offline-demo-key \
  agent-learn run examples/run_manifest.json \
  --output artifacts/first-run.json
```

Note: `agent-learn` resolves a relative `--output` against your current
working directory, so the artifact lands at `artifacts/first-run.json`.

SDK (same operation):

```python
import asyncio
import os
from fi.alk import simulate

os.environ.setdefault("AGENT_LEARNING_RUN_EXAMPLE_KEY", "offline-demo-key")
result = asyncio.run(simulate.run_manifest_file("examples/run_manifest.json"))
```

The backing example, `examples/sdk_agent_integration_simulation.py`, drives
this same engine path — it builds a run manifest with
`simulate.build_agent_integration_run_manifest`, writes it with
`simulate.write_manifest_file`, and executes it with
`simulate.run_manifest_file`. Its verdict is re-proven on every release-check
by the `agent_integration_readiness` gate.

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/first-run.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
```

Inside the artifact: `status` and `exit_code` for CI, a `summary` block with
case counts, and the per-case record of the conversation the scripted agent
produced. The `kind` field is the contract every downstream command
(`baseline`, `compare`, `report`) reads.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| manifest rejected / `required_env` missing | config fault | `summary.public_boundary_passed` + the manifest error line |
| artifact missing after exit 0 | output path resolution | a relative `--output` lands in your current working directory (see §2 note) |

## 5. Prove it / keep it

Swap the scripted agent for your own (an adapter target, an HTTP endpoint,
or a framework object — see [`simulate-any-framework.md`](simulate-any-framework.md)),
turn `evaluation.enabled` on, and re-run. Then freeze the passing artifact:

```bash
agent-learn simulate baseline artifacts/first-run.json --output first-run-baseline.json
```

That baseline is the entry point to the full
[`regression-lifecycle.md`](regression-lifecycle.md) journey — every future
run gets compared against it instead of being judged by eye.
