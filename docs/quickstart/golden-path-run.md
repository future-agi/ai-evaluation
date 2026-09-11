---
kind: agent-learning.docs-page.v1
track: quickstart
backing: []
artifact_kinds:
  - agent-learning.run.v1
commands:
  - agent-learn init . --preset run
  - agent-learn run manifests/run.json --output artifacts/run.json
postcondition: python -c "import json; p=json.load(open('artifacts/run.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; assert p['status']=='passed', p['status']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - api_key_configured
opt_in_lane: false
---

# Golden path: first simulation run

> **Twin:** the `agent-learn init --preset run` scaffold (`backing: []` — scaffold-backed
> by rule; proven offline by `tests/test_init_golden_paths.py`) · emits
> `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

This page takes you from an empty directory to a replayable run artifact in two
commands, fully offline — no Docker, no cloud login, no API key. The point of
the first run is not to exercise a real agent; it is to prove the harness:
manifest loading, the simulation engine, evaluation, and artifact emission all
work on your machine before any of your own code enters the loop.

`agent-learn init . --preset run` scaffolds four things: `manifests/run.json`,
a `README.md` that lists this exact command path with per-command checks and a
doctor table, and empty `artifacts/` and `regressions/` directories. The run
manifest describes a single-turn scenario: a persona (`Kai`, a `ci-operator`)
who needs a local CLI smoke test to pass, a `scripted` agent whose reply is the
expected outcome verbatim, the `local_text` simulation engine, and an
`agent_report` evaluation with a 0.7 threshold. Because the agent is scripted,
the run is deterministic — if anything fails, the fault is in your environment,
not in model behavior.

The scaffolded manifest carries `required_env: ["AGENT_LEARNING_API_KEY"]`.
This is CI metadata, not an offline precondition: the per-preset golden-path
test runs this exact sequence with that variable explicitly unset and passes.

## 2. Run it

CLI:

```bash
agent-learn init . --preset run
agent-learn run manifests/run.json --output artifacts/run.json
```

SDK, same operation:

```python
import asyncio
from fi.alk import simulate

result = asyncio.run(simulate.run_manifest_file("manifests/run.json"))
assert result["kind"] == "agent-learning.run.v1"
assert result["status"] == "passed"
```

## 3. What you built

Postcondition (machine-checkable — the same check the scaffold README carries):

```bash
python -c "import json; p=json.load(open('artifacts/run.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; assert p['status']=='passed', p['status']; print('ok')"
```

`artifacts/run.json` is a complete run record: `kind` and `schema_version`
headers, `status`, a `summary` block, the simulated transcript, and the
`agent_report` evaluation against the 0.7 threshold. It is the unit every later
stage consumes — replay re-executes it, regression promotion freezes it, and
`agent-learn report` renders it.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` from any command | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| key/credential errors on a platform-connected lane | config fault | `agent-learn doctor` → `summary.api_key_configured` |
| `init would overwrite existing file(s); use --force` | scaffold state | not a doctor fault — rerun with `--force` or use a clean directory |

The offline path needs no key; `api_key_configured` matters once you point the
same manifests at platform-connected features.

## 5. Prove it / keep it

A passing scripted run is a working harness, not a tested agent. Next steps, in
spine order:

1. Replace the `agent` block in `manifests/run.json` with your real agent (see
   the framework adapters under `../frameworks/` or
   [`../simulate/simulate-any-framework.md`](../simulate/simulate-any-framework.md)).
2. Promote the run into `regressions/` and replay it on every change —
   [`../simulate/regression-lifecycle.md`](../simulate/regression-lifecycle.md)
   is the full baseline → compare → replay → promote journey.
3. When you want run + red-team + replay wired for CI in one scaffold, use
   [`golden-path-ci.md`](golden-path-ci.md).
