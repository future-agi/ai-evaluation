---
kind: agent-learning.docs-page.v1
track: quickstart
backing: []
artifact_kinds:
  - agent-learning.run.v1
  - agent-learning.redteam.v1
  - agent-learning.replay.v1
commands:
  - agent-learn init . --preset ci
  - agent-learn run manifests/run.json --output artifacts/run.json
  - agent-learn redteam manifests/redteam.json --output artifacts/redteam.json
  - agent-learn replay manifests --output artifacts/replay.json
postcondition: python -c "import json; p=json.load(open('artifacts/replay.json')); assert p['kind']=='agent-learning.replay.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - api_key_configured
opt_in_lane: false
---

# Golden path: the CI spine

> **Twin:** the `agent-learn init --preset ci` scaffold (`backing: []` —
> scaffold-backed by rule; proven offline by `tests/test_init_golden_paths.py`)
> · emits `agent-learning.replay.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

The `ci` preset scaffolds the smallest honest CI pipeline for an agent: a
behavior run, a red-team gate, and a replay over both — in that order. The
order is the lesson. Replay re-executes existing manifests and compares; it
means nothing until baselines exist. So the spine is run first, red-team
second, replay last, and the scaffold's README and init output list exactly
that sequence. Your first replayable artifact lands at the second command —
the path to a first artifact stays within three commands, offline.

`agent-learn init . --preset ci` writes both manifests at once:
`manifests/run.json` (the deterministic scripted smoke run — a `scripted`
agent, the `local_text` engine, one turn, an `agent_report` threshold of 0.7)
and `manifests/redteam.json` (a tool-surface prompt-injection attack pack and
campaign mapped to the OWASP LLM and agentic taxonomies, with a 0.9
evaluation threshold). These are the same manifests the single-purpose `run`
and `redteam` presets produce; the `ci` preset exists to wire them into one
directory that a pipeline can execute end to end and archive.

The final command, `agent-learn replay manifests`, re-runs every manifest in
the directory and emits one combined verdict. In a pipeline you commit the
manifests, run all three commands on every change, and archive `artifacts/` —
the replay artifact is the thing CI gates on.

## 2. Run it

CLI:

```bash
agent-learn init . --preset ci
agent-learn run manifests/run.json --output artifacts/run.json
agent-learn redteam manifests/redteam.json --output artifacts/redteam.json
agent-learn replay manifests --output artifacts/replay.json
```

SDK, same operations:

```python
import asyncio
from fi.alk import redteam, simulate

run_result = asyncio.run(simulate.run_manifest_file("manifests/run.json"))
rt_result = asyncio.run(redteam.redteam_manifest_file("manifests/redteam.json"))
replay_result = simulate.replay_manifests(["manifests"])
assert replay_result["kind"] == "agent-learning.replay.v1"
```

## 3. What you built

Postcondition (machine-checkable — the same check the scaffold README carries):

```bash
python -c "import json; p=json.load(open('artifacts/replay.json')); assert p['kind']=='agent-learning.replay.v1', p['kind']; print('ok')"
```

Three artifacts now sit in `artifacts/`: `run.json`
(`agent-learning.run.v1`, with `status: "passed"`), `redteam.json`
(`agent-learning.redteam.v1`, findings and campaign coverage), and
`replay.json` (`agent-learning.replay.v1`, the per-manifest replay results and
the combined verdict). Each carries its own `kind` header, so a pipeline can
verify every step with the same one-line check pattern.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` from any command | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| key/credential errors on a platform-connected lane | config fault | `agent-learn doctor` → `summary.api_key_configured` |
| `init would overwrite existing file(s); use --force` | scaffold state | not a doctor fault — rerun with `--force` or use a clean directory |

All four commands run with no API key set — `required_env` in the scaffolded
manifests is CI metadata, and the per-preset golden-path test executes this
sequence with the key explicitly unset.

## 5. Prove it / keep it

This spine becomes real the moment the scripted agents are replaced with your
own and the manifests are committed. Then:

1. Promote passing runs and red-team findings into `regressions/` so replay
   compares against frozen baselines instead of fresh manifests —
   [`../simulate/regression-lifecycle.md`](../simulate/regression-lifecycle.md)
   is the full baseline → compare → replay → promote → shrink journey.
2. Grow the red-team half into real campaigns via
   [`../redteam/red-team-anything.md`](../redteam/red-team-anything.md).
3. When you also want eval suites and optimization in one gate, the `all`
   preset and [`../prove/trinity-suite.md`](../prove/trinity-suite.md) collapse
   the whole spine into a single `agent-learn suite` command.
