---
kind: agent-learning.docs-page.v1
track: eval
objective: reliability
stage: evaluate
backing:
  - examples/sdk_evaluation_hook_optimization.py
  - examples/sdk_evaluation_hook_probe_optimization.py
artifact_kinds:
  - agent-learning.optimization.v1
  - agent-learning.run.v1
commands:
  - AGENT_LEARNING_SDK_EVALUATION_HOOK_KEY=local-demo-key python examples/sdk_evaluation_hook_optimization.py artifacts/eval-hook-optimization.json
postcondition: python -c "import json; p=json.load(open('artifacts/eval-hook-optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - api_key_configured
  - missing_engine_modules
opt_in_lane: false
---

# Eval hooks

> **Twin:** [`examples/sdk_evaluation_hook_optimization.py`](../../examples/sdk_evaluation_hook_optimization.py)
> · emits `agent-learning.optimization.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

Whether your own scoring service can sit in the loop. An evaluation hook is
an HTTP endpoint that receives a case (messages, evidence) via authenticated
POST and returns a score — your domain rubric, your compliance checker, your
existing QA service, exposed at one URL. The kit then treats that endpoint
as a first-class metric: in one-off evaluations, and as the objective an
optimizer climbs.

The two backing examples cover both roles. The first one optimizes agent
candidates *scored by* a hook: it builds an evaluation-hook optimization
manifest (`optimize.build_evaluation_hook_optimization_manifest`), declares
the endpoint and the env var holding its bearer token, and runs the
optimization with the hook as judge. The second probes hook-scored
candidates directly and then promotes the probe result into a runnable
simulation manifest
(`build_evaluation_hook_run_manifest_from_probe_optimization`) — the
probe-to-run promotion path, which is why this page lists two artifact
kinds.

The integration risks a hook introduces are exactly what the examples
exercise offline: both spin up a local `ThreadingHTTPServer` standing in
for your service, and the server enforces bearer auth, rejects malformed
JSON, and scores assistant messages against required terms. Nothing leaves
localhost; the "key" is whatever string you export, checked only by the
local stand-in.

## 2. Run it

CLI — the twin self-hosts its hook when no endpoint is configured:

```bash
AGENT_LEARNING_SDK_EVALUATION_HOOK_KEY=local-demo-key \
  python examples/sdk_evaluation_hook_optimization.py artifacts/eval-hook-optimization.json
```

SDK — the same operation against a hook you run:

```python
from fi.alk import optimize

result = optimize.optimize_evaluation_hooks(
    name="my-hook-optimization",
    endpoint="http://127.0.0.1:8768/eval/task",
    required_env=["AGENT_LEARNING_SDK_EVALUATION_HOOK_KEY"],
    api_key_env="AGENT_LEARNING_SDK_EVALUATION_HOOK_KEY",
)
```

For one-off scoring rather than optimization, `agent-learn eval-task`
accepts `--eval-hook <endpoint>` (with `--eval-hook-api-key-env` and
`--eval-hook-metric-name`) and merges the hook's score into the task
evidence evaluation — that form needs your endpoint reachable when the
command runs, so it stays out of this page's offline command list.

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/eval-hook-optimization.json')); assert p['kind']=='agent-learning.optimization.v1', p['kind']; print('ok')"
```

The artifact is a standard optimization result — candidate history, scores,
the selected candidate — except every score came from your endpoint. The
probe-promotion example additionally writes an `agent-learning.run.v1`
artifact from the manifest it derived, demonstrating that hook-scored
selection survives into a replayable simulation.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `Set AGENT_LEARNING_SDK_EVALUATION_HOOK_KEY ...` | keys | `agent-learn doctor` → `summary.api_key_configured` |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| 401 from the hook | auth fault | bearer token sent does not match what the endpoint expects |
| hook returns non-JSON or missing score | contract fault | your service must return JSON with a score the metric can read |

## 5. Prove it / keep it

Both backing examples are executed by their readiness gates on every
`agent-learn release-check`, so the hook contract — auth, request shape,
score extraction — is continuously re-verified in this repo. Keep your own
hook honest the same way: run a stand-in server with your real handler
logic in CI, point the optimization at it, and assert the artifact kind and
selected-candidate score. When the hook is the judge for decisions that
ship, its contract test belongs in the same suite as the agents it judges.
