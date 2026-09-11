---
kind: agent-learning.docs-page.v1
track: frameworks
objective: behavior
stage: simulate
backing:
  - examples/sdk_framework_adapter_browser_cua_trace.py
  - examples/sdk_browser_cua_probe_optimization.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - python examples/sdk_framework_adapter_browser_cua_trace.py artifacts/framework-browser-use.json
  - agent-learn run artifacts/framework-browser-use.manifest.json --output artifacts/framework-browser-use-cli.json
postcondition: python -c "import json; p=json.load(open('artifacts/framework-browser-use.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Browser Use: offline framework-adapter simulation

> **Twin:** [`examples/sdk_framework_adapter_browser_cua_trace.py`](../../examples/sdk_framework_adapter_browser_cua_trace.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

Browser Use coverage in the kit is probe-promoted, and the unit of evidence is the
browser/CUA trace, not a screenshot folder. The twin,
[`examples/sdk_framework_adapter_browser_cua_trace.py`](../../examples/sdk_framework_adapter_browser_cua_trace.py),
builds a local `LocalBrowserUseAgent` whose export is typed: `BrowserSnapshot`
entries (url, title, DOM, screenshot URI), `BrowserAction` entries (action,
selector, coordinates, region) with explicit `success`/`matched`/`blocked` flags, a
`prompt_injection_touched` marker, and a named mutation
(`selector_drift_safe_fallback` of type `selector_drift`). The adapter must carry
all of that into the run artifact so a reviewer can answer "what did the agent
click, and did the selector still match after drift" from the artifact alone.

The failure class this catches is invisible action drift: a browser agent whose
selector silently falls back, or whose action touched injected page content, looks
identical to a clean run if the harness only checks the final answer. The trace
flags make each of those conditions a checkable field.

The second twin, [`examples/sdk_browser_cua_probe_optimization.py`](../../examples/sdk_browser_cua_probe_optimization.py),
closes the loop: it builds a browser-CUA probe optimization, derives a run manifest
from the winning candidate, and executes it. The red-team variant of this lane is
[`examples/browser_cua_optimization.json`](../../examples/browser_cua_optimization.json).
Everything here runs offline, deterministic, no real browser and no provider keys.

## 2. Run it

CLI — the twin is executable and writes both the run artifact and the manifest it
ran (`artifacts/framework-browser-use.manifest.json`), which you can then replay
through `agent-learn`:

```bash
python examples/sdk_framework_adapter_browser_cua_trace.py artifacts/framework-browser-use.json
agent-learn run artifacts/framework-browser-use.manifest.json \
  --output artifacts/framework-browser-use-cli.json
```

SDK, same operation:

```python
from sdk_framework_adapter_browser_cua_trace import run  # examples/ on sys.path

result = run("artifacts/framework-browser-use.json")
assert result["kind"] == "agent-learning.run.v1"
```

## 3. What you built

Postcondition (machine-checkable — the same check the docs gate pattern uses):

```bash
python -c "import json; p=json.load(open('artifacts/framework-browser-use.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
```

The artifact carries `status`, the simulated transcript, the evaluation report,
and the full browser trace export — snapshots, actions with selector and region
data, the drift mutation record, and the prompt-injection flag — plus the exact
manifest that produced it. It is a replayable record, not a log line: the same
file feeds `baseline`, `compare`, and `replay`.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| manifest replay rejected | config fault | `agent-learn doctor` → `summary.public_boundary_passed` plus the manifest error line |
| actions report `matched: false` or `prompt_injection_touched: true` | behavior regression | re-run the twin and diff the action flags against the previous artifact |

## 5. Prove it / keep it

The first twin is admitted by the `framework_adapter_probe_readiness` release gate
and the probe-optimization twin by `browser_cua_probe_readiness`, so every
`agent-learn release-check` re-executes both paths — the page stays true or the
release fails. To keep your own browser agent honest, promote the run artifact into
a regression baseline with the `baseline` / `promote-to-regression` / `compare`
command family, and graduate to the browser-CUA red-team optimization manifest
linked above when you want adversarial pressure on the same trace contract. The
reader's job here is maintenance of a living proof, not a one-off demo.
