---
kind: agent-learning.docs-page.v1
track: frameworks
objective: behavior
stage: simulate
backing:
  - examples/sdk_framework_adapter_mcp_tool_session.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - python examples/sdk_framework_adapter_mcp_tool_session.py artifacts/framework-mcp.json
  - agent-learn run artifacts/framework-mcp.manifest.json --output artifacts/framework-mcp-cli.json
postcondition: python -c "import json; p=json.load(open('artifacts/framework-mcp.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# MCP: offline framework-adapter simulation

> **Twin:** [`examples/sdk_framework_adapter_mcp_tool_session.py`](../../examples/sdk_framework_adapter_mcp_tool_session.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

MCP coverage in the kit is probe-promoted at the protocol layer: what gets
simulated is an MCP client/server tool session export, not a string answer. The
twin, [`examples/sdk_framework_adapter_mcp_tool_session.py`](../../examples/sdk_framework_adapter_mcp_tool_session.py),
builds a local `LocalMCPToolSessionAgent` whose verified entrypoint returns a typed
`MCPToolSessionExport`: the server name, a session id, the tool list with JSON
`inputSchema` declarations (for example `refund_policy_lookup`), the resource list,
and the session's protocol events. A weak `run(text)` path with no tool-protocol
evidence exists on the same object, and promotion records it as weak.

The failure class this catches is protocol-evidence loss: an agent wired to an MCP
server can produce plausible answers while the harness never confirms which tools
were advertised, which were called, and in what session. The session export makes
tool inventory and call history first-class, checkable fields of the artifact.

There is no separate manifest file for this page: the twin builds its run manifest
in code, writes it next to the output
(`artifacts/framework-mcp.manifest.json`), and executes it through the same
`simulate.run_manifest_file` path the CLI uses. Everything runs on the local
engine: offline, deterministic, no MCP server process and no provider keys.

## 2. Run it

CLI — the twin is executable and writes both the run artifact and the manifest it
ran, which you can then replay through `agent-learn`:

```bash
python examples/sdk_framework_adapter_mcp_tool_session.py artifacts/framework-mcp.json
agent-learn run artifacts/framework-mcp.manifest.json \
  --output artifacts/framework-mcp-cli.json
```

SDK, same operation:

```python
from sdk_framework_adapter_mcp_tool_session import run  # examples/ on sys.path

result = run("artifacts/framework-mcp.json")
assert result["kind"] == "agent-learning.run.v1"
```

## 3. What you built

Postcondition (machine-checkable — the same check the docs gate pattern uses):

```bash
python -c "import json; p=json.load(open('artifacts/framework-mcp.json')); assert p['kind']=='agent-learning.run.v1', p['kind']; print('ok')"
```

The artifact carries `status`, the simulated transcript, the evaluation report,
and the MCP session export — tools with their input schemas, resources, session
id, and protocol events — plus the exact manifest that produced it. It is a
replayable record, not a log line: the same file feeds `baseline`, `compare`, and
`replay`.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| manifest replay rejected | config fault | `agent-learn doctor` → `summary.public_boundary_passed` plus the manifest error line |
| export missing tools/resources/session events (weak text path) | behavior regression | re-run the twin promotion and compare the session export against the text fallback |

## 5. Prove it / keep it

The twin is admitted by the `protocol_adapter_readiness` release gate, so every
`agent-learn release-check` re-executes this exact tool-session path — the page
stays true or the release fails. To keep your own MCP integration honest, promote
the run artifact into a regression baseline with the `baseline` /
`promote-to-regression` / `compare` command family, and treat the session export as
the contract: when your server adds or renames tools, the diff shows up in the
artifact before it shows up in production. The reader's job here is maintenance of
a living proof, not a one-off demo.
