# Vapi and Retell targets in the hosted harness

The hosted harness supports two deliberately separate modes for Vapi and Retell.

- `connect_only` calls an existing Vapi assistant ID or Retell agent ID. ALK does not modify,
  clone, or delete that target.
- `environment_backed` runs repository-owned code to create a temporary provider target for each
  isolated world. ALK calls it, then invokes the repository's destroy command during cleanup.

ALK does not infer a provider definition from arbitrary source. The repository remains the source
of truth for prompts, models, voices, tools, webhooks, knowledge bases, and provider-specific
options.

## Existing target

Submit `agent.mode: connect_only`, the target ID in non-secret config, and the matching provider
API key as a `target_provider` secret reference.

```json
{
  "agent": {
    "connector": "vapi",
    "mode": "connect_only",
    "config": {"assistant_id": "existing-assistant-id"},
    "secret_refs": {
      "VAPI_API_KEY": {
        "manager": "platform-vault",
        "key": "opaque-platform-key",
        "version": "1",
        "purpose": "target_provider"
      }
    }
  }
}
```

For Retell, use `connector: retell`, `agent_id`, and `RETELL_API_KEY`.

## Repository-created target

Submit `agent.mode: environment_backed`. The repository must contain `alk.yaml` (or the
repository-relative path selected in `agent.config.lifecycle_manifest`).

```yaml
schema_version: "1"

provider:
  type: vapi                 # vapi or retell
  scope: world               # one provider target per isolated test world
  process: tools-api         # source process whose writable build tree runs these commands
  public_capability: tools_http
  event_path: /provider/events
  tool_path: /provider/tools
  required_secrets:
    - VAPI_API_KEY
  provision:
    command: [python, scripts/provider_target.py, provision]
    timeout_seconds: 120
  destroy:
    command: [python, scripts/provider_target.py, destroy]
    timeout_seconds: 120
  output: provider-target.json
```

The named `public_capability` must be an HTTP capability in the generated environment bundle. Its
process must bind to `0.0.0.0` so Daytona can expose it. ALK provides a short-lived signed HTTPS
base URL; the repository code puts the supplied event and tool URLs into the provider definition.

Commands are executed directly without a shell, from the selected process's built source tree.
They receive:

| Environment variable | Meaning |
| --- | --- |
| `ALK_PROVIDER_CONTEXT` | Path to the complete, non-secret JSON context |
| `ALK_PROVIDER_OUTPUT` | Path where provision must write its receipt |
| `ALK_PROVIDER_RECEIPT` | Provision receipt path, present during destroy |
| `ALK_PUBLIC_BASE_URL` | Signed Daytona preview base URL |
| `ALK_EVENT_URL` | Base URL plus `event_path` |
| `ALK_TOOL_BASE_URL` | Base URL plus `tool_path` |
| `ALK_ATTEMPT_ID` / `ALK_WORLD_ID` | Ownership scope |
| declared `required_secrets` | Run-scoped provider credentials |

Provision must be idempotent for the supplied context `idempotency_key` and write:

```json
{
  "schema_version": "1",
  "provider": "vapi",
  "attempt_id": "value from ALK_PROVIDER_CONTEXT",
  "world_id": "value from ALK_PROVIDER_CONTEXT",
  "target": {
    "kind": "assistant",
    "id": "temporary-provider-target-id",
    "version": null
  },
  "resources": [
    {"kind": "assistant", "id": "temporary-provider-target-id", "owned": true},
    {"kind": "tool", "id": "temporary-tool-id", "owned": true}
  ],
  "cleanup": {
    "receipt_version": "1",
    "idempotency_key": "value from ALK_PROVIDER_CONTEXT"
  },
  "metadata": {"definition_fingerprint": "sha256:non-secret-digest"}
}
```

Retell uses `provider: retell` and `target.kind: voice_agent`.

The receipt must not contain provider keys or signed callback URLs. Destroy must read the receipt,
delete only entries with `owned: true`, tolerate resources that are already absent, and return zero
only after cleanup is complete.

## Provider wiring

Repository code is responsible for mapping the supplied URLs into its provider definition:

- Vapi: set the assistant server/webhook URL and each API-request or function tool endpoint to the
  supplied event/tool URL as appropriate.
- Retell: set the agent webhook and custom-function endpoint to the supplied event/tool URL as
  appropriate.

The platform automatically permits the Vapi or Retell API and call transport domains when the
matching run-scoped API key is present. Provider keys remain encrypted at rest, are materialized
only into the hosted attempt, and are never stored in the bundle or lifecycle receipt.

## Failure behavior

Missing manifests, missing declared secrets, invalid receipts, unavailable callback ingress, and
non-zero lifecycle commands fail before calls begin as environment or infrastructure failures.
They are not reported as agent-quality failures. Cleanup runs before process teardown on success,
failure, or cancellation.
