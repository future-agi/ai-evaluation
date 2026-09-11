---
kind: agent-learning.docs-page.v1
track: prove
objective: reliability
stage: prove
backing:
  - examples/sdk_account_sync.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - python examples/sdk_account_sync.py artifacts/account-sync.json
  - AGENT_LEARNING_LEDGER_PATH=examples/telemetry_ledger_fixture agent-learn runs sync --queued --dry-run
postcondition: python -c "import json; p=json.load(open('artifacts/account-sync.json')); assert p['kind']=='agent-learning.account-sync-dryrun.v1', p['kind']; assert p['sent'] is False, p; assert p['destination']['endpoint'].endswith('/tracer/v1/traces'), p['destination']; assert p['identity']['local_run_id']==p['identity']['encoded_run_id'], p['identity']; print('ok')"
claims: []
doctor_checks:
  - api_key_configured
  - public_boundary_passed
opt_in_lane: false
---

# Account sync: keyed, explicit, metadata by default

> **Twin:** [`examples/sdk_account_sync.py`](../../examples/sdk_account_sync.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

The kit has exactly two telemetry channels: the always-on local run ledger
([run-ledger](./run-ledger.md)) and keyed sync to *your own* Future AGI
account. There is no third, anonymous channel — not "off by default",
structurally absent, and the `telemetry_boundary` release gate scans both
`src/fi/alk/` and vendored `src/fi/` for any analytics endpoint or
network emission reachable without keys.

Consent is the key: when `AGENT_LEARNING_API_KEY` / `FUTURE_AGI_API_KEY` /
`FI_API_KEY` resolve, ledger rows sync to your account over the existing
fi-instrumentation-otel path (`POST {FI_BASE_URL}/tracer/v1/traces`) — the
same collector your traces already use. The default payload is **metadata
only**: `run_id`, kind, phase, verdicts, scores, gate outcomes, semconv
version, asset hashes. Content — transcripts, prompts, tool I/O — requires
the same capture+redaction contract the `live_lane_boundary` gate demands on
captured fixtures; without it, content stays on your machine even with valid
keys. The single kill switch `AGENT_LEARNING_TELEMETRY=off` overrides keys
and binds every component, vendored `fi/*` included.

The failure classes this page targets: a sync surface you cannot inspect
before bytes leave, content leaving without a redaction contract, and a
telemetry failure changing a run's verdict.

## 2. Run it

The dry-run prints the literal JSON a real sync would transmit — destination,
header names as present/missing (names always, values never), channel, and
the canonical row — and never opens a socket:

```bash
python examples/sdk_account_sync.py artifacts/account-sync.json
AGENT_LEARNING_LEDGER_PATH=examples/telemetry_ledger_fixture \
  agent-learn runs sync --queued --dry-run
```

The same flow from the SDK:

```python
from fi.alk import telemetry
from fi.alk.telemetry import _sync

row = telemetry.RunLedger().rows()[0]
print(_sync.sync_destination())          # endpoint + header NAMES only
print(_sync.encode_metadata_row(row))    # the literal metadata payload
print(_sync.sync_enabled())              # False without keys / with the kill switch
```

With your own keys in env, `agent-learn runs sync <id>` sends the metadata
row; re-running is a no-op because the content address is the identity —
the same `run_id` appears locally and in your account.

## 3. What you built

Run the postcondition verbatim:

```bash
python -c "import json; p=json.load(open('artifacts/account-sync.json')); assert p['kind']=='agent-learning.account-sync-dryrun.v1', p['kind']; assert p['sent'] is False, p; assert p['destination']['endpoint'].endswith('/tracer/v1/traces'), p['destination']; assert p['identity']['local_run_id']==p['identity']['encoded_run_id'], p['identity']; print('ok')"
```

The artifact records the destination a real sync would use, the kill-switch
state, whether keys resolved, the metadata channel, the encoded row, and the
identity check — the locally computed `run_id` equals the sync-encoder
address byte-for-byte.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `sync_enabled` is False with keys exported | `AGENT_LEARNING_TELEMETRY=off` is set — the kill switch overrides keys | `api_key_configured` |
| `runs sync --content` refuses with `capture_contract_missing` | the run has no reviewed capture+redaction map — metadata still syncs | `api_key_configured` |
| `runs sync` reports `deferred` | collector unreachable — the row stays local and syncs later, idempotently | `public_boundary_passed` |
| a run's exit code changed after enabling sync | telemetry must never block or alter a run — file a bug; never ship | `public_boundary_passed` |

## 5. Prove it / keep it

The `telemetry_boundary` gate keeps this honest on every `release-check`:
zero network emission in the no-key path across both source trees, an
analytics-endpoint denylist over all kit source, the content-contract
discipline on every fixture row, and the local-vs-encoder identity check.
The one real-key validation is owner-run: `python examples/sdk_account_sync.py
--send` against a real account, then re-run to confirm the no-op.
