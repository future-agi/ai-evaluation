---
kind: agent-learning.docs-page.v1
track: prove
objective: reliability
stage: prove
backing:
  - examples/sdk_run_ledger.py
artifact_kinds:
  - agent-learning.run.v1
  - agent-learning.ledger-row.v1
commands:
  - python examples/sdk_run_ledger.py artifacts/run-ledger.json
  - AGENT_LEARNING_LEDGER_PATH=examples/telemetry_ledger_fixture agent-learn runs list
  - AGENT_LEARNING_LEDGER_PATH=examples/telemetry_ledger_fixture agent-learn runs verify
postcondition: python -c "import json; p=json.load(open('artifacts/run-ledger.json')); assert p['kind']=='agent-learning.telemetry-ledger-readiness.v1', p['kind']; assert p['chain_intact'] is True, p; assert p['identity']['equal'] is True, p; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Run ledger: every run leaves a verifiable local row

> **Twin:** [`examples/sdk_run_ledger.py`](../../examples/sdk_run_ledger.py)
> · emits `agent-learning.run.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

A run you cannot find tomorrow is a run you cannot learn from. Every kit
workflow — simulate, evals, optimize, redteam, suite, live lanes — already
emits an `agent-learning.run.v1` artifact through one shared normalization
boundary, and the run ledger hooks that single boundary once: each run
appends one small row to a local, append-only JSONL ledger you own, at
`${AGENT_LEARNING_HOME:-~/.agent-learning}/ledger/runs.jsonl`
(`AGENT_LEARNING_LEDGER_PATH` overrides the directory). No keys, no server,
no network — the ledger is a product artifact for you, not phone-home.

Each row is content-addressed (`run_id` = SHA-256 of the canonical row, after
deterministic redaction) and hash-chained
(`chain_i = H(chain_{i-1} || run_id_i)` from the documented genesis sentinel
`"agent-learning.ledger.genesis.v1"`). One linear pass recomputes both, so a
row edited in place is detected, never trusted. Rows carry asset
*references* — content addresses of manifests, personas, scenarios, traceAI
trace ids — never asset copies, and declared env var VALUES are rewritten to
`[redacted:NAME]` before any byte is addressed or written. Forgetting is an
append too: `agent-learn runs forget` adds a tombstone row, the chain stays
verifiable, the content disappears from resolution.

The failure classes this page targets: a history that silently loses runs, a
ledger that can be edited without detection, and a redaction step that runs
after — instead of before — serialization.

## 2. Run it

Generate the committed fixture ledger (this also exercises the seeded-secret
redaction, the tombstone flow, the fault injection, and the local-vs-encoder
identity check), then inspect it with the zero-infrastructure viewer:

```bash
python examples/sdk_run_ledger.py artifacts/run-ledger.json
AGENT_LEARNING_LEDGER_PATH=examples/telemetry_ledger_fixture agent-learn runs list
AGENT_LEARNING_LEDGER_PATH=examples/telemetry_ledger_fixture agent-learn runs verify
```

The same flow from the SDK:

```python
from fi.alk import telemetry

ledger = telemetry.RunLedger()          # ~/.agent-learning/ledger by default
for row in ledger.iter_rows():
    print(row["run_id"], row.get("verdict"))
print(ledger.verify()["chain_intact"])
```

`agent-learn runs show <id> --json` prints the exact canonical bytes the
`run_id` is computed over, so `runs show <id> --json | shasum -a 256` lets
you recompute the address independently.

## 3. What you built

Run the postcondition verbatim:

```bash
python -c "import json; p=json.load(open('artifacts/run-ledger.json')); assert p['kind']=='agent-learning.telemetry-ledger-readiness.v1', p['kind']; assert p['chain_intact'] is True, p; assert p['identity']['equal'] is True, p; print('ok')"
```

The artifact records the fixture row count, `chain_intact` from a full
recompute, the tombstone count, the seeded-secret redaction result (zero
sentinel bytes on disk), the fault-injection comparison (a failing ledger
write leaves the run payload byte-identical), and the identity check (the
locally computed `run_id` equals the sync-encoder address for the same row).

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `runs verify` exits 1 with `content_address_mismatch` | a row's bytes were altered after write | `missing_engine_modules` |
| `runs verify` exits 1 with `chain_mismatch` | rows were reordered/inserted/removed | `missing_engine_modules` |
| `runs list` prints `no runs yet` after a run | `AGENT_LEARNING_TELEMETRY=off` was set, or the ledger path points elsewhere | `public_boundary_passed` |
| sentinel value visible in a row | redaction ran after serialization — file a bug; never ship | `public_boundary_passed` |

## 5. Prove it / keep it

The `telemetry_boundary` release gate recomputes this fixture ledger on every
`release-check`: chain integrity, evidence-class discipline, the
seeded-secret residue scan, the fault-injection equality, and the identity
equivalence all gate the release. Account sync for these rows is the next
step: [account-sync](./account-sync.md).
