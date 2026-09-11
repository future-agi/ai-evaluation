"""Keyed account-sync example — metadata-only, DRY-RUN by default (Phase 8).

With no keys this prints the encoded metadata row plus the destination a real
sync would use and sends NOTHING — there is no anonymous channel to fall back
to (P8-D1; the ``telemetry_boundary`` gate proves the absence structurally).
With Future AGI keys in env, ``python examples/sdk_account_sync.py --send``
performs the single owner-keyed real-sync acceptance (BUILD §7 step 5): the
metadata row reaches the user's OWN account via the existing
fi-instrumentation-otel -> ``POST {FI_BASE_URL}/tracer/v1/traces`` path, the
same ``run_id`` appears locally and in the account, a re-send is a no-op, and
``AGENT_LEARNING_TELEMETRY=off`` with the same keys produces zero network
calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import telemetry
from fi.alk.telemetry import _sync

OUTPUT_KIND = "agent-learning.account-sync-dryrun.v1"


def _demo_row() -> dict[str, Any]:
    """A deterministic metadata-only row (no capture contract — content sync
    would be refused; the metadata channel is the OTel-2026 default)."""

    payload = {
        "status": "passed",
        "summary": {"verdict": "pass", "icc": 0.88, "repeats": 8},
        "manifest": {"name": "account_sync_demo", "scenario": "refund_dispute"},
    }
    return telemetry.build_ledger_row(payload)


def run(output_path: str | Path, *, send: bool = False) -> dict[str, Any]:
    row = _demo_row()
    destination = _sync.sync_destination()
    encoded = _sync.encode_metadata_row(row)
    payload: dict[str, Any] = {
        "kind": OUTPUT_KIND,
        "status": "passed",
        "exit_code": 0,
        "sent": False,
        "kill_switch": {
            "env": telemetry.TELEMETRY_ENV,
            "active": telemetry.kill_switch_on(),
        },
        "sync_enabled": _sync.sync_enabled(),
        "destination": destination,  # header NAMES + present/missing only
        "channel": "metadata",
        "identity": {
            "local_run_id": str(row["run_id"]),
            "encoded_run_id": _sync.encoded_run_id(row),
        },
        "payload": encoded,
        "note": (
            "dry-run: nothing was sent. metadata only — run_id, kind, "
            "verdicts, scores, gate outcomes, semconv, asset hashes. "
            "content requires the capture+redaction contract."
        ),
    }
    if send:
        result = _sync.sync_run(row)
        payload["sent"] = bool(result.get("sent"))
        payload["sync_result"] = result
        payload["note"] = (
            "owner-keyed real-sync acceptance: see sync_result; a re-run "
            "must be a no-op (idempotent by content address)."
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return payload


if __name__ == "__main__":
    args = [arg for arg in sys.argv[1:] if arg != "--send"]
    target = args[0] if args else "artifacts/account-sync.json"
    run(target, send="--send" in sys.argv[1:])
