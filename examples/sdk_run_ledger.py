"""Run-ledger readiness example + gate-fixture generator (Phase 8, gate #72).

Deterministic and fully OFFLINE: no network, no API keys. ``run(output_path)``
exercises the whole always-on ledger lane end-to-end —

    fire ``agent-learning.run.v1`` payloads through ``public_payload`` (the
    single emission hook) -> hash-chained rows land in the ledger ->
    seeded-secret redaction (a sentinel env VALUE never reaches disk) ->
    tombstone forget (append, never rewrite) -> chain verify -> fault
    injection (a failing ledger leaves the run payload byte-identical) ->
    identity equivalence (local run_id == sync-encoder run_id)

— and regenerates the committed fixtures under
``examples/telemetry_ledger_fixture/`` that the ``telemetry_boundary`` gate
recomputes statically: ``runs.jsonl`` (valid chain from the genesis
sentinel), ``sentinel.json`` (seeded secret env + value, redacted out of the
ledger), ``faults.json`` (verdict with/without telemetry, equal), and
``identity.json`` (local ``run_id`` == sync-encoder ``run_id``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import telemetry
from fi.alk._schema import public_payload
from fi.alk.telemetry import _sync

EXAMPLE_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = EXAMPLE_DIR / "telemetry_ledger_fixture"
READINESS_KIND = "agent-learning.telemetry-ledger-readiness.v1"
RUN_KIND = "agent-learning.run.v1"

SENTINEL_ENV = "SENTINEL_TOKEN"
SENTINEL_VALUE = "sk-sentinel-do-not-leak-7f3a"

_ENV_KEYS = (
    "AGENT_LEARNING_LEDGER_PATH",
    "AGENT_LEARNING_TELEMETRY",
    "OTEL_SEMCONV_STABILITY_OPT_IN",
    SENTINEL_ENV,
    # cleared during generation so regeneration is hermetic even on a keyed
    # machine — the fixture must never record a sync attempt:
    "AGENT_LEARNING_API_KEY",
    "FUTURE_AGI_API_KEY",
    "FI_API_KEY",
    "AGENT_LEARNING_SECRET_KEY",
    "FUTURE_AGI_SECRET_KEY",
    "FI_SECRET_KEY",
)


def _payloads() -> list[dict[str, Any]]:
    """Three deterministic run payloads spanning the row shapes the gate
    audits: metadata-only, content-bearing with the capture+redaction
    contract, and a suite-phase row with gate outcomes."""

    return [
        {
            "status": "passed",
            "summary": {"verdict": "pass", "icc": 0.91, "repeats": 8},
            "manifest": {"name": "ledger_demo", "scenario": "refund_dispute"},
        },
        {
            "status": "failed",
            "summary": {"verdict": "fail", "icc": 0.42, "repeats": 8},
            "evidence_class": "captured_fixture",
            "capture": {
                "redaction": {SENTINEL_ENV: "redact_env_values"},
                "reviewed": True,
            },
            "required_env": [SENTINEL_ENV],
            # The sentinel VALUE rides a real string field pre-redaction —
            # the on-disk row must carry [redacted:SENTINEL_TOKEN] instead:
            "trace_ids": [f"9f0b-{os.environ[SENTINEL_ENV]}"],
            "asset_refs": [
                {
                    "kind": "persona",
                    "content_address": "sha256:" + "c8" * 32,
                    "account_object_id": "obj-4f2c",
                },
                {"kind": "transcript", "content_address": "sha256:" + "11" * 32},
            ],
        },
        {
            "status": "passed",
            "summary": {"verdict": "pass", "scenarios": 3},
            "suite": {"name": "trinity"},
            "gate_outcomes": {"refund_flow": True, "escalation_flow": True},
        },
    ]


def _generate_fixture(fixture_dir: Path) -> dict[str, Any]:
    for name in ("runs.jsonl", "chain.head", "sync.cursor"):
        path = fixture_dir / name
        if path.exists():
            path.unlink()
    fixture_dir.mkdir(parents=True, exist_ok=True)

    os.environ["AGENT_LEARNING_LEDGER_PATH"] = str(fixture_dir)
    for payload in _payloads():
        result = public_payload(payload, kind=RUN_KIND)  # fires the ONE hook
        assert result["kind"] == RUN_KIND
    assert telemetry.flush(10.0), "telemetry queue did not drain"

    ledger = telemetry.RunLedger(fixture_dir)
    rows = [
        row
        for row in ledger.rows()
        if row.get("schema") == telemetry.LEDGER_ROW_SCHEMA
    ]
    assert len(rows) == 3, f"expected 3 rows, found {len(rows)}"
    blob = json.dumps(rows, default=str)
    assert SENTINEL_VALUE not in blob, "sentinel VALUE leaked into the ledger"
    assert f"[redacted:{SENTINEL_ENV}]" in blob, "redaction marker missing"

    # Forget-by-tombstone: append, never rewrite — chain stays verifiable.
    content_row = rows[1]
    tombstone = ledger.append_tombstone(
        target_run_id=str(content_row["run_id"]),
        reason="redaction",
        redacted_fields=["asset_refs", "trace_ids"],
        evidence_class=str(content_row["evidence_class"]),
    )
    verify = ledger.verify()
    assert verify["chain_intact"], verify
    assert verify["tombstone_count"] == 1, verify

    (fixture_dir / "sentinel.json").write_text(
        json.dumps(
            {
                "kind": "agent-learning.ledger-sentinel.v1",
                "seeded_secret_env": SENTINEL_ENV,
                "seeded_secret_value": SENTINEL_VALUE,
                "expected_marker": f"[redacted:{SENTINEL_ENV}]",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "rows": rows,
        "tombstone": tombstone,
        "verify": verify,
        "ledger": ledger,
    }


def _fault_injection(fixture_dir: Path, tmp_root: Path) -> dict[str, Any]:
    """Never-run-blocking, executed for real: the same run payload flows
    through the hook (a) with telemetry disabled and (b) with the ledger
    write forced to fail (the ledger directory path points THROUGH a file).
    The returned payloads must be byte-identical (PRD §4.3, R§3.5)."""

    payload = {
        "status": "passed",
        "summary": {"verdict": "pass", "icc": 0.77},
    }

    os.environ["AGENT_LEARNING_TELEMETRY"] = "off"
    clean = public_payload(payload, kind=RUN_KIND)
    telemetry.flush(10.0)
    os.environ.pop("AGENT_LEARNING_TELEMETRY", None)

    blocker = tmp_root / "not-a-directory"
    blocker.write_text("a ledger dir cannot live under a file\n", encoding="utf-8")
    os.environ["AGENT_LEARNING_LEDGER_PATH"] = str(blocker / "ledger")
    faulted = public_payload(payload, kind=RUN_KIND)
    telemetry.flush(10.0)
    os.environ["AGENT_LEARNING_LEDGER_PATH"] = str(fixture_dir)

    import hashlib

    clean_bytes = json.dumps(clean, sort_keys=True, default=str)
    faulted_bytes = json.dumps(faulted, sort_keys=True, default=str)
    assert clean_bytes == faulted_bytes, "telemetry fault altered the payload"
    record = {
        "kind": "agent-learning.ledger-fault-injection.v1",
        "verdict_without_telemetry": {
            "verdict": clean["summary"]["verdict"],
            "payload_sha256": hashlib.sha256(
                clean_bytes.encode("utf-8")
            ).hexdigest(),
        },
        "verdict_with_failing_ledger": {
            "verdict": faulted["summary"]["verdict"],
            "payload_sha256": hashlib.sha256(
                faulted_bytes.encode("utf-8")
            ).hexdigest(),
        },
        "byte_identical": clean_bytes == faulted_bytes,
    }
    (fixture_dir / "faults.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def _identity_equivalence(fixture_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    """One fixture row serialized locally and through the sync metadata
    encoder yields the IDENTICAL content address (gate #72 check 6)."""

    record = {
        "kind": "agent-learning.ledger-identity.v1",
        "local_run_id": str(row["run_id"]),
        "encoded_run_id": _sync.encoded_run_id(row),
    }
    (fixture_dir / "identity.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def run(
    output_path: str | Path,
    *,
    fixture_dir: str | Path | None = None,
    tmp_root: str | Path | None = None,
) -> dict[str, Any]:
    fixture = Path(fixture_dir) if fixture_dir is not None else FIXTURE_DIR
    scratch = Path(tmp_root) if tmp_root is not None else fixture
    previous = {key: os.environ.get(key) for key in _ENV_KEYS}
    try:
        for key in _ENV_KEYS:
            os.environ.pop(key, None)
        os.environ[SENTINEL_ENV] = SENTINEL_VALUE
        generated = _generate_fixture(fixture)
        faults = _fault_injection(fixture, scratch)
        identity = _identity_equivalence(fixture, generated["rows"][0])
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        blocker = scratch / "not-a-directory"
        if blocker.is_file():
            blocker.unlink()

    verify = generated["verify"]
    payload: dict[str, Any] = {
        "kind": READINESS_KIND,
        "status": "passed",
        "exit_code": 0,
        "fixture_dir": str(fixture),
        "row_count": verify["row_count"],
        "chain_intact": verify["chain_intact"],
        "tombstone_count": verify["tombstone_count"],
        "genesis": telemetry.GENESIS,
        "redaction": {
            "seeded_secret_env": SENTINEL_ENV,
            "sentinel_bytes_on_disk": 0,
            "marker": f"[redacted:{SENTINEL_ENV}]",
        },
        "fault_injection": {"byte_identical": faults["byte_identical"]},
        "identity": {
            "local_run_id": identity["local_run_id"],
            "encoded_run_id": identity["encoded_run_id"],
            "equal": identity["local_run_id"] == identity["encoded_run_id"],
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "artifacts/run-ledger.json"
    run(target)
