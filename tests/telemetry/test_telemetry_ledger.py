"""Phase 8 telemetry: ledger, chain, tombstone, queue, sync, CLI (gate #72
substrate). Everything here runs offline — the only "network" is a local
``http.server`` stub collector bound to 127.0.0.1.
"""

from __future__ import annotations

import hashlib
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from fi.alk import telemetry
from fi.alk._schema import public_payload
from fi.alk.cli import main
from fi.alk.telemetry import _sync
from fi.alk.telemetry._queue import TelemetryQueue

RUN_KIND = "agent-learning.run.v1"

_SCRUB_ENV = (
    "AGENT_LEARNING_TELEMETRY",
    "AGENT_LEARNING_HOME",
    "OTEL_SEMCONV_STABILITY_OPT_IN",
    "AGENT_LEARNING_API_KEY",
    "FUTURE_AGI_API_KEY",
    "FI_API_KEY",
    "AGENT_LEARNING_SECRET_KEY",
    "FUTURE_AGI_SECRET_KEY",
    "FI_SECRET_KEY",
    "FI_BASE_URL",
)


@pytest.fixture
def ledger_env(tmp_path, monkeypatch):
    """Hermetic telemetry env: tmp ledger dir, no keys, no kill switch."""

    ledger_dir = tmp_path / "ledger"
    monkeypatch.setenv("AGENT_LEARNING_LEDGER_PATH", str(ledger_dir))
    for name in _SCRUB_ENV:
        monkeypatch.delenv(name, raising=False)
    return ledger_dir


def _payload(**overrides):
    base = {
        "status": "passed",
        "summary": {"verdict": "pass", "icc": 0.91, "repeats": 8},
        "manifest": {"name": "demo", "scenario": "refund"},
    }
    base.update(overrides)
    return base


# --- canonical row -----------------------------------------------------------


def test_build_ledger_row_is_deterministic(ledger_env):
    row_a = telemetry.build_ledger_row(_payload())
    row_b = telemetry.build_ledger_row(dict(reversed(list(_payload().items()))))
    assert row_a["run_id"] == row_b["run_id"]
    assert telemetry.canonical_row_bytes(row_a) == telemetry.canonical_row_bytes(
        row_b
    )


def test_row_field_set_matches_canon(ledger_env):
    ledger = telemetry.RunLedger()
    appended = ledger.append(telemetry.build_ledger_row(_payload()))
    assert set(appended) == set(telemetry.ROW_FIELDS)


def test_envelope_fields_excluded_from_address(ledger_env):
    ledger = telemetry.RunLedger()
    appended = ledger.append(telemetry.build_ledger_row(_payload()))
    # created_at/chain joined after addressing; the address still recomputes:
    assert appended["run_id"] == telemetry.canonical_row_address(appended)
    sha = hashlib.sha256(telemetry.canonical_row_bytes(appended)).hexdigest()
    assert sha == appended["run_id"]


def test_redaction_runs_before_addressing(ledger_env, monkeypatch):
    monkeypatch.setenv("SENTINEL_TOKEN", "sk-sentinel-row-test")
    payload = _payload(
        trace_ids=["t-sk-sentinel-row-test"], required_env=["SENTINEL_TOKEN"]
    )
    row = telemetry.build_ledger_row(
        payload, required_env=telemetry.declared_required_env(payload)
    )
    blob = telemetry.canonical_row_bytes(row).decode("utf-8")
    assert "sk-sentinel-row-test" not in blob
    assert row["trace_ids"] == ["t-[redacted:SENTINEL_TOKEN]"]
    # the address is computed over the REDACTED bytes:
    assert row["run_id"] == hashlib.sha256(blob.encode("utf-8")).hexdigest()


def test_scores_fixed_precision_rounding(ledger_env):
    row = telemetry.build_ledger_row(
        _payload(summary={"verdict": "pass", "score": 0.1 + 0.2})
    )
    assert row["scores"]["score"] == 0.3


def test_phase_inference(ledger_env):
    assert telemetry.build_ledger_row(_payload())["phase"] == "simulate"
    assert (
        telemetry.build_ledger_row(_payload(optimization={"x": 1}))["phase"]
        == "optimize"
    )
    assert (
        telemetry.build_ledger_row(_payload(redteam={"x": 1}))["phase"]
        == "redteam"
    )
    assert (
        telemetry.build_ledger_row(_payload(suite={"x": 1}))["phase"] == "suite"
    )
    assert (
        telemetry.build_ledger_row(_payload(evaluations=[{"x": 1}]))["phase"]
        == "evals"
    )
    assert (
        telemetry.build_ledger_row(_payload(live_lane={"lane": "mcp"}))["phase"]
        == "live"
    )


def test_evidence_class_defaults_to_local_gate(ledger_env):
    assert telemetry.build_ledger_row(_payload())["evidence_class"] == (
        "local_gate"
    )
    assert telemetry.build_ledger_row(
        _payload(evidence_class="not-a-class")
    )["evidence_class"] == "local_gate"
    assert telemetry.build_ledger_row(
        _payload(evidence_class="captured_fixture")
    )["evidence_class"] == "captured_fixture"


def test_content_bearing_and_redaction_contract(ledger_env):
    plain = telemetry.build_ledger_row(_payload())
    assert plain["content_bearing"] is False
    assert plain["redaction"] is None
    captured = telemetry.build_ledger_row(
        _payload(
            capture={"redaction": {"TOKEN_X": "redact_env_values"},
                     "reviewed": True}
        )
    )
    assert captured["content_bearing"] is True
    assert captured["redaction"] == {"TOKEN_X": "redact_env_values"}
    assert telemetry.content_admissible(
        {"capture": {"redaction": {"A": "x"}, "reviewed": True}}
    )
    assert not telemetry.content_admissible(
        {"capture": {"redaction": {}, "reviewed": True}}
    )
    assert not telemetry.content_admissible(
        {"capture": {"redaction": {"A": "x"}, "reviewed": False}}
    )


# --- ledger + chain ----------------------------------------------------------


def test_append_chains_from_genesis_and_verifies(ledger_env):
    ledger = telemetry.RunLedger()
    first = ledger.append(telemetry.build_ledger_row(_payload()))
    second = ledger.append(
        telemetry.build_ledger_row(_payload(status="failed",
                                            summary={"verdict": "fail"}))
    )
    expected_chain_0 = hashlib.sha256(
        (telemetry.GENESIS + first["run_id"]).encode("utf-8")
    ).hexdigest()
    assert first["chain"] == expected_chain_0
    expected_chain_1 = hashlib.sha256(
        (first["chain"] + second["run_id"]).encode("utf-8")
    ).hexdigest()
    assert second["chain"] == expected_chain_1
    verify = ledger.verify()
    assert verify["chain_intact"] is True
    assert verify["row_count"] == 2


def test_tampered_body_breaks_verify(ledger_env):
    ledger = telemetry.RunLedger()
    ledger.append(telemetry.build_ledger_row(_payload()))
    lines = ledger.rows_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["scores"]["icc"] = 0.0  # rewrite in place — the forbidden act
    ledger.rows_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    verify = ledger.verify()
    assert verify["chain_intact"] is False
    assert any(
        item["reason"] == "content_address_mismatch" for item in verify["breaks"]
    )


def test_tampered_chain_link_breaks_verify(ledger_env):
    ledger = telemetry.RunLedger()
    ledger.append(telemetry.build_ledger_row(_payload()))
    lines = ledger.rows_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["chain"] = "f" * 64
    ledger.rows_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    verify = ledger.verify()
    assert any(item["reason"] == "chain_mismatch" for item in verify["breaks"])


def test_tombstone_keeps_chain_verifiable(ledger_env):
    ledger = telemetry.RunLedger()
    appended = ledger.append(telemetry.build_ledger_row(_payload()))
    tomb = ledger.append_tombstone(
        target_run_id=appended["run_id"],
        reason="forget",
        redacted_fields=["*"],
        evidence_class=appended["evidence_class"],
    )
    assert tomb["schema"] == telemetry.TOMBSTONE_SCHEMA
    assert set(tomb) == set(telemetry.TOMBSTONE_FIELDS)
    verify = ledger.verify()
    assert verify["chain_intact"] is True
    assert verify["tombstone_count"] == 1
    assert verify["unresolved_tombstones"] == []


def test_gap_marker_is_a_chained_row(ledger_env):
    ledger = telemetry.RunLedger()
    ledger.append(telemetry.build_ledger_row(_payload()))
    gap = ledger.append_gap(3)
    assert gap["schema"] == telemetry.GAP_SCHEMA
    assert gap["dropped"] == 3
    verify = ledger.verify()
    assert verify["chain_intact"] is True
    assert verify["gap_count"] == 1
    assert verify["gap_dropped_total"] == 3


def test_unreadable_line_is_reported_not_fatal(ledger_env):
    ledger = telemetry.RunLedger()
    ledger.append(telemetry.build_ledger_row(_payload()))
    with open(ledger.rows_path, "a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    verify = ledger.verify()
    assert verify["chain_intact"] is False
    assert any(item["reason"] == "unreadable_line" for item in verify["breaks"])


def test_chain_head_sidecar_fallback(ledger_env):
    ledger = telemetry.RunLedger()
    first = ledger.append(telemetry.build_ledger_row(_payload()))
    ledger.head_path.unlink()  # sidecar gone -> linear-scan fallback
    second = ledger.append(
        telemetry.build_ledger_row(_payload(summary={"verdict": "pass", "n": 2}))
    )
    assert second["chain"] == hashlib.sha256(
        (first["chain"] + second["run_id"]).encode("utf-8")
    ).hexdigest()
    assert ledger.verify()["chain_intact"] is True


# --- emission hook + queue ---------------------------------------------------


def test_public_payload_run_kind_appends_exactly_one_row(ledger_env):
    result = public_payload(_payload(), kind=RUN_KIND)
    assert result["kind"] == RUN_KIND
    assert telemetry.flush(10.0)
    rows = [
        row
        for row in telemetry.RunLedger().rows()
        if row.get("schema") == telemetry.LEDGER_ROW_SCHEMA
    ]
    assert len(rows) == 1


def test_public_payload_other_kind_appends_nothing(ledger_env):
    public_payload(_payload(), kind="agent-learning.report.v1")
    telemetry.flush(10.0)
    assert not telemetry.RunLedger().rows_path.exists()


def test_kill_switch_suppresses_ledger_and_sync(ledger_env, monkeypatch):
    monkeypatch.setenv("AGENT_LEARNING_TELEMETRY", "off")
    assert telemetry.kill_switch_on()
    public_payload(_payload(), kind=RUN_KIND)
    telemetry.flush(10.0)
    assert not telemetry.RunLedger().rows_path.exists()
    assert _sync.sync_enabled() is False


def test_failing_ledger_never_alters_the_payload(ledger_env, monkeypatch, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir\n", encoding="utf-8")
    clean = public_payload(_payload(), kind=RUN_KIND)
    telemetry.flush(10.0)
    monkeypatch.setenv("AGENT_LEARNING_LEDGER_PATH", str(blocker / "ledger"))
    faulted = public_payload(_payload(), kind=RUN_KIND)
    telemetry.flush(10.0)
    assert json.dumps(clean, sort_keys=True, default=str) == json.dumps(
        faulted, sort_keys=True, default=str
    )


def test_queue_drops_with_gap_marker_on_overflow(ledger_env):
    ledger_holder = {}
    appended = []

    def slow_handler(row, dropped):
        ledger = telemetry.RunLedger()
        ledger_holder["ledger"] = ledger
        if dropped:
            ledger.append_gap(dropped)
        appended.append(ledger.append(row))

    q = TelemetryQueue(slow_handler, maxsize=1)
    rows = [
        telemetry.build_ledger_row(_payload(summary={"verdict": "pass", "n": n}))
        for n in range(6)
    ]
    accepted = sum(1 for row in rows if q.enqueue(row))
    assert q.flush(10.0)
    dropped = len(rows) - accepted
    if dropped:  # burst raced the worker — the loss must be RECORDED
        # drain one more row so the pending gap is written:
        assert q.enqueue(
            telemetry.build_ledger_row(_payload(summary={"verdict": "pass"}))
        )
        assert q.flush(10.0)
        verify = ledger_holder["ledger"].verify()
        assert verify["gap_dropped_total"] + len(appended) >= len(rows)
        assert verify["chain_intact"] is True


def test_queue_handler_exception_never_propagates(ledger_env):
    def exploding_handler(row, dropped):
        raise RuntimeError("boom")

    q = TelemetryQueue(exploding_handler, maxsize=4)
    assert q.enqueue(telemetry.build_ledger_row(_payload())) is True
    assert q.flush(10.0) is True  # drained despite the handler raising


# --- sync client --------------------------------------------------------------


def test_sync_enabled_requires_keys_and_no_kill_switch(ledger_env, monkeypatch):
    assert _sync.sync_enabled() is False
    monkeypatch.setenv("AGENT_LEARNING_API_KEY", "key-x")
    monkeypatch.setenv("AGENT_LEARNING_SECRET_KEY", "secret-x")
    assert _sync.sync_enabled() is True
    monkeypatch.setenv("AGENT_LEARNING_TELEMETRY", "off")
    assert _sync.sync_enabled() is False


def test_identity_equivalence_local_vs_encoder(ledger_env):
    row = telemetry.RunLedger().append(telemetry.build_ledger_row(_payload()))
    assert _sync.encoded_run_id(row) == row["run_id"]


def test_sync_run_without_keys_sends_nothing(ledger_env, monkeypatch):
    # any socket use would blow up — the no-key path must never get there:
    monkeypatch.setattr(
        socket, "create_connection", _raise_socket, raising=True
    )
    row = telemetry.RunLedger().append(telemetry.build_ledger_row(_payload()))
    result = _sync.sync_run(row)
    assert result == {"status": "no_keys", "sent": False}


def _raise_socket(*args, **kwargs):
    raise AssertionError("socket opened in a no-network path")


def test_sync_run_content_refused_without_contract(ledger_env, monkeypatch):
    monkeypatch.setenv("AGENT_LEARNING_API_KEY", "key-x")
    monkeypatch.setenv("AGENT_LEARNING_SECRET_KEY", "secret-x")
    monkeypatch.setattr(
        socket, "create_connection", _raise_socket, raising=True
    )
    row = telemetry.RunLedger().append(telemetry.build_ledger_row(_payload()))
    result = _sync.sync_run(row, content=True)
    assert result["status"] == "refused"
    assert result["reason"] == "capture_contract_missing"


def test_sync_run_defers_when_collector_unreachable(ledger_env, monkeypatch):
    monkeypatch.setenv("AGENT_LEARNING_API_KEY", "key-x")
    monkeypatch.setenv("AGENT_LEARNING_SECRET_KEY", "secret-x")
    # a port nothing listens on:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
    monkeypatch.setenv("FI_BASE_URL", f"http://127.0.0.1:{dead_port}")
    ledger = telemetry.RunLedger()
    row = ledger.append(telemetry.build_ledger_row(_payload()))
    result = _sync.sync_run(row, ledger=ledger)
    assert result["status"] == "deferred"
    assert result["sent"] is False
    assert ledger.read_cursor()["synced"] == {}  # cursor unmoved


class _StubCollector(BaseHTTPRequestHandler):
    requests: list[str] = []

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        type(self).requests.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):  # silence
        return


@pytest.fixture
def stub_collector():
    _StubCollector.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubCollector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_sync_run_posts_to_local_stub_and_is_idempotent(
    ledger_env, monkeypatch, stub_collector
):
    monkeypatch.setenv("AGENT_LEARNING_API_KEY", "key-x")
    monkeypatch.setenv("AGENT_LEARNING_SECRET_KEY", "secret-x")
    monkeypatch.setenv("FI_BASE_URL", stub_collector)
    ledger = telemetry.RunLedger()
    row = ledger.append(telemetry.build_ledger_row(_payload()))
    result = _sync.sync_run(row, ledger=ledger)
    assert result["status"] == "synced", result
    assert result["channel"] == "metadata"
    assert result["endpoint"] == f"{stub_collector}/tracer/v1/traces"
    assert any(
        path.endswith("/tracer/v1/traces") for path in _StubCollector.requests
    ), _StubCollector.requests
    cursor = ledger.read_cursor()
    assert cursor["synced"][row["run_id"]] == "metadata"
    assert cursor["cursor"] == row["run_id"]
    # idempotent by content address — the re-send is a no-op:
    again = _sync.sync_run(row, ledger=ledger)
    assert again["status"] == "noop"
    assert again["sent"] is False


# --- CLI: runs list | show | verify | sync | forget ---------------------------


def _seed_two_runs():
    public_payload(_payload(), kind=RUN_KIND)
    public_payload(
        _payload(status="failed", summary={"verdict": "fail", "icc": 0.42}),
        kind=RUN_KIND,
    )
    assert telemetry.flush(10.0)
    return [
        row
        for row in telemetry.RunLedger().rows()
        if row.get("schema") == telemetry.LEDGER_ROW_SCHEMA
    ]


def test_cli_runs_list_table_and_footer(ledger_env, capsys):
    rows = _seed_two_runs()
    assert main(["runs", "list"]) == 0
    out = capsys.readouterr().out
    assert "RUN_ID" in out and "SYNCED" in out
    for row in rows:
        assert row["run_id"][:8] in out
    assert "chain OK" in out
    assert str(telemetry.RunLedger().rows_path) in out


def test_cli_runs_list_json_and_filters(ledger_env, capsys):
    _seed_two_runs()
    assert main(["runs", "list", "--verdict", "fail", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert len(listed) == 1
    assert listed[0]["verdict"] == "fail"


def test_cli_runs_show_json_reproduces_run_id(ledger_env, capsys):
    rows = _seed_two_runs()
    target = rows[0]
    assert main(["runs", "show", target["run_id"][:8], "--json"]) == 0
    out = capsys.readouterr().out
    # exact canonical bytes, no trailing newline -> sha256 == run_id:
    assert hashlib.sha256(out.encode("utf-8")).hexdigest() == target["run_id"]


def test_cli_runs_show_refuses_ambiguous_prefix(ledger_env, capsys):
    _seed_two_runs()
    assert main(["runs", "show", ""]) == 1  # empty prefix matches both
    err = capsys.readouterr().err
    assert "ambiguous" in err


def test_cli_runs_verify_exit_codes(ledger_env, capsys):
    _seed_two_runs()
    assert main(["runs", "verify"]) == 0
    out = capsys.readouterr().out
    assert "CHAIN OK" in out
    assert telemetry.GENESIS in out
    ledger = telemetry.RunLedger()
    lines = ledger.rows_path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["scores"]["icc"] = -1.0
    lines[0] = json.dumps(tampered)
    ledger.rows_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert main(["runs", "verify"]) == 1
    assert "CHAIN BROKEN" in capsys.readouterr().out


def test_cli_runs_sync_dry_run_opens_no_socket(ledger_env, capsys, monkeypatch):
    rows = _seed_two_runs()
    monkeypatch.setattr(
        socket, "create_connection", _raise_socket, raising=True
    )
    monkeypatch.setattr(socket.socket, "connect", _raise_socket, raising=True)
    # no keys -> the §4.2 "no anonymous channel" state:
    assert main(["runs", "sync", rows[0]["run_id"][:8], "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "no anonymous channel" in out
    # with keys -> the literal-JSON transparency surface, still no socket:
    monkeypatch.setenv("AGENT_LEARNING_API_KEY", "key-x")
    monkeypatch.setenv("AGENT_LEARNING_SECRET_KEY", "secret-x")
    assert main(["runs", "sync", rows[0]["run_id"][:8], "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN — nothing is sent." in out
    assert "/tracer/v1/traces" in out
    assert "X-Api-Key=[present]" in out
    assert "key-x" not in out  # names always, values never
    assert rows[0]["run_id"] in out  # the literal canonical row
    assert "0 residual sentinel bytes" in out
    assert "nothing was sent" in out


def test_cli_runs_sync_kill_switch_refusal(ledger_env, capsys, monkeypatch):
    rows = _seed_two_runs()
    monkeypatch.setenv("AGENT_LEARNING_TELEMETRY", "off")
    monkeypatch.setattr(
        socket, "create_connection", _raise_socket, raising=True
    )
    assert main(["runs", "sync", rows[0]["run_id"][:8]]) == 0
    out = capsys.readouterr().out
    assert "sync disabled" in out
    assert "AGENT_LEARNING_TELEMETRY=off" in out


def test_cli_runs_forget_appends_tombstone_and_verify_stays_green(
    ledger_env, capsys
):
    rows = _seed_two_runs()
    assert main(
        ["runs", "forget", rows[1]["run_id"][:8], "--run", "--yes"]
    ) == 0
    out = capsys.readouterr().out
    assert "tombstone appended" in out
    assert main(["runs", "verify"]) == 0
    out = capsys.readouterr().out
    assert "1 redaction rows" in out
    assert main(["runs", "list"]) == 0
    out = capsys.readouterr().out
    assert "[redacted]" in out  # tombstoned row renders [redacted]


def test_cli_ledger_hidden_alias(ledger_env, capsys):
    _seed_two_runs()
    assert main(["ledger", "list"]) == 0
    assert "RUN_ID" in capsys.readouterr().out


def test_cli_help_does_not_document_ledger_alias(capsys):
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "runs" in out
    assert "ledger" not in out


# --- example generators (offline, deterministic) ------------------------------


def test_sdk_run_ledger_example_regenerates_fixture(tmp_path, monkeypatch):
    import importlib.util

    for name in _SCRUB_ENV + ("AGENT_LEARNING_LEDGER_PATH",):
        monkeypatch.delenv(name, raising=False)
    spec = importlib.util.spec_from_file_location(
        "sdk_run_ledger",
        Path(__file__).resolve().parents[2] / "examples" / "sdk_run_ledger.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture_dir = tmp_path / "fixture"
    payload = module.run(
        tmp_path / "out.json", fixture_dir=fixture_dir, tmp_root=tmp_path
    )
    assert payload["chain_intact"] is True
    assert payload["row_count"] == 4
    assert payload["tombstone_count"] == 1
    assert payload["fault_injection"]["byte_identical"] is True
    assert payload["identity"]["equal"] is True
    blob = (fixture_dir / "runs.jsonl").read_text(encoding="utf-8")
    assert module.SENTINEL_VALUE not in blob
    assert "[redacted:SENTINEL_TOKEN]" in blob


def test_sdk_account_sync_example_dry_run(tmp_path, monkeypatch):
    import importlib.util

    for name in _SCRUB_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(
        "AGENT_LEARNING_LEDGER_PATH", str(tmp_path / "ledger")
    )
    monkeypatch.setattr(
        socket, "create_connection", _raise_socket, raising=True
    )
    spec = importlib.util.spec_from_file_location(
        "sdk_account_sync",
        Path(__file__).resolve().parents[2]
        / "examples"
        / "sdk_account_sync.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = module.run(tmp_path / "out.json")
    assert payload["sent"] is False
    assert payload["sync_enabled"] is False
    assert payload["destination"]["endpoint"].endswith("/tracer/v1/traces")
    assert payload["destination"]["headers"] == {
        "X-Api-Key": "missing",
        "X-Secret-Key": "missing",
    }
    assert payload["identity"]["local_run_id"] == (
        payload["identity"]["encoded_run_id"]
    )
