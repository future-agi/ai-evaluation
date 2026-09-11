"""Phase 14 — dashboard telemetry wiring (W&B / promptfoo model).

Credential-free proofs of the four acceptance criteria that DON'T need a live
collector (AC1/AC2/AC3 + URL construction + framework-agnostic side-channel). The
live data-plane E2E (AC4/AC5) is owner-gated (a collector that accepts the FI key)
and lives in the build guide's Step 7, not here.

The headline regression test is ``test_sync_run_reports_export_failed_not_synced``:
it pins the fix for the false-``synced`` bug (RESEARCH §1.2) — the OTLP exporter
swallows a 401, so the old path reported success while nothing landed.
"""

from __future__ import annotations

from opentelemetry.sdk.trace.export import SpanExportResult

from fi.alk import tasks
from fi.alk.config import AgentLearningConfig
from fi.alk.telemetry import _emit, _run, _sync, _url


# --- AC2/AC3: the export-result-aware recorder is the source of truth ----------
class _FakeInner:
    def __init__(self, result):
        self._result = result

    def export(self, spans):
        return self._result

    def shutdown(self):
        return None

    def force_flush(self, timeout_millis: int = 30_000):
        return True


def test_recording_exporter_tracks_success_and_failure() -> None:
    ok = _emit._recording_exporter(_FakeInner(SpanExportResult.SUCCESS))
    ok.export(["s"])
    assert ok.ok is True

    bad = _emit._recording_exporter(_FakeInner(SpanExportResult.FAILURE))
    bad.export(["s"])
    assert bad.ok is False
    assert bad.last_reason == "export_rejected"

    none = _emit._recording_exporter(_FakeInner(SpanExportResult.SUCCESS))
    assert none.ok is False  # nothing exported yet -> NOT ok (no false success)


# --- keyed_emit status mapping (no network: fake provider) ---------------------
class _FakeSpanCtx:
    trace_id = int("2f2c9da95d04412db12911adc6c65530", 16)


class _FakeSpan:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_span_context(self):
        return _FakeSpanCtx()

    def set_attribute(self, *a):
        return None


class _FakeTracer:
    def start_as_current_span(self, name):
        return _FakeSpan()


class _FakeProvider:
    def get_tracer(self, name):
        return _FakeTracer()

    def force_flush(self):
        return True

    def shutdown(self):
        return None


def _patch_provider(monkeypatch, ok: bool):
    class _Rec:
        @property
        def ok(self):
            return ok

        @property
        def last_reason(self):
            return "ok" if ok else "export_rejected"

    monkeypatch.setattr(_emit, "_build_provider", lambda pn, hd: (_FakeProvider(), _Rec()))


def test_keyed_emit_synced_on_observed_success(monkeypatch) -> None:
    _patch_provider(monkeypatch, ok=True)
    out = _emit.keyed_emit(
        span_name="t", root_attrs={"a": 1}, project_name="p",
        headers={"X-Api-Key": "k", "X-Secret-Key": "s"},
    )
    assert out["status"] == "synced"
    assert out["trace_id"] == "2f2c9da95d04412db12911adc6c65530"


def test_keyed_emit_export_failed_on_observed_failure(monkeypatch) -> None:
    _patch_provider(monkeypatch, ok=False)
    out = _emit.keyed_emit(
        span_name="t", root_attrs={"a": 1}, project_name="p",
        headers={"X-Api-Key": "k", "X-Secret-Key": "s"},
    )
    assert out["status"] == "export_failed"
    assert out["trace_id"] is None  # nothing landed -> no viewable trace


# --- URL construction (verified route shapes; no network) ---------------------
def test_url_deep_link_from_explicit_project_id() -> None:
    cfg = AgentLearningConfig(
        api_key="k", secret_key="s",
        api_url="https://api.futureagi.com", project_id="proj-123",
    )
    u = _url.build_dashboard_url("agent-learning", "2f2c9da95d04412db12911adc6c65530", config=cfg)
    assert u["kind"] == "deep_link"
    assert u["url"] == (
        "https://app.futureagi.com/dashboard/observe/proj-123/"
        "trace/2f2c9da9-5d04-412d-b129-11adc6c65530"
    )


def test_url_list_fallback_when_no_project_id() -> None:
    cfg = AgentLearningConfig(api_url="https://api.futureagi.com")  # no keys -> no resolve
    u = _url.build_dashboard_url("agent-learning", None, config=cfg)
    assert u["kind"] == "list_fallback"
    assert u["url"] == "https://app.futureagi.com/dashboard/observe"


def test_url_self_hosted_host_not_invented() -> None:
    cfg = AgentLearningConfig(api_url="https://collector.internal.example", project_id="p9")
    u = _url.build_dashboard_url("x", "a" * 32, config=cfg)
    # no api.* -> keep the base host as-is (do not invent an app.* host)
    assert u["url"].startswith("https://collector.internal.example/dashboard/observe/p9")


# --- AC2 REGRESSION: sync_run must NOT report synced when the export failed ----
def test_sync_run_reports_export_failed_not_synced(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AGENT_LEARNING_LEDGER_PATH", str(tmp_path))
    monkeypatch.setenv("FI_API_KEY", "k")
    monkeypatch.setenv("FI_SECRET_KEY", "s")
    monkeypatch.delenv("AGENT_LEARNING_TELEMETRY", raising=False)
    # collector reachable (TCP ok) but the EXPORT is rejected (the 401 case):
    monkeypatch.setattr(_sync, "_collector_reachable", lambda base, timeout=3.0: (True, "ok"))
    monkeypatch.setattr(
        _emit, "keyed_emit",
        lambda **kw: {"status": "export_failed", "trace_id": None, "reason": "export_rejected"},
    )
    from fi.alk.telemetry import build_ledger_row
    from fi.alk.telemetry._ledger import RunLedger

    row = build_ledger_row({"status": "passed", "summary": {"verdict": "pass"},
                            "manifest": {"name": "regress"}})
    out = _sync.sync_run(row)
    assert out["status"] == "export_failed"
    assert out["sent"] is False
    # cursor UNMOVED — the row is not marked synced (degrade-to-local, R§3.5)
    cursor = RunLedger().read_cursor()
    assert row["run_id"] not in cursor["synced"]


def test_sync_run_synced_only_on_observed_success(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AGENT_LEARNING_LEDGER_PATH", str(tmp_path))
    monkeypatch.setenv("FI_API_KEY", "k")
    monkeypatch.setenv("FI_SECRET_KEY", "s")
    monkeypatch.delenv("AGENT_LEARNING_TELEMETRY", raising=False)
    monkeypatch.setattr(_sync, "_collector_reachable", lambda base, timeout=3.0: (True, "ok"))
    monkeypatch.setattr(
        _emit, "keyed_emit",
        lambda **kw: {"status": "synced", "trace_id": "a" * 32, "reason": None},
    )
    from fi.alk.telemetry import build_ledger_row
    from fi.alk.telemetry._ledger import RunLedger

    row = build_ledger_row({"status": "passed", "summary": {"verdict": "pass"},
                            "manifest": {"name": "good"}})
    out = _sync.sync_run(row)
    assert out["status"] == "synced" and out["sent"] is True
    assert row["run_id"] in RunLedger().read_cursor()["synced"]


# --- AC1: local path makes NO network call, appends a ledger row --------------
def test_run_telemetry_local_only_no_network(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AGENT_LEARNING_LEDGER_PATH", str(tmp_path))
    for var in ("FI_API_KEY", "FUTURE_AGI_API_KEY", "AGENT_LEARNING_API_KEY",
                "FI_SECRET_KEY", "FUTURE_AGI_SECRET_KEY", "AGENT_LEARNING_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)

    def _boom(**kw):  # the cloud path must NEVER be reached with no keys
        raise AssertionError("keyed_emit called on the no-key local path")

    monkeypatch.setattr(_emit, "keyed_emit", _boom)
    with _run.run_telemetry(kind="benchmark", name="local_ds") as rec:
        rec.set_metrics(pass_rate=0.5)
    assert rec.summary is not None
    assert rec.summary.status == "local"
    assert rec.summary.dashboard_url is None
    assert rec.summary.run_id  # ledger row was built/appended


def test_run_telemetry_auto_mode_emits_url(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AGENT_LEARNING_LEDGER_PATH", str(tmp_path))
    monkeypatch.setenv("AGENT_LEARNING_SYNC", "auto")
    monkeypatch.setenv("FI_API_KEY", "k")
    monkeypatch.setenv("FI_SECRET_KEY", "s")
    monkeypatch.setenv("AGENT_LEARNING_PROJECT_ID", "proj-xyz")
    monkeypatch.setattr(
        _emit, "keyed_emit",
        lambda **kw: {"status": "synced", "trace_id": "b" * 32, "reason": None},
    )
    with _run.run_telemetry(kind="optimize", name="auto_ds") as rec:
        rec.set_metrics(lift=0.3)
    assert rec.summary.status == "synced"
    assert rec.summary.url_kind == "deep_link"
    assert "proj-xyz/trace/" in rec.summary.dashboard_url


def test_local_mode_with_keys_does_not_emit(monkeypatch, tmp_path) -> None:
    """Keys present but mode=local (the test/gate default) => queue locally, no
    network. This is the P8 'stray key in CI' safety reconciliation."""
    monkeypatch.setenv("AGENT_LEARNING_LEDGER_PATH", str(tmp_path))
    monkeypatch.setenv("AGENT_LEARNING_SYNC", "local")
    monkeypatch.setenv("FI_API_KEY", "k")
    monkeypatch.setenv("FI_SECRET_KEY", "s")
    monkeypatch.setattr(
        _emit, "keyed_emit",
        lambda **kw: (_ for _ in ()).throw(AssertionError("emitted in local mode")),
    )
    with _run.run_telemetry(kind="benchmark", name="x") as rec:
        rec.set_metrics(pass_rate=1.0)
    assert rec.summary.status == "local"


# --- FR1: pipeline result carries the telemetry summary (framework-agnostic) ---
def test_run_benchmark_result_carries_telemetry(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AGENT_LEARNING_LEDGER_PATH", str(tmp_path))
    monkeypatch.setenv("AGENT_LEARNING_SYNC", "local")
    ds = tasks.compile_task_dataset({
        "name": "tele-bench",
        "tasks": [{
            "id": "t1", "title": "t1",
            "world": {"kind": "tool_api", "spec": {"max_turns": 2}},
            "difficulty": "easy",
            "objective": {"source": "declared",
                          "evals": [{"eval": "task_success", "weight": 1.0, "anchor": True}],
                          "guards": {"sentinel_rows": [{"id": "s"}], "min_guard_count": 1}},
            "scenario": {"name": "t1", "kind": "task",
                         "dataset": [{"persona": {"name": "P"}, "situation": "hi",
                                      "outcome": "done"}]},
            "verification": {"checks": [{"type": "contains", "value": "x"}], "threshold": 0.5},
        }],
    })

    def _runner(manifest):
        return {"results": [{"verdict": "pass", "scores": {"task_success": 1.0}}]}

    out = tasks.run_benchmark(ds, {"type": "python", "callable": "x:y"}, runner=_runner)
    assert "telemetry" in out
    assert out["telemetry"]["kind"] == "benchmark"
    assert out["telemetry"]["status"] == "local"
    assert out["telemetry"]["dashboard_url"] is None
