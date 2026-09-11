"""``run_telemetry`` (Phase 14, ARCH §4) — the ONE telemetry surface every kit run
wraps its body in. The W&B / promptfoo model:

  * local path  — ALWAYS: append the ledger row + return/print a RunSummary. No
    network. Works credential-free (promptfoo-local).
  * cloud path  — ADDITIVE, only when keys resolve and the collector is reachable:
    emit the run as a real trace and print a clickable dashboard URL (W&B
    "View run at …"). The URL is printed ONLY on an OBSERVED export success.

Logs go to STDERR (W&B convention) so a kit run's STDOUT stays clean (the gate
example asserts empty stdout). Keys gate the destination, never the capability
(P8-D2); ``AGENT_LEARNING_TELEMETRY=off`` binds everything (P8-D6).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from ..config import AgentLearningConfig
from ._contract import SYNC_MODE_AUTO, kill_switch_on, ledger_dir, sync_mode
from ._ledger import RunLedger
from ._row import build_ledger_row


@dataclass
class RunSummary:
    """What a kit run reports — the value the caller attaches to its result and
    ``_log_summary`` renders."""

    kind: str
    name: str
    status: str = "local"  # local | synced | export_failed | deferred | off
    metrics: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    trace_id: str | None = None
    dashboard_url: str | None = None
    url_kind: str | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "status": self.status,
            "metrics": dict(self.metrics),
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "dashboard_url": self.dashboard_url,
            "url_kind": self.url_kind,
            "reason": self.reason,
        }


class RunRecorder:
    """Collects metrics + child-span specs during a run; ``summary`` is populated
    on context exit and readable by the caller afterwards."""

    def __init__(self, kind: str, name: str, world: str | None = None) -> None:
        self.kind = kind
        self.name = name
        self.world = world
        self.metrics: dict[str, Any] = {}
        self.children: list[tuple[str, dict[str, Any]]] = []
        self.verdict: str | None = None
        self.summary: RunSummary | None = None

    def set_metrics(self, **metrics: Any) -> None:
        self.metrics.update(metrics)

    def set_verdict(self, verdict: str) -> None:
        self.verdict = verdict

    def add_child(self, name: str, attrs: Mapping[str, Any]) -> None:
        self.children.append((str(name), dict(attrs)))


def _ledger_payload(rec: RunRecorder) -> dict[str, Any]:
    return {
        "kind": rec.kind,
        "summary": {
            "name": rec.name,
            "verdict": rec.verdict,
            "metrics": rec.metrics,
        },
    }


def _log_summary(summary: RunSummary) -> None:
    """Render the W&B / promptfoo line to stderr."""

    metric_bits = " · ".join(
        f"{k} {v}" for k, v in summary.metrics.items()
        if isinstance(v, (int, float, str))
    )
    head = f"agent-learning: {summary.kind} '{summary.name}'"
    if metric_bits:
        head += f" · {metric_bits}"
    print(head, file=sys.stderr)

    if summary.status == "synced" and summary.dashboard_url:
        if summary.url_kind == "deep_link":
            print(f"agent-learning: 🔗 view in dashboard → {summary.dashboard_url}", file=sys.stderr)
        elif summary.url_kind == "project":
            print(f"agent-learning: 🔗 view in dashboard (project) → {summary.dashboard_url}", file=sys.stderr)
        else:  # list_fallback
            print(
                f"agent-learning: 🔗 dashboard → {summary.dashboard_url} "
                f"(find project '{summary.name}' / '{summary.metrics.get('project_name', '')}')",
                file=sys.stderr,
            )
    elif summary.status == "export_failed":
        print(
            f"agent-learning: ⚠ dashboard export not accepted ({summary.reason}) — logged locally only",
            file=sys.stderr,
        )
    elif summary.status == "deferred":
        print(
            f"agent-learning: dashboard unreachable ({summary.reason}) — logged locally only",
            file=sys.stderr,
        )
    elif summary.status == "off":
        print("agent-learning: telemetry off — nothing logged or sent", file=sys.stderr)
    else:  # local
        print(f"agent-learning: logged locally → {ledger_dir()}", file=sys.stderr)
        print(
            "agent-learning: set FI_API_KEY + FI_SECRET_KEY to view runs in the dashboard",
            file=sys.stderr,
        )


def _finalize(rec: RunRecorder, *, project_name: str | None) -> RunSummary:
    summary = RunSummary(kind=rec.kind, name=rec.name, metrics=dict(rec.metrics))

    # Kill switch binds EVERYTHING incl. the ledger (P8-D6).
    if kill_switch_on():
        summary.status = "off"
        return summary

    # Local path — always (FR2).
    row = build_ledger_row(_ledger_payload(rec))
    try:
        RunLedger().append(row)
    except Exception:  # noqa: BLE001 — a ledger write failure must not break the run
        pass
    summary.run_id = row.get("run_id")

    config = AgentLearningConfig.from_env()
    # Cloud path requires keys AND mode=auto (W&B-online). mode=local (the test/
    # gate default) queues locally — no surprise network in release/CI (P8).
    if not (config.api_key and config.secret_key) or sync_mode() != SYNC_MODE_AUTO:
        summary.status = "local"
        return summary

    # Cloud path — additive (FR3). Import network-capable code only here.
    from . import _emit, _url

    proj = project_name or "agent-learning"
    summary.metrics.setdefault("project_name", proj)
    emit = _emit.keyed_emit(
        span_name=_emit.RUN_SPAN_NAME,
        root_attrs={"kind": rec.kind, "name": rec.name, **rec.metrics},
        children=rec.children,
        project_name=proj,
        headers={"X-Api-Key": config.api_key, "X-Secret-Key": config.secret_key},
        run_id=summary.run_id or "",
        phase=rec.kind,
        world=rec.world,
    )
    summary.status = emit["status"]
    summary.reason = emit.get("reason")
    if emit["status"] == "synced":
        summary.trace_id = emit.get("trace_id")
        url = _url.build_dashboard_url(proj, summary.trace_id, config=config)
        summary.dashboard_url = url["url"]
        summary.url_kind = url["kind"]
    return summary


def emit_run(
    *,
    kind: str,
    name: str,
    metrics: Mapping[str, Any] | None = None,
    verdict: str | None = None,
    children: list[tuple[str, dict[str, Any]]] | None = None,
    world: str | None = None,
    project_name: str | None = None,
) -> RunSummary:
    """Non-context entrypoint for code paths that have already computed their
    results (``run_benchmark`` / ``optimize_against_dataset`` / ``improve_agent_
    code``): finalize one run (local ledger always; cloud emit when mode=auto +
    keys), log the W&B/promptfoo line, and return the summary. Never raises into
    the caller."""

    try:
        rec = RunRecorder(kind=kind, name=name, world=world)
        if metrics:
            rec.set_metrics(**dict(metrics))
        if verdict:
            rec.set_verdict(verdict)
        for child_name, attrs in children or []:
            rec.add_child(child_name, attrs)
        rec.summary = _finalize(rec, project_name=project_name)
        _log_summary(rec.summary)
        return rec.summary
    except Exception:  # noqa: BLE001 — telemetry is a side-channel, never fatal
        return RunSummary(kind=kind, name=name, status="local")


@contextmanager
def run_telemetry(
    *,
    kind: str,
    name: str,
    world: str | None = None,
    project_name: str | None = None,
) -> Iterator[RunRecorder]:
    """Wrap a kit run. Yields a ``RunRecorder``; after the block, ``rec.summary``
    holds the finalized ``RunSummary`` (status + dashboard URL). Telemetry never
    raises into the wrapped run."""

    rec = RunRecorder(kind=kind, name=name, world=world)
    try:
        yield rec
    finally:
        try:
            rec.summary = _finalize(rec, project_name=project_name)
            _log_summary(rec.summary)
        except Exception:  # noqa: BLE001 — telemetry is a side-channel, never fatal
            rec.summary = RunSummary(kind=kind, name=name, status="local")
