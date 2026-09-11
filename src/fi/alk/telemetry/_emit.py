"""Export-result-aware OTLP emit (Phase 14, ARCH §2) — the truthful span sender.

THE FIX for the false-``synced`` bug (RESEARCH §1.2): the OTLP HTTP exporter
swallows export failures (it logs a 401 to stderr and returns
``SpanExportResult.FAILURE`` without raising), so the old ``register(...)`` path
completed its ``try`` block and reported ``synced`` while *nothing landed*. This
module owns the exporter via a recording wrapper, so a reported success is an
*observed* ``SpanExportResult.SUCCESS`` — never an assumption.

Vendored-engine boundary (gate #72): every ``fi_instrumentation`` import is
in-function. This module is imported only from inside the keyed branch of
``_run`` / ``_sync`` — never on the keyless import graph.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ._contract import (
    FI_KIT_PHASE_ATTR,
    FI_KIT_RUN_ID_ATTR,
    FI_KIT_WORLD_ATTR,
)

# project_type/version/semconv resource attrs the collector resolves a project by
# (verified constants, fi_instrumentation/otel.py:57-89; RESEARCH §2.3).
_RES_PROJECT_NAME = "project_name"
_RES_PROJECT_TYPE = "project_type"
_RES_PROJECT_VERSION_NAME = "project_version_name"
_RES_PROJECT_VERSION_ID = "project_version_id"
_RES_EVAL_TAGS = "eval_tags"
_RES_METADATA = "metadata"
_RES_SEMCONV = "semantic_convention"

DEFAULT_PROJECT_NAME = "agent-learning"
RUN_SPAN_NAME = "agent-learning.run"


def _recording_exporter(inner: Any) -> Any:
    """Wrap an OTLP SpanExporter so every ``export()`` result is recorded — the
    single source of truth for 'did the span actually land'."""

    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    class _RecordingExporter(SpanExporter):  # type: ignore[misc]
        def __init__(self, delegate: Any) -> None:
            self._inner = delegate
            self.results: list[Any] = []

        def export(self, spans: Any) -> Any:
            result = self._inner.export(spans)
            self.results.append(result)
            return result

        def shutdown(self) -> Any:
            return self._inner.shutdown()

        def force_flush(self, timeout_millis: int = 30_000) -> bool:
            flush = getattr(self._inner, "force_flush", None)
            return flush(timeout_millis) if flush else True

        @property
        def ok(self) -> bool:
            # success == at least one export, and EVERY export succeeded.
            return bool(self.results) and all(
                r is SpanExportResult.SUCCESS for r in self.results
            )

        @property
        def last_reason(self) -> str:
            if not self.results:
                return "no_export_attempted"
            return "ok" if self.ok else "export_rejected"

    return _RecordingExporter(inner)


def _build_provider(project_name: str, headers: Mapping[str, str]) -> tuple[Any, Any]:
    """Build a private tracer provider (NOT the global one — RESEARCH §4) with the
    FI resource attrs the dashboard filters on, and swap in the recording exporter
    so we can observe the real export result. Returns (provider, recorder)."""

    import uuid

    import fi_instrumentation.otel as fio
    from fi_instrumentation.otel import SimpleSpanProcessor, Transport
    from fi_instrumentation.settings import UuidIdGenerator
    from opentelemetry.sdk.resources import Resource

    # SimpleSpanProcessor(headers, transport) builds the correctly-configured OTLP
    # exporter (endpoint resolved from FI_BASE_URL/BASE_URL). We then swap its
    # exporter for the recording wrapper — reusing fi's construction, observing
    # the result (the HTTPSpanExporter ctor itself does not accept transport).
    processor = SimpleSpanProcessor(
        headers=dict(headers), transport=Transport.HTTP
    )
    recorder = _recording_exporter(processor.span_exporter)
    processor.span_exporter = recorder

    resource = Resource(
        attributes={
            _RES_PROJECT_NAME: project_name,
            _RES_PROJECT_TYPE: "experiment",
            _RES_PROJECT_VERSION_NAME: DEFAULT_PROJECT_NAME,
            _RES_PROJECT_VERSION_ID: str(uuid.uuid4()),
            _RES_EVAL_TAGS: "[]",
            _RES_METADATA: "{}",
            _RES_SEMCONV: "fi",
        }
    )
    provider = fio.TracerProvider(
        resource=resource,
        id_generator=UuidIdGenerator(),
        transport=Transport.HTTP,
        verbose=False,
    )
    provider.add_span_processor(processor)
    return provider, recorder


def _scalar(value: Any) -> Any:
    """OTel span attributes accept str/bool/int/float (and homogeneous lists).
    Coerce anything else to a compact string so a metric mapping never breaks
    the emit."""

    if isinstance(value, (str, bool, int, float)):
        return value
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)[:1024]


def keyed_emit(
    *,
    span_name: str,
    root_attrs: Mapping[str, Any],
    children: Sequence[tuple[str, Mapping[str, Any]]] = (),
    project_name: str,
    headers: Mapping[str, str],
    run_id: str = "",
    phase: str = "",
    world: str | None = None,
) -> dict[str, Any]:
    """Emit one run as a trace (root span + child spans) to the user's account and
    report the OBSERVED export result.

    Returns ``{status, trace_id, reason}`` where ``status`` is ``"synced"`` only
    when the recording exporter saw ``SpanExportResult.SUCCESS``; otherwise
    ``"export_failed"`` with a reason. Any exception degrades to local
    (``"deferred"``) — never propagates into the caller's run (R§3.5).
    """

    try:
        provider, recorder = _build_provider(project_name, headers)
        tracer = provider.get_tracer("fi.alk.telemetry")
        trace_id_hex = ""
        with tracer.start_as_current_span(span_name) as root:
            trace_id_hex = format(root.get_span_context().trace_id, "032x")
            if run_id:
                root.set_attribute(FI_KIT_RUN_ID_ATTR, run_id)
            if phase:
                root.set_attribute(FI_KIT_PHASE_ATTR, phase)
            if world:
                root.set_attribute(FI_KIT_WORLD_ATTR, str(world))
            for key, value in root_attrs.items():
                root.set_attribute(str(key), _scalar(value))
            for child_name, child_attrs in children:
                with tracer.start_as_current_span(str(child_name)) as child:
                    for key, value in (child_attrs or {}).items():
                        child.set_attribute(str(key), _scalar(value))
        provider.force_flush()
        provider.shutdown()
    except BaseException as exc:  # noqa: BLE001 — degrade-to-local (R§3.5)
        return {"status": "deferred", "trace_id": None, "reason": f"{type(exc).__name__}: {exc}"}

    if recorder.ok:
        return {"status": "synced", "trace_id": trace_id_hex, "reason": None}
    return {
        "status": "export_failed",
        "trace_id": None,  # nothing landed → no viewable trace
        "reason": recorder.last_reason,
    }
