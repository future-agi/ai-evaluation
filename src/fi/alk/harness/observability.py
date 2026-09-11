"""Emit a hosted harness run to Observe: one session per job, a span per stage and per scenario,
with the model and tool calls each stage made nested inside it.

Every job reports into the platform's own account, so tenancy travels as attributes. Identifiers
only. Configured by HARNESS_OBSERVABILITY, FI_API_KEY, FI_SECRET_KEY, FI_BASE_URL and
FI_HARNESS_PROJECT; absent credentials mean untraced, and every failure here is swallowed.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Iterator, Mapping

_OFF = {"0", "off", "false", "no"}

_provider: Any = None
_tracer: Any = None
_session = contextlib.ExitStack()
_stage_span: Any = None
_stage_token: Any = None
_context: dict[str, Any] = {}


def enabled() -> bool:
    if os.getenv("HARNESS_OBSERVABILITY", "").strip().lower() in _OFF:
        return False
    return bool(os.getenv("FI_API_KEY") and os.getenv("FI_SECRET_KEY"))


def begin(job_id: str, run_id: str, context: Mapping[str, Any] | None = None) -> None:
    """Open the job's session, once, as early as the job id is known."""
    global _provider, _tracer, _context
    if not enabled() or _provider is not None:
        return
    _context = {str(k): v for k, v in (context or {}).items() if v not in ("", None)}
    _context.setdefault("job_id", job_id)
    _context.setdefault("run_id", run_id)
    try:
        from fi_instrumentation import (
            FITracer,
            register,
            using_metadata,
            using_session,
            using_tags,
            using_user,
        )
        from fi_instrumentation.fi_types import ProjectType

        _provider = register(
            project_name=os.getenv("FI_HARNESS_PROJECT") or "hosted-harness",
            project_type=ProjectType.OBSERVE,
            # The platform binds the global provider to its own infrastructure tracing.
            set_global_tracer_provider=False,
            verbose=False,
        )
        _instrument(_provider)
        # Only FITracer writes the baggage-carried session, user, metadata and tags onto a span.
        _tracer = FITracer(_provider.get_tracer(__name__))
        _session.enter_context(using_session(job_id))
        organization = str(_context.get("organization_id") or "")
        if organization:
            _session.enter_context(using_user(organization))
        _session.enter_context(using_metadata(dict(_context)))
        _session.enter_context(using_tags(_tags()))
    except Exception:
        _provider = None
        _tracer = None


def stage_event(event_type: str, name: str, payload: Mapping[str, Any] | None = None) -> None:
    """Drive stage spans from the harness's own stage announcements."""
    if _tracer is None:
        return
    with contextlib.suppress(Exception):
        if event_type.endswith(".started"):
            _open_stage(name)
        elif event_type.endswith((".completed", ".failed")) and _stage_span is not None:
            record(
                _stage_span,
                stage_outcome="failed" if event_type.endswith(".failed") else "completed",
                stage_status=(payload or {}).get("status"),
            )
            _end_stage()


def stage(name: str) -> None:
    if _tracer is None:
        return
    with contextlib.suppress(Exception):
        _open_stage(name)


@contextlib.contextmanager
def scenario(key: str, index: int) -> Iterator[Any]:
    if _tracer is None:
        yield None
        return
    span = None
    token = None
    try:
        from opentelemetry import context as otel_context
        from opentelemetry import trace as otel_trace

        span = _tracer.start_span(f"harness.scenario.{key}")
        span.set_attribute("gen_ai.span.kind", "CHAIN")
        span.set_attribute("harness.scenario.key", key)
        span.set_attribute("harness.scenario.index", index)
        _apply_context(span)
        # Current for this task only, so the scenario's own model and tool calls nest inside it.
        token = otel_context.attach(otel_trace.set_span_in_context(span))
    except Exception:
        span = None
    try:
        yield span
    finally:
        with contextlib.suppress(Exception):
            if token is not None:
                from opentelemetry import context as otel_context

                otel_context.detach(token)
        with contextlib.suppress(Exception):
            if span is not None:
                span.end()


def record(span: Any, **attributes: Any) -> None:
    if span is None:
        return
    with contextlib.suppress(Exception):
        for name, value in attributes.items():
            if value is not None:
                span.set_attribute(f"harness.{name}", _scalar(value))


def end() -> None:
    """Close the session and flush; the guest exits next, taking the exporter with it."""
    global _provider, _tracer
    with contextlib.suppress(Exception):
        _end_stage()
    with contextlib.suppress(Exception):
        _session.close()
    if _provider is not None:
        with contextlib.suppress(Exception):
            _provider.force_flush()
    _provider = None
    _tracer = None


def _open_stage(name: str) -> None:
    global _stage_span, _stage_token
    from opentelemetry import context as otel_context
    from opentelemetry import trace as otel_trace

    _end_stage()
    _stage_span = _tracer.start_span(f"harness.stage.{name}")
    _stage_span.set_attribute("gen_ai.span.kind", "CHAIN")
    _stage_span.set_attribute("harness.stage", name)
    _apply_context(_stage_span)
    # Current, so the stage's model and tool calls nest inside it rather than arriving as roots.
    _stage_token = otel_context.attach(otel_trace.set_span_in_context(_stage_span))


def _end_stage() -> None:
    global _stage_span, _stage_token
    if _stage_token is not None:
        with contextlib.suppress(Exception):
            from opentelemetry import context as otel_context

            otel_context.detach(_stage_token)
        _stage_token = None
    if _stage_span is not None:
        _stage_span.end()
        _stage_span = None


def _tags() -> list[str]:
    tags = ["hosted-harness"]
    for key in ("connector", "deployment"):
        value = str(_context.get(key) or "")
        if value:
            tags.append(f"{key}:{value}")
    return tags


def _scalar(value: Any) -> Any:
    return value if isinstance(value, (str, bool, int, float)) else str(value)


def _apply_context(span: Any) -> None:
    for name, value in _context.items():
        with contextlib.suppress(Exception):
            span.set_attribute(f"harness.{name}", _scalar(value))


def _instrument(provider: Any) -> None:
    """Enable whichever instrumentors the image ships; the backend in use is the one that emits."""
    for module, name in (
        ("traceai_google_adk", "GoogleADKInstrumentor"),
        ("traceai_claude_agent_sdk", "ClaudeAgentInstrumentor"),
        ("traceai_google_genai", "GoogleGenAIInstrumentor"),
        ("traceai_litellm", "LiteLLMInstrumentor"),
        ("traceai_openai", "OpenAIInstrumentor"),
        ("traceai_anthropic", "AnthropicInstrumentor"),
    ):
        with contextlib.suppress(Exception):
            instrumentor = getattr(__import__(module, fromlist=[name]), name)
            instrumentor().instrument(tracer_provider=provider)
