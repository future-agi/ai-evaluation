"""Emitting a hosted harness run to Observe.

Everything opaque about a harness job happens inside the sandbox: contract authoring, environment
generation and validation, the calls, the judge. Until now that was readable only as log text in the
diagnostics archive, pulled out by hand after the fact.

What lands is one session per job, carrying a span per harness stage and per scenario, with whatever
the model instrumentors record nested inside. Stage spans come from the guest's own transitions, so
the timeline is the harness's real one rather than a second one invented here.

**Every job reports into one platform-owned account, never the customer's.** That is what makes a
fleet debuggable: runs from every organization land in one project, and tenancy travels as
attributes so a single run can still be found. The account is whatever ``FI_API_KEY`` and
``FI_SECRET_KEY`` the deployment sets.

Nothing personal is attached here. The identifiers are organization, workspace, job, run and
scenario ids plus the shape of the run; transcripts, prompts, personas, credentials and customer
names are never set by this module.

Two rules hold everywhere. The SDK is imported in-function, because the bundle image may not carry
it and an import error must not reach the run. And every failure is swallowed: a verdict may never
depend on telemetry, the same rule the diagnostics upload already follows.

Environment, so a deployment decides without a code change:

    HARNESS_OBSERVABILITY     off/false/0/no disables it outright, whatever else is set
    FI_API_KEY/FI_SECRET_KEY  the platform account every job reports into
    FI_BASE_URL               which instance receives it, including the collector's own port where
                              one is needed; unset means the public endpoint
    FI_HARNESS_PROJECT        which project, defaulting to hosted-harness

The module holds process state because the guest handles exactly one job per process.
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
_context: dict[str, Any] = {}


def enabled() -> bool:
    """Whether this run reports to Observe.

    The switch is explicit so an operator can turn tracing off while leaving the account
    credentials in place, which unsetting keys alone cannot express.
    """
    if os.getenv("HARNESS_OBSERVABILITY", "").strip().lower() in _OFF:
        return False
    return bool(os.getenv("FI_API_KEY") and os.getenv("FI_SECRET_KEY"))


def begin(job_id: str, run_id: str, context: Mapping[str, Any] | None = None) -> None:
    """Open the job's session. Called once, as early as the job id is known."""
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
            using_simulator_attributes,
            using_tags,
            using_user,
        )
        from fi_instrumentation.fi_types import ProjectType

        _provider = register(
            project_name=os.getenv("FI_HARNESS_PROJECT") or "hosted-harness",
            project_type=ProjectType.OBSERVE,
            # The platform binds the global provider to its own infrastructure tracing; taking it
            # here would redirect that into Observe.
            set_global_tracer_provider=False,
            verbose=False,
        )
        _instrument(_provider)
        # FITracer, not the raw tracer: session, user, metadata and tags ride on baggage and are
        # only written onto a span by the SDK's own wrapper.
        _tracer = FITracer(_provider.get_tracer(__name__))
        # One session per job. The organization occupies the user dimension so a fleet can be
        # sliced by tenant without giving each tenant its own account.
        _session.enter_context(using_session(job_id))
        organization = str(_context.get("organization_id") or "")
        if organization:
            _session.enter_context(using_user(organization))
        _session.enter_context(using_metadata(dict(_context)))
        _session.enter_context(using_tags(_tags()))
        _session.enter_context(
            using_simulator_attributes(
                {
                    "is_simulator_trace": True,
                    "run_test_id": run_id,
                    "test_execution_id": job_id,
                }
            )
        )
    except Exception:
        _provider = None
        _tracer = None


def stage(name: str) -> None:
    """Close the stage that was open and open one for ``name``.

    Stages are consecutive rather than nested, so the previous span ends where the next begins and
    the trace reads as the harness's own timeline.
    """
    global _stage_span
    if _tracer is None:
        return
    with contextlib.suppress(Exception):
        _end_stage()
        _stage_span = _tracer.start_span(f"harness.stage.{name}")
        _stage_span.set_attribute("gen_ai.span.kind", "CHAIN")
        _stage_span.set_attribute("harness.stage", name)
        _apply_context(_stage_span)


@contextlib.contextmanager
def scenario(key: str, index: int) -> Iterator[Any]:
    """One scenario, so a failure can be read against the run it belongs to."""
    if _tracer is None:
        yield None
        return
    span = None
    try:
        span = _tracer.start_span(f"harness.scenario.{key}")
        span.set_attribute("gen_ai.span.kind", "CHAIN")
        span.set_attribute("harness.scenario.key", key)
        span.set_attribute("harness.scenario.index", index)
        _apply_context(span)
    except Exception:
        span = None
    try:
        yield span
    finally:
        with contextlib.suppress(Exception):
            if span is not None:
                span.end()


def record(span: Any, **attributes: Any) -> None:
    """Attach an outcome to a span already opened here. Silent when untraced."""
    if span is None:
        return
    with contextlib.suppress(Exception):
        for name, value in attributes.items():
            if value is None:
                continue
            span.set_attribute(f"harness.{name}", _scalar(value))


def end() -> None:
    """Close the session and flush. The guest exits next, taking the exporter with it."""
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


def _tags() -> list[str]:
    """Coarse filters: the axes worth scanning a whole project by."""
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


def _end_stage() -> None:
    global _stage_span
    if _stage_span is not None:
        _stage_span.end()
        _stage_span = None


def _instrument(provider: Any) -> None:
    """Turn on whichever instrumentors the image actually ships.

    The first two cover the harness's own two stage backends, so authoring, scenario writing and
    the judge are recorded whichever model a job runs on. The rest are there for agents and tools
    that reach a provider directly.

    ``traceai-google-adk`` still pins ``google-genai<2`` while ``google-adk`` 2.7 requires
    ``>=2.12``; the override in pyproject installs it anyway because it works against 2.12.
    """
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
