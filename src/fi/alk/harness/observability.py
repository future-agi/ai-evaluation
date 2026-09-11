"""Emitting a hosted harness run to Observe.

The run happens inside the sandbox, which is where everything opaque about a harness job lives:
contract authoring, environment generation and validation, the calls, the judge. Until now those
were readable only as log text in the diagnostics archive, which is why "what went wrong" is still
a manual read.

What lands is one session per job, holding a span per harness stage plus whatever the model
instrumentors record inside each. Stage spans come from the guest's own stage transitions, so the
timeline is the harness's own, not a second one invented here.

Two rules hold everywhere. The SDK is imported in-function, because the bundle image may not carry
it and an import error must not reach the run. And every failure is swallowed: a verdict may never
depend on telemetry, the same rule the diagnostics upload already follows.

Everything is environment-driven, so a deployment decides without a code change:

    HARNESS_OBSERVABILITY     off/false/0/no disables it outright, whatever else is set
    FI_API_KEY/FI_SECRET_KEY  which account the run reports into; absent means untraced
    FI_BASE_URL               which instance receives it
    FI_HARNESS_PROJECT        which project it lands in, defaulting to hosted-harness

The module holds process state because the guest handles exactly one job per process.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

_OFF = {"0", "off", "false", "no"}

_provider: Any = None
_tracer: Any = None
_session = contextlib.ExitStack()
_stage_span: Any = None


def enabled() -> bool:
    """Whether this run reports to Observe.

    The switch is explicit so an operator can turn tracing off while leaving the account
    credentials in place, which unsetting keys alone cannot express.
    """
    if os.getenv("HARNESS_OBSERVABILITY", "").strip().lower() in _OFF:
        return False
    return bool(os.getenv("FI_API_KEY") and os.getenv("FI_SECRET_KEY"))


def begin(job_id: str, run_id: str) -> None:
    """Open the job's session. Called once, as early as the job id is known."""
    global _provider, _tracer
    if not enabled() or _provider is not None:
        return
    try:
        from fi_instrumentation import register, using_session, using_simulator_attributes
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
        _tracer = _provider.get_tracer(__name__)
        _session.enter_context(using_session(job_id))
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


def _end_stage() -> None:
    global _stage_span
    if _stage_span is not None:
        _stage_span.end()
        _stage_span = None


def _instrument(provider: Any) -> None:
    """Turn on whichever model instrumentors the image actually ships."""
    for module, name in (
        ("traceai_google_genai", "GoogleGenAIInstrumentor"),
        ("traceai_litellm", "LiteLLMInstrumentor"),
        ("traceai_openai", "OpenAIInstrumentor"),
        ("traceai_anthropic", "AnthropicInstrumentor"),
    ):
        with contextlib.suppress(Exception):
            instrumentor = getattr(__import__(module, fromlist=[name]), name)
            instrumentor().instrument(tracer_provider=provider)
