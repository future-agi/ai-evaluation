"""The harness's span shape: stages and scenarios must parent the calls made inside them."""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from fi.alk.harness import observability


@pytest.fixture
def exporter(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(observability, "_tracer", provider.get_tracer(__name__))
    monkeypatch.setattr(observability, "_context", {"job_id": "job-1"})
    yield exporter, provider.get_tracer(__name__)
    observability._end_stage()


def _by_name(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


def test_a_stage_span_parents_the_calls_made_during_it(exporter):
    exporter, tracer = exporter
    observability.stage_event("harness.stage.started", "environment")
    with tracer.start_as_current_span("call_llm"):
        pass
    observability.stage_event("harness.stage.completed", "environment")

    spans = _by_name(exporter)
    stage = spans["harness.stage.environment"]
    assert spans["call_llm"].parent.span_id == stage.context.span_id
    assert stage.parent is None


def test_a_scenario_span_parents_the_calls_made_inside_it(exporter):
    exporter, tracer = exporter
    with observability.scenario("checkout", 0):
        with tracer.start_as_current_span("execute_tool"):
            pass

    spans = _by_name(exporter)
    scenario = spans["harness.scenario.checkout"]
    assert spans["execute_tool"].parent.span_id == scenario.context.span_id


def test_a_stage_records_the_outcome_it_ended_with(exporter):
    exporter, _ = exporter
    observability.stage_event("harness.stage.started", "calls")
    observability.stage_event("harness.stage.failed", "calls", {"status": "boom"})

    stage = _by_name(exporter)["harness.stage.calls"]
    assert stage.attributes["harness.stage_outcome"] == "failed"
    assert stage.attributes["harness.stage_status"] == "boom"


def test_a_second_stage_closes_the_one_before_it(exporter):
    exporter, _ = exporter
    observability.stage_event("harness.stage.started", "understand")
    observability.stage_event("harness.stage.started", "environment")

    assert "harness.stage.understand" in _by_name(exporter)


def test_tracing_off_is_silent_rather_than_fatal(monkeypatch):
    monkeypatch.setattr(observability, "_tracer", None)
    observability.stage_event("harness.stage.started", "environment")
    with observability.scenario("k", 0) as span:
        assert span is None
