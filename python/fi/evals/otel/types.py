"""
OpenTelemetry Types and Data Structures.

Core type definitions for the OTEL integration module.
"""

from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime


class ExporterType(str, Enum):
    """Supported exporter backends."""

    # Standard OTEL exporters
    OTLP_GRPC = "otlp_grpc"
    OTLP_HTTP = "otlp_http"
    JAEGER = "jaeger"
    ZIPKIN = "zipkin"
    CONSOLE = "console"

    # Vendor-specific
    DATADOG = "datadog"
    HONEYCOMB = "honeycomb"
    NEWRELIC = "newrelic"

    # LLM Observability platforms
    ARIZE = "arize"
    LANGFUSE = "langfuse"
    PHOENIX = "phoenix"

    # Custom
    FUTUREAGI = "futureagi"
    CUSTOM = "custom"


class SpanKind(str, Enum):
    """Span kinds for LLM operations (uppercase, matching traceAI)."""

    LLM = "LLM"
    CHAIN = "CHAIN"
    TOOL = "TOOL"
    AGENT = "AGENT"
    GUARDRAIL = "GUARDRAIL"
    EVALUATOR = "EVALUATOR"
    RETRIEVER = "RETRIEVER"
    EMBEDDING = "EMBEDDING"
    RERANKER = "RERANKER"
    CONVERSATION = "CONVERSATION"
    VECTOR_DB = "VECTOR_DB"
    UNKNOWN = "UNKNOWN"


class ProcessorType(str, Enum):
    """Available span processor types."""

    LLM = "llm"
    EVALUATION = "evaluation"
    COST = "cost"
    GUARDRAIL = "guardrail"
    BATCH = "batch"
    SIMPLE = "simple"


@dataclass
class TokenPricing:
    """Token pricing for a model."""

    model: str
    input_per_1k: float  # USD per 1K input tokens
    output_per_1k: float  # USD per 1K output tokens

    @property
    def input_per_token(self) -> float:
        return self.input_per_1k / 1000

    @property
    def output_per_token(self) -> float:
        return self.output_per_1k / 1000


@dataclass
class SpanAttributes:
    """Standard span attributes for LLM operations."""

    # GenAI standard attributes
    system: Optional[str] = None  # gen_ai.system
    operation_name: Optional[str] = None  # gen_ai.operation.name
    request_model: Optional[str] = None  # gen_ai.request.model
    response_model: Optional[str] = None  # gen_ai.response.model

    # Token usage
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Request parameters
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None

    # Content (may be redacted)
    prompt: Optional[str] = None
    completion: Optional[str] = None

    # Cost
    cost_input_usd: Optional[float] = None
    cost_output_usd: Optional[float] = None
    cost_total_usd: Optional[float] = None

    # Evaluation scores
    eval_scores: Dict[str, float] = field(default_factory=dict)
    eval_reasons: Dict[str, str] = field(default_factory=dict)

    # Metadata
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OTEL attribute dictionary using gen_ai.* namespace."""
        attrs = {}

        if self.system:
            attrs["gen_ai.provider.name"] = self.system
        if self.operation_name:
            attrs["gen_ai.operation.name"] = self.operation_name
        if self.request_model:
            attrs["gen_ai.request.model"] = self.request_model
        if self.response_model:
            attrs["gen_ai.response.model"] = self.response_model

        if self.input_tokens is not None:
            attrs["gen_ai.usage.input_tokens"] = self.input_tokens
        if self.output_tokens is not None:
            attrs["gen_ai.usage.output_tokens"] = self.output_tokens
        if self.total_tokens is not None:
            attrs["gen_ai.usage.total_tokens"] = self.total_tokens

        if self.temperature is not None:
            attrs["gen_ai.request.temperature"] = self.temperature
        if self.max_tokens is not None:
            attrs["gen_ai.request.max_tokens"] = self.max_tokens
        if self.top_p is not None:
            attrs["gen_ai.request.top_p"] = self.top_p

        if self.prompt:
            attrs["gen_ai.input.messages"] = self.prompt
        if self.completion:
            attrs["gen_ai.output.messages"] = self.completion

        if self.cost_input_usd is not None:
            attrs["gen_ai.cost.input"] = self.cost_input_usd
        if self.cost_output_usd is not None:
            attrs["gen_ai.cost.output"] = self.cost_output_usd
        if self.cost_total_usd is not None:
            attrs["gen_ai.cost.total"] = self.cost_total_usd

        for metric, score in self.eval_scores.items():
            attrs["gen_ai.evaluation.name"] = metric
            attrs["gen_ai.evaluation.score.value"] = score
        for metric, reason in self.eval_reasons.items():
            attrs["gen_ai.evaluation.explanation"] = reason

        if self.user_id:
            attrs["user.id"] = self.user_id
        if self.session_id:
            attrs["session.id"] = self.session_id

        return attrs


@dataclass
class EvaluationResult:
    """Result from span evaluation."""

    metric: str
    score: float
    reason: Optional[str] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportResult:
    """Result from span export."""

    success: bool
    spans_exported: int
    spans_failed: int = 0
    error: Optional[str] = None
    backend: Optional[str] = None
    latency_ms: Optional[float] = None


@dataclass
class TraceContext:
    """Context for a trace/span."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    trace_flags: int = 1  # Sampled
    trace_state: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return bool(self.trace_id and self.span_id)

    @property
    def is_sampled(self) -> bool:
        return bool(self.trace_flags & 0x01)


# Type aliases for callbacks
EvaluatorFn = Callable[[str, str, str], List[EvaluationResult]]  # (prompt, response, model) -> results
CostCalculatorFn = Callable[[str, int, int], float]  # (model, input_tokens, output_tokens) -> cost
SpanFilterFn = Callable[[Any], bool]  # (span) -> should_process


__all__ = [
    "ExporterType",
    "SpanKind",
    "ProcessorType",
    "TokenPricing",
    "SpanAttributes",
    "EvaluationResult",
    "ExportResult",
    "TraceContext",
    "EvaluatorFn",
    "CostCalculatorFn",
    "SpanFilterFn",
]
