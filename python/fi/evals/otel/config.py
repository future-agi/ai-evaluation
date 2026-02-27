"""
OpenTelemetry Configuration.

Configuration types and defaults for the OTEL integration.
Supports multiple exporters for the "export anywhere" philosophy.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from .types import ExporterType, ProcessorType


class SamplingStrategy(str, Enum):
    """Trace sampling strategies."""

    ALWAYS_ON = "always_on"  # Sample all traces
    ALWAYS_OFF = "always_off"  # Sample no traces
    RATIO = "ratio"  # Sample based on ratio
    PARENT_BASED = "parent_based"  # Follow parent's sampling decision


@dataclass
class ExporterConfig:
    """Configuration for a single exporter."""

    type: ExporterType
    endpoint: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timeout_ms: int = 30000
    compression: Optional[str] = None  # "gzip", "none"
    insecure: bool = False  # Allow HTTP (not HTTPS)

    # Exporter-specific options
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessorConfig:
    """Configuration for a span processor."""

    type: ProcessorType
    enabled: bool = True

    # Batch processor settings
    max_queue_size: int = 2048
    max_export_batch_size: int = 512
    export_timeout_ms: int = 30000
    schedule_delay_ms: int = 5000

    # Processor-specific options
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    """Configuration for automatic evaluation of spans."""

    enabled: bool = True
    metrics: List[str] = field(default_factory=lambda: ["relevance", "coherence"])
    sample_rate: float = 1.0  # Fraction of spans to evaluate (0.0 to 1.0)
    async_evaluation: bool = True  # Don't block the request path
    timeout_ms: int = 5000  # Timeout for evaluation
    cache_enabled: bool = True  # Cache evaluation results
    cache_ttl_seconds: int = 3600  # Cache TTL

    # Model for LLM-based evaluation
    evaluator_model: Optional[str] = None  # None = use default


@dataclass
class CostConfig:
    """Configuration for cost tracking."""

    enabled: bool = True
    pricing_source: str = "litellm"  # Source for pricing data
    custom_pricing: Dict[str, Dict[str, float]] = field(default_factory=dict)
    currency: str = "USD"

    # Alerts
    alert_threshold_usd: Optional[float] = None  # Alert if exceeded
    alert_callback: Optional[str] = None  # Callback function name


@dataclass
class ContentConfig:
    """Configuration for content capture (prompts/completions)."""

    capture_prompts: bool = True
    capture_completions: bool = True
    max_content_length: int = 10000  # Truncate if longer
    redact_patterns: List[str] = field(default_factory=list)  # Regex patterns to redact

    # PII handling
    redact_pii: bool = False
    pii_types: List[str] = field(default_factory=lambda: ["email", "phone", "ssn"])


@dataclass
class ResourceConfig:
    """Configuration for OTEL resource attributes."""

    service_name: str = "llm-service"
    service_version: Optional[str] = None
    service_namespace: Optional[str] = None
    deployment_environment: str = "development"

    # Additional attributes
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class TraceConfig:
    """
    Complete configuration for OpenTelemetry tracing.

    This is the main configuration object for setting up tracing.
    Supports multiple exporters for maximum flexibility.

    Example:
        config = TraceConfig(
            service_name="my-ai-service",
            exporters=[
                ExporterConfig(type=ExporterType.OTLP_GRPC, endpoint="localhost:4317"),
                ExporterConfig(type=ExporterType.CONSOLE),
            ],
            evaluation=EvaluationConfig(
                metrics=["relevance", "faithfulness"],
                sample_rate=0.1,
            ),
        )
    """

    # Resource identification
    service_name: str = "llm-service"
    service_version: Optional[str] = None
    deployment_environment: str = "development"

    # Exporters - can export to multiple backends
    exporters: List[ExporterConfig] = field(default_factory=lambda: [
        ExporterConfig(type=ExporterType.CONSOLE)
    ])

    # Processors
    processors: List[ProcessorConfig] = field(default_factory=lambda: [
        ProcessorConfig(type=ProcessorType.LLM),
        ProcessorConfig(type=ProcessorType.BATCH),
    ])

    # Sampling
    sampling_strategy: SamplingStrategy = SamplingStrategy.ALWAYS_ON
    sampling_ratio: float = 1.0  # For RATIO strategy

    # Feature configurations
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    content: ContentConfig = field(default_factory=ContentConfig)
    resource: ResourceConfig = field(default_factory=ResourceConfig)

    # Propagation
    propagators: List[str] = field(default_factory=lambda: ["tracecontext", "baggage"])

    # Global settings
    enabled: bool = True
    debug: bool = False

    def __post_init__(self):
        """Sync resource config with top-level settings."""
        self.resource.service_name = self.service_name
        if self.service_version:
            self.resource.service_version = self.service_version
        self.resource.deployment_environment = self.deployment_environment

    @classmethod
    def development(cls, service_name: str = "llm-service") -> "TraceConfig":
        """
        Create a development configuration.

        Outputs to console, evaluates all spans, captures all content.
        """
        return cls(
            service_name=service_name,
            deployment_environment="development",
            exporters=[ExporterConfig(type=ExporterType.CONSOLE)],
            evaluation=EvaluationConfig(
                sample_rate=1.0,
                async_evaluation=False,
            ),
            content=ContentConfig(
                capture_prompts=True,
                capture_completions=True,
            ),
            debug=True,
        )

    @classmethod
    def production(
        cls,
        service_name: str,
        otlp_endpoint: str,
        service_version: Optional[str] = None,
        eval_sample_rate: float = 0.1,
    ) -> "TraceConfig":
        """
        Create a production configuration.

        Exports to OTLP, samples evaluations, async processing.
        """
        return cls(
            service_name=service_name,
            service_version=service_version,
            deployment_environment="production",
            exporters=[
                ExporterConfig(
                    type=ExporterType.OTLP_GRPC,
                    endpoint=otlp_endpoint,
                )
            ],
            evaluation=EvaluationConfig(
                sample_rate=eval_sample_rate,
                async_evaluation=True,
            ),
            content=ContentConfig(
                capture_prompts=True,
                capture_completions=True,
                redact_pii=True,
            ),
            sampling_strategy=SamplingStrategy.RATIO,
            sampling_ratio=0.5,  # Sample 50% of traces
            debug=False,
        )

    @classmethod
    def multi_backend(
        cls,
        service_name: str,
        backends: List[Dict[str, Any]],
    ) -> "TraceConfig":
        """
        Create a multi-backend configuration.

        Export to multiple observability platforms simultaneously.

        Args:
            service_name: Service name
            backends: List of backend configs, each with 'type' and optional 'endpoint'

        Example:
            config = TraceConfig.multi_backend(
                service_name="my-service",
                backends=[
                    {"type": "otlp_grpc", "endpoint": "localhost:4317"},
                    {"type": "jaeger", "endpoint": "localhost:6831"},
                    {"type": "console"},
                ]
            )
        """
        exporters = []
        for backend in backends:
            exporter_type = backend.get("type", "console")
            if isinstance(exporter_type, str):
                exporter_type = ExporterType(exporter_type)

            exporters.append(ExporterConfig(
                type=exporter_type,
                endpoint=backend.get("endpoint"),
                headers=backend.get("headers", {}),
                options=backend.get("options", {}),
            ))

        return cls(
            service_name=service_name,
            exporters=exporters,
        )

    def add_exporter(
        self,
        exporter_type: Union[str, ExporterType],
        endpoint: Optional[str] = None,
        **kwargs
    ) -> "TraceConfig":
        """
        Add an exporter to the configuration.

        Args:
            exporter_type: Type of exporter
            endpoint: Exporter endpoint
            **kwargs: Additional exporter options

        Returns:
            Self for chaining
        """
        if isinstance(exporter_type, str):
            exporter_type = ExporterType(exporter_type)

        self.exporters.append(ExporterConfig(
            type=exporter_type,
            endpoint=endpoint,
            **kwargs
        ))
        return self

    def with_evaluation(
        self,
        metrics: List[str],
        sample_rate: float = 1.0,
        async_evaluation: bool = True,
    ) -> "TraceConfig":
        """
        Configure evaluation settings.

        Args:
            metrics: List of metrics to evaluate
            sample_rate: Fraction of spans to evaluate
            async_evaluation: Whether to evaluate asynchronously

        Returns:
            Self for chaining
        """
        self.evaluation = EvaluationConfig(
            enabled=True,
            metrics=metrics,
            sample_rate=sample_rate,
            async_evaluation=async_evaluation,
        )
        return self

    def with_cost_tracking(
        self,
        alert_threshold: Optional[float] = None,
        custom_pricing: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> "TraceConfig":
        """
        Configure cost tracking.

        Args:
            alert_threshold: Alert if cost exceeds this (USD)
            custom_pricing: Custom pricing overrides

        Returns:
            Self for chaining
        """
        self.cost = CostConfig(
            enabled=True,
            alert_threshold_usd=alert_threshold,
            custom_pricing=custom_pricing or {},
        )
        return self


# Predefined exporter configurations for common backends
EXPORTER_PRESETS: Dict[str, ExporterConfig] = {
    "jaeger": ExporterConfig(
        type=ExporterType.JAEGER,
        endpoint="localhost:6831",
    ),
    "zipkin": ExporterConfig(
        type=ExporterType.ZIPKIN,
        endpoint="http://localhost:9411/api/v2/spans",
    ),
    "datadog": ExporterConfig(
        type=ExporterType.DATADOG,
        endpoint="https://trace.agent.datadoghq.com",
    ),
    "honeycomb": ExporterConfig(
        type=ExporterType.HONEYCOMB,
        endpoint="https://api.honeycomb.io",
    ),
    "grafana_tempo": ExporterConfig(
        type=ExporterType.OTLP_GRPC,
        endpoint="localhost:4317",
    ),
    "arize": ExporterConfig(
        type=ExporterType.ARIZE,
        endpoint="https://otlp.arize.com",
    ),
}


def get_exporter_preset(name: str) -> ExporterConfig:
    """Get a predefined exporter configuration."""
    if name not in EXPORTER_PRESETS:
        available = ", ".join(EXPORTER_PRESETS.keys())
        raise ValueError(f"Unknown exporter preset: {name}. Available: {available}")
    return EXPORTER_PRESETS[name]


__all__ = [
    "SamplingStrategy",
    "ExporterConfig",
    "ProcessorConfig",
    "EvaluationConfig",
    "CostConfig",
    "ContentConfig",
    "ResourceConfig",
    "TraceConfig",
    "EXPORTER_PRESETS",
    "get_exporter_preset",
]
