"""
Cost Span Processor.

Calculates and attaches cost information to LLM spans
based on token usage and model pricing.
"""

from typing import Any, Dict, Optional, Callable
import logging

from .base import BaseSpanProcessor, OTEL_AVAILABLE
from ..conventions import GenAIAttributes, CostAttributes
from ..types import TokenPricing

if OTEL_AVAILABLE:
    from opentelemetry.sdk.trace import ReadableSpan
else:
    ReadableSpan = Any

logger = logging.getLogger(__name__)


# Default pricing data (USD per 1K tokens)
# Updated as of 2024 - users should provide custom pricing for accuracy
DEFAULT_PRICING: Dict[str, TokenPricing] = {
    # OpenAI Models
    "gpt-4o": TokenPricing("gpt-4o", 0.005, 0.015),
    "gpt-4o-mini": TokenPricing("gpt-4o-mini", 0.00015, 0.0006),
    "gpt-4-turbo": TokenPricing("gpt-4-turbo", 0.01, 0.03),
    "gpt-4": TokenPricing("gpt-4", 0.03, 0.06),
    "gpt-4-32k": TokenPricing("gpt-4-32k", 0.06, 0.12),
    "gpt-3.5-turbo": TokenPricing("gpt-3.5-turbo", 0.0005, 0.0015),
    "gpt-3.5-turbo-16k": TokenPricing("gpt-3.5-turbo-16k", 0.003, 0.004),

    # Anthropic Models
    "claude-3-opus-20240229": TokenPricing("claude-3-opus-20240229", 0.015, 0.075),
    "claude-3-sonnet-20240229": TokenPricing("claude-3-sonnet-20240229", 0.003, 0.015),
    "claude-3-haiku-20240307": TokenPricing("claude-3-haiku-20240307", 0.00025, 0.00125),
    "claude-3-5-sonnet-20241022": TokenPricing("claude-3-5-sonnet-20241022", 0.003, 0.015),
    "claude-2.1": TokenPricing("claude-2.1", 0.008, 0.024),
    "claude-2.0": TokenPricing("claude-2.0", 0.008, 0.024),
    "claude-instant-1.2": TokenPricing("claude-instant-1.2", 0.0008, 0.0024),

    # Google Models
    "gemini-1.5-pro": TokenPricing("gemini-1.5-pro", 0.00125, 0.005),
    "gemini-1.5-flash": TokenPricing("gemini-1.5-flash", 0.000075, 0.0003),
    "gemini-pro": TokenPricing("gemini-pro", 0.00025, 0.0005),

    # Mistral Models
    "mistral-large-latest": TokenPricing("mistral-large-latest", 0.004, 0.012),
    "mistral-medium-latest": TokenPricing("mistral-medium-latest", 0.0027, 0.0081),
    "mistral-small-latest": TokenPricing("mistral-small-latest", 0.001, 0.003),
    "open-mixtral-8x7b": TokenPricing("open-mixtral-8x7b", 0.0007, 0.0007),
    "open-mistral-7b": TokenPricing("open-mistral-7b", 0.00025, 0.00025),

    # Cohere Models
    "command-r-plus": TokenPricing("command-r-plus", 0.003, 0.015),
    "command-r": TokenPricing("command-r", 0.0005, 0.0015),
    "command": TokenPricing("command", 0.001, 0.002),
    "command-light": TokenPricing("command-light", 0.0003, 0.0006),

    # Meta/Llama Models (typical cloud pricing)
    "llama-3-70b": TokenPricing("llama-3-70b", 0.0009, 0.0009),
    "llama-3-8b": TokenPricing("llama-3-8b", 0.0002, 0.0002),
    "llama-2-70b": TokenPricing("llama-2-70b", 0.0009, 0.0009),
    "llama-2-13b": TokenPricing("llama-2-13b", 0.0002, 0.0002),

    # Embedding Models
    "text-embedding-3-large": TokenPricing("text-embedding-3-large", 0.00013, 0.0),
    "text-embedding-3-small": TokenPricing("text-embedding-3-small", 0.00002, 0.0),
    "text-embedding-ada-002": TokenPricing("text-embedding-ada-002", 0.0001, 0.0),
}

# Model name aliases for normalization
MODEL_ALIASES: Dict[str, str] = {
    # OpenAI
    "gpt-4-0613": "gpt-4",
    "gpt-4-1106-preview": "gpt-4-turbo",
    "gpt-4-0125-preview": "gpt-4-turbo",
    "gpt-4-turbo-preview": "gpt-4-turbo",
    "gpt-3.5-turbo-0125": "gpt-3.5-turbo",
    "gpt-3.5-turbo-1106": "gpt-3.5-turbo",

    # Anthropic
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",

    # Google
    "gemini-pro-vision": "gemini-pro",
    "gemini-1.0-pro": "gemini-pro",

    # Mistral
    "mistral-large": "mistral-large-latest",
    "mistral-medium": "mistral-medium-latest",
    "mistral-small": "mistral-small-latest",
}


class CostSpanProcessor(BaseSpanProcessor):
    """
    Processor that calculates and attaches cost to LLM spans.

    Uses token counts and model pricing to calculate the
    cost of each LLM call. Supports custom pricing and
    cost alerts.

    Example:
        processor = CostSpanProcessor(
            custom_pricing={
                "my-fine-tuned-model": TokenPricing("my-model", 0.01, 0.02),
            },
            alert_threshold_usd=1.0,
            on_cost_alert=lambda cost, span_id: notify_slack(cost),
        )
    """

    def __init__(
        self,
        custom_pricing: Optional[Dict[str, TokenPricing]] = None,
        pricing_source: str = "default",
        currency: str = "USD",
        alert_threshold_usd: Optional[float] = None,
        on_cost_alert: Optional[Callable[[float, str], None]] = None,
        on_cost_calculated: Optional[Callable[[str, float, float, float], None]] = None,
        enabled: bool = True,
    ):
        """
        Initialize the cost processor.

        Args:
            custom_pricing: Custom model pricing to override defaults
            pricing_source: Source identifier for pricing data
            currency: Currency for cost values
            alert_threshold_usd: Alert if single call exceeds this
            on_cost_alert: Callback when cost exceeds threshold
            on_cost_calculated: Callback with (span_id, input_cost, output_cost, total_cost)
            enabled: Whether processor is active
        """
        super().__init__(enabled=enabled)
        self._custom_pricing = custom_pricing or {}
        self._pricing_source = pricing_source
        self._currency = currency
        self._alert_threshold_usd = alert_threshold_usd
        self._on_cost_alert = on_cost_alert
        self._on_cost_calculated = on_cost_calculated

        # Build combined pricing table
        self._pricing = {**DEFAULT_PRICING, **self._custom_pricing}

        # Track cumulative costs
        self._total_cost_usd = 0.0
        self._total_input_cost_usd = 0.0
        self._total_output_cost_usd = 0.0
        self._total_calls = 0

    @property
    def total_cost_usd(self) -> float:
        """Get total cost tracked so far."""
        return self._total_cost_usd

    @property
    def total_calls(self) -> int:
        """Get total number of calls tracked."""
        return self._total_calls

    def should_process(self, span: ReadableSpan) -> bool:
        """Check if this span should have cost calculated."""
        if not self.enabled:
            return False

        attrs = self._get_attributes(span)

        # Need token counts to calculate cost
        has_tokens = (
            GenAIAttributes.USAGE_INPUT_TOKENS in attrs or
            GenAIAttributes.USAGE_OUTPUT_TOKENS in attrs or
            "llm.usage.prompt_tokens" in attrs or
            "llm.usage.completion_tokens" in attrs
        )

        return has_tokens

    def _get_attributes(self, span: ReadableSpan) -> Dict[str, Any]:
        """Get span attributes safely."""
        try:
            return dict(span.attributes or {})
        except Exception:
            return {}

    def _get_span_id(self, span: ReadableSpan) -> str:
        """Get unique span identifier."""
        try:
            ctx = span.get_span_context()
            return f"{ctx.trace_id:032x}:{ctx.span_id:016x}"
        except Exception:
            return str(id(span))

    def on_end(self, span: ReadableSpan) -> None:
        """Calculate and record cost for the span."""
        if not self.should_process(span):
            return

        try:
            attrs = self._get_attributes(span)
            span_id = self._get_span_id(span)

            # Extract token counts
            input_tokens = self._extract_input_tokens(attrs)
            output_tokens = self._extract_output_tokens(attrs)

            if input_tokens is None and output_tokens is None:
                return

            # Get model name and pricing
            model = self._extract_model(attrs)
            pricing = self._get_pricing(model)

            if pricing is None:
                logger.debug(f"No pricing found for model: {model}")
                return

            # Calculate costs
            input_tokens = input_tokens or 0
            output_tokens = output_tokens or 0

            input_cost = input_tokens * pricing.input_per_token
            output_cost = output_tokens * pricing.output_per_token
            total_cost = input_cost + output_cost

            # Update cumulative totals
            self._total_cost_usd += total_cost
            self._total_input_cost_usd += input_cost
            self._total_output_cost_usd += output_cost
            self._total_calls += 1

            # Log cost
            logger.info(
                f"Cost [{span_id}] model={model} "
                f"input=${input_cost:.6f} output=${output_cost:.6f} "
                f"total=${total_cost:.6f}"
            )

            # Check alert threshold
            if self._alert_threshold_usd and total_cost > self._alert_threshold_usd:
                logger.warning(
                    f"Cost alert: ${total_cost:.4f} exceeds threshold "
                    f"${self._alert_threshold_usd:.4f}"
                )
                if self._on_cost_alert:
                    try:
                        self._on_cost_alert(total_cost, span_id)
                    except Exception as e:
                        logger.warning(f"Cost alert callback error: {e}")

            # Call callback
            if self._on_cost_calculated:
                try:
                    self._on_cost_calculated(span_id, input_cost, output_cost, total_cost)
                except Exception as e:
                    logger.warning(f"Cost callback error: {e}")

        except Exception as e:
            logger.warning(f"Cost calculation error: {e}")

    def _extract_input_tokens(self, attrs: Dict[str, Any]) -> Optional[int]:
        """Extract input token count from attributes."""
        for key in [
            GenAIAttributes.USAGE_INPUT_TOKENS,
            "llm.usage.prompt_tokens",
            "input_tokens",
            "prompt_tokens",
        ]:
            if key in attrs:
                try:
                    return int(attrs[key])
                except (ValueError, TypeError):
                    pass
        return None

    def _extract_output_tokens(self, attrs: Dict[str, Any]) -> Optional[int]:
        """Extract output token count from attributes."""
        for key in [
            GenAIAttributes.USAGE_OUTPUT_TOKENS,
            "llm.usage.completion_tokens",
            "output_tokens",
            "completion_tokens",
        ]:
            if key in attrs:
                try:
                    return int(attrs[key])
                except (ValueError, TypeError):
                    pass
        return None

    def _extract_model(self, attrs: Dict[str, Any]) -> Optional[str]:
        """Extract model name from attributes."""
        for key in [
            GenAIAttributes.REQUEST_MODEL,
            GenAIAttributes.RESPONSE_MODEL,
            "llm.model",
            "model",
            "model_name",
        ]:
            if key in attrs:
                return str(attrs[key])
        return None

    def _get_pricing(self, model: Optional[str]) -> Optional[TokenPricing]:
        """Get pricing for a model."""
        if model is None:
            return None

        # Check direct match
        if model in self._pricing:
            return self._pricing[model]

        # Check alias
        if model in MODEL_ALIASES:
            aliased = MODEL_ALIASES[model]
            if aliased in self._pricing:
                return self._pricing[aliased]

        # Try partial match (for versioned models)
        model_lower = model.lower()
        for pricing_model, pricing in self._pricing.items():
            if pricing_model.lower() in model_lower or model_lower in pricing_model.lower():
                return pricing

        return None

    def get_cost_attributes(
        self,
        input_cost: float,
        output_cost: float,
        total_cost: float,
    ) -> Dict[str, Any]:
        """
        Create cost attributes for a span.

        Args:
            input_cost: Cost for input tokens
            output_cost: Cost for output tokens
            total_cost: Total cost

        Returns:
            Dictionary of cost attributes
        """
        return {
            CostAttributes.INPUT: input_cost,
            CostAttributes.OUTPUT: output_cost,
            CostAttributes.TOTAL: total_cost,
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of costs tracked.

        Returns:
            Dictionary with cost summary
        """
        return {
            "total_cost_usd": self._total_cost_usd,
            "total_input_cost_usd": self._total_input_cost_usd,
            "total_output_cost_usd": self._total_output_cost_usd,
            "total_calls": self._total_calls,
            "average_cost_per_call": (
                self._total_cost_usd / self._total_calls
                if self._total_calls > 0
                else 0.0
            ),
            "currency": self._currency,
        }

    def reset_totals(self) -> None:
        """Reset cumulative cost totals."""
        self._total_cost_usd = 0.0
        self._total_input_cost_usd = 0.0
        self._total_output_cost_usd = 0.0
        self._total_calls = 0

    def add_custom_pricing(self, model: str, pricing: TokenPricing) -> None:
        """Add custom pricing for a model."""
        self._pricing[model] = pricing
        self._custom_pricing[model] = pricing


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    custom_pricing: Optional[Dict[str, TokenPricing]] = None,
) -> Dict[str, float]:
    """
    Calculate cost for an LLM call.

    Standalone utility function for cost calculation.

    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        custom_pricing: Optional custom pricing overrides

    Returns:
        Dictionary with input_cost, output_cost, total_cost
    """
    pricing_table = {**DEFAULT_PRICING, **(custom_pricing or {})}

    pricing = pricing_table.get(model)
    if pricing is None:
        # Try alias
        aliased = MODEL_ALIASES.get(model)
        if aliased:
            pricing = pricing_table.get(aliased)

    if pricing is None:
        # Try partial match
        model_lower = model.lower()
        for pricing_model, p in pricing_table.items():
            if pricing_model.lower() in model_lower:
                pricing = p
                break

    if pricing is None:
        return {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}

    input_cost = input_tokens * pricing.input_per_token
    output_cost = output_tokens * pricing.output_per_token

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


__all__ = [
    "CostSpanProcessor",
    "DEFAULT_PRICING",
    "MODEL_ALIASES",
    "calculate_cost",
]
