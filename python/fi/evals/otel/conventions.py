"""
OpenTelemetry Semantic Conventions for GenAI.

Follows the OpenTelemetry GenAI semantic conventions specification:
https://opentelemetry.io/docs/specs/semconv/gen-ai/

Aligned with traceAI SDK attribute namespace (gen_ai.*) so that
fi.evals spans render correctly alongside traceAI instrumentation
spans on the FutureAGI dashboard.
"""

from typing import Dict, Any, Optional
import warnings


class GenAIAttributes:
    """
    GenAI semantic conventions for span attributes.

    These are the standard attribute names defined by OpenTelemetry
    for GenAI/LLM operations. Using these conventions ensures your
    traces are compatible with any OTEL-compliant backend.
    """

    # System identification
    SYSTEM = "gen_ai.system"  # Deprecated: use PROVIDER_NAME
    PROVIDER_NAME = "gen_ai.provider.name"  # e.g., "openai", "anthropic"

    # Span kind (traceAI convention)
    SPAN_KIND = "gen_ai.span.kind"  # "LLM", "CHAIN", "GUARDRAIL", etc.

    # Operation
    OPERATION_NAME = "gen_ai.operation.name"  # e.g., "chat", "completion", "embedding"

    # Model information
    REQUEST_MODEL = "gen_ai.request.model"  # Model requested, e.g., "gpt-4"
    RESPONSE_MODEL = "gen_ai.response.model"  # Model that responded (may differ)

    # Request parameters
    REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    REQUEST_TOP_P = "gen_ai.request.top_p"
    REQUEST_TOP_K = "gen_ai.request.top_k"
    REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
    REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
    REQUEST_SEED = "gen_ai.request.seed"
    REQUEST_PARAMETERS = "gen_ai.request.parameters"  # Full invocation params JSON

    # Token usage
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"

    # Response metadata
    RESPONSE_FINISH_REASON = "gen_ai.response.finish_reason"  # singular (compat)
    RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"  # plural, list
    RESPONSE_ID = "gen_ai.response.id"

    # Messages (JSON arrays — replaces indexed prompt/completion)
    INPUT_MESSAGES = "gen_ai.input.messages"  # JSON array of messages
    OUTPUT_MESSAGES = "gen_ai.output.messages"  # JSON array of messages
    SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"

    # Conversation
    CONVERSATION_ID = "gen_ai.conversation.id"

    # Content (indexed for multi-turn) — DEPRECATED, prefer INPUT/OUTPUT_MESSAGES
    PROMPT_TEMPLATE = "gen_ai.prompt.{index}.content"
    PROMPT_ROLE_TEMPLATE = "gen_ai.prompt.{index}.role"
    COMPLETION_TEMPLATE = "gen_ai.completion.{index}.content"
    COMPLETION_ROLE_TEMPLATE = "gen_ai.completion.{index}.role"

    @classmethod
    def prompt_content(cls, index: int = 0) -> str:
        """Deprecated: use INPUT_MESSAGES instead."""
        return cls.PROMPT_TEMPLATE.format(index=index)

    @classmethod
    def prompt_role(cls, index: int = 0) -> str:
        """Deprecated: use INPUT_MESSAGES instead."""
        return cls.PROMPT_ROLE_TEMPLATE.format(index=index)

    @classmethod
    def completion_content(cls, index: int = 0) -> str:
        """Deprecated: use OUTPUT_MESSAGES instead."""
        return cls.COMPLETION_TEMPLATE.format(index=index)

    @classmethod
    def completion_role(cls, index: int = 0) -> str:
        """Deprecated: use OUTPUT_MESSAGES instead."""
        return cls.COMPLETION_ROLE_TEMPLATE.format(index=index)


class CostAttributes:
    """
    Cost tracking attributes (gen_ai.cost.* namespace).

    Aligned with traceAI conventions.
    """

    TOTAL = "gen_ai.cost.total"
    INPUT = "gen_ai.cost.input"
    OUTPUT = "gen_ai.cost.output"
    CACHE_WRITE = "gen_ai.cost.cache_write"


# Backward-compat alias
class LLMCostAttributes:
    """
    Deprecated: use CostAttributes instead.

    Legacy llm.cost.* namespace kept for backward compatibility reads.
    """

    INPUT_COST_USD = "llm.cost.input_usd"
    OUTPUT_COST_USD = "llm.cost.output_usd"
    TOTAL_COST_USD = "llm.cost.total_usd"
    CURRENCY = "llm.cost.currency"
    PRICING_MODEL = "llm.cost.pricing_model"


class EvaluationAttributes:
    """
    Evaluation score attributes.

    New namespace: gen_ai.evaluation.*
    Legacy namespace: eval.{metric} (kept for dual-write transition)
    """

    # --- New gen_ai.evaluation.* namespace ---
    NAME = "gen_ai.evaluation.name"
    SCORE_VALUE = "gen_ai.evaluation.score.value"
    SCORE_LABEL = "gen_ai.evaluation.score.label"
    EXPLANATION = "gen_ai.evaluation.explanation"
    TARGET_SPAN_ID = "gen_ai.evaluation.target_span_id"

    # --- Legacy eval.{metric} namespace (dual-write) ---
    SCORE_TEMPLATE = "eval.{metric}"
    REASON_TEMPLATE = "eval.{metric}.reason"
    LATENCY_TEMPLATE = "eval.{metric}.latency_ms"

    # Metadata (legacy)
    EVALUATED_AT = "eval.evaluated_at"
    EVALUATOR_VERSION = "eval.evaluator_version"
    SAMPLE_RATE = "eval.sample_rate"
    EVALUATION_MODEL = "eval.model"

    @classmethod
    def score(cls, metric: str) -> str:
        """Legacy: eval.{metric} score key."""
        return cls.SCORE_TEMPLATE.format(metric=metric)

    @classmethod
    def reason(cls, metric: str) -> str:
        """Legacy: eval.{metric}.reason key."""
        return cls.REASON_TEMPLATE.format(metric=metric)

    @classmethod
    def latency(cls, metric: str) -> str:
        """Legacy: eval.{metric}.latency_ms key."""
        return cls.LATENCY_TEMPLATE.format(metric=metric)


class GuardrailAttributes:
    """
    Guardrail attributes.

    New namespace: gen_ai.guardrail.*
    Dashboard-facing attrs (guardrail.*) kept for frontend compat.
    """

    # --- New gen_ai.guardrail.* namespace ---
    GEN_AI_NAME = "gen_ai.guardrail.name"
    GEN_AI_TYPE = "gen_ai.guardrail.type"
    GEN_AI_RESULT = "gen_ai.guardrail.result"  # "allow" / "block" / "warn"
    GEN_AI_SCORE = "gen_ai.guardrail.score"
    GEN_AI_CATEGORIES = "gen_ai.guardrail.categories"
    GEN_AI_MODIFIED_OUTPUT = "gen_ai.guardrail.modified_output"

    # --- Dashboard-facing legacy attrs ---
    PASSED = "guardrail.passed"
    BLOCKED = "guardrail.blocked"
    CATEGORIES = "guardrail.categories"
    SCORE = "guardrail.score"
    BACKEND = "guardrail.backend"
    LATENCY_MS = "guardrail.latency_ms"
    STATUS = "guardrail.status"
    RULES = "guardrail.rules"
    FAILED_RULE = "guardrail.failed_rule"
    COMPLETED_RULES = "guardrail.completed_rules"
    REASONS = "guardrail.reasons"




class RAGAttributes:
    """
    Custom attributes for RAG (Retrieval-Augmented Generation) operations.
    """

    NUM_DOCUMENTS = "rag.num_documents"
    CONTEXT_LENGTH = "rag.context_length"
    RETRIEVER_TYPE = "rag.retriever"
    TOP_K = "rag.top_k"
    SIMILARITY_THRESHOLD = "rag.similarity_threshold"
    RERANKER = "rag.reranker"
    DOCUMENTS_TEMPLATE = "rag.document.{index}.content"
    DOCUMENT_SCORE_TEMPLATE = "rag.document.{index}.score"
    DOCUMENT_ID_TEMPLATE = "rag.document.{index}.id"

    @classmethod
    def document_content(cls, index: int) -> str:
        return cls.DOCUMENTS_TEMPLATE.format(index=index)

    @classmethod
    def document_score(cls, index: int) -> str:
        return cls.DOCUMENT_SCORE_TEMPLATE.format(index=index)

    @classmethod
    def document_id(cls, index: int) -> str:
        return cls.DOCUMENT_ID_TEMPLATE.format(index=index)


class SpanNames:
    """
    Standard span names for LLM operations.
    """

    # Core LLM operations
    LLM_CHAT = "llm.chat"
    LLM_COMPLETION = "llm.completion"
    LLM_EMBEDDING = "llm.embedding"

    # RAG operations
    RAG_RETRIEVE = "rag.retrieve"
    RAG_RERANK = "rag.rerank"
    RAG_GENERATE = "rag.generate"
    RAG_PIPELINE = "rag.pipeline"

    # Agent operations
    AGENT_STEP = "agent.step"
    AGENT_TOOL_CALL = "agent.tool_call"
    AGENT_PLANNING = "agent.planning"

    # Safety operations
    GUARDRAIL_INPUT = "guardrail.input"
    GUARDRAIL_OUTPUT = "guardrail.output"
    GUARDRAIL_RETRIEVAL = "guardrail.retrieval"

    # Evaluation operations
    EVAL_RUN = "eval.run"
    EVAL_METRIC = "eval.metric"


# Provider system names (for gen_ai.system / gen_ai.provider.name attribute)
SYSTEM_OPENAI = "openai"
SYSTEM_ANTHROPIC = "anthropic"
SYSTEM_COHERE = "cohere"
SYSTEM_GOOGLE = "google"
SYSTEM_MISTRAL = "mistral"
SYSTEM_META = "meta"
SYSTEM_AWS_BEDROCK = "aws_bedrock"
SYSTEM_AZURE_OPENAI = "azure_openai"
SYSTEM_HUGGINGFACE = "huggingface"
SYSTEM_OLLAMA = "ollama"
SYSTEM_TOGETHER = "together"
SYSTEM_ANYSCALE = "anyscale"
SYSTEM_GROQ = "groq"
SYSTEM_FIREWORKS = "fireworks"
SYSTEM_REPLICATE = "replicate"
SYSTEM_CUSTOM = "custom"

# Operation names
OPERATION_CHAT = "chat"
OPERATION_COMPLETION = "completion"
OPERATION_EMBEDDING = "embedding"
OPERATION_RERANK = "rerank"

# Finish reasons
FINISH_STOP = "stop"
FINISH_LENGTH = "length"
FINISH_TOOL_CALLS = "tool_calls"
FINISH_CONTENT_FILTER = "content_filter"
FINISH_ERROR = "error"


def normalize_system_name(provider: str) -> str:
    """
    Normalize provider name to standard system name.

    Args:
        provider: Provider name (various formats)

    Returns:
        Normalized system name for gen_ai.provider.name attribute
    """
    provider_lower = provider.lower().strip()

    mapping = {
        "openai": SYSTEM_OPENAI,
        "gpt": SYSTEM_OPENAI,
        "chatgpt": SYSTEM_OPENAI,
        "anthropic": SYSTEM_ANTHROPIC,
        "claude": SYSTEM_ANTHROPIC,
        "cohere": SYSTEM_COHERE,
        "command": SYSTEM_COHERE,
        "google": SYSTEM_GOOGLE,
        "gemini": SYSTEM_GOOGLE,
        "palm": SYSTEM_GOOGLE,
        "mistral": SYSTEM_MISTRAL,
        "mixtral": SYSTEM_MISTRAL,
        "meta": SYSTEM_META,
        "llama": SYSTEM_META,
        "bedrock": SYSTEM_AWS_BEDROCK,
        "aws": SYSTEM_AWS_BEDROCK,
        "azure": SYSTEM_AZURE_OPENAI,
        "huggingface": SYSTEM_HUGGINGFACE,
        "hf": SYSTEM_HUGGINGFACE,
        "ollama": SYSTEM_OLLAMA,
        "together": SYSTEM_TOGETHER,
        "anyscale": SYSTEM_ANYSCALE,
        "groq": SYSTEM_GROQ,
        "fireworks": SYSTEM_FIREWORKS,
        "replicate": SYSTEM_REPLICATE,
    }

    for key, value in mapping.items():
        if key in provider_lower:
            return value

    return SYSTEM_CUSTOM


def create_llm_span_attributes(
    system: str,
    model: str,
    operation: str = OPERATION_CHAT,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    finish_reason: Optional[str] = None,
    **extra_attrs
) -> Dict[str, Any]:
    """
    Create a standardized attribute dictionary for an LLM span.

    Args:
        system: Provider system name (will be normalized)
        model: Model name
        operation: Operation type (chat, completion, embedding)
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        temperature: Temperature setting
        max_tokens: Max tokens setting
        finish_reason: Finish reason
        **extra_attrs: Additional attributes

    Returns:
        Dictionary of OTEL-compliant attributes
    """
    normalized = normalize_system_name(system)
    attrs: Dict[str, Any] = {
        GenAIAttributes.SPAN_KIND: "LLM",
        GenAIAttributes.PROVIDER_NAME: normalized,
        GenAIAttributes.OPERATION_NAME: operation,
        GenAIAttributes.REQUEST_MODEL: model,
    }

    if input_tokens is not None:
        attrs[GenAIAttributes.USAGE_INPUT_TOKENS] = input_tokens
    if output_tokens is not None:
        attrs[GenAIAttributes.USAGE_OUTPUT_TOKENS] = output_tokens
    if input_tokens is not None and output_tokens is not None:
        attrs[GenAIAttributes.USAGE_TOTAL_TOKENS] = input_tokens + output_tokens
    if temperature is not None:
        attrs[GenAIAttributes.REQUEST_TEMPERATURE] = temperature
    if max_tokens is not None:
        attrs[GenAIAttributes.REQUEST_MAX_TOKENS] = max_tokens
    if finish_reason is not None:
        attrs[GenAIAttributes.RESPONSE_FINISH_REASON] = finish_reason

    # Add extra attributes
    attrs.update(extra_attrs)

    return attrs


def create_evaluation_attributes(
    metric: str,
    score: float,
    reason: Optional[str] = None,
    latency_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Create attributes for an evaluation result.

    Uses gen_ai.evaluation.* namespace.

    Args:
        metric: Evaluation metric name
        score: Score value (0.0 to 1.0)
        reason: Explanation for the score
        latency_ms: Time taken to evaluate

    Returns:
        Dictionary of evaluation attributes
    """
    attrs: Dict[str, Any] = {
        EvaluationAttributes.NAME: metric,
        EvaluationAttributes.SCORE_VALUE: score,
    }

    if reason:
        attrs[EvaluationAttributes.EXPLANATION] = reason
    if latency_ms is not None:
        attrs[EvaluationAttributes.latency(metric)] = latency_ms

    return attrs


__all__ = [
    # Attribute classes
    "GenAIAttributes",
    "CostAttributes",
    "LLMCostAttributes",
    "EvaluationAttributes",
    "RAGAttributes",
    "GuardrailAttributes",
    "SpanNames",
    # System constants
    "SYSTEM_OPENAI",
    "SYSTEM_ANTHROPIC",
    "SYSTEM_COHERE",
    "SYSTEM_GOOGLE",
    "SYSTEM_MISTRAL",
    "SYSTEM_META",
    "SYSTEM_AWS_BEDROCK",
    "SYSTEM_AZURE_OPENAI",
    "SYSTEM_HUGGINGFACE",
    "SYSTEM_OLLAMA",
    "SYSTEM_TOGETHER",
    "SYSTEM_ANYSCALE",
    "SYSTEM_GROQ",
    "SYSTEM_FIREWORKS",
    "SYSTEM_REPLICATE",
    "SYSTEM_CUSTOM",
    # Operation constants
    "OPERATION_CHAT",
    "OPERATION_COMPLETION",
    "OPERATION_EMBEDDING",
    "OPERATION_RERANK",
    # Finish reason constants
    "FINISH_STOP",
    "FINISH_LENGTH",
    "FINISH_TOOL_CALLS",
    "FINISH_CONTENT_FILTER",
    "FINISH_ERROR",
    # Helper functions
    "normalize_system_name",
    "create_llm_span_attributes",
    "create_evaluation_attributes",
]
