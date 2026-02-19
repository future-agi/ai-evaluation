from .heuristics.aggregation_metrics import AggregatedMetric
from .heuristics.json_metrics import JsonSchema, ContainsJson, IsJson
from .heuristics.similarity_metrics import (
    BLEUScore,
    ROUGEScore,
    LevenshteinSimilarity,
    NumericSimilarity,
    EmbeddingSimilarity,
    SemanticListContains,
    RecallScore,
)
from .heuristics.string_metrics import (
    Regex,
    Contains,
    ContainsAny,
    ContainsAll,
    ContainsNone,
    Equals,
    StartsWith,
    EndsWith,
    LengthLessThan,
    LengthGreaterThan,
    LengthBetween,
    OneLine,
    ContainsEmail,
    IsEmail,
    ContainsLink,
    ContainsValidLink,
)
from .llm_as_judges import CustomLLMJudge

# RAG Metrics (optional import - may fail if dependencies missing)
try:
    from .rag import (
        # Types
        RAGInput,
        RAGRetrievalInput,
        RAGRankingInput,
        # Retrieval metrics
        ContextRecall,
        ContextPrecision,
        ContextEntityRecall,
        NoiseSensitivity,
        NDCG,
        MRR,
        # Generation metrics
        AnswerRelevancy,
        ContextUtilization,
        Groundedness,
        RAGFaithfulness,
        # Advanced metrics
        MultiHopReasoning,
        SourceAttribution,
        # Comprehensive scorers
        RAGScore,
        RAGScoreDetailed,
    )
    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False

# Structured Output Metrics
try:
    from .structured import (
        # Types
        ValidationMode,
        JSONInput,
        PydanticInput,
        YAMLInput,
        StructuredInput,
        ValidationError,
        ValidationResult,
        # Validators
        JSONValidator,
        PydanticValidator,
        YAMLValidator,
        # Metrics
        JSONValidation,
        JSONSyntaxOnly,
        SchemaCompliance,
        TypeCompliance,
        FieldCompleteness,
        RequiredFieldsOnly,
        FieldCoverage,
        HierarchyScore,
        TreeEditDistance,
        StructuredOutputScore,
        QuickStructuredCheck,
    )
    _STRUCTURED_AVAILABLE = True
except ImportError:
    _STRUCTURED_AVAILABLE = False

__all__ = [
    # Aggregation
    "AggregatedMetric",
    # JSON
    "JsonSchema",
    "ContainsJson",
    "IsJson",
    # Similarity
    "BLEUScore",
    "ROUGEScore",
    "LevenshteinSimilarity",
    "EmbeddingSimilarity",
    "NumericSimilarity",
    "SemanticListContains",
    "RecallScore",
    # String
    "Regex",
    "Contains",
    "ContainsAny",
    "ContainsAll",
    "ContainsNone",
    "Equals",
    "StartsWith",
    "EndsWith",
    "LengthLessThan",
    "LengthGreaterThan",
    "LengthBetween",
    "OneLine",
    "ContainsEmail",
    "IsEmail",
    "ContainsLink",
    "ContainsValidLink",
    # LLM as Judges
    "CustomLLMJudge",
    # RAG Metrics
    "RAGInput",
    "RAGRetrievalInput",
    "RAGRankingInput",
    "ContextRecall",
    "ContextPrecision",
    "ContextEntityRecall",
    "NoiseSensitivity",
    "NDCG",
    "MRR",
    "AnswerRelevancy",
    "ContextUtilization",
    "Groundedness",
    "RAGFaithfulness",
    "MultiHopReasoning",
    "SourceAttribution",
    "RAGScore",
    "RAGScoreDetailed",
    # Structured Output Metrics
    "ValidationMode",
    "JSONInput",
    "PydanticInput",
    "YAMLInput",
    "StructuredInput",
    "ValidationError",
    "ValidationResult",
    "JSONValidator",
    "PydanticValidator",
    "YAMLValidator",
    "JSONValidation",
    "JSONSyntaxOnly",
    "SchemaCompliance",
    "TypeCompliance",
    "FieldCompleteness",
    "RequiredFieldsOnly",
    "FieldCoverage",
    "HierarchyScore",
    "TreeEditDistance",
    "StructuredOutputScore",
    "QuickStructuredCheck",
]
