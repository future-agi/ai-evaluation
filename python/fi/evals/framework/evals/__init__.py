"""
Built-in evaluations using the evaluation framework.

This module provides ready-to-use evaluations for common use cases:
- Semantic evaluations (similarity, coherence, entailment)
- Multi-modal evaluations (image-text consistency, VQA)
- Safety evaluations (toxicity, bias, PII detection)
- Quality evaluations (relevance, completeness, accuracy)

All evaluations implement BaseEvaluation and work with the Evaluator class.

Example:
    from fi.evals.framework import Evaluator, ExecutionMode
    from fi.evals.framework.evals import (
        SemanticSimilarityEval,
        CoherenceEval,
        FactualConsistencyEval,
        ImageTextConsistencyEval,
    )

    evaluator = Evaluator(
        evaluations=[
            SemanticSimilarityEval(),
            CoherenceEval(),
            FactualConsistencyEval(),
            ImageTextConsistencyEval(),
        ],
        mode=ExecutionMode.NON_BLOCKING,
    )

    result = evaluator.run({
        "response": "The capital of France is Paris.",
        "reference": "Paris is the capital city of France.",
        "context": "France is a country in Western Europe.",
        "image_description": "A map of France",
        "text": "This shows the country of France",
    })
"""

from .semantic import (
    SemanticSimilarityEval,
    CoherenceEval,
    EntailmentEval,
    ContradictionEval,
    FactualConsistencyEval,
    SemanticEvalResult,
)

from .multimodal import (
    ImageTextConsistencyEval,
    CaptionQualityEval,
    VisualQAEval,
    ImageSafetyEval,
    CrossModalConsistencyEval,
    MultiModalEvalResult,
)

from .agentic import (
    ToolUseCorrectnessEval,
    TrajectoryEfficiencyEval,
    GoalCompletionEval,
    ActionSafetyEval,
    ReasoningQualityEval,
    AgenticEvalResult,
    AgentAction,
)

from .builder import (
    EvalBuilder,
    CustomEvaluation,
    CustomEvalResult,
    custom_eval,
    simple_eval,
    comparison_eval,
    threshold_eval,
    pattern_match_eval,
)

__all__ = [
    # Semantic
    "SemanticSimilarityEval",
    "CoherenceEval",
    "EntailmentEval",
    "ContradictionEval",
    "FactualConsistencyEval",
    "SemanticEvalResult",
    # Multi-modal
    "ImageTextConsistencyEval",
    "CaptionQualityEval",
    "VisualQAEval",
    "ImageSafetyEval",
    "CrossModalConsistencyEval",
    "MultiModalEvalResult",
    # Agentic
    "ToolUseCorrectnessEval",
    "TrajectoryEfficiencyEval",
    "GoalCompletionEval",
    "ActionSafetyEval",
    "ReasoningQualityEval",
    "AgenticEvalResult",
    "AgentAction",
    # Builder
    "EvalBuilder",
    "CustomEvaluation",
    "CustomEvalResult",
    "custom_eval",
    "simple_eval",
    "comparison_eval",
    "threshold_eval",
    "pattern_match_eval",
]
