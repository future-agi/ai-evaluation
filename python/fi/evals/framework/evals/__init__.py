"""
Built-in evaluations using the evaluation framework.

This module provides ready-to-use evaluations for common use cases:
- Semantic evaluations (coherence)
- Multi-modal evaluations (image-text consistency, VQA)
- Agentic evaluations (action safety, reasoning quality)
- Builder utilities (custom eval creation)

All evaluations implement BaseEvaluation and work with the Evaluator class.

Example:
    from fi.evals.framework import Evaluator, ExecutionMode
    from fi.evals.framework.evals import (
        CoherenceEval,
        ImageTextConsistencyEval,
        ActionSafetyEval,
    )

    evaluator = Evaluator(
        evaluations=[
            CoherenceEval(),
            ImageTextConsistencyEval(),
            ActionSafetyEval(),
        ],
        mode=ExecutionMode.NON_BLOCKING,
    )
"""

from .semantic import (
    CoherenceEval,
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
    "CoherenceEval",
    "SemanticEvalResult",
    # Multi-modal
    "ImageTextConsistencyEval",
    "CaptionQualityEval",
    "VisualQAEval",
    "ImageSafetyEval",
    "CrossModalConsistencyEval",
    "MultiModalEvalResult",
    # Agentic
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
