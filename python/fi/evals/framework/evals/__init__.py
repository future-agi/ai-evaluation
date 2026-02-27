"""
Built-in evaluations using the evaluation framework.

This module provides ready-to-use evaluations for common use cases:
- Semantic evaluations (coherence)
- Agentic evaluations (action safety, reasoning quality)
- Builder utilities (custom eval creation)

All evaluations implement BaseEvaluation and work with the FrameworkEvaluator class.

Example:
    from fi.evals.framework import FrameworkEvaluator, ExecutionMode
    from fi.evals.framework.evals import (
        CoherenceEval,
        ActionSafetyEval,
    )

    evaluator = FrameworkEvaluator(
        evaluations=[
            CoherenceEval(),
            ActionSafetyEval(),
        ],
        mode=ExecutionMode.NON_BLOCKING,
    )
"""

from .semantic import (
    CoherenceEval,
    SemanticEvalResult,
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
