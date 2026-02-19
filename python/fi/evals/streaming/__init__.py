"""Streaming Evaluation Module.

Provides real-time evaluation of LLM outputs as tokens stream in,
with support for early stopping based on configurable policies.

Example:
    from fi.evals.streaming import StreamingEvaluator, StreamingConfig, EarlyStopPolicy

    # Create evaluator
    evaluator = StreamingEvaluator(
        config=StreamingConfig(
            min_chunk_size=10,
            max_chunk_size=100,
            enable_early_stop=True,
        ),
        policy=EarlyStopPolicy.default(),
    )

    # Add evaluation functions
    evaluator.add_eval("toxicity", toxicity_scorer, threshold=0.7, pass_above=False)

    # Process stream
    for token in llm_stream:
        result = evaluator.process_token(token)
        if result and result.should_stop:
            print(f"Early stop: {result.stop_reason}")
            break

    # Get final results
    final_result = evaluator.finalize()
    print(final_result.summary())
"""

from .types import (
    ChunkResult,
    EarlyStopCondition,
    EarlyStopReason,
    StreamingConfig,
    StreamingEvalResult,
    StreamingState,
)
from .buffer import BufferState, ChunkBuffer
from .policy import EarlyStopPolicy, PolicyState
from .evaluator import EvalSpec, StreamingEvaluator
from .scorers import (
    toxicity_scorer,
    safety_scorer,
    pii_scorer,
    jailbreak_scorer,
    coherence_scorer,
    quality_scorer,
    safety_composite_scorer,
    quality_composite_scorer,
    create_keyword_scorer,
    create_pattern_scorer,
    CompositeScorer,
)

__all__ = [
    # Types
    "ChunkResult",
    "EarlyStopCondition",
    "EarlyStopReason",
    "StreamingConfig",
    "StreamingEvalResult",
    "StreamingState",
    # Buffer
    "BufferState",
    "ChunkBuffer",
    # Policy
    "EarlyStopPolicy",
    "PolicyState",
    # Evaluator
    "EvalSpec",
    "StreamingEvaluator",
    # Scorers
    "toxicity_scorer",
    "safety_scorer",
    "pii_scorer",
    "jailbreak_scorer",
    "coherence_scorer",
    "quality_scorer",
    "safety_composite_scorer",
    "quality_composite_scorer",
    "create_keyword_scorer",
    "create_pattern_scorer",
    "CompositeScorer",
]
