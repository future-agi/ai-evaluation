"""Integration hooks for wiring feedback into the pipeline.

These functions are called from the augmentation flow when a feedback_store
is provided. Kept in a separate module to avoid circular imports.
"""

import logging
from typing import Any, Dict, Optional

from .store import FeedbackStore
from .retriever import FeedbackRetriever

logger = logging.getLogger(__name__)

# Module-level default store (set via configure_feedback)
_default_store: Optional[FeedbackStore] = None
_default_max_examples: int = 3


def configure_feedback(
    store: FeedbackStore,
    max_examples: int = 3,
) -> None:
    """Set a global default feedback store for all augmented metric runs.

    After calling this, all augmented runs will automatically retrieve
    feedback examples -- no need to pass feedback_store= every time.

    Args:
        store: The FeedbackStore to use globally.
        max_examples: Max few-shot examples per query.

    Usage:
        from fi.evals.feedback import ChromaFeedbackStore, configure_feedback

        store = ChromaFeedbackStore()
        configure_feedback(store)

        # Now all augmented runs automatically use feedback
        result = run_metric("faithfulness", ..., augment=True, model="gemini/...")
    """
    global _default_store, _default_max_examples
    _default_store = store
    _default_max_examples = max_examples
    logger.info(f"Feedback configured globally (max_examples={max_examples})")


def get_default_store() -> Optional[FeedbackStore]:
    """Get the globally configured feedback store, if any."""
    return _default_store


def retrieve_feedback_config(
    metric_name: str,
    inputs: Dict[str, Any],
    store: Optional[FeedbackStore] = None,
    config: Optional[Dict[str, Any]] = None,
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """Retrieve feedback examples and inject into config dict.

    Called from the augmentation flow. Can also be called directly.

    Args:
        metric_name: Metric being run.
        inputs: Current inputs.
        store: Explicit store override. Falls back to global default.
        config: Existing config dict to merge into.
        max_examples: Override for max examples.

    Returns:
        Config dict with few_shot_examples populated (or unchanged if
        no store is configured / no feedback found).
    """
    effective_store = store or _default_store
    if effective_store is None:
        return dict(config or {})

    n = max_examples or _default_max_examples
    retriever = FeedbackRetriever(store=effective_store, max_examples=n)
    return retriever.inject_into_config(metric_name, inputs, config)
