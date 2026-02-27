"""Feedback retrieval and few-shot formatting.

Retrieves semantically similar feedback from the store and formats it
as few-shot examples for the LLM judge pipeline.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from .store import FeedbackStore
from .types import FeedbackEntry

logger = logging.getLogger(__name__)


class FeedbackRetriever:
    """Retrieves semantically similar feedback and formats as few-shot examples.

    This is the bridge between the feedback store and the LLM judge pipeline.
    When a metric is run with a feedback store, the retriever:

    1. Builds an embedding query from the current inputs
    2. Searches the store for similar past feedback entries
    3. Converts matching entries to the few_shot_examples format
       expected by CustomLLMJudge's Jinja2 template

    Args:
        store: The FeedbackStore to search.
        max_examples: Maximum number of few-shot examples to inject. Default 3.
    """

    def __init__(
        self,
        store: FeedbackStore,
        max_examples: int = 3,
    ):
        self.store = store
        self.max_examples = max_examples

    def build_query_text(self, metric_name: str, inputs: Dict[str, Any]) -> str:
        """Build a query string from inputs for semantic search.

        Uses the same concatenation strategy as FeedbackEntry.to_embedding_text()
        to ensure query-document alignment.
        """
        parts = [f"metric: {metric_name}"]
        for key in ("output", "response", "context", "input", "query"):
            val = inputs.get(key)
            if val:
                text = val if isinstance(val, str) else json.dumps(val, default=str)
                parts.append(f"{key}: {text[:500]}")
        return "\n".join(parts)

    def retrieve_few_shot_examples(
        self,
        metric_name: str,
        inputs: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Retrieve few-shot examples from feedback store.

        Returns a list of dicts in the format expected by
        CustomLLMJudge's config["few_shot_examples"]:

            [{"inputs": {...}, "output": '{"score": 0.8, "reason": "..."}'}]

        Args:
            metric_name: The metric being run.
            inputs: The current inputs.

        Returns:
            List of few-shot example dicts, possibly empty if no feedback exists.
        """
        if self.store.count(metric_name) == 0:
            return []

        query_text = self.build_query_text(metric_name, inputs)

        similar_entries = self.store.query_similar(
            metric_name=metric_name,
            text=query_text,
            n_results=self.max_examples,
        )

        if not similar_entries:
            return []

        examples = []
        for entry in similar_entries:
            # Only include entries where the developer provided a correction
            if entry.correct_score is None and not entry.correct_reason:
                continue
            examples.append(entry.to_few_shot())

        if examples:
            logger.debug(
                f"Retrieved {len(examples)} feedback examples for '{metric_name}'"
            )

        return examples

    def inject_into_config(
        self,
        metric_name: str,
        inputs: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Retrieve feedback and merge into a config dict for LLMEngine/CustomLLMJudge.

        This is the primary integration point. Call this before passing config
        to LLMEngine.run() to inject few-shot examples.

        Args:
            metric_name: Metric name.
            inputs: Current inputs.
            config: Existing config dict (will not be mutated).

        Returns:
            New config dict with few_shot_examples populated.
        """
        config = dict(config or {})

        examples = self.retrieve_few_shot_examples(metric_name, inputs)
        if examples:
            # Merge with any existing few-shot examples
            existing = config.get("few_shot_examples", [])
            config["few_shot_examples"] = existing + examples

        return config
