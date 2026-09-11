"""
OpenAI Moderation Backend for Guardrails.

Uses OpenAI's Moderation API (FREE) for content screening.
Supports the omni-moderation-latest model with 13 categories.

Docs: https://platform.openai.com/docs/guides/moderation
"""

import os
import time
from typing import Any, Dict, List, Optional

from fi.evals.guardrails.backends.base import BaseBackend
from fi.evals.guardrails.config import GuardrailModel, RailType
from fi.evals.guardrails.types import GuardrailResult


# Category mapping from OpenAI to our standard categories
OPENAI_CATEGORY_MAP = {
    "hate": "hate_speech",
    "hate/threatening": "hate_speech",
    "harassment": "harassment",
    "harassment/threatening": "harassment",
    "self-harm": "self_harm",
    "self-harm/intent": "self_harm",
    "self-harm/instructions": "self_harm",
    "sexual": "sexual_content",
    "sexual/minors": "sexual_content",
    "violence": "violence",
    "violence/graphic": "violence",
    "illicit": "illegal_activity",
    "illicit/violent": "illegal_activity",
}


class OpenAIBackend(BaseBackend):
    """
    OpenAI Moderation API backend.

    This is a FREE API provided by OpenAI for content moderation.
    It supports 13 categories with confidence scores from 0 to 1.

    Usage:
        backend = OpenAIBackend(
            model=GuardrailModel.OPENAI_MODERATION,
            api_key="sk-..."  # or set OPENAI_API_KEY env var
        )
        results = backend.classify("some content", RailType.INPUT)

    Rate Limits:
        - 1,000 requests per minute
        - 150,000 tokens per minute
        - Does not count toward API usage limits
    """

    def __init__(
        self,
        model: GuardrailModel,
        api_key: Optional[str] = None,
        moderation_model: str = "omni-moderation-latest",
    ):
        """
        Initialize OpenAI Moderation backend.

        Args:
            model: The guardrail model enum value
            api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)
            moderation_model: OpenAI moderation model to use
        """
        super().__init__(model)

        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self._moderation_model = moderation_model
        self._client = None

    def _get_client(self):
        """Lazy-load the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package required for OpenAI backend. "
                    "Install with: pip install openai"
                )
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def classify(
        self,
        content: str,
        rail_type: RailType,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[GuardrailResult]:
        """
        Classify content using OpenAI Moderation API.

        Args:
            content: Content to classify
            rail_type: Type of rail (input, output, retrieval)
            context: Optional context (not used by OpenAI)
            metadata: Optional metadata (not used by OpenAI)

        Returns:
            List of GuardrailResult objects for flagged categories
        """
        start_time = time.time()
        results = []

        # Handle empty content
        if not content or not content.strip():
            return [
                GuardrailResult(
                    passed=True,
                    category="empty",
                    score=0.0,
                    model=self.model_name,
                    reason="Empty or whitespace-only content",
                    action="pass",
                    latency_ms=(time.time() - start_time) * 1000,
                )
            ]

        try:
            client = self._get_client()

            # Call OpenAI Moderation API
            response = client.moderations.create(
                input=content,
                model=self._moderation_model,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            if not response.results:
                return [
                    GuardrailResult(
                        passed=True,
                        category="safe",
                        score=0.0,
                        model=self.model_name,
                        reason="No moderation results returned",
                        action="pass",
                        latency_ms=elapsed_ms,
                    )
                ]

            # Process the first result (single input)
            mod_result = response.results[0]

            # Check if any category was flagged
            if not mod_result.flagged:
                return [
                    GuardrailResult(
                        passed=True,
                        category="safe",
                        score=0.0,
                        model=self.model_name,
                        reason="Content passed all moderation checks",
                        action="pass",
                        latency_ms=elapsed_ms,
                    )
                ]

            # Extract flagged categories and scores
            categories = mod_result.categories
            scores = mod_result.category_scores

            # Convert to dict for easier access
            categories_dict = categories.model_dump() if hasattr(categories, 'model_dump') else dict(categories)
            scores_dict = scores.model_dump() if hasattr(scores, 'model_dump') else dict(scores)

            # Create results for each flagged category
            for openai_category, is_flagged in categories_dict.items():
                if is_flagged:
                    # Map to our category name
                    our_category = OPENAI_CATEGORY_MAP.get(
                        openai_category, openai_category.replace("/", "_").replace("-", "_")
                    )

                    # Get the score (handle nested category names)
                    score_key = openai_category.replace("/", "_").replace("-", "_")
                    score = scores_dict.get(score_key, scores_dict.get(openai_category, 0.5))

                    results.append(
                        GuardrailResult(
                            passed=False,
                            category=our_category,
                            score=float(score),
                            model=self.model_name,
                            reason=f"Content flagged for {openai_category}",
                            action="block",
                            latency_ms=elapsed_ms,
                        )
                    )

            # If flagged but no specific categories (shouldn't happen), add generic
            if not results:
                results.append(
                    GuardrailResult(
                        passed=False,
                        category="harmful_content",
                        score=1.0,
                        model=self.model_name,
                        reason="Content flagged by OpenAI moderation",
                        action="block",
                        latency_ms=elapsed_ms,
                    )
                )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            results.append(
                GuardrailResult(
                    passed=False,
                    category="error",
                    score=0.0,
                    model=self.model_name,
                    reason=f"OpenAI API error: {str(e)}",
                    action="block",
                    latency_ms=elapsed_ms,
                )
            )

        return results

    async def classify_async(
        self,
        content: str,
        rail_type: RailType,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[GuardrailResult]:
        """
        Async version using OpenAI's async client.

        Args:
            content: Content to classify
            rail_type: Type of rail
            context: Optional context
            metadata: Optional metadata

        Returns:
            List of GuardrailResult objects
        """
        start_time = time.time()

        # Handle empty content
        if not content or not content.strip():
            return [
                GuardrailResult(
                    passed=True,
                    category="empty",
                    score=0.0,
                    model=self.model_name,
                    reason="Empty or whitespace-only content",
                    action="pass",
                    latency_ms=(time.time() - start_time) * 1000,
                )
            ]

        try:
            from openai import AsyncOpenAI
        except ImportError:
            # Fall back to sync implementation
            return await super().classify_async(content, rail_type, context, metadata)

        try:
            async_client = AsyncOpenAI(api_key=self._api_key)

            response = await async_client.moderations.create(
                input=content,
                model=self._moderation_model,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            if not response.results:
                return [
                    GuardrailResult(
                        passed=True,
                        category="safe",
                        score=0.0,
                        model=self.model_name,
                        reason="No moderation results returned",
                        action="pass",
                        latency_ms=elapsed_ms,
                    )
                ]

            mod_result = response.results[0]

            if not mod_result.flagged:
                return [
                    GuardrailResult(
                        passed=True,
                        category="safe",
                        score=0.0,
                        model=self.model_name,
                        reason="Content passed all moderation checks",
                        action="pass",
                        latency_ms=elapsed_ms,
                    )
                ]

            results = []
            categories_dict = mod_result.categories.model_dump() if hasattr(mod_result.categories, 'model_dump') else dict(mod_result.categories)
            scores_dict = mod_result.category_scores.model_dump() if hasattr(mod_result.category_scores, 'model_dump') else dict(mod_result.category_scores)

            for openai_category, is_flagged in categories_dict.items():
                if is_flagged:
                    our_category = OPENAI_CATEGORY_MAP.get(
                        openai_category, openai_category.replace("/", "_").replace("-", "_")
                    )
                    score_key = openai_category.replace("/", "_").replace("-", "_")
                    score = scores_dict.get(score_key, scores_dict.get(openai_category, 0.5))

                    results.append(
                        GuardrailResult(
                            passed=False,
                            category=our_category,
                            score=float(score),
                            model=self.model_name,
                            reason=f"Content flagged for {openai_category}",
                            action="block",
                            latency_ms=elapsed_ms,
                        )
                    )

            if not results:
                results.append(
                    GuardrailResult(
                        passed=False,
                        category="harmful_content",
                        score=1.0,
                        model=self.model_name,
                        reason="Content flagged by OpenAI moderation",
                        action="block",
                        latency_ms=elapsed_ms,
                    )
                )

            return results

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return [
                GuardrailResult(
                    passed=False,
                    category="error",
                    score=0.0,
                    model=self.model_name,
                    reason=f"OpenAI API error: {str(e)}",
                    action="block",
                    latency_ms=elapsed_ms,
                )
            ]
