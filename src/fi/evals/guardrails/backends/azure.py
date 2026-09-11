"""
Azure Content Safety Backend for Guardrails.

Uses Azure AI Content Safety API for content screening.
Supports 4 categories with severity levels 0-7.

Docs: https://learn.microsoft.com/en-us/azure/ai-services/content-safety/
"""

import os
import time
from typing import Any, Dict, List, Optional

from fi.evals.guardrails.backends.base import BaseBackend
from fi.evals.guardrails.config import GuardrailModel, RailType
from fi.evals.guardrails.types import GuardrailResult


# Category mapping from Azure to our standard categories
AZURE_CATEGORY_MAP = {
    "Hate": "hate_speech",
    "SelfHarm": "self_harm",
    "Sexual": "sexual_content",
    "Violence": "violence",
}

# Severity level to score mapping (0-7 to 0-1)
SEVERITY_TO_SCORE = {
    0: 0.0,
    1: 0.14,
    2: 0.29,
    3: 0.43,
    4: 0.57,
    5: 0.71,
    6: 0.86,
    7: 1.0,
}


class AzureBackend(BaseBackend):
    """
    Azure Content Safety API backend.

    Uses Azure AI Content Safety for content moderation.
    Supports 4 categories: Hate, SelfHarm, Sexual, Violence.
    Each category has severity levels from 0 (safe) to 7 (severe).

    Usage:
        backend = AzureBackend(
            model=GuardrailModel.AZURE_CONTENT_SAFETY,
            endpoint="https://your-resource.cognitiveservices.azure.com/",
            api_key="your-key"  # or set AZURE_CONTENT_SAFETY_KEY env var
        )
        results = backend.classify("some content", RailType.INPUT)

    Environment Variables:
        AZURE_CONTENT_SAFETY_ENDPOINT: Azure endpoint URL
        AZURE_CONTENT_SAFETY_KEY: Azure API key
    """

    def __init__(
        self,
        model: GuardrailModel,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        severity_threshold: int = 2,
    ):
        """
        Initialize Azure Content Safety backend.

        Args:
            model: The guardrail model enum value
            endpoint: Azure Content Safety endpoint URL
            api_key: Azure API key
            severity_threshold: Minimum severity level to flag (0-7, default: 2)
        """
        super().__init__(model)

        self._endpoint = endpoint or os.environ.get("AZURE_CONTENT_SAFETY_ENDPOINT")
        self._api_key = api_key or os.environ.get("AZURE_CONTENT_SAFETY_KEY")

        if not self._endpoint:
            raise ValueError(
                "Azure endpoint required. Set AZURE_CONTENT_SAFETY_ENDPOINT environment variable "
                "or pass endpoint parameter."
            )

        if not self._api_key:
            raise ValueError(
                "Azure API key required. Set AZURE_CONTENT_SAFETY_KEY environment variable "
                "or pass api_key parameter."
            )

        self._severity_threshold = severity_threshold
        self._client = None

    def _get_client(self):
        """Lazy-load the Azure Content Safety client."""
        if self._client is None:
            try:
                from azure.ai.contentsafety import ContentSafetyClient
                from azure.core.credentials import AzureKeyCredential
            except ImportError:
                raise ImportError(
                    "azure-ai-contentsafety package required for Azure backend. "
                    "Install with: pip install azure-ai-contentsafety"
                )

            self._client = ContentSafetyClient(
                endpoint=self._endpoint,
                credential=AzureKeyCredential(self._api_key),
            )
        return self._client

    def classify(
        self,
        content: str,
        rail_type: RailType,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[GuardrailResult]:
        """
        Classify content using Azure Content Safety API.

        Args:
            content: Content to classify
            rail_type: Type of rail (input, output, retrieval)
            context: Optional context (not used by Azure)
            metadata: Optional metadata (not used by Azure)

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
            from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory

            client = self._get_client()

            # Truncate content if too long (Azure limit is 10,000 characters)
            truncated_content = content[:10000] if len(content) > 10000 else content

            # Create the analysis request
            request = AnalyzeTextOptions(
                text=truncated_content,
                categories=[
                    TextCategory.HATE,
                    TextCategory.SELF_HARM,
                    TextCategory.SEXUAL,
                    TextCategory.VIOLENCE,
                ],
            )

            # Call Azure Content Safety API
            response = client.analyze_text(request)

            elapsed_ms = (time.time() - start_time) * 1000

            # Check if any category exceeds threshold
            flagged = False
            for category_result in response.categories_analysis:
                category_name = category_result.category
                severity = category_result.severity

                if severity >= self._severity_threshold:
                    flagged = True
                    our_category = AZURE_CATEGORY_MAP.get(
                        category_name, category_name.lower()
                    )
                    score = SEVERITY_TO_SCORE.get(severity, severity / 7.0)

                    results.append(
                        GuardrailResult(
                            passed=False,
                            category=our_category,
                            score=score,
                            model=self.model_name,
                            reason=f"Content flagged for {category_name} (severity: {severity})",
                            action="block",
                            latency_ms=elapsed_ms,
                        )
                    )

            if not flagged:
                return [
                    GuardrailResult(
                        passed=True,
                        category="safe",
                        score=0.0,
                        model=self.model_name,
                        reason="Content passed all Azure safety checks",
                        action="pass",
                        latency_ms=elapsed_ms,
                    )
                ]

        except ImportError as e:
            elapsed_ms = (time.time() - start_time) * 1000
            results.append(
                GuardrailResult(
                    passed=False,
                    category="error",
                    score=0.0,
                    model=self.model_name,
                    reason=str(e),
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
                    reason=f"Azure API error: {str(e)}",
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
        Async version using Azure's async client.

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
            from azure.ai.contentsafety.aio import ContentSafetyClient as AsyncContentSafetyClient
            from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory
            from azure.core.credentials import AzureKeyCredential
        except ImportError:
            # Fall back to sync implementation
            return await super().classify_async(content, rail_type, context, metadata)

        try:
            async with AsyncContentSafetyClient(
                endpoint=self._endpoint,
                credential=AzureKeyCredential(self._api_key),
            ) as async_client:

                truncated_content = content[:10000] if len(content) > 10000 else content

                request = AnalyzeTextOptions(
                    text=truncated_content,
                    categories=[
                        TextCategory.HATE,
                        TextCategory.SELF_HARM,
                        TextCategory.SEXUAL,
                        TextCategory.VIOLENCE,
                    ],
                )

                response = await async_client.analyze_text(request)

                elapsed_ms = (time.time() - start_time) * 1000

                results = []
                flagged = False

                for category_result in response.categories_analysis:
                    category_name = category_result.category
                    severity = category_result.severity

                    if severity >= self._severity_threshold:
                        flagged = True
                        our_category = AZURE_CATEGORY_MAP.get(
                            category_name, category_name.lower()
                        )
                        score = SEVERITY_TO_SCORE.get(severity, severity / 7.0)

                        results.append(
                            GuardrailResult(
                                passed=False,
                                category=our_category,
                                score=score,
                                model=self.model_name,
                                reason=f"Content flagged for {category_name} (severity: {severity})",
                                action="block",
                                latency_ms=elapsed_ms,
                            )
                        )

                if not flagged:
                    return [
                        GuardrailResult(
                            passed=True,
                            category="safe",
                            score=0.0,
                            model=self.model_name,
                            reason="Content passed all Azure safety checks",
                            action="pass",
                            latency_ms=elapsed_ms,
                        )
                    ]

                return results

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return [
                GuardrailResult(
                    passed=False,
                    category="error",
                    score=0.0,
                    model=self.model_name,
                    reason=f"Azure API error: {str(e)}",
                    action="block",
                    latency_ms=elapsed_ms,
                )
            ]
