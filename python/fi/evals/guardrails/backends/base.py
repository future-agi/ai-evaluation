"""
Base Backend Interface for Guardrails.

Defines the abstract interface that all guardrail backends must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import asyncio

from fi.evals.guardrails.config import RailType, GuardrailModel
from fi.evals.guardrails.types import GuardrailResult


class BaseBackend(ABC):
    """
    Abstract base class for guardrail backends.

    All backend implementations must inherit from this class and implement
    the classify and classify_async methods.
    """

    def __init__(self, model: GuardrailModel):
        """
        Initialize the backend.

        Args:
            model: The model this backend represents
        """
        self.model = model

    @abstractmethod
    def classify(
        self,
        content: str,
        rail_type: RailType,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[GuardrailResult]:
        """
        Classify content synchronously.

        Args:
            content: The content to classify
            rail_type: Type of rail (input, output, retrieval)
            context: Optional context for the classification
            metadata: Optional metadata

        Returns:
            List of GuardrailResult objects for each category detected
        """
        pass

    async def classify_async(
        self,
        content: str,
        rail_type: RailType,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[GuardrailResult]:
        """
        Classify content asynchronously.

        Default implementation wraps the sync method.
        Backends can override for true async support.

        Args:
            content: The content to classify
            rail_type: Type of rail (input, output, retrieval)
            context: Optional context for the classification
            metadata: Optional metadata

        Returns:
            List of GuardrailResult objects for each category detected
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.classify(content, rail_type, context, metadata)
        )

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.model.value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model.value})"
