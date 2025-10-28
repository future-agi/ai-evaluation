from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type

from pydantic import BaseModel


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    This defines the standard interface that all LLM inference backends
    must implement to be compatible with the LLM-as-a-judge framework.
    """

    @abstractmethod
    def get_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Optional[Type[BaseModel] | Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generates a text completion from a list of messages.

        Args:
            model (str): The name or identifier of the model to use.
            messages (List[Dict[str, str]]): The chat messages, following the OpenAI format.
            **kwargs: Provider-specific arguments like temperature, max_tokens, etc.

        Returns:
            str: The content of the generated message.
        """
        raise NotImplementedError
