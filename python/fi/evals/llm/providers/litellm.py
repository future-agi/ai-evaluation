from typing import Any, Dict, List, Optional, Type
import litellm
from pydantic import BaseModel
from ..base_llm_provider import LLMProvider
import openai


class LiteLLMProvider(LLMProvider):
    """The default provider, using the litellm library to connect to any API."""

    def __init__(self, credentials: Optional[Dict[str, Any]] = None):
        """
        Initializes the LiteLLMProvider.

        Args:
            credentials (Optional[Dict[str, Any]]): A dictionary containing authentication
                details that map directly to litellm.completion() arguments.

                Examples:
                - For OpenAI: `{"api_key": "sk-..."}`
                - For Azure: `{"api_key": "...", "api_base": "...", "api_version": "..."}`
                - For other providers, see LiteLLM documentation.

                If not provided, LiteLLM will fall back to its default behavior
                (i.e., checking for environment variables like OPENAI_API_KEY).
        """
        self.credentials = credentials or {}
        self.schema_support_cache: Dict[str, bool] = {}

    def _supports_schema(self, model: str) -> bool:
        """Checks if a model supports response_format with caching."""
        if model not in self.schema_support_cache:
            try:
                self.schema_support_cache[model] = litellm.supports_response_schema(
                    model
                )
            except Exception:
                self.schema_support_cache[model] = False
        return self.schema_support_cache[model]

    def get_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Optional[Type[BaseModel] | Dict[str, str]] = None,
        **kwargs: Any,
    ):
        completion_args = {**self.credentials, **kwargs}

        if response_format and self._supports_schema(model):
            completion_args["response_format"] = response_format

        try:
            # drop unknown/unsupported params in kwargs
            litellm.drop_params = True
            response = litellm.completion(
                model=model, messages=messages, **completion_args
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Received null content from the LLM API.")
            return content

        except openai.APIError as openai_error:
            # Wrap litellm exceptions in a standard error
            raise RuntimeError(
                f"LiteLLM provider failed: {openai_error}"
            ) from openai_error
        except Exception as e:
            raise RuntimeError(f"Unknown error occured in LiteLLM :{e}")
