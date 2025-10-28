from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Generic, TypeVar
from pydantic import BaseModel, ValidationError
import re
import json

from .base_metric import BaseMetric
from .heuristics.json_metrics import ContainsJson
from ..types import TextMetricInput
from .llm_as_judges.types import BaseLLMJudgeInput
from ..llm.base_llm_provider import LLMProvider
from ..llm.providers.litellm import LiteLLMProvider

LLMJudgeInputType = TypeVar("LLMJudgeInputType", bound=BaseLLMJudgeInput)


class BaseLLMJudgeMetric(BaseMetric[LLMJudgeInputType], ABC):
    """
    The definitive, Pydantic-driven base class for LLM-as-a-judge metrics.
    It orchestrates prompt creation, calling a provider for a structured response,
    and robustly parsing the output into a validated Pydantic model.
    """

    def __init__(
        self,
        provider: LLMProvider,
        config: Optional[Dict[str, Any]] = None,
        **litellm_kwargs,
    ):
        super().__init__(config)
        self.provider = provider
        self.model = self.config.get("model", "gpt-4o")
        self.provider_kwargs = {**litellm_kwargs}

    @property
    @abstractmethod
    def output_pydantic_model(self) -> Type[BaseModel]:
        """The Pydantic model defining the structure of the judge's output."""
        pass

    @abstractmethod
    def _create_prompt_messages(
        self, inputs: LLMJudgeInputType
    ) -> List[Dict[str, str]]:
        """Creates the full prompt message list for the API call."""
        raise NotImplementedError

    @abstractmethod
    def _normalize_score(self, parsed_output: BaseModel) -> Dict[str, Any]:
        """
        Takes the validated Pydantic output object and converts it into the
        final metric result (a dict with 'output' and 'reason' keys).
        """
        raise NotImplementedError

    def _parse_response_with_fallback(self, response_text: str) -> BaseModel:
        """
        Attempts to parse the LLM's response string into the target Pydantic model.
        It first tries a direct parse, and if that fails, it uses a regex-based
        fallback to extract a JSON object and tries again.
        """
        try:
            return self.output_pydantic_model.model_validate_json(response_text)
        except (ValidationError, json.JSONDecodeError) as e:
            # Fallback attempt: extract JSON from messy text and retry
            # use our metric to check if it contains json
            if (
                ContainsJson().compute_one(
                    inputs=TextMetricInput(response=response_text)
                )["output"]
                == 1.0
            ):
                match = re.search(r"\{.*\}|\[.*\]", response_text, re.DOTALL)
                cleaned_json_str = match.group(0)
                try:
                    return self.output_pydantic_model.model_validate_json(
                        cleaned_json_str
                    )
                except (ValidationError, json.JSONDecodeError) as final_e:
                    raise ValueError(
                        f"Failed to validate the extracted JSON."
                    ) from final_e
            else:
                raise ValueError(f"Failed to find JSON in the response.")

    def compute_one(self, inputs: LLMJudgeInputType) -> Dict[str, Any]:
        messages = self._create_prompt_messages(inputs)

        response_format = self.output_pydantic_model
        # response schema check
        if isinstance(self.provider, LiteLLMProvider):
            if not self.provider._supports_schema(self.model):
                response_format = {"type": "json_object"}
        try:
            response_text = self.provider.get_completion(
                model=self.model,
                messages=messages,
                response_format=response_format,
                **self.provider_kwargs,
            )
        except Exception as e:
            return {"output": 0.0, "reason": f"LLM provider failed: {e}"}

        try:
            parsed_output = self._parse_response_with_fallback(response_text)
            return self._normalize_score(parsed_output)
        except Exception as e:
            return {
                "output": 0.0,
                "reason": f"Failed to parse or validate LLM response. Raw response: '{response_text}'. Error: {e}",
            }
