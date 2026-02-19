"""Local LLM integration for running LLM-as-judge evaluations without cloud API.

This module provides support for local LLM inference via Ollama, enabling
offline LLM-based evaluations for air-gapped environments or faster iteration.

Example:
    >>> from fi.evals.local.llm import OllamaLLM, LocalLLMConfig
    >>>
    >>> # Initialize with default model
    >>> llm = OllamaLLM()
    >>>
    >>> # Generate completion
    >>> response = llm.generate("What is 2+2?")
    >>>
    >>> # Use as LLM judge
    >>> result = llm.judge(
    ...     query="What is the capital of France?",
    ...     response="The capital of France is Paris.",
    ...     criteria="Evaluate if the response correctly answers the question."
    ... )
    >>> print(result["score"])
    0.9
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import json
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class LocalLLMConfig:
    """Configuration for local LLM.

    Attributes:
        model: The model name to use (e.g., "llama3.2", "mistral", "phi3").
        base_url: The Ollama API base URL.
        temperature: Sampling temperature (0.0 for deterministic).
        max_tokens: Maximum tokens to generate.
        timeout: Request timeout in seconds.
    """

    model: str = "llama3.2"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: int = 120


class OllamaLLM:
    """Interface to Ollama for local LLM inference.

    This class provides methods for text generation and LLM-as-judge
    evaluations using locally running Ollama models.

    Example:
        >>> llm = OllamaLLM()
        >>> llm.is_available()
        True
        >>> response = llm.generate("Hello, how are you?")
        "I'm doing well, thank you for asking!"
    """

    def __init__(
        self,
        config: Optional[LocalLLMConfig] = None,
        auto_check: bool = True,
    ) -> None:
        """Initialize the Ollama LLM interface.

        Args:
            config: Configuration for the LLM.
            auto_check: If True, check Ollama availability on init.
        """
        self.config = config or LocalLLMConfig()
        self._available: Optional[bool] = None
        self._models: Optional[List[str]] = None

        if auto_check:
            self._check_availability()

    def _check_availability(self) -> None:
        """Check if Ollama is available and the model is installed."""
        try:
            import requests

            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()

            data = response.json()
            self._models = [m.get("name", "") for m in data.get("models", [])]
            self._available = True

            # Check if requested model is available
            model_base = self.config.model.split(":")[0]
            if not any(model_base in m for m in self._models):
                logger.warning(
                    f"Model '{self.config.model}' not found in Ollama. "
                    f"Available models: {self._models}"
                )

        except Exception as e:
            self._available = False
            logger.warning(f"Ollama not available: {e}")

    def is_available(self) -> bool:
        """Check if Ollama is available.

        Returns:
            True if Ollama is running and accessible.
        """
        if self._available is None:
            self._check_availability()
        return self._available or False

    def list_models(self) -> List[str]:
        """List available models in Ollama.

        Returns:
            List of installed model names.
        """
        if self._models is None:
            self._check_availability()
        return self._models or []

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a completion using Ollama.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.
            temperature: Override temperature for this request.
            max_tokens: Override max tokens for this request.

        Returns:
            The generated text response.

        Raises:
            ConnectionError: If Ollama is not available.
            RuntimeError: If generation fails.
        """
        if not self.is_available():
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.config.base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )

        import requests

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.config.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.config.max_tokens,
            }
        }

        if system:
            payload["system"] = system

        try:
            response = requests.post(
                f"{self.config.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()

            return response.json().get("response", "")

        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama request timed out after {self.config.timeout}s"
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Chat completion using Ollama's chat API.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: Override temperature for this request.
            max_tokens: Override max tokens for this request.

        Returns:
            The assistant's response.
        """
        if not self.is_available():
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.config.base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )

        import requests

        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.config.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.config.max_tokens,
            }
        }

        try:
            response = requests.post(
                f"{self.config.base_url}/api/chat",
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()

            return response.json().get("message", {}).get("content", "")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama chat request failed: {e}")

    def judge(
        self,
        query: str,
        response: str,
        criteria: str,
        context: Optional[str] = None,
        output_format: str = "json",
    ) -> Dict[str, Any]:
        """Use LLM as judge for evaluation.

        Args:
            query: The original query/question.
            response: The response to evaluate.
            criteria: The evaluation criteria/rubric.
            context: Optional context/reference information.
            output_format: Output format ("json" or "text").

        Returns:
            Dictionary with evaluation result:
                - score: Float between 0 and 1
                - passed: Boolean indicating if evaluation passed
                - reason: Explanation of the evaluation
        """
        system_prompt = """You are an AI evaluation judge. Evaluate the response based on the given criteria.

Your evaluation must be fair, consistent, and based solely on the criteria provided.

Output your evaluation as JSON with the following format:
{
    "score": <float between 0.0 and 1.0>,
    "passed": <true or false, based on whether score >= 0.5>,
    "reason": "<brief explanation of your evaluation>"
}

Only output the JSON object, nothing else."""

        user_prompt = f"""## Evaluation Criteria
{criteria}

## Query
{query}

## Response to Evaluate
{response}
"""

        if context:
            user_prompt += f"""
## Context/Reference
{context}
"""

        user_prompt += """
## Your Evaluation (JSON only)"""

        try:
            result_text = self.generate(user_prompt, system=system_prompt)
            return self._parse_judge_response(result_text)

        except Exception as e:
            logger.error(f"LLM judge failed: {e}")
            return {
                "score": 0.0,
                "passed": False,
                "reason": f"Evaluation failed: {str(e)}",
                "error": True,
            }

    def _parse_judge_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM judge response into structured format.

        Args:
            response: The raw LLM response text.

        Returns:
            Parsed evaluation result dictionary.
        """
        response = response.strip()

        # Try direct JSON parse first
        try:
            result = json.loads(response)
            return self._validate_judge_result(result)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        match = re.search(code_block_pattern, response)
        if match:
            try:
                result = json.loads(match.group(1).strip())
                return self._validate_judge_result(result)
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in response
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                return self._validate_judge_result(result)
            except json.JSONDecodeError:
                pass

        # Fallback: try to extract score and reason from text
        score = 0.5
        score_pattern = r"(?:score|rating)[:\s]*([0-9]*\.?[0-9]+)"
        score_match = re.search(score_pattern, response, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                # Normalize if score is > 1 (e.g., 1-10 scale)
                if score > 1:
                    score = score / 10
                score = max(0.0, min(1.0, score))
            except ValueError:
                pass

        return {
            "score": score,
            "passed": score >= 0.5,
            "reason": response[:500] if response else "Unable to parse evaluation",
            "parse_error": True,
        }

    def _validate_judge_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize the judge result.

        Args:
            result: Raw parsed result dictionary.

        Returns:
            Validated and normalized result.
        """
        score = result.get("score", 0.5)

        # Handle various score formats
        if isinstance(score, str):
            try:
                score = float(score)
            except ValueError:
                score = 0.5

        # Normalize score to 0-1 range
        if score > 1:
            score = score / 10 if score <= 10 else score / 100
        score = max(0.0, min(1.0, float(score)))

        passed = result.get("passed")
        if passed is None:
            passed = score >= 0.5
        elif isinstance(passed, str):
            passed = passed.lower() in ("true", "yes", "1", "pass")

        reason = result.get("reason", result.get("explanation", ""))
        if not isinstance(reason, str):
            reason = str(reason)

        return {
            "score": score,
            "passed": bool(passed),
            "reason": reason,
        }

    def batch_judge(
        self,
        evaluations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run multiple judge evaluations.

        Args:
            evaluations: List of evaluation specifications, each containing:
                - query: The query
                - response: The response to evaluate
                - criteria: Evaluation criteria
                - context: Optional context

        Returns:
            List of evaluation results.
        """
        results = []
        for eval_spec in evaluations:
            result = self.judge(
                query=eval_spec.get("query", ""),
                response=eval_spec.get("response", ""),
                criteria=eval_spec.get("criteria", ""),
                context=eval_spec.get("context"),
            )
            results.append(result)
        return results


class LocalLLMFactory:
    """Factory for creating local LLM instances.

    This factory provides a unified interface for creating different
    types of local LLM backends.
    """

    _backends = {
        "ollama": OllamaLLM,
    }

    @classmethod
    def create(
        cls,
        backend: str = "ollama",
        **kwargs,
    ) -> OllamaLLM:
        """Create a local LLM instance.

        Args:
            backend: The LLM backend to use (currently only "ollama").
            **kwargs: Additional arguments passed to the LLM constructor.

        Returns:
            An initialized LLM instance.

        Raises:
            ValueError: If backend is not supported.
        """
        if backend not in cls._backends:
            raise ValueError(
                f"Unsupported LLM backend: {backend}. "
                f"Supported backends: {list(cls._backends.keys())}"
            )

        return cls._backends[backend](**kwargs)

    @classmethod
    def from_string(cls, spec: str) -> OllamaLLM:
        """Create a local LLM from a string specification.

        Args:
            spec: String specification in format "backend/model"
                  (e.g., "ollama/llama3.2").

        Returns:
            An initialized LLM instance.
        """
        parts = spec.split("/", 1)
        backend = parts[0].lower()
        model = parts[1] if len(parts) > 1 else None

        config = LocalLLMConfig()
        if model:
            config.model = model

        return cls.create(backend, config=config)
