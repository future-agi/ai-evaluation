"""
Base class for Local Model Backends.

Provides common functionality for running HuggingFace models locally,
either via VLLM server or direct transformers loading.
"""

import os
import time
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from fi.evals.guardrails.backends.base import BaseBackend
from fi.evals.guardrails.backends.vllm_client import VLLMClient, get_vllm_url
from fi.evals.guardrails.config import GuardrailModel, RailType
from fi.evals.guardrails.types import GuardrailResult


class LocalModelBackend(BaseBackend):
    """
    Base class for local HuggingFace model backends.

    Supports two modes:
    1. VLLM Server Mode: Use external VLLM server for inference
    2. Direct Mode: Load model directly using transformers

    Subclasses must implement:
    - _format_prompt(): Model-specific prompt formatting
    - _parse_response(): Model-specific response parsing
    - HF_MODEL_NAME: HuggingFace model identifier

    Usage:
        class WildGuardBackend(LocalModelBackend):
            HF_MODEL_NAME = "allenai/wildguard"

            def _format_prompt(self, content, context=None):
                return f"Human user:\\n{content}\\n..."

            def _parse_response(self, response):
                # Parse "harmful_request: yes/no" format
                ...
    """

    # Subclasses should override these
    HF_MODEL_NAME: str = ""
    MAX_NEW_TOKENS: int = 128
    TEMPERATURE: float = 0.1

    def __init__(
        self,
        model: GuardrailModel,
        vllm_url: Optional[str] = None,
        device: str = "auto",
        hf_token: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize local model backend.

        Args:
            model: The guardrail model enum value
            vllm_url: VLLM server URL (auto-detected from env if not provided)
            device: Device to use ("auto", "cuda", "mps", "cpu")
            hf_token: HuggingFace token for gated models
            load_in_8bit: Use 8-bit quantization (reduces memory)
            load_in_4bit: Use 4-bit quantization (reduces memory further)
        """
        super().__init__(model)

        self._vllm_url = vllm_url or get_vllm_url(model.value)
        self._device = device
        self._hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit

        # Lazy-loaded resources
        self._vllm_client: Optional[VLLMClient] = None
        self._transformers_model = None
        self._tokenizer = None

        # Check what's available
        self._use_vllm = False
        if self._vllm_url:
            client = VLLMClient(self._vllm_url)
            if client.health_check():
                self._use_vllm = True
                self._vllm_client = client

    def _get_hf_model_name(self) -> str:
        """Get the HuggingFace model name. Override in subclasses if needed."""
        return self.HF_MODEL_NAME

    def _load_transformers_model(self):
        """Load the model using transformers library."""
        if self._transformers_model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers and torch packages required for local model backends. "
                "Install with: pip install torch transformers accelerate"
            )

        model_name = self._get_hf_model_name()

        # Determine device
        if self._device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = self._device

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=self._hf_token,
        )

        # Determine dtype and load kwargs
        if device in ("cuda", "mps"):
            dtype = torch.float16
        else:
            dtype = torch.float32

        load_kwargs = {
            "torch_dtype": dtype,
            "token": self._hf_token,
            "low_cpu_mem_usage": True,
        }

        if self._load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif self._load_in_4bit:
            load_kwargs["load_in_4bit"] = True

        # Load model
        self._transformers_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs,
        )

        if not (self._load_in_8bit or self._load_in_4bit):
            self._transformers_model.to(device)

        self._transformers_model.eval()
        self._transformers_device = device

    def _generate_with_transformers(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate text using local transformers model.

        Args:
            prompt: Input prompt
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text (response only, not including prompt)
        """
        import torch

        self._load_transformers_model()

        max_tokens = max_new_tokens or self.MAX_NEW_TOKENS
        temp = temperature or self.TEMPERATURE

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._transformers_device)
        prompt_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self._transformers_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temp, 0.01),
                do_sample=temp > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (excluding prompt)
        generated_tokens = outputs[0][prompt_length:]
        response = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response

    @abstractmethod
    def _format_prompt(
        self,
        content: str,
        rail_type: RailType,
        context: Optional[str] = None,
    ) -> str:
        """
        Format content into model-specific prompt.

        Args:
            content: Content to classify
            rail_type: Type of rail (input, output, retrieval)
            context: Optional context (e.g., for output rail)

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def _parse_response(
        self,
        response: str,
        content: str,
        rail_type: RailType,
    ) -> List[GuardrailResult]:
        """
        Parse model response into GuardrailResult objects.

        Args:
            response: Raw model response text
            content: Original content that was classified
            rail_type: Type of rail

        Returns:
            List of GuardrailResult objects
        """
        pass

    def classify(
        self,
        content: str,
        rail_type: RailType,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[GuardrailResult]:
        """
        Classify content using local model.

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
            # Format prompt
            prompt = self._format_prompt(content, rail_type, context)

            # Generate response
            if self._use_vllm and self._vllm_client:
                vllm_response = self._vllm_client.generate(
                    prompt=prompt,
                    max_tokens=self.MAX_NEW_TOKENS,
                    temperature=self.TEMPERATURE,
                )
                response_text = vllm_response.text
            else:
                response_text = self._generate_with_transformers(prompt)

            elapsed_ms = (time.time() - start_time) * 1000

            # Parse response
            results = self._parse_response(response_text, content, rail_type)

            # Update latency in results
            for result in results:
                result.latency_ms = elapsed_ms

            return results

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return [
                GuardrailResult(
                    passed=False,
                    category="error",
                    score=0.0,
                    model=self.model_name,
                    reason=f"Local model error: {str(e)}",
                    action="block",
                    latency_ms=elapsed_ms,
                )
            ]

    async def classify_async(
        self,
        content: str,
        rail_type: RailType,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[GuardrailResult]:
        """
        Async version of classify.

        Uses VLLM async client if available, otherwise falls back to sync.

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
            # Format prompt
            prompt = self._format_prompt(content, rail_type, context)

            # Generate response
            if self._use_vllm and self._vllm_client:
                vllm_response = await self._vllm_client.generate_async(
                    prompt=prompt,
                    max_tokens=self.MAX_NEW_TOKENS,
                    temperature=self.TEMPERATURE,
                )
                response_text = vllm_response.text
            else:
                # Fall back to sync for transformers
                import asyncio
                response_text = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._generate_with_transformers(prompt)
                )

            elapsed_ms = (time.time() - start_time) * 1000

            # Parse response
            results = self._parse_response(response_text, content, rail_type)

            # Update latency in results
            for result in results:
                result.latency_ms = elapsed_ms

            return results

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return [
                GuardrailResult(
                    passed=False,
                    category="error",
                    score=0.0,
                    model=self.model_name,
                    reason=f"Local model error: {str(e)}",
                    action="block",
                    latency_ms=elapsed_ms,
                )
            ]

    def is_available(self) -> bool:
        """
        Check if this backend is available.

        Returns:
            True if VLLM server is up or transformers can load the model
        """
        if self._use_vllm:
            return True

        # Check if we can load via transformers
        try:
            import torch
            has_gpu = torch.cuda.is_available() or (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            )
            return has_gpu or self._device == "cpu"
        except ImportError:
            return False
