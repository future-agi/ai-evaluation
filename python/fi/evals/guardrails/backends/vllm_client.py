"""
VLLM Client for Guardrails.

Provides a client for OpenAI-compatible VLLM servers.
Used by local model backends to communicate with VLLM instances.
"""

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import asyncio
import httpx


@dataclass
class VLLMResponse:
    """Response from VLLM server."""
    text: str
    model: str
    usage: Dict[str, int]
    latency_ms: float


class VLLMClient:
    """
    Client for OpenAI-compatible VLLM servers.

    Supports both sync and async operations.

    Usage:
        client = VLLMClient(base_url="http://localhost:28000")

        # Check health
        if client.health_check():
            response = client.generate("prompt", max_tokens=128)
            print(response.text)

        # Async usage
        response = await client.generate_async("prompt", max_tokens=128)
    """

    def __init__(
        self,
        base_url: str,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """
        Initialize VLLM client.

        Args:
            base_url: VLLM server base URL (e.g., http://localhost:28000)
            model: Model name (if not specified, uses server default)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def health_check(self) -> bool:
        """
        Check if VLLM server is healthy.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False

    async def health_check_async(self) -> bool:
        """Async version of health_check."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False

    def get_models(self) -> List[str]:
        """
        Get list of available models from VLLM server.

        Returns:
            List of model names
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/v1/models")
                if response.status_code == 200:
                    data = response.json()
                    return [m["id"] for m in data.get("data", [])]
        except Exception:
            pass
        return []

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.1,
        stop: Optional[List[str]] = None,
    ) -> VLLMResponse:
        """
        Generate completion from VLLM server.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences

        Returns:
            VLLMResponse with generated text
        """
        start_time = time.time()

        # Determine model to use
        model = self.model or self._get_default_model()

        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/v1/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        elapsed_ms = (time.time() - start_time) * 1000

        return VLLMResponse(
            text=data["choices"][0]["text"],
            model=data.get("model", model),
            usage=data.get("usage", {}),
            latency_ms=elapsed_ms,
        )

    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.1,
        stop: Optional[List[str]] = None,
    ) -> VLLMResponse:
        """
        Async version of generate.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences

        Returns:
            VLLMResponse with generated text
        """
        start_time = time.time()

        model = self.model or await self._get_default_model_async()

        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        elapsed_ms = (time.time() - start_time) * 1000

        return VLLMResponse(
            text=data["choices"][0]["text"],
            model=data.get("model", model),
            usage=data.get("usage", {}),
            latency_ms=elapsed_ms,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 128,
        temperature: float = 0.1,
    ) -> VLLMResponse:
        """
        Generate chat completion from VLLM server.

        Args:
            messages: List of chat messages [{"role": "user", "content": "..."}]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            VLLMResponse with generated text
        """
        start_time = time.time()

        model = self.model or self._get_default_model()

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        elapsed_ms = (time.time() - start_time) * 1000

        return VLLMResponse(
            text=data["choices"][0]["message"]["content"],
            model=data.get("model", model),
            usage=data.get("usage", {}),
            latency_ms=elapsed_ms,
        )

    async def chat_async(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 128,
        temperature: float = 0.1,
    ) -> VLLMResponse:
        """Async version of chat."""
        start_time = time.time()

        model = self.model or await self._get_default_model_async()

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        elapsed_ms = (time.time() - start_time) * 1000

        return VLLMResponse(
            text=data["choices"][0]["message"]["content"],
            model=data.get("model", model),
            usage=data.get("usage", {}),
            latency_ms=elapsed_ms,
        )

    def _get_default_model(self) -> str:
        """Get default model from server."""
        models = self.get_models()
        return models[0] if models else "default"

    async def _get_default_model_async(self) -> str:
        """Async version of _get_default_model."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/v1/models")
                if response.status_code == 200:
                    data = response.json()
                    models = [m["id"] for m in data.get("data", [])]
                    return models[0] if models else "default"
        except Exception:
            pass
        return "default"


def get_vllm_url(model_name: str) -> Optional[str]:
    """
    Get VLLM server URL for a model from environment variables.

    Checks for model-specific env var first, then falls back to generic.

    Args:
        model_name: Model name (e.g., "wildguard-7b")

    Returns:
        VLLM server URL or None
    """
    # Try model-specific env var (e.g., VLLM_WILDGUARD_7B_URL)
    env_var = f"VLLM_{model_name.upper().replace('-', '_')}_URL"
    url = os.environ.get(env_var)
    if url:
        return url

    # Fall back to generic VLLM_SERVER_URL
    return os.environ.get("VLLM_SERVER_URL")
