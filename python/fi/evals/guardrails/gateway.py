"""
Guardrails Gateway - High-level API for content screening.

Provides a simple, ergonomic interface for screening content
with automatic backend management and context managers.
"""

from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, Iterator, List, Optional, Union

from fi.evals.guardrails.base import Guardrails
from fi.evals.guardrails.config import GuardrailModel, GuardrailsConfig, AggregationStrategy
from fi.evals.guardrails.types import GuardrailsResponse
from fi.evals.guardrails.discovery import BackendDiscovery, discover_backends, get_backend_details


class ScreeningSession:
    """A screening session for synchronous operations."""

    def __init__(self, guardrails: Guardrails):
        self._guardrails = guardrails
        self._history: List[GuardrailsResponse] = []

    def input(self, content: str, metadata: Optional[dict] = None) -> GuardrailsResponse:
        """Screen user input.

        Args:
            content: The user input to screen.
            metadata: Optional metadata for the request.

        Returns:
            GuardrailsResponse with screening results.
        """
        result = self._guardrails.screen_input(content, metadata=metadata)
        self._history.append(result)
        return result

    def output(
        self,
        content: str,
        context: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> GuardrailsResponse:
        """Screen LLM output.

        Args:
            content: The LLM response to screen.
            context: Optional original user query for context.
            metadata: Optional metadata for the request.

        Returns:
            GuardrailsResponse with screening results.
        """
        result = self._guardrails.screen_output(content, context=context, metadata=metadata)
        self._history.append(result)
        return result

    def retrieval(
        self,
        chunks: List[str],
        query: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> List[GuardrailsResponse]:
        """Screen retrieval chunks.

        Args:
            chunks: List of document chunks to screen.
            query: Optional user query for context.
            metadata: Optional metadata for the request.

        Returns:
            List of GuardrailsResponse, one per chunk.
        """
        results = self._guardrails.screen_retrieval(chunks, query=query, metadata=metadata)
        self._history.extend(results)
        return results

    @property
    def history(self) -> List[GuardrailsResponse]:
        """Get the history of all screening results in this session."""
        return self._history.copy()

    @property
    def all_passed(self) -> bool:
        """Check if all screenings in this session passed."""
        return all(r.passed for r in self._history)


class AsyncScreeningSession:
    """A screening session for async operations."""

    def __init__(self, guardrails: Guardrails):
        self._guardrails = guardrails
        self._history: List[GuardrailsResponse] = []

    async def input(self, content: str, metadata: Optional[dict] = None) -> GuardrailsResponse:
        """Screen user input asynchronously.

        Args:
            content: The user input to screen.
            metadata: Optional metadata for the request.

        Returns:
            GuardrailsResponse with screening results.
        """
        result = await self._guardrails.screen_input_async(content, metadata=metadata)
        self._history.append(result)
        return result

    async def output(
        self,
        content: str,
        context: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> GuardrailsResponse:
        """Screen LLM output asynchronously.

        Args:
            content: The LLM response to screen.
            context: Optional original user query for context.
            metadata: Optional metadata for the request.

        Returns:
            GuardrailsResponse with screening results.
        """
        result = await self._guardrails.screen_output_async(content, context=context, metadata=metadata)
        self._history.append(result)
        return result

    async def retrieval(
        self,
        chunks: List[str],
        query: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> List[GuardrailsResponse]:
        """Screen retrieval chunks asynchronously.

        Args:
            chunks: List of document chunks to screen.
            query: Optional user query for context.
            metadata: Optional metadata for the request.

        Returns:
            List of GuardrailsResponse, one per chunk.
        """
        results = await self._guardrails.screen_retrieval_async(chunks, query=query, metadata=metadata)
        self._history.extend(results)
        return results

    async def batch(
        self,
        contents: List[str],
        metadata: Optional[dict] = None,
    ) -> List[GuardrailsResponse]:
        """Screen multiple contents in batch asynchronously.

        Args:
            contents: List of contents to screen.
            metadata: Optional metadata for the request.

        Returns:
            List of GuardrailsResponse, one per content.
        """
        results = await self._guardrails.screen_batch_async(contents, metadata=metadata)
        self._history.extend(results)
        return results

    @property
    def history(self) -> List[GuardrailsResponse]:
        """Get the history of all screening results in this session."""
        return self._history.copy()

    @property
    def all_passed(self) -> bool:
        """Check if all screenings in this session passed."""
        return all(r.passed for r in self._history)


class GuardrailsGateway:
    """High-level gateway for content screening.

    Provides convenient methods for creating screening sessions
    and managing guardrails configuration.

    Example:
        # Simple usage
        gateway = GuardrailsGateway()
        result = gateway.screen("Hello world")

        # With context manager
        with gateway.screening() as session:
            input_result = session.input("user message")
            if input_result.passed:
                response = call_llm("user message")
                output_result = session.output(response)

        # Async context manager
        async with gateway.screening_async() as session:
            input_result = await session.input("user message")
            output_result = await session.output(response)
    """

    def __init__(
        self,
        models: Optional[List[GuardrailModel]] = None,
        config: Optional[GuardrailsConfig] = None,
        auto_discover: bool = False,
    ):
        """Initialize the gateway.

        Args:
            models: List of models to use. If None, uses default.
            config: Full configuration object. Takes precedence over models.
            auto_discover: If True and no models specified, auto-discover available backends.
        """
        if config:
            self._config = config
        elif models:
            self._config = GuardrailsConfig(models=models)
        elif auto_discover:
            available = discover_backends()
            if not available:
                raise ValueError("No backends available. Set API keys or start VLLM server.")
            self._config = GuardrailsConfig(models=available[:1])  # Use first available
        else:
            self._config = GuardrailsConfig()

        self._guardrails = Guardrails(config=self._config)

    @classmethod
    def with_openai(cls, api_key: Optional[str] = None) -> "GuardrailsGateway":
        """Create a gateway using OpenAI Moderation (FREE).

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.

        Returns:
            GuardrailsGateway configured for OpenAI.
        """
        import os
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        config = GuardrailsConfig(
            models=[GuardrailModel.OPENAI_MODERATION],
            timeout_ms=30000,
        )
        return cls(config=config)

    @classmethod
    def with_azure(
        cls,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> "GuardrailsGateway":
        """Create a gateway using Azure Content Safety.

        Args:
            endpoint: Azure endpoint URL. If None, uses env var.
            api_key: Azure API key. If None, uses env var.

        Returns:
            GuardrailsGateway configured for Azure.
        """
        import os
        if endpoint:
            os.environ["AZURE_CONTENT_SAFETY_ENDPOINT"] = endpoint
        if api_key:
            os.environ["AZURE_CONTENT_SAFETY_KEY"] = api_key

        config = GuardrailsConfig(
            models=[GuardrailModel.AZURE_CONTENT_SAFETY],
            timeout_ms=30000,
        )
        return cls(config=config)

    @classmethod
    def with_local_model(
        cls,
        model: GuardrailModel,
        vllm_url: Optional[str] = None,
    ) -> "GuardrailsGateway":
        """Create a gateway using a local model via VLLM.

        Args:
            model: The local model to use (e.g., GuardrailModel.WILDGUARD_7B).
            vllm_url: VLLM server URL. If None, uses env var.

        Returns:
            GuardrailsGateway configured for the local model.
        """
        import os
        if vllm_url:
            os.environ["VLLM_SERVER_URL"] = vllm_url

        config = GuardrailsConfig(
            models=[model],
            timeout_ms=60000,  # Local models may be slower
        )
        return cls(config=config)

    @classmethod
    def with_ensemble(
        cls,
        models: List[GuardrailModel],
        aggregation: AggregationStrategy = AggregationStrategy.ANY,
        parallel: bool = True,
    ) -> "GuardrailsGateway":
        """Create a gateway with ensemble of multiple backends.

        Args:
            models: List of models to use.
            aggregation: How to combine results (ANY, ALL, MAJORITY).
            parallel: Whether to run backends in parallel.

        Returns:
            GuardrailsGateway configured for ensemble mode.
        """
        config = GuardrailsConfig(
            models=models,
            aggregation=aggregation,
            parallel=parallel,
            timeout_ms=60000,
        )
        return cls(config=config)

    @classmethod
    def auto(cls) -> "GuardrailsGateway":
        """Create a gateway that auto-discovers available backends.

        Returns:
            GuardrailsGateway with the best available backend.

        Raises:
            ValueError: If no backends are available.
        """
        return cls(auto_discover=True)

    def screen(self, content: str, metadata: Optional[dict] = None) -> GuardrailsResponse:
        """Quick screen content (alias for screen_input).

        Args:
            content: Content to screen.
            metadata: Optional metadata.

        Returns:
            GuardrailsResponse with results.
        """
        return self._guardrails.screen_input(content, metadata=metadata)

    async def screen_async(self, content: str, metadata: Optional[dict] = None) -> GuardrailsResponse:
        """Quick screen content asynchronously.

        Args:
            content: Content to screen.
            metadata: Optional metadata.

        Returns:
            GuardrailsResponse with results.
        """
        return await self._guardrails.screen_input_async(content, metadata=metadata)

    @contextmanager
    def screening(self) -> Iterator[ScreeningSession]:
        """Create a synchronous screening session.

        Yields:
            ScreeningSession for screening operations.

        Example:
            with gateway.screening() as session:
                input_result = session.input("user message")
                if input_result.passed:
                    response = generate_response()
                    output_result = session.output(response)
        """
        session = ScreeningSession(self._guardrails)
        yield session

    @asynccontextmanager
    async def screening_async(self) -> AsyncIterator[AsyncScreeningSession]:
        """Create an async screening session.

        Yields:
            AsyncScreeningSession for async screening operations.

        Example:
            async with gateway.screening_async() as session:
                input_result = await session.input("user message")
                if input_result.passed:
                    response = await generate_response()
                    output_result = await session.output(response)
        """
        session = AsyncScreeningSession(self._guardrails)
        yield session

    @property
    def available_backends(self) -> List[GuardrailModel]:
        """Get list of available backends."""
        return discover_backends()

    @property
    def configured_models(self) -> List[GuardrailModel]:
        """Get list of configured models."""
        return self._config.models

    @staticmethod
    def discover() -> List[GuardrailModel]:
        """Discover available backends.

        Returns:
            List of available GuardrailModel values.
        """
        return discover_backends()

    @staticmethod
    def get_details() -> dict:
        """Get detailed backend availability info.

        Returns:
            Dict mapping model names to availability details.
        """
        return get_backend_details()


# Convenience alias
Gateway = GuardrailsGateway


__all__ = [
    "GuardrailsGateway",
    "Gateway",
    "ScreeningSession",
    "AsyncScreeningSession",
    "discover_backends",
    "get_backend_details",
]
