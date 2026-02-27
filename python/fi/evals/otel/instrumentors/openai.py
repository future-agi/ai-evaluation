"""
OpenAI Instrumentor.

Automatic instrumentation for the OpenAI Python client library.
Supports both sync and async APIs.
"""

from typing import Any, Dict, Optional, Callable
import functools
import logging
import json

from .base import BaseInstrumentor
from ..processors import OTEL_AVAILABLE
from ..conventions import (
    GenAIAttributes,
    SYSTEM_OPENAI,
    OPERATION_CHAT,
    OPERATION_COMPLETION,
    OPERATION_EMBEDDING,
    FINISH_STOP,
    FINISH_LENGTH,
    FINISH_TOOL_CALLS,
)

if OTEL_AVAILABLE:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)


class OpenAIInstrumentor(BaseInstrumentor):
    """
    Instrumentor for the OpenAI Python client.

    Automatically traces:
    - chat.completions.create
    - completions.create (legacy)
    - embeddings.create

    Example:
        from fi.evals.otel.instrumentors import OpenAIInstrumentor

        # Instrument OpenAI
        instrumentor = OpenAIInstrumentor()
        instrumentor.instrument()

        # All OpenAI calls are now traced
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        # Span automatically created with model, tokens, etc.

        # To disable
        instrumentor.uninstrument()
    """

    system_name = "openai"
    library_name = "openai"

    def __init__(
        self,
        capture_prompts: bool = True,
        capture_completions: bool = True,
        capture_streaming: bool = True,
    ):
        """
        Initialize the OpenAI instrumentor.

        Args:
            capture_prompts: Whether to capture input prompts
            capture_completions: Whether to capture outputs
            capture_streaming: Whether to trace streaming responses
        """
        super().__init__()
        self._capture_prompts = capture_prompts
        self._capture_completions = capture_completions
        self._capture_streaming = capture_streaming

    def instrument(self, **kwargs) -> None:
        """Apply instrumentation to OpenAI client."""
        if self._instrumented:
            return

        try:
            import openai
            from openai.resources.chat import completions as chat_completions
            from openai.resources import embeddings

            # Store original methods
            self._original_methods["chat_completions_create"] = (
                chat_completions.Completions.create
            )
            self._original_methods["async_chat_completions_create"] = (
                chat_completions.AsyncCompletions.create
            )
            self._original_methods["embeddings_create"] = (
                embeddings.Embeddings.create
            )
            self._original_methods["async_embeddings_create"] = (
                embeddings.AsyncEmbeddings.create
            )

            # Patch chat completions
            chat_completions.Completions.create = self._wrap_chat_create(
                self._original_methods["chat_completions_create"],
                is_async=False,
            )
            chat_completions.AsyncCompletions.create = self._wrap_chat_create(
                self._original_methods["async_chat_completions_create"],
                is_async=True,
            )

            # Patch embeddings
            embeddings.Embeddings.create = self._wrap_embeddings_create(
                self._original_methods["embeddings_create"],
                is_async=False,
            )
            embeddings.AsyncEmbeddings.create = self._wrap_embeddings_create(
                self._original_methods["async_embeddings_create"],
                is_async=True,
            )

            # Try to patch legacy completions if available
            try:
                from openai.resources import completions as legacy_completions
                self._original_methods["completions_create"] = (
                    legacy_completions.Completions.create
                )
                legacy_completions.Completions.create = self._wrap_completions_create(
                    self._original_methods["completions_create"],
                    is_async=False,
                )
            except (ImportError, AttributeError):
                pass  # Legacy completions not available

            self._instrumented = True
            logger.info("OpenAI instrumentation applied")

        except ImportError:
            logger.warning("OpenAI library not installed, skipping instrumentation")
        except Exception as e:
            logger.error(f"Failed to instrument OpenAI: {e}")

    def uninstrument(self, **kwargs) -> None:
        """Remove instrumentation from OpenAI client."""
        if not self._instrumented:
            return

        try:
            from openai.resources.chat import completions as chat_completions
            from openai.resources import embeddings

            # Restore original methods
            if "chat_completions_create" in self._original_methods:
                chat_completions.Completions.create = (
                    self._original_methods["chat_completions_create"]
                )
            if "async_chat_completions_create" in self._original_methods:
                chat_completions.AsyncCompletions.create = (
                    self._original_methods["async_chat_completions_create"]
                )
            if "embeddings_create" in self._original_methods:
                embeddings.Embeddings.create = (
                    self._original_methods["embeddings_create"]
                )
            if "async_embeddings_create" in self._original_methods:
                embeddings.AsyncEmbeddings.create = (
                    self._original_methods["async_embeddings_create"]
                )

            try:
                from openai.resources import completions as legacy_completions
                if "completions_create" in self._original_methods:
                    legacy_completions.Completions.create = (
                        self._original_methods["completions_create"]
                    )
            except (ImportError, AttributeError):
                pass

            self._original_methods.clear()
            self._instrumented = False
            logger.info("OpenAI instrumentation removed")

        except Exception as e:
            logger.error(f"Failed to uninstrument OpenAI: {e}")

    def _wrap_chat_create(self, original: Callable, is_async: bool) -> Callable:
        """Wrap chat.completions.create."""
        instrumentor = self

        if is_async:
            @functools.wraps(original)
            async def async_wrapper(self_arg, *args, **kwargs):
                return await instrumentor._trace_chat_completion(
                    original, self_arg, args, kwargs, is_async=True
                )
            return async_wrapper
        else:
            @functools.wraps(original)
            def wrapper(self_arg, *args, **kwargs):
                return instrumentor._trace_chat_completion(
                    original, self_arg, args, kwargs, is_async=False
                )
            return wrapper

    def _trace_chat_completion(
        self,
        original: Callable,
        self_arg: Any,
        args: tuple,
        kwargs: dict,
        is_async: bool,
    ):
        """Trace a chat completion call."""
        tracer = self.get_tracer()

        # Extract parameters
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")

        # Build attributes
        attributes = {
            GenAIAttributes.SYSTEM: SYSTEM_OPENAI,
            GenAIAttributes.OPERATION_NAME: OPERATION_CHAT,
            GenAIAttributes.REQUEST_MODEL: model,
        }

        if temperature is not None:
            attributes[GenAIAttributes.REQUEST_TEMPERATURE] = temperature
        if max_tokens is not None:
            attributes[GenAIAttributes.REQUEST_MAX_TOKENS] = max_tokens

        # Capture prompt
        if self._capture_prompts and messages:
            prompt_text = self._format_messages(messages)
            attributes[GenAIAttributes.prompt_content(0)] = prompt_text[:10000]

        span_name = f"openai.chat.completions.create"

        if is_async:
            return self._trace_async(
                original, self_arg, args, kwargs, span_name, attributes, stream
            )
        else:
            return self._trace_sync(
                original, self_arg, args, kwargs, span_name, attributes, stream
            )

    def _trace_sync(
        self,
        original: Callable,
        self_arg: Any,
        args: tuple,
        kwargs: dict,
        span_name: str,
        attributes: Dict[str, Any],
        stream: bool,
    ):
        """Trace a synchronous call."""
        tracer = self.get_tracer()

        with tracer.start_as_current_span(span_name, attributes=attributes) as span:
            try:
                response = original(self_arg, *args, **kwargs)

                if stream and self._capture_streaming:
                    # Wrap streaming response
                    return self._wrap_stream_response(response, span)

                # Extract response data
                self._extract_response_data(response, span)

                if OTEL_AVAILABLE:
                    span.set_status(Status(StatusCode.OK))

                return response

            except Exception as e:
                if OTEL_AVAILABLE:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                raise

    async def _trace_async(
        self,
        original: Callable,
        self_arg: Any,
        args: tuple,
        kwargs: dict,
        span_name: str,
        attributes: Dict[str, Any],
        stream: bool,
    ):
        """Trace an asynchronous call."""
        tracer = self.get_tracer()

        with tracer.start_as_current_span(span_name, attributes=attributes) as span:
            try:
                response = await original(self_arg, *args, **kwargs)

                if stream and self._capture_streaming:
                    # Wrap async streaming response
                    return self._wrap_async_stream_response(response, span)

                self._extract_response_data(response, span)

                if OTEL_AVAILABLE:
                    span.set_status(Status(StatusCode.OK))

                return response

            except Exception as e:
                if OTEL_AVAILABLE:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                raise

    def _extract_response_data(self, response: Any, span: Any) -> None:
        """Extract data from a chat completion response."""
        try:
            # Model used
            if hasattr(response, "model"):
                span.set_attribute(GenAIAttributes.RESPONSE_MODEL, response.model)

            # Response ID
            if hasattr(response, "id"):
                span.set_attribute(GenAIAttributes.RESPONSE_ID, response.id)

            # Token usage
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                if hasattr(usage, "prompt_tokens"):
                    span.set_attribute(
                        GenAIAttributes.USAGE_INPUT_TOKENS,
                        usage.prompt_tokens
                    )
                if hasattr(usage, "completion_tokens"):
                    span.set_attribute(
                        GenAIAttributes.USAGE_OUTPUT_TOKENS,
                        usage.completion_tokens
                    )
                if hasattr(usage, "total_tokens"):
                    span.set_attribute(
                        GenAIAttributes.USAGE_TOTAL_TOKENS,
                        usage.total_tokens
                    )

            # Completion content
            if self._capture_completions and hasattr(response, "choices"):
                choices = response.choices
                if choices and len(choices) > 0:
                    choice = choices[0]

                    # Finish reason
                    if hasattr(choice, "finish_reason") and choice.finish_reason:
                        span.set_attribute(
                            GenAIAttributes.RESPONSE_FINISH_REASON,
                            choice.finish_reason
                        )

                    # Content
                    if hasattr(choice, "message") and choice.message:
                        msg = choice.message
                        if hasattr(msg, "content") and msg.content:
                            span.set_attribute(
                                GenAIAttributes.completion_content(0),
                                msg.content[:10000]
                            )
                        if hasattr(msg, "role"):
                            span.set_attribute(
                                GenAIAttributes.completion_role(0),
                                msg.role
                            )

        except Exception as e:
            logger.debug(f"Failed to extract response data: {e}")

    def _format_messages(self, messages: list) -> str:
        """Format messages list as string."""
        parts = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                parts.append(f"[{role}]: {content}")
            else:
                parts.append(str(msg))
        return "\n".join(parts)

    def _wrap_stream_response(self, stream: Any, span: Any):
        """Wrap a streaming response to capture data."""
        # Return wrapped iterator that accumulates content
        return _StreamWrapper(stream, span, self._capture_completions)

    def _wrap_async_stream_response(self, stream: Any, span: Any):
        """Wrap an async streaming response."""
        return _AsyncStreamWrapper(stream, span, self._capture_completions)

    def _wrap_embeddings_create(self, original: Callable, is_async: bool) -> Callable:
        """Wrap embeddings.create."""
        instrumentor = self

        if is_async:
            @functools.wraps(original)
            async def async_wrapper(self_arg, *args, **kwargs):
                tracer = instrumentor.get_tracer()
                model = kwargs.get("model", "text-embedding-ada-002")

                attributes = {
                    GenAIAttributes.SYSTEM: SYSTEM_OPENAI,
                    GenAIAttributes.OPERATION_NAME: OPERATION_EMBEDDING,
                    GenAIAttributes.REQUEST_MODEL: model,
                }

                # Capture input text
                input_text = kwargs.get("input", "")
                if instrumentor._capture_prompts and input_text:
                    if isinstance(input_text, list):
                        input_text = " ".join(str(t) for t in input_text[:5])
                    attributes[GenAIAttributes.prompt_content(0)] = str(input_text)[:10000]

                with tracer.start_as_current_span(
                    "openai.embeddings.create",
                    attributes=attributes
                ) as span:
                    try:
                        response = await original(self_arg, *args, **kwargs)

                        # Extract usage
                        if hasattr(response, "usage") and response.usage:
                            span.set_attribute(
                                GenAIAttributes.USAGE_INPUT_TOKENS,
                                response.usage.prompt_tokens
                            )
                            span.set_attribute(
                                GenAIAttributes.USAGE_TOTAL_TOKENS,
                                response.usage.total_tokens
                            )

                        if OTEL_AVAILABLE:
                            span.set_status(Status(StatusCode.OK))

                        return response
                    except Exception as e:
                        if OTEL_AVAILABLE:
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                            span.record_exception(e)
                        raise

            return async_wrapper
        else:
            @functools.wraps(original)
            def wrapper(self_arg, *args, **kwargs):
                tracer = instrumentor.get_tracer()
                model = kwargs.get("model", "text-embedding-ada-002")

                attributes = {
                    GenAIAttributes.SYSTEM: SYSTEM_OPENAI,
                    GenAIAttributes.OPERATION_NAME: OPERATION_EMBEDDING,
                    GenAIAttributes.REQUEST_MODEL: model,
                }

                input_text = kwargs.get("input", "")
                if instrumentor._capture_prompts and input_text:
                    if isinstance(input_text, list):
                        input_text = " ".join(str(t) for t in input_text[:5])
                    attributes[GenAIAttributes.prompt_content(0)] = str(input_text)[:10000]

                with tracer.start_as_current_span(
                    "openai.embeddings.create",
                    attributes=attributes
                ) as span:
                    try:
                        response = original(self_arg, *args, **kwargs)

                        if hasattr(response, "usage") and response.usage:
                            span.set_attribute(
                                GenAIAttributes.USAGE_INPUT_TOKENS,
                                response.usage.prompt_tokens
                            )
                            span.set_attribute(
                                GenAIAttributes.USAGE_TOTAL_TOKENS,
                                response.usage.total_tokens
                            )

                        if OTEL_AVAILABLE:
                            span.set_status(Status(StatusCode.OK))

                        return response
                    except Exception as e:
                        if OTEL_AVAILABLE:
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                            span.record_exception(e)
                        raise

            return wrapper

    def _wrap_completions_create(self, original: Callable, is_async: bool) -> Callable:
        """Wrap legacy completions.create."""
        instrumentor = self

        @functools.wraps(original)
        def wrapper(self_arg, *args, **kwargs):
            tracer = instrumentor.get_tracer()
            model = kwargs.get("model", "unknown")

            attributes = {
                GenAIAttributes.SYSTEM: SYSTEM_OPENAI,
                GenAIAttributes.OPERATION_NAME: OPERATION_COMPLETION,
                GenAIAttributes.REQUEST_MODEL: model,
            }

            prompt = kwargs.get("prompt", "")
            if instrumentor._capture_prompts and prompt:
                attributes[GenAIAttributes.prompt_content(0)] = str(prompt)[:10000]

            with tracer.start_as_current_span(
                "openai.completions.create",
                attributes=attributes
            ) as span:
                try:
                    response = original(self_arg, *args, **kwargs)

                    if hasattr(response, "usage") and response.usage:
                        span.set_attribute(
                            GenAIAttributes.USAGE_INPUT_TOKENS,
                            response.usage.prompt_tokens
                        )
                        span.set_attribute(
                            GenAIAttributes.USAGE_OUTPUT_TOKENS,
                            response.usage.completion_tokens
                        )
                        span.set_attribute(
                            GenAIAttributes.USAGE_TOTAL_TOKENS,
                            response.usage.total_tokens
                        )

                    if instrumentor._capture_completions and hasattr(response, "choices"):
                        if response.choices:
                            text = response.choices[0].text
                            span.set_attribute(
                                GenAIAttributes.completion_content(0),
                                text[:10000]
                            )

                    if OTEL_AVAILABLE:
                        span.set_status(Status(StatusCode.OK))

                    return response
                except Exception as e:
                    if OTEL_AVAILABLE:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                    raise

        return wrapper


class _StreamWrapper:
    """Wrapper for sync streaming responses."""

    def __init__(self, stream, span, capture_content: bool):
        self._stream = stream
        self._span = span
        self._capture_content = capture_content
        self._content_parts = []
        self._usage = {"input_tokens": 0, "output_tokens": 0}

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._stream)
            self._process_chunk(chunk)
            return chunk
        except StopIteration:
            self._finalize()
            raise

    def _process_chunk(self, chunk):
        """Process a stream chunk."""
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                self._content_parts.append(delta.content)

        if hasattr(chunk, "usage") and chunk.usage:
            self._usage["input_tokens"] = chunk.usage.prompt_tokens
            self._usage["output_tokens"] = chunk.usage.completion_tokens

    def _finalize(self):
        """Finalize the stream and update span."""
        if self._capture_content and self._content_parts:
            content = "".join(self._content_parts)
            self._span.set_attribute(
                GenAIAttributes.completion_content(0),
                content[:10000]
            )

        if self._usage["input_tokens"]:
            self._span.set_attribute(
                GenAIAttributes.USAGE_INPUT_TOKENS,
                self._usage["input_tokens"]
            )
        if self._usage["output_tokens"]:
            self._span.set_attribute(
                GenAIAttributes.USAGE_OUTPUT_TOKENS,
                self._usage["output_tokens"]
            )

        if OTEL_AVAILABLE:
            self._span.set_status(Status(StatusCode.OK))


class _AsyncStreamWrapper:
    """Wrapper for async streaming responses."""

    def __init__(self, stream, span, capture_content: bool):
        self._stream = stream
        self._span = span
        self._capture_content = capture_content
        self._content_parts = []
        self._usage = {"input_tokens": 0, "output_tokens": 0}

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._stream.__anext__()
            self._process_chunk(chunk)
            return chunk
        except StopAsyncIteration:
            self._finalize()
            raise

    def _process_chunk(self, chunk):
        """Process a stream chunk."""
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                self._content_parts.append(delta.content)

        if hasattr(chunk, "usage") and chunk.usage:
            self._usage["input_tokens"] = chunk.usage.prompt_tokens
            self._usage["output_tokens"] = chunk.usage.completion_tokens

    def _finalize(self):
        """Finalize the stream and update span."""
        if self._capture_content and self._content_parts:
            content = "".join(self._content_parts)
            self._span.set_attribute(
                GenAIAttributes.completion_content(0),
                content[:10000]
            )

        if self._usage["input_tokens"]:
            self._span.set_attribute(
                GenAIAttributes.USAGE_INPUT_TOKENS,
                self._usage["input_tokens"]
            )
        if self._usage["output_tokens"]:
            self._span.set_attribute(
                GenAIAttributes.USAGE_OUTPUT_TOKENS,
                self._usage["output_tokens"]
            )

        if OTEL_AVAILABLE:
            self._span.set_status(Status(StatusCode.OK))


__all__ = ["OpenAIInstrumentor"]
