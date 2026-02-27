"""
Anthropic Instrumentor.

Automatic instrumentation for the Anthropic Python client library.
Supports both sync and async APIs.
"""

from typing import Any, Dict, Optional, Callable
import functools
import logging

from .base import BaseInstrumentor
from ..processors import OTEL_AVAILABLE
from ..conventions import (
    GenAIAttributes,
    SYSTEM_ANTHROPIC,
    OPERATION_CHAT,
    FINISH_STOP,
    FINISH_LENGTH,
)

if OTEL_AVAILABLE:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)


class AnthropicInstrumentor(BaseInstrumentor):
    """
    Instrumentor for the Anthropic Python client.

    Automatically traces:
    - messages.create

    Example:
        from fi.evals.otel.instrumentors import AnthropicInstrumentor

        # Instrument Anthropic
        instrumentor = AnthropicInstrumentor()
        instrumentor.instrument()

        # All Anthropic calls are now traced
        from anthropic import Anthropic
        client = Anthropic()
        response = client.messages.create(
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        # Span automatically created with model, tokens, etc.

        # To disable
        instrumentor.uninstrument()
    """

    system_name = "anthropic"
    library_name = "anthropic"

    def __init__(
        self,
        capture_prompts: bool = True,
        capture_completions: bool = True,
        capture_streaming: bool = True,
    ):
        """
        Initialize the Anthropic instrumentor.

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
        """Apply instrumentation to Anthropic client."""
        if self._instrumented:
            return

        try:
            import anthropic
            from anthropic.resources import messages

            # Store original methods
            self._original_methods["messages_create"] = messages.Messages.create
            self._original_methods["async_messages_create"] = (
                messages.AsyncMessages.create
            )

            # Patch messages.create
            messages.Messages.create = self._wrap_messages_create(
                self._original_methods["messages_create"],
                is_async=False,
            )
            messages.AsyncMessages.create = self._wrap_messages_create(
                self._original_methods["async_messages_create"],
                is_async=True,
            )

            self._instrumented = True
            logger.info("Anthropic instrumentation applied")

        except ImportError:
            logger.warning("Anthropic library not installed, skipping instrumentation")
        except Exception as e:
            logger.error(f"Failed to instrument Anthropic: {e}")

    def uninstrument(self, **kwargs) -> None:
        """Remove instrumentation from Anthropic client."""
        if not self._instrumented:
            return

        try:
            from anthropic.resources import messages

            # Restore original methods
            if "messages_create" in self._original_methods:
                messages.Messages.create = self._original_methods["messages_create"]
            if "async_messages_create" in self._original_methods:
                messages.AsyncMessages.create = (
                    self._original_methods["async_messages_create"]
                )

            self._original_methods.clear()
            self._instrumented = False
            logger.info("Anthropic instrumentation removed")

        except Exception as e:
            logger.error(f"Failed to uninstrument Anthropic: {e}")

    def _wrap_messages_create(self, original: Callable, is_async: bool) -> Callable:
        """Wrap messages.create."""
        instrumentor = self

        if is_async:
            @functools.wraps(original)
            async def async_wrapper(self_arg, *args, **kwargs):
                return await instrumentor._trace_messages_create(
                    original, self_arg, args, kwargs, is_async=True
                )
            return async_wrapper
        else:
            @functools.wraps(original)
            def wrapper(self_arg, *args, **kwargs):
                return instrumentor._trace_messages_create(
                    original, self_arg, args, kwargs, is_async=False
                )
            return wrapper

    def _trace_messages_create(
        self,
        original: Callable,
        self_arg: Any,
        args: tuple,
        kwargs: dict,
        is_async: bool,
    ):
        """Trace a messages.create call."""
        tracer = self.get_tracer()

        # Extract parameters
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        system = kwargs.get("system", "")
        stream = kwargs.get("stream", False)
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")

        # Build attributes
        attributes = {
            GenAIAttributes.SYSTEM: SYSTEM_ANTHROPIC,
            GenAIAttributes.OPERATION_NAME: OPERATION_CHAT,
            GenAIAttributes.REQUEST_MODEL: model,
        }

        if temperature is not None:
            attributes[GenAIAttributes.REQUEST_TEMPERATURE] = temperature
        if max_tokens is not None:
            attributes[GenAIAttributes.REQUEST_MAX_TOKENS] = max_tokens

        # Capture prompt
        if self._capture_prompts:
            prompt_text = self._format_messages(messages, system)
            attributes[GenAIAttributes.prompt_content(0)] = prompt_text[:10000]

        span_name = "anthropic.messages.create"

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
        """Extract data from a messages response."""
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
                if hasattr(usage, "input_tokens"):
                    span.set_attribute(
                        GenAIAttributes.USAGE_INPUT_TOKENS,
                        usage.input_tokens
                    )
                if hasattr(usage, "output_tokens"):
                    span.set_attribute(
                        GenAIAttributes.USAGE_OUTPUT_TOKENS,
                        usage.output_tokens
                    )
                    # Calculate total
                    input_tokens = getattr(usage, "input_tokens", 0)
                    output_tokens = getattr(usage, "output_tokens", 0)
                    span.set_attribute(
                        GenAIAttributes.USAGE_TOTAL_TOKENS,
                        input_tokens + output_tokens
                    )

            # Stop reason
            if hasattr(response, "stop_reason") and response.stop_reason:
                span.set_attribute(
                    GenAIAttributes.RESPONSE_FINISH_REASON,
                    response.stop_reason
                )

            # Content
            if self._capture_completions and hasattr(response, "content"):
                content = response.content
                if content and len(content) > 0:
                    # Anthropic returns list of content blocks
                    text_parts = []
                    for block in content:
                        if hasattr(block, "text"):
                            text_parts.append(block.text)
                        elif hasattr(block, "type") and block.type == "text":
                            text_parts.append(getattr(block, "text", ""))

                    if text_parts:
                        span.set_attribute(
                            GenAIAttributes.completion_content(0),
                            "\n".join(text_parts)[:10000]
                        )
                        span.set_attribute(
                            GenAIAttributes.completion_role(0),
                            "assistant"
                        )

        except Exception as e:
            logger.debug(f"Failed to extract response data: {e}")

    def _format_messages(self, messages: list, system: str = "") -> str:
        """Format messages list as string."""
        parts = []

        # Add system prompt if present
        if system:
            parts.append(f"[system]: {system}")

        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                # Content can be string or list of content blocks
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    content = "\n".join(text_parts)
                parts.append(f"[{role}]: {content}")
            else:
                parts.append(str(msg))

        return "\n".join(parts)

    def _wrap_stream_response(self, stream: Any, span: Any):
        """Wrap a streaming response to capture data."""
        return _AnthropicStreamWrapper(stream, span, self._capture_completions)

    def _wrap_async_stream_response(self, stream: Any, span: Any):
        """Wrap an async streaming response."""
        return _AnthropicAsyncStreamWrapper(stream, span, self._capture_completions)


class _AnthropicStreamWrapper:
    """Wrapper for sync streaming responses from Anthropic."""

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
            event = next(self._stream)
            self._process_event(event)
            return event
        except StopIteration:
            self._finalize()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._finalize()

    def _process_event(self, event):
        """Process a stream event."""
        # Anthropic uses different event types
        event_type = getattr(event, "type", None)

        if event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta and hasattr(delta, "text"):
                self._content_parts.append(delta.text)

        elif event_type == "message_delta":
            usage = getattr(event, "usage", None)
            if usage:
                if hasattr(usage, "output_tokens"):
                    self._usage["output_tokens"] = usage.output_tokens

        elif event_type == "message_start":
            message = getattr(event, "message", None)
            if message and hasattr(message, "usage"):
                self._usage["input_tokens"] = message.usage.input_tokens

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
        if self._usage["input_tokens"] or self._usage["output_tokens"]:
            self._span.set_attribute(
                GenAIAttributes.USAGE_TOTAL_TOKENS,
                self._usage["input_tokens"] + self._usage["output_tokens"]
            )

        if OTEL_AVAILABLE:
            self._span.set_status(Status(StatusCode.OK))


class _AnthropicAsyncStreamWrapper:
    """Wrapper for async streaming responses from Anthropic."""

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
            event = await self._stream.__anext__()
            self._process_event(event)
            return event
        except StopAsyncIteration:
            self._finalize()
            raise

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self._finalize()

    def _process_event(self, event):
        """Process a stream event."""
        event_type = getattr(event, "type", None)

        if event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta and hasattr(delta, "text"):
                self._content_parts.append(delta.text)

        elif event_type == "message_delta":
            usage = getattr(event, "usage", None)
            if usage:
                if hasattr(usage, "output_tokens"):
                    self._usage["output_tokens"] = usage.output_tokens

        elif event_type == "message_start":
            message = getattr(event, "message", None)
            if message and hasattr(message, "usage"):
                self._usage["input_tokens"] = message.usage.input_tokens

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
        if self._usage["input_tokens"] or self._usage["output_tokens"]:
            self._span.set_attribute(
                GenAIAttributes.USAGE_TOTAL_TOKENS,
                self._usage["input_tokens"] + self._usage["output_tokens"]
            )

        if OTEL_AVAILABLE:
            self._span.set_status(Status(StatusCode.OK))


__all__ = ["AnthropicInstrumentor"]
