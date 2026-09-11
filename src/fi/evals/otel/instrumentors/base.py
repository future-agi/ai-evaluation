"""
Base Instrumentor.

Abstract base class for LLM client instrumentors.
Instrumentors automatically add tracing to LLM client libraries.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set
import functools
import logging

from ..processors import OTEL_AVAILABLE
from ..conventions import GenAIAttributes, normalize_system_name

if OTEL_AVAILABLE:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)


class BaseInstrumentor(ABC):
    """
    Abstract base class for LLM client instrumentors.

    Instrumentors patch LLM client libraries to automatically
    create spans and capture attributes for LLM calls.

    To create a custom instrumentor:
    1. Subclass BaseInstrumentor
    2. Implement instrument() and uninstrument()
    3. Use the helper methods to create standardized spans

    Example:
        class MyLLMInstrumentor(BaseInstrumentor):
            def instrument(self, **kwargs):
                import mylib
                mylib.Client.generate = self.wrap_method(
                    mylib.Client.generate,
                    span_name="mylib.generate",
                    system="mylib",
                )

            def uninstrument(self, **kwargs):
                import mylib
                # Restore original method
    """

    # Override in subclass
    system_name: str = "custom"
    library_name: str = "unknown"

    def __init__(self):
        self._instrumented = False
        self._original_methods: Dict[str, Callable] = {}
        self._tracer: Optional[Any] = None

    @property
    def is_instrumented(self) -> bool:
        """Check if instrumentation is active."""
        return self._instrumented

    def get_tracer(self) -> Any:
        """Get or create a tracer for this instrumentor."""
        if self._tracer is None:
            if OTEL_AVAILABLE:
                self._tracer = trace.get_tracer(
                    f"fi.evals.otel.instrumentors.{self.library_name}"
                )
            else:
                # Return a no-op tracer
                from ..tracer import _NoOpTracer
                self._tracer = _NoOpTracer()
        return self._tracer

    @abstractmethod
    def instrument(self, **kwargs) -> None:
        """
        Apply instrumentation to the library.

        Override this to patch the target library's methods.
        Store original methods in self._original_methods.

        Args:
            **kwargs: Optional configuration
        """
        pass

    @abstractmethod
    def uninstrument(self, **kwargs) -> None:
        """
        Remove instrumentation from the library.

        Override this to restore original methods.
        """
        pass

    def wrap_method(
        self,
        original: Callable,
        span_name: str,
        system: Optional[str] = None,
        operation: str = "chat",
        extract_model: Optional[Callable[[tuple, dict], str]] = None,
        extract_input: Optional[Callable[[tuple, dict], str]] = None,
        extract_output: Optional[Callable[[Any], str]] = None,
        extract_tokens: Optional[Callable[[Any], Dict[str, int]]] = None,
    ) -> Callable:
        """
        Wrap a method to add tracing.

        Args:
            original: Original method to wrap
            span_name: Name for the span
            system: System name (default: self.system_name)
            operation: Operation type (chat, completion, embedding)
            extract_model: Function to extract model from args
            extract_input: Function to extract input/prompt from args
            extract_output: Function to extract output from response
            extract_tokens: Function to extract token counts from response

        Returns:
            Wrapped method with tracing
        """
        instrumentor = self

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            tracer = instrumentor.get_tracer()

            # Build initial attributes
            attributes = {
                GenAIAttributes.SYSTEM: normalize_system_name(system or instrumentor.system_name),
                GenAIAttributes.OPERATION_NAME: operation,
            }

            # Extract model if possible
            if extract_model:
                try:
                    model = extract_model(args, kwargs)
                    if model:
                        attributes[GenAIAttributes.REQUEST_MODEL] = model
                except Exception as e:
                    logger.debug(f"Failed to extract model: {e}")

            # Extract input if possible
            if extract_input:
                try:
                    input_text = extract_input(args, kwargs)
                    if input_text:
                        attributes[GenAIAttributes.prompt_content(0)] = input_text[:10000]
                except Exception as e:
                    logger.debug(f"Failed to extract input: {e}")

            with tracer.start_as_current_span(span_name, attributes=attributes) as span:
                try:
                    # Call original method
                    result = original(*args, **kwargs)

                    # Extract response data
                    if extract_output:
                        try:
                            output = extract_output(result)
                            if output:
                                span.set_attribute(
                                    GenAIAttributes.completion_content(0),
                                    output[:10000]
                                )
                        except Exception as e:
                            logger.debug(f"Failed to extract output: {e}")

                    if extract_tokens:
                        try:
                            tokens = extract_tokens(result)
                            if tokens.get("input_tokens"):
                                span.set_attribute(
                                    GenAIAttributes.USAGE_INPUT_TOKENS,
                                    tokens["input_tokens"]
                                )
                            if tokens.get("output_tokens"):
                                span.set_attribute(
                                    GenAIAttributes.USAGE_OUTPUT_TOKENS,
                                    tokens["output_tokens"]
                                )
                            if tokens.get("input_tokens") and tokens.get("output_tokens"):
                                span.set_attribute(
                                    GenAIAttributes.USAGE_TOTAL_TOKENS,
                                    tokens["input_tokens"] + tokens["output_tokens"]
                                )
                        except Exception as e:
                            logger.debug(f"Failed to extract tokens: {e}")

                    if OTEL_AVAILABLE:
                        span.set_status(Status(StatusCode.OK))

                    return result

                except Exception as e:
                    if OTEL_AVAILABLE:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                    raise

        return wrapper

    def wrap_async_method(
        self,
        original: Callable,
        span_name: str,
        system: Optional[str] = None,
        operation: str = "chat",
        extract_model: Optional[Callable[[tuple, dict], str]] = None,
        extract_input: Optional[Callable[[tuple, dict], str]] = None,
        extract_output: Optional[Callable[[Any], str]] = None,
        extract_tokens: Optional[Callable[[Any], Dict[str, int]]] = None,
    ) -> Callable:
        """
        Wrap an async method to add tracing.

        Same as wrap_method but for async functions.
        """
        instrumentor = self

        @functools.wraps(original)
        async def wrapper(*args, **kwargs):
            tracer = instrumentor.get_tracer()

            attributes = {
                GenAIAttributes.SYSTEM: normalize_system_name(system or instrumentor.system_name),
                GenAIAttributes.OPERATION_NAME: operation,
            }

            if extract_model:
                try:
                    model = extract_model(args, kwargs)
                    if model:
                        attributes[GenAIAttributes.REQUEST_MODEL] = model
                except Exception as e:
                    logger.debug(f"Failed to extract model: {e}")

            if extract_input:
                try:
                    input_text = extract_input(args, kwargs)
                    if input_text:
                        attributes[GenAIAttributes.prompt_content(0)] = input_text[:10000]
                except Exception as e:
                    logger.debug(f"Failed to extract input: {e}")

            with tracer.start_as_current_span(span_name, attributes=attributes) as span:
                try:
                    result = await original(*args, **kwargs)

                    if extract_output:
                        try:
                            output = extract_output(result)
                            if output:
                                span.set_attribute(
                                    GenAIAttributes.completion_content(0),
                                    output[:10000]
                                )
                        except Exception as e:
                            logger.debug(f"Failed to extract output: {e}")

                    if extract_tokens:
                        try:
                            tokens = extract_tokens(result)
                            if tokens.get("input_tokens"):
                                span.set_attribute(
                                    GenAIAttributes.USAGE_INPUT_TOKENS,
                                    tokens["input_tokens"]
                                )
                            if tokens.get("output_tokens"):
                                span.set_attribute(
                                    GenAIAttributes.USAGE_OUTPUT_TOKENS,
                                    tokens["output_tokens"]
                                )
                            if tokens.get("input_tokens") and tokens.get("output_tokens"):
                                span.set_attribute(
                                    GenAIAttributes.USAGE_TOTAL_TOKENS,
                                    tokens["input_tokens"] + tokens["output_tokens"]
                                )
                        except Exception as e:
                            logger.debug(f"Failed to extract tokens: {e}")

                    if OTEL_AVAILABLE:
                        span.set_status(Status(StatusCode.OK))

                    return result

                except Exception as e:
                    if OTEL_AVAILABLE:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                    raise

        return wrapper


class InstrumentorManager:
    """
    Manager for multiple instrumentors.

    Provides a unified interface to instrument/uninstrument
    multiple LLM libraries at once.

    Example:
        manager = InstrumentorManager()
        manager.add(OpenAIInstrumentor())
        manager.add(AnthropicInstrumentor())

        # Instrument all
        manager.instrument_all()

        # Later...
        manager.uninstrument_all()
    """

    def __init__(self):
        self._instrumentors: Dict[str, BaseInstrumentor] = {}

    def add(self, instrumentor: BaseInstrumentor) -> None:
        """Add an instrumentor to the manager."""
        self._instrumentors[instrumentor.library_name] = instrumentor

    def remove(self, library_name: str) -> None:
        """Remove an instrumentor by library name."""
        if library_name in self._instrumentors:
            instrumentor = self._instrumentors[library_name]
            if instrumentor.is_instrumented:
                instrumentor.uninstrument()
            del self._instrumentors[library_name]

    def get(self, library_name: str) -> Optional[BaseInstrumentor]:
        """Get an instrumentor by library name."""
        return self._instrumentors.get(library_name)

    def instrument_all(self, **kwargs) -> List[str]:
        """
        Instrument all registered libraries.

        Returns list of successfully instrumented library names.
        """
        instrumented = []
        for name, instrumentor in self._instrumentors.items():
            try:
                if not instrumentor.is_instrumented:
                    instrumentor.instrument(**kwargs)
                    instrumented.append(name)
            except Exception as e:
                logger.warning(f"Failed to instrument {name}: {e}")
        return instrumented

    def uninstrument_all(self, **kwargs) -> List[str]:
        """
        Uninstrument all registered libraries.

        Returns list of successfully uninstrumented library names.
        """
        uninstrumented = []
        for name, instrumentor in self._instrumentors.items():
            try:
                if instrumentor.is_instrumented:
                    instrumentor.uninstrument(**kwargs)
                    uninstrumented.append(name)
            except Exception as e:
                logger.warning(f"Failed to uninstrument {name}: {e}")
        return uninstrumented

    @property
    def instrumented_libraries(self) -> Set[str]:
        """Get set of currently instrumented library names."""
        return {
            name for name, inst in self._instrumentors.items()
            if inst.is_instrumented
        }


__all__ = [
    "BaseInstrumentor",
    "InstrumentorManager",
]
