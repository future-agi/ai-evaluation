"""
Protocol definitions for evaluations.

This module defines the BaseEvaluation protocol that all evaluations must implement,
and the EvalRegistry for registering and looking up evaluations by name/version.

The protocol-based approach allows:
- Type checking at development time
- Runtime validation of evaluation implementations
- Decoupled evaluation logic from execution mode
"""

from typing import (
    Protocol,
    TypeVar,
    Dict,
    Any,
    Optional,
    runtime_checkable,
    Type,
    List,
    Callable,
)
from abc import abstractmethod
import asyncio

T = TypeVar("T")


@runtime_checkable
class BaseEvaluation(Protocol[T]):
    """
    Protocol that all evaluations must implement.

    Evaluations are stateless functions that:
    1. Take inputs (dict)
    2. Return a typed result
    3. Can convert results to span attributes

    The protocol is runtime checkable, so you can verify implementations:
        isinstance(my_eval, BaseEvaluation)  # True if properly implemented

    Attributes:
        name: Unique name for this evaluation (e.g., "faithfulness")
        version: Semantic version string (e.g., "1.0.0")

    Example:
        class FaithfulnessEval(BaseEvaluation[FaithfulnessResult]):
            name = "faithfulness"
            version = "1.0.0"

            def evaluate(self, inputs: Dict[str, Any]) -> FaithfulnessResult:
                query = inputs["query"]
                response = inputs["response"]
                context = inputs["context"]
                # ... evaluation logic ...
                return FaithfulnessResult(score=0.95, reason="...")

            def get_span_attributes(self, result: FaithfulnessResult) -> Dict[str, Any]:
                return {
                    "score": result.score,
                    "passed": result.score >= 0.7,
                }
    """

    name: str
    version: str

    @abstractmethod
    def evaluate(self, inputs: Dict[str, Any]) -> T:
        """
        Synchronous evaluation - implement this.

        This is the main evaluation method. It should be stateless and
        deterministic given the same inputs.

        Args:
            inputs: Dict containing evaluation inputs. Keys depend on
                    the specific evaluation (e.g., "query", "response", "context")

        Returns:
            Typed result object containing evaluation output

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If evaluation fails for other reasons
        """
        ...

    async def evaluate_async(self, inputs: Dict[str, Any]) -> T:
        """
        Async evaluation - defaults to sync wrapped.

        Override this method for true async implementations that can
        benefit from non-blocking I/O (e.g., API calls, database queries).

        Args:
            inputs: Dict containing evaluation inputs

        Returns:
            Typed result object
        """
        # Default implementation runs sync version in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.evaluate, inputs)

    @abstractmethod
    def get_span_attributes(self, result: T) -> Dict[str, Any]:
        """
        Convert result to span attributes.

        Returns a flat dict suitable for OTEL span attributes.
        Keys should NOT include the eval name prefix - that's added by the framework.

        The returned values must be OTEL-compatible types:
        - str, int, float, bool
        - Lists of the above types

        Args:
            result: The evaluation result

        Returns:
            Dict of attribute name to value

        Example:
            def get_span_attributes(self, result: MyResult) -> Dict[str, Any]:
                return {
                    "score": result.score,
                    "passed": result.passed,
                    "category": result.category,
                }
            # Framework adds prefix: eval.my_eval.score, eval.my_eval.passed, etc.
        """
        ...

    def validate_inputs(self, inputs: Dict[str, Any]) -> Optional[str]:
        """
        Validate inputs before evaluation.

        Override to add custom validation logic. Called before evaluate().

        Args:
            inputs: Dict containing evaluation inputs

        Returns:
            None if valid, error message string if invalid
        """
        return None

    def get_required_inputs(self) -> List[str]:
        """
        Get list of required input keys.

        Override to declare required inputs for documentation and validation.

        Returns:
            List of required input key names
        """
        return []


class EvalRegistry:
    """
    Registry for evaluation classes.

    Allows lookups by name/version for distributed execution where
    the evaluation class needs to be instantiated on a worker.

    The registry is a singleton - all registrations go to the same global registry.

    Example:
        # Register an evaluation
        @register_evaluation
        class MyEval(BaseEvaluation[MyResult]):
            name = "my_eval"
            version = "1.0.0"
            ...

        # Look up later
        eval_class = EvalRegistry.get("my_eval", "1.0.0")
        eval_instance = eval_class()
    """
    _registry: Dict[str, Dict[str, Type[BaseEvaluation]]] = {}

    @classmethod
    def register(cls, eval_class: Type[BaseEvaluation]) -> Type[BaseEvaluation]:
        """
        Register an evaluation class.

        Args:
            eval_class: The evaluation class to register

        Returns:
            The same class (for use as decorator)

        Raises:
            ValueError: If class doesn't have name or version attributes
        """
        name = getattr(eval_class, 'name', None)
        version = getattr(eval_class, 'version', '1.0.0')

        if not name:
            name = eval_class.__name__

        if name not in cls._registry:
            cls._registry[name] = {}

        cls._registry[name][version] = eval_class
        return eval_class

    @classmethod
    def get(cls, name: str, version: str = "latest") -> Type[BaseEvaluation]:
        """
        Get evaluation class by name and version.

        Args:
            name: Evaluation name
            version: Version string or "latest" for highest version

        Returns:
            The evaluation class

        Raises:
            ValueError: If evaluation or version not found
        """
        if name not in cls._registry:
            raise ValueError(f"Evaluation '{name}' not found in registry")

        versions = cls._registry[name]

        if version == "latest":
            # Get highest version using semantic versioning comparison
            version = cls._get_latest_version(list(versions.keys()))

        if version not in versions:
            available = list(versions.keys())
            raise ValueError(
                f"Version '{version}' not found for evaluation '{name}'. "
                f"Available versions: {available}"
            )

        return versions[version]

    @classmethod
    def _get_latest_version(cls, versions: List[str]) -> str:
        """Get the latest version from a list of version strings."""
        def version_key(v: str) -> tuple:
            # Parse semver: major.minor.patch
            parts = v.split(".")
            result = []
            for part in parts:
                try:
                    result.append(int(part))
                except ValueError:
                    result.append(0)
            # Pad to 3 elements
            while len(result) < 3:
                result.append(0)
            return tuple(result)

        return max(versions, key=version_key)

    @classmethod
    def get_instance(
        cls,
        name: str,
        version: str = "latest",
        **kwargs,
    ) -> BaseEvaluation:
        """
        Get an instantiated evaluation.

        Args:
            name: Evaluation name
            version: Version string or "latest"
            **kwargs: Arguments to pass to evaluation constructor

        Returns:
            Instantiated evaluation object
        """
        eval_class = cls.get(name, version)
        return eval_class(**kwargs)

    @classmethod
    def list_all(cls) -> Dict[str, List[str]]:
        """
        List all registered evaluations.

        Returns:
            Dict mapping evaluation names to list of available versions
        """
        return {name: list(versions.keys()) for name, versions in cls._registry.items()}

    @classmethod
    def is_registered(cls, name: str, version: Optional[str] = None) -> bool:
        """
        Check if an evaluation is registered.

        Args:
            name: Evaluation name
            version: Optional specific version to check

        Returns:
            True if registered, False otherwise
        """
        if name not in cls._registry:
            return False
        if version is None:
            return True
        return version in cls._registry[name]

    @classmethod
    def unregister(cls, name: str, version: Optional[str] = None) -> bool:
        """
        Remove an evaluation from the registry.

        Args:
            name: Evaluation name
            version: Specific version to remove, or None to remove all versions

        Returns:
            True if something was removed, False otherwise
        """
        if name not in cls._registry:
            return False

        if version is None:
            del cls._registry[name]
            return True

        if version in cls._registry[name]:
            del cls._registry[name][version]
            if not cls._registry[name]:
                del cls._registry[name]
            return True

        return False

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (for testing)."""
        cls._registry.clear()


def register_evaluation(cls: Type[BaseEvaluation]) -> Type[BaseEvaluation]:
    """
    Decorator to register an evaluation class.

    Example:
        @register_evaluation
        class MyEval(BaseEvaluation[MyResult]):
            name = "my_eval"
            version = "1.0.0"

            def evaluate(self, inputs: Dict[str, Any]) -> MyResult:
                ...

            def get_span_attributes(self, result: MyResult) -> Dict[str, Any]:
                ...
    """
    return EvalRegistry.register(cls)


def create_evaluation(
    name: str,
    version: str = "1.0.0",
    evaluate_fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
    span_attributes_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
) -> Type[BaseEvaluation]:
    """
    Factory function to create an evaluation class from functions.

    Useful for simple evaluations that don't need a full class definition.

    Args:
        name: Evaluation name
        version: Version string
        evaluate_fn: The evaluation function
        span_attributes_fn: Function to convert result to span attributes

    Returns:
        A new evaluation class

    Example:
        MyEval = create_evaluation(
            name="simple_check",
            evaluate_fn=lambda inputs: {"passed": len(inputs["response"]) > 10},
            span_attributes_fn=lambda result: {"passed": result["passed"]},
        )
    """
    class DynamicEvaluation:
        def __init__(self):
            pass

        def evaluate(self, inputs: Dict[str, Any]) -> Any:
            if evaluate_fn is None:
                raise NotImplementedError("evaluate_fn not provided")
            return evaluate_fn(inputs)

        def get_span_attributes(self, result: Any) -> Dict[str, Any]:
            if span_attributes_fn is None:
                # Default: try to convert result to dict
                if isinstance(result, dict):
                    return {k: v for k, v in result.items() if isinstance(v, (str, int, float, bool))}
                return {}
            return span_attributes_fn(result)

        def validate_inputs(self, inputs: Dict[str, Any]) -> Optional[str]:
            return None

    DynamicEvaluation.name = name
    DynamicEvaluation.version = version

    return DynamicEvaluation
