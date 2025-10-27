from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
)
from pydantic import ValidationError
from typing_extensions import get_original_bases
import time

from ..types import BatchRunResult, EvalResult, BaseMetricInput

BaseMetricInputType = TypeVar("BaseMetricInputType", bound=BaseMetricInput)


class BaseMetric(Generic[BaseMetricInputType], ABC):
    """The abstract base class for all metric evaluations.

    It provides a unified structure, centralizes batch processing,
    and handles standardized error logging and timing. Subclasses are only
    required to implement the logic for a single computation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

        # magic code to get the input metric basemodel to use for input validation.
        # This traverses the MRO to find the generic BaseMetric specialization.
        self.input_model: Type[BaseMetricInputType]
        for cls in self.__class__.__mro__:
            for base in get_original_bases(cls):
                if hasattr(base, "__origin__") and base.__origin__ is BaseMetric:
                    type_args = get_args(base)
                    if type_args:
                        self.input_model = type_args[0]
                        break
            if hasattr(self, "input_model"):
                break

        if not hasattr(self, "input_model"):
            raise TypeError(
                f"Could not determine input model for {self.__class__.__name__}. "
                "Ensure it inherits from BaseMetric with a specific input type, "
                "e.g., class MyMetric(BaseMetric[TextMetricInput]):"
            )

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """The official name of the metric"""
        pass

    def _validate_and_yield_inputs(
        self,
        inputs: List[Any],
    ) -> Iterator[Union[BaseMetricInputType, Exception]]:
        """
        A generator that validates each item from an input list.
        """
        for i, item in enumerate(inputs):
            try:
                if isinstance(item, dict):
                    # Validate and yield the dictionary as a Pydantic model
                    yield self.input_model.model_validate(item)
                elif isinstance(item, self.input_model):
                    # The item is already a valid Pydantic model, yield it directly
                    yield item
                else:
                    # The item is an unsupported type
                    raise TypeError(
                        f"Item at index {i} has an invalid type '{type(item).__name__}'. "
                        f"Expected a 'dict' or a '{self.input_model.__name__}' instance."
                    )
            except (ValidationError, TypeError) as e:
                # If validation fails for any reason, yield the exception
                yield e

    @abstractmethod
    def compute_one(self, inputs: BaseMetricInputType) -> Dict[str, Any]:
        """
        Computes the metric for a single, strongly-typed input.
        Pydantic validates the data before this method is ever called.

        Returns:
            A dictionary containing at least an 'output' key with the score.
            Can optionally include a 'reason' key for more details.
        """
        raise NotImplementedError

    def evaluate(
        self, inputs: Union[List[BaseMetricInputType], List[Dict[str, Any]]]
    ) -> BatchRunResult:
        """
        Evaluates the metric over a batch of strongly-typed inputs.
        """
        if not inputs:
            return BatchRunResult(eval_results=[])

        eval_results: List[Optional[EvalResult]] = []

        # Use the generator to process each item one by one
        for prepared_item in self._validate_and_yield_inputs(inputs):
            start_time = time.perf_counter()
            output = 0.0
            reason = ""

            # Check if the prepared_item is an exception from the validation step
            if isinstance(prepared_item, Exception):
                reason = f"Input validation failed: {prepared_item}"
            else:
                # If validation succeeded, run the actual metric computation
                result_dict = self.compute_one(prepared_item)
                output = result_dict.get("output")
                reason = result_dict.get("reason")

            end_time = time.perf_counter()
            runtime_ms = int((end_time - start_time) * 1000)

            eval_result = EvalResult(
                name=self.metric_name,
                output=output,
                reason=reason,
                runtime=runtime_ms,
                output_type="score",  # Default, can be customized if needed
            )
            eval_results.append(eval_result)

        return BatchRunResult(eval_results=eval_results)
