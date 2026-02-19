"""
Core types for the evaluation framework.

This module defines the foundational types used throughout the framework:
- ExecutionMode: How evaluations are executed (blocking, non-blocking, distributed)
- EvalStatus: Status of an evaluation execution
- EvalResult: Result from a single evaluation
- BatchEvalResult: Aggregated results from batch evaluation
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Any, Dict, Optional, List, Union
from datetime import datetime, timezone

T = TypeVar("T")


class ExecutionMode(Enum):
    """
    How an evaluation should be executed.

    - BLOCKING: Synchronous execution, waits for result. Introduces latency.
    - NON_BLOCKING: Asynchronous execution, returns immediately. Zero latency impact.
    - DISTRIBUTED: Distributed across workers for batch processing.
    """
    BLOCKING = "blocking"
    NON_BLOCKING = "non_blocking"
    DISTRIBUTED = "distributed"


class EvalStatus(Enum):
    """Status of an evaluation execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class EvalResult(Generic[T]):
    """
    Result from any evaluation.

    Attributes:
        value: The evaluation result value (typed)
        eval_name: Name of the evaluation that produced this result
        eval_version: Version of the evaluation
        latency_ms: Time taken to run the evaluation in milliseconds
        status: Current status of the evaluation
        error: Error message if status is FAILED
        metadata: Additional metadata about the evaluation run
        timestamp: When the evaluation completed

    Example:
        result = EvalResult(
            value={"score": 0.95, "passed": True},
            eval_name="faithfulness",
            eval_version="1.0.0",
            latency_ms=150.5,
        )
    """
    value: T
    eval_name: str
    eval_version: str
    latency_ms: float
    status: EvalStatus = EvalStatus.COMPLETED
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def succeeded(self) -> bool:
        """Whether the evaluation completed successfully."""
        return self.status == EvalStatus.COMPLETED

    @property
    def failed(self) -> bool:
        """Whether the evaluation failed."""
        return self.status == EvalStatus.FAILED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "value": self.value,
            "eval_name": self.eval_name,
            "eval_version": self.eval_version,
            "latency_ms": self.latency_ms,
            "status": self.status.value,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalResult":
        """Create from dictionary."""
        return cls(
            value=data["value"],
            eval_name=data["eval_name"],
            eval_version=data["eval_version"],
            latency_ms=data["latency_ms"],
            status=EvalStatus(data["status"]),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.utcnow(),
        )

    def to_span_attributes(self) -> Dict[str, Any]:
        """
        Convert to flat dict for span attributes.

        Returns attributes suitable for OTEL span.set_attribute().
        """
        attrs = {
            "eval_name": self.eval_name,
            "eval_version": self.eval_version,
            "latency_ms": self.latency_ms,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.error:
            attrs["error"] = self.error
        return attrs

    @classmethod
    def failure(
        cls,
        eval_name: str,
        eval_version: str,
        error: str,
        latency_ms: float = 0.0,
    ) -> "EvalResult":
        """Create a failed result."""
        return cls(
            value=None,
            eval_name=eval_name,
            eval_version=eval_version,
            latency_ms=latency_ms,
            status=EvalStatus.FAILED,
            error=error,
        )


@dataclass
class BatchEvalResult:
    """
    Aggregated result from batch evaluation.

    Attributes:
        results: List of individual EvalResult objects
        total_count: Total number of evaluations attempted
        success_count: Number of successful evaluations
        failure_count: Number of failed evaluations
        total_latency_ms: Sum of all evaluation latencies

    Example:
        batch = BatchEvalResult.from_results(results)
        print(f"Success rate: {batch.success_rate:.1%}")
    """
    results: List[EvalResult]
    total_count: int
    success_count: int
    failure_count: int
    total_latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Percentage of successful evaluations (0.0 to 1.0)."""
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per evaluation in milliseconds."""
        if self.total_count == 0:
            return 0.0
        return self.total_latency_ms / self.total_count

    @classmethod
    def from_results(cls, results: List[EvalResult], **metadata) -> "BatchEvalResult":
        """Create from a list of EvalResult objects."""
        return cls(
            results=results,
            total_count=len(results),
            success_count=sum(1 for r in results if r.status == EvalStatus.COMPLETED),
            failure_count=sum(1 for r in results if r.status == EvalStatus.FAILED),
            total_latency_ms=sum(r.latency_ms for r in results),
            metadata=metadata,
        )

    def get_by_name(self, eval_name: str) -> List[EvalResult]:
        """Get all results for a specific evaluation name."""
        return [r for r in self.results if r.eval_name == eval_name]

    def get_failures(self) -> List[EvalResult]:
        """Get all failed results."""
        return [r for r in self.results if r.status == EvalStatus.FAILED]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "results": [r.to_dict() for r in self.results],
            "total_count": self.total_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_latency_ms": self.total_latency_ms,
            "metadata": self.metadata,
        }


# Type aliases for common patterns
EvalInputs = Dict[str, Any]
SpanAttributes = Dict[str, Union[str, int, float, bool]]
