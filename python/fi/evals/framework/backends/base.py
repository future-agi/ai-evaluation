"""
Base backend abstraction for evaluation execution.

This module defines the abstract interface for evaluation backends,
enabling pluggable execution strategies (thread pool, Temporal, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Callable, TypeVar, Generic, List
from datetime import datetime, timezone


T = TypeVar("T")


class TaskStatus(Enum):
    """Status of a backend task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class BackendConfig:
    """
    Base configuration for backends.

    Subclasses add backend-specific options.
    """
    max_workers: int = 4
    timeout_seconds: float = 300.0  # 5 minutes default
    retry_count: int = 0
    retry_delay_seconds: float = 1.0


@dataclass
class TaskHandle(Generic[T]):
    """
    Handle to a submitted task.

    Allows checking status and retrieving results.
    """
    task_id: str
    backend_name: str
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Internal state - set by backend
    _status: TaskStatus = TaskStatus.PENDING
    _result: Optional[T] = None
    _error: Optional[str] = None
    _completed_at: Optional[datetime] = None

    @property
    def status(self) -> TaskStatus:
        """Current task status."""
        return self._status

    @property
    def is_done(self) -> bool:
        """Whether task has completed (success or failure)."""
        return self._status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMEOUT,
        )

    @property
    def succeeded(self) -> bool:
        """Whether task completed successfully."""
        return self._status == TaskStatus.COMPLETED

    @property
    def result(self) -> Optional[T]:
        """Task result if completed successfully."""
        return self._result

    @property
    def error(self) -> Optional[str]:
        """Error message if task failed."""
        return self._error


class Backend(ABC):
    """
    Abstract base class for evaluation backends.

    Backends handle the actual execution of evaluation tasks,
    whether locally (thread pool), distributed (Temporal, Celery),
    or in a compute cluster (Ray).

    Example implementation:
        class MyBackend(Backend):
            name = "my_backend"

            def submit(self, fn, args, kwargs, context):
                # Submit task to execution system
                task_id = my_system.submit(fn, *args, **kwargs)
                return TaskHandle(task_id=task_id, backend_name=self.name)

            def get_result(self, handle, timeout):
                # Wait for and return result
                return my_system.wait(handle.task_id, timeout)

    Thread Safety:
        Backends must be thread-safe. Multiple threads may call
        submit() and get_result() concurrently.
    """

    name: str = "base"

    @abstractmethod
    def submit(
        self,
        fn: Callable[..., T],
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskHandle[T]:
        """
        Submit a task for execution.

        Args:
            fn: The function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            context: Trace context for propagation (trace_id, span_id, etc.)

        Returns:
            TaskHandle to track the task
        """
        pass

    @abstractmethod
    def get_result(
        self,
        handle: TaskHandle[T],
        timeout: Optional[float] = None,
    ) -> T:
        """
        Get the result of a submitted task.

        Args:
            handle: The task handle from submit()
            timeout: Maximum seconds to wait (None = wait forever)

        Returns:
            The task result

        Raises:
            TimeoutError: If timeout exceeded
            Exception: If task failed
        """
        pass

    @abstractmethod
    def get_status(self, handle: TaskHandle) -> TaskStatus:
        """
        Get current status of a task.

        Args:
            handle: The task handle

        Returns:
            Current TaskStatus
        """
        pass

    @abstractmethod
    def cancel(self, handle: TaskHandle) -> bool:
        """
        Attempt to cancel a task.

        Args:
            handle: The task handle

        Returns:
            True if cancelled, False if already running/complete
        """
        pass

    def submit_batch(
        self,
        tasks: List[tuple],  # List of (fn, args, kwargs, context)
    ) -> List[TaskHandle]:
        """
        Submit multiple tasks.

        Default implementation submits sequentially.
        Override for backends with batch submission support.

        Args:
            tasks: List of (function, args, kwargs, context) tuples

        Returns:
            List of TaskHandles
        """
        handles = []
        for fn, args, kwargs, context in tasks:
            handle = self.submit(fn, args, kwargs or {}, context)
            handles.append(handle)
        return handles

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the backend.

        Args:
            wait: Whether to wait for pending tasks to complete
        """
        pass  # Default: no-op

    def __enter__(self) -> "Backend":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown(wait=True)
