"""
Thread pool backend for local evaluation execution.

Provides a simple, lightweight backend for running evaluations
in a local thread pool. Suitable for development and single-machine
production deployments.
"""

from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, TypeVar, List
from datetime import datetime, timezone
import threading
import uuid

from .base import Backend, BackendConfig, TaskHandle, TaskStatus


T = TypeVar("T")


@dataclass
class ThreadPoolConfig(BackendConfig):
    """Configuration for thread pool backend."""
    thread_name_prefix: str = "eval_"


class ThreadPoolBackend(Backend):
    """
    Thread pool backend for local execution.

    Runs evaluation tasks in a local thread pool. Best for:
    - Development and testing
    - Single-machine deployments
    - Low-latency requirements

    Example:
        config = ThreadPoolConfig(max_workers=8)
        backend = ThreadPoolBackend(config)

        handle = backend.submit(my_eval_fn, args=(inputs,))
        result = backend.get_result(handle, timeout=30.0)

    Thread Safety:
        This class is thread-safe.
    """

    name = "thread_pool"

    def __init__(self, config: Optional[ThreadPoolConfig] = None):
        """
        Initialize the thread pool backend.

        Args:
            config: Configuration options (uses defaults if None)
        """
        self.config = config or ThreadPoolConfig()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures: Dict[str, Future] = {}
        self._handles: Dict[str, TaskHandle] = {}
        self._lock = threading.Lock()

    @property
    def executor(self) -> ThreadPoolExecutor:
        """Get or create the thread pool executor."""
        if self._executor is None:
            with self._lock:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        max_workers=self.config.max_workers,
                        thread_name_prefix=self.config.thread_name_prefix,
                    )
        return self._executor

    def submit(
        self,
        fn: Callable[..., T],
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskHandle[T]:
        """
        Submit a task to the thread pool.

        Args:
            fn: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            context: Trace context (stored in handle.metadata)

        Returns:
            TaskHandle to track the task
        """
        task_id = uuid.uuid4().hex[:16]
        kwargs = kwargs or {}

        # Create handle
        handle: TaskHandle[T] = TaskHandle(
            task_id=task_id,
            backend_name=self.name,
            metadata={"context": context} if context else {},
        )
        handle._status = TaskStatus.PENDING

        # Submit to executor
        future = self.executor.submit(fn, *args, **kwargs)

        # Store mappings
        with self._lock:
            self._futures[task_id] = future
            self._handles[task_id] = handle

        # Update status when running
        def on_start():
            handle._status = TaskStatus.RUNNING

        # Update status on completion
        def on_done(f: Future):
            with self._lock:
                handle._completed_at = datetime.now(timezone.utc)
                try:
                    result = f.result(timeout=0)  # Don't block
                    handle._result = result
                    handle._status = TaskStatus.COMPLETED
                except FuturesTimeout:
                    handle._status = TaskStatus.TIMEOUT
                    handle._error = "Task timed out"
                except Exception as e:
                    handle._status = TaskStatus.FAILED
                    handle._error = str(e)

        future.add_done_callback(on_done)

        return handle

    def get_result(
        self,
        handle: TaskHandle[T],
        timeout: Optional[float] = None,
    ) -> T:
        """
        Get result from a submitted task.

        Args:
            handle: The task handle
            timeout: Maximum seconds to wait

        Returns:
            The task result

        Raises:
            TimeoutError: If timeout exceeded
            KeyError: If handle not found
            Exception: If task raised an exception
        """
        with self._lock:
            future = self._futures.get(handle.task_id)
            if future is None:
                raise KeyError(f"Task not found: {handle.task_id}")

        effective_timeout = timeout or self.config.timeout_seconds
        return future.result(timeout=effective_timeout)

    def get_status(self, handle: TaskHandle) -> TaskStatus:
        """
        Get current status of a task.

        Args:
            handle: The task handle

        Returns:
            Current TaskStatus
        """
        with self._lock:
            future = self._futures.get(handle.task_id)
            if future is None:
                return TaskStatus.FAILED  # Not found

            if future.cancelled():
                return TaskStatus.CANCELLED
            elif future.done():
                try:
                    future.result(timeout=0)
                    return TaskStatus.COMPLETED
                except Exception:
                    return TaskStatus.FAILED
            elif future.running():
                return TaskStatus.RUNNING
            else:
                return TaskStatus.PENDING

    def cancel(self, handle: TaskHandle) -> bool:
        """
        Attempt to cancel a task.

        Args:
            handle: The task handle

        Returns:
            True if cancelled, False otherwise
        """
        with self._lock:
            future = self._futures.get(handle.task_id)
            if future is None:
                return False

            cancelled = future.cancel()
            if cancelled:
                handle._status = TaskStatus.CANCELLED
            return cancelled

    def submit_batch(
        self,
        tasks: List[tuple],
    ) -> List[TaskHandle]:
        """
        Submit multiple tasks to the thread pool.

        Args:
            tasks: List of (fn, args, kwargs, context) tuples

        Returns:
            List of TaskHandles
        """
        handles = []
        for task in tasks:
            fn = task[0]
            args = task[1] if len(task) > 1 else ()
            kwargs = task[2] if len(task) > 2 else {}
            context = task[3] if len(task) > 3 else None
            handle = self.submit(fn, args, kwargs, context)
            handles.append(handle)
        return handles

    def wait_all(
        self,
        handles: List[TaskHandle],
        timeout: Optional[float] = None,
    ) -> List[Any]:
        """
        Wait for all tasks and return results.

        Args:
            handles: List of task handles
            timeout: Maximum total wait time

        Returns:
            List of results (in same order as handles)
        """
        results = []
        for handle in handles:
            try:
                result = self.get_result(handle, timeout=timeout)
                results.append(result)
            except Exception as e:
                results.append(e)
        return results

    def pending_count(self) -> int:
        """Get count of pending/running tasks."""
        with self._lock:
            return sum(
                1 for f in self._futures.values()
                if not f.done()
            )

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the thread pool.

        Args:
            wait: Whether to wait for pending tasks
        """
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None

        with self._lock:
            self._futures.clear()
            self._handles.clear()

    def __enter__(self) -> "ThreadPoolBackend":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown(wait=True)
