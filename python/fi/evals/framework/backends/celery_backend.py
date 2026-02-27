"""
Celery backend for distributed task execution.

Provides distributed task execution using Celery with Redis/RabbitMQ.
Requires: pip install 'celery[redis]'
"""

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

from .base import Backend, BackendConfig, TaskHandle, TaskStatus
from ._utils import CELERY

T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass
class CeleryConfig(BackendConfig):
    """
    Configuration for Celery backend.

    Attributes:
        broker_url: Message broker URL (Redis/RabbitMQ)
        result_backend: Result backend URL
        task_queue: Queue name for tasks
        task_priority: Default task priority (0-9, higher = more priority)
        task_serializer: Serialization format ('json', 'pickle')
        result_serializer: Result serialization format
        task_acks_late: Acknowledge tasks after execution
        task_reject_on_worker_lost: Reject tasks if worker dies
        task_track_started: Track when tasks start executing
        worker_prefetch_multiplier: Number of tasks to prefetch
    """

    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/1"
    task_queue: str = "eval_tasks"
    task_priority: int = 0
    task_serializer: str = "cloudpickle"
    result_serializer: str = "cloudpickle"
    task_acks_late: bool = True
    task_reject_on_worker_lost: bool = True
    task_track_started: bool = True
    worker_prefetch_multiplier: int = 1


class CeleryBackend(Backend):
    """
    Celery backend for distributed task execution.

    Uses Celery for distributed task execution across multiple workers.
    Integrates with existing Celery infrastructure (Redis, RabbitMQ).

    Example:
        config = CeleryConfig(
            broker_url="redis://localhost:6379/0",
            result_backend="redis://localhost:6379/1",
            task_queue="eval_tasks",
        )
        backend = CeleryBackend(config)

        handle = backend.submit(my_eval_func, args=(input_data,))
        result = backend.get_result(handle)

    Note:
        Requires Celery workers to be running to execute tasks.
        Start workers with: celery -A your_app worker -Q eval_tasks
    """

    name: str = "celery"

    def __init__(self, config: Optional[CeleryConfig] = None):
        """
        Initialize Celery backend.

        Args:
            config: Celery configuration

        Raises:
            ImportError: If celery is not installed
        """
        CELERY.require()

        self.config = config or CeleryConfig()
        self._app: Optional[Any] = None
        self._task: Optional[Any] = None
        self._handles: Dict[str, TaskHandle] = {}
        self._async_results: Dict[str, Any] = {}
        self._lock = threading.Lock()

        self._setup_celery()

    def _setup_celery(self) -> None:
        """Set up Celery application and task from shared worker module."""
        from .celery_worker import app, eval_task

        # Override broker/backend if config differs from env defaults
        app.conf.update(
            broker_url=self.config.broker_url,
            result_backend=self.config.result_backend,
            task_serializer=self.config.task_serializer,
            result_serializer=self.config.result_serializer,
            task_acks_late=self.config.task_acks_late,
            task_reject_on_worker_lost=self.config.task_reject_on_worker_lost,
            task_track_started=self.config.task_track_started,
            worker_prefetch_multiplier=self.config.worker_prefetch_multiplier,
            task_default_queue=self.config.task_queue,
        )

        self._app = app
        self._task = eval_task

    def submit(
        self,
        fn: Callable[..., T],
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskHandle[T]:
        """
        Submit a task to Celery.

        Args:
            fn: The function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            context: Trace context (stored in task metadata)

        Returns:
            TaskHandle to track the task
        """
        kwargs = kwargs or {}
        task_id = str(uuid.uuid4())

        # Create handle
        handle = TaskHandle(
            task_id=task_id,
            backend_name=self.name,
            metadata={
                "queue": self.config.task_queue,
                "priority": self.config.task_priority,
                "function": fn.__name__ if hasattr(fn, "__name__") else str(fn),
                "context": context,
            },
        )
        handle._status = TaskStatus.PENDING

        with self._lock:
            self._handles[task_id] = handle

        try:
            # Submit task to Celery
            async_result = self._task.apply_async(
                args=(fn, args, kwargs),
                task_id=task_id,
                queue=self.config.task_queue,
                priority=self.config.task_priority,
            )

            with self._lock:
                self._async_results[task_id] = async_result
                handle._status = TaskStatus.PENDING

            logger.debug(f"Submitted task {task_id} to queue {self.config.task_queue}")

        except Exception as e:
            handle._status = TaskStatus.FAILED
            handle._error = str(e)
            logger.error(f"Failed to submit task {task_id}: {e}")

        return handle

    def get_result(
        self,
        handle: TaskHandle[T],
        timeout: Optional[float] = None,
    ) -> T:
        """
        Get the result of a Celery task.

        Args:
            handle: The task handle from submit()
            timeout: Maximum seconds to wait

        Returns:
            The task result

        Raises:
            TimeoutError: If timeout exceeded
            Exception: If task failed
        """
        timeout = timeout or self.config.timeout_seconds

        with self._lock:
            async_result = self._async_results.get(handle.task_id)

        if async_result is None:
            # Try to get result from backend
            from celery.result import AsyncResult
            async_result = AsyncResult(handle.task_id, app=self._app)

        try:
            result = async_result.get(timeout=timeout)

            with self._lock:
                if handle.task_id in self._handles:
                    self._handles[handle.task_id]._status = TaskStatus.COMPLETED
                    self._handles[handle.task_id]._result = result

            return result

        except Exception as e:
            error_name = type(e).__name__
            if "TimeoutError" in error_name or "TimeLimitExceeded" in error_name:
                with self._lock:
                    if handle.task_id in self._handles:
                        self._handles[handle.task_id]._status = TaskStatus.TIMEOUT
                raise TimeoutError(f"Task {handle.task_id} timed out after {timeout}s")

            with self._lock:
                if handle.task_id in self._handles:
                    self._handles[handle.task_id]._status = TaskStatus.FAILED
                    self._handles[handle.task_id]._error = str(e)
            raise

    def get_status(self, handle: TaskHandle) -> TaskStatus:
        """
        Get current status of a Celery task.

        Args:
            handle: The task handle

        Returns:
            Current TaskStatus
        """
        with self._lock:
            async_result = self._async_results.get(handle.task_id)

        if async_result is None:
            from celery.result import AsyncResult
            async_result = AsyncResult(handle.task_id, app=self._app)

        return self._map_celery_status(async_result.status)

    def _map_celery_status(self, celery_status: str) -> TaskStatus:
        """Map Celery task status to TaskStatus."""
        status_map = {
            "PENDING": TaskStatus.PENDING,
            "STARTED": TaskStatus.RUNNING,
            "SUCCESS": TaskStatus.COMPLETED,
            "FAILURE": TaskStatus.FAILED,
            "REVOKED": TaskStatus.CANCELLED,
            "RETRY": TaskStatus.RUNNING,
        }
        return status_map.get(celery_status, TaskStatus.PENDING)

    def cancel(self, handle: TaskHandle) -> bool:
        """
        Revoke/cancel a Celery task.

        Args:
            handle: The task handle

        Returns:
            True if revoked, False otherwise
        """
        try:
            with self._lock:
                async_result = self._async_results.get(handle.task_id)

            if async_result is None:
                from celery.result import AsyncResult
                async_result = AsyncResult(handle.task_id, app=self._app)

            async_result.revoke(terminate=True)

            with self._lock:
                if handle.task_id in self._handles:
                    self._handles[handle.task_id]._status = TaskStatus.CANCELLED

            logger.debug(f"Revoked task {handle.task_id}")
            return True

        except Exception as e:
            logger.warning(f"Failed to revoke task {handle.task_id}: {e}")
            return False

    def submit_batch(
        self,
        tasks: List[tuple],
    ) -> List[TaskHandle]:
        """
        Submit multiple tasks efficiently using Celery group.

        Args:
            tasks: List of (function, args, kwargs, context) tuples

        Returns:
            List of TaskHandles
        """
        from celery import group

        handles = []
        signatures = []

        for fn, args, kwargs, context in tasks:
            kwargs = kwargs or {}
            task_id = str(uuid.uuid4())

            handle = TaskHandle(
                task_id=task_id,
                backend_name=self.name,
                metadata={
                    "queue": self.config.task_queue,
                    "function": fn.__name__ if hasattr(fn, "__name__") else str(fn),
                    "context": context,
                },
            )
            handle._status = TaskStatus.PENDING
            handles.append(handle)

            with self._lock:
                self._handles[task_id] = handle

            sig = self._task.signature(
                args=(fn, args, kwargs),
                task_id=task_id,
                queue=self.config.task_queue,
            )
            signatures.append(sig)

        # Submit as a group
        try:
            job = group(signatures)
            group_result = job.apply_async()

            # Store individual results
            for handle, async_result in zip(handles, group_result.results):
                with self._lock:
                    self._async_results[handle.task_id] = async_result

        except Exception as e:
            logger.error(f"Failed to submit batch: {e}")
            for handle in handles:
                handle._status = TaskStatus.FAILED
                handle._error = str(e)

        return handles

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the Celery backend.

        Revokes only our tracked tasks, not the entire Celery queue.

        Args:
            wait: Whether to wait for pending tasks
        """
        if self._app is not None:
            with self._lock:
                task_ids = list(self._async_results.keys())

            for task_id in task_ids:
                try:
                    self._app.control.revoke(task_id, terminate=not wait)
                except Exception:
                    pass

            with self._lock:
                self._handles.clear()
                self._async_results.clear()

            logger.info(f"Celery backend shut down, revoked {len(task_ids)} tasks")

    def get_queue_length(self) -> int:
        """
        Get the number of tasks waiting in the queue.

        Returns:
            Number of pending tasks in the queue
        """
        try:
            with self._app.pool.acquire(block=True) as conn:
                return conn.default_channel.client.llen(self.config.task_queue)
        except Exception:
            return -1

    def get_stats(self) -> dict:
        """
        Get Celery backend statistics.

        Returns:
            Dictionary with queue and worker stats
        """
        stats = {
            "queue": self.config.task_queue,
            "broker": self.config.broker_url,
            "pending_tasks": len([
                h for h in self._handles.values()
                if h._status == TaskStatus.PENDING
            ]),
            "running_tasks": len([
                h for h in self._handles.values()
                if h._status == TaskStatus.RUNNING
            ]),
        }

        try:
            stats["queue_length"] = self.get_queue_length()
        except Exception:
            stats["queue_length"] = -1

        return stats
