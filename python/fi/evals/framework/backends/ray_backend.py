"""
Ray backend for distributed computing.

Provides distributed task execution using Ray for batch evaluations.
Requires: pip install 'ray[default]'
"""

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

from .base import Backend, BackendConfig, TaskHandle, TaskStatus
from ._utils import RAY

T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass
class RayConfig(BackendConfig):
    """
    Configuration for Ray backend.

    Attributes:
        address: Ray cluster address ('auto', 'local', or specific address)
        num_cpus: Default CPUs per task (None = auto)
        num_gpus: Default GPUs per task
        memory: Memory limit per task in bytes (None = auto)
        runtime_env: Runtime environment for tasks
        namespace: Ray namespace for isolation
        ignore_reinit_error: Ignore error if Ray already initialized
        log_to_driver: Send task logs to driver
        max_retries: Max retries for failed tasks
        retry_exceptions: Whether to retry on exceptions
    """

    address: str = "auto"
    num_cpus: Optional[float] = None
    num_gpus: float = 0.0
    memory: Optional[int] = None
    runtime_env: Optional[Dict[str, Any]] = None
    namespace: Optional[str] = None
    ignore_reinit_error: bool = True
    log_to_driver: bool = True
    max_retries: int = 3
    retry_exceptions: bool = True


class RayBackend(Backend):
    """
    Ray backend for distributed computing.

    Uses Ray for distributed task execution across a cluster.
    Optimized for batch processing with auto-scaling.

    Example:
        config = RayConfig(
            address="auto",  # Connect to existing cluster
            num_cpus=2.0,
            num_gpus=0.5,
        )
        backend = RayBackend(config)

        handle = backend.submit(my_eval_func, args=(input_data,))
        result = backend.get_result(handle)

        # Or batch submission
        handles = backend.submit_batch([
            (eval_func, (data1,), {}, None),
            (eval_func, (data2,), {}, None),
        ])

    Note:
        Ray will start a local cluster if none exists when address='auto'.
    """

    name: str = "ray"

    def __init__(self, config: Optional[RayConfig] = None):
        """
        Initialize Ray backend.

        Args:
            config: Ray configuration

        Raises:
            ImportError: If ray is not installed
        """
        RAY.require()

        self.config = config or RayConfig()
        self._initialized = False
        self._handles: Dict[str, TaskHandle] = {}
        self._object_refs: Dict[str, Any] = {}
        self._lock = threading.Lock()

        self._init_ray()

    def _init_ray(self) -> None:
        """Initialize Ray runtime."""
        import ray

        init_kwargs = {
            "ignore_reinit_error": self.config.ignore_reinit_error,
            "log_to_driver": self.config.log_to_driver,
        }

        if self.config.address:
            init_kwargs["address"] = self.config.address

        if self.config.namespace:
            init_kwargs["namespace"] = self.config.namespace

        if self.config.runtime_env:
            init_kwargs["runtime_env"] = self.config.runtime_env

        ray.init(**init_kwargs)
        self._initialized = True
        logger.info(f"Ray initialized with address={self.config.address}")

    def submit(
        self,
        fn: Callable[..., T],
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskHandle[T]:
        """
        Submit a task to Ray.

        Args:
            fn: The function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            context: Trace context (stored in metadata)

        Returns:
            TaskHandle to track the task
        """
        import ray

        kwargs = kwargs or {}
        task_id = str(uuid.uuid4())

        # Create handle
        handle = TaskHandle(
            task_id=task_id,
            backend_name=self.name,
            metadata={
                "function": fn.__name__ if hasattr(fn, "__name__") else str(fn),
                "context": context,
                "num_cpus": self.config.num_cpus,
                "num_gpus": self.config.num_gpus,
            },
        )
        handle._status = TaskStatus.PENDING

        with self._lock:
            self._handles[task_id] = handle

        try:
            # Create Ray remote function with resources
            remote_options = {
                "max_retries": self.config.max_retries,
                "retry_exceptions": self.config.retry_exceptions,
            }

            if self.config.num_cpus is not None:
                remote_options["num_cpus"] = self.config.num_cpus

            if self.config.num_gpus > 0:
                remote_options["num_gpus"] = self.config.num_gpus

            if self.config.memory is not None:
                remote_options["memory"] = self.config.memory

            # Create remote function and submit
            remote_fn = ray.remote(**remote_options)(fn)
            object_ref = remote_fn.remote(*args, **kwargs)

            with self._lock:
                self._object_refs[task_id] = object_ref
                handle._status = TaskStatus.RUNNING

            logger.debug(f"Submitted task {task_id} to Ray")

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
        Get the result of a Ray task.

        Args:
            handle: The task handle from submit()
            timeout: Maximum seconds to wait

        Returns:
            The task result

        Raises:
            TimeoutError: If timeout exceeded
            Exception: If task failed
        """
        import ray

        timeout = timeout or self.config.timeout_seconds

        with self._lock:
            object_ref = self._object_refs.get(handle.task_id)

        if object_ref is None:
            raise ValueError(f"Unknown task: {handle.task_id}")

        try:
            result = ray.get(object_ref, timeout=timeout)

            with self._lock:
                if handle.task_id in self._handles:
                    self._handles[handle.task_id]._status = TaskStatus.COMPLETED
                    self._handles[handle.task_id]._result = result

            return result

        except ray.exceptions.GetTimeoutError:
            with self._lock:
                if handle.task_id in self._handles:
                    self._handles[handle.task_id]._status = TaskStatus.TIMEOUT
            raise TimeoutError(f"Task {handle.task_id} timed out after {timeout}s")

        except Exception as e:
            with self._lock:
                if handle.task_id in self._handles:
                    self._handles[handle.task_id]._status = TaskStatus.FAILED
                    self._handles[handle.task_id]._error = str(e)
            raise

    def get_status(self, handle: TaskHandle) -> TaskStatus:
        """
        Get current status of a Ray task.

        Args:
            handle: The task handle

        Returns:
            Current TaskStatus
        """
        import ray

        with self._lock:
            if handle.task_id in self._handles:
                cached_status = self._handles[handle.task_id]._status
                if cached_status in (TaskStatus.COMPLETED, TaskStatus.FAILED,
                                     TaskStatus.CANCELLED, TaskStatus.TIMEOUT):
                    return cached_status

            object_ref = self._object_refs.get(handle.task_id)

        if object_ref is None:
            return TaskStatus.FAILED

        # Check if task is ready (completed)
        ready, _ = ray.wait([object_ref], timeout=0)
        if ready:
            # Task is done, check if it succeeded
            try:
                ray.get(object_ref, timeout=0)
                return TaskStatus.COMPLETED
            except Exception:
                return TaskStatus.FAILED

        return TaskStatus.RUNNING

    def cancel(self, handle: TaskHandle) -> bool:
        """
        Cancel a Ray task.

        Args:
            handle: The task handle

        Returns:
            True if cancelled, False otherwise
        """
        import ray

        with self._lock:
            object_ref = self._object_refs.get(handle.task_id)

        if object_ref is None:
            return False

        try:
            ray.cancel(object_ref, force=True)

            with self._lock:
                if handle.task_id in self._handles:
                    self._handles[handle.task_id]._status = TaskStatus.CANCELLED

            logger.debug(f"Cancelled task {handle.task_id}")
            return True

        except Exception as e:
            logger.warning(f"Failed to cancel task {handle.task_id}: {e}")
            return False

    def submit_batch(
        self,
        tasks: List[tuple],
    ) -> List[TaskHandle]:
        """
        Submit multiple tasks efficiently to Ray.

        Uses Ray's ability to schedule multiple tasks in parallel.

        Args:
            tasks: List of (function, args, kwargs, context) tuples

        Returns:
            List of TaskHandles
        """
        import ray

        handles = []
        refs_to_submit = []

        # Prepare all tasks
        for fn, args, kwargs, context in tasks:
            kwargs = kwargs or {}
            task_id = str(uuid.uuid4())

            handle = TaskHandle(
                task_id=task_id,
                backend_name=self.name,
                metadata={
                    "function": fn.__name__ if hasattr(fn, "__name__") else str(fn),
                    "context": context,
                },
            )
            handle._status = TaskStatus.PENDING
            handles.append(handle)

            with self._lock:
                self._handles[task_id] = handle

            # Create remote function
            remote_options = {
                "max_retries": self.config.max_retries,
            }
            if self.config.num_cpus is not None:
                remote_options["num_cpus"] = self.config.num_cpus
            if self.config.num_gpus > 0:
                remote_options["num_gpus"] = self.config.num_gpus

            remote_fn = ray.remote(**remote_options)(fn)
            refs_to_submit.append((task_id, remote_fn, args, kwargs))

        # Submit all tasks
        for task_id, remote_fn, args, kwargs in refs_to_submit:
            try:
                object_ref = remote_fn.remote(*args, **kwargs)
                with self._lock:
                    self._object_refs[task_id] = object_ref
                    self._handles[task_id]._status = TaskStatus.RUNNING
            except Exception as e:
                with self._lock:
                    self._handles[task_id]._status = TaskStatus.FAILED
                    self._handles[task_id]._error = str(e)

        return handles

    def get_batch_results(
        self,
        handles: List[TaskHandle],
        timeout: Optional[float] = None,
    ) -> List[T]:
        """
        Get results for multiple tasks efficiently.

        Uses ray.get() on multiple object refs for better performance.

        Args:
            handles: List of task handles
            timeout: Maximum seconds to wait for all results

        Returns:
            List of results in same order as handles
        """
        import ray

        timeout = timeout or self.config.timeout_seconds

        with self._lock:
            object_refs = [
                self._object_refs.get(h.task_id)
                for h in handles
            ]

        # Filter out None refs
        valid_refs = [(h, ref) for h, ref in zip(handles, object_refs) if ref is not None]

        if not valid_refs:
            return [None] * len(handles)

        try:
            refs_only = [ref for _, ref in valid_refs]
            results = ray.get(refs_only, timeout=timeout)

            # Update handles and build result list
            result_map = {}
            for (handle, _), result in zip(valid_refs, results):
                with self._lock:
                    if handle.task_id in self._handles:
                        self._handles[handle.task_id]._status = TaskStatus.COMPLETED
                        self._handles[handle.task_id]._result = result
                result_map[handle.task_id] = result

            return [result_map.get(h.task_id) for h in handles]

        except ray.exceptions.GetTimeoutError:
            for handle in handles:
                with self._lock:
                    if handle.task_id in self._handles:
                        if self._handles[handle.task_id]._status == TaskStatus.RUNNING:
                            self._handles[handle.task_id]._status = TaskStatus.TIMEOUT
            raise TimeoutError(f"Batch get timed out after {timeout}s")

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the Ray backend.

        Args:
            wait: Whether to wait for pending tasks
        """
        import ray

        if self._initialized:
            if not wait:
                # Cancel all pending tasks
                with self._lock:
                    for task_id, object_ref in self._object_refs.items():
                        try:
                            ray.cancel(object_ref, force=True)
                        except Exception:
                            pass

            ray.shutdown()
            self._initialized = False
            logger.info("Ray backend shut down")

    def get_cluster_resources(self) -> dict:
        """
        Get available cluster resources.

        Returns:
            Dictionary with CPU, GPU, memory info
        """
        import ray

        resources = ray.cluster_resources()
        available = ray.available_resources()

        return {
            "total": resources,
            "available": available,
            "used": {
                k: resources.get(k, 0) - available.get(k, 0)
                for k in resources
            },
        }

    def get_stats(self) -> dict:
        """
        Get Ray backend statistics.

        Returns:
            Dictionary with task and resource stats
        """
        with self._lock:
            pending = len([
                h for h in self._handles.values()
                if h._status == TaskStatus.PENDING
            ])
            running = len([
                h for h in self._handles.values()
                if h._status == TaskStatus.RUNNING
            ])
            completed = len([
                h for h in self._handles.values()
                if h._status == TaskStatus.COMPLETED
            ])
            failed = len([
                h for h in self._handles.values()
                if h._status == TaskStatus.FAILED
            ])

        stats = {
            "address": self.config.address,
            "pending_tasks": pending,
            "running_tasks": running,
            "completed_tasks": completed,
            "failed_tasks": failed,
            "total_tasks": len(self._handles),
        }

        try:
            resources = self.get_cluster_resources()
            stats["cluster_resources"] = resources
        except Exception:
            pass

        return stats
