"""
Temporal backend for durable workflow execution.

Provides distributed, fault-tolerant task execution using Temporal.io.
Requires: pip install temporalio
"""

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, TypeVar

from .base import Backend, BackendConfig, TaskHandle, TaskStatus
from ._utils import TEMPORAL

T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass
class TemporalConfig(BackendConfig):
    """
    Configuration for Temporal backend.

    Attributes:
        host: Temporal server address (host:port)
        namespace: Temporal namespace
        task_queue: Task queue name for workers
        workflow_id_prefix: Prefix for generated workflow IDs
        execution_timeout_seconds: Maximum workflow execution time
        task_timeout_seconds: Maximum time for a single task
        retry_policy_max_attempts: Max retry attempts for activities
        retry_policy_initial_interval: Initial retry interval in seconds
        retry_policy_backoff_coefficient: Backoff multiplier
        identity: Worker identity string
    """

    host: str = "localhost:7233"
    namespace: str = "default"
    task_queue: str = "eval-tasks"
    workflow_id_prefix: str = "eval-"
    execution_timeout_seconds: float = 3600.0  # 1 hour
    task_timeout_seconds: float = 300.0  # 5 minutes
    retry_policy_max_attempts: int = 3
    retry_policy_initial_interval: float = 1.0
    retry_policy_backoff_coefficient: float = 2.0
    identity: Optional[str] = None


class TemporalBackend(Backend):
    """
    Temporal backend for durable workflow execution.

    Uses Temporal.io for distributed, fault-tolerant task execution.
    Workflows survive process restarts and infrastructure failures.

    Example:
        config = TemporalConfig(
            host="temporal.example.com:7233",
            namespace="evaluations",
            task_queue="eval-tasks",
        )
        backend = TemporalBackend(config)

        handle = backend.submit(my_eval_func, args=(input_data,))
        result = backend.get_result(handle)

    Note:
        Requires a running Temporal server and worker processes.
        This backend submits workflows - workers must be running
        to execute them.
    """

    name: str = "temporal"

    def __init__(self, config: Optional[TemporalConfig] = None):
        """
        Initialize Temporal backend.

        Args:
            config: Temporal configuration

        Raises:
            ImportError: If temporalio is not installed
        """
        TEMPORAL.require()

        self.config = config or TemporalConfig()
        self._client: Optional[Any] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._handles: Dict[str, TaskHandle] = {}
        self._lock = threading.Lock()

    def _ensure_client(self) -> Any:
        """Ensure Temporal client is connected."""
        if self._client is None:
            self._setup_event_loop()
            self._client = self._run_async(self._connect())
        return self._client

    def _setup_event_loop(self) -> None:
        """Set up dedicated event loop for async operations."""
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self._loop.run_forever,
                daemon=True,
                name="temporal-event-loop",
            )
            self._loop_thread.start()

    def _run_async(self, coro) -> Any:
        """Run async coroutine in the dedicated event loop."""
        if self._loop is None:
            self._setup_event_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=60)

    async def _connect(self) -> Any:
        """Connect to Temporal server."""
        from temporalio.client import Client

        client = await Client.connect(
            self.config.host,
            namespace=self.config.namespace,
        )
        logger.info(f"Connected to Temporal at {self.config.host}")
        return client

    def submit(
        self,
        fn: Callable[..., T],
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskHandle[T]:
        """
        Submit a task as a Temporal workflow.

        Args:
            fn: The function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            context: Trace context for propagation

        Returns:
            TaskHandle to track the workflow
        """
        kwargs = kwargs or {}
        client = self._ensure_client()

        workflow_id = f"{self.config.workflow_id_prefix}{uuid.uuid4().hex[:12]}"

        # Create handle before starting workflow
        handle = TaskHandle(
            task_id=workflow_id,
            backend_name=self.name,
            metadata={
                "namespace": self.config.namespace,
                "task_queue": self.config.task_queue,
                "function": fn.__name__ if hasattr(fn, "__name__") else str(fn),
                "context": context,
            },
        )
        handle._status = TaskStatus.PENDING

        with self._lock:
            self._handles[workflow_id] = handle

        # Start workflow asynchronously
        try:
            workflow_handle = self._run_async(
                self._start_workflow(client, workflow_id, fn, args, kwargs)
            )
            handle._status = TaskStatus.RUNNING
            handle.metadata["workflow_run_id"] = workflow_handle.run_id
        except Exception as e:
            handle._status = TaskStatus.FAILED
            handle._error = str(e)
            logger.error(f"Failed to start workflow {workflow_id}: {e}")

        return handle

    async def _start_workflow(
        self,
        client: Any,
        workflow_id: str,
        fn: Callable,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """Start a Temporal workflow with cloudpickle-serialized payload."""
        from temporalio.client import WorkflowHandle

        from .temporal_worker import serialize_task

        payload_b64 = serialize_task(fn, args, kwargs)

        handle: WorkflowHandle = await client.start_workflow(
            "EvalTaskWorkflow",
            payload_b64,
            id=workflow_id,
            task_queue=self.config.task_queue,
            execution_timeout=timedelta(seconds=self.config.execution_timeout_seconds),
        )

        return handle

    def get_result(
        self,
        handle: TaskHandle[T],
        timeout: Optional[float] = None,
    ) -> T:
        """
        Get the result of a workflow.

        Args:
            handle: The task handle from submit()
            timeout: Maximum seconds to wait

        Returns:
            The workflow result

        Raises:
            TimeoutError: If timeout exceeded
            Exception: If workflow failed
        """
        client = self._ensure_client()
        timeout = timeout or self.config.timeout_seconds

        try:
            result = self._run_async(
                self._get_workflow_result(client, handle.task_id, timeout)
            )
            with self._lock:
                if handle.task_id in self._handles:
                    self._handles[handle.task_id]._status = TaskStatus.COMPLETED
                    self._handles[handle.task_id]._result = result
            return result
        except asyncio.TimeoutError:
            with self._lock:
                if handle.task_id in self._handles:
                    self._handles[handle.task_id]._status = TaskStatus.TIMEOUT
            raise TimeoutError(f"Workflow {handle.task_id} timed out after {timeout}s")
        except Exception as e:
            with self._lock:
                if handle.task_id in self._handles:
                    self._handles[handle.task_id]._status = TaskStatus.FAILED
                    self._handles[handle.task_id]._error = str(e)
            raise

    async def _get_workflow_result(
        self, client: Any, workflow_id: str, timeout: float
    ) -> Any:
        """Get workflow result with timeout, deserializing cloudpickle payload."""
        from .temporal_worker import deserialize_result

        handle = client.get_workflow_handle(workflow_id)
        result_b64 = await asyncio.wait_for(handle.result(), timeout=timeout)
        return deserialize_result(result_b64)

    def get_status(self, handle: TaskHandle) -> TaskStatus:
        """
        Get current status of a workflow.

        Args:
            handle: The task handle

        Returns:
            Current TaskStatus
        """
        with self._lock:
            if handle.task_id in self._handles:
                return self._handles[handle.task_id]._status

        # Query Temporal for actual status
        try:
            client = self._ensure_client()
            description = self._run_async(
                self._describe_workflow(client, handle.task_id)
            )
            return self._map_workflow_status(description.status)
        except Exception:
            return TaskStatus.FAILED

    async def _describe_workflow(self, client: Any, workflow_id: str) -> Any:
        """Describe a workflow execution."""
        handle = client.get_workflow_handle(workflow_id)
        return await handle.describe()

    def _map_workflow_status(self, temporal_status: Any) -> TaskStatus:
        """Map Temporal workflow status to TaskStatus."""
        from temporalio.client import WorkflowExecutionStatus

        status_map = {
            WorkflowExecutionStatus.RUNNING: TaskStatus.RUNNING,
            WorkflowExecutionStatus.COMPLETED: TaskStatus.COMPLETED,
            WorkflowExecutionStatus.FAILED: TaskStatus.FAILED,
            WorkflowExecutionStatus.CANCELED: TaskStatus.CANCELLED,
            WorkflowExecutionStatus.TERMINATED: TaskStatus.CANCELLED,
            WorkflowExecutionStatus.TIMED_OUT: TaskStatus.TIMEOUT,
        }
        return status_map.get(temporal_status, TaskStatus.PENDING)

    def cancel(self, handle: TaskHandle) -> bool:
        """
        Cancel a workflow.

        Args:
            handle: The task handle

        Returns:
            True if cancelled, False otherwise
        """
        try:
            client = self._ensure_client()
            self._run_async(self._cancel_workflow(client, handle.task_id))
            with self._lock:
                if handle.task_id in self._handles:
                    self._handles[handle.task_id]._status = TaskStatus.CANCELLED
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel workflow {handle.task_id}: {e}")
            return False

    async def _cancel_workflow(self, client: Any, workflow_id: str) -> None:
        """Cancel a workflow execution."""
        handle = client.get_workflow_handle(workflow_id)
        await handle.cancel()

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the Temporal backend.

        Args:
            wait: Whether to wait for pending workflows (not implemented)
        """
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=5)
            self._loop = None
            self._loop_thread = None

        self._client = None
        logger.info("Temporal backend shut down")
