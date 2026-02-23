"""
Kubernetes backend for cloud-native job execution.

Provides distributed task execution using Kubernetes Jobs.
Each task is submitted as a K8s Job that runs a container which
deserializes and executes the function, then writes JSON results to stdout.

Requires: pip install kubernetes cloudpickle
"""

import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

from .base import Backend, BackendConfig, TaskHandle, TaskStatus
from ._utils import KUBERNETES
from ._container import (
    DEFAULT_IMAGE,
    EVAL_PAYLOAD_ENV,
    RUNNER_COMMAND,
    RUNNER_SCRIPT,
    parse_result_from_logs,
    serialize_task,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass
class KubernetesConfig(BackendConfig):
    """
    Configuration for Kubernetes backend.

    A pre-built eval runner Dockerfile is provided at
    ``fi/evals/framework/backends/Dockerfile.eval-runner``.
    Build it with::

        docker build -f Dockerfile.eval-runner -t fi-eval-runner:latest .

    Then pass ``image="fi-eval-runner:latest"`` (or your registry path).

    Attributes:
        namespace: Kubernetes namespace for Jobs
        image: Container image (must have cloudpickle installed)
        cpu_request: CPU request per Job pod
        cpu_limit: CPU limit per Job pod
        memory_request: Memory request per Job pod
        memory_limit: Memory limit per Job pod
        job_prefix: Prefix for Job names (DNS-1123 compliant)
        backoff_limit: Number of retries for failed Jobs
        active_deadline_seconds: Maximum seconds a Job may run
        ttl_seconds_after_finished: Seconds to keep finished Jobs before cleanup
        service_account_name: K8s service account for the Job pod
        image_pull_secrets: Names of image pull secrets
        image_pull_policy: Image pull policy (Always, IfNotPresent, Never)
        kubeconfig_path: Path to kubeconfig file (None = auto-detect)
        context: Kubeconfig context to use (None = current context)
        in_cluster: Force in-cluster config loading
        labels: Extra labels to apply to Jobs
        annotations: Extra annotations to apply to Jobs
        poll_interval: Seconds between status polls
        log_tail_lines: Number of log lines to tail (None = all)
    """

    namespace: str = "default"
    image: str = "fi-eval-runner:latest"
    cpu_request: str = "500m"
    cpu_limit: str = "1"
    memory_request: str = "512Mi"
    memory_limit: str = "2Gi"
    job_prefix: str = "eval-"
    backoff_limit: int = 0
    active_deadline_seconds: int = 600
    ttl_seconds_after_finished: int = 300
    service_account_name: Optional[str] = None
    image_pull_secrets: Optional[List[str]] = None
    image_pull_policy: str = "IfNotPresent"
    kubeconfig_path: Optional[str] = None
    context: Optional[str] = None
    in_cluster: Optional[bool] = None
    labels: Optional[Dict[str, str]] = None
    annotations: Optional[Dict[str, str]] = None
    poll_interval: float = 2.0
    log_tail_lines: Optional[int] = None


class KubernetesBackend(Backend):
    """
    Kubernetes backend for cloud-native job execution.

    Submits evaluation tasks as Kubernetes Jobs. Each task is serialized
    with cloudpickle, base64-encoded, and passed to the container as an env var.
    Results are read from pod logs as JSON.

    Example:
        config = KubernetesConfig(
            namespace="evaluations",
            image="my-eval-image:latest",
            cpu_request="1",
            memory_limit="4Gi",
        )
        backend = KubernetesBackend(config)

        handle = backend.submit(my_eval_func, args=(input_data,))
        result = backend.get_result(handle)

        # Or batch submission
        handles = backend.submit_batch([
            (eval_func, (data1,), {}, None),
            (eval_func, (data2,), {}, None),
        ])

    Note:
        Requires a running Kubernetes cluster. Uses in-cluster config
        when running inside a pod, or falls back to kubeconfig for local dev.
    """

    name: str = "kubernetes"

    def __init__(self, config: Optional[KubernetesConfig] = None):
        """
        Initialize Kubernetes backend.

        Args:
            config: Kubernetes configuration

        Raises:
            ImportError: If kubernetes package is not installed
        """
        KUBERNETES.require()

        self.config = config or KubernetesConfig()
        self._batch_api: Optional[Any] = None
        self._core_api: Optional[Any] = None
        self._handles: Dict[str, TaskHandle] = {}
        self._job_names: Dict[str, str] = {}
        self._lock = threading.Lock()

        self._setup_kubernetes()

    def _setup_kubernetes(self) -> None:
        """Load kubeconfig and create API clients."""
        from kubernetes import client, config

        if self.config.in_cluster is True:
            config.load_incluster_config()
        elif self.config.in_cluster is False:
            config.load_kube_config(
                config_file=self.config.kubeconfig_path,
                context=self.config.context,
            )
        else:
            # Auto-detect: try in-cluster first, fall back to kubeconfig
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config(
                    config_file=self.config.kubeconfig_path,
                    context=self.config.context,
                )

        self._batch_api = client.BatchV1Api()
        self._core_api = client.CoreV1Api()
        logger.info(
            f"Kubernetes backend initialized (namespace={self.config.namespace})"
        )

    def _make_job_name(self, task_id: str) -> str:
        """
        Generate a DNS-1123 compliant Job name.

        The name is ``{job_prefix}{uuid_hex[:12]}``, lowercased and
        stripped of any characters that are not lowercase alphanumeric
        or hyphens. The total length is capped at 63 characters.
        """
        short_id = uuid.uuid4().hex[:12]
        raw = f"{self.config.job_prefix}{short_id}"
        # DNS-1123: lowercase, alphanumeric and hyphens only
        name = re.sub(r"[^a-z0-9-]", "", raw.lower())
        # Must start/end with alphanumeric
        name = name.strip("-")
        return name[:63]

    def _build_job_manifest(
        self, job_name: str, serialized_payload: str
    ) -> Any:
        """Build a V1Job manifest for the given payload."""
        from kubernetes import client

        labels = {"app": "fi-eval", "job-name": job_name}
        if self.config.labels:
            labels.update(self.config.labels)

        annotations = self.config.annotations or {}

        # Resource requirements
        resources = client.V1ResourceRequirements(
            requests={
                "cpu": self.config.cpu_request,
                "memory": self.config.memory_request,
            },
            limits={
                "cpu": self.config.cpu_limit,
                "memory": self.config.memory_limit,
            },
        )

        container = client.V1Container(
            name="eval-runner",
            image=self.config.image,
            image_pull_policy=self.config.image_pull_policy,
            command=RUNNER_COMMAND,
            env=[
                client.V1EnvVar(
                    name=EVAL_PAYLOAD_ENV, value=serialized_payload
                ),
            ],
            resources=resources,
        )

        # Image pull secrets
        image_pull_secrets = None
        if self.config.image_pull_secrets:
            image_pull_secrets = [
                client.V1LocalObjectReference(name=s)
                for s in self.config.image_pull_secrets
            ]

        pod_spec = client.V1PodSpec(
            containers=[container],
            restart_policy="Never",
            service_account_name=self.config.service_account_name,
            image_pull_secrets=image_pull_secrets,
        )

        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels=labels, annotations=annotations),
            spec=pod_spec,
        )

        job_spec = client.V1JobSpec(
            template=template,
            backoff_limit=self.config.backoff_limit,
            active_deadline_seconds=self.config.active_deadline_seconds,
            ttl_seconds_after_finished=self.config.ttl_seconds_after_finished,
        )

        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_name,
                namespace=self.config.namespace,
                labels=labels,
                annotations=annotations,
            ),
            spec=job_spec,
        )

        return job

    def submit(
        self,
        fn: Callable[..., T],
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskHandle[T]:
        """
        Submit a task as a Kubernetes Job.

        The function and its arguments are serialized with cloudpickle,
        base64-encoded, and passed to the container as an env var.

        Args:
            fn: The function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            context: Trace context (stored in metadata)

        Returns:
            TaskHandle to track the task
        """
        kwargs = kwargs or {}
        task_id = str(uuid.uuid4())
        job_name = self._make_job_name(task_id)

        handle = TaskHandle(
            task_id=task_id,
            backend_name=self.name,
            metadata={
                "function": fn.__name__ if hasattr(fn, "__name__") else str(fn),
                "context": context,
                "job_name": job_name,
                "namespace": self.config.namespace,
            },
        )
        handle._status = TaskStatus.PENDING

        with self._lock:
            self._handles[task_id] = handle
            self._job_names[task_id] = job_name

        try:
            serialized = serialize_task(fn, args, kwargs)

            # Build and create the Job
            job_manifest = self._build_job_manifest(job_name, serialized)
            self._batch_api.create_namespaced_job(
                namespace=self.config.namespace,
                body=job_manifest,
            )

            with self._lock:
                handle._status = TaskStatus.RUNNING

            logger.debug(
                f"Submitted task {task_id} as Job {job_name} "
                f"in namespace {self.config.namespace}"
            )

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
        Get the result of a Kubernetes Job by polling until completion.

        Args:
            handle: The task handle from submit()
            timeout: Maximum seconds to wait

        Returns:
            The task result

        Raises:
            TimeoutError: If timeout exceeded
            RuntimeError: If the Job failed
            ValueError: If task is unknown
        """
        timeout = timeout or self.config.timeout_seconds

        with self._lock:
            job_name = self._job_names.get(handle.task_id)

        if job_name is None:
            raise ValueError(f"Unknown task: {handle.task_id}")

        deadline = time.monotonic() + timeout
        while True:
            status = self._poll_job_status(job_name)

            if status == TaskStatus.COMPLETED:
                result = self._read_job_result(job_name)
                with self._lock:
                    if handle.task_id in self._handles:
                        self._handles[handle.task_id]._status = TaskStatus.COMPLETED
                        self._handles[handle.task_id]._result = result
                return result

            if status == TaskStatus.FAILED:
                error_msg = f"Job {job_name} failed"
                # Try to read logs for error details
                try:
                    self._read_job_result(job_name)
                except RuntimeError as re_err:
                    error_msg = str(re_err)
                except Exception:
                    pass
                with self._lock:
                    if handle.task_id in self._handles:
                        self._handles[handle.task_id]._status = TaskStatus.FAILED
                        self._handles[handle.task_id]._error = error_msg
                raise RuntimeError(error_msg)

            if time.monotonic() > deadline:
                with self._lock:
                    if handle.task_id in self._handles:
                        self._handles[handle.task_id]._status = TaskStatus.TIMEOUT
                raise TimeoutError(
                    f"Task {handle.task_id} timed out after {timeout}s"
                )

            time.sleep(self.config.poll_interval)

    def _poll_job_status(self, job_name: str) -> TaskStatus:
        """Poll the K8s API for the Job's current status."""
        job = self._batch_api.read_namespaced_job_status(
            name=job_name,
            namespace=self.config.namespace,
        )
        return self._map_job_status(job.status)

    def _map_job_status(self, status: Any) -> TaskStatus:
        """Map a V1JobStatus object to TaskStatus."""
        # Check conditions first (Complete / Failed)
        if status.conditions:
            for condition in status.conditions:
                if condition.type == "Complete" and condition.status == "True":
                    return TaskStatus.COMPLETED
                if condition.type == "Failed" and condition.status == "True":
                    return TaskStatus.FAILED

        # If pods are still active, the job is running
        if status.active and status.active > 0:
            return TaskStatus.RUNNING

        # Succeeded / failed counts
        if status.succeeded and status.succeeded > 0:
            return TaskStatus.COMPLETED
        if status.failed and status.failed > 0:
            return TaskStatus.FAILED

        return TaskStatus.PENDING

    def _read_job_result(self, job_name: str) -> Any:
        """
        Read the result from the Job's pod logs.

        Uses :func:`_container.parse_result_from_logs` to extract the
        JSON result printed by the runner script.
        """
        from kubernetes.client.rest import ApiException

        # Find pods belonging to this Job
        label_selector = f"job-name={job_name}"
        pods = self._core_api.list_namespaced_pod(
            namespace=self.config.namespace,
            label_selector=label_selector,
        )

        if not pods.items:
            raise RuntimeError(f"No pods found for Job {job_name}")

        pod_name = pods.items[0].metadata.name

        log_kwargs = {"name": pod_name, "namespace": self.config.namespace}
        if self.config.log_tail_lines is not None:
            log_kwargs["tail_lines"] = self.config.log_tail_lines

        try:
            logs = self._core_api.read_namespaced_pod_log(**log_kwargs)
        except ApiException as e:
            raise RuntimeError(f"Failed to read logs for pod {pod_name}: {e}")

        return parse_result_from_logs(logs)

    def get_status(self, handle: TaskHandle) -> TaskStatus:
        """
        Get current status of a Kubernetes Job.

        Returns cached terminal status if available, otherwise polls K8s.

        Args:
            handle: The task handle

        Returns:
            Current TaskStatus
        """
        with self._lock:
            if handle.task_id in self._handles:
                cached = self._handles[handle.task_id]._status
                if cached in (
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                    TaskStatus.TIMEOUT,
                ):
                    return cached

            job_name = self._job_names.get(handle.task_id)

        if job_name is None:
            return TaskStatus.FAILED

        try:
            status = self._poll_job_status(job_name)
            with self._lock:
                if handle.task_id in self._handles:
                    self._handles[handle.task_id]._status = status
            return status
        except Exception:
            return TaskStatus.FAILED

    def cancel(self, handle: TaskHandle) -> bool:
        """
        Cancel a Kubernetes Job.

        Deletes the Job with foreground propagation policy so
        associated pods are terminated immediately.

        Args:
            handle: The task handle

        Returns:
            True if cancelled, False otherwise
        """
        from kubernetes.client.rest import ApiException

        with self._lock:
            job_name = self._job_names.get(handle.task_id)

        if job_name is None:
            return False

        try:
            from kubernetes import client

            self._batch_api.delete_namespaced_job(
                name=job_name,
                namespace=self.config.namespace,
                body=client.V1DeleteOptions(
                    propagation_policy="Foreground",
                ),
            )

            with self._lock:
                if handle.task_id in self._handles:
                    self._handles[handle.task_id]._status = TaskStatus.CANCELLED

            logger.debug(f"Cancelled Job {job_name}")
            return True

        except ApiException as e:
            logger.warning(f"Failed to cancel Job {job_name}: {e}")
            return False

    def submit_batch(
        self,
        tasks: List[tuple],
    ) -> List[TaskHandle]:
        """
        Submit multiple tasks as Kubernetes Jobs.

        Submits tasks sequentially (each creates an independent K8s Job).

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
        Shutdown the Kubernetes backend.

        Args:
            wait: If False, cancels all pending/running tasks
        """
        if not wait:
            with self._lock:
                pending_tasks = [
                    (tid, jn)
                    for tid, jn in self._job_names.items()
                    if tid in self._handles
                    and self._handles[tid]._status
                    in (TaskStatus.PENDING, TaskStatus.RUNNING)
                ]

            for task_id, _job_name in pending_tasks:
                try:
                    handle = TaskHandle(task_id=task_id, backend_name=self.name)
                    self.cancel(handle)
                except Exception:
                    pass

        logger.info("Kubernetes backend shut down")

    def get_job_logs(self, handle: TaskHandle) -> Dict[str, str]:
        """
        Get raw logs for all pods of a Job.

        Args:
            handle: The task handle

        Returns:
            Dict mapping pod name to log text

        Raises:
            ValueError: If task is unknown
        """
        from kubernetes.client.rest import ApiException

        with self._lock:
            job_name = self._job_names.get(handle.task_id)

        if job_name is None:
            raise ValueError(f"Unknown task: {handle.task_id}")

        label_selector = f"job-name={job_name}"
        pods = self._core_api.list_namespaced_pod(
            namespace=self.config.namespace,
            label_selector=label_selector,
        )

        result: Dict[str, str] = {}
        for pod in pods.items:
            pod_name = pod.metadata.name
            try:
                logs = self._core_api.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=self.config.namespace,
                )
                result[pod_name] = logs
            except ApiException as e:
                result[pod_name] = f"<error reading logs: {e}>"

        return result

    def get_stats(self) -> dict:
        """
        Get Kubernetes backend statistics.

        Returns:
            Dictionary with task counts and config info
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

        return {
            "namespace": self.config.namespace,
            "image": self.config.image,
            "pending_tasks": pending,
            "running_tasks": running,
            "completed_tasks": completed,
            "failed_tasks": failed,
            "total_tasks": len(self._handles),
        }
