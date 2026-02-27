"""
Temporal worker for executing evaluation tasks.

This module defines the real Temporal workflow and activity that the
TemporalBackend submits work to. It uses cloudpickle for serialization,
matching the pattern used by the Kubernetes backend (_container.py).

Run as a standalone worker process::

    python -m fi.evals.framework.backends.temporal_worker

Or with custom settings::

    TEMPORAL_HOST=temporal.example.com:7233 \\
    TEMPORAL_NAMESPACE=evaluations \\
    TEMPORAL_TASK_QUEUE=eval-tasks \\
    python -m fi.evals.framework.backends.temporal_worker

Requires: pip install temporalio cloudpickle

Security note:
    cloudpickle is the industry-standard serializer used by Kubeflow Pipelines,
    Ray, Dask, etc. It is only used here in trusted evaluation environments
    (your own code running on your own infrastructure) — never for untrusted input.
"""

import asyncio
import base64
import logging
import os
from datetime import timedelta
from typing import Any

import cloudpickle
from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.worker import Worker

logger = logging.getLogger(__name__)


def serialize_task(fn, args: tuple = (), kwargs: dict | None = None) -> str:
    """Serialize a task into a base64 cloudpickle payload."""
    kwargs = kwargs or {}
    payload = cloudpickle.dumps((fn, args, kwargs))
    return base64.b64encode(payload).decode("utf-8")


def deserialize_task(payload_b64: str) -> tuple:
    """Deserialize a base64 cloudpickle payload into (fn, args, kwargs)."""
    raw = base64.b64decode(payload_b64)
    return cloudpickle.loads(raw)


def serialize_result(result: Any) -> str:
    """Serialize a result into a base64 cloudpickle payload."""
    return base64.b64encode(cloudpickle.dumps(result)).decode("utf-8")


def deserialize_result(payload_b64: str) -> Any:
    """Deserialize a base64 cloudpickle result."""
    return cloudpickle.loads(base64.b64decode(payload_b64))


@activity.defn
async def execute_eval_task(payload_b64: str) -> str:
    """
    Temporal activity that executes a serialized evaluation task.

    Receives a cloudpickle-serialized (fn, args, kwargs) tuple as base64,
    executes the function, and returns the cloudpickle-serialized result
    as base64.
    """
    fn, args, kwargs = deserialize_task(payload_b64)
    logger.info(f"Executing task: {getattr(fn, '__name__', str(fn))}")
    result = fn(*args, **kwargs)
    return serialize_result(result)


@workflow.defn
class EvalTaskWorkflow:
    """Workflow that executes a single evaluation task via activity."""

    @workflow.run
    async def run(self, payload_b64: str) -> str:
        return await workflow.execute_activity(
            execute_eval_task,
            payload_b64,
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=RetryPolicy(
                maximum_attempts=3,
                initial_interval=timedelta(seconds=1),
                backoff_coefficient=2.0,
            ),
        )


async def run_worker(
    host: str = "localhost:7233",
    namespace: str = "default",
    task_queue: str = "eval-tasks",
) -> None:
    """Connect to Temporal and run the worker until interrupted."""
    client = await Client.connect(host, namespace=namespace)
    logger.info(f"Connected to Temporal at {host}, namespace={namespace}")

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[EvalTaskWorkflow],
        activities=[execute_eval_task],
    )
    logger.info(f"Worker listening on task queue: {task_queue}")
    await worker.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(
        run_worker(
            host=os.environ.get("TEMPORAL_HOST", "localhost:7233"),
            namespace=os.environ.get("TEMPORAL_NAMESPACE", "default"),
            task_queue=os.environ.get("TEMPORAL_TASK_QUEUE", "eval-tasks"),
        )
    )
