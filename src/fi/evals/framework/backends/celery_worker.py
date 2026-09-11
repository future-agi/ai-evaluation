"""
Shared Celery app and task definition.

This module is importable by both the SDK client (CeleryBackend) and a
standalone Celery worker process, solving the problem of task registration
needing to be visible to workers.

Run as a worker::

    celery -A fi.evals.framework.backends.celery_worker worker \\
        -Q eval_tasks --loglevel=info

Or configure via environment variables::

    CELERY_BROKER_URL=redis://redis:6379/0 \\
    CELERY_RESULT_BACKEND=redis://redis:6379/1 \\
    celery -A fi.evals.framework.backends.celery_worker worker \\
        -Q eval_tasks --loglevel=info

Requires: pip install 'celery[redis]'

Security note:
    Pickle serialization is used here because Celery tasks transport arbitrary
    Python callables (evaluation functions). This is the standard approach for
    Celery task serialization in trusted environments. Never expose the broker
    to untrusted networks.
"""

import os

import cloudpickle
from celery import Celery
from kombu.serialization import register

# ---------------------------------------------------------------------------
# Register cloudpickle as a Celery serializer.
# Unlike stdlib pickle, cloudpickle can serialize lambdas, closures, and
# functions defined in __main__ — exactly what we need for ad-hoc eval fns.
# ---------------------------------------------------------------------------

register(
    "cloudpickle",
    cloudpickle.dumps,
    cloudpickle.loads,
    content_type="application/x-cloudpickle",
    content_encoding="binary",
)

# ---------------------------------------------------------------------------
# App configuration — shared between SDK client and Docker worker
# ---------------------------------------------------------------------------

BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
TASK_QUEUE = os.environ.get("CELERY_TASK_QUEUE", "eval_tasks")

app = Celery(
    "eval_tasks",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
)

app.conf.update(
    task_serializer="cloudpickle",
    result_serializer="cloudpickle",
    accept_content=["cloudpickle", "json"],
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_default_queue=TASK_QUEUE,
)


@app.task(bind=True, name="eval_task")
def eval_task(self, fn, args, kwargs):
    """Generic task that executes the provided function."""
    return fn(*args, **kwargs)
