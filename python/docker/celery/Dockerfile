# Celery worker for fi-evals distributed evaluation backend.
#
# Build:
#   docker build -t fi-celery-worker:latest .
#
# The worker connects to Redis (broker + result backend) and processes
# evaluation tasks submitted via CeleryBackend in the SDK.
#
# Security note: pickle serialization is used intentionally for Celery tasks
# that transport arbitrary Python callables (evaluation functions) in trusted
# environments. Never expose the broker to untrusted networks.

FROM python:3.13-slim

LABEL maintainer="Future AGI" \
      description="Celery worker for fi-evals distributed evaluations"

WORKDIR /app

# Install Celery with Redis support and cloudpickle for task serialization
RUN pip install --no-cache-dir 'celery[redis]' cloudpickle>=3.0

# Copy the SDK source
COPY fi ./fi

# Drop to non-root
RUN useradd --create-home worker
USER worker

# Use --pool=threads to avoid billiard re-pickling (which can't handle
# cloudpickle-deserialized closures). Thread pool is fine for I/O-bound
# eval tasks; use --pool=solo for CPU-bound workloads.
CMD ["celery", "-A", "fi.evals.framework.backends.celery_worker", "worker", \
     "-Q", "eval_tasks", "--loglevel=info", "--concurrency=4", "--pool=threads"]
