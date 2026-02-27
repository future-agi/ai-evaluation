# Temporal worker for fi-evals distributed evaluation backend.
#
# Build:
#   docker build -t fi-temporal-worker:latest .
#
# The worker connects to the Temporal server and polls the eval-tasks
# task queue for workflows submitted via TemporalBackend in the SDK.
#
# Security note: cloudpickle is the industry-standard serializer used by
# Kubeflow, Ray, Dask, etc. Only used in trusted evaluation environments.

FROM python:3.13-slim

LABEL maintainer="Future AGI" \
      description="Temporal worker for fi-evals distributed evaluations"

WORKDIR /app

# Install Temporal SDK and cloudpickle for task serialization
RUN pip install --no-cache-dir temporalio cloudpickle>=3.0

# Copy the SDK source
COPY fi ./fi

# Drop to non-root
RUN useradd --create-home worker
USER worker

CMD ["python", "-m", "fi.evals.framework.backends.temporal_worker"]
