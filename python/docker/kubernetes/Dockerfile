# Eval runner image for the Kubernetes backend.
#
# This image is used by KubernetesBackend to execute serialized evaluation
# tasks inside Kubernetes Jobs. Contains cloudpickle for deserialization
# and a minimal Python runtime.
#
# Build:
#   docker build -t fi-eval-runner:latest .
#
# Security note: cloudpickle is the industry-standard serializer used by
# Kubeflow, Ray, Dask, etc. Only used in trusted evaluation environments.

FROM python:3.13-slim

LABEL maintainer="Future AGI" \
      description="Eval runner for fi-evals Kubernetes backend"

# Install cloudpickle for task deserialization
RUN pip install --no-cache-dir cloudpickle>=3.0

# Drop to non-root user for safety
RUN useradd --create-home evalrunner
USER evalrunner
WORKDIR /home/evalrunner

# The actual task code is injected via the EVAL_PAYLOAD env var at runtime.
# The KubernetesBackend sets the container command to execute the runner script.
