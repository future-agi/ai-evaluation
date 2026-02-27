"""
Shared container utilities for container-based backends.

Provides the serialization protocol, runner script, and result parsing
used by any backend that executes eval tasks inside containers (Kubernetes,
Docker, ECS, Nomad, etc.).

Custom backends can reuse these utilities::

    from fi.evals.framework.backends._container import (
        serialize_task,
        parse_result_from_logs,
        RUNNER_SCRIPT,
        EVAL_PAYLOAD_ENV,
        DEFAULT_IMAGE,
    )

    # Serialize (fn, args, kwargs) into a base64 string
    payload = serialize_task(my_func, (arg1,), {"key": "val"})

    # Pass payload as env var EVAL_PAYLOAD into any container that runs RUNNER_SCRIPT

    # After container finishes, parse the JSON result from its stdout
    result = parse_result_from_logs(container_stdout)

Security note:
    cloudpickle is the industry-standard serializer used by Kubeflow Pipelines,
    Ray, Dask, etc. It is only used here in trusted evaluation environments
    (your own code running on your own infrastructure) — never for untrusted input.
"""

import base64
import json
import logging
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default eval runner image. Build from ``Dockerfile.eval-runner``.
DEFAULT_IMAGE: str = "fi-eval-runner:latest"

#: Environment variable name for the serialized task payload.
EVAL_PAYLOAD_ENV: str = "EVAL_PAYLOAD"

#: Python bootstrap script to inject into containers.
#: Reads EVAL_PAYLOAD env var, deserializes with cloudpickle, runs the
#: function, and prints a JSON result line to stdout.
RUNNER_SCRIPT: str = """
import base64, cloudpickle, json, sys, traceback
try:
    import os
    payload = base64.b64decode(os.environ["EVAL_PAYLOAD"])
    fn, args, kwargs = cloudpickle.loads(payload)
    result = fn(*args, **kwargs)
    print(json.dumps({"status": "success", "result": result}))
except Exception:
    tb = traceback.format_exc()
    print(json.dumps({"status": "error", "error": tb}))
    sys.exit(1)
"""

#: Container command that executes the runner script.
RUNNER_COMMAND: list = ["python", "-c", RUNNER_SCRIPT]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def serialize_task(
    fn: Callable,
    args: tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Serialize a task (function + arguments) into a base64-encoded string.

    The result is safe to pass as an environment variable to a container
    running ``RUNNER_SCRIPT``.

    Args:
        fn: The function to execute.
        args: Positional arguments.
        kwargs: Keyword arguments.

    Returns:
        Base64-encoded string of the cloudpickle-serialized payload.

    Raises:
        ImportError: If cloudpickle is not installed.
    """
    import cloudpickle

    kwargs = kwargs or {}
    payload = cloudpickle.dumps((fn, args, kwargs))
    return base64.b64encode(payload).decode("utf-8")


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def parse_result_from_logs(logs: str) -> Any:
    """
    Parse the task result from container stdout logs.

    The runner script prints a JSON line as its last output. This function
    walks the log lines in reverse to find and parse that JSON.

    Args:
        logs: The full stdout text from the container.

    Returns:
        The deserialized result value.

    Raises:
        RuntimeError: If the logs contain an error result, are empty,
                      or don't contain a valid JSON result line.
    """
    lines = logs.strip().splitlines()
    if not lines:
        raise RuntimeError("Empty container logs — no result found")

    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if data.get("status") == "success":
                return data["result"]
            elif data.get("status") == "error":
                raise RuntimeError(data["error"])
        except json.JSONDecodeError:
            continue

    raise RuntimeError("No JSON result found in container logs")
