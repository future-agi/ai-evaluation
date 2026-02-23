"""
Backend implementations for distributed evaluation.

Provides pluggable backends for running evaluations:
- ThreadPoolBackend: Local execution using thread pool (default)
- TemporalBackend: Durable workflows via Temporal (optional)
- CeleryBackend: Distributed tasks via Celery (optional)
- RayBackend: Distributed computing via Ray (optional)
- KubernetesBackend: Cloud-native jobs via Kubernetes (optional)

Container utilities for custom backends:
- serialize_task / parse_result_from_logs: serialization protocol
- RUNNER_SCRIPT / RUNNER_COMMAND / DEFAULT_IMAGE: container constants
- Dockerfile.eval-runner: pre-built eval runner image

Optional backends are lazily imported to avoid requiring their dependencies.
Install optional dependencies with:
    pip install fi-evals[temporal]    # For Temporal
    pip install fi-evals[celery]      # For Celery
    pip install fi-evals[ray]         # For Ray
    pip install fi-evals[kubernetes]  # For Kubernetes
"""

from typing import TYPE_CHECKING

from .base import (
    Backend,
    BackendConfig,
    TaskHandle,
    TaskStatus,
)
from .thread_pool import ThreadPoolBackend, ThreadPoolConfig
from ._container import (
    DEFAULT_IMAGE,
    EVAL_PAYLOAD_ENV,
    RUNNER_COMMAND,
    RUNNER_SCRIPT,
    parse_result_from_logs,
    serialize_task,
)

# Type hints for lazy imports (only used by type checkers)
if TYPE_CHECKING:
    from .temporal import TemporalBackend, TemporalConfig
    from .celery_backend import CeleryBackend, CeleryConfig
    from .ray_backend import RayBackend, RayConfig
    from .kubernetes_backend import KubernetesBackend, KubernetesConfig

__all__ = [
    # Base
    "Backend",
    "BackendConfig",
    "TaskHandle",
    "TaskStatus",
    # Thread Pool
    "ThreadPoolBackend",
    "ThreadPoolConfig",
    # Temporal (optional)
    "TemporalBackend",
    "TemporalConfig",
    # Celery (optional)
    "CeleryBackend",
    "CeleryConfig",
    # Ray (optional)
    "RayBackend",
    "RayConfig",
    # Kubernetes (optional)
    "KubernetesBackend",
    "KubernetesConfig",
    # Container utilities (for custom backends)
    "DEFAULT_IMAGE",
    "EVAL_PAYLOAD_ENV",
    "RUNNER_COMMAND",
    "RUNNER_SCRIPT",
    "serialize_task",
    "parse_result_from_logs",
]

# Lazy imports for optional backends
_LAZY_IMPORTS = {
    "TemporalBackend": (".temporal", "TemporalBackend"),
    "TemporalConfig": (".temporal", "TemporalConfig"),
    "CeleryBackend": (".celery_backend", "CeleryBackend"),
    "CeleryConfig": (".celery_backend", "CeleryConfig"),
    "RayBackend": (".ray_backend", "RayBackend"),
    "RayConfig": (".ray_backend", "RayConfig"),
    "KubernetesBackend": (".kubernetes_backend", "KubernetesBackend"),
    "KubernetesConfig": (".kubernetes_backend", "KubernetesConfig"),
}


def __getattr__(name: str):
    """Lazy import for optional backend dependencies."""
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_name, __package__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
