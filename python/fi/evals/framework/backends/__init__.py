"""
Backend implementations for distributed evaluation.

Provides pluggable backends for running evaluations:
- ThreadPoolBackend: Local execution using thread pool (default)
- TemporalBackend: Durable workflows via Temporal (optional)
- CeleryBackend: Distributed tasks via Celery (optional)
- RayBackend: Distributed computing via Ray (optional)

Optional backends are lazily imported to avoid requiring their dependencies.
Install optional dependencies with:
    pip install fi-evals[temporal]  # For Temporal
    pip install fi-evals[celery]    # For Celery
    pip install fi-evals[ray]       # For Ray
"""

from typing import TYPE_CHECKING

from .base import (
    Backend,
    BackendConfig,
    TaskHandle,
    TaskStatus,
)
from .thread_pool import ThreadPoolBackend, ThreadPoolConfig

# Type hints for lazy imports (only used by type checkers)
if TYPE_CHECKING:
    from .temporal import TemporalBackend, TemporalConfig
    from .celery_backend import CeleryBackend, CeleryConfig
    from .ray_backend import RayBackend, RayConfig

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
]

# Lazy imports for optional backends
_LAZY_IMPORTS = {
    "TemporalBackend": (".temporal", "TemporalBackend"),
    "TemporalConfig": (".temporal", "TemporalConfig"),
    "CeleryBackend": (".celery_backend", "CeleryBackend"),
    "CeleryConfig": (".celery_backend", "CeleryConfig"),
    "RayBackend": (".ray_backend", "RayBackend"),
    "RayConfig": (".ray_backend", "RayConfig"),
}


def __getattr__(name: str):
    """Lazy import for optional backend dependencies."""
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_name, __package__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
