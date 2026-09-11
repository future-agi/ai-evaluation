"""
Utilities for optional dependency handling.

Provides helpers for gracefully handling missing optional dependencies
with clear error messages guiding users to install them.
"""

from typing import Any, Optional


class OptionalDependency:
    """
    Wrapper for optional imports with helpful error messages.

    Example:
        TEMPORAL = OptionalDependency("temporalio", "pip install temporalio")

        # Check availability
        if TEMPORAL.is_available():
            client = TEMPORAL.require()

        # Or just require (raises if not available)
        temporalio = TEMPORAL.require()
    """

    def __init__(
        self,
        module_name: str,
        install_hint: str,
        extra_name: Optional[str] = None,
    ):
        """
        Initialize optional dependency wrapper.

        Args:
            module_name: The Python module name to import
            install_hint: Installation command to show in error message
            extra_name: Optional extra name for pip install (e.g., 'temporal')
        """
        self.module_name = module_name
        self.install_hint = install_hint
        self.extra_name = extra_name or module_name
        self._module: Optional[Any] = None
        self._checked = False
        self._available = False

    def is_available(self) -> bool:
        """Check if the dependency is installed."""
        if not self._checked:
            try:
                self._module = __import__(self.module_name)
                self._available = True
            except ImportError:
                self._available = False
            self._checked = True
        return self._available

    def require(self) -> Any:
        """
        Require the dependency, raising ImportError if not available.

        Returns:
            The imported module

        Raises:
            ImportError: If dependency is not installed
        """
        if not self.is_available():
            raise ImportError(
                f"'{self.module_name}' is required but not installed. "
                f"Install it with: {self.install_hint}\n"
                f"Or install the extra: pip install fi-evals[{self.extra_name}]"
            )
        return self._module

    def import_from(self, *names: str) -> tuple:
        """
        Import specific names from the module.

        Args:
            *names: Names to import from the module

        Returns:
            Tuple of imported objects

        Raises:
            ImportError: If dependency is not installed
        """
        module = self.require()
        result = []
        for name in names:
            parts = name.split(".")
            obj = module
            for part in parts:
                obj = getattr(obj, part)
            result.append(obj)
        return tuple(result) if len(result) > 1 else result[0]


# Pre-configured optional dependencies
TEMPORAL = OptionalDependency(
    "temporalio",
    "pip install temporalio",
    "temporal",
)

CELERY = OptionalDependency(
    "celery",
    "pip install 'celery[redis]'",
    "celery",
)

RAY = OptionalDependency(
    "ray",
    "pip install 'ray[default]'",
    "ray",
)

KUBERNETES = OptionalDependency(
    "kubernetes",
    "pip install kubernetes",
    "kubernetes",
)


def check_dependency(name: str) -> bool:
    """
    Check if a named dependency is available.

    Args:
        name: Dependency name ('temporal', 'celery', 'ray', 'kubernetes')

    Returns:
        True if available, False otherwise
    """
    deps = {
        "temporal": TEMPORAL,
        "celery": CELERY,
        "ray": RAY,
        "kubernetes": KUBERNETES,
    }
    dep = deps.get(name.lower())
    if dep:
        return dep.is_available()
    return False
