from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from typing import Any

from .config import AgentLearningConfig, configure, current_config, get_api_key

try:
    __version__ = _package_version("agent-learning-kit")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.0.0+unknown"

_SUBMODULES = {
    "actions",
    "bench",
    "capabilities",
    "evals",
    "optimize",
    "redteam",
    "simulate",
    "studio",
    "suite",
    "trinity",
}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_SUBMODULES})


__all__ = [
    "AgentLearningConfig",
    "__version__",
    "actions",
    "bench",
    "capabilities",
    "configure",
    "current_config",
    "evals",
    "get_api_key",
    "optimize",
    "redteam",
    "simulate",
    "studio",
    "suite",
    "trinity",
]
