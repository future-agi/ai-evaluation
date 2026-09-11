from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any


def optional_module(module_name: str, extra: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"`{module_name}` is not available in this `agent-learning-kit` "
            "installation. The Python simulation, evaluation, and optimization "
            "engines are vendored in the base package; reinstall "
            "`agent-learning-kit`. Use `agent-learning-kit[trinity]` only for "
            "optional heavier integrations."
        ) from exc


def proxy_getattr(module_name: str, extra: str, name: str) -> Any:
    module = optional_module(module_name, extra)
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(f"module `{module_name}` has no attribute `{name}`") from exc


def proxy_dir(module_name: str, extra: str) -> list[str]:
    module = optional_module(module_name, extra)
    return sorted(set(dir(module)))
