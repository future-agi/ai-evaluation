"""Adapter registries (canonical plan §3/§4).

The registries are keyed by the string names already carried on
``SimulationSpec`` — ``environment.adapter`` / ``target.adapter`` /
``simulator.adapter`` — so the declarative spec stays the stable contract while
dispatch moves from hardcoded ``if``/dict branches (``runner.py``, ``planner.py``)
to plug-and-play lookup.

Two registration paths, so "anyone can add anything and it just works":

1. In-process decorator on a factory::

       @register_environment("chat")
       class ChatEnvironmentPlugin: ...

2. Third-party plugins via ``importlib.metadata`` entry-point groups. A
   pip-installed package declares, in its own ``pyproject.toml``::

       [project.entry-points."fi.simulate.environments"]
       my_world = "my_pkg.env:MyEnvironmentPlugin"

   and it is discovered on first lookup without editing this codebase.
"""

from __future__ import annotations

import logging
import threading
from importlib import metadata
from typing import Callable, Dict, Generic, Iterable, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

ENVIRONMENT_ENTRY_POINT_GROUP = "fi.simulate.environments"
ENDPOINT_ENTRY_POINT_GROUP = "fi.simulate.endpoints"
SIMULATOR_ENTRY_POINT_GROUP = "fi.simulate.simulators"


class AdapterNotFound(KeyError):
    """Raised when a spec references an adapter name nobody registered."""

    def __init__(self, kind: str, name: str, available: Iterable[str]) -> None:
        self.kind = kind
        self.name = name
        self.available = sorted(available)
        super().__init__(
            f"{kind}_adapter_unsupported: {name!r} is not registered; "
            f"available: {self.available}"
        )


class AdapterRegistry(Generic[T]):
    """Thread-safe name -> factory registry with lazy entry-point discovery."""

    def __init__(self, kind: str, entry_point_group: Optional[str] = None) -> None:
        self._kind = kind
        self._entry_point_group = entry_point_group
        self._factories: Dict[str, Callable[..., T]] = {}
        self._lock = threading.RLock()
        self._entry_points_loaded = False

    def register(
        self,
        name: str,
        factory: Optional[Callable[..., T]] = None,
        *,
        override: bool = False,
    ):
        """Register ``factory`` under ``name``. Usable as a decorator."""

        def _apply(f: Callable[..., T]) -> Callable[..., T]:
            with self._lock:
                existing = self._factories.get(name)
                if existing is not None and existing is not f and not override:
                    raise ValueError(
                        f"{self._kind}_adapter_already_registered: {name!r}"
                    )
                self._factories[name] = f
            return f

        return _apply if factory is None else _apply(factory)

    def _load_entry_points(self) -> None:
        if self._entry_points_loaded:
            return
        with self._lock:
            if self._entry_points_loaded:
                return
            if self._entry_point_group:
                try:
                    eps = list(metadata.entry_points(group=self._entry_point_group))
                except Exception:  # pragma: no cover - importlib version quirks
                    eps = []
                for ep in eps:
                    if ep.name in self._factories:
                        continue
                    try:
                        self._factories[ep.name] = ep.load()
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.warning(
                            "Failed to load %s plugin %r: %s",
                            self._kind,
                            ep.name,
                            exc,
                        )
            self._entry_points_loaded = True

    def get(self, name: str) -> Callable[..., T]:
        with self._lock:
            factory = self._factories.get(name)
        if factory is not None:
            return factory
        self._load_entry_points()
        with self._lock:
            factory = self._factories.get(name)
            if factory is None:
                raise AdapterNotFound(self._kind, name, self._factories.keys())
            return factory

    def get_or_none(self, name: str) -> Optional[Callable[..., T]]:
        try:
            return self.get(name)
        except AdapterNotFound:
            return None

    def create(self, name: str, *args, **kwargs) -> T:
        return self.get(name)(*args, **kwargs)

    def has(self, name: str) -> bool:
        return self.get_or_none(name) is not None

    def names(self) -> List[str]:
        self._load_entry_points()
        with self._lock:
            return sorted(self._factories)


environment_registry: AdapterRegistry = AdapterRegistry(
    "environment", ENVIRONMENT_ENTRY_POINT_GROUP
)
endpoint_registry: AdapterRegistry = AdapterRegistry(
    "endpoint", ENDPOINT_ENTRY_POINT_GROUP
)
simulator_registry: AdapterRegistry = AdapterRegistry(
    "simulator", SIMULATOR_ENTRY_POINT_GROUP
)


def register_environment(name: str, factory=None, *, override: bool = False):
    """Register a **runnable** environment plugin factory into ``environment_registry``.

    Canon correspondence (assessment §8 Gap B): a same-named sibling
    ``fi.alk.extensions.register_environment`` writes a **metadata record** into
    the studio extension registry (and, for a ``world.kind`` extension, writes
    the contract's extension side-table via ``contract.register_world_kind`` —
    the frozen canon constants never mutate). That one is *descriptive*; this one
    is *executable*. Two systems on purpose — a metadata
    record is not a runnable factory, so never auto-wire one into the other.
    """
    return environment_registry.register(name, factory, override=override)


def register_endpoint(name: str, factory=None, *, override: bool = False):
    return endpoint_registry.register(name, factory, override=override)


def register_simulator(name: str, factory=None, *, override: bool = False):
    return simulator_registry.register(name, factory, override=override)


__all__ = [
    "AdapterNotFound",
    "AdapterRegistry",
    "ENDPOINT_ENTRY_POINT_GROUP",
    "ENVIRONMENT_ENTRY_POINT_GROUP",
    "SIMULATOR_ENTRY_POINT_GROUP",
    "endpoint_registry",
    "environment_registry",
    "register_endpoint",
    "register_environment",
    "register_simulator",
    "simulator_registry",
]
