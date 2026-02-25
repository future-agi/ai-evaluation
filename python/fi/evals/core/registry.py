"""
Unified registry — resolves eval names to engines.

Routing logic:
    1. If the name is in the local metric registry → "local"
    2. If a turing model is specified → "turing"
    3. If engine is explicitly set → use that
    4. Otherwise → None (caller must provide engine)
"""

from typing import Optional


TURING_MODEL_PREFIXES = ("turing",)


def is_turing_model(model: Optional[str]) -> bool:
    """Check if the model string indicates a Turing platform model."""
    if not model:
        return False
    return model.lower().startswith(TURING_MODEL_PREFIXES)


class UnifiedRegistry:
    """Resolves an eval name + model to its engine type."""

    def __init__(self) -> None:
        self._local_registry = None

    @property
    def local_registry(self):
        if self._local_registry is None:
            from ..local.registry import get_registry
            self._local_registry = get_registry()
        return self._local_registry

    def is_local(self, name: str) -> bool:
        return self.local_registry.is_registered(name)

    def resolve_engine(
        self,
        name: Optional[str],
        *,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        engine: Optional[str] = None,
    ) -> Optional[str]:
        """Auto-detect which engine should handle the request.

        Priority:
            1. Explicit engine kwarg
            2. Name in local registry → "local"
            3. Turing model specified → "turing"
            4. Non-turing model specified → "llm"
            5. None (ambiguous)
        """
        if engine:
            return engine

        if name and self.is_local(name):
            return "local"

        if is_turing_model(model):
            return "turing"

        if model:
            return "llm"

        return None

    def list_local(self):
        return self.local_registry.list_metrics()


# Singleton
_unified_registry: Optional[UnifiedRegistry] = None


def get_unified_registry() -> UnifiedRegistry:
    global _unified_registry
    if _unified_registry is None:
        _unified_registry = UnifiedRegistry()
    return _unified_registry
