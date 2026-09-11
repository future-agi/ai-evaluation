"""Database catalogue adapters that produce the canonical harness source model."""

from .sqlite import inspect_sqlite

__all__ = ["inspect_sqlite"]
