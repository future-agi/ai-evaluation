"""Database catalogue adapters that produce the canonical harness source model."""

from .postgres import inspect_postgres
from .sqlite import inspect_sqlite

__all__ = ["inspect_postgres", "inspect_sqlite"]
