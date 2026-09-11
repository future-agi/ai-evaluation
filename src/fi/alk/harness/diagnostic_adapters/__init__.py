"""Adapters from backend-native exceptions to stable harness diagnostics."""

from .postgres import diagnose_postgres_error

__all__ = ["diagnose_postgres_error"]
