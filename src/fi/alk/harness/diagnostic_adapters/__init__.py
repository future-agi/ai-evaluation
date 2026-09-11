"""Adapters from backend-native exceptions to stable harness diagnostics."""

from .postgres import diagnose_postgres_error
from .world_ir import diagnose_world_ir_error

__all__ = ["diagnose_postgres_error", "diagnose_world_ir_error"]
