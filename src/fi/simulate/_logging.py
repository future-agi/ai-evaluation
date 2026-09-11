from __future__ import annotations

from types import TracebackType


def redacted_exc_info(
    exc: BaseException,
) -> tuple[type[RuntimeError], RuntimeError, TracebackType | None]:
    redacted = RuntimeError(f"{type(exc).__name__}: details redacted")
    return RuntimeError, redacted, exc.__traceback__
