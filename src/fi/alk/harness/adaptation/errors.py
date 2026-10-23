from __future__ import annotations


class AdaptationFailure(RuntimeError):
    """Adaptation stops with `result.json` status `failed` and this `{code, message}`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
