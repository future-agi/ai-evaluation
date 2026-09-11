from __future__ import annotations

try:
    import audioop
except ImportError as exc:  # pragma: no cover - Python 3.13 without audioop-lts
    raise ImportError(
        "LiveKit bridge audio requires 'audioop-lts' on Python 3.13+"
    ) from exc


class PCMResampler:
    def __init__(self, *, from_rate: int, to_rate: int, channels: int = 1) -> None:
        self._from_rate = from_rate
        self._to_rate = to_rate
        self._channels = channels
        self._state = None

    def convert(self, data: bytes) -> bytes:
        if self._from_rate == self._to_rate:
            return data
        converted, self._state = audioop.ratecv(
            data,
            2,
            self._channels,
            self._from_rate,
            self._to_rate,
            self._state,
        )
        return converted
