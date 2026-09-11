from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


def content_hash(value: BaseModel | Mapping[str, Any]) -> str:
    payload = (
        value.model_dump(mode="json", exclude_none=True)
        if isinstance(value, BaseModel)
        else dict(value)
    )
    encoded = json.dumps(
        payload,
        default=_json_default,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
