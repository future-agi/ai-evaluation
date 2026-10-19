from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

SETUP_HASH_PREFIX = "sh1:"


def compute_setup_hash(agent: Mapping[str, Any]) -> str:
    hashed = copy.deepcopy(dict(agent))
    hashed.pop("setup_hash", None)
    hashed.pop("direction_answer", None)
    if isinstance(hashed.get("provider_ref"), dict):
        hashed["provider_ref"].pop("version", None)
    body = json.dumps(
        hashed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return SETUP_HASH_PREFIX + hashlib.sha256(body).hexdigest()
