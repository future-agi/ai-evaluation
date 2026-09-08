"""Static LiveKit worker metadata recovered from untrusted source code."""

from __future__ import annotations

import re
from collections.abc import Collection
from pathlib import Path


_ENV_AGENT_NAME = re.compile(
    r"agent_name\s*=\s*os\.(?:environ\.get|getenv)\(\s*"
    r"[\"']LIVEKIT_AGENT_NAME[\"']\s*,\s*[\"'](?P<name>[^\"']+)[\"']"
)
_STATIC_RTC_SESSION_AGENT_NAME = re.compile(
    r"@(?:[A-Za-z_][A-Za-z0-9_]*\.)*rtc_session\s*\("
    r"(?:(?!\)\s*(?:\r?\n|$)).)*?"
    r"agent_name\s*=\s*[\"'](?P<name>[^\"']+)[\"']",
    re.DOTALL,
)


def infer_livekit_agent_name_from_source(
    source_root: Path, *, ignored_parts: Collection[str] = ()
) -> str:
    """Return a literal LiveKit dispatch name declared by common SDK forms.

    Dynamic expressions remain unresolved deliberately: guessing could route a call to the wrong
    worker on a shared LiveKit project.
    """

    try:
        if not source_root.is_dir():
            return ""
        for path in source_root.rglob("*.py"):
            relative = path.relative_to(source_root)
            if any(part in ignored_parts for part in relative.parts):
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for pattern in (_ENV_AGENT_NAME, _STATIC_RTC_SESSION_AGENT_NAME):
                if match := pattern.search(source):
                    return match.group("name").strip()
    except OSError:
        pass
    return ""


__all__ = ["infer_livekit_agent_name_from_source"]
