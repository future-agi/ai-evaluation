from __future__ import annotations

import json
import re
from collections.abc import Mapping
from urllib.parse import quote, quote_plus, urlsplit, urlunsplit

_SECRET_FIELD = re.compile(
    r"(?i)(authorization|cookie|password|passwd|secret|token|api[_-]?key|private[_-]?key|credential)"
)
_URL = re.compile(r'https?://[^\s<>"\\]+')
_FIELD = re.compile(
    r"""(?i)((?:["']?)[\w-]*(?:authorization|cookie|password|passwd|secret|token|api[_-]?key|private[_-]?key|credential)[\w-]*["']?\s*[:=]\s*)("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\r\n,;}]+)"""
)


def safe_endpoint(value: str) -> str:
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        if parsed.port:
            host += f":{parsed.port}"
        return urlunsplit(
            (parsed.scheme, host, parsed.path, "<redacted>" if parsed.query else "", "")
        )
    except ValueError:
        return "<invalid endpoint>"


def sanitize_diagnostic(
    text: str, secrets: Mapping[str, str] | None, *, limit: int = 8192
) -> str:
    if secrets is None:
        return "<diagnostic omitted: credential context unavailable>"
    values: set[str] = set()

    def leaves(value):
        if isinstance(value, str) and value:
            values.add(value)
            values.update(line for line in value.splitlines() if line)
        elif isinstance(value, (int, float, bool)):
            values.add(json.dumps(value))
        elif isinstance(value, Mapping):
            for item in value.values():
                leaves(item)
        elif isinstance(value, list):
            for item in value:
                leaves(item)

    for value in secrets.values():
        leaves(value)
        bearer = re.fullmatch(r"(?i)(?:bearer|basic)\s+(\S+)", value)
        if bearer:
            values.add(bearer.group(1))
        try:
            leaves(json.loads(value))
        except (ValueError, TypeError, RecursionError):
            pass
    variants = set(values)
    if sum(map(len, values)) > 1024 * 1024:
        return "<diagnostic omitted: credential context exceeds safe limit>"
    for value in values:
        for ascii_only in (False, True):
            escaped = json.dumps(value, ensure_ascii=ascii_only)[1:-1]
            variants.update(
                (escaped, json.dumps(escaped)[1:-1], escaped.replace("/", "\\/"))
            )
        variants.update((quote(value, safe=""), quote_plus(value)))
    for value in sorted(variants, key=len, reverse=True):
        text = text.replace(value, "<redacted>")
    text = _URL.sub(lambda match: safe_endpoint(match.group()), text)
    text = re.sub(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        "<redacted>",
        text,
        flags=re.DOTALL,
    )
    try:
        document = json.loads(text)
    except (ValueError, RecursionError):
        text = _FIELD.sub(lambda match: match.group(1) + '"<redacted>"', text)
    else:

        def scrub(value):
            if isinstance(value, dict):
                return {
                    key: "<redacted>" if _SECRET_FIELD.search(key) else scrub(item)
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [scrub(item) for item in value]
            if isinstance(value, str):
                return _FIELD.sub(lambda match: match.group(1) + '"<redacted>"', value)
            return value

        try:
            text = json.dumps(scrub(document), ensure_ascii=True)
        except RecursionError:
            return "<diagnostic omitted: nesting exceeds safe limit>"
    return text[:limit]
