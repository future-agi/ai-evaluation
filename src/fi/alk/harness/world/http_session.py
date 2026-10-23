"""Bounded private HTTP session transport between isolated scenario phases."""

from __future__ import annotations

import http.cookiejar
import json
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from fi.simulate.agent.wrappers.http import validate_http_headers

from .errors import WorldUsageError

MAX_SESSION_BYTES = 65536
MAX_REDACTIONS = 256
MAX_REDACTION_BYTES = 32768


class _Cookie(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: int
    name: str
    value: str | None
    port: str | None
    port_specified: bool
    domain: str
    domain_specified: bool
    domain_initial_dot: bool
    path: str
    path_specified: bool
    secure: bool
    expires: int | None
    discard: bool
    comment: str | None
    comment_url: str | None
    rest: dict[str, str | None]
    rfc2109: bool = False


class _SessionState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    headers: dict[str, str]
    cookies: list[_Cookie] = Field(max_length=64)
    redactions: list[str] = Field(default_factory=list, max_length=256)


def headers(values):
    try:
        return validate_http_headers(values)
    except (TypeError, ValueError):
        raise WorldUsageError("native_http_headers_invalid") from None


def capability_name(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_.-]{0,127}", value
    ):
        raise WorldUsageError("native_http_capability_invalid")
    return value


@dataclass
class HttpSession:
    headers: dict[str, str] = field(default_factory=dict)
    cookies: http.cookiejar.CookieJar = field(default_factory=http.cookiejar.CookieJar)
    redactions: list[str] = field(default_factory=list)

    def remember(self, values) -> None:
        for value in values:
            if value and value not in self.redactions:
                self.redactions.append(value)
        del self.redactions[:-MAX_REDACTIONS]
        while (
            self.redactions
            and sum(len(value.encode()) for value in self.redactions)
            > MAX_REDACTION_BYTES
        ):
            self.redactions.pop(0)

    def state(self):
        return {
            "headers": headers(self.headers),
            "redactions": self.redactions,
            "cookies": [
                {
                    name: getattr(cookie, "_rest" if name == "rest" else name)
                    for name in _Cookie.model_fields
                }
                for cookie in self.cookies
            ],
        }


def restore_sessions(payload, endpoints):
    try:
        if (
            not isinstance(payload, dict)
            or len(payload) > 32
            or len(json.dumps(payload).encode()) > MAX_SESSION_BYTES
        ):
            raise ValueError
        result = {}
        for capability, raw in payload.items():
            capability_name(capability)
            if capability not in endpoints:
                raise ValueError
            state = _SessionState.model_validate(raw)
            session = HttpSession(
                headers=headers(state.headers), redactions=state.redactions
            )
            for cookie in state.cookies:
                if any(
                    "\r" in value or "\n" in value or "\0" in value
                    for value in (
                        cookie.name,
                        cookie.value or "",
                        cookie.domain,
                        cookie.path,
                    )
                ):
                    raise ValueError
                session.cookies.set_cookie(http.cookiejar.Cookie(**cookie.model_dump()))
            result[capability] = session
        return result
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise WorldUsageError("native_http_session_invalid") from None


def export_sessions(sessions, endpoints):
    payload = {name: session.state() for name, session in sessions.items()}
    restore_sessions(payload, endpoints)
    return payload


def validate_events(events, endpoints):
    import httpx

    error_names = {"WorldUsageError", "UnicodeDecodeError", "UnicodeEncodeError"}
    error_names.update(
        name
        for name in dir(httpx)
        if isinstance(getattr(httpx, name), type)
        and issubclass(getattr(httpx, name), httpx.HTTPError)
    )
    if not isinstance(events, list) or len(events) > 128:
        raise WorldUsageError("native_http_events_invalid")
    for event in events:
        if (
            not isinstance(event, dict)
            or set(event) - {"capability", "method", "route_hash", "status", "error"}
            or event.get("capability") not in endpoints
            or event.get("method")
            not in {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"}
            or not isinstance(event.get("route_hash"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", event["route_hash"])
            or (
                "status" in event
                and (
                    type(event["status"]) is not int
                    or not 100 <= event["status"] <= 999
                )
            )
            or ("error" in event and event["error"] not in error_names)
        ):
            raise WorldUsageError("native_http_events_invalid")
    return events
