"""Native state accessed through declared application APIs and recorded responses."""

from __future__ import annotations

import json as json_module
import random
from urllib.parse import urlsplit, urlunsplit

import httpx

from .errors import WorldUsageError


class ObservationWorld:
    def __init__(
        self, world_index: int, rng: random.Random, endpoints=None, *, read_only=False
    ):
        self.world_index = world_index
        self.rng = rng
        self.calls = []
        self._endpoints = dict(endpoints or {})
        self._read_only = read_only

    def read_only(self):
        return ObservationWorld(
            self.world_index, self.rng, self._endpoints, read_only=True
        )

    def request(
        self, capability: str, method: str, path: str, *, json=None, headers=None
    ):
        """Use one declared HTTP capability; return bounded status/text/JSON without redirects."""
        method = method.upper()
        if method not in {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"}:
            raise WorldUsageError("native_http_method_unsupported")
        if self._read_only and method not in {"GET", "HEAD"}:
            raise WorldUsageError("native_http_read_only")
        base = urlsplit(self._endpoints.get(capability, ""))
        relative = urlsplit(path)
        if (
            base.scheme not in {"http", "https"}
            or not base.hostname
            or base.username
            or base.password
        ):
            raise WorldUsageError("native_http_capability_unavailable")
        if (
            not path.startswith("/")
            or path.startswith("//")
            or relative.scheme
            or relative.netloc
            or relative.fragment
            or "\\" in path
            or any(ord(char) < 32 for char in path)
        ):
            raise WorldUsageError("native_http_relative_path_required")
        headers = dict(headers or {})
        forbidden = {
            "host",
            "content-length",
            "transfer-encoding",
            "connection",
            "proxy-authorization",
        }
        if (
            any(
                not isinstance(key, str)
                or not isinstance(value, str)
                or key.lower() in forbidden
                or "\r" in key + value
                or "\n" in key + value
                for key, value in headers.items()
            )
            or sum(len(key) + len(value) for key, value in headers.items()) > 16384
        ):
            raise WorldUsageError("native_http_headers_invalid")
        body = None if json is None else json_module.dumps(json).encode("utf-8")
        if body is not None and len(body) > 65536:
            raise WorldUsageError("native_http_request_too_large")
        if body is not None:
            headers.setdefault("Content-Type", "application/json")
        url = urlunsplit((base.scheme, base.netloc, relative.path, relative.query, ""))
        try:
            with httpx.Client(
                timeout=10, follow_redirects=False, trust_env=False
            ) as client:
                with client.stream(
                    method, url, content=body, headers=headers
                ) as response:
                    chunks = []
                    size = 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > 1048576:
                            raise WorldUsageError("native_http_response_too_large")
                        chunks.append(chunk)
                    text = b"".join(chunks).decode("utf-8")
                    try:
                        parsed = json_module.loads(text)
                    except ValueError:
                        parsed = None
                    return {
                        "status": response.status_code,
                        "text": text,
                        "json": parsed,
                    }
        except (httpx.HTTPError, UnicodeError) as exc:
            raise WorldUsageError(
                f"native_http_request_failed: {type(exc).__name__}"
            ) from None

    def _execution_state(self):
        return {
            "kind": "native_observation",
            "world_index": self.world_index,
            "rng": self.rng.getstate(),
            "read_only": self._read_only,
            "endpoints": self._endpoints,
        }

    def _unsupported(self, *args, **kwargs):
        raise WorldUsageError(
            "native_observation: state access and harness tool execution are unsupported; observe native responses"
        )

    state = query = put = change = drop = call = _unsupported
    read_file = write_file = checkpoint = restore = handle_tool_call = _unsupported


def http_endpoints(runtime):
    return {
        name: endpoint.address
        for name, endpoint in runtime.endpoints.items()
        if endpoint.protocol in {"http", "https"}
    }


class ObservationWorldFactory:
    async def create(self, runtime, *, rng):
        return ObservationWorld(runtime.world_index, rng, http_endpoints(runtime))
