"""Native state accessed through declared application APIs and recorded responses."""

from __future__ import annotations

import hashlib
import json as json_module
import random
from urllib.parse import urlencode, urlsplit, urlunsplit

import httpx

from .errors import WorldUsageError
from .http_session import (
    HttpSession,
    capability_name,
    export_sessions,
    restore_sessions,
    headers as validate_headers,
)


class HttpObservationHandle:
    """Forward declared HTTP operations to the handle's scoped observation client."""

    _http_client: ObservationWorld

    @property
    def capabilities(self):
        return self._http_client.capabilities

    def request(self, capability, method, path, **kwargs):
        return self._http_client.request(capability, method, path, **kwargs)

    def observe(self, name, parameters=None):
        return self._http_client.observe(name, parameters)

    def set_headers(self, capability, headers):
        return self._http_client.set_headers(capability, headers)


class ObservationWorld:
    def __init__(
        self,
        world_index: int,
        rng: random.Random,
        endpoints=None,
        *,
        read_only=False,
        observations=None,
    ):
        from ..contract import Runtime

        self._observations = Runtime.model_validate_json(
            Runtime(observations=observations or {}).model_dump_json()
        ).observations
        if any(
            spec.capability not in (endpoints or {})
            for spec in self._observations.values()
        ):
            raise WorldUsageError("native_http_observation_capability_unavailable")
        self.world_index = world_index
        self.rng = rng
        self.calls = []
        self._endpoints = dict(endpoints or {})
        self._read_only = read_only
        self._sessions = {}
        self._http_events = []

    @property
    def capabilities(self):
        """Declared HTTP capability names, without endpoints or session credentials."""
        return tuple(sorted(self._endpoints))

    def read_only(self):
        world = ObservationWorld(
            self.world_index,
            self.rng,
            self._endpoints,
            read_only=True,
            observations=self._observations,
        )
        world._sessions = restore_sessions(self._private_http_state(), self._endpoints)
        world._http_events = self._http_events
        return world

    def set_headers(self, capability: str, headers):
        if self._read_only:
            raise WorldUsageError("native_http_read_only")
        capability_name(capability)
        if capability not in self._endpoints:
            raise WorldUsageError("native_http_capability_unavailable")
        values = validate_headers(headers)
        session = self._sessions.setdefault(capability, HttpSession())
        previous = session.headers
        session.remember(_sensitive_header_values(previous))
        session.headers = values
        try:
            self._private_http_state()
        except WorldUsageError:
            session.headers = previous
            raise

    def _private_http_state(self):
        return export_sessions(self._sessions, self._endpoints)

    def _restore_private_http_state(self, state):
        sessions = restore_sessions(state, self._endpoints)
        self._sessions.clear()
        self._sessions.update(sessions)

    def _redact_http_text(self, value):
        from fi._diagnostics import sanitize_diagnostic

        values = [
            item
            for session in self._sessions.values()
            for item in session.headers.values()
        ]
        values.extend(
            value for session in self._sessions.values() for value in session.redactions
        )
        values.extend(
            cookie.value
            for session in self._sessions.values()
            for cookie in session.cookies
            if cookie.value
        )
        return sanitize_diagnostic(
            value, {str(index): item for index, item in enumerate(values)}
        )

    def _drain_http_events(self):
        events = self._http_events[:]
        self._http_events.clear()
        return events

    def request(
        self,
        capability: str,
        method: str,
        path: str,
        *,
        json=None,
        form=None,
        headers=None,
        capture_headers=(),
    ):
        """Use one declared HTTP capability; return bounded status/text/JSON without redirects."""
        capability_name(capability)
        method = method.upper()
        if method not in {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"}:
            raise WorldUsageError("native_http_method_unsupported")
        if self._read_only and method not in {"GET", "HEAD"}:
            raise WorldUsageError("native_http_read_only")
        return self._request(
            capability,
            method,
            path,
            json=json,
            form=form,
            headers=headers,
            capture_headers=capture_headers,
        )

    def observe(self, name: str, parameters=None):
        """Invoke a sealed read operation; callers cannot replace its request shape."""
        if not isinstance(name, str) or name not in self._observations:
            raise WorldUsageError("native_http_observation_unknown")
        spec = self._observations[name]
        try:
            body = spec.render({} if parameters is None else parameters)
        except ValueError as exc:
            raise WorldUsageError(str(exc)) from None
        return self._request(spec.capability, "POST", spec.path, json=body)

    def _request(
        self,
        capability,
        method,
        path,
        *,
        json=None,
        form=None,
        headers=None,
        capture_headers=(),
    ):
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
        session = self._sessions.setdefault(capability, HttpSession())
        override = validate_headers(headers or {})
        merged = {
            key: value
            for key, value in session.headers.items()
            if key.lower() not in {name.lower() for name in override}
        }
        headers = validate_headers({**merged, **override})
        if (
            not isinstance(capture_headers, (list, tuple))
            or len(capture_headers) > 16
            or any(not isinstance(name, str) for name in capture_headers)
        ):
            raise WorldUsageError("native_http_capture_headers_invalid")
        validate_headers({name: "" for name in capture_headers})
        if json is not None and form is not None:
            raise WorldUsageError("native_http_body_encoding_conflict")
        if form is not None:
            if not isinstance(form, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in form.items()
            ):
                raise WorldUsageError("native_http_form_invalid")
            body = urlencode(form).encode("utf-8")
            if not any(name.lower() == "content-type" for name in headers):
                headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            body = None if json is None else json_module.dumps(json).encode("utf-8")
            if body is not None:
                if not any(name.lower() == "content-type" for name in headers):
                    headers["Content-Type"] = "application/json"
        if body is not None and len(body) > 65536:
            raise WorldUsageError("native_http_request_too_large")
        event = {
            "capability": capability,
            "method": method,
            "route_hash": hashlib.sha256(relative.path.encode()).hexdigest(),
        }
        url = urlunsplit((base.scheme, base.netloc, relative.path, relative.query, ""))
        session.remember(_sensitive_header_values(headers))
        session.remember(cookie.value for cookie in session.cookies if cookie.value)
        self._private_http_state()
        try:
            with httpx.Client(
                timeout=10,
                follow_redirects=False,
                trust_env=False,
                cookies=session.cookies,
            ) as client:
                with client.stream(
                    method, url, content=body, headers=headers
                ) as response:
                    event["status"] = response.status_code
                    session.cookies = client.cookies.jar
                    session.remember(
                        cookie.value for cookie in session.cookies if cookie.value
                    )
                    self._private_http_state()
                    selected_headers = {
                        name: response.headers[name]
                        for name in capture_headers
                        if name in response.headers
                    }
                    validate_headers(selected_headers)
                    session.remember(selected_headers.values())
                    self._private_http_state()
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
                        "headers": selected_headers,
                    }
        except (httpx.HTTPError, UnicodeError) as exc:
            event["error"] = type(exc).__name__
            raise WorldUsageError(
                f"native_http_request_failed: {type(exc).__name__}"
            ) from None
        except WorldUsageError as exc:
            event["error"] = type(exc).__name__
            raise
        finally:
            if len(self._http_events) < 128:
                self._http_events.append(event)

    def _execution_state(self):
        return {
            "kind": "native_observation",
            "world_index": self.world_index,
            "rng": self.rng.getstate(),
            "read_only": self._read_only,
            "endpoints": self._endpoints,
            "observations": {
                name: spec.model_dump(mode="json")
                for name, spec in self._observations.items()
            },
        }

    def _unsupported(self, *args, **kwargs):
        raise WorldUsageError(
            "native_observation: state access and harness tool execution are unsupported; observe native responses"
        )

    state = query = put = change = drop = call = _unsupported
    read_file = write_file = checkpoint = restore = handle_tool_call = _unsupported


def _sensitive_header_values(headers):
    return (
        value
        for name, value in headers.items()
        if any(
            marker in name.lower()
            for marker in ("authorization", "cookie", "token", "key")
        )
    )


def http_endpoints(runtime):
    return {
        name: endpoint.address
        for name, endpoint in runtime.endpoints.items()
        if endpoint.protocol in {"http", "https"}
    }


def load_observations(bundle_dir):
    from ..contract import AgentContract

    if bundle_dir is None:
        return {}
    runtime = AgentContract.model_validate_json(
        (bundle_dir / "contract.json").read_text()
    ).runtime
    return runtime.observations if runtime else {}


class ObservationWorldFactory:
    def __init__(self, bundle_dir=None):
        self._observations = load_observations(bundle_dir)

    async def create(self, runtime, *, rng):
        return ObservationWorld(
            runtime.world_index,
            rng,
            http_endpoints(runtime),
            observations=self._observations,
        )
