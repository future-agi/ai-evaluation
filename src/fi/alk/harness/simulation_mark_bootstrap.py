"""Startup hook that stamps the simulation mark on every OpenTelemetry resource a process builds.

The sandbox copies this file into a customer process's world directory beside a
``sitecustomize.py`` that calls :func:`install`, so it runs inside the CUSTOMER's interpreter,
which has no ``fi``. It therefore uses the standard library only, imports nothing of
OpenTelemetry itself, and is written for Python 3.9 syntax.

Why it exists: OpenTelemetry's own environment parsing reaches ``Resource.create`` only. Code
that builds ``Resource(attributes=...)`` directly, or that names its own project, never sees
``OTEL_RESOURCE_ATTRIBUTES``, so its traces would carry neither the mark that hides test traces
from production views nor the project they must land in. This hook wraps the one constructor
every resource goes through and merges those pairs over whatever the caller passed.

When the launcher also hands the process the agent's ingest key, the hook makes the two base
OTLP span exporters, and traceAI's exporters built on them, send to our collector with that key
instead of the customer's own ``FI_API_KEY``: only the Future AGI authentication headers are
replaced, only on exports to the collector, and the customer's key stays in the environment for
the rest of their code.

When the sandbox names a call file, every span the process starts is also stamped with the id of
the call the sandbox is running at that moment. The stamp rides on the SDK tracer rather than a
span processor, because a provider subclass such as traceAI's may drop processors it did not add.

It never reads or writes the customer's files and never imports their modules on its own. It does
run the next ``sitecustomize`` on the path, which is theirs, so a repository's own startup hook is
not lost behind this one; a failure of that hook is reported as Python reports it, never as ours.
A failure of this patch leaves the process as it would be without the hook and writes one fixed
line to stderr.
"""

from __future__ import annotations

import collections.abc
import functools
import importlib.machinery
import importlib.util
import inspect
import os
import sys
import urllib.parse

_RESOURCES = "opentelemetry.sdk.resources"
_TRACE = "opentelemetry.sdk.trace"
_CALL_FILE = "FI_SIMULATION_CALL_FILE"
_CALL_KEY = "fi.simulator.call_execution_id"
_EXPORTERS = {
    "opentelemetry.exporter.otlp.proto.http.trace_exporter": False,
    "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": True,
}
_INGEST_API_KEY = "FI_SIMULATION_INGEST_API_KEY"
_INGEST_SECRET_KEY = "FI_SIMULATION_INGEST_SECRET_KEY"
_TRACES_ENDPOINT = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
_GENERIC_ENDPOINT = "OTEL_EXPORTER_OTLP_ENDPOINT"
_TRACES_HEADERS = "OTEL_EXPORTER_OTLP_TRACES_HEADERS"
_GENERIC_HEADERS = "OTEL_EXPORTER_OTLP_HEADERS"
_AUTH_HEADERS = ("x-api-key", "x-secret-key")
_DEFAULT_PORTS = {"http": 80, "https": 443}
_MARK_PREFIX = "fi.simulation."
_ORIGIN_KEY = "fi.simulation.origin"
_PROJECT_KEY = "project_name"
_PATCHED = "_alk_simulation_mark"
_LOADER = "_alk_simulation_mark_loader"
_FINDER = "_alk_simulation_mark_finder"
_CHAINED_NAME = "_alk_chained_sitecustomize"

_installed = False
_failed_written = False
_PATCHES: dict = {}


def _write(outcome: str) -> None:
    """Write the one fixed stderr line. ``failed`` at most once, so a broken merge on every
    resource the process builds cannot flood its log."""
    global _failed_written
    try:
        if outcome == "failed":
            if _failed_written:
                return
            _failed_written = True
        sys.stderr.write(f"simulation mark hook: {outcome}\n")
    except Exception:
        pass


def _read_pairs() -> dict[str, str] | None:
    """The pairs of ``OTEL_RESOURCE_ATTRIBUTES`` that are ours, parsed as the SDK parses the
    variable. The customer's own pairs are left to the SDK, whose precedence is already defined."""
    raw = os.environ.get("OTEL_RESOURCE_ATTRIBUTES")
    if not raw:
        return None
    pairs: dict[str, str] = {}
    for item in raw.split(","):
        key, separator, value = item.partition("=")
        if not separator:
            continue
        key = key.strip()
        if key == _PROJECT_KEY or key.startswith(_MARK_PREFIX):
            pairs[key] = urllib.parse.unquote(value.strip())
    if _ORIGIN_KEY not in pairs:
        return None
    return pairs


def _patch(module, pairs: dict[str, str]) -> None:
    cls = module.Resource
    original = cls.__init__
    if getattr(original, _PATCHED, False):
        return

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        new_args, new_kwargs = args, kwargs
        try:
            # Only attributes that can be merged are touched. Any other call reaches the
            # constructor as the caller wrote it, so it behaves as it would without the hook.
            if args:
                supplied = args[0]
                if supplied is None or isinstance(supplied, collections.abc.Mapping):
                    merged = dict(supplied or {})
                    merged.update(pairs)
                    new_args = (merged,) + args[1:]
            elif "attributes" in kwargs:
                supplied = kwargs["attributes"]
                if supplied is None or isinstance(supplied, collections.abc.Mapping):
                    merged = dict(supplied or {})
                    merged.update(pairs)
                    new_kwargs = dict(kwargs, attributes=merged)
        except Exception:
            _write("failed")
        # Outside the try: a failure of the original constructor is the caller's, not ours.
        return original(self, *new_args, **new_kwargs)

    setattr(wrapper, _PATCHED, True)
    cls.__init__ = wrapper
    _write("applied")


def _wrap_loader(loader) -> None:
    if getattr(loader, _LOADER, False):
        return
    exec_module = getattr(loader, "exec_module", None)
    if exec_module is None:
        # A third-party import hook written before ``exec_module`` existed still loads modules
        # with ``load_module``; the patch cannot land there, and the log must say the SDK is
        # going to load unmarked.
        _write("failed")
        return

    def patching_exec_module(module):
        exec_module(module)
        # One zipimporter serves every module of its archive, so the name is looked up here.
        _apply(module)

    loader.exec_module = patching_exec_module
    setattr(loader, _LOADER, True)


def _apply(module) -> None:
    try:
        patch = _PATCHES.get(module.__name__)
        if patch is not None:
            patch(module)
    except Exception:
        _write("failed")


class _MarkFinder:
    """Sits first on ``sys.meta_path`` for the life of the process, so the SDK is patched when
    it is imported later and again when it is reloaded."""

    def __init__(self) -> None:
        setattr(self, _FINDER, True)

    def find_spec(self, fullname, path=None, target=None):
        # Not removed after the first import: live-reload dev servers and test runners call
        # ``importlib.reload``, which builds a new constructor that must be patched again.
        if fullname not in _PATCHES:
            return None
        spec = None
        try:
            for finder in list(sys.meta_path):
                # The module can run twice under two names (the world directory copy and a copy
                # on the customer's own path); two finders would otherwise call each other until
                # ``RecursionError``.
                if finder is self or getattr(finder, _FINDER, False):
                    continue
                find_spec = getattr(finder, "find_spec", None)
                if find_spec is None:
                    continue
                try:
                    spec = find_spec(fullname, path, target)
                except Exception:
                    # Not ours: the import system reaches the same finder next and reports it.
                    return None
                if spec is not None:
                    break
            if spec is not None:
                _wrap_loader(spec.loader)
        except Exception:
            _write("failed")
        return spec


def _read_ingest() -> tuple[list[tuple[str, str]], tuple[str, int | None]] | None:
    api_key = os.environ.get(_INGEST_API_KEY)
    secret_key = os.environ.get(_INGEST_SECRET_KEY)
    collector = _host_port(os.environ.get(_TRACES_ENDPOINT))
    if not api_key or not secret_key or collector is None:
        return None
    return [(_AUTH_HEADERS[0], api_key), (_AUTH_HEADERS[1], secret_key)], collector


def _host_port(endpoint) -> tuple[str, int | None] | None:
    if not endpoint:
        return None
    parts = urllib.parse.urlsplit(endpoint if "://" in endpoint else f"//{endpoint}")
    host = (parts.hostname or "").lower()
    if not host:
        return None
    return host, parts.port or _DEFAULT_PORTS.get(parts.scheme)


def _header_entries(headers) -> list[tuple[str, str]]:
    if headers is None:
        return []
    if isinstance(headers, str):
        entries = []
        for item in headers.split(","):
            name, separator, value = item.partition("=")
            if separator and name.strip():
                entries.append(
                    (
                        urllib.parse.unquote(name.strip()),
                        urllib.parse.unquote(value.strip()),
                    )
                )
        return entries
    if isinstance(headers, collections.abc.Mapping):
        return list(headers.items())
    if isinstance(headers, (list, tuple)):
        return [(name, value) for name, value in headers]
    raise TypeError("unsupported headers")


def _environment_headers() -> list[tuple[str, str]]:
    merged = {
        name.lower(): value
        for name, value in _header_entries(os.environ.get(_GENERIC_HEADERS))
    }
    merged.update(
        (name.lower(), value)
        for name, value in _header_entries(os.environ.get(_TRACES_HEADERS))
    )
    return list(merged.items())


def _with_ingest(
    entries: list[tuple[str, str]], ours: list[tuple[str, str]], lower: bool
) -> dict[str, str]:
    kept = [
        (name, value) for name, value in entries if name.lower() not in _AUTH_HEADERS
    ]
    return {(name.lower() if lower else name): value for name, value in kept + ours}


def _without_auth(bound, supplied) -> None:
    """The fail-closed form of the headers: the customer's own key is replaced by blank ones, so
    our collector refuses the span instead of storing it under the customer's project."""
    try:
        entries = (
            _environment_headers() if supplied is None else _header_entries(supplied)
        )
    except (TypeError, ValueError):
        entries = []
    kept = [
        (name, value) for name, value in entries if name.lower() not in _AUTH_HEADERS
    ]
    bound.arguments["headers"] = dict(kept + [(name, "") for name in _AUTH_HEADERS])


def _patch_exporter(module, ingest, lower: bool) -> None:
    cls = module.OTLPSpanExporter
    original = cls.__init__
    if getattr(original, _PATCHED, False):
        return
    signature = inspect.signature(original)
    ours, collector = ingest

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        call_args, call_kwargs = (self,) + args, kwargs
        aimed_at_us = None
        try:
            bound = signature.bind(self, *args, **kwargs)
            endpoint = (
                bound.arguments.get("endpoint")
                or os.environ.get(_TRACES_ENDPOINT)
                or os.environ.get(_GENERIC_ENDPOINT)
            )
            if _host_port(endpoint) == collector:
                aimed_at_us = bound
                supplied = bound.arguments.get("headers")
                entries = (
                    _environment_headers()
                    if supplied is None
                    else _header_entries(supplied)
                )
                bound.arguments["headers"] = _with_ingest(entries, ours, lower)
                call_args, call_kwargs = bound.args, bound.kwargs
        except Exception:
            _write("failed")
            if aimed_at_us is not None:
                try:
                    _without_auth(aimed_at_us, aimed_at_us.arguments.get("headers"))
                    call_args, call_kwargs = aimed_at_us.args, aimed_at_us.kwargs
                except Exception:
                    pass
        return original(*call_args, **call_kwargs)

    setattr(wrapper, _PATCHED, True)
    cls.__init__ = wrapper
    _write("applied")


class _CallFile:
    def __init__(self, path: str) -> None:
        self._path = path
        self._seen = None
        self._call = None

    def current(self):
        try:
            stat = os.stat(self._path)
        except FileNotFoundError:
            return None
        seen = (stat.st_ino, stat.st_mtime_ns)
        if seen != self._seen:
            with open(self._path, encoding="utf-8") as handle:
                self._call = handle.read().strip() or None
            self._seen = seen
        return self._call


def _patch_tracer(module, path: str) -> None:
    cls = module.Tracer
    original = cls.start_span
    if getattr(original, _PATCHED, False):
        return

    call_file = _CallFile(path)

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        span = original(self, *args, **kwargs)
        try:
            call = call_file.current()
            if call and span.is_recording():
                span.set_attribute(_CALL_KEY, call)
        except Exception:
            pass
        return span

    setattr(wrapper, _PATCHED, True)
    cls.start_span = wrapper
    _write("applied")


def _report_chained_failure(err: Exception) -> None:
    """Say what Python itself says when a ``sitecustomize`` fails, in its words, so the owner of
    that hook still learns of the failure. It is theirs, not a failure of our patch, so it is
    never our ``failed`` line."""
    try:
        if sys.flags.verbose:
            sys.excepthook(*sys.exc_info())
        else:
            sys.stderr.write(
                "Error in sitecustomize; set PYTHONVERBOSE for traceback:\n%s: %s\n"
                % (err.__class__.__name__, err)
            )
    except Exception:
        pass


def _chain() -> None:
    """Run the next ``sitecustomize`` on the path: ours sits first, so a repository's or the
    environment's own hook would otherwise never run."""
    try:
        own_dir = os.path.dirname(os.path.realpath(__file__))
        own_hook = os.path.realpath(os.path.join(own_dir, "sitecustomize.py"))
        remaining = None
        for index, entry in enumerate(sys.path):
            try:
                if os.path.realpath(entry) == own_dir:
                    remaining = sys.path[index + 1 :]
                    break
            except Exception:
                continue
        if remaining is None:
            return
        for entry in remaining:
            try:
                spec = importlib.machinery.PathFinder.find_spec(
                    "sitecustomize", [entry]
                )
                origin = getattr(spec, "origin", None)
                # The world directory may sit on the path twice; our own file is not "next".
                if not origin or os.path.realpath(origin) == own_hook:
                    continue
                chained = importlib.util.spec_from_file_location(_CHAINED_NAME, origin)
                module = importlib.util.module_from_spec(chained)
            except Exception:
                continue
            sys.modules[_CHAINED_NAME] = module
            try:
                chained.loader.exec_module(module)
            except BaseException as err:
                sys.modules.pop(_CHAINED_NAME, None)
                if not isinstance(err, Exception):
                    raise
                _report_chained_failure(err)
            return
    except Exception:
        # Looking for the next hook must never cost the process its start.
        pass


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True
    try:
        pairs = _read_pairs()
        if pairs is not None:
            _PATCHES[_RESOURCES] = functools.partial(_patch, pairs=pairs)
        call_file = os.environ.get(_CALL_FILE)
        if call_file:
            _PATCHES[_TRACE] = functools.partial(_patch_tracer, path=call_file)
        ingest = None
        try:
            ingest = _read_ingest()
        except ValueError:
            _write("failed")
        if ingest is not None:
            for name, lower in _EXPORTERS.items():
                _PATCHES[name] = functools.partial(
                    _patch_exporter, ingest=ingest, lower=lower
                )
        if not _PATCHES:
            return
        # The finder goes in first so that a failed patch of an already imported module still
        # leaves a later reload to be patched.
        sys.meta_path.insert(0, _MarkFinder())
        for name in list(_PATCHES):
            if name in sys.modules:
                _apply(sys.modules[name])
    except Exception:
        _write("failed")
    finally:
        _chain()
