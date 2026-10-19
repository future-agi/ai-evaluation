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
import os
import sys
import urllib.parse

_TARGET = "opentelemetry.sdk.resources"
_MARK_PREFIX = "fi.simulation."
_ORIGIN_KEY = "fi.simulation.origin"
_PROJECT_KEY = "project_name"
_PATCHED = "_alk_simulation_mark"
_LOADER = "_alk_simulation_mark_loader"
_FINDER = "_alk_simulation_mark_finder"
_CHAINED_NAME = "_alk_chained_sitecustomize"

_installed = False
_failed_written = False


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


def _wrap_loader(loader, pairs: dict[str, str]) -> None:
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
        try:
            # One zipimporter serves every module of its archive, so the name is tested here.
            if module.__name__ == _TARGET:
                _patch(module, pairs)
        except Exception:
            _write("failed")

    loader.exec_module = patching_exec_module
    setattr(loader, _LOADER, True)


class _ResourceFinder:
    """Sits first on ``sys.meta_path`` for the life of the process, so the SDK is patched when
    it is imported later and again when it is reloaded."""

    def __init__(self, pairs: dict[str, str]) -> None:
        self._pairs = pairs
        setattr(self, _FINDER, True)

    def find_spec(self, fullname, path=None, target=None):
        # Not removed after the first import: live-reload dev servers and test runners call
        # ``importlib.reload``, which builds a new constructor that must be patched again.
        if fullname != _TARGET:
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
                _wrap_loader(spec.loader, self._pairs)
        except Exception:
            _write("failed")
        return spec


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
        if pairs is None:
            return
        # The finder goes in first so that a failed patch of an already imported SDK still
        # leaves a later reload to be patched.
        sys.meta_path.insert(0, _ResourceFinder(pairs))
        if _TARGET in sys.modules:
            _patch(sys.modules[_TARGET], pairs)
    except Exception:
        _write("failed")
    finally:
        _chain()
