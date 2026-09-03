"""Tests for `livekit_tool_trace_bootstrap.py`'s LiveKit worker-isolation half: the harness
flips a W>1 worker into livekit-agents' own side-by-side mode (or a port/load-threshold
fallback) using only the `FI_WORKER_HEALTH_PORT` env var, without any change to the agent
under test.

livekit-agents is NOT installed in this test venv (see the CI split: the ALK suite must run
without the wheel). Every test below injects a `types.ModuleType` fake for `livekit.agents.worker`
into `sys.modules` before loading the bootstrap module fresh (via `importlib.util.spec_from_file_
location` under a unique module name per test, matching `test_scenario_source.py`'s pattern) so
the module's bottom-of-file `_install()` / `_install_worker_isolation()` calls run against a
controlled fake rather than a missing package. `test_livekit_worker_isolation_wheel_shape` is the
exception: it `pytest.importorskip`s the real wheel and is expected to skip in this venv,
exercised for real only via the scratch-venv probe recorded alongside the implementation.

Every fake's `run` is keyword-only (`*, devmode=False, unregistered=False`), matching the real
`AgentServer.run` signature, and writes exactly ONE attribute of its own -- `entry_state`,
captured ON ENTRY before returning -- rather than appending to a call log. This matters for two
things a weaker fake would hide: (1) checking `entry_state` (not the server's attributes after
`await` returns) proves the wrapper applies isolation BEFORE calling through, not merely that the
attributes end up right eventually by some other order; (2) because `entry_state` is the fake's
only self-inflicted mutation, a before/after `vars(server)` diff across `await server.run()`
reveals exactly what the isolation wrapper itself changed, with no risk of a call-log attribute
masking an accidental extra write.

`_apply_worker_isolation` gates every attribute write with TWO checks, and the fakes below are
split into two families to exercise each independently:

- `hasattr` first, so the shim never CREATES an attribute a wheel never had (`FakeAgentServer
  PortOnly` and `FakeAgentServerNoKnobs` model this: they leave `_simulation` off the class
  entirely, a plain slot-free object that would otherwise happily accept the assignment and
  silently invent it).
- A post-write re-read second, so a wheel that carries an attribute but rejects assignment (a
  `@property` with a raising setter) is not mistaken for one where the write actually took
  (`FakeAgentServerSimulationRejected` and `FakeAgentServerAllRejected` model this: `_simulation`
  exists and is readable, but every assignment to it raises).

Both families must fall through to the same weaker path, so most assertions apply to both; a few
tests are specific to one family precisely because the two former defects (inventing an attribute,
versus trusting an unverified write) are independent bugs with independent regressions.
"""

from __future__ import annotations

import asyncio
import importlib.util
import itertools
import logging
import math
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

_BOOTSTRAP_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "fi"
    / "alk"
    / "harness"
    / "livekit_tool_trace_bootstrap.py"
)

_MODULE_COUNTER = itertools.count()


def _install_fake_livekit(monkeypatch: pytest.MonkeyPatch, agent_server_cls: type) -> None:
    """Register a `livekit.agents.worker` fake (plus its parent packages) in
    `sys.modules`, restored by `monkeypatch` teardown."""
    livekit_mod = types.ModuleType("livekit")
    agents_mod = types.ModuleType("livekit.agents")
    worker_mod = types.ModuleType("livekit.agents.worker")
    worker_mod.AgentServer = agent_server_cls
    livekit_mod.agents = agents_mod
    agents_mod.worker = worker_mod
    monkeypatch.setitem(sys.modules, "livekit", livekit_mod)
    monkeypatch.setitem(sys.modules, "livekit.agents", agents_mod)
    monkeypatch.setitem(sys.modules, "livekit.agents.worker", worker_mod)


def _load_bootstrap(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load a fresh copy of the bootstrap module under a unique name, so its
    module-bottom `_install()`/`_install_worker_isolation()` calls run exactly
    once per test against whatever fake `sys.modules` state that test set up."""
    module_name = f"_test_livekit_tool_trace_bootstrap_{next(_MODULE_COUNTER)}"
    spec = importlib.util.spec_from_file_location(module_name, _BOOTSTRAP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


class FakeAgentServerFull:
    """Every knob is a plain, freely-settable instance attribute -- the wheel shape that
    fully supports the primary `_simulation` path."""

    def __init__(self) -> None:
        self._simulation = False
        self._port = 8081
        self._load_threshold = 0.0
        self._num_idle_processes = 0
        self.entry_state: tuple[Any, ...] | None = None

    async def run(self, *, devmode: bool = False, unregistered: bool = False) -> str:
        self.entry_state = (
            self._simulation,
            self._num_idle_processes,
            getattr(self, "_port", None),
            devmode,
            unregistered,
        )
        return "sentinel-return"


class FakeAgentServerRaisingRun(FakeAgentServerFull):
    """Same shape as `FakeAgentServerFull`, but `run` itself raises after recording
    `entry_state` -- the wrapper must propagate this, never swallow it."""

    async def run(self, *, devmode: bool = False, unregistered: bool = False) -> str:
        await super().run(devmode=devmode, unregistered=unregistered)
        raise RuntimeError("agent-under-test failure")


class FakeAgentServerSimulationRejected:
    """`_simulation` is readable (a property, always False) but every assignment to it
    raises -- models a wheel where the attribute exists but assignment does not take.
    `_port`/`_load_threshold` are plain, freely-settable attributes, so the fallback path
    is exercised. Being a property, `_simulation` never appears in `vars(instance)`."""

    def __init__(self) -> None:
        self._port = 8081
        self._load_threshold = 0.0
        self._num_idle_processes = 0
        self.entry_state: tuple[Any, ...] | None = None

    @property
    def _simulation(self) -> bool:
        return False

    @_simulation.setter
    def _simulation(self, value: bool) -> None:
        raise RuntimeError("wheel refuses simulation assignment")

    async def run(self, *, devmode: bool = False, unregistered: bool = False) -> str:
        self.entry_state = (
            self._simulation,
            self._num_idle_processes,
            getattr(self, "_port", None),
            devmode,
            unregistered,
        )
        return "sentinel-return"


class FakeAgentServerAllRejected:
    """Neither `_simulation` nor `_port`/`_load_threshold` assignment takes -- a wheel
    shape the shim has no working isolation path for at all."""

    def __init__(self) -> None:
        self._num_idle_processes = 0
        self.entry_state: tuple[Any, ...] | None = None

    @property
    def _simulation(self) -> bool:
        return False

    @_simulation.setter
    def _simulation(self, value: bool) -> None:
        raise RuntimeError("wheel refuses simulation assignment")

    @property
    def _port(self) -> int:
        return 8081

    @_port.setter
    def _port(self, value: int) -> None:
        raise RuntimeError("wheel refuses port override")

    @property
    def _load_threshold(self) -> float:
        return 0.0

    @_load_threshold.setter
    def _load_threshold(self, value: float) -> None:
        raise RuntimeError("wheel refuses load-threshold override")

    async def run(self, *, devmode: bool = False, unregistered: bool = False) -> str:
        self.entry_state = (
            self._simulation,
            self._num_idle_processes,
            getattr(self, "_port", None),
            devmode,
            unregistered,
        )
        return "sentinel-return"


class FakeAgentServerPortOnly:
    """Genuinely has NO `_simulation` attribute at all -- not a rejecting property, just
    absent -- the future-wheel shape the `_port` fallback exists for. `_port`/
    `_load_threshold`/`_num_idle_processes` are plain, freely-settable attributes. The
    shim must never invent `_simulation` on an object that never had it."""

    def __init__(self) -> None:
        self._port = 8081
        self._load_threshold = 0.0
        self._num_idle_processes = 0
        self.entry_state: tuple[Any, ...] | None = None

    async def run(self, *, devmode: bool = False, unregistered: bool = False) -> str:
        self.entry_state = (
            getattr(self, "_simulation", None),
            getattr(self, "_num_idle_processes", None),
            getattr(self, "_port", None),
            devmode,
            unregistered,
        )
        return "sentinel-return"


class FakeAgentServerNoKnobs:
    """None of the four knobs exist at all -- a wheel shape the shim has nothing to work
    with, and must not invent anything on."""

    def __init__(self) -> None:
        self.entry_state: tuple[Any, ...] | None = None

    async def run(self, *, devmode: bool = False, unregistered: bool = False) -> str:
        self.entry_state = (
            getattr(self, "_simulation", None),
            getattr(self, "_num_idle_processes", None),
            getattr(self, "_port", None),
            devmode,
            unregistered,
        )
        return "sentinel-return"


class FakeAgentServerSimulationTruthyNonBool:
    """`_simulation`'s setter genuinely stores the write, but the getter always returns `1`
    (truthy, but not `True`) -- verification must compare `is True`, not truthiness, or this
    would be wrongly accepted as "the write took"."""

    def __init__(self) -> None:
        self._port = 8081
        self._load_threshold = 0.0
        self._num_idle_processes = 0
        self._sim_backing: bool = False
        self.entry_state: tuple[Any, ...] | None = None

    @property
    def _simulation(self) -> int:
        return 1

    @_simulation.setter
    def _simulation(self, value: bool) -> None:
        self._sim_backing = value

    async def run(self, *, devmode: bool = False, unregistered: bool = False) -> str:
        self.entry_state = (
            self._simulation,
            self._num_idle_processes,
            getattr(self, "_port", None),
            devmode,
            unregistered,
        )
        return "sentinel-return"


class FakeAgentServerNoOpPortSetter:
    """NO `_simulation` at all. `_port`'s setter silently discards every write (no exception,
    the getter always returns the original value) -- the fallback's post-write re-read must
    catch this even though the assignment statement itself raised nothing. `_load_threshold`
    is a plain, freely-settable attribute: since both writes happen inside the SAME `try`
    block, a no-op (non-raising) `_port` setter does not stop the subsequent `_load_threshold`
    write from landing -- this fake exists to pin that real, accepted asymmetry down in a test
    rather than leave it as an unstated assumption."""

    def __init__(self) -> None:
        self._port_value = 8081
        self._load_threshold = 0.0
        self._num_idle_processes = 0
        self.entry_state: tuple[Any, ...] | None = None

    @property
    def _port(self) -> int:
        return self._port_value

    @_port.setter
    def _port(self, value: int) -> None:
        pass  # silently discarded: no exception, no effect

    async def run(self, *, devmode: bool = False, unregistered: bool = False) -> str:
        self.entry_state = (
            getattr(self, "_simulation", None),
            getattr(self, "_num_idle_processes", None),
            getattr(self, "_port", None),
            devmode,
            unregistered,
        )
        return "sentinel-return"


class FakeAgentServerNoIdleAttr:
    """A plain, freely-settable `_simulation` (so the primary path succeeds) but genuinely NO
    `_num_idle_processes` at all -- the shim must not invent it just because simulation mode
    otherwise succeeded."""

    def __init__(self) -> None:
        self._simulation = False
        self._port = 8081
        self._load_threshold = 0.0
        self.entry_state: tuple[Any, ...] | None = None

    async def run(self, *, devmode: bool = False, unregistered: bool = False) -> str:
        self.entry_state = (
            self._simulation,
            getattr(self, "_num_idle_processes", None),
            getattr(self, "_port", None),
            devmode,
            unregistered,
        )
        return "sentinel-return"


class FakeAgentServerWithoutRun:
    """No `run` attribute at all -- `_install_worker_isolation`'s wheel-shape guard must
    treat this the same as any other unsupported shape and skip silently."""

    async def serve(self, *args: Any, **kwargs: Any) -> str:
        return "served"


# --- T1: env absent / blank ---------------------------------------------------------------


def test_env_absent_leaves_run_unwrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FI_WORKER_HEALTH_PORT", raising=False)
    _install_fake_livekit(monkeypatch, FakeAgentServerFull)
    module = _load_bootstrap(monkeypatch)

    from livekit.agents.worker import AgentServer

    assert getattr(AgentServer.run, "__alk_worker_isolation__", None) is None

    server = AgentServer()
    result = module._apply_worker_isolation(server, {})
    assert result == "noop"
    assert server._simulation is False
    assert server._port == 8081
    assert server._load_threshold == 0.0
    assert server._num_idle_processes == 0


def test_blank_env_value_is_noop_and_leaves_run_unwrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A blank (present-but-empty) value must be treated identically to absent -- the
    # install-time gate and the pure function must agree on this, not just on outright
    # absence, since an authored-but-unrendered template value could land here as "".
    monkeypatch.setenv("FI_WORKER_HEALTH_PORT", "")
    _install_fake_livekit(monkeypatch, FakeAgentServerFull)
    module = _load_bootstrap(monkeypatch)

    from livekit.agents.worker import AgentServer

    assert getattr(AgentServer.run, "__alk_worker_isolation__", None) is None
    assert (
        module._apply_worker_isolation(AgentServer(), {"FI_WORKER_HEALTH_PORT": "   "})
        == "noop"
    )


# --- T2: env present, primary path ----------------------------------------------------------


def test_env_present_sets_simulation_and_awaits_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FI_WORKER_HEALTH_PORT", "18081")
    _install_fake_livekit(monkeypatch, FakeAgentServerFull)
    _load_bootstrap(monkeypatch)

    from livekit.agents.worker import AgentServer

    assert getattr(AgentServer.run, "__alk_worker_isolation__", None) is True

    server = AgentServer()
    result = asyncio.run(server.run(devmode=True, unregistered=True))

    assert result == "sentinel-return"
    # Checked via `entry_state` (captured ON ENTRY to the real `run`), not the server's
    # post-await attributes -- this is what actually proves isolation ran BEFORE the
    # original `run` was called, not merely that it ran at some point.
    assert server.entry_state == (True, 1, 8081, True, True)


# --- T3: fallback ------------------------------------------------------------------------


def test_apply_worker_isolation_falls_back_to_port_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_livekit(monkeypatch, FakeAgentServerSimulationRejected)
    module = _load_bootstrap(monkeypatch)

    server = FakeAgentServerSimulationRejected()
    env = {"FI_WORKER_HEALTH_PORT": "18081"}
    result = module._apply_worker_isolation(server, env)

    assert result == "port_override"
    assert server._port == 18081
    assert server._load_threshold == math.inf
    assert server._num_idle_processes == 1


# --- T4: unsupported shape -----------------------------------------------------------------


def test_apply_worker_isolation_unsupported_shape_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_livekit(monkeypatch, FakeAgentServerAllRejected)
    module = _load_bootstrap(monkeypatch)

    server = FakeAgentServerAllRejected()
    env = {"FI_WORKER_HEALTH_PORT": "18081"}
    result = module._apply_worker_isolation(server, env)

    assert result == "unsupported"
    assert server._simulation is False
    assert server._port == 8081  # unchanged default from the (rejecting) property
    assert server._num_idle_processes == 0  # never attempted in "unsupported" mode


# --- T5: idempotent ------------------------------------------------------------------------


def test_wrapping_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FI_WORKER_HEALTH_PORT", "18081")
    _install_fake_livekit(monkeypatch, FakeAgentServerFull)
    module = _load_bootstrap(monkeypatch)

    from livekit.agents.worker import AgentServer

    # Calling the installer again (as the module-bottom call already did once at
    # import time) must not wrap an already-wrapped `run` a second time.
    module._install_worker_isolation()
    module._install_worker_isolation()

    depth = 0
    wrapped = AgentServer.run
    while getattr(wrapped, "__wrapped__", None) is not None:
        depth += 1
        wrapped = wrapped.__wrapped__
    assert depth == 1


# --- T6: broken livekit import ---------------------------------------------------------------


def test_broken_livekit_import_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FI_WORKER_HEALTH_PORT", "18081")
    monkeypatch.setitem(sys.modules, "livekit", None)
    monkeypatch.setitem(sys.modules, "livekit.agents", None)
    monkeypatch.setitem(sys.modules, "livekit.agents.worker", None)

    # Must not raise at import time.
    module = _load_bootstrap(monkeypatch)
    assert module is not None


def test_broken_livekit_import_raising_module_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FI_WORKER_HEALTH_PORT", "18081")

    class ExplodingModule(types.ModuleType):
        def __getattr__(self, name: str) -> Any:
            raise RuntimeError("boom")

    livekit_mod = types.ModuleType("livekit")
    agents_mod = types.ModuleType("livekit.agents")
    worker_mod = ExplodingModule("livekit.agents.worker")
    livekit_mod.agents = agents_mod
    agents_mod.worker = worker_mod
    monkeypatch.setitem(sys.modules, "livekit", livekit_mod)
    monkeypatch.setitem(sys.modules, "livekit.agents", agents_mod)
    monkeypatch.setitem(sys.modules, "livekit.agents.worker", worker_mod)

    module = _load_bootstrap(monkeypatch)
    assert module is not None


def test_agent_server_without_run_is_skipped_silently(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A wheel shape with no `run` at all must not turn into an import-time error --
    # that would print on every interpreter this hook loads into, including every
    # per-job child process it spawns.
    monkeypatch.setenv("FI_WORKER_HEALTH_PORT", "18081")
    _install_fake_livekit(monkeypatch, FakeAgentServerWithoutRun)

    module = _load_bootstrap(monkeypatch)
    assert module is not None

    captured = capsys.readouterr()
    assert captured.err == ""


# --- T7: fallback hardcodes _load_threshold unconditionally ----------------------------------


def test_fallback_hardcodes_load_threshold_to_inf(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_livekit(monkeypatch, FakeAgentServerSimulationRejected)
    module = _load_bootstrap(monkeypatch)

    server = FakeAgentServerSimulationRejected()
    module._apply_worker_isolation(server, {"FI_WORKER_HEALTH_PORT": "18081"})

    assert server._load_threshold == math.inf


# --- T8: setattr failure is swallowed, and the fallback beneath it still runs -----------------


def test_simulation_rejected_falls_through_to_port_override_via_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FI_WORKER_HEALTH_PORT", "18081")
    _install_fake_livekit(monkeypatch, FakeAgentServerSimulationRejected)
    module = _load_bootstrap(monkeypatch)

    from livekit.agents.worker import AgentServer

    server = AgentServer()
    result = asyncio.run(server.run())

    assert result == "sentinel-return"
    assert server._port == 18081
    assert server._load_threshold == math.inf

    # Direct unit check of the pure function's return value too.
    direct_server = FakeAgentServerSimulationRejected()
    outcome = module._apply_worker_isolation(
        direct_server, {"FI_WORKER_HEALTH_PORT": "18081"}
    )
    assert outcome == "port_override"


def test_everything_rejected_is_unsupported_and_run_still_awaited(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("FI_WORKER_HEALTH_PORT", "18081")
    _install_fake_livekit(monkeypatch, FakeAgentServerAllRejected)
    module = _load_bootstrap(monkeypatch)

    from livekit.agents.worker import AgentServer

    server = AgentServer()
    with caplog.at_level(logging.WARNING, logger="fi.alk.harness.worker_isolation"):
        result = asyncio.run(server.run())

    assert result == "sentinel-return"
    assert server.entry_state is not None  # run was awaited despite total rejection
    assert server._simulation is False

    outcome = module._apply_worker_isolation(
        FakeAgentServerAllRejected(), {"FI_WORKER_HEALTH_PORT": "18081"}
    )
    assert outcome == "unsupported"

    messages = [record.getMessage() for record in caplog.records]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected an operator-usable WARNING when isolation could not be applied"
    assert any("could not be applied" in m for m in messages)
    assert not any("18081" in m for m in messages)  # keys only, never values


def test_unparseable_port_with_simulation_settable_is_still_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Port parsing is only ever consulted by the fallback -- a wheel that supports the
    # primary path doesn't care whether the harness's own port value is well-formed.
    _install_fake_livekit(monkeypatch, FakeAgentServerFull)
    module = _load_bootstrap(monkeypatch)

    server = FakeAgentServerFull()
    result = module._apply_worker_isolation(
        server, {"FI_WORKER_HEALTH_PORT": "not-a-number"}
    )

    assert result == "simulation"
    assert server._simulation is True


def test_unparseable_port_with_no_working_simulation_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_livekit(monkeypatch, FakeAgentServerSimulationRejected)
    module = _load_bootstrap(monkeypatch)

    server = FakeAgentServerSimulationRejected()
    result = module._apply_worker_isolation(
        server, {"FI_WORKER_HEALTH_PORT": "not-a-number"}
    )

    assert result == "unsupported"
    assert server._port == 8081  # untouched: parsing failed before any write attempt
    assert server._num_idle_processes == 0


def test_attribute_genuinely_absent_falls_back_without_inventing_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: a server that never had `_simulation` at all must never gain one.
    # A plain, slot-free Python object accepts `server._simulation = True` unconditionally
    # (it just creates the attribute) -- verifying the value afterward is not enough to
    # catch this, since the invented attribute reads back exactly as if the write "worked".
    # `_apply_worker_isolation` must refuse to attempt the write at all when `hasattr`
    # says the attribute was never there to begin with.
    monkeypatch.setenv("FI_WORKER_HEALTH_PORT", "18081")
    _install_fake_livekit(monkeypatch, FakeAgentServerPortOnly)
    module = _load_bootstrap(monkeypatch)

    from livekit.agents.worker import AgentServer

    server = AgentServer()
    result = asyncio.run(server.run())

    assert result == "sentinel-return"
    assert "_simulation" not in vars(server)
    assert server.entry_state == (None, 1, 18081, False, False)
    assert server._port == 18081
    assert server._load_threshold == math.inf
    assert server._num_idle_processes == 1

    # Direct unit check of the pure function's return value too.
    outcome = module._apply_worker_isolation(
        FakeAgentServerPortOnly(), {"FI_WORKER_HEALTH_PORT": "18081"}
    )
    assert outcome == "port_override"


def test_no_knobs_at_all_is_unsupported_and_invents_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("FI_WORKER_HEALTH_PORT", "18081")
    _install_fake_livekit(monkeypatch, FakeAgentServerNoKnobs)
    module = _load_bootstrap(monkeypatch)

    from livekit.agents.worker import AgentServer

    server = AgentServer()
    before = dict(vars(server))
    with caplog.at_level(logging.WARNING, logger="fi.alk.harness.worker_isolation"):
        result = asyncio.run(server.run())
    after = dict(vars(server))

    assert result == "sentinel-return"
    changed = {k for k in after if before.get(k, object()) != after[k]}
    assert changed == {"entry_state"}  # nothing invented
    assert not any(hasattr(server, attr) for attr in ("_simulation", "_port", "_load_threshold"))

    outcome = module._apply_worker_isolation(
        FakeAgentServerNoKnobs(), {"FI_WORKER_HEALTH_PORT": "18081"}
    )
    assert outcome == "unsupported"

    messages = [record.getMessage() for record in caplog.records]
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert any("could not be applied" in m for m in messages)


def test_simulation_verification_uses_is_true_not_truthiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A `_simulation` getter returning a truthy-but-not-`True` value (e.g. `1`) must NOT be
    # accepted as "the write took" -- verification compares `is True`, never plain truthiness.
    monkeypatch.setenv("FI_WORKER_HEALTH_PORT", "18081")
    _install_fake_livekit(monkeypatch, FakeAgentServerSimulationTruthyNonBool)
    module = _load_bootstrap(monkeypatch)

    from livekit.agents.worker import AgentServer

    server = AgentServer()
    result = asyncio.run(server.run())

    assert result == "sentinel-return"
    assert server._port == 18081
    assert server._load_threshold == math.inf

    outcome = module._apply_worker_isolation(
        FakeAgentServerSimulationTruthyNonBool(), {"FI_WORKER_HEALTH_PORT": "18081"}
    )
    assert outcome == "port_override"


def test_port_writes_are_verified_even_when_setter_does_not_raise(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A `_port` setter that silently discards the write (no exception) must still be caught
    # by the post-write re-read -- a write that "succeeded" (raised nothing) is not the same
    # as a write that actually landed.
    monkeypatch.setenv("FI_WORKER_HEALTH_PORT", "18081")
    _install_fake_livekit(monkeypatch, FakeAgentServerNoOpPortSetter)
    module = _load_bootstrap(monkeypatch)

    from livekit.agents.worker import AgentServer

    server = AgentServer()
    with caplog.at_level(logging.WARNING, logger="fi.alk.harness.worker_isolation"):
        result = asyncio.run(server.run())

    assert result == "sentinel-return"
    assert server._port == 8081  # the setter discarded the write; re-read caught it
    # Documenting the code's actual, accepted behavior: `_port` and `_load_threshold` are
    # written inside the SAME `try` block, so a `_port` setter that no-ops (rather than
    # raising) does not stop the subsequent `_load_threshold` write from landing, even though
    # the overall mode still correctly falls through to "unsupported" once `_port` fails its
    # own re-read.
    assert server._load_threshold == math.inf

    outcome = module._apply_worker_isolation(
        FakeAgentServerNoOpPortSetter(), {"FI_WORKER_HEALTH_PORT": "18081"}
    )
    assert outcome == "unsupported"

    messages = [record.getMessage() for record in caplog.records]
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert any("could not be applied" in m for m in messages)


def test_num_idle_processes_is_not_invented_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FI_WORKER_HEALTH_PORT", "18081")
    _install_fake_livekit(monkeypatch, FakeAgentServerNoIdleAttr)
    module = _load_bootstrap(monkeypatch)

    from livekit.agents.worker import AgentServer

    server = AgentServer()
    result = asyncio.run(server.run())

    assert result == "sentinel-return"
    assert server._simulation is True
    assert "_num_idle_processes" not in vars(server)

    outcome = module._apply_worker_isolation(
        FakeAgentServerNoIdleAttr(), {"FI_WORKER_HEALTH_PORT": "18081"}
    )
    assert outcome == "simulation"


def test_wrapped_run_is_still_a_coroutine_function(monkeypatch: pytest.MonkeyPatch) -> None:
    # If the wrapper regressed to a plain `def` (rather than `async def`), calling it could
    # still incidentally behave correctly by returning the original coroutine unawaited -- so
    # this must be checked structurally, in the fake-based suite (not only under the real
    # wheel in T9), rather than relying on behavior alone to surface it.
    monkeypatch.setenv("FI_WORKER_HEALTH_PORT", "18081")
    _install_fake_livekit(monkeypatch, FakeAgentServerFull)
    _load_bootstrap(monkeypatch)

    from livekit.agents.worker import AgentServer

    assert asyncio.iscoroutinefunction(AgentServer.run)


# --- propagation: a raising `run` is never swallowed ------------------------------------------


def test_original_run_exception_propagates_through_the_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FI_WORKER_HEALTH_PORT", "18081")
    _install_fake_livekit(monkeypatch, FakeAgentServerRaisingRun)
    _load_bootstrap(monkeypatch)

    from livekit.agents.worker import AgentServer

    server = AgentServer()
    with pytest.raises(RuntimeError, match="agent-under-test failure"):
        asyncio.run(server.run())

    # Isolation still ran, and the original `run` was still entered, before the raise.
    assert server.entry_state is not None
    assert server._simulation is True


# --- non-interference: isolation changes exactly what it says it changes, nothing else -------


def test_simulation_mode_changes_only_the_documented_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FI_WORKER_HEALTH_PORT", "18081")
    _install_fake_livekit(monkeypatch, FakeAgentServerFull)
    _load_bootstrap(monkeypatch)

    from livekit.agents.worker import AgentServer

    server = AgentServer()
    before = dict(vars(server))
    asyncio.run(server.run())
    after = dict(vars(server))

    changed = {k for k in after if before.get(k, object()) != after[k]}
    # "entry_state" is the fake's own single self-inflicted write (see module docstring);
    # the other two are exactly what simulation-mode isolation is documented to touch.
    assert changed == {"_simulation", "_num_idle_processes", "entry_state"}


def test_port_override_mode_changes_only_the_documented_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FI_WORKER_HEALTH_PORT", "18081")
    _install_fake_livekit(monkeypatch, FakeAgentServerSimulationRejected)
    _load_bootstrap(monkeypatch)

    from livekit.agents.worker import AgentServer

    server = AgentServer()
    before = dict(vars(server))
    asyncio.run(server.run())
    after = dict(vars(server))

    changed = {k for k in after if before.get(k, object()) != after[k]}
    assert changed == {"_port", "_load_threshold", "_num_idle_processes", "entry_state"}


def test_port_only_fake_changes_only_the_documented_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Same assertion as above, but against a server that genuinely never had `_simulation`
    # (rather than one where it exists as a rejecting property) -- the regression case: an
    # invented `_simulation` key would break this exact-set assertion.
    monkeypatch.setenv("FI_WORKER_HEALTH_PORT", "18081")
    _install_fake_livekit(monkeypatch, FakeAgentServerPortOnly)
    _load_bootstrap(monkeypatch)

    from livekit.agents.worker import AgentServer

    server = AgentServer()
    before = dict(vars(server))
    asyncio.run(server.run())
    after = dict(vars(server))

    changed = {k for k in after if before.get(k, object()) != after[k]}
    assert changed == {"_port", "_load_threshold", "_num_idle_processes", "entry_state"}
    assert "_simulation" not in after


# --- T9: real-wheel shape probe (skipped in this venv) ----------------------------------------


def test_livekit_worker_isolation_wheel_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("livekit.agents")
    from livekit.agents.worker import AgentServer

    monkeypatch.setenv("FI_WORKER_HEALTH_PORT", "18081")

    module = _load_bootstrap(monkeypatch)

    server = AgentServer()
    for attr in ("_simulation", "_port", "_load_threshold", "_num_idle_processes"):
        assert hasattr(server, attr)

    assert asyncio.iscoroutinefunction(AgentServer.run)

    result = module._apply_worker_isolation(server, dict(os.environ))
    assert result == "simulation"
    assert server._simulation is True


# --- logging: INFO on success, WARNING on unsupported, silence on noop -----------------------


def test_apply_worker_isolation_logs_mode_without_port_value(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _install_fake_livekit(monkeypatch, FakeAgentServerFull)
    module = _load_bootstrap(monkeypatch)

    server = FakeAgentServerFull()
    with caplog.at_level(logging.INFO, logger="fi.alk.harness.worker_isolation"):
        module._apply_worker_isolation(server, {"FI_WORKER_HEALTH_PORT": "18081"})

    messages = [record.getMessage() for record in caplog.records]
    assert any("simulation" in message for message in messages)
    assert not any("18081" in message for message in messages)


def test_apply_worker_isolation_does_not_log_on_noop(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _install_fake_livekit(monkeypatch, FakeAgentServerFull)
    module = _load_bootstrap(monkeypatch)

    server = FakeAgentServerFull()
    with caplog.at_level(logging.INFO, logger="fi.alk.harness.worker_isolation"):
        module._apply_worker_isolation(server, {})

    assert caplog.records == []
