"""C3 dispatch-ack ladder — the engine half (Track B-ack), c3-call-affinity.md v0.4 §4.3/§4.5.

The ladder logic lives in `engines/livekit.py`, whose module import requires the optional
`livekit` dependency. When the real wheel is absent we install a lightweight `sys.modules` stub so
the ladder's timing/scheduling code — the part C3 calls "the hard part" — runs for real here; when
the real wheel IS present (CI's livekit lane) the stub is skipped and these same tests exercise the
genuine module. Either way the ladder is driven through injected fakes: no real LiveKit server, no
`rtc.Room`, no `AgentSession`.

Convention (`test_livekit_engine.py`): `asyncio.run` drives every `async def` seam directly.
"""

from __future__ import annotations

import asyncio
import importlib.abc
import importlib.machinery
import sys
import time
import types
from types import SimpleNamespace

import pytest


# --- livekit stub (only when the real optional dependency is absent) --------------------------
def _install_livekit_stub():
    class _Dummy:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return _Dummy()

        def __getattr__(self, name):
            return _Dummy()

        def __iter__(self):
            return iter(())

        def __mro_entries__(self, bases):
            return (object,)

    class _StubModule(types.ModuleType):
        __path__: list[str] = []

        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _Dummy()

    class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "livekit" or fullname.startswith("livekit."):
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
            return None

        def create_module(self, spec):
            module = _StubModule(spec.name)
            module.__path__ = []
            return module

        def exec_module(self, module):
            pass

    finder = _Finder()
    sys.meta_path.insert(0, finder)
    return finder


_STUB_FINDER = None
try:  # pragma: no cover - exercised by whichever branch matches the environment
    import livekit  # noqa: F401
except Exception:  # noqa: BLE001
    _STUB_FINDER = _install_livekit_stub()

from fi.simulate.simulation.engines import livekit as lk  # noqa: E402
from fi.simulate.simulation.engines.livekit import (  # noqa: E402
    DISPATCH_ACK_MARKS,
    DispatchUnacknowledgedError,
    _dispatch_ack_enabled,
    _run_dispatch_ack_ladder,
)

if _STUB_FINDER is not None:
    # The engine module (and everything it imported) now holds its own references to the stub
    # symbols, so tear the global stub back down: remove the finder and drop the `livekit*`
    # sys.modules entries. Otherwise a sibling module's `pytest.importorskip("livekit")` would find
    # the stub and RUN instead of SKIP in a livekit-less env. `lk` keeps working — it never
    # re-imports livekit at call time.
    sys.meta_path = [f for f in sys.meta_path if f is not _STUB_FINDER]
    for _name in [n for n in list(sys.modules) if n == "livekit" or n.startswith("livekit.")]:
        del sys.modules[_name]

# Small marks keep the real-clock timing tests fast while preserving generous headroom over the
# sub-millisecond fake dispatch ops (~150ms between marks). The +N ordering is what matters.
_MARKS = (0.15, 0.30, 0.45)
_TARGET = SimpleNamespace(identity="agent-participant")


# --- fakes ------------------------------------------------------------------------------------
class _Joiner:
    """Models `await_join(timeout)`: returns the target if the agent joins within `timeout`, else
    None on timeout. `join_at` is an absolute offset (seconds) from the first call."""

    def __init__(self, join_at: float | None = None) -> None:
        self.join_at = join_at
        self.start: float | None = None
        self.calls = 0

    async def __call__(self, timeout: float):
        self.calls += 1
        now = time.monotonic()
        if self.start is None:
            self.start = now
        if self.join_at is not None:
            remaining = (self.start + self.join_at) - now
            if remaining <= timeout:
                await asyncio.sleep(max(remaining, 0.0))
                return _TARGET
        await asyncio.sleep(max(timeout, 0.0))
        return None


class _Ops:
    def __init__(
        self,
        *,
        listed: list | None = None,
        list_error: bool = False,
        delete_error: bool = False,
        create_hang: bool = False,
        create_error: bool = False,
    ) -> None:
        self.listed = list(listed or [])
        self.list_error = list_error
        self.delete_error = delete_error
        self.create_hang = create_hang
        self.create_error = create_error
        self.list_calls = 0
        self.creates = 0
        self.deletes: list = []

    async def list_dispatches(self):
        self.list_calls += 1
        if self.list_error:
            raise RuntimeError("ListAgentDispatch boom")
        return list(self.listed)

    async def delete_dispatch(self, dispatch) -> None:
        self.deletes.append(dispatch)
        if self.delete_error:
            raise RuntimeError("DeleteAgentDispatch boom")
        try:
            self.listed.remove(dispatch)
        except ValueError:
            pass

    async def create_dispatch(self) -> None:
        self.creates += 1
        if self.create_hang:
            await asyncio.sleep(3600)  # never returns inside the window
        if self.create_error:
            raise RuntimeError("create_dispatch boom")


def _name_of(d) -> str | None:
    return getattr(d, "agent_name", None)


def _ladder(ops: _Ops, joiner: _Joiner, *, agent_name: str = "agent-x"):
    first = time.monotonic()
    return _run_dispatch_ack_ladder(
        await_join=joiner,
        list_dispatches=ops.list_dispatches,
        delete_dispatch=ops.delete_dispatch,
        create_dispatch=ops.create_dispatch,
        dispatch_name_of=_name_of,
        agent_name=agent_name,
        room_name="room-1",
        first_dispatch_at=first,
        marks=_MARKS,
    )


# --- gate + constants -------------------------------------------------------------------------
def test_default_marks_are_20_40_60() -> None:
    assert DISPATCH_ACK_MARKS == (20.0, 40.0, 60.0)


def test_gate_is_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("FI_HOSTED_DISPATCH_ACK", raising=False)
    assert _dispatch_ack_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_gate_on_for_truthy_values(monkeypatch, value) -> None:
    monkeypatch.setenv("FI_HOSTED_DISPATCH_ACK", value)
    assert _dispatch_ack_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "", "no", "off", "garbage"])
def test_gate_off_for_non_truthy_values(monkeypatch, value) -> None:
    monkeypatch.setenv("FI_HOSTED_DISPATCH_ACK", value)
    assert _dispatch_ack_enabled() is False


def test_authored_hosted_bundle_value_arms_the_gate(monkeypatch) -> None:
    # D32 cross-track seam (A′ bundle author -> B-ack engine gate): the value the hosted bundle
    # author writes into the LiveKit control worker's environment must be the EXACT env var name
    # the engine reads, at a value the gate accepts as armed. Pull the authored knob env straight
    # from `bundle_author_v2._worker_knob_env` (the hosted bundle producer) and drive it through
    # the engine's own gate, so authoring and runtime cannot silently drift apart.
    from fi.alk.harness.bundle_author_v2 import _worker_knob_env

    authored = _worker_knob_env("agent")
    assert lk._DISPATCH_ACK_ENV == "FI_HOSTED_DISPATCH_ACK"
    assert lk._DISPATCH_ACK_ENV in authored
    monkeypatch.setenv(lk._DISPATCH_ACK_ENV, authored[lk._DISPATCH_ACK_ENV])
    assert _dispatch_ack_enabled() is True


# --- exhaustion exception is typed, structured, and NOT a TimeoutError ------------------------
def test_exhaustion_error_is_not_asyncio_timeout() -> None:
    # The whole point of the ladder's own exception: it must NOT be an asyncio.TimeoutError, which
    # would map to AGENT_UNAVAILABLE -> WorldUnavailable -> world retirement (C3 §4.3 step 4).
    assert not issubclass(DispatchUnacknowledgedError, asyncio.TimeoutError)
    exc = DispatchUnacknowledgedError(
        room_name="r", agent_name="a", attempts=3, marks=(20.0, 40.0, 60.0)
    )
    assert not isinstance(exc, asyncio.TimeoutError)


def test_exhaustion_marker_is_structured_code() -> None:
    exc = DispatchUnacknowledgedError(
        room_name="r", agent_name="a", attempts=3, marks=_MARKS
    )
    assert exc.marker == "voice_dispatch_unacknowledged"
    assert exc.room_name == "r"
    assert exc.agent_name == "a"
    assert exc.attempts == 3
    assert exc.marks == _MARKS


def test_engine_maps_marker_to_the_receipt_code() -> None:
    # The engine's except-handler feeds exc.marker straight into failure.code.
    exc = DispatchUnacknowledgedError(
        room_name="r", agent_name="a", attempts=3, marks=_MARKS
    )
    outcome = lk._failure_outcome(
        lk.TestCaseStatus.FAILED,
        lk.FailureStage.READINESS,
        exc.marker,
        "not acked",
        retryable=True,
    )
    assert outcome.failure.code == "voice_dispatch_unacknowledged"
    assert outcome.status is not lk.TestCaseStatus.AGENT_UNAVAILABLE


# --- ladder behaviour -------------------------------------------------------------------------
def test_join_before_first_mark_returns_target_without_redispatch() -> None:
    ops = _Ops()
    joiner = _Joiner(join_at=0.05)  # before the +0.15 mark
    result = asyncio.run(_ladder(ops, joiner))
    assert result is _TARGET
    assert ops.creates == 0
    assert ops.list_calls == 0
    assert ops.deletes == []


def test_never_joins_redispatches_twice_then_exhausts() -> None:
    ops = _Ops()
    joiner = _Joiner(join_at=None)
    with pytest.raises(DispatchUnacknowledgedError) as ei:
        asyncio.run(_ladder(ops, joiner))
    # attempts 2 and 3 are the re-creations at +0.15 and +0.30; initial dispatch is attempt 1.
    assert ops.creates == 2
    assert ei.value.attempts == 3
    assert ei.value.marker == "voice_dispatch_unacknowledged"


def test_join_between_marks_stops_after_one_redispatch() -> None:
    ops = _Ops()
    joiner = _Joiner(join_at=0.20)  # between +0.15 and +0.30
    result = asyncio.run(_ladder(ops, joiner))
    assert result is _TARGET
    assert ops.creates == 1  # exactly one re-dispatch, none after the join edge


def test_lists_before_redispatch_and_deletes_stale_first() -> None:
    stale = SimpleNamespace(agent_name="agent-x", id="d1")
    ops = _Ops(listed=[stale])
    joiner = _Joiner(join_at=None)
    with pytest.raises(DispatchUnacknowledgedError):
        asyncio.run(_ladder(ops, joiner))
    # The room was listed before the first re-dispatch, and the stale same-name dispatch was
    # deleted before a new one was created (delete-first, C3 §4.3 step 1).
    assert ops.list_calls >= 1
    assert stale in ops.deletes
    assert ops.creates == 2


def test_failed_list_never_redispatches() -> None:
    ops = _Ops(list_error=True)
    joiner = _Joiner(join_at=None)
    with pytest.raises(DispatchUnacknowledgedError):
        asyncio.run(_ladder(ops, joiner))
    # A failed ListAgentDispatch is "possibly-outstanding": MUST NOT re-dispatch (C3 §4.3).
    assert ops.creates == 0
    assert ops.list_calls == 2  # attempted at both re-dispatch marks, never a create


def test_undeletable_stale_dispatch_blocks_recreate() -> None:
    stale = SimpleNamespace(agent_name="agent-x", id="d1")
    ops = _Ops(listed=[stale], delete_error=True)
    joiner = _Joiner(join_at=None)
    with pytest.raises(DispatchUnacknowledgedError):
        asyncio.run(_ladder(ops, joiner))
    # Could not delete the listed dispatch -> it is still outstanding -> no second create
    # (at-most-one-outstanding wins).
    assert ops.creates == 0


def test_inflight_create_is_not_double_dispatched() -> None:
    # The first re-create hangs past its window; at the next mark the ladder must reconcile the
    # in-flight create (possibly-outstanding) and issue NO blind second create (C3 §4.3 forfeit
    # rule / in-flight-create race).
    ops = _Ops(create_hang=True)
    joiner = _Joiner(join_at=None)
    with pytest.raises(DispatchUnacknowledgedError):
        asyncio.run(_ladder(ops, joiner))
    assert ops.creates == 1  # only the single hung create was ever started


def test_create_failure_spends_the_attempt_and_folds_into_exhaustion() -> None:
    # A create_dispatch failure on a ladder attempt counts against the budget and converts to the
    # typed exhaustion, not livekit_dispatch_failed (C3 §4.3 step 4).
    ops = _Ops(create_error=True)
    joiner = _Joiner(join_at=None)
    with pytest.raises(DispatchUnacknowledgedError) as ei:
        asyncio.run(_ladder(ops, joiner))
    assert ops.creates == 2  # both re-creates attempted
    assert ei.value.marker == "voice_dispatch_unacknowledged"


# --- production wrapper: readiness task + ops closures + cancellation --------------------------
def _fake_api(ops: _Ops) -> SimpleNamespace:
    dispatch = SimpleNamespace(
        list_dispatch=lambda req: ops.list_dispatches(),
        delete_dispatch=lambda req: ops.delete_dispatch(req),
        create_dispatch=lambda req: ops.create_dispatch(),
    )
    return SimpleNamespace(agent_dispatch=dispatch)


def _run_wrapper(ops: _Ops, join_delay: float | None, monkeypatch):
    async def fake_wait(room, *, excluded_identities, target_identity, timeout):
        if join_delay is None:
            await asyncio.sleep(timeout)  # never joins; cancelled at exhaustion
            raise asyncio.TimeoutError
        await asyncio.sleep(join_delay)
        return _TARGET

    monkeypatch.setattr(lk, "_wait_for_target_audio", fake_wait)

    async def go():
        return await lk._await_target_audio_with_dispatch_ack(
            SimpleNamespace(),  # room (unused by the fake)
            excluded_identities=set(),
            target_identity=None,
            readiness_timeout=120.0,
            api_client=_fake_api(ops),
            room_name="room-1",
            agent_name="agent-x",
            metadata="{}",
            first_dispatch_at=time.monotonic(),
            marks=_MARKS,
        )

    return asyncio.run(go())


def test_wrapper_returns_target_on_join(monkeypatch) -> None:
    ops = _Ops()
    result = _run_wrapper(ops, join_delay=0.05, monkeypatch=monkeypatch)
    assert result is _TARGET
    assert ops.creates == 0


def test_wrapper_raises_typed_exhaustion(monkeypatch) -> None:
    ops = _Ops()
    with pytest.raises(DispatchUnacknowledgedError):
        _run_wrapper(ops, join_delay=None, monkeypatch=monkeypatch)
    assert ops.creates == 2
