from __future__ import annotations

import ast
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from fi.simulate.endpoints.originators import (
    CallOriginator,
    OriginatorFinalizeResult,
    build_call_originator,
    finalize_originator,
)
from fi.simulate.endpoints.retell import RetellCallOriginator
from fi.simulate.endpoints.vapi import VapiCallOriginator

_ENGINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "fi"
    / "simulate"
    / "simulation"
    / "engines"
    / "livekit.py"
)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


class _Transport:
    def __init__(
        self,
        *,
        inbound_call_originator: str | None,
        originator_agent_id: str | None = None,
        originator_from_number: str | None = None,
    ) -> None:
        self.inbound_call_originator = inbound_call_originator
        self.originator_agent_id = originator_agent_id
        self.originator_from_number = originator_from_number


# --- build_call_originator ------------------------------------------------


def test_factory_vapi_transport_calls_from_env_with_no_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VAPI_API_KEY", "env-key")
    monkeypatch.setenv("VAPI_ASSISTANT_ID", "env-assistant")
    monkeypatch.setenv("VAPI_PHONE_NUMBER_ID", "env-phone")
    monkeypatch.setenv("LIVEKIT_INBOUND_DID", "+15550000099")

    captured: dict[str, tuple] = {}
    original = VapiCallOriginator.from_env.__func__

    def _spy(cls: type) -> VapiCallOriginator:
        captured["called"] = True
        return original(cls)

    monkeypatch.setattr(VapiCallOriginator, "from_env", classmethod(_spy))

    transport = _Transport(inbound_call_originator="vapi")
    originator = build_call_originator(transport)

    assert captured.get("called") is True
    assert isinstance(originator, VapiCallOriginator)


def test_factory_retell_transport_calls_from_env_with_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RETELL_API_KEY", "env-key")
    monkeypatch.setenv("LIVEKIT_INBOUND_DID", "+15550000099")
    monkeypatch.delenv("RETELL_AGENT_ID", raising=False)
    monkeypatch.delenv("RETELL_FROM_NUMBER", raising=False)

    transport = _Transport(
        inbound_call_originator="retell",
        originator_agent_id="transport-agent",
        originator_from_number="+15550000001",
    )
    originator = build_call_originator(transport)

    assert isinstance(originator, RetellCallOriginator)
    assert originator._agent_id == "transport-agent"
    assert originator._from_number == "+15550000001"
    assert originator._destination == "+15550000099"


def test_factory_unsupported_originator_raises_typed_error() -> None:
    transport = _Transport(inbound_call_originator="bland")
    with pytest.raises(ValueError, match=r"unsupported_inbound_call_originator: bland"):
        build_call_originator(transport)


# --- structural dial-ordering assertion (C3) -------------------------------


def _find_function(tree: ast.AST, name: str) -> ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _contains_call_no_nested_funcs(node: ast.AST, func_name: str) -> bool:
    """True if node's subtree calls func_name, without descending into a
    nested FunctionDef/AsyncFunctionDef (a call inside a closure has a line
    number but no runtime ordering relative to the enclosing statement list).
    """

    found = False

    class _Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, inner: ast.FunctionDef) -> None:  # noqa: N802
            return

        def visit_AsyncFunctionDef(self, inner: ast.AsyncFunctionDef) -> None:  # noqa: N802
            return

        def visit_Call(self, call: ast.Call) -> None:  # noqa: N802
            nonlocal found
            target = call.func
            if isinstance(target, ast.Name) and target.id == func_name:
                found = True
            elif isinstance(target, ast.Attribute) and target.attr == func_name:
                found = True
            self.generic_visit(call)

    _Visitor().visit(node)
    return found


def _is_outcome_none_guard(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.If)
        and ast.unparse(stmt.test) == "outcome is not None"
        and len(stmt.body) == 1
        and isinstance(stmt.body[0], ast.Return)
    )


def test_dial_ordering_verify_before_guard_before_dial() -> None:
    tree = ast.parse(_ENGINE_PATH.read_text())
    fn = _find_function(tree, "_run_single_test_case")

    trys = [node for node in fn.body if isinstance(node, ast.Try)]
    assert len(trys) == 1, "expected exactly one top-level Try in _run_single_test_case"
    try_node = trys[0]

    verify_hits = [
        i
        for i, stmt in enumerate(try_node.body)
        if _contains_call_no_nested_funcs(stmt, "_ensure_sip_inbound_dispatch")
    ]
    guard_hits = [
        i for i, stmt in enumerate(try_node.body) if _is_outcome_none_guard(stmt)
    ]
    dial_hits = [
        i
        for i, stmt in enumerate(try_node.body)
        if _contains_call_no_nested_funcs(stmt, "build_call_originator")
    ]

    assert len(verify_hits) == 1, verify_hits
    assert len(guard_hits) == 1, guard_hits
    assert len(dial_hits) == 1, dial_hits

    i_verify, i_guard, i_dial = verify_hits[0], guard_hits[0], dial_hits[0]
    assert i_verify < i_guard < i_dial, (i_verify, i_guard, i_dial)


def test_engine_has_no_bare_vapi_failure_code_literals() -> None:
    source = _ENGINE_PATH.read_text()
    assert '"vapi_call_start_timeout"' not in source
    assert '"vapi_call_start_failed"' not in source


def test_engine_retell_call_id_expression_checks_originator() -> None:
    source = _ENGINE_PATH.read_text()
    assert 'inbound_call_originator == "retell"' in source


def _find_sync_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    """Like _find_function, but for a plain (non-async) def — engines/livekit.py
    itself can't be imported here without the livekit optional dependency
    (it raises ImportError at module scope), so this is a source-level
    assertion, not an import + call."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def test_engine_safe_provider_error_details_falls_back_to_status_code() -> None:
    """retell-sdk's APIStatusError exposes status_code, not status, so
    without this fallback a Retell start-failure outcome's details never
    carry an http_status. AST-checked (not imported + called) because
    engines/livekit.py requires the livekit optional dependency, which this
    environment does not have."""
    tree = ast.parse(_ENGINE_PATH.read_text())
    fn = _find_sync_function(tree, "_safe_provider_error_details")

    status_assigns = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "status"
    ]
    assert len(status_assigns) == 1, status_assigns
    value_source = ast.unparse(status_assigns[0].value)
    # ast.unparse normalises string literals to single quotes, so compare
    # against the unparse of an equivalent double-quoted reference
    # expression rather than hand-writing the expected quote style.
    expected_source = ast.unparse(
        ast.parse(
            'getattr(exc, "status", None) or getattr(exc, "status_code", None)',
            mode="eval",
        ).body
    )
    assert value_source == expected_source, value_source


def test_engine_finally_writes_reconciled_call_ids_onto_outcome_metadata() -> None:
    """Reconcile results must reach the outcome from inside the `finally` —
    start-failure paths `return outcome` from inside the `try`, so a write
    that only happens in the later `outcome.metadata.update(...)` block would
    never fire on exactly the paths where a reconcile can happen."""
    tree = ast.parse(_ENGINE_PATH.read_text())
    fn = _find_function(tree, "_run_single_test_case")
    trys = [node for node in fn.body if isinstance(node, ast.Try)]
    assert len(trys) == 1, "expected exactly one top-level Try in _run_single_test_case"
    try_node = trys[0]

    def _is_outcome_metadata_reconciled_write(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Store)
            and ast.unparse(node.value) == "outcome.metadata"
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "reconciled_call_ids"
        )

    hits = [
        node
        for stmt in try_node.finalbody
        for node in ast.walk(stmt)
        if _is_outcome_metadata_reconciled_write(node)
    ]
    assert len(hits) == 1, hits

    # Existence alone isn't enough: a write with the key present but the
    # value gutted to `[]`, or a guard flipped to `outcome is None`, would
    # still satisfy the check above while silently reintroducing the bug
    # this test exists to catch (ids dropped, or the write never firing on
    # a live outcome). Tie the store to its enclosing guard and confirm the
    # assigned value is a variable, not a constant/empty-list literal.
    guarded_writes = [
        (if_node, assign)
        for stmt in try_node.finalbody
        for if_node in ast.walk(stmt)
        if isinstance(if_node, ast.If)
        for assign in if_node.body
        if isinstance(assign, ast.Assign)
        and len(assign.targets) == 1
        and _is_outcome_metadata_reconciled_write(assign.targets[0])
    ]
    assert len(guarded_writes) == 1, guarded_writes
    guard_if, write_assign = guarded_writes[0]
    assert ast.unparse(guard_if.test) == "outcome is not None", ast.unparse(
        guard_if.test
    )
    assert isinstance(write_assign.value, ast.Name), (
        "outcome.metadata['reconciled_call_ids'] must be assigned a variable "
        "(the finalize result's ids), not a constant/empty-list literal that "
        "would mask a lost value"
    )


def test_engine_finally_falls_back_provider_call_id_to_reconciled_orphan() -> None:
    """A reconciled orphan's first id must feed the evidence hint — a
    fallback that only lived in the outer scope (or fired unconditionally,
    clobbering a real provider_call_id) would silently reintroduce
    "we stopped a billed call and then declined to fetch its record"."""
    tree = ast.parse(_ENGINE_PATH.read_text())
    fn = _find_function(tree, "_run_single_test_case")
    trys = [node for node in fn.body if isinstance(node, ast.Try)]
    assert len(trys) == 1, "expected exactly one top-level Try in _run_single_test_case"
    try_node = trys[0]

    fallback_writes = [
        (if_node, assign)
        for stmt in try_node.finalbody
        for if_node in ast.walk(stmt)
        if isinstance(if_node, ast.If)
        for assign in if_node.body
        if isinstance(assign, ast.Assign)
        and len(assign.targets) == 1
        and isinstance(assign.targets[0], ast.Name)
        and assign.targets[0].id == "provider_call_id"
    ]
    assert len(fallback_writes) == 1, fallback_writes
    guard_if, assign = fallback_writes[0]
    guard_source = ast.unparse(guard_if.test)
    assert "provider_call_id is None" in guard_source, guard_source
    assert "reconciled_call_ids" in guard_source, guard_source


def test_engine_finally_writes_cleanup_status_as_last_statement() -> None:
    """cleanup_status/cleanup_errors must be computed after every other
    cleanup step in the finally has run its `_record_cleanup_error` calls —
    anything earlier would read cleanup_errors before it is fully populated,
    so the write belongs at the end, and nowhere else in the finally."""
    tree = ast.parse(_ENGINE_PATH.read_text())
    fn = _find_function(tree, "_run_single_test_case")
    trys = [node for node in fn.body if isinstance(node, ast.Try)]
    assert len(trys) == 1, "expected exactly one top-level Try in _run_single_test_case"
    try_node = trys[0]

    assert try_node.finalbody, "finally body is empty"
    last_stmt = try_node.finalbody[-1]
    assert isinstance(last_stmt, ast.If), last_stmt
    assert "outcome" in ast.unparse(last_stmt.test), ast.unparse(last_stmt.test)

    def _is_cleanup_status_write(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Store)
            and ast.unparse(node.value) == "outcome.metadata"
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "cleanup_status"
        )

    last_stmt_writes = [
        node
        for stmt in last_stmt.body
        for node in ast.walk(stmt)
        if _is_cleanup_status_write(node)
    ]
    assert len(last_stmt_writes) == 1, last_stmt_writes

    earlier_writes = [
        node
        for stmt in try_node.finalbody[:-1]
        for node in ast.walk(stmt)
        if _is_cleanup_status_write(node)
    ]
    assert earlier_writes == [], earlier_writes


# --- finalize_originator (C3) ----------------------------------------------


@dataclass
class _Call:
    call_id: str = "call-1"


@dataclass
class FakeOriginator:
    start_result: Any = field(default_factory=_Call)
    stop_exc: Exception | None = None
    reconcile_result: list[str] = field(default_factory=list)
    reconcile_exc: Exception | None = None
    close_exc: Exception | None = None

    stop_calls: list[str] = field(default_factory=list)
    reconcile_calls: list[dict[str, Any]] = field(default_factory=list)
    close_calls: int = 0

    async def start(self) -> Any:
        return self.start_result

    async def stop(self, call_id: str) -> None:
        self.stop_calls.append(call_id)
        if self.stop_exc is not None:
            raise self.stop_exc

    async def reconcile_and_stop(
        self, *, started_after_ms: int, ended_before_ms: int
    ) -> list[str]:
        self.reconcile_calls.append(
            {"started_after_ms": started_after_ms, "ended_before_ms": ended_before_ms}
        )
        if self.reconcile_exc is not None:
            raise self.reconcile_exc
        return self.reconcile_result

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_exc is not None:
            raise self.close_exc


_CASE_STARTED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)
_NOW = lambda: _CASE_STARTED_AT + timedelta(seconds=5)  # noqa: E731


def test_finalize_with_call_id_stops_and_sets_termination_source() -> None:
    fake = FakeOriginator()
    result = _run(
        finalize_originator(
            fake,
            provider_call_id="call-1",
            originator_name="retell",
            case_started_at=_CASE_STARTED_AT,
            cleanup_timeout=5.0,
            now=_NOW,
        )
    )
    assert fake.stop_calls == ["call-1"]
    assert fake.reconcile_calls == []
    assert fake.close_calls == 1
    assert result.termination_source == "sdk_originator_cleanup"
    assert result.reconciled_call_ids == []
    assert result.cleanup_errors == []


def test_finalize_without_call_id_reconciles_with_int_ms_and_ordered_window() -> None:
    fake = FakeOriginator(reconcile_result=["orphan-1"])
    result = _run(
        finalize_originator(
            fake,
            provider_call_id=None,
            originator_name="retell",
            case_started_at=_CASE_STARTED_AT,
            cleanup_timeout=5.0,
            now=_NOW,
        )
    )
    assert fake.stop_calls == []
    assert len(fake.reconcile_calls) == 1
    call = fake.reconcile_calls[0]
    assert isinstance(call["started_after_ms"], int)
    assert isinstance(call["ended_before_ms"], int)
    assert call["started_after_ms"] < call["ended_before_ms"]
    assert call["started_after_ms"] == int(_CASE_STARTED_AT.timestamp() * 1000)
    assert call["ended_before_ms"] == int(_NOW().timestamp() * 1000)
    assert result.termination_source == "sdk_originator_cleanup"
    assert result.reconciled_call_ids == ["orphan-1"]
    assert result.cleanup_errors == []
    assert fake.close_calls == 1


def test_finalize_treats_a_naive_case_started_at_as_utc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A naive datetime must be read as UTC, not local time — assert the naive
    # case produces the identical started_after_ms as its aware UTC
    # equivalent, rather than pinning a machine-local-timezone offset. Under
    # TZ=UTC (e.g. most CI runners) a naive datetime and its aware twin
    # already share an epoch, so this assertion has no teeth there unless we
    # force a non-UTC process timezone for the duration of the test.
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is POSIX-only; cannot pin the process timezone")
    monkeypatch.setenv("TZ", "Asia/Kolkata")
    time.tzset()
    try:
        naive_started_at = datetime(2026, 1, 1)
        fake = FakeOriginator(reconcile_result=["orphan-1"])
        result = _run(
            finalize_originator(
                fake,
                provider_call_id=None,
                originator_name="retell",
                case_started_at=naive_started_at,
                cleanup_timeout=5.0,
                now=_NOW,
            )
        )
        call = fake.reconcile_calls[0]
        assert call["started_after_ms"] == int(_CASE_STARTED_AT.timestamp() * 1000)
        assert result.reconciled_call_ids == ["orphan-1"]
    finally:
        # Restore TZ before re-syncing the process timezone: calling tzset()
        # while the patched TZ is still in os.environ would just reapply
        # Asia/Kolkata and leave later tests running under it. monkeypatch's
        # own automatic teardown restores os.environ but never calls
        # tzset() itself, so undo the env change here, synchronously, before
        # resetting the C library's idea of local time.
        monkeypatch.undo()
        time.tzset()


def test_finalize_without_call_id_empty_reconcile_leaves_termination_source_none() -> (
    None
):
    fake = FakeOriginator(reconcile_result=[])
    result = _run(
        finalize_originator(
            fake,
            provider_call_id=None,
            originator_name="vapi",
            case_started_at=_CASE_STARTED_AT,
            cleanup_timeout=5.0,
            now=_NOW,
        )
    )
    assert result.termination_source is None
    assert result.reconciled_call_ids == []
    assert result.cleanup_errors == []


def test_finalize_stop_raising_still_closes_and_records_one_error() -> None:
    fake = FakeOriginator(stop_exc=RuntimeError("boom"))
    result = _run(
        finalize_originator(
            fake,
            provider_call_id="call-1",
            originator_name="retell",
            case_started_at=_CASE_STARTED_AT,
            cleanup_timeout=5.0,
            now=_NOW,
        )
    )
    assert fake.close_calls == 1
    assert result.termination_source is None
    assert len(result.cleanup_errors) == 1
    op, exc = result.cleanup_errors[0]
    assert op == "retell_call_stop"
    assert isinstance(exc, RuntimeError)


def test_finalize_reconcile_raising_records_reconcile_tag() -> None:
    fake = FakeOriginator(reconcile_exc=RuntimeError("list-calls down"))
    result = _run(
        finalize_originator(
            fake,
            provider_call_id=None,
            originator_name="retell",
            case_started_at=_CASE_STARTED_AT,
            cleanup_timeout=5.0,
            now=_NOW,
        )
    )
    assert result.termination_source is None
    assert result.reconciled_call_ids == []
    assert len(result.cleanup_errors) == 1
    op, exc = result.cleanup_errors[0]
    assert op == "retell_call_reconcile"
    assert isinstance(exc, RuntimeError)


def test_finalize_reconcile_type_error_gets_distinct_tag() -> None:
    fake = FakeOriginator(reconcile_exc=TypeError("bad epoch ms"))
    result = _run(
        finalize_originator(
            fake,
            provider_call_id=None,
            originator_name="retell",
            case_started_at=_CASE_STARTED_AT,
            cleanup_timeout=5.0,
            now=_NOW,
        )
    )
    assert len(result.cleanup_errors) == 1
    op, exc = result.cleanup_errors[0]
    assert op == "retell_call_reconcile_bad_arguments"
    assert isinstance(exc, TypeError)


def test_finalize_close_raising_records_close_error() -> None:
    fake = FakeOriginator(close_exc=RuntimeError("client already closed"))
    result = _run(
        finalize_originator(
            fake,
            provider_call_id="call-1",
            originator_name="retell",
            case_started_at=_CASE_STARTED_AT,
            cleanup_timeout=5.0,
            now=_NOW,
        )
    )
    assert result.termination_source == "sdk_originator_cleanup"
    assert len(result.cleanup_errors) == 1
    op, exc = result.cleanup_errors[0]
    assert op == "retell_call_close"
    assert isinstance(exc, RuntimeError)


def test_finalize_stop_and_close_both_raising_records_both_errors() -> None:
    fake = FakeOriginator(
        stop_exc=RuntimeError("stop failed"),
        close_exc=RuntimeError("close failed"),
    )
    result = _run(
        finalize_originator(
            fake,
            provider_call_id="call-1",
            originator_name="vapi",
            case_started_at=_CASE_STARTED_AT,
            cleanup_timeout=5.0,
            now=_NOW,
        )
    )
    ops = [op for op, _exc in result.cleanup_errors]
    assert ops == ["vapi_call_stop", "vapi_call_close"]
    assert result.termination_source is None


def test_finalize_never_raises_even_when_close_raises_and_no_call_id() -> None:
    fake = FakeOriginator(
        reconcile_exc=RuntimeError("boom"), close_exc=RuntimeError("boom2")
    )
    # Must not raise.
    result = _run(
        finalize_originator(
            fake,
            provider_call_id=None,
            originator_name="retell",
            case_started_at=_CASE_STARTED_AT,
            cleanup_timeout=5.0,
            now=_NOW,
        )
    )
    assert isinstance(result, OriginatorFinalizeResult)
    ops = [op for op, _exc in result.cleanup_errors]
    assert ops == ["retell_call_reconcile", "retell_call_close"]


def test_call_originator_protocol_is_satisfied_by_both_concrete_classes() -> None:
    # Structural check only (no runtime_checkable) — both classes expose the
    # full Protocol surface with matching method names.
    for cls in (VapiCallOriginator, RetellCallOriginator):
        for method in ("start", "stop", "reconcile_and_stop", "close"):
            assert hasattr(cls, method)
    # FakeOriginator (used above) also satisfies CallOriginator structurally.
    fake: CallOriginator = FakeOriginator()
    assert hasattr(fake, "start")
    assert hasattr(fake, "stop")
    assert hasattr(fake, "reconcile_and_stop")
    assert hasattr(fake, "close")


# --------------------------------------------------------------------------- #
# Leased-room reuse: structural pins (livekit-free, AST-only)
# --------------------------------------------------------------------------- #
def _find_if_with_test(stmts: list[ast.stmt], test_src: str) -> ast.If:
    for stmt in stmts:
        if isinstance(stmt, ast.If) and ast.unparse(stmt.test) == test_src:
            return stmt
    raise AssertionError(f"no If with test {test_src!r} in {stmts}")


def _top_level_try(fn: ast.AsyncFunctionDef) -> ast.Try:
    trys = [node for node in fn.body if isinstance(node, ast.Try)]
    assert len(trys) == 1, "expected exactly one top-level Try in _run_single_test_case"
    return trys[0]


def test_drain_precedes_room_create_under_verbatim_guard() -> None:
    tree = ast.parse(_ENGINE_PATH.read_text())
    fn = _find_function(tree, "_run_single_test_case")
    try_node = _top_level_try(fn)

    managed_if = _find_if_with_test(try_node.body, "managed_room_owned")
    inner_if = _find_if_with_test(managed_if.body, "not profile.places_outbound_call")

    drain_hits = [
        i
        for i, stmt in enumerate(inner_if.body)
        if _contains_call_no_nested_funcs(stmt, "_ensure_room_absent")
    ]
    create_hits = [
        i
        for i, stmt in enumerate(inner_if.body)
        if _contains_call_no_nested_funcs(stmt, "create_room")
    ]
    assert len(drain_hits) == 1, drain_hits
    assert len(create_hits) == 1, create_hits
    i_drain, i_create = drain_hits[0], create_hits[0]
    assert i_drain < i_create, (i_drain, i_create)

    drain_stmt = inner_if.body[i_drain]
    assert isinstance(drain_stmt, ast.If)
    assert ast.unparse(drain_stmt.test) == "runtime.room_name_verbatim"

    create_stmt = inner_if.body[i_create]
    assert isinstance(create_stmt, ast.If)
    assert ast.unparse(create_stmt.test) == "outcome is None"


def test_occupancy_check_immediately_precedes_dial() -> None:
    tree = ast.parse(_ENGINE_PATH.read_text())
    fn = _find_function(tree, "_run_single_test_case")
    try_node = _top_level_try(fn)

    occ_hits = [
        i
        for i, stmt in enumerate(try_node.body)
        if _contains_call_no_nested_funcs(stmt, "_unexpected_participants")
    ]
    dial_hits = [
        i
        for i, stmt in enumerate(try_node.body)
        if _contains_call_no_nested_funcs(stmt, "build_call_originator")
    ]
    assert len(occ_hits) == 1, occ_hits
    assert len(dial_hits) == 1, dial_hits
    i_occ, i_dial = occ_hits[0], dial_hits[0]
    assert i_occ == i_dial - 1, (i_occ, i_dial)

    occ_stmt = try_node.body[i_occ]
    assert isinstance(occ_stmt, ast.If)
    occ_test_src = ast.unparse(occ_stmt.test)
    assert "runtime.room_name_verbatim" in occ_test_src
    assert "profile.receives_inbound_call" in occ_test_src


def test_occupancy_assigns_outcome_before_return() -> None:
    tree = ast.parse(_ENGINE_PATH.read_text())
    fn = _find_function(tree, "_run_single_test_case")
    try_node = _top_level_try(fn)

    occ_hits = [
        stmt
        for stmt in try_node.body
        if _contains_call_no_nested_funcs(stmt, "_unexpected_participants")
    ]
    assert len(occ_hits) == 1, occ_hits
    occ_stmt = occ_hits[0]
    assert isinstance(occ_stmt, ast.If)

    # The occupancy guard body is: compute `unexpected`, then `if unexpected:`
    # whose own body assigns `outcome` and then returns it.
    inner_ifs = [
        node
        for node in ast.walk(occ_stmt)
        if isinstance(node, ast.If) and node is not occ_stmt
    ]
    assert len(inner_ifs) == 1, inner_ifs
    inner_if = inner_ifs[0]

    assign_indices = [
        i
        for i, stmt in enumerate(inner_if.body)
        if isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id == "outcome"
    ]
    return_indices = [
        i for i, stmt in enumerate(inner_if.body) if isinstance(stmt, ast.Return)
    ]
    assert len(assign_indices) == 1, assign_indices
    assert len(return_indices) == 1, return_indices
    assert assign_indices[0] < return_indices[0]


def test_caller_check_immediately_follows_readiness_wait() -> None:
    tree = ast.parse(_ENGINE_PATH.read_text())
    fn = _find_function(tree, "_run_single_test_case")
    try_node = _top_level_try(fn)

    wait_hits = [
        i
        for i, stmt in enumerate(try_node.body)
        if _contains_call_no_nested_funcs(stmt, "_wait_for_target_audio")
    ]
    caller_hits = [
        i
        for i, stmt in enumerate(try_node.body)
        if _contains_call_no_nested_funcs(stmt, "_caller_matches")
    ]
    assert len(wait_hits) == 1, wait_hits
    assert len(caller_hits) == 1, caller_hits
    i_wait, i_caller = wait_hits[0], caller_hits[0]
    assert i_caller == i_wait + 1, (i_wait, i_caller)

    caller_stmt = try_node.body[i_caller]
    assert isinstance(caller_stmt, ast.If)
    caller_test_src = ast.unparse(caller_stmt.test)
    for token in (
        "runtime.room_name_verbatim",
        "profile.receives_inbound_call",
        "transport.originator_from_number",
    ):
        assert token in caller_test_src, (token, caller_test_src)

    returns = [node for node in ast.walk(caller_stmt) if isinstance(node, ast.Return)]
    assert returns == [], "the caller check must not return directly"

    # No SECOND `if outcome is not None: return outcome` guard was added
    # anywhere in the function beyond the one pre-existing occurrence.
    guard_hits = [node for node in ast.walk(fn) if _is_outcome_none_guard(node)]
    assert len(guard_hits) == 1, guard_hits


def test_wrong_caller_fails_through_exception_handler() -> None:
    tree = ast.parse(_ENGINE_PATH.read_text())
    fn = _find_function(tree, "_run_single_test_case")
    try_node = _top_level_try(fn)

    caller_hits = [
        stmt
        for stmt in try_node.body
        if _contains_call_no_nested_funcs(stmt, "_caller_matches")
    ]
    assert len(caller_hits) == 1, caller_hits
    caller_stmt = caller_hits[0]

    raises = [
        node
        for node in ast.walk(caller_stmt)
        if isinstance(node, ast.Raise)
        and node.exc is not None
        and ast.unparse(node.exc).startswith("_LeasedRoomCallerMismatch")
    ]
    assert len(raises) == 1, raises

    handler_types = [
        ast.unparse(handler.type) if handler.type is not None else None
        for handler in try_node.handlers
    ]
    assert "_LeasedRoomCallerMismatch" in handler_types, handler_types
    idx_mismatch = handler_types.index("_LeasedRoomCallerMismatch")
    exception_indices = [i for i, t in enumerate(handler_types) if t == "Exception"]
    assert exception_indices, handler_types
    assert all(idx_mismatch < i for i in exception_indices), (
        idx_mismatch,
        exception_indices,
    )

    mismatch_handler = try_node.handlers[idx_mismatch]
    outcome_assigns = [
        node
        for node in ast.walk(mismatch_handler)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "outcome"
    ]
    assert len(outcome_assigns) == 1, outcome_assigns


def test_leased_room_failure_codes_and_flags() -> None:
    tree = ast.parse(_ENGINE_PATH.read_text())
    pairs: set[tuple[str, bool]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != "_failure_outcome":
            continue
        args = node.args
        if len(args) < 3 or not isinstance(args[2], ast.Constant):
            continue
        code = args[2].value
        if not isinstance(code, str) or not code.startswith("livekit_room_"):
            continue
        retryable = False
        for kw in node.keywords:
            if kw.arg == "retryable" and isinstance(kw.value, ast.Constant):
                retryable = kw.value.value
        pairs.add((code, retryable))

    assert pairs == {
        ("livekit_room_create_timeout", True),
        ("livekit_room_create_failed", False),
        ("livekit_room_drain_timeout", True),
        ("livekit_room_drain_failed", False),
        ("livekit_room_occupied", True),
        ("livekit_room_wrong_caller", True),
    }, pairs


def test_leased_room_log_lines_carry_no_identities() -> None:
    tree = ast.parse(_ENGINE_PATH.read_text())
    targets = {
        "leased room drained": {"run_id", "test_case_id", "room_name", "polls"},
        "leased room occupied before dial": {
            "run_id",
            "test_case_id",
            "room_name",
            "unexpected_participants",
        },
        "leased room wrong caller": {
            "run_id",
            "test_case_id",
            "expected_digits",
            "observed_digits",
        },
        "leased room caller unverified": {"run_id", "test_case_id"},
    }
    found: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name not in ("info", "warning"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        message = first.value
        if message not in targets:
            continue
        extra_kw = next((kw for kw in node.keywords if kw.arg == "extra"), None)
        assert extra_kw is not None, message
        assert isinstance(extra_kw.value, ast.Dict), (
            f"{message!r} extra must be an inline dict literal"
        )
        keys = {
            key.value for key in extra_kw.value.keys if isinstance(key, ast.Constant)
        }
        found[message] = keys

    for message, expected_keys in targets.items():
        assert found.get(message) == expected_keys, (message, found.get(message))
