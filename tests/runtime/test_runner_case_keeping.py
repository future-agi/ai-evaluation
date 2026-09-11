"""The wrapper that keeps finished cases must call the real callback, once, not itself."""

from __future__ import annotations

import asyncio
import inspect

from fi.simulate.runtime.runner import SimulationRunner


def test_the_case_wrapper_does_not_call_itself():
    """It closed over the name it then rebound, so it recursed until the interpreter gave up.

    Read off the source rather than by running a whole simulation: the recursion only appeared once
    cases started completing, which is exactly the condition a unit test would not reproduce.
    """
    source = inspect.getsource(SimulationRunner.run)
    assert "report_case = on_case_complete" in source
    assert "await report_case(index, legacy_case)" in source
    assert "await on_case_complete(index, legacy_case)" not in source


def test_a_wrapper_built_this_way_calls_through_once():
    """The same shape, in miniature, so the property is checked and not just the spelling."""
    seen: list[int] = []

    async def real(index: int, case: object) -> None:
        seen.append(index)

    on_case_complete = real
    report_case = on_case_complete

    async def wrapper(index: int, case: object) -> None:
        if report_case is not None:
            await report_case(index, case)

    on_case_complete = wrapper
    asyncio.run(on_case_complete(7, object()))
    assert seen == [7]
