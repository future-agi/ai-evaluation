"""Framework-neutral observation of declared in-process Python tools."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def test_declared_python_tool_is_traced_without_changing_result(tmp_path: Path) -> None:
    bootstrap = Path(__file__).resolve().parents[2] / "src/fi/alk/harness/livekit_tool_trace_bootstrap.py"
    shutil.copy2(bootstrap, tmp_path / "sitecustomize.py")
    (tmp_path / "agent.py").write_text(
        "import asyncio\n"
        "def get_weather(location):\n    return f'sunny in {location}'\n"
        "async def get_forecast(location):\n"
        "    await asyncio.sleep(0.2)\n"
        "    return f'rain in {location}'\n"
    )
    trace = tmp_path / "calls.jsonl"
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path),
        "HARNESS_TOOL_TRACE": str(trace),
        "ALK_TOOL_TRACE_BINDINGS": json.dumps(
            [
                {"name": "get_weather", "module": "agent", "callable": "get_weather"},
                {"name": "get_forecast", "module": "agent", "callable": "get_forecast"},
            ]
        ),
    }

    started = time.time()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import asyncio\n"
            "from agent import get_forecast, get_weather\n"
            "print(get_weather('Paris'))\n"
            "print(asyncio.run(get_forecast('Oslo')))\n",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    finished = time.time()

    assert result.stdout.split() == ["sunny", "in", "Paris", "rain", "in", "Oslo"]
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    # A suspended coroutine is also traced (its awaited value as output); the
    # completed call is the one that carries the tool's result.
    completed = {
        record["name"]: record
        for record in records
        if record["output"] in {"sunny in Paris", "rain in Oslo"}
    }
    weather, forecast = completed["get_weather"], completed["get_forecast"]
    assert weather == {
        "name": "get_weather",
        "arguments": {"location": "Paris"},
        "output": "sunny in Paris",
        "is_error": False,
        "at": weather["at"],
    }
    assert forecast["output"] == "rain in Oslo"
    assert started <= weather["at"] <= forecast["at"] <= finished
    # The tool began before its 200 ms await, not when it resumed after it.
    assert finished - forecast["at"] >= 0.2


def test_langchain_tool_bound_through_its_invoke_method_is_traced(tmp_path: Path) -> None:
    """A builder may bind a LangChain ``@tool`` as ``check_availability.invoke``.

    ``invoke`` is defined in langchain_core, not in the agent's module, so the hook must watch
    the decorated function itself. The stand-in package below keeps that shape: the wrapper and
    its ``invoke`` live in ``langchain_core/tools.py``, the function in ``agent.py``.
    """
    bootstrap = Path(__file__).resolve().parents[2] / "src/fi/alk/harness/livekit_tool_trace_bootstrap.py"
    shutil.copy2(bootstrap, tmp_path / "sitecustomize.py")
    (tmp_path / "langchain_core").mkdir()
    (tmp_path / "langchain_core" / "__init__.py").write_text("")
    (tmp_path / "langchain_core" / "tools.py").write_text(
        "import asyncio\n"
        "class StructuredTool:\n"
        "    def __init__(self, function):\n"
        "        self.name = function.__name__\n"
        "        coroutine = asyncio.iscoroutinefunction(function)\n"
        "        self.func = None if coroutine else function\n"
        "        self.coroutine = function if coroutine else None\n"
        "    def invoke(self, tool_input):\n"
        "        return self.func(**tool_input)\n"
        "    async def ainvoke(self, tool_input):\n"
        "        return await self.coroutine(**tool_input)\n"
        "def tool(function):\n"
        "    return StructuredTool(function)\n"
    )
    (tmp_path / "agent.py").write_text(
        "import asyncio\n"
        "from langchain_core.tools import tool\n"
        "class Booking:\n"
        "    @staticmethod\n"
        "    def invoke(room):\n"
        "        return f'held {room}'\n"
        "@tool\n"
        "def check_availability(date):\n"
        "    return f'free on {date}'\n"
        "@tool\n"
        "async def book_room(room):\n"
        "    await asyncio.sleep(0)\n"
        "    return f'booked {room}'\n"
    )
    trace = tmp_path / "calls.jsonl"
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path),
        "HARNESS_TOOL_TRACE": str(trace),
        "ALK_TOOL_TRACE_BINDINGS": json.dumps(
            [
                {"name": "check_availability", "module": "agent", "callable": "check_availability.invoke"},
                {"name": "book_room", "module": "agent", "callable": "book_room.ainvoke"},
                # A class's own method named like a wrapper method is still matched by that name.
                {"name": "hold_room", "module": "agent", "callable": "Booking.invoke"},
            ]
        ),
    }

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import asyncio\n"
            "from agent import Booking, book_room, check_availability\n"
            "print(check_availability.invoke({'date': 'Friday'}))\n"
            "print(asyncio.run(book_room.ainvoke({'room': '12'})))\n"
            "print(Booking.invoke('7'))\n",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.stdout.splitlines() == ["free on Friday", "booked 12", "held 7"]
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    completed = [
        (record["name"], record["arguments"], record["output"])
        for record in records
        if record["output"] is not None
    ]
    assert completed == [
        ("check_availability", {"date": "Friday"}, "free on Friday"),
        ("book_room", {"room": "12"}, "booked 12"),
        ("hold_room", {"room": "7"}, "held 7"),
    ]
    # The Booking class body runs at import under a watched name; it is not a call.
    assert [record["output"] for record in records if record["name"] == "hold_room"] == [
        "held 7"
    ]
