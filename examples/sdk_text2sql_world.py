"""A Text2SQL environment — proof that "simulation" is not just chat/voice.

The world is a real in-memory SQLite database. The agent's **action space** is two
tools (`inspect_schema`, `run_sql`); its **observation** is the returned rows or a
SQL error; the world **state** is `solved` (did the query return the gold rows).
Scoring is the world contract (final state), not conversation quality.

This is a *world you plug in* (an ``EnvironmentAdapter``), driven by the ordinary
``chat`` loop — the same shape as ``RefundWorld`` in the v2 demo, but the tools do
real work. No credentials; deterministic.

Run:  python examples/sdk_text2sql_world.py artifacts/text2sql.json
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import fi.alk.simulate as S
from fi.simulate.agent.wrapper import AgentInput, AgentResponse
from fi.simulate.environment import EnvironmentAdapter, EnvironmentSnapshot, ToolExecutionResult

_SCHEMA = (
    "CREATE TABLE orders (id TEXT, customer TEXT, amount REAL, status TEXT);"
)
_ROWS = [
    ("A1", "Sam", 50.0, "damaged"),
    ("A2", "Alex", 20.0, "delivered"),
    ("A3", "Morgan", 75.0, "damaged"),
]
_QUESTION = "List the order ids whose status is 'damaged', ascending."
_GOLD = [("A1",), ("A3",)]

_TOOLS = [
    {"name": "inspect_schema", "description": "Return the SQL schema."},
    {"name": "run_sql", "description": "Execute a read-only SQL query, return rows."},
]


class Text2SQLWorld(EnvironmentAdapter):
    """SQLite-backed world: agent writes SQL, world executes + scores it."""

    name = "text2sql"

    def __init__(self) -> None:
        self._db = sqlite3.connect(":memory:")
        self._db.executescript(_SCHEMA)
        self._db.executemany("INSERT INTO orders VALUES (?,?,?,?)", _ROWS)
        self._db.commit()
        self.state: dict[str, Any] = {"solved": False, "attempts": 0, "last_error": None}

    def reset(self, **_context: Any) -> EnvironmentSnapshot:
        self.state = {"solved": False, "attempts": 0, "last_error": None}
        return EnvironmentSnapshot(
            tools=list(_TOOLS),
            state={"schema": _SCHEMA, "question": _QUESTION, **self.state},
        )

    def handle_tool_call(
        self, tool_call: Mapping[str, Any], **_context: Any
    ) -> Optional[ToolExecutionResult]:
        name = tool_call.get("name") or (tool_call.get("function") or {}).get("name")
        call_id = tool_call.get("id") or tool_call.get("tool_call_id")
        args = tool_call.get("arguments") or {}
        if name == "inspect_schema":
            return ToolExecutionResult(tool_call_id=call_id, tool_name=name, content=_SCHEMA)
        if name == "run_sql":
            self.state["attempts"] += 1
            query = str(args.get("query") or args.get("sql") or "")
            try:
                rows = self._db.execute(query).fetchall()
            except Exception as exc:  # invalid SQL becomes the next observation
                self.state["last_error"] = str(exc)
                return ToolExecutionResult(
                    tool_call_id=call_id, tool_name=name,
                    content=f"SQL error: {exc}", success=False, error=str(exc),
                    state_updates={"last_error": str(exc)})
            solved = rows == _GOLD
            self.state["solved"] = solved
            return ToolExecutionResult(
                tool_call_id=call_id, tool_name=name, content=str(rows),
                result={"rows": rows, "solved": solved},
                state_updates={"solved": solved, "last_error": None})
        return None


class Text2SQLAgent:
    """Scripted target: inspect the schema, then write the correct query."""

    async def call(self, agent_input: AgentInput) -> AgentResponse:
        turn = agent_input.turn_index
        if turn == 0:
            return AgentResponse(content="Let me check the schema.",
                                 tool_calls=[{"id": "s0", "name": "inspect_schema", "arguments": {}}])
        if turn == 1:
            return AgentResponse(
                content="Now the query.",
                tool_calls=[{"id": "s1", "name": "run_sql",
                             "arguments": {"query": "SELECT id FROM orders WHERE status='damaged' ORDER BY id"}}])
        return AgentResponse(content="Done — the damaged orders are A1 and A3.")


def run(output_path: str | os.PathLike[str]) -> dict[str, Any]:
    world = Text2SQLWorld()
    spec = S.SimulationSpec(
        run_id="text2sql_demo",
        environment=S.EnvironmentSpec(adapter=S.EnvironmentAdapters.CHAT,
                                      world_kind=S.WorldKinds.TOOL_API,
                                      config={"max_turns": 3, "min_turns": 1}),
        target=S.AgentEndpointSpec(adapter=S.TargetAdapters.CALLABLE),
        simulator=S.SimulatorPolicySpec(adapter=S.SimulatorAdapters.SYNTHETIC_USER),
        scenario=S.Scenario(name="text2sql", dataset=[
            S.Persona(persona={"name": "Analyst"}, situation=_QUESTION,
                      outcome="the correct order ids are returned")]),
    )
    report = asyncio.run(S.SimulationRunner().run(spec, target=Text2SQLAgent(), environment=world))
    data = {
        "kind": "agent-learning.text2sql-world.v1",
        "status": "passed" if (report.status == S.RunStatus.COMPLETED and world.state["solved"]) else "failed",
        "run_status": report.status.value,
        "solved": world.state["solved"],
        "attempts": world.state["attempts"],
        "transcript": report.test_cases[0].result.transcript if report.test_cases else "",
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


if __name__ == "__main__":
    target_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/text2sql.json"
    result = run(target_path)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "passed" else 1)
