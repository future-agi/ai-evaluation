"""A chat server over the harness.

One conversation, held open, talked to over HTTP. The stages already emit typed events; this
streams them to whoever is listening and serves the artifacts the stages write. The page in
static/ is one renderer over that stream, and deliberately not the only possible one: anything
that can read server-sent events can draw this.

Run from the repo root, with the same environment the CLI uses. The last two variables are what
the run stage needs to place live calls; without them it still opens and says what is missing.

    set -a; . ./.env.acceptance; set +a
    export CLOUD_ML_REGION=global ALK_HARNESS_MODEL=claude-haiku-4-5
    export ACCEPTANCE_LIVEKIT_URL=ws://localhost:7880 ACCEPTANCE_MAX_SECONDS=210
    .venv/bin/python harness-ui/server.py

Then open http://localhost:8777
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from fi.alk.harness import sessions  # noqa: E402
from fi.alk.harness.chat import Conversation  # noqa: E402
from fi.alk.harness.config import chosen_model, credentials_hint  # noqa: E402
from fi.alk.harness.run import run_suite  # noqa: E402
from fi.alk.harness.scenarios import load as load_scenarios  # noqa: E402
from fi.alk.harness.understand import load as load_contract  # noqa: E402

app = FastAPI(title="harness")

# Where agents live. Almost never inside this repo: the harness is in one place and the agent
# being tested is somewhere else on disk nearly every time.
WORKSPACE = REPO.parent
SESSIONS = REPO / "artifacts" / "sessions"
# Which session was last opened, so a restart or a refresh comes back to it instead of to a
# blank page. One line on disk, because anything held only in the process is lost by restarting
# it — which is exactly when you most want it back.
OPEN = REPO / "artifacts" / ".open-session"

sessions.SESSIONS = SESSIONS

# The conversation currently open, and which session folder it belongs to.
conversation: Conversation | None = None
current: sessions.Session | None = None

# The task doing work, so it can be stopped. A stage that has started thrashing costs money
# every turn, and watching it without being able to intervene is the worst seat in the house.
running: asyncio.Task | None = None
busy = asyncio.Lock()


def _remember_open() -> None:
    OPEN.parent.mkdir(parents=True, exist_ok=True)
    OPEN.write_text(current.id if current else "", encoding="utf-8")


def _adopt(session: sessions.Session) -> None:
    """Make one session the open one, rebuilding its conversation from its folder."""
    global conversation, current
    from fi.alk.harness.sources import resolve

    source = None
    if session.source and session.kind:
        try:
            source = resolve(session.kind, name=session.agent or session.id, root=session.source)
        except Exception:
            source = None
    current = session
    conversation = Conversation(source=source, out=session.path, workspace=WORKSPACE)
    _remember_open()


def _runs(path) -> list:
    return sessions._runs(path) if path else []


class Said(BaseModel):
    text: str = ""


def _payload(event) -> str:
    body = {"kind": event.kind, "text": event.text, "tool": event.tool, "detail": event.detail}
    return f"data: {json.dumps(body, default=str)}\n\n"


def _status() -> dict:
    """Everything the page needs to draw itself, read from the open session's folder."""
    if conversation is None or current is None:
        return {
            "session": None,
            "stage": "",
            "stages": {},
            "agent": None,
            "model": chosen_model(),
            "credentials": credentials_hint().splitlines()[0],
            "spent_usd": 0.0,
            "have": {},
            "out": None,
            "busy": busy.locked(),
        }
    return {
        "session": current.meta(),
        "stage": conversation.stage_name,
        # Which stages can be opened right now, and why not where they cannot. Stages are not a
        # wizard: going back to fix a contract after the world is built is the ordinary case.
        "stages": conversation.reachable(),
        "agent": current.agent or current.id,
        "model": chosen_model(),
        "credentials": credentials_hint().splitlines()[0],
        "spent_usd": round(
            conversation.spent_usd
            + (conversation.stage.spent_usd if conversation.stage else 0.0),
            4,
        ),
        "have": current.has(),
        "out": str(current.path),
        # A refresh must be able to tell that work is still going on. Without this the page comes
        # back looking idle, and the next thing typed is rejected for no visible reason.
        "busy": busy.locked(),
    }


async def _stream_turn(coro_factory):
    """Run one piece of work and stream its events, ending with the fresh status."""
    global running
    queue: asyncio.Queue = asyncio.Queue()

    async def work():
        try:
            await coro_factory(lambda event: queue.put_nowait(_payload(event)))
        except asyncio.CancelledError:
            queue.put_nowait(
                _payload(
                    type("E", (), {"kind": "done", "text": "", "tool": "", "detail": {
                        "outcome": "stopped",
                        "error": "stopped. The stage is closed; say something to start it again."}})()
                )
            )
            raise
        except Exception as failed:
            queue.put_nowait(
                _payload(
                    type("E", (), {"kind": "done", "text": "", "tool": "", "detail": {
                        "outcome": "failed", "error": f"{type(failed).__name__}: {failed}"}})()
                )
            )
        finally:
            queue.put_nowait(None)

    task = asyncio.create_task(work())
    running = task
    while True:
        item = await queue.get()
        if item is None:
            break
        yield item
    running = None
    try:
        await task
    except asyncio.CancelledError:
        # Whatever the stage was in the middle of is not resumable, so it is closed and the
        # next message opens it again. Everything already saved to disk is untouched.
        await conversation.close()
    yield f"data: {json.dumps({'kind': 'status', 'detail': _status()}, default=str)}\n\n"


@app.on_event("startup")
async def _startup() -> None:
    """Come back to whichever session was last open."""
    SESSIONS.mkdir(parents=True, exist_ok=True)
    wanted = OPEN.read_text(encoding="utf-8").strip() if OPEN.exists() else ""
    session = sessions.load(wanted, SESSIONS) if wanted else None
    if session is None:
        found = sessions.every(SESSIONS)
        session = found[0] if found else None
    if session is not None:
        _adopt(session)


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/sessions")
async def list_sessions():
    """Every conversation, newest first, with what each one has produced."""
    return {
        "sessions": [
            {**one.meta(), "has": one.has(), "path": str(one.path)}
            for one in sessions.every(SESSIONS)
        ],
        "open": current.id if current else None,
    }


class Started(BaseModel):
    agent: str = ""


@app.post("/api/sessions")
async def start_session(started: Started):
    """Begin a new conversation, with its own folder."""
    if busy.locked():
        return JSONResponse({"error": "still working on the last thing"}, status_code=409)
    if conversation is not None:
        await conversation.close()
    _adopt(sessions.create(agent=started.agent, base=SESSIONS))
    return _status()


class Opened(BaseModel):
    id: str


@app.post("/api/sessions/open")
async def open_session(opened: Opened):
    """Reopen a conversation. Everything about it is read back from its folder."""
    if busy.locked():
        return JSONResponse({"error": "still working on the last thing"}, status_code=409)
    session = sessions.load(opened.id, SESSIONS)
    if session is None:
        return JSONResponse({"error": f"no session {opened.id}"}, status_code=404)
    if conversation is not None:
        await conversation.close()
    _adopt(session)
    return _status()


@app.delete("/api/sessions/{identifier}")
async def delete_session(identifier: str):
    """Delete a conversation and everything in it."""
    global conversation, current
    if busy.locked():
        return JSONResponse({"error": "still working on the last thing"}, status_code=409)
    if not sessions.remove(identifier, SESSIONS):
        return JSONResponse({"error": f"no session {identifier}"}, status_code=404)
    if current and current.id == identifier:
        if conversation is not None:
            await conversation.close()
        conversation, current = None, None
        found = sessions.every(SESSIONS)
        if found:
            _adopt(found[0])
        else:
            _remember_open()
    return _status()


@app.get("/api/history")
async def chat_history():
    """This conversation, as it was, so a refresh does not lose it."""
    if current is None:
        return {"messages": []}
    return {"messages": sessions.history(current.path)}


class Chosen(BaseModel):
    stage: str


@app.post("/api/stage")
async def choose_stage(chosen: Chosen):
    """Open one stage directly, whether or not it is the next one.

    Opening is not starting: the stage is made current and its tools become available, but it is
    not told to begin, because choosing to look at a stage is not asking it to spend anything.
    """
    if busy.locked():
        return JSONResponse({"error": "still working on the last thing"}, status_code=409)
    if conversation is None or current is None:
        return JSONResponse({"error": "no session open"}, status_code=404)
    try:
        await conversation.go_to(chosen.stage)
    except Exception as failed:
        return JSONResponse({"error": str(failed)}, status_code=400)
    current.stage = conversation.stage_name
    sessions.save(current)
    return _status()


@app.post("/api/stop")
async def stop():
    """Interrupt whatever is running. Anything already written to disk stays written."""
    if running is None or running.done():
        return {"stopped": False, "why": "nothing is running"}
    running.cancel()
    return {"stopped": True}


@app.get("/api/status")
async def status():
    return _status()


@app.post("/api/say")
async def say(said: Said):
    if conversation is None or current is None:
        return JSONResponse({"error": "no session open"}, status_code=404)
    if busy.locked():
        return JSONResponse(
            {"error": f"still working on the {conversation.stage_name} stage — one moment"},
            status_code=409,
        )

    text = said.text.strip()
    if text:
        sessions.remember(
            current.path,
            sessions.Message(role="you", text=text, stage=conversation.stage_name),
        )

    async def run(on_event):
        # What the harness says back, kept as it is produced, so reopening this conversation
        # shows the work and not only the conclusion.
        spoken: list[str] = []
        tools: list[dict] = []

        def watch(event):
            if event.kind == "text":
                spoken.append(event.text)
            elif event.kind == "tool":
                tools.append(
                    {
                        "label": (event.detail or {}).get("label") or event.tool,
                        "target": (event.detail or {}).get("target", ""),
                    }
                )
            elif event.kind == "result" and tools:
                tools[-1]["said"] = (event.text or "").splitlines()[:1]
                tools[-1]["failed"] = bool((event.detail or {}).get("is_error"))
            on_event(event)

        async with busy:
            if not text:
                entered = await conversation.advance(on_event=watch)
                if entered is None and conversation.stage is None:
                    await conversation.start(on_event=watch)
            else:
                await conversation.say(text, on_event=watch)

        sessions.remember(
            current.path,
            sessions.Message(
                role="harness",
                # Blank-line joined: one turn can speak several times, once per stage it passes
                # through, and running those together reads as one garbled paragraph when the
                # conversation is restored.
                text="\n\n".join(one.strip() for one in spoken if one.strip()),
                stage=conversation.stage_name,
                tools=tools,
            ),
        )
        # The folder knows what it is about, so the list can show it without opening it.
        current.stage = conversation.stage_name
        if not current.agent and conversation.contract:
            current.agent = conversation.contract.agent
            current.title = conversation.contract.one_liner or current.agent
        if conversation.source and not current.source:
            current.source = str(getattr(conversation.source, "root", "") or "")
            current.kind = conversation.source.kind
        sessions.save(current)

    return StreamingResponse(_stream_turn(run), media_type="text/event-stream")


@app.post("/api/run")
async def run_scenarios(said: Said):
    """Run the written scenarios against the world, streaming the conversations as they happen."""
    if busy.locked():
        return JSONResponse({"error": "still working on the last thing"}, status_code=409)
    out = current.path if current else None
    contract = load_contract(out) if out else None
    scenarios = load_scenarios(out) if out else []
    if not contract or not scenarios:
        return JSONResponse({"error": "nothing to run yet: need a contract and scenarios"}, 400)

    only = [name for name in said.text.split() if name]
    chosen = [s for s in scenarios if s.name in only] if only else scenarios

    async def run(on_event):
        def exchange(spoken):
            on_event(type("E", (), {
                "kind": "exchange", "text": spoken.text, "tool": "",
                "detail": {"speaker": spoken.speaker}})())

        def result(one):
            on_event(type("E", (), {
                "kind": "result_card", "text": one.line(), "tool": "",
                "detail": {
                    "scenario": one.scenario, "passed": one.passed,
                    "met": one.met, "of": len(one.checkpoints),
                    "checkpoints": [asdict(check) for check in one.checkpoints],
                    "ended": one.ended, "turns": one.turns, "calls": one.calls,
                    "transcript": one.transcript, "actions": one.actions,
                }})())

        async with busy:
            await run_suite(chosen, contract, out, on_result=result, on_exchange=exchange)

    return StreamingResponse(_stream_turn(run), media_type="text/event-stream")


@app.get("/api/contract")
async def contract():
    out = current.path if current else None
    path = out / "contract.json" if out else None
    if not path or not path.exists():
        return JSONResponse({})
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/world")
async def world():
    out = current.path if current else None
    path = out / "world.sqlite" if out else None
    if not path or not path.exists():
        return JSONResponse({"tables": []})
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        names = [row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        tables = []
        for name in names:
            count = db.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            cursor = db.execute(f"SELECT * FROM {name} LIMIT 200")
            columns = [col[0] for col in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            tables.append({"name": name, "count": count, "columns": columns, "rows": rows})
        manifest = {}
        manifest_path = out / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        handlers = [
            {"name": source.stem, "source": source.read_text(encoding="utf-8")}
            for source in sorted((out / "handlers").glob("*.py"))
        ] if (out / "handlers").exists() else []
        return {
            "tables": tables,
            "tools": manifest.get("tools", []),
            "tool_specs": manifest.get("tool_specs", []),
            "handlers": handlers,
            "sequences": manifest.get("sequences", []),
            "notes": manifest.get("notes", ""),
        }
    finally:
        db.close()


@app.get("/api/scenarios")
async def scenarios():
    """Every scenario, with its files and its three gates re-run.

    The gates are re-run rather than remembered. They are milliseconds of pure code, and a
    scenario shown as validated when the world has since changed underneath it is worse than one
    shown as unknown.
    """
    from fi.alk.harness.environment import load_catalogue
    from fi.alk.harness.folder import folder_for
    from fi.alk.harness.prove import prove

    out = current.path if current else None
    if not out:
        return []
    catalogue = load_catalogue(out)
    built = (out / "world.sqlite").exists()
    found = []
    for one in load_scenarios(out):
        body = one.model_dump()
        here = folder_for(out, one.name)
        body["folder"] = str(here)
        body["files"] = (
            sorted(str(f.relative_to(here)) for f in here.rglob("*") if f.is_file())
            if here.exists()
            else []
        )
        body["checks"] = [
            {
                "name": name,
                "settled_by": "code"
                if (g := catalogue.named(name)) and g.deterministic()
                else "a judge",
                "what": g.what if (g := catalogue.named(name)) else "",
                "source": g.check if (g := catalogue.named(name)) else "",
            }
            for name in one.sub_goals
        ]
        if built:
            proof = prove(one, catalogue, out)
            body["gates"] = proof.gates()
            body["validated"] = proof.holds
            body["why"] = "" if proof.holds else proof.why()
        else:
            body["gates"] = {}
            body["validated"] = None
            body["why"] = "no world to check against yet"
        found.append(body)
    return found


@app.get("/api/scenario-file")
async def scenario_file(name: str, path: str):
    """One file out of a scenario's folder, so the page can show what will actually run.

    Resolved and then checked to be inside that scenario's own folder: the path comes from a
    query string, and a page is not a trustworthy source of one.
    """
    from fi.alk.harness.folder import folder_for

    out = current.path if current else None
    if not out:
        return JSONResponse({"error": "no agent open"}, status_code=404)
    here = folder_for(out, name).resolve()
    asked = (here / path).resolve()
    if not asked.is_relative_to(here) or not asked.is_file():
        return JSONResponse({"error": "no such file"}, status_code=404)
    return {"path": path, "source": asked.read_text(encoding="utf-8")}


@app.get("/api/subgoals")
async def subgoals():
    """The shared catalogue. What every scenario is checked against."""
    from fi.alk.harness.environment import load_catalogue, load_simulator_prompt

    out = current.path if current else None
    if not out:
        return {"sub_goals": [], "simulator_prompt": ""}
    catalogue = load_catalogue(out)
    return {
        "sub_goals": [
            {
                "name": one.name,
                "what": one.what,
                "settled_by": "code" if one.deterministic() else "a judge",
                "check": one.check,
                "judged": one.judged,
            }
            for one in catalogue.sub_goals
        ],
        "simulator_prompt": load_simulator_prompt(out),
    }


@app.get("/api/runs")
async def runs():
    return _runs(current.path if current else None)


if __name__ == "__main__":
    import uvicorn

    print(f"model:       {chosen_model()}")
    print(credentials_hint())
    print("\nopen http://localhost:8777\n")
    uvicorn.run(app, host="127.0.0.1", port=8777, log_level="warning")
