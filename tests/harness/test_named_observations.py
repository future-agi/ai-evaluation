import asyncio
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
import threading

import pytest
from pydantic import ValidationError

from fi.alk.harness.contract import Runtime
from fi.alk.harness.hosted_scheduler import _invoke
from fi.alk.harness.scenario_source import _compile_entry
from fi.alk.harness.world.errors import WorldUsageError
from fi.alk.harness.world.observation import ObservationWorld


CATALOGUE = {
    "history": {
        "capability": "app",
        "path": "/read",
        "body": {
            "operation": "history",
            "args": {"session": "{{session}}", "limit": "{{limit}}"},
        },
        "parameters": {
            "session": {"type": "string", "max_length": 16},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        },
    }
}


@pytest.fixture
def application():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, body, self.headers.get("Authorization")))
            self.reply(
                {"events": ["completed"], "echo": self.headers.get("Authorization")}
            )

        def do_GET(self):
            self.reply({"ok": True})

        def reply(self, body):
            encoded = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    world = ObservationWorld(
        0,
        random.Random(0),
        {"app": f"http://127.0.0.1:{server.server_port}"},
        observations=CATALOGUE,
    )
    world.set_headers("app", {"Authorization": "Bearer private-test-credential"})
    yield world, calls
    server.shutdown()
    thread.join()
    server.server_close()


def test_real_read_operation_and_unchanged_get(application):
    world, calls = application
    readonly = world.read_only()
    assert readonly.observe("history", {"session": "abc", "limit": 3})["json"][
        "events"
    ] == ["completed"]
    assert calls == [
        (
            "/read",
            {"operation": "history", "args": {"session": "abc", "limit": 3}},
            "Bearer private-test-credential",
        )
    ]
    assert readonly.request("app", "GET", "/health")["json"] == {"ok": True}
    assert "private-test-credential" not in json.dumps(world._drain_http_events())


@pytest.mark.parametrize(
    "parameters",
    [
        None,
        {},
        {"session": "a", "limit": 1, "operation": "delete"},
        {"session": {}, "limit": 1},
        {"session": "x" * 17, "limit": 1},
        {"session": "a", "limit": True},
        {"session": "a", "limit": 11},
        {"session": "a", "limit": 0},
        {"session": "a", "limit": "1"},
        {"session": "a", "limit": 1, "path": "/delete"},
    ],
)
def test_invalid_parameters_never_reach_server(application, parameters):
    world, calls = application
    with pytest.raises(WorldUsageError):
        world.read_only().observe("history", parameters)
    assert calls == []


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_direct_mutation_stays_denied(application, method):
    world, calls = application
    with pytest.raises(WorldUsageError, match="read_only"):
        world.read_only().request("app", method, "/read", json={"operation": "delete"})
    with pytest.raises(WorldUsageError, match="unknown"):
        world.read_only().observe("delete", {})
    assert calls == []


@pytest.mark.parametrize(
    "change",
    [
        {"path": "/{{session}}"},
        {"path": "//evil/read"},
        {"body": {"operation": "history {{session}}"}},
        {"body": {"{{session}}": "history"}},
        {"parameters": {"session": {"type": "string"}}},
        {"parameters": {"session": {"type": "object"}}},
        {"parameters": {"session": {"type": "integer", "minimum": True, "maximum": 2}}},
    ],
)
def test_contract_rejects_unbounded_or_nonliteral_shapes(change):
    with pytest.raises(ValidationError):
        Runtime(observations={"history": {**CATALOGUE["history"], **change}})


def test_actual_worker_keeps_catalogue_private_session_and_redaction(
    application, monkeypatch, tmp_path
):
    from fi.alk.harness import process_runtime, hosted_scheduler

    original = process_runtime._allowlisted_ambient_env
    monkeypatch.setattr(
        process_runtime,
        "_allowlisted_ambient_env",
        lambda env: {
            **original(env),
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
    )
    monkeypatch.setattr(hosted_scheduler.tempfile, "gettempdir", lambda: str(tmp_path))
    world, calls = application
    source = """def ready(world):
    response = world.observe('history', {'session': 'abc', 'limit': 3})
    assert response['json']['events'] == ['completed']
    try:
        world.request('app', 'POST', '/read', json={'operation': 'delete'})
    except Exception:
        pass
    else:
        raise AssertionError('direct POST accepted')
    return response['json']['echo']
"""
    fn = _compile_entry(source, label="named-observation", entry="ready")
    for _ in range(2):
        with ThreadPoolExecutor(max_workers=1) as executor:
            value = asyncio.run(
                _invoke(
                    fn, world.read_only(), timeout=15, phase="ready", executor=executor
                )
            )
        assert "private-test-credential" not in str(value)
    assert len(calls) == 2
    assert all(
        call[1]["operation"] == "history"
        and call[2] == "Bearer private-test-credential"
        for call in calls
    )
    assert list((tmp_path / "alk-phase-workers").iterdir()) == []


def test_catalogue_is_snapshotted_and_selectors_remain_distinct(application):
    base, calls = application
    declarations = json.loads(json.dumps(CATALOGUE))
    declarations["summary"] = {
        **declarations["history"],
        "body": {
            "operation": "summary",
            "session": "{{session}}",
            "limit": "{{limit}}",
        },
    }
    world = ObservationWorld(
        0, random.Random(0), base._endpoints, observations=declarations
    )
    declarations["history"]["body"]["operation"] = "delete"
    for name in ("history", "summary"):
        world.read_only().observe(name, {"session": "abc", "limit": 3})
    assert [call[1]["operation"] for call in calls] == ["history", "summary"]
    assert all(call[0] == "/read" for call in calls)


def test_rendered_body_remains_bounded(application):
    base, calls = application
    declaration = {
        "large": {
            "capability": "app",
            "path": "/read",
            "body": {"values": ["{{text}}"] * 20},
            "parameters": {"text": {"type": "string", "max_length": 4096}},
        }
    }
    world = ObservationWorld(
        0, random.Random(0), base._endpoints, observations=declaration
    )
    with pytest.raises(WorldUsageError, match="too_large"):
        world.read_only().observe("large", {"text": "x" * 4096})
    assert calls == []
