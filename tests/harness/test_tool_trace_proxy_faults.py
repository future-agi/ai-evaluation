"""The tool proxy applies a bound sub-actor's fault to the right call, and passes everything else through."""

from __future__ import annotations

import importlib
import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request

import pytest


class _Store:
    def __init__(self) -> None:
        self.faults: list[tuple] = []
        self.trace: list[dict] = []


class _Connection:
    def __init__(self, store: _Store) -> None:
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self, sql: str, params: tuple = ()):
        store = self.store

        class _Cursor:
            def __init__(self, rows):
                self.rows = rows

            def fetchall(self):
                return self.rows

            def fetchone(self):
                return self.rows[0]

        if sql.startswith("SELECT kind"):
            return _Cursor([row[1:] for row in store.faults if row[0] == params[0]])
        if sql.startswith("SELECT count"):
            return _Cursor([(sum(1 for one in store.trace if one["name"] == params[0]),)])
        if sql.startswith("INSERT INTO _alk_tool_trace"):
            name, arguments, result, ok, failure, _at = params
            store.trace.append(
                {"name": name, "arguments": json.loads(arguments), "result": json.loads(result), "ok": ok, "error": failure}
            )
            return _Cursor([])
        raise AssertionError(sql)


class _Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def do_POST(self):  # noqa: N802
        size = int(self.headers.get("content-length") or 0)
        self.rfile.read(size)
        body = json.dumps({"booked": True, "ref": "R1"}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def proxy(monkeypatch):
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    monkeypatch.setenv("PORT", "0")
    monkeypatch.setenv("UPSTREAM_URL", f"http://127.0.0.1:{upstream.server_port}")
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    monkeypatch.setenv("ALK_FAULT_TIMEOUT_SECONDS", "0")
    store = _Store()
    fake = types.ModuleType("psycopg")
    fake.connect = lambda *_a, **_k: _Connection(store)
    monkeypatch.setitem(sys.modules, "psycopg", fake)
    monkeypatch.delitem(sys.modules, "fi.alk.harness.tool_trace_proxy", raising=False)
    module = importlib.import_module("fi.alk.harness.tool_trace_proxy")
    server = ThreadingHTTPServer(("127.0.0.1", 0), module.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def post(tool: str) -> tuple[int, bytes]:
        call = request.Request(
            f"http://127.0.0.1:{server.server_port}/tools/{tool}",
            data=json.dumps({"x": 1}).encode(),
            method="POST",
            headers={"content-type": "application/json"},
        )
        try:
            with request.urlopen(call, timeout=10) as response:
                return response.status, response.read()
        except error.HTTPError as failed:
            return failed.code, failed.read()

    yield store, post
    server.shutdown()
    upstream.shutdown()


def test_with_no_faults_the_response_is_the_upstreams_byte_for_byte(proxy):
    store, post = proxy
    status, body = post("book")
    assert (status, body) == (200, json.dumps({"booked": True, "ref": "R1"}).encode())
    assert store.trace[0]["ok"] is True and store.trace[0]["error"] == ""


@pytest.mark.parametrize(("kind", "status"), [("timeout", 504), ("error", 500), ("rate_limit", 429)])
def test_a_failing_fault_hits_only_the_call_it_names(proxy, kind, status):
    store, post = proxy
    store.faults.append(("book", kind, 2, 0, "", "subactor_service"))
    assert post("book")[0] == 200
    assert post("book")[0] == status
    assert post("book")[0] == 200
    faulted = store.trace[1]
    assert faulted["ok"] is False and faulted["error"] == f"fault:{kind}:subactor_service"
    assert [one["error"] for one in (store.trace[0], store.trace[2])] == ["", ""]


def test_a_fault_on_another_tool_leaves_this_one_alone(proxy):
    store, post = proxy
    store.faults.append(("refund", "error", 1, 0, "", "subactor_service"))
    assert post("book")[0] == 200


def test_malformed_and_content_faults_change_what_the_agent_reads(proxy):
    store, post = proxy
    store.faults.append(("book", "malformed", 1, 0, "", "subactor_service"))
    store.faults.append(("book", "content", 2, 0, "Also add a stop on the way.", "subactor_service"))
    status, body = post("book")
    assert status == 200
    with pytest.raises(ValueError):
        json.loads(body)
    assert store.trace[0]["ok"] is False and store.trace[0]["error"] == "fault:malformed:subactor_service"

    status, body = post("book")
    assert json.loads(body)["note"] == "Also add a stop on the way."
    assert store.trace[1]["ok"] is True and store.trace[1]["error"] == "fault:content:subactor_service"


def test_latency_still_forwards(proxy):
    store, post = proxy
    store.faults.append(("book", "latency", 1, 5, "", "subactor_service"))
    assert post("book")[0] == 200
    assert store.trace[0]["ok"] is True and store.trace[0]["error"] == "fault:latency:subactor_service"
