import asyncio
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import random
import threading
from urllib.parse import parse_qs

import pytest

from fi.alk.harness.hosted_scheduler import _invoke, _PhaseCrashed
from fi.alk.harness.scenario_source import _compile_entry
from fi.alk.harness.world.errors import WorldUsageError
from fi.alk.harness.world.observation import ObservationWorld


@pytest.fixture
def application():
    secrets = {
        "jwt": "private-login-opaque-value",
        "cookie": "private-cookie-opaque-value",
        "api": "private-api-opaque-value",
    }
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
            requests.append((self.path, dict(self.headers), raw))
            assert self.headers.get_all("Content-Type") == [
                "application/x-www-form-urlencoded"
            ]
            assert parse_qs(raw) == {"username": ["fixture"], "password": ["fixture"]}
            self.respond(
                200, {"token": secrets["jwt"], "api_key": secrets["api"]}, cookies=True
            )

        def do_GET(self):
            requests.append((self.path, dict(self.headers), ""))
            if self.path == "/private/overflow":
                self.send_response(200)
                for index in range(65):
                    self.send_header(
                        "Set-Cookie", f"extra{index}=value{index}; Path=/private"
                    )
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path.startswith("/private/rotate"):
                self.send_response(200)
                self.send_header("X-Rotating-Token", self.path.rsplit("/", 1)[-1])
                self.send_header(
                    "Set-Cookie", "session=read-only-mutation; Path=/private"
                )
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            cookie = self.headers.get("Cookie", "")
            if self.path == "/private/ready":
                valid = (
                    "session=" + secrets["cookie"] in cookie
                    and self.headers.get("Authorization") == "Bearer " + secrets["jwt"]
                )
            elif self.path == "/private/api":
                valid = self.headers.get("Authorization") == "Bearer " + secrets["api"]
            elif self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/private/ready")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            else:
                valid = False
            self.respond(200 if valid else 401, {"ok": valid})

        def respond(self, status, body, cookies=False):
            encoded = json.dumps(body).encode()
            self.send_response(status)
            if cookies:
                self.send_header(
                    "Set-Cookie",
                    "session=" + secrets["cookie"] + "; Path=/private; HttpOnly",
                )
                self.send_header(
                    "Set-Cookie", "secure_only=secure-fixture; Path=/; Secure"
                )
                self.send_header(
                    "Set-Cookie",
                    "expired=expired-fixture; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT",
                )
                self.send_header("X-Session", secrets["jwt"])
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", secrets, requests
    server.shutdown()
    thread.join()
    server.server_close()


@pytest.fixture
def phase(monkeypatch, tmp_path):
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

    def invoke(source, world, entry="setup"):
        fn = _compile_entry(source, label="private-http-session", entry=entry)
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = asyncio.run(
                _invoke(fn, world, timeout=15, phase=entry, executor=executor)
            )
        assert list((tmp_path / "alk-phase-workers").iterdir()) == []
        return result

    return invoke


SETUP = """def setup(world):
    response = world.request('app', 'POST', '/login', form={'username': 'fixture', 'password': 'fixture'}, capture_headers=['X-Session'])
    assert response['status'] == 200
    assert response['headers']['X-Session'] == response['json']['token']
    world.set_headers('app', {'Authorization': 'Bearer ' + response['json']['token']})
    assert world.request('app', 'GET', '/private/api', headers={'authorization': 'Bearer ' + response['json']['api_key']})['status'] == 200
"""
READY = """def ready(world):
    return None if world.request('app', 'GET', '/private/ready')['status'] == 200 else 'not authenticated'
"""


def test_phase_output_cannot_publish_login_credentials(
    application, phase, capfd, caplog
):
    address, secrets, _ = application
    world = ObservationWorld(0, random.Random(1), {"app": address})
    caplog.set_level(logging.INFO, logger="fi.alk.harness.hosted_scheduler")
    source = (
        SETUP
        + """    import os, subprocess, sys
    print(response, flush=True)
    print(response, file=sys.stderr, flush=True)
    os.write(1, response['json']['token'].encode())
    os.write(2, response['json']['token'].encode())
    subprocess.run([sys.executable, '-c', 'import sys; print(sys.argv[1])', response['json']['token']], check=True)
"""
    )
    assert phase(source, world) is None
    assert phase(READY, world.read_only(), "ready") is None
    captured = capfd.readouterr()
    public = captured.out + captured.err + caplog.text
    assert all(secret not in public for secret in secrets.values())
    assert '"status": 200' in caplog.text
    assert "route_hash" in caplog.text


def test_real_setup_ready_check_workers_keep_private_auth(application, phase, caplog):
    address, secrets, requests = application
    world = ObservationWorld(0, random.Random(1), {"app": address, "other": address})
    caplog.set_level(logging.INFO, logger="fi.alk.harness.hosted_scheduler")
    assert phase(SETUP, world) is None
    clone = world.read_only()
    assert phase(READY, clone, "ready") is None
    assert (
        phase(READY.replace("def ready", "def check"), world.read_only(), "check")
        is None
    )
    assert world.request("other", "GET", "/private/ready")["status"] == 401
    fresh = ObservationWorld(0, random.Random(1), {"app": address})
    assert phase(READY, fresh.read_only(), "ready") == "not authenticated"
    assert world.request("app", "GET", "/redirect")["status"] == 302
    assert requests[-1][0] == "/redirect"
    # Cookie path, Secure and expiry attributes survive worker transfer.
    assert "session=" not in requests[-1][1].get("Cookie", "")
    for _, headers, _ in requests:
        assert "secure_only=" not in headers.get("Cookie", "")
        assert "expired=" not in headers.get("Cookie", "")
    public = json.dumps(world._execution_state()) + caplog.text
    assert all(secret not in public for secret in secrets.values())
    assert '"status": 401' in caplog.text
    assert '"status": 200' in caplog.text
    assert "/private/ready" not in caplog.text
    assert "route_hash" in caplog.text


def test_failed_ready_preserves_safe_evidence_and_redacts_verdict(
    application, phase, caplog
):
    address, secrets, _ = application
    world = ObservationWorld(0, random.Random(1), {"app": address})
    phase(SETUP, world)
    caplog.set_level(logging.INFO, logger="fi.alk.harness.hosted_scheduler")
    source = """def ready(world):
    world.request('app', 'GET', '/token-in-path?secret=query-token')
    return world._sessions['app'].headers['Authorization']
"""
    verdict = phase(source, world.read_only(), "ready")
    assert "<redacted>" in verdict
    assert secrets["jwt"] not in verdict + caplog.text
    assert "token-in-path" not in caplog.text and "query-token" not in caplog.text
    assert '"status": 401' in caplog.text


@pytest.mark.parametrize(
    "state",
    [
        {"missing": {"headers": {}, "cookies": []}},
        {"app": {"headers": {"Host": "host"}, "cookies": []}},
        {"app": {"headers": {"X-Test": "x" * 65537}, "cookies": []}},
        {"app": {"headers": {}, "cookies": [{}] * 65}},
        {"app": {"headers": {}, "cookies": [{"name": "malformed"}]}},
    ],
)
def test_private_auth_import_is_bounded_and_validated(state):
    world = ObservationWorld(0, random.Random(1), {"app": "http://127.0.0.1:1"})
    with pytest.raises(WorldUsageError):
        world._restore_private_http_state(state)
    assert world._private_http_state() == {}


def test_read_only_auth_mutation_and_body_conflict_rejected(application):
    address, _, requests = application
    world = ObservationWorld(0, random.Random(1), {"app": address})
    with pytest.raises(WorldUsageError, match="read_only"):
        world.read_only().set_headers("app", {"Authorization": "Bearer forbidden"})
    with pytest.raises(WorldUsageError, match="body_encoding_conflict"):
        world.request("app", "POST", "/login", json={}, form={})
    with pytest.raises(WorldUsageError, match="capture_headers_invalid"):
        world.request("app", "GET", "/private/ready", capture_headers=[{}])
    assert requests == []
    world.request(
        "app",
        "POST",
        "/login",
        form={"username": "fixture", "password": "fixture"},
        headers={"content-type": "application/x-www-form-urlencoded"},
    )


def test_read_only_requests_do_not_mutate_writable_cookies(application):
    address, secrets, _ = application
    world = ObservationWorld(0, random.Random(1), {"app": address})
    world.request(
        "app",
        "POST",
        "/login",
        form={"username": "fixture", "password": "fixture"},
    )
    view = world.read_only()
    assert view.request("app", "GET", "/private/rotate/one")["status"] == 200
    assert world.request("app", "GET", "/private/ready")["status"] == 401
    world.set_headers("app", {"Authorization": "Bearer " + secrets["jwt"]})
    assert world.request("app", "GET", "/private/ready")["status"] == 200


def test_rotating_capture_values_are_bounded_and_common_headers_are_not_redacted(
    application,
):
    address, _, _ = application
    world = ObservationWorld(0, random.Random(1), {"app": address})
    for index in range(300):
        world.request(
            "app",
            "GET",
            f"/private/rotate/token-{index}",
            headers={"Content-Type": "application/json"},
            capture_headers=["X-Rotating-Token"],
        )
    session = world._sessions["app"]
    assert len(session.redactions) == 256
    assert "token-44" not in session.redactions
    assert "token-45" in session.redactions
    assert session.redactions[-1] == "token-299"
    assert "application/json" not in session.redactions


def test_worker_error_keeps_safe_401_evidence(application, phase, caplog):
    address, _, _ = application
    world = ObservationWorld(0, random.Random(1), {"app": address})
    phase(SETUP, world)
    caplog.set_level(logging.INFO, logger="fi.alk.harness.hosted_scheduler")
    with pytest.raises(_PhaseCrashed):
        phase(
            "def ready(world):\n    world.request('app', 'GET', '/secret-route?token=secret-query')\n    raise AssertionError('private-detail')\n",
            world.read_only(),
            "ready",
        )
    assert '"status": 401' in caplog.text
    assert "secret-route" not in caplog.text and "secret-query" not in caplog.text
    assert phase(READY, world.read_only(), "ready") is None


def test_replaced_auth_is_still_redacted(application, phase):
    address, secrets, _ = application
    world = ObservationWorld(0, random.Random(1), {"app": address})
    phase(SETUP, world)
    verdict = phase(
        "def setup(world):\n    old = world._sessions['app'].headers['Authorization']\n    world.set_headers('app', {})\n    return old\n",
        world,
    )
    assert secrets["jwt"] not in verdict
    assert "<redacted>" in verdict


def test_read_only_cookie_overflow_cannot_mutate_parent_or_expose_credential(
    application, phase, caplog
):
    address, secrets, _ = application
    world = ObservationWorld(0, random.Random(1), {"app": address})
    phase(SETUP, world)
    caplog.set_level(logging.INFO, logger="fi.alk.harness.hosted_scheduler")
    source = """def ready(world):
    old = world._sessions['app'].headers['Authorization']
    try:
        world.request('app', 'GET', '/private/overflow')
    except Exception:
        return old
"""
    verdict = phase(source, world.read_only(), "ready")
    assert verdict == "<redacted>"
    assert world._private_http_state()
    assert secrets["jwt"] not in caplog.text
