import asyncio
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from fi.alk.harness.build import blockers
from fi.alk.harness.catalogue import Catalogue, SubGoal
from fi.alk.harness.contract import AgentContract
from fi.alk.harness.folder import apply_setup, check_ready
from fi.alk.harness.prove import prove
from fi.alk.harness.scenario import Scenario
from fi.alk.harness.scenario_tools import accept_scenario
from fi.alk.harness.world.errors import WorldUsageError
from fi.alk.harness.world.observation import ObservationWorld
from fi.alk.harness.world.snapshot import restore, save
from fi.alk.harness.world.tools import world_tools


def native_world(tmp_path):
    contract = AgentContract(
        agent="native", modality="chat", tools=[{"name": "lookup"}]
    )
    server, world = world_tools(
        contract, tmp_path, native_runtime=True, deferred_runtime=True
    )
    return contract, {tool.name: tool.handler for tool in server.tools}, world


def test_native_authoring_blocks_shadow_state_and_replay(tmp_path):
    contract, tools, world = native_world(tmp_path)
    assert blockers(contract, str(tmp_path), native_runtime=True) == []
    assert blockers(contract, native_runtime=True)
    assert blockers(contract)
    for name, payload in [
        ("create_schema", {"sql": "CREATE TABLE fake(id text)"}),
        ("seed", {"table": "fake", "rows": [{"id": "fake"}]}),
        ("adopt_state", {"module": "os", "callable": "getcwd"}),
        ("run_tool", {"tool_name": "lookup"}),
        ("declare_sequence", {"calls": [{"tool": "lookup"}]}),
        ("run_env_command", {"command": "docker ps"}),
    ]:
        assert asyncio.run(tools[name](payload))["is_error"], name
    assert world.store.key == "in_process"
    assert world.state() == {}
    assert not world.handlers
    assert "DEFERRED" in asyncio.run(tools["check_world"]({}))["content"][0]["text"]
    save(world, tmp_path)
    reopened = restore(tmp_path)
    assert reopened.native_runtime and not reopened.external_runtime
    assert reopened.store.key == "in_process"
    assert not (tmp_path / "world.sqlite").exists()
    world.close()
    reopened.close()


def test_native_proof_is_deferred_and_never_executes_reference(tmp_path):
    _, _, world = native_world(tmp_path)
    save(world, tmp_path)
    catalogue = Catalogue(
        sub_goals=[
            SubGoal(
                name="lookup",
                check="def check(world, calls):\n    return None if any(c.ok and c.result == 42 for c in calls) else 'missing value'\n",
            )
        ]
    )
    scenario = Scenario(
        name="lookup",
        instruction="Find the stored value",
        tests="Retrieves the exact stored integer",
        sub_goals=["lookup"],
        solution=[
            {"tool": "lookup", "arguments": {"id": "first"}},
            {"tool": "lookup", "arguments": {"id": "second"}},
        ],
        setup_code="def setup(world):\n    raise RuntimeError('must never run locally')\n",
        ready_code="def ready(world):\n    return 'must run live'\n",
    )
    proof = prove(scenario, catalogue, tmp_path)
    assert proof.deferred and not proof.holds and not proof.ready and not proof.solvable
    kept = []
    result = accept_scenario(
        scenario.model_dump(),
        world_root=tmp_path,
        catalogue=catalogue,
        kept=kept,
        spoken=False,
    )
    assert not result.get("is_error"), result
    assert "DEFERRED" in result["content"][0]["text"]
    assert kept[0].solution == scenario.solution
    bad = scenario.model_copy(
        update={"setup_code": "def setup(world):\n    world.put('fake', {})\n"}
    )
    assert prove(bad, catalogue, tmp_path).broken
    bad = scenario.model_copy(
        update={"ready_code": "def ready(world):\n syntax error !"}
    )
    assert prove(bad, catalogue, tmp_path).broken
    world.close()


@pytest.fixture
def native_api():
    records = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            records.update(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(201)
            self.end_headers()
            self.wfile.write(b"{}")

        def do_GET(self):
            self.send_response(302 if self.path == "/redirect" else 200)
            if self.path == "/redirect":
                self.send_header("Location", "http://127.0.0.1:1/private")
            self.end_headers()
            self.wfile.write(json.dumps(records).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", records
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_real_api_fixture_setup_readiness_and_read_only(native_api):
    address, records = native_api
    world = ObservationWorld(0, random.Random(1), {"agent": address})
    scenario = Scenario(
        name="fixture",
        setup_code="def setup(world):\n    assert world.request('agent', 'POST', '/records', json={'item': 42})['status'] == 201\n",
        ready_code="def ready(world):\n    return None if world.request('agent', 'GET', '/records')['json'] == {'item': 42} else 'wrong native state'\n",
    )
    assert apply_setup(scenario, world).ok
    assert records == {"item": 42}
    assert check_ready(scenario, world).ok
    records["item"] = 43
    assert not check_ready(scenario, world).ok
    with pytest.raises(WorldUsageError, match="read_only"):
        world.read_only().request("agent", "POST", "/records", json={"item": 0})
    assert records == {"item": 43}
    assert world.request("agent", "GET", "/redirect")["status"] == 302
    for path in [
        "https://example.com/",
        "//example.com/",
        "/\\example.com",
        "/records#fragment",
    ]:
        with pytest.raises(WorldUsageError, match="relative_path"):
            world.request("agent", "GET", path)
    assert world._execution_state()["endpoints"] == {"agent": address}


def test_native_phase_worker_receives_endpoints_and_preserves_read_only(
    native_api, tmp_path
):
    import subprocess
    import sys

    address, records = native_api
    world = ObservationWorld(0, random.Random(1), {"agent": address})
    for index, (handle, source, entry) in enumerate(
        [
            (
                world,
                "def setup(world):\n    world.request('agent', 'POST', '/records', json={'item': 42})\n",
                "setup",
            ),
            (
                world.read_only(),
                "def ready(world):\n    return None if world.request('agent', 'GET', '/records')['json']['item'] == 42 else 'wrong value'\n",
                "ready",
            ),
            (
                world.read_only(),
                "def ready(world):\n    world.request('agent', 'POST', '/records', json={'item': 0})\n",
                "ready",
            ),
        ]
    ):
        result_path = tmp_path / f"phase-{index}.json"
        subprocess.run(
            [sys.executable, "-m", "fi.alk.harness.phase_worker", str(result_path)],
            input=json.dumps(
                {"world": handle._execution_state(), "source": source, "entry": entry}
            ),
            text=True,
            check=True,
            capture_output=True,
            timeout=20,
        )
        result = json.loads(result_path.read_text())
        if index == 2:
            assert result["world_error"] == "WorldUsageError"
        else:
            assert result["value"] is None and not result["invalid_verdict"]
    assert records == {"item": 42}


def test_native_save_preserves_checks_without_claiming_proof(tmp_path):
    from fi.alk.harness.catalogue import load_catalogue

    _, tools, world = native_world(tmp_path)
    added = asyncio.run(
        tools["add_sub_goal"](
            {
                "name": "exact_value",
                "what": "Returns the exact native record value",
                "check": "def check(world, calls):\n    return None if world.request('target_http', 'GET', '/value')['json'] == {'value': 42} else 'wrong value'\n",
            }
        )
    )
    assert not added.get("is_error"), added
    prompt = "You are a caller with persona {{ persona }}. Follow {{ instruction }} and ask natural follow-up questions. Do not act as the assistant or invent its responses."
    result = asyncio.run(tools["write_simulator_prompt"]({"prompt": prompt}))
    assert not result.get("is_error"), result
    result = asyncio.run(tools["save_world"]({}))
    assert not result.get("is_error"), result
    assert "DEFERRED" in result["content"][0]["text"]
    assert load_catalogue(tmp_path).sub_goals[0].deterministic()
    assert json.loads((tmp_path / "manifest.json").read_text())["native_runtime"]
    world.close()


def test_native_selection_requires_optin_and_actual_compose_selection(
    tmp_path, monkeypatch
):
    from fi.alk.harness.build import uses_native_runtime

    (tmp_path / "compose.yaml").write_text(
        'services:\n  app:\n    image: python:3.12\n    ports: ["3000:3000"]\n'
    )
    contract = AgentContract(
        agent="native",
        modality="chat",
        runtime={
            "execution": "compose",
            "compose_file": "compose.yaml",
            "service": "app",
            "interface": {"kind": "http", "port": 3000, "path": "/chat"},
        },
    )
    monkeypatch.delenv("ALK_TRUSTED_COMPOSE_SOCKET", raising=False)
    monkeypatch.delenv("ALK_COMPOSE_BROKER_SOCKET", raising=False)
    assert not uses_native_runtime(contract, str(tmp_path))
    monkeypatch.setenv("ALK_COMPOSE_BROKER_SOCKET", "/private/broker.sock")
    assert uses_native_runtime(contract, str(tmp_path))
    contract.runtime = contract.runtime.model_copy(
        update={
            "execution": "native",
            "compose_file": "",
            "service": "",
            "command": ["python", "app.py"],
        }
    )
    assert not uses_native_runtime(contract, str(tmp_path))


def test_native_selection_uses_public_configuration_without_optional_dependencies(
    tmp_path, monkeypatch
):
    from fi.alk.harness.build import uses_native_runtime

    (tmp_path / "compose.yaml").write_text("""services:
  app:
    image: ${APP_IMAGE}
    depends_on: [db]
    environment: ["OPTIONAL_URL=http://optional:8000"]
  db:
    image: db:1
  optional:
    image: optional:1
""")
    contract = AgentContract(
        agent="native",
        modality="chat",
        runtime={
            "execution": "compose",
            "compose_file": "compose.yaml",
            "service": "app",
            "config_files": {".env": {"values": {"APP_IMAGE": "example/app:1"}}},
            "interface": {"kind": "http", "port": 3000, "path": "/chat"},
        },
    )
    monkeypatch.setenv("ALK_COMPOSE_BROKER_SOCKET", "/private/broker.sock")
    assert uses_native_runtime(contract, str(tmp_path))
