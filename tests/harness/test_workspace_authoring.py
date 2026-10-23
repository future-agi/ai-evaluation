import sys
import subprocess
import urllib.request


from fi.alk.harness.catalogue import Catalogue, SubGoal
from fi.alk.harness.contract import AgentContract, Runtime
from fi.alk.harness.folder import apply_setup, check_ready
from fi.alk.harness.prove import prove, workspace_bindings, workspace_fixture
from fi.alk.harness.scenario import Scenario
from fi.alk.harness.scenario_tools import accept_scenario, world_summary
from fi.alk.harness.world.runtime import GeneratedWorld
from fi.alk.harness.world.snapshot import save
from test_native_workspace import world


SETUP = "def setup(world):\n    world.write_file('project', 'settings.py', 'PORT = 8000\\n')\n"
READY = "def ready(world):\n    return None if world.read_file('project', 'settings.py') == 'PORT = 8000\\n' else 'wrong fixture'\n"


def authoring(tmp_path, runtime):
    contract = AgentContract(agent="file-agent", runtime=runtime)
    (tmp_path / "contract.json").write_text(contract.model_dump_json())
    w = GeneratedWorld(":memory:")
    save(w, tmp_path)
    w.close()


def test_context_exposes_bindings_and_absence_without_optional_runtime_crash(tmp_path):
    authoring(tmp_path, None)
    assert workspace_bindings(tmp_path) == {}
    summary = world_summary(tmp_path)
    assert "NO TARGET FILE WORKSPACE" in summary
    assert "self-contained" in summary
    assert "uploaded agent source is not" in summary
    authoring(tmp_path, Runtime(workspaces={"project": "workspace/project"}))
    summary = world_summary(tmp_path)
    assert '"project": "workspace/project"' in summary
    assert "world.write_file" in summary and "world.read_file" in summary
    assert "DEFERRED" in summary


def test_workspace_proof_is_deferred_and_rejects_missing_binding_or_readback(tmp_path):
    scenario = Scenario(name="edit", setup_code=SETUP, ready_code=READY)
    catalogue = Catalogue()
    authoring(tmp_path, Runtime(workspaces={"project": "workspace/project"}))
    proof = prove(scenario, catalogue, tmp_path)
    assert proof.deferred and not proof.ready and not proof.solvable and not proof.holds
    assert not proof.broken
    assert prove(
        scenario.model_copy(
            update={"ready_code": "def ready(world):\n    return None\n"}
        ),
        catalogue,
        tmp_path,
    ).broken
    authoring(tmp_path, Runtime())
    assert prove(scenario, catalogue, tmp_path).broken


def test_variable_fixture_paths_allowed_and_check_syntax_validated(tmp_path):
    authoring(tmp_path, Runtime(workspaces={"project": "workspace/project"}))
    scenario = Scenario(
        name="edit",
        setup_code="def setup(world):\n    for p in ['settings.py']:\n        world.write_file('project', p, 'PORT = 8000\\n')\n",
        ready_code=READY,
        sub_goals=["edited"],
    )
    catalogue = Catalogue(
        sub_goals=[
            SubGoal(
                name="edited",
                check='def check(world, calls):\n    return world.read_file("project", "settings.py")\n',
            )
        ]
    )
    assert not prove(scenario, catalogue, tmp_path).broken
    catalogue.sub_goals[0].check = "def check(world, calls):\n    syntax error !\n"
    assert prove(scenario, catalogue, tmp_path).broken


def test_binding_does_not_defer_ordinary_scenario(tmp_path):
    authoring(tmp_path, Runtime(workspaces={"project": "workspace/project"}))
    proof = prove(Scenario(name="chat"), Catalogue(), tmp_path)
    assert not proof.deferred


def test_real_child_http_sees_setup_ready_and_reset(tmp_path):
    root = tmp_path / "private"
    w = world(root)
    baseline = w.checkpoint()
    scenario = Scenario(name="edit", setup_code=SETUP, ready_code=READY)
    assert apply_setup(scenario, w).ok
    assert check_ready(scenario, w).ok
    code = """from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
class Handler(BaseHTTPRequestHandler):
 def log_message(self, *args): pass
 def do_GET(self):
  self.send_response(200); self.end_headers()
  self.wfile.write(Path('settings.py').read_bytes())
server=HTTPServer(('127.0.0.1',0),Handler)
print(server.server_port,flush=True)
server.serve_forever()
"""
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", code], cwd=root, stdout=subprocess.PIPE, text=True
    )
    try:
        port = int(child.stdout.readline())
        with urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=5) as response:
            assert response.read() == b"PORT = 8000\n"
        (root / "settings.py").write_text("wrong contents")
        assert not check_ready(scenario, w).ok
        w.revert(baseline)
        assert not (root / "settings.py").exists()
        assert not check_ready(scenario, w).ok
    finally:
        child.terminate()
        child.wait(timeout=5)
        child.stdout.close()


def test_workspace_scenario_admitted_without_shadow_fixture_execution(tmp_path):
    authoring(tmp_path, Runtime(workspaces={"project": "workspace/project"}))
    catalogue = Catalogue(
        sub_goals=[
            SubGoal(
                name="edited",
                what="Updated content",
                check='def check(world, calls):\n    return None if world.read_file("project", "settings.py") == "PORT = 9000\\n" else "not updated"\n',
            )
        ]
    )
    scenario = Scenario(
        name="edit",
        instruction="Change the port in settings.py to 9000.",
        tests="Updates the configured port",
        setup_code=SETUP,
        ready_code=READY,
        sub_goals=["edited"],
    )
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
    assert "All three gates pass" not in result["content"][0]["text"]
    assert len(kept) == 1


def test_file_mentions_do_not_defer_proof():
    scenario = Scenario(
        name="ordinary",
        setup_code='def setup(world):\n    # world.write_file("unused", "path", "text")\n    pass\n',
        ready_code="def ready(world):\n    return None\n",
    )
    assert not workspace_fixture(scenario, Catalogue())


def test_workspace_declaration_preserves_table_inspection_and_nonfile_trials(tmp_path):
    import asyncio
    from fi.alk.harness.scenario_tools import scenario_tools

    contract = AgentContract(
        agent="mixed", runtime=Runtime(workspaces={"project": "workspace/project"})
    )
    (tmp_path / "contract.json").write_text(contract.model_dump_json())
    w = GeneratedWorld(":memory:")
    w.connection.execute("CREATE TABLE records (name TEXT)")
    w.connection.execute("INSERT INTO records VALUES ('existing')")
    w.connection.commit()
    save(w, tmp_path)
    w.close()
    server, _ = scenario_tools(contract, tmp_path, tmp_path, wanted=1)
    tools = {tool.name: tool.handler for tool in server.tools}
    result = asyncio.run(tools["inspect_world"]({"table": "records"}))
    assert "existing" in result["content"][0]["text"]
    result = asyncio.run(tools["try_calls"]({"calls": []}))
    assert "DEFERRED" not in result["content"][0]["text"]
    result = asyncio.run(tools["try_calls"]({"calls": [], "setup_code": SETUP}))
    assert "DEFERRED" in result["content"][0]["text"]
