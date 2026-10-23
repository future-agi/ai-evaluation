import json
import logging
import random

import pytest

from fi.alk.harness.world.handle import HostedWorld
from fi.alk.harness.world.stores.postgres import AttachedPostgresStore
import test_observation_http_sessions as sessions
import test_named_observations as named

application = sessions.application
phase = sessions.phase
named_application = named.application
SETUP, READY = sessions.SETUP, sessions.READY
CATALOGUE = named.CATALOGUE


def hosted(tmp_path, endpoints, observations=None):
    return HostedWorld(
        AttachedPostgresStore("postgresql://unused/unused"),
        0,
        random.Random(0),
        {},
        _validate_baseline=False,
        workspace_roots={"files": tmp_path.resolve()},
        endpoints=endpoints,
        observations=observations,
    )


def test_real_process_workers_combine_files_http_and_private_auth(
    application, phase, tmp_path, caplog
):
    address, secrets, requests = application
    world = hosted(tmp_path, {"app": address})
    caplog.set_level(logging.INFO, logger="fi.alk.harness.hosted_scheduler")
    readonly = world.read_only()
    assert (
        phase(
            SETUP + "    world.write_file('files', 'result.txt', 'verified')\n", world
        )
        is None
    )
    assert (
        phase(
            READY.replace(
                "    return",
                "    assert world.read_file('files', 'result.txt') == 'verified'\n    return",
            ),
            readonly,
            "ready",
        )
        is None
    )
    check = """def check(world):
    assert world.read_file('files', 'result.txt') == 'verified'
    response = world.request('app', 'GET', '/private/ready')
    assert response['status'] == 200
    return response['text']
"""
    verdict = phase(check, readonly, "check")
    public = json.dumps(world._execution_state()) + caplog.text + verdict
    assert all(secret not in public for secret in secrets.values())
    assert '"status": 200' in caplog.text
    assert len(requests) >= 4


def test_real_process_worker_named_post_and_denials(named_application, phase, tmp_path):
    observation, calls = named_application
    world = hosted(tmp_path, observation._endpoints, CATALOGUE)
    assert (
        phase(
            "def setup(world):\n    world.set_headers('app', {'Authorization': 'Bearer private-test-credential'})\n",
            world,
        )
        is None
    )
    source = """def check(world):
    from fi.alk.harness.world.errors import WorldUsageError
    assert world.request('app', 'GET', '/health')['json'] == {'ok': True}
    assert world.observe('history', {'session': 'abc', 'limit': 3})['json']['events'] == ['completed']
    for capability, method, path in [('app', 'POST', '/read'), ('missing', 'GET', '/health')]:
        try:
            world.request(capability, method, path)
        except WorldUsageError:
            pass
        else:
            raise AssertionError('request accepted')
    try:
        world.set_headers('app', {'Authorization': 'changed'})
    except WorldUsageError:
        pass
    else:
        raise AssertionError('headers changed')
"""
    assert phase(source, world.read_only(), "check") is None
    verdict = phase(
        "def check(world):\n    return world.observe('history', {'session': 'abc', 'limit': 3})['json']['echo']\n",
        world.read_only(),
        "check",
    )
    assert "private-test-credential" not in verdict
    assert (
        calls
        == [
            (
                "/read",
                {"operation": "history", "args": {"session": "abc", "limit": 3}},
                "Bearer private-test-credential",
            )
        ]
        * 2
    )


def test_factory_composes_only_declared_http_and_sealed_catalogue(
    tmp_path, monkeypatch
):
    import asyncio
    from fi.alk.harness import hosted_entrypoint as he
    from fi.alk.harness.process_runtime import EnvironmentRuntime, RuntimeEndpoint
    from fi.alk.harness.world.errors import WorldUsageError

    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/build.json").write_text(
        json.dumps({"stores": [{"capability": "db", "row_counts": {}}]})
    )
    (tmp_path / "contract.json").write_text(
        json.dumps({"runtime": {"observations": CATALOGUE}})
    )
    monkeypatch.setattr(HostedWorld, "_visible_tables", lambda self: [])
    runtime = EnvironmentRuntime(
        runtime_id="fixture",
        world_index=0,
        bundle_digest="fixture",
        state="ready",
        endpoints={
            "db": RuntimeEndpoint(
                capability="db",
                protocol="postgres",
                address="postgresql://unused/unused",
            ),
            "app": RuntimeEndpoint(
                capability="app", protocol="http", address="http://127.0.0.1:1"
            ),
        },
    )
    factory = he.ProcessWorldFactory(tmp_path, tmp_path)
    world = asyncio.run(factory.create(runtime, rng=random.Random(0)))
    assert world.capabilities == ("app",)
    assert world._execution_state()["observations"]["history"]["path"] == "/read"
    del runtime.endpoints["app"]
    with pytest.raises(WorldUsageError, match="capability_unavailable"):
        asyncio.run(factory.create(runtime, rng=random.Random(0)))
