import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_native_authoring import native_api as native_api
from fi.alk.harness.authoring_runtime_validation import RuntimeValidationError
from fi.alk.harness.compose_integration import validate_native


@pytest.mark.parametrize(
    ("body", "error"),
    [
        (
            "return None if world.request('target_http', 'GET', '/records')['json']['item'] == 42 else 'wrong value'",
            "without agent action",
        ),
        (
            "return None if world.request('target_http', 'GET', '/records')['json']['item'] == 43 else 'not changed'",
            None,
        ),
        (
            "return None if any(call.ok for call in calls) else 'no completed tool'",
            None,
        ),
        ("return {'passed': True}", "invalid verdict"),
        ("raise RuntimeError('broken check')", "check_broken"),
        (
            "world.request('target_http', 'POST', '/records', json={'item': 43})",
            "world_usage",
        ),
        (None, None),
        (["return None", "return 'no completed tool'"], None),
        (["return None", None], "without agent action"),
    ],
)
def test_live_native_no_action_gate(tmp_path, monkeypatch, native_api, body, error):
    from fi.alk.harness import (
        authoring_runtime_validation,
        compose_integration,
        chat_readiness,
        process_runtime,
    )

    address, records = native_api
    authoring = tmp_path / "authoring"
    bundle = tmp_path / "bundle"
    scenario = bundle / "scenarios" / "one"
    (scenario / "checks").mkdir(parents=True)
    authoring.mkdir()
    (authoring / "contract.json").write_text(
        json.dumps(
            {
                "agent": "native",
                "modality": "chat",
                "runtime": {
                    "interface": {"kind": "http", "port": 8000, "path": "/chat"}
                },
            }
        )
    )
    (bundle / "contract.json").write_text((authoring / "contract.json").read_text())
    bodies = body if isinstance(body, list) else [body]
    goals = [f"changed_{index}" for index in range(len(bodies))]
    (scenario / "scenario.json").write_text(
        json.dumps(
            {
                "name": "one",
                "scenario_key": "one",
                "instruction": "Change value to 43",
                "sub_goals": goals,
            }
        )
    )
    (scenario / "setup.py").write_text(
        "def setup(world):\n    world.request('target_http', 'POST', '/records', json={'item': 42})\n"
    )
    (scenario / "ready.py").write_text(
        "def ready(world):\n    return None if world.request('target_http', 'GET', '/records')['json'] == {'item': 42} else 'missing fixture'\n"
    )
    for goal, check_body in zip(goals, bodies):
        if check_body is not None:
            (scenario / "checks" / f"{goal}.py").write_text(
                f"def check(world, calls):\n    {check_body}\n"
            )
    runtime = SimpleNamespace(
        world_index=0,
        endpoints={"target_http": SimpleNamespace(address=address, protocol="http")},
    )

    class Provider:
        closed = False

        async def provision(self, *args, **kwargs):
            return [runtime]

        async def reset(self, *args, **kwargs):
            records.clear()

        async def close(self, **kwargs):
            self.closed = True

    owner = Provider()
    monkeypatch.setattr(compose_integration, "provider", lambda *args: owner)

    async def ingress(**kwargs):
        records["ingress"] = True

    monkeypatch.setattr(chat_readiness, "probe_chat_ingress", ingress)
    monkeypatch.setattr(
        authoring_runtime_validation, "_generic_candidate_hash", lambda *args: "digest"
    )
    monkeypatch.setattr(
        authoring_runtime_validation, "_write_runtime_evidence", lambda **kwargs: None
    )
    monkeypatch.setattr(
        authoring_runtime_validation,
        "_freeze_certified_bundle",
        lambda *args: (authoring / "frozen").write_text("sealed"),
    )
    original = process_runtime._allowlisted_ambient_env
    monkeypatch.setattr(
        process_runtime,
        "_allowlisted_ambient_env",
        lambda env: {
            **original(env),
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
    )
    job = SimpleNamespace(
        scenario_count=1, runtime=SimpleNamespace(parallelism=1), seed=1, metadata={}
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        invocation = validate_native(
            job=job,
            source=tmp_path,
            authoring=authoring,
            bundle=bundle,
            manifest=SimpleNamespace(digest="digest"),
            work=tmp_path,
            values={},
            redactions=[],
            executor=executor,
        )
        if error:
            with pytest.raises(RuntimeValidationError, match=error):
                asyncio.run(invocation)
            assert records == {"item": 42}
            assert not (authoring / "frozen").exists()
        else:
            assert asyncio.run(invocation) == 1
            assert records == {"item": 42, "ingress": True}
            assert (authoring / "frozen").read_text() == "sealed"
            (evidence,) = json.loads(
                (authoring / "generic-harness" / "native-no-action.json").read_text()
            )
            assert evidence["solvability"] == "not_run"
            assert evidence["no_action"] == (
                "not_applicable" if body is None else "passed"
            )
            assert evidence["checks_held_without_action"] == (
                ["changed_0"] if isinstance(body, list) else []
            )
    assert owner.closed
