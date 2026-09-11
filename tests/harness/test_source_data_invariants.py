import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest

from fi.alk.harness import source_data_invariants as subject


def test_probe_only_exposes_local_http_capabilities_and_never_follows_redirects(
    monkeypatch,
):
    import requests

    services = subject.local_services(
        {
            "tools": SimpleNamespace(address="http://127.0.0.1:15000"),
            "external": SimpleNamespace(address="https://provider.example"),
            "database": SimpleNamespace(address="postgresql://localhost/db"),
        }
    )
    assert services == {"tools": "http://127.0.0.1:15000"}

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def request(self, method, url, **kwargs):
            assert self.trust_env is False
            assert kwargs["allow_redirects"] is False
            assert kwargs["timeout"] == 10
            assert url == "http://127.0.0.1:15000/lookup"
            return SimpleNamespace(status_code=302, json=lambda: {"redirect": True})

    monkeypatch.setattr(requests, "Session", Session)
    args = {"service": "tools", "path": "/lookup", "method": "POST", "arguments": {}}
    assert subject.probe_local_service(services, args)[0] == 302
    for path in ("https://provider.example", "//provider.example", "/lookup#fragment"):
        with pytest.raises(ValueError):
            subject.probe_local_service(services, args | {"path": path})
    with pytest.raises(ValueError):
        subject.probe_local_service(services, args | {"service": "external"})


class ReadWorld:
    def __init__(self):
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(
            "CREATE TABLE aliases (target TEXT); CREATE TABLE records (id TEXT);"
            "INSERT INTO aliases VALUES ('missing');"
        )
        self.db.execute("PRAGMA query_only=ON")

    def query(self, sql):
        return [dict(row) for row in self.db.execute(sql).fetchall()]


def test_data_free_review_is_explicit_and_bound_to_source_and_contract(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    code = source / "agent.py"
    code.write_text("class Assistant: pass\n")
    out = tmp_path / "authoring"
    out.mkdir()
    contract = out / "contract.json"
    contract.write_text(
        json.dumps({"agent": "starter", "tools": [], "real_use_cases": ["Chat"]})
    )
    from fi.alk.harness.world.handle import ReadOnlyWorld

    world = ReadOnlyWorld(
        SimpleNamespace(
            world_index=0,
            rng=None,
            state=lambda table=None: {
                "harness_seed_sentinel": [{"id": "ready"}],
                "_alk_tool_trace": [],
            },
        )
    )
    assert asyncio.run(subject.author_invariants(source, out, world)) == []
    review = json.loads((out / subject.ARTIFACT).read_text())
    assert review["status"] == "not_applicable"
    assert not review["tool_execution_proven"]
    assert review["source_sha256"]["agent.py"]
    assert asyncio.run(subject.author_invariants(source, out, world)) == []
    code.write_text("class Assistant: changed = True\n")
    with pytest.raises(ValueError, match="changed"):
        asyncio.run(subject.author_invariants(source, out, world))


@pytest.mark.parametrize(
    "contract_patch,tables",
    [
        ({}, ["orders"]),
        ({"data_store": {"kind": "postgres"}}, []),
        ({"data_schema": {"orders": {}}}, []),
        ({"tools": [{"name": "order"}]}, []),
    ],
)
def test_empty_checks_never_exempt_data_bearing_worlds(
    tmp_path, contract_patch, tables
):
    (tmp_path / "contract.json").write_text(
        json.dumps(
            {"agent": "starter", "tools": [], "real_use_cases": ["Chat"]}
            | contract_patch
        )
    )
    (tmp_path / subject.ARTIFACT).write_text('{"checks": []}')
    with pytest.raises(ValueError, match="no executable checks"):
        asyncio.run(
            subject.author_invariants(
                tmp_path,
                tmp_path,
                SimpleNamespace(state=lambda: dict.fromkeys(tables, [])),
            )
        )


def declaration():
    return {
        "name": "aliases resolve to source records",
        "violations_sql": "SELECT a.target FROM aliases a LEFT JOIN records r ON a.target=r.id WHERE r.id IS NULL LIMIT 100",
        "evidence": [{"path": "lookup.py", "quote": "return records[alias.target]"}],
    }


def test_source_evidence_is_exact_and_contained(tmp_path):
    (tmp_path / "lookup.py").write_text("return records[alias.target]\n")
    (tmp_path / ".env").write_text("SECRET=do-not-read")
    files = subject.source_files(tmp_path)
    assert set(files) == {"lookup.py"}
    check = subject.validate_evidence(declaration(), files)
    assert len(check["evidence"][0]["sha256"]) == 64
    bad = declaration()
    bad["evidence"][0]["quote"] = "this quote does not exist in the submitted code"
    with pytest.raises(ValueError, match="does not occur"):
        subject.validate_evidence(bad, files)


def test_generic_relationship_without_database_fk_is_rejected_then_passes():
    world = ReadWorld()
    with pytest.raises(ValueError, match="aliases resolve"):
        asyncio.run(subject.check_invariants(world, [declaration()]))
    world.db.execute("PRAGMA query_only=OFF")
    world.db.execute("INSERT INTO records VALUES ('missing')")
    world.db.execute("PRAGMA query_only=ON")
    asyncio.run(subject.check_invariants(world, [declaration()]))


def test_all_failing_relationships_are_reported_in_one_repair_set():
    world = ReadWorld()
    second = {
        **declaration(),
        "name": "every alias has a corresponding label",
    }
    with pytest.raises(ValueError) as failed:
        asyncio.run(subject.check_invariants(world, [declaration(), second]))
    assert "aliases resolve to source records" in str(failed.value)
    assert "every alias has a corresponding label" in str(failed.value)


def test_validation_queries_cannot_change_the_database():
    world = ReadWorld()
    check = {**declaration(), "violations_sql": "DELETE FROM aliases"}
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        asyncio.run(subject.check_invariants(world, [check]))
    assert len(world.query("SELECT * FROM aliases")) == 1


def test_scenario_precondition_is_not_imposed_on_baseline_or_negative_sibling():
    world = ReadWorld()
    check = {**declaration(), "scenarios": ["positive-case"]}
    asyncio.run(subject.check_invariants(world, [check]))
    asyncio.run(subject.check_invariants(world, [check], scenario_key="negative-case"))
    with pytest.raises(ValueError, match="aliases resolve"):
        asyncio.run(
            subject.check_invariants(world, [check], scenario_key="positive-case")
        )


def test_author_preserves_failing_check_and_reuses_it_during_data_repair(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "lookup.py").write_text("return records[alias.target]\n")
    out = tmp_path / "authoring"
    out.mkdir()
    stages = []

    class Stage:
        def __init__(self, spec, **kwargs):
            stages.append(spec)
            self.tools = {
                tool.name: tool.handler for tool in spec.servers["source_data"].tools
            }

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def say(self, *args):
            result = await self.tools["declare_invariant"](declaration())
            assert json.loads(result["content"][0]["text"])["violating_rows"] == 1
            await self.tools["finish_review"]({})

    monkeypatch.setattr(subject, "Stage", Stage)
    world = ReadWorld()
    checks = asyncio.run(subject.author_invariants(source, out, world))
    assert len(checks) == 1
    assert asyncio.run(subject.author_invariants(source, out, world)) == checks
    assert len(stages) == 1
    assert not json.loads((out / subject.ARTIFACT).read_text())["tool_execution_proven"]
    (source / "lookup.py").write_text(
        "# changed source\nreturn records[alias.target]\n"
    )
    with pytest.raises(ValueError, match="Source changed"):
        asyncio.run(subject.author_invariants(source, out, world))


def test_large_suite_can_be_reviewed_in_batches_within_bounded_turns(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "lookup.py").write_text("return records[alias.target]\n")
    out = tmp_path / "authoring"
    scenarios = out / "scenarios"
    scenarios.mkdir(parents=True)
    names = [f"case-{index:02d}" for index in range(25)]
    for name in names:
        folder = scenarios / name
        folder.mkdir()
        (folder / "scenario.json").write_text(
            json.dumps({"scenario_key": name, "goal": "exercise lookup"})
        )
        (folder / "setup.py").write_text("def setup(world):\n    return None\n")
        (folder / "ready.py").write_text("def ready(world):\n    return True\n")

    class Stage:
        def __init__(self, spec, **kwargs):
            self.tools = {
                tool.name: tool.handler for tool in spec.servers["source_data"].tools
            }

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def say(self, *args):
            for start in range(0, len(names), 10):
                result = await self.tools["read_scenarios"](
                    {"names": names[start : start + 10]}
                )
                assert len(json.loads(result["content"][0]["text"])["scenarios"]) <= 10
            await self.tools["declare_invariant"](declaration())
            result = await self.tools["finish_review"]({})
            assert json.loads(result["content"][0]["text"])["saved"] == 1

    monkeypatch.setattr(subject, "Stage", Stage)
    checks = asyncio.run(subject.author_invariants(source, out, ReadWorld()))
    assert [check["name"] for check in checks] == [declaration()["name"]]
