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


@pytest.mark.parametrize("store_kind", ["in_process", "none", "filesystem"])
def test_non_relational_state_does_not_require_sql_invariants(tmp_path, monkeypatch, store_kind):
    def unexpected_review(*args, **kwargs):
        pytest.fail("SQL authoring invoked for an explicitly non-relational empty world")

    monkeypatch.setattr(subject, "Stage", unexpected_review)
    monkeypatch.setattr(subject, "chosen_model", lambda: "unused")
    source = tmp_path / "source"
    source.mkdir()
    (source / "graph.py").write_text("graph = object()\n")
    out = tmp_path / "authoring"
    out.mkdir()
    (out / "contract.json").write_text(
        json.dumps(
            {
                "agent": "graph",
                "tools": [{"name": "research"}],
                "data_store": {"kind": store_kind},
                "data_schema": {"AgentState": {"messages": "list"}},
            }
        )
    )
    world = SimpleNamespace(state=lambda: {"harness_seed_sentinel": []})

    assert asyncio.run(subject.author_invariants(source, out, world)) == []
    review = json.loads((out / subject.ARTIFACT).read_text())
    assert review["status"] == "not_applicable"
    assert "no runtime SQL business tables" in review["reason"]
    assert not review["tool_execution_proven"]
    assert asyncio.run(subject.author_invariants(source, out, world)) == []
    (source / "graph.py").write_text("graph = changed()\n")
    with pytest.raises(ValueError, match="changed"):
        asyncio.run(subject.author_invariants(source, out, world))


@pytest.mark.parametrize(
    "contract_patch,tables",
    [
        ({}, ["orders"]),
        ({"data_store": {"kind": "postgres"}}, []),
        ({"data_store": {"kind": "sqlite"}}, []),
        ({"data_store": {"kind": "none"}}, ["orders"]),
        ({"data_store": {"kind": "filesystem"}}, ["orders"]),
        ({"data_store": {"kind": "in_process"}}, ["orders"]),
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


def test_large_suite_is_shared_across_conversations_told_the_world_checks(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ALK_INVARIANT_REVIEW_SHARE", "10")
    names = [f"case-{index:02d}" for index in range(25)]
    source, out = _suite(tmp_path, names)
    monkeypatch.setattr(subject, "Stage", ReviewStage)
    ReviewStage.sessions = []
    ReviewStage.prompts = []

    checks = asyncio.run(subject.author_invariants(source, out, ReadWorld()))

    assert ReviewStage.sessions == [[names[:10]], [names[10:20]], [names[20:]]]
    assert all(declaration()["violations_sql"] in prompt for prompt in ReviewStage.prompts[1:])
    assert [check["name"] for check in checks if not check["scenarios"]] == [
        declaration()["name"]
    ]
    assert sorted(check["name"] for check in checks if check["scenarios"]) == sorted(
        f"for {name}" for name in names
    )


def test_a_small_suite_with_nothing_earlier_is_one_review_as_always(tmp_path, monkeypatch):
    source, out = _suite(tmp_path, ["a", "b"])
    monkeypatch.setattr(subject, "Stage", ReviewStage)
    ReviewStage.sessions = []
    ReviewStage.prompts = []

    asyncio.run(subject.author_invariants(source, out, ReadWorld()))

    assert ReviewStage.sessions == [[["a", "b"]]]
    assert "already declared" not in ReviewStage.prompts[0]
    assert "More scenarios are listed" not in ReviewStage.prompts[0]


def test_a_conversation_reviews_a_changed_scenario_again_and_withdraws_its_checks(
    tmp_path, monkeypatch
):
    source, out = _suite(tmp_path, ["a"])
    monkeypatch.setattr(subject, "Stage", ReviewStage)
    ReviewStage.sessions = []
    path = out / "scenarios" / "a" / "scenario.json"

    async def run():
        turns, records = asyncio.Queue(), []
        for handed in ({"a": path}, {"a": path}, None):
            turns.put_nowait(handed)
        checks, _ = await subject._review_session(
            subject.source_files(source), {}, ReadWorld(), {}, turns=turns, record=records.append
        )
        assert [check["name"] for check in checks] == [declaration()["name"], "for a"]
        return records

    opening, first, again = asyncio.run(run())
    assert ReviewStage.sessions == [[[], ["a"], ["a"]]]
    assert [check["name"] for check in opening["checks"]] == [declaration()["name"]]
    assert [check["name"] for check in first["checks"]] == ["for a"]
    assert [check["name"] for check in again["checks"]] == ["for a"]
    assert first["session"] == again["session"] and set(again["scenarios"]) == {"a"}


def test_a_failed_relationship_names_the_rows_to_repair():
    world = ReadWorld()
    with pytest.raises(ValueError) as failed:
        asyncio.run(subject.check_invariants(world, [declaration()]))
    message = str(failed.value)
    assert "1 violating rows: {" in message
    assert "missing" in message


def _suite(tmp_path, names):
    source = tmp_path / "source"
    source.mkdir()
    (source / "lookup.py").write_text("return records[alias.target]\n")
    out = tmp_path / "authoring"
    out.mkdir()
    (out / "world.sqlite").write_text("rows v1")
    for name in names:
        for root, indent in ((out / "scenarios", None), (out / "environment-bundle" / "scenarios", 2)):
            (root / name).mkdir(parents=True)
            (root / name / "scenario.json").write_text(
                json.dumps({"scenario_key": name, "goal": "g"}, indent=indent)
            )
            (root / name / "setup.py").write_text("def setup(world): pass\n")
    return source, out


class ReviewStage:
    """A review conversation as the model holds one: each turn reads the scenarios it is handed,
    declares the universal check on its first turn unless told one exists, and one scoped check per
    scenario."""

    sessions: list = []
    prompts: list = []

    def __init__(self, spec, **kwargs):
        listed = spec.system_prompt.split("Scenarios:\n")[1].split("\nSource files:")[0]
        self.initial = [name for name in listed.split("\n") if name]
        self.known = "already declared by another review" in spec.system_prompt
        self.turns = []
        ReviewStage.sessions.append(self.turns)
        ReviewStage.prompts.append(spec.system_prompt)
        self.tools = {tool.name: tool.handler for tool in spec.servers["source_data"].tools}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def say(self, message, *args):
        if message.startswith("Review is incomplete"):
            return
        names = (
            [name for name in message.split(":\n", 1)[1].split("\n") if name]
            if self.turns
            else self.initial
        )
        self.turns.append(names)
        for start in range(0, len(names), 10):
            await self.tools["read_scenarios"]({"names": names[start : start + 10]})
        if not self.known and len(self.turns) == 1:
            await self.tools["declare_invariant"](declaration())
        for name in names:
            await self.tools["declare_invariant"](
                declaration() | {"name": f"for {name}", "scenarios": [name]}
            )
        await self.tools["finish_review"]({})


def test_scenarios_reviewed_while_writing_are_not_reviewed_again(tmp_path, monkeypatch):
    import threading

    monkeypatch.setattr(subject, "Stage", ReviewStage)
    ReviewStage.sessions = []
    source, out = _suite(tmp_path, ["a", "b", "c"])
    done = threading.Event()
    done.set()
    asyncio.run(
        subject.review_while_writing(
            source,
            out,
            ReadWorld(),
            endpoints={},
            world_key=subject.world_digest(out),
            done=done,
            expected=3,
        )
    )
    assert ReviewStage.sessions == [[[], ["a", "b", "c"]]]

    (out / "scenarios" / "b" / "setup.py").write_text("def setup(world): changed\n")
    checks = asyncio.run(subject.author_invariants(source, out, ReadWorld()))

    assert ReviewStage.sessions == [[[], ["a", "b", "c"]], [["b"]]]
    assert "already declared by another review" in ReviewStage.prompts[-1]
    assert sorted(check["name"] for check in checks) == [
        declaration()["name"],
        "for a",
        "for b",
        "for c",
    ]


class UnfinishedTurnStage(ReviewStage):
    """Reviews the world as usual, then reads handed scenarios without declaring or finishing."""

    async def say(self, message, *args):
        if not self.turns or message.startswith("Review is incomplete"):
            return await super().say(message, *args)
        names = [name for name in message.split(":\n", 1)[1].split("\n") if name]
        self.turns.append(names)
        await self.tools["read_scenarios"]({"names": names})


def test_scenarios_an_unfinished_turn_was_handed_are_reviewed_again_at_the_end(
    tmp_path, monkeypatch
):
    import threading

    monkeypatch.setattr(subject, "Stage", UnfinishedTurnStage)
    ReviewStage.sessions = []
    source, out = _suite(tmp_path, ["a", "b"])
    done = threading.Event()
    done.set()
    asyncio.run(
        subject.review_while_writing(
            source,
            out,
            ReadWorld(),
            endpoints={},
            world_key=subject.world_digest(out),
            done=done,
            expected=2,
        )
    )
    checks = asyncio.run(subject.author_invariants(source, out, ReadWorld()))

    assert ReviewStage.sessions == [[[], ["a", "b"]], [["a", "b"]]]
    assert sorted(check["name"] for check in checks) == [declaration()["name"], "for a", "for b"]


def test_a_changed_world_discards_what_was_reviewed_while_writing(tmp_path, monkeypatch):
    import threading

    monkeypatch.setattr(subject, "Stage", ReviewStage)
    ReviewStage.sessions = []
    source, out = _suite(tmp_path, ["a", "b"])
    done = threading.Event()
    done.set()
    asyncio.run(
        subject.review_while_writing(
            source,
            out,
            ReadWorld(),
            endpoints={},
            world_key=subject.world_digest(out),
            done=done,
            expected=2,
        )
    )
    (out / "world.sqlite").write_text("rows v2")
    asyncio.run(subject.author_invariants(source, out, ReadWorld()))

    assert ReviewStage.sessions == [[[], ["a", "b"]], [["a", "b"]]]
    assert "already declared" not in ReviewStage.prompts[-1]


def test_saved_scenarios_are_compared_by_content_not_formatting(tmp_path):
    _, out = _suite(tmp_path, ["a"])
    (out / "scenarios" / "half").mkdir()
    (out / "scenarios" / "half" / "scenario.json").write_text('{"scenario_key": ')

    assert list(subject.sealed_scenarios(out)) == ["a"]
    assert subject.scenario_digest(out / "scenarios" / "a") == subject.scenario_digest(
        out / "environment-bundle" / "scenarios" / "a"
    )
    before = subject.scenario_digest(out / "scenarios" / "a")
    document = out / "scenarios" / "a" / "scenario.json"
    document.write_text(json.dumps(json.loads(document.read_text()) | {"keywords": ["x"]}))
    assert subject.scenario_digest(out / "scenarios" / "a") == before
    document.write_text(json.dumps(json.loads(document.read_text()) | {"goal": "other"}))
    assert subject.scenario_digest(out / "scenarios" / "a") != before


def test_a_large_suite_is_shared_while_writing_and_full_conversations_are_replaced(
    tmp_path, monkeypatch
):
    import threading

    monkeypatch.setenv("ALK_INVARIANT_REVIEW_SHARE", "2")
    monkeypatch.setattr(subject, "Stage", ReviewStage)
    ReviewStage.sessions = []
    ReviewStage.prompts = []
    names = [f"s{index}" for index in range(12)]
    source, out = _suite(tmp_path, names)
    done = threading.Event()
    done.set()
    asyncio.run(
        subject.review_while_writing(
            source,
            out,
            ReadWorld(),
            endpoints={},
            world_key=subject.world_digest(out),
            done=done,
            expected=12,
        )
    )

    owner, *shares = ReviewStage.prompts
    assert "already declared by another review" not in owner
    assert all("already declared by another review" in prompt for prompt in shares)
    assert all(len(sum(turns, [])) <= 4 for turns in ReviewStage.sessions)
    assert sorted(sum((sum(turns, []) for turns in ReviewStage.sessions), [])) == sorted(names)
    opened = len(ReviewStage.sessions)
    checks = asyncio.run(subject.author_invariants(source, out, ReadWorld()))
    assert len(ReviewStage.sessions) == opened
    assert [check["name"] for check in checks if not check["scenarios"]] == [
        declaration()["name"]
    ]
    assert len([check for check in checks if check["scenarios"]]) == 12
