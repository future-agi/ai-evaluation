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


def test_in_process_graph_state_does_not_require_sql_invariants(tmp_path):
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
                "data_store": {"kind": "in_process"},
                "data_schema": {"AgentState": {"messages": "list"}},
            }
        )
    )
    world = SimpleNamespace(state=lambda: {"harness_seed_sentinel": []})

    assert asyncio.run(subject.author_invariants(source, out, world)) == []
    review = json.loads((out / subject.ARTIFACT).read_text())
    assert review["status"] == "not_applicable"
    assert "in-process" in review["reason"].lower()


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


def test_postgres_invariant_sql_runs_on_a_sqlite_world_with_json_arrays():
    connection = sqlite3.connect(":memory:")
    connection.create_function("regexp", 2, subject.sqlite_regexp)
    connection.executescript(
        "CREATE TABLE places (id TEXT, address TEXT, aliases TEXT, needs TEXT);"
        """INSERT INTO places VALUES ('sfo', 'SFO Airport', '["sfo"]', '[]');"""
        """INSERT INTO places VALUES ('tower', 'Salesforce Tower', '[]', '[]');"""
    )
    sql = subject.sqlite_dialect(
        "SELECT id FROM places WHERE ('sfo' = ANY(aliases) OR address ILIKE '%sfo%') "
        "AND COALESCE(needs, '{}') = '{}' AND round(1.5::numeric, 0) = 2 "
        "AND btrim(address) !~ '^x~y' ORDER BY id"
    )

    assert "'^x~y'" in sql and "ILIKE" not in sql
    assert [row[0] for row in connection.execute(sql)] == ["sfo", "tower"]
    with pytest.raises(sqlite3.Error):
        connection.execute(
            subject.sqlite_dialect("SELECT a FROM unnest(ARRAY[1]) AS a")
        )


SEED = """INSERT INTO market_config (market, cash_supported, currency, surge_multiplier, available_products) VALUES
  ('US-SF',  FALSE, 'USD', 1.00, ARRAY['uberx','comfort','uberxl','black','wav']),
  ('US-NYC', FALSE, 'USD', 1.35, ARRAY['uberx','comfort','uberxl','black']),
  ('IN-BLR', TRUE,  'INR', 1.00, ARRAY['uberx','uberxl']);

INSERT INTO places (place_id, formatted_address, city, market, lat, lng, aliases) VALUES
  ('plc_main_st_sf', 'Main Street, San Francisco, CA', 'San Francisco', 'US-SF', 37.791400, -122.392600,
     ARRAY['main street']),
  ('plc_main_st_la', 'Main Street, Los Angeles, CA', 'Los Angeles', 'US-LA', 34.050400, -118.242000,
     ARRAY['main street']);
"""

TOOLS = """    mk = one("SELECT * FROM market_config WHERE market = %s", (p["market"],))
    if not mk:
        raise HTTPException(404, "unknown_market")
    surge = _f(mk["surge_multiplier"]) or 1.0
"""


def market_check(**extra):
    return {
        "name": "places_market_must_exist_in_market_config",
        "violations_sql": "SELECT p.place_id, p.market FROM places p WHERE NOT EXISTS "
        "(SELECT 1 FROM market_config mc WHERE mc.market = p.market) LIMIT 100",
        "evidence": [
            {
                "path": "tools-api/main.py",
                "quote": 'raise HTTPException(404, "unknown_market")',
            }
        ],
        **extra,
    }


SEED_PLACES = [("plc_main_st_sf", "US-SF"), ("plc_main_st_la", "US-LA")]
GENERATED_LA = ("plc_gen_lax_1", "US-LA")


class MarketWorld(ReadWorld):
    def __init__(self, places=SEED_PLACES):
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(
            "CREATE TABLE market_config (market TEXT);"
            "CREATE TABLE places (place_id TEXT, market TEXT);"
            "INSERT INTO market_config VALUES ('US-SF'), ('US-NYC'), ('IN-BLR');"
        )
        self.db.executemany("INSERT INTO places VALUES (?, ?)", places)
        self.db.execute("PRAGMA query_only=ON")


def _ride_source(tmp_path):
    source = tmp_path / "source"
    (source / "db").mkdir(parents=True)
    (source / "tools-api").mkdir()
    (source / "db" / "seed.sql").write_text(SEED)
    (source / "tools-api" / "main.py").write_text(TOOLS)
    return source


def _declare(monkeypatch, source, world, check, scenarios=None):
    replies = []

    class Stage(ReviewStage):
        async def say(self, message, *args):
            if not replies:
                if scenarios:
                    await self.tools["read_scenarios"]({"names": list(scenarios)})
                result = await self.tools["declare_invariant"](check)
                replies.append(json.loads(result["content"][0]["text"]))
                await self.tools["finish_review"]({})

    monkeypatch.setattr(subject, "Stage", Stage)
    checks, _ = asyncio.run(
        subject._review_session(
            subject.source_files(source), scenarios or {}, world, {}
        )
    )
    return replies[0], checks


def test_a_universal_check_only_the_submitted_seed_violates_is_an_intended_case(
    tmp_path, monkeypatch
):
    reply, checks = _declare(
        monkeypatch, _ride_source(tmp_path), MarketWorld(), market_check()
    )
    assert "intended cases" in reply["error"]
    assert "scenarios" not in reply["error"]
    assert checks == []


@pytest.mark.parametrize(
    "places,select",
    [
        ([SEED_PLACES[0], GENERATED_LA], "p.place_id, p.market"),
        ([*SEED_PLACES, GENERATED_LA], "DISTINCT p.market"),
        (
            [("plc_main_st_sf", "US-NYC"), ("plc_main_st_sf", "US-LA")],
            "p.place_id, p.market",
        ),
    ],
    ids=["generated", "market-only", "seed-id-with-changed-market"],
)
def test_generated_rows_still_violate_for_repair(tmp_path, monkeypatch, places, select):
    check = market_check()
    check["violations_sql"] = check["violations_sql"].replace(
        "p.place_id, p.market", select
    )
    reply, checks = _declare(
        monkeypatch, _ride_source(tmp_path), MarketWorld(places), check
    )
    assert reply["violating_rows"] == 1
    assert [check["name"] for check in checks] == [market_check()["name"]]


def test_a_seed_row_among_generated_violations_must_be_excluded(tmp_path, monkeypatch):
    source = _ride_source(tmp_path)
    world = MarketWorld([*SEED_PLACES, GENERATED_LA])
    reply, checks = _declare(monkeypatch, source, world, market_check())
    assert "plc_main_st_la" in reply["error"]
    assert "plc_gen_lax_1" not in reply["error"]
    assert checks == []

    check = market_check()
    check["violations_sql"] = check["violations_sql"].replace(
        "WHERE", "WHERE p.place_id <> 'plc_main_st_la' AND", 1
    )
    reply, checks = _declare(monkeypatch, source, world, check)
    assert reply["violating_rows"] == 1
    assert [check["name"] for check in checks] == [check["name"]]


def test_a_scoped_check_the_seed_violates_is_kept(tmp_path, monkeypatch):
    source = _ride_source(tmp_path)
    (tmp_path / "suite").mkdir()
    _, out = _suite(tmp_path / "suite", ["book-la"])
    scenarios = {"book-la": out / "scenarios" / "book-la" / "scenario.json"}
    reply, checks = _declare(
        monkeypatch,
        source,
        MarketWorld(),
        market_check(scenarios=["book-la"]),
        scenarios,
    )
    assert reply["violating_rows"] == 1
    assert [check["scenarios"] for check in checks] == [["book-la"]]


def test_authoring_drops_a_recorded_universal_check_only_the_seed_violates(
    tmp_path, monkeypatch
):
    source = _ride_source(tmp_path)
    (tmp_path / "suite").mkdir()
    _, out = _suite(tmp_path / "suite", ["book-sf"])
    files = subject.source_files(source)
    seed_only = subject.validate_evidence(market_check(), files)
    scoped = subject.validate_evidence(
        market_check(name="sf place exists", scenarios=["book-sf"])
        | {"violations_sql": "SELECT 1 FROM places WHERE place_id = 'none' LIMIT 100"},
        files,
    )
    (out / subject.BATCHES).parent.mkdir(parents=True, exist_ok=True)
    (out / subject.BATCHES).write_text(
        json.dumps(
            [
                {
                    "world": subject.world_digest(out),
                    "session": "writing",
                    "scenarios": {
                        "book-sf": subject.scenario_digest(
                            out / "scenarios" / "book-sf"
                        )
                    },
                    "checks": [seed_only, scoped],
                    "probes": [],
                }
            ]
        )
    )
    monkeypatch.setattr(subject, "Stage", None)
    world = MarketWorld()

    checks = asyncio.run(subject.author_invariants(source, out, world))

    assert [check["name"] for check in checks] == ["sf place exists"]
    assert json.loads((out / subject.ARTIFACT).read_text())["checks"] == checks
    asyncio.run(subject.check_invariants(world, checks))
    asyncio.run(subject.check_invariants(world, checks, scenario_key="book-sf"))
