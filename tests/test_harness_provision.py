"""The provisioning build stage, and the engine support it writes for itself.

The point of this path is that the harness stands up the engine the agent already uses rather
than replicating its tools, and that it can do so for an engine nobody shipped support for. So
the test that matters is the last one: ops written from scratch, for an engine registered at
run time, cleared by the same gate that has never heard of it -- and the same ops with one
thing wrong, caught.
"""

from __future__ import annotations

import pytest

from fi.alk.bench._docker import docker_available
from fi.alk.harness import build as build_stage
from fi.alk.harness.config import provisioning
from fi.alk.harness.contract import AgentContract
from fi.alk.harness.world.provision import MANIFEST, TOOL_NAMES
from fi.alk.harness.world.stores import StoreError, resolve, supported
from fi.alk.harness.world.stores.prove import prove_store
from fi.alk.harness.world.stores.written import register_written

# --- the flag ----------------------------------------------------------------------------


def test_provisioning_is_off_unless_asked_for(monkeypatch) -> None:
    """Two stages prove different things, so the shipped one stays the shipped one."""
    monkeypatch.delenv("ALK_HARNESS_PROVISION", raising=False)
    assert provisioning() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_the_flag_turns_it_on(monkeypatch, value: str) -> None:
    monkeypatch.setenv("ALK_HARNESS_PROVISION", value)
    assert provisioning() is True


def test_an_explicit_argument_beats_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("ALK_HARNESS_PROVISION", "1")
    assert provisioning(False) is False


def test_the_opening_tells_the_stage_not_to_touch_the_agent(monkeypatch) -> None:
    monkeypatch.setenv("ALK_HARNESS_PROVISION", "1")
    said = build_stage.opening(AgentContract(agent="orders"))
    assert "its OWN migrations" in said
    assert "do not touch the" in said


def test_the_old_opening_is_untouched_with_the_flag_off(monkeypatch) -> None:
    monkeypatch.delenv("ALK_HARNESS_PROVISION", raising=False)
    said = build_stage.opening(AgentContract(agent="orders"))
    assert "one handler per tool" in said


def test_success_is_a_different_artifact_on_each_path() -> None:
    assert MANIFEST == "environment.json"


# --- the surface itself -------------------------------------------------------------------


def test_there_is_no_tool_that_writes_into_the_agents_repo() -> None:
    """The rule is enforced by there being no verb for it, not by asking in a skill file."""
    forbidden = ("edit", "write_file", "patch", "apply_patch", "shell", "bash")
    assert not [one for one in TOOL_NAMES if any(bad in one for bad in forbidden)]


def test_the_surface_provisions_rather_than_replicating() -> None:
    assert "declare_engine" in TOOL_NAMES
    assert "run_migrations" in TOOL_NAMES
    # The two that made the old path build a replica.
    assert "define_handler" not in TOOL_NAMES
    assert "create_schema" not in TOOL_NAMES


# --- teaching the harness an engine at build time ------------------------------------------

WORKING_OPS = """
import psycopg

def connect(dsn):
    return psycopg.connect(dsn, autocommit=True)

def apply(db, script):
    db.execute(script)

def _tables(db):
    rows = db.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
    ).fetchall()
    return [r[0] for r in rows]

def state(db):
    out = {}
    for t in _tables(db):
        cur = db.execute('SELECT * FROM "%s" ORDER BY 1' % t)
        cols = [d[0] for d in cur.description]
        out[t] = [dict(zip(cols, r)) for r in cur.fetchall()]
    return out

def freeze(db):
    rows = db.execute(
        "SELECT sequencename, last_value FROM pg_sequences WHERE schemaname='public'"
    ).fetchall()
    return state(db), {r[0]: r[1] for r in rows if r[1] is not None}

def restore(db, rows, counters):
    tables = _tables(db)
    if tables:
        listed = ", ".join('"%s"' % t for t in tables)
        db.execute("TRUNCATE TABLE " + listed + " RESTART IDENTITY CASCADE")
    db.execute("SET session_replication_role = replica")
    for t, rs in rows.items():
        if not rs or t not in tables:
            continue
        cols = list(rs[0])
        q = ", ".join('"%s"' % c for c in cols)
        ph = ", ".join(["%s"] * len(cols))
        with db.cursor() as cur:
            cur.executemany(
                'INSERT INTO "%s" (%s) VALUES (%s)' % (t, q, ph),
                [tuple(r.get(c) for c in cols) for r in rs],
            )
    db.execute("SET session_replication_role = DEFAULT")
    COUNTERS

def add(db, group, record):
    cols = list(record)
    names = ", ".join('"%s"' % c for c in cols)
    values = ", ".join(["%s"] * len(cols))
    db.execute(
        'INSERT INTO "%s" (%s) VALUES (%s)' % (group, names, values),
        tuple(record[c] for c in cols),
    )
    return 1

def amend(db, group, key, changes, by):
    cols = list(changes)
    sets = ", ".join('"%s" = %%s' % c for c in cols)
    return db.execute(
        'UPDATE "%s" SET %s WHERE "%s" = %%s' % (group, sets, by),
        (*[changes[c] for c in cols], key),
    ).rowcount

def remove(db, group, key, by):
    if not key:
        return db.execute('DELETE FROM "%s"' % group).rowcount
    return db.execute(
        'DELETE FROM "%s" WHERE "%s" = %%s' % (group, by), (key,)
    ).rowcount
"""

RESTORES_COUNTERS = """
    for s, v in counters.items():
        db.execute("SELECT setval(%s, %s, true)", ('public."' + s + '"', v))
"""

BOOT = {
    "POSTGRES_USER": "{user}",
    "POSTGRES_PASSWORD": "{password}",
    "POSTGRES_DB": "{database}",
}
TEMPLATE = "postgresql://{user}:{password}@{host}:{port}/{database}"


def written(engine: str, counters: str):
    return register_written(
        engine=engine,
        image="postgres:16",
        container_port=5432,
        boot_env=BOOT,
        dsn_template=TEMPLATE,
        code=WORKING_OPS.replace("COUNTERS", counters),
    )


def test_ops_that_do_not_parse_are_refused() -> None:
    with pytest.raises(StoreError, match="do not parse"):
        register_written(
            engine="broken", image="x:1", container_port=1, code="def connect(:"
        )


def test_ops_missing_a_function_are_told_which_one() -> None:
    with pytest.raises(StoreError, match="restore"):
        register_written(
            engine="partial",
            image="x:1",
            container_port=1,
            code="def connect(d): pass\ndef apply(d,s): pass\n"
            "def state(d): return {}\ndef freeze(d): return {}, {}\n",
        )


def test_ops_importing_something_missing_say_so() -> None:
    with pytest.raises(StoreError, match="not installed"):
        register_written(
            engine="absent",
            image="x:1",
            container_port=1,
            code="import nope_not_real\n",
        )


def test_an_engine_needs_an_image_and_a_port() -> None:
    with pytest.raises(StoreError, match="image"):
        register_written(engine="e", image="", container_port=1, code="")
    with pytest.raises(StoreError, match="port"):
        register_written(engine="e", image="x:1", container_port=0, code="")


def test_registering_makes_the_engine_resolvable() -> None:
    written("pg_registered", RESTORES_COUNTERS)
    assert "pg_registered" in supported()
    assert resolve("pg_registered").engine == "pg_registered"


# --- and the gate judges it, having never heard of it ---------------------------------------

SCHEMA = "CREATE TABLE orders (id serial PRIMARY KEY, item text NOT NULL, quantity int NOT NULL)"
SEED = "INSERT INTO orders (item, quantity) VALUES ('turkey', 2)"
MUTATION = "INSERT INTO orders (item, quantity) VALUES ('ham', 1)"

pg = pytest.mark.skipif(not docker_available(), reason="docker daemon unavailable")


def stood_up(engine: str, counters: str):
    written(engine, counters)
    store = resolve(engine)
    store.start()
    store.apply(SCHEMA)
    store.apply(SEED)
    return store


@pg
def test_written_ops_stand_up_a_real_engine_and_clear_the_gate() -> None:
    """The whole thesis in one test: support written at build time, proved generically."""
    store = stood_up("pg_written_good", RESTORES_COUNTERS)
    try:
        assert store.state()["orders"][0]["item"] == "turkey"
        report = prove_store(store, MUTATION)
        assert [r.name for r in report.results if not r.passed] == [], report.summary()
    finally:
        store.stop()


@pg
def test_written_ops_that_forget_the_counter_are_caught() -> None:
    """Rows go back perfectly. Only the counter is wrong, and the gate never asks what a
    counter is called on this engine -- it runs the same change twice and compares."""
    store = stood_up("pg_written_bad", "")
    try:
        report = prove_store(store, MUTATION)
        failed = [r.name for r in report.results if not r.passed]
        assert "ids do not drift" in failed
        assert "restore is exact" not in failed
    finally:
        store.stop()
