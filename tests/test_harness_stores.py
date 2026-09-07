"""The stores the harness stands up for an agent under test.

Two lanes. The registry tests are offline and always run: they are about what the harness
does when asked for an engine it does not have, which is the behaviour that keeps a run from
quietly grading an agent against the wrong database. The Postgres tests need a Docker daemon
and are skipped without one, following the same rule as the bench Docker lane.
"""

from __future__ import annotations

import pytest

from fi.alk.bench._docker import docker_available
from fi.alk.harness.bundle_author_v2 import _constraint_checked_seed_sql
from fi.alk.harness.world.stores import (
    PostgresStore,
    Snapshot,
    StoreError,
    resolve,
    supported,
)

# --- offline: what the harness will and will not stand up -------------------------------


def test_postgres_is_registered() -> None:
    assert "postgres" in supported()


def test_resolve_returns_a_store_for_a_known_engine() -> None:
    assert resolve("postgres").engine == "postgres"


def test_an_unknown_engine_is_refused_and_names_what_there_is() -> None:
    """No fallback. An engine we cannot run is a gap to report, not one to substitute.

    Handing a ClickHouse agent a Postgres would produce a green suite about SQL the agent
    never executes, which is worse than no suite at all.
    """
    with pytest.raises(StoreError) as raised:
        resolve("clickhouse")
    assert "clickhouse" in str(raised.value)
    assert "postgres" in str(raised.value)


def test_a_store_has_no_address_before_it_starts() -> None:
    with pytest.raises(StoreError):
        PostgresStore().dsn()


def test_stopping_a_store_that_never_started_is_safe() -> None:
    PostgresStore().stop()


def test_snapshot_counts_its_rows() -> None:
    snapshot = Snapshot(rows={"orders": [{"id": 1}, {"id": 2}], "menu": []})
    assert snapshot.counts() == {"orders": 2, "menu": 0}


# --- with docker: a real engine, really reset -------------------------------------------

pg = pytest.mark.skipif(not docker_available(), reason="docker daemon unavailable")

SCHEMA = """
CREATE TABLE customers (
    id    serial PRIMARY KEY,
    name  text NOT NULL,
    prefs jsonb
);
CREATE TABLE orders (
    id          serial PRIMARY KEY,
    customer_id int  NOT NULL REFERENCES customers(id),
    item        text NOT NULL,
    quantity    int  NOT NULL CHECK (quantity > 0)
);
CREATE TABLE boolean_rows (
    id             text PRIMARY KEY,
    phone_verified boolean NOT NULL
);
CREATE TABLE array_rows (
    id   text PRIMARY KEY,
    tags text[] NOT NULL
);
"""

SEED = """
INSERT INTO customers (name, prefs) VALUES ('ana', '{"spice": "hot"}'), ('bo', '{"spice": "mild"}');
INSERT INTO orders (customer_id, item, quantity) VALUES (1, 'turkey', 2);
"""


@pytest.fixture(scope="module")
def store():
    """One container for the module, because starting costs seconds and resetting does not.

    That is the same shape a suite has: the environment is stood up once and its contents move
    between scenarios.
    """
    running = PostgresStore(version="16")
    running.start()
    try:
        running.apply(SCHEMA)
        yield running
    finally:
        running.stop()


@pytest.fixture()
def seeded(store):
    """A store holding the seed, put back after whatever the test does to it."""
    store.restore(Snapshot())
    store.apply(SEED)
    return store


@pg
def test_generated_seed_loads_child_before_parent_without_disabling_constraints(seeded):
    seeded.apply(_constraint_checked_seed_sql([
        "INSERT INTO orders (customer_id, item, quantity) VALUES (99, 'test', 1)",
        "INSERT INTO customers (id, name) VALUES (99, 'parent')",
    ]))
    assert any(row["customer_id"] == 99 for row in seeded.state()["orders"])


@pg
def test_generated_seed_rejects_orphans_and_rolls_back_partial_load(seeded):
    with pytest.raises(Exception, match="seed_dependency_unresolved"):
        seeded.apply(_constraint_checked_seed_sql([
            "INSERT INTO customers (id, name) VALUES (98, 'must rollback')",
            "INSERT INTO orders (customer_id, item, quantity) VALUES (9999, 'orphan', 1)",
        ]))
    assert not any(row["id"] == 98 for row in seeded.state()["customers"])


@pg
def test_the_schema_is_the_one_that_was_applied(seeded) -> None:
    """The harness never invents tables. What is here is what the migration created."""
    assert sorted(seeded.state()) == [
        "array_rows",
        "boolean_rows",
        "customers",
        "orders",
    ]


@pg
def test_state_reads_rows_back_including_json(seeded) -> None:
    assert seeded.state()["customers"][0] == {
        "id": 1,
        "name": "ana",
        "prefs": {"spice": "hot"},
    }


@pg
def test_rows_come_back_in_a_stable_order(seeded) -> None:
    """Ordered by primary key, so a check reading the first row is not reading a coin toss."""
    assert [row["id"] for row in seeded.state()["customers"]] == [1, 2]


@pg
def test_restore_puts_the_rows_back_exactly(seeded) -> None:
    baseline = seeded.freeze()
    seeded.apply(
        "INSERT INTO orders (customer_id, item, quantity) VALUES (2, 'ham', 5)"
    )
    seeded.apply("DELETE FROM orders WHERE customer_id = 2")
    seeded.apply("DELETE FROM customers WHERE name = 'bo'")
    assert seeded.state() != baseline.rows

    seeded.restore(baseline)
    assert seeded.state() == baseline.rows


@pg
def test_restore_puts_the_counters_back_too(seeded) -> None:
    """Without this the next scenario's first insert gets an id continuing from the last one,
    and a check naming a specific id fails for a reason that is not the agent's doing."""
    baseline = seeded.freeze()
    seeded.apply(
        "INSERT INTO orders (customer_id, item, quantity) VALUES (1, 'ham', 1)"
    )
    seeded.restore(baseline)

    seeded.apply(
        "INSERT INTO orders (customer_id, item, quantity) VALUES (1, 'swiss', 1)"
    )
    fresh = [row for row in seeded.state()["orders"] if row["item"] == "swiss"]
    assert [row["id"] for row in fresh] == [2]


@pg
def test_restore_survives_foreign_keys_without_ordering_the_tables(seeded) -> None:
    """The snapshot came from a consistent database, so what goes back is consistent.

    Suspending the constraints beats sorting the tables into dependency order, which is a
    problem the snapshot means we do not have.
    """
    baseline = seeded.freeze()
    seeded.apply("DELETE FROM orders")
    seeded.apply("DELETE FROM customers")
    assert seeded.state()["customers"] == []

    seeded.restore(baseline)
    assert seeded.state() == baseline.rows


@pg
def test_boolean_equivalents_work_for_scenario_insert_update_and_restore(
    seeded,
) -> None:
    """Generated setup commonly crosses JSON/SQLite as 0/1 before reaching Postgres.

    All three runtime write paths share ``_adapt``. Prove each one against a real PostgreSQL
    boolean column so the hosted runner cannot regress to binding a smallint again.
    """
    inserted = seeded.add("boolean_rows", {"id": "inserted", "phone_verified": 1})
    assert inserted["phone_verified"] is True

    assert (
        seeded.amend(
            "boolean_rows",
            "inserted",
            {"phone_verified": 0},
            by="id",
        )
        == 1
    )
    assert seeded.state()["boolean_rows"] == [
        {"id": "inserted", "phone_verified": False}
    ]

    seeded.restore(
        Snapshot(rows={"boolean_rows": [{"id": "restored", "phone_verified": "true"}]})
    )
    assert seeded.state()["boolean_rows"] == [
        {"id": "restored", "phone_verified": True}
    ]


def test_postgres_boolean_adaptation_rejects_ambiguous_values() -> None:
    from fi.alk.harness.world.stores.postgres import _adapt

    with pytest.raises(StoreError, match="expected true/false"):
        _adapt(2, "boolean")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("true", True),
        ("FALSE", False),
        (" t ", True),
        (" 0 ", False),
    ],
)
def test_postgres_normalizes_unambiguous_boolean_equivalents(value, expected) -> None:
    from fi.alk.harness.world.stores.postgres import _adapt

    adapted = _adapt(value, "BOOLEAN")
    assert adapted is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("[]", []),
        ('["uberx", "comfort"]', ["uberx", "comfort"]),
        (("black", "xl"), ["black", "xl"]),
    ],
)
def test_postgres_normalizes_json_encoded_array_values(value, expected) -> None:
    from fi.alk.harness.world.stores.postgres import _adapt

    assert _adapt(value, "ARRAY") == expected


def test_postgres_array_adaptation_rejects_malformed_json_list() -> None:
    from fi.alk.harness.world.stores.postgres import _adapt

    with pytest.raises(StoreError, match="malformed JSON array text"):
        _adapt('["uberx",]', "ARRAY")


@pytest.mark.parametrize(
    "value", [{"enabled": True}, ["a", "b"], "plain text", 7, False]
)
def test_postgres_adapts_every_json_value_through_the_json_codec(value) -> None:
    from psycopg.types.json import Jsonb

    from fi.alk.harness.world.stores.postgres import _adapt

    adapted = _adapt(value, "jsonb")
    assert isinstance(adapted, Jsonb)
    assert adapted.obj == value


def test_postgres_decodes_catalogue_types_from_a_sql_ascii_database() -> None:
    class Result:
        def fetchall(self):
            return [(b"enabled", b"boolean"), (b"tags", b"ARRAY")]

    class Connection:
        def execute(self, _statement, _params):
            return Result()

    assert PostgresStore()._column_types(Connection(), "records") == {
        "enabled": "boolean",
        "tags": "ARRAY",
    }


@pg
def test_json_encoded_arrays_work_for_scenario_insert_update_and_restore(
    seeded,
) -> None:
    """Every generated setup write path accepts the JSON representation of a SQL array."""
    inserted = seeded.add("array_rows", {"id": "inserted", "tags": '["uberx"]'})
    assert inserted["tags"] == ["uberx"]

    assert seeded.amend("array_rows", "inserted", {"tags": "[]"}, by="id") == 1
    assert seeded.state()["array_rows"] == [{"id": "inserted", "tags": []}]

    seeded.restore(
        Snapshot(rows={"array_rows": [{"id": "restored", "tags": '["black", "xl"]'}]})
    )
    assert seeded.state()["array_rows"] == [{"id": "restored", "tags": ["black", "xl"]}]


@pg
def test_the_engine_produces_its_own_refusals(seeded) -> None:
    """A refusal here is the database's, not one we wrote into a handler.

    This is the difference the provisioning approach buys: an agent inserting a row that
    cannot exist is refused by the same constraint that would refuse it in production.
    """
    import psycopg

    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        seeded.apply(
            "INSERT INTO orders (customer_id, item, quantity) VALUES (999, 'x', 1)"
        )


@pg
def test_a_check_constraint_refuses_too(seeded) -> None:
    import psycopg

    with pytest.raises(psycopg.errors.CheckViolation):
        seeded.apply(
            "INSERT INTO orders (customer_id, item, quantity) VALUES (1, 'turkey', 0)"
        )


@pg
def test_the_dsn_is_what_the_agent_would_be_pointed_at(store) -> None:
    assert store.dsn().startswith("postgresql://")
    assert store.env("DATABASE_URL") == {"DATABASE_URL": store.dsn()}
    assert store.env("PG_DSN")["PG_DSN"] == store.dsn()
