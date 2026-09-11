"""The hosted world handle: `HostedWorld`, its read-only view, and the postgres deltas under it.

Two lanes, like the store tests this sits beside. Everything that is `HostedWorld`'s own
judgement — the token check, the reserved-name refusal, the row-count cap, the by-resolution,
the read-only view, `call`'s refusal — is proven against a fake store and needs neither Docker
nor a real database nor even psycopg installed. Only "is this really enforced by Postgres, not
just by us" needs the container, and those tests are skipped without one, following the same
rule as the bench Docker lane.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from fi.alk.bench._docker import docker_available
from fi.alk.harness.world.errors import (
    WorldError,
    WorldQueryRejected,
    WorldReadOnly,
    WorldReservedName,
    WorldStateTooLarge,
    WorldUnavailable,
    WorldUsageError,
)
from fi.alk.harness.world.handle import CONFORMANCE_TABLE, STATE_ROW_CAP, HostedWorld
from fi.alk.harness.world.stores import Snapshot
from fi.alk.harness.world.stores.postgres import PostgresStore

# --- errors.py: the vocabulary, offline ---------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    [
        WorldQueryRejected,
        WorldReadOnly,
        WorldReservedName,
        WorldStateTooLarge,
        WorldUnavailable,
        WorldUsageError,
    ],
)
def test_every_world_exception_is_a_world_error(kind) -> None:
    assert issubclass(kind, WorldError)
    assert isinstance(kind("because"), RuntimeError)


# --- a fake store: HostedWorld's own judgement, without a database --------------------------


class FakeStore:
    """Just enough of `PostgresStore` for `HostedWorld` to run on, without a database.

    `table`/`state` mirror the store's own per-table ordering directly off `primary_keys`, the
    way `PostgresStore` resolves it itself; `query` only fakes the three catalogue lookups
    `HostedWorld` actually issues (`pg_tables`, `pg_index`, `information_schema.columns`) plus
    one quoted-table-name passthrough for `world.query()` itself, since nothing here needs to
    parse real SQL to prove what is `HostedWorld`'s own judgement rather than the database's.
    """

    def __init__(
        self,
        tables: dict[str, list[dict[str, Any]]],
        primary_keys: dict[str, list[str]] | None = None,
        columns: dict[str, set[str]] | None = None,
    ) -> None:
        self.tables = {name: list(rows) for name, rows in tables.items()}
        self.primary_keys = primary_keys or {}
        # A table's columns otherwise come from whatever its rows happen to hold, which is
        # nothing for an empty table; an explicit map lets a test give a table its columns
        # without needing a seed row just to make a by= resolve, the way it would against a
        # real, empty Postgres table.
        self.columns = columns or {}

    def _ordered(self, name: str) -> list[dict[str, Any]]:
        rows = self.tables.get(name, [])
        key = self.primary_keys.get(name)
        if key:
            return sorted((dict(row) for row in rows), key=lambda row: row[key[0]])
        return [dict(row) for row in rows]

    def state(self, only: list[str] | None = None) -> dict[str, list[dict[str, Any]]]:
        names = self.tables if only is None else only
        return {name: self._ordered(name) for name in names}

    def table(self, name: str) -> list[dict[str, Any]]:
        return self._ordered(name)

    def query(self, statement: str, params: tuple = ()) -> list[dict[str, Any]]:
        if "pg_tables" in statement:
            return [{"tablename": name} for name in sorted(self.tables)]
        if "pg_index" in statement:
            table = params[0]
            return [{"attname": column} for column in self.primary_keys.get(table, [])]
        if "information_schema.columns" in statement:
            table = params[0]
            known = self.columns.get(table)
            if known is None:
                known = {column for row in self.tables.get(table, []) for column in row}
            return [{"column_name": column} for column in sorted(known)]
        for name, rows in self.tables.items():
            if f'"{name}"' in statement:
                return [dict(row) for row in rows]
        return []

    def add(self, collection: str, record: dict[str, Any]) -> dict[str, Any]:
        stored = dict(record)
        stored.setdefault("id", len(self.tables[collection]) + 1)
        self.tables[collection].append(stored)
        return stored

    def amend(self, collection: str, key: str, changes: dict[str, Any], *, by: str = "") -> int:
        changed = 0
        for row in self.tables.get(collection, []):
            if str(row.get(by)) == str(key):
                row.update(changes)
                changed += 1
        return changed

    def remove(self, collection: str, key: str = "", *, by: str = "") -> int:
        rows = self.tables.get(collection, [])
        if not key:
            count = len(rows)
            rows.clear()
            return count
        kept = [row for row in rows if str(row.get(by)) != str(key)]
        removed = len(rows) - len(kept)
        rows[:] = kept
        return removed


def _world(
    tables: dict[str, list[dict[str, Any]]],
    *,
    baseline: dict[str, int] | None = None,
    primary_keys: dict[str, list[str]] | None = None,
    columns: dict[str, set[str]] | None = None,
) -> HostedWorld:
    visible = {name: len(rows) for name, rows in tables.items() if name != CONFORMANCE_TABLE}
    store = FakeStore(tables, primary_keys, columns)
    return HostedWorld(
        store,
        world_index=3,
        rng=random.Random(7),
        baseline_row_counts=baseline if baseline is not None else visible,
    )


def test_world_index_and_rng_are_carried_through_unchanged() -> None:
    rng = random.Random(11)
    world = HostedWorld(
        FakeStore({"orders": []}), world_index=4, rng=rng, baseline_row_counts={"orders": 0}
    )
    assert world.world_index == 4
    assert world.rng is rng


# --- state() -----------------------------------------------------------------------------


def test_state_reports_every_visible_table_empty_ones_included() -> None:
    world = _world({"customers": [{"id": 1, "name": "ana"}], "orders": []})
    assert world.state() == {"customers": [{"id": 1, "name": "ana"}], "orders": []}


def test_state_excludes_the_conformance_table() -> None:
    world = _world(
        {"orders": [{"id": 1}], CONFORMANCE_TABLE: [{"id": 1, "marker": "alive"}]},
    )
    assert CONFORMANCE_TABLE not in world.state()


def test_state_selects_one_table() -> None:
    world = _world({"customers": [{"id": 1}], "orders": [{"id": 9}]})
    assert world.state("orders") == {"orders": [{"id": 9}]}


def test_state_selector_reflects_the_fakestores_own_ordering() -> None:
    """`table()`'s primary-key ordering is `PostgresStore._select_ordered`'s job, proven against
    a real database in the docker lane below; this only pins that `HostedWorld` passes through
    whatever order `FakeStore.table()` hands back, not that the ordering itself is correct."""
    world = _world(
        {"orders": [{"id": 2, "item": "b"}, {"id": 1, "item": "a"}]},
        primary_keys={"orders": ["id"]},
    )
    assert [row["id"] for row in world.state("orders")["orders"]] == [1, 2]


def test_state_selector_passes_through_the_fakes_row_order_when_it_has_no_primary_key() -> None:
    """Same caveat as above: this is `FakeStore`'s own behaviour when it has no key to sort by,
    not a guarantee `HostedWorld` makes about row order."""
    given = [{"item": "b"}, {"item": "a"}]
    world = _world({"orders": given})
    assert world.state("orders")["orders"] == given


def test_naming_the_conformance_table_is_reserved_not_a_usage_mistake() -> None:
    world = _world({"orders": [], CONFORMANCE_TABLE: [{"id": 1}]})
    with pytest.raises(WorldReservedName):
        world.state(CONFORMANCE_TABLE)


def test_naming_a_table_this_world_does_not_have_is_a_usage_error() -> None:
    world = _world({"orders": []})
    with pytest.raises(WorldUsageError):
        world.state("bookings")


def test_zero_visible_tables_is_unavailable_not_an_empty_snapshot() -> None:
    world = _world({CONFORMANCE_TABLE: [{"id": 1}]})
    with pytest.raises(WorldUnavailable):
        world.state()


def test_state_selector_on_an_empty_schema_is_unavailable_not_a_usage_error() -> None:
    """The empty-schema case is `WorldUnavailable` before the membership test even runs: a world
    with nothing in it cannot serve any table name, so blaming the scenario for asking about one
    in particular is backwards."""
    world = _world({CONFORMANCE_TABLE: [{"id": 1}]})
    with pytest.raises(WorldUnavailable):
        world.state("bookings")


def test_a_table_over_its_baseline_cap_raises_without_touching_its_rows() -> None:
    """The cap is decided from what was measured at freeze, never recounted here.

    The fake table itself holds nothing near 5,000 rows; only the baseline claims it does. If
    `state()` re-measured, this would pass instead of raising.
    """
    world = _world({"orders": [{"id": 1}]}, baseline={"orders": STATE_ROW_CAP + 1})
    with pytest.raises(WorldStateTooLarge):
        world.state("orders")


def test_a_table_at_or_under_the_cap_is_read_normally() -> None:
    world = _world({"orders": [{"id": 1}]}, baseline={"orders": STATE_ROW_CAP})
    assert world.state("orders") == {"orders": [{"id": 1}]}


def test_bare_state_excludes_an_over_cap_table_without_raising() -> None:
    """The exclusion is a property of the bare snapshot, not a failure — one seeded audit table
    must not make the primary read verb inert for the whole run. Naming it explicitly still
    raises: the exclusion never becomes a way to read the table around its own cap."""
    world = _world(
        {"orders": [{"id": 1}], "audit_log": [{"id": 1}]},
        baseline={"orders": 1, "audit_log": STATE_ROW_CAP + 1},
    )
    snapshot = world.state()
    assert snapshot == {"orders": [{"id": 1}]}
    assert "audit_log" not in snapshot
    with pytest.raises(WorldStateTooLarge):
        world.state("audit_log")


def test_bare_state_raises_when_every_table_is_over_the_cap() -> None:
    """Excluding every over-cap table would leave the snapshot `{}` while the schema is not
    empty — exactly the vacuous observation `state()` must never produce, since a check reading
    an empty snapshot as "nothing there" would pass on a world it never actually looked at — so
    the bare call raises instead of quietly handing that back."""
    world = _world(
        {"orders": [{"id": 1}], "audit_log": [{"id": 1}]},
        baseline={"orders": STATE_ROW_CAP + 1, "audit_log": STATE_ROW_CAP + 1},
    )
    with pytest.raises(WorldStateTooLarge):
        world.state()


def test_bare_state_names_both_causes_when_nothing_survives_the_exclusion() -> None:
    """The widened message covers the mixed case too: one baseline table over the cap plus one
    table the agent created since construction — excluding both would leave the snapshot `{}`,
    so the raised message must name both `orders` (over-cap) and `late_table` (never measured),
    not just whichever cause the wording happens to lead with."""
    world = _world({"orders": [{"id": 1}]}, baseline={"orders": STATE_ROW_CAP + 1})
    world._store.tables["late_table"] = [{"id": 1}]
    with pytest.raises(WorldStateTooLarge) as raised:
        world.state()
    assert "never measured" in str(raised.value)
    assert "late_table" in str(raised.value) and "orders" in str(raised.value)


def test_a_table_missing_from_the_baseline_dict_is_unavailable_not_under_cap() -> None:
    """A missing entry used to default to a row count of 0 and read as under the cap; the cap
    cannot be decided for a table nobody measured at freeze. The check runs once, at
    construction, rather than waiting for a scenario's first access to discover the gap."""
    with pytest.raises(WorldUnavailable):
        _world({"orders": [{"id": 1}]}, baseline={})


def test_a_table_appearing_after_construction_is_unavailable_on_the_selector_path() -> None:
    """`_require_baseline_coverage` only sees the tables visible when the handle was built; a
    table that shows up afterward is still absent from `_baseline_row_counts`. Naming it
    explicitly must reach the typed `WorldUnavailable` naming the table, not a bare `KeyError`
    off an unguarded index — a `WorldUnavailable("")` would pass this test for the wrong reason,
    so the message is asserted, not just the type."""
    world = _world({"orders": [{"id": 1}]})
    world._store.tables["late_table"] = [{"id": 1}]
    with pytest.raises(WorldUnavailable) as raised:
        world.state("late_table")
    assert "late_table" in str(raised.value)


def test_a_table_appearing_after_construction_is_excluded_from_the_bare_snapshot() -> None:
    """Nothing the agent under test does during a call may decide whether bare `state()` raises
    — a table it creates since construction is the only way an unmeasured table can exist, so the
    bare path excludes it the same way it excludes an over-cap table rather than going
    unavailable for the whole snapshot."""
    world = _world({"orders": [{"id": 1}]})
    world._store.tables["late_table"] = [{"id": 1}]
    snapshot = world.state()
    assert snapshot == {"orders": [{"id": 1}]}
    assert "late_table" not in snapshot


# --- put / change / drop -------------------------------------------------------------------


def test_put_returns_the_stored_record_key_included() -> None:
    world = _world({"customers": []})
    stored = world.put("customers", {"name": "ana"})
    assert stored == {"id": 1, "name": "ana"}


def test_put_into_a_reserved_name_is_refused() -> None:
    world = _world({CONFORMANCE_TABLE: []})
    with pytest.raises(WorldReservedName):
        world.put(CONFORMANCE_TABLE, {"marker": "x"})


def test_put_into_something_that_is_not_a_table_is_a_usage_error() -> None:
    world = _world({"customers": []})
    with pytest.raises(WorldUsageError):
        world.put("not_a_table", {"name": "ana"})


def test_put_with_a_mapping_key_value_is_a_usage_error() -> None:
    """A mapping-style key value must not silently become a hosted table column hint."""
    world = _world({"customers": []})
    with pytest.raises(WorldUsageError):
        world.put("customers", {"name": "ana"}, key="c1")


def test_put_accepts_redundant_single_primary_key_column_hint() -> None:
    """GeneratedWorld ignores this table hint, so hosted execution must behave identically."""
    world = _world(
        {"customers": []},
        primary_keys={"customers": ["customer_id"]},
        columns={"customers": {"customer_id", "name"}},
    )
    stored = world.put(
        "customers", {"customer_id": "c1", "name": "ana"}, key="customer_id"
    )
    assert stored["customer_id"] == "c1"
    assert stored["name"] == "ana"


def test_put_accepts_redundant_single_primary_key_value_hint() -> None:
    """Generated table scenarios commonly spell key= as the record key's value."""
    world = _world(
        {"customers": []},
        primary_keys={"customers": ["customer_id"]},
        columns={"customers": {"customer_id", "name"}},
    )
    stored = world.put(
        "customers", {"customer_id": "c1", "name": "ana"}, key="c1"
    )
    assert stored["customer_id"] == "c1"
    assert stored["name"] == "ana"


def test_put_rejects_key_value_that_does_not_match_the_record_primary_key() -> None:
    world = _world(
        {"customers": []},
        primary_keys={"customers": ["customer_id"]},
        columns={"customers": {"customer_id", "name"}},
    )
    with pytest.raises(WorldUsageError):
        world.put("customers", {"customer_id": "c1", "name": "ana"}, key="c2")


def test_put_rejects_non_primary_column_hint() -> None:
    world = _world(
        {"customers": []},
        primary_keys={"customers": ["customer_id"]},
        columns={"customers": {"customer_id", "name"}},
    )
    with pytest.raises(WorldUsageError):
        world.put("customers", {"customer_id": "c1", "name": "ana"}, key="name")


def test_change_without_by_is_a_usage_error() -> None:
    world = _world({"orders": [{"id": 1, "item": "a"}]})
    with pytest.raises(WorldUsageError):
        world.change("orders", "1", {"item": "b"})


def test_change_with_by_updates_and_reports_the_count() -> None:
    world = _world({"orders": [{"id": 1, "item": "a"}]})
    assert world.change("orders", "1", {"item": "b"}, by="id") == 1


def test_change_without_by_resolves_a_single_column_primary_key() -> None:
    world = _world({"orders": [{"id": 1, "item": "a"}]}, primary_keys={"orders": ["id"]})
    assert world.change("orders", "1", {"item": "b"}) == 1


def test_change_on_something_that_is_not_a_table_is_a_usage_error() -> None:
    world = _world({"orders": []})
    with pytest.raises(WorldUsageError):
        world.change("not_a_table", "1", {"item": "b"}, by="id")


def test_change_a_reserved_name_is_refused() -> None:
    world = _world({CONFORMANCE_TABLE: [{"id": 1}]})
    with pytest.raises(WorldReservedName):
        world.change(CONFORMANCE_TABLE, "1", {"marker": "x"}, by="id")


def test_change_with_a_by_that_is_not_a_column_is_a_usage_error() -> None:
    world = _world({"orders": [{"id": 1, "item": "a"}]})
    with pytest.raises(WorldUsageError):
        world.change("orders", "1", {"item": "b"}, by="nope")


def test_change_by_a_column_works_on_an_empty_table_when_the_fake_declares_its_columns() -> None:
    """Proves the columns map on `FakeStore` actually does something: without it an empty
    table's columns default to empty and this would raise `WorldUsageError` in the fake even
    though a real, empty Postgres table would resolve `by="id"` just fine."""
    world = _world({"orders": []}, columns={"orders": {"id", "item"}})
    assert world.change("orders", "1", {"item": "b"}, by="id") == 0


def test_drop_with_a_key_and_no_by_is_a_usage_error() -> None:
    world = _world({"orders": [{"id": 1}]})
    with pytest.raises(WorldUsageError):
        world.drop("orders", "1")


def test_drop_with_no_key_needs_no_by() -> None:
    world = _world({"orders": [{"id": 1}, {"id": 2}]})
    assert world.drop("orders") == 2


def test_drop_without_by_resolves_a_single_column_primary_key() -> None:
    world = _world({"orders": [{"id": 1}, {"id": 2}]}, primary_keys={"orders": ["id"]})
    assert world.drop("orders", "1") == 1


def test_drop_on_something_that_is_not_a_table_is_a_usage_error() -> None:
    world = _world({"orders": []})
    with pytest.raises(WorldUsageError):
        world.drop("not_a_table", "1", by="id")


def test_drop_a_reserved_name_is_refused() -> None:
    world = _world({CONFORMANCE_TABLE: [{"id": 1}]})
    with pytest.raises(WorldReservedName):
        world.drop(CONFORMANCE_TABLE)


def test_drop_with_a_by_that_is_not_a_column_is_a_usage_error() -> None:
    world = _world({"orders": [{"id": 1, "item": "a"}]})
    with pytest.raises(WorldUsageError):
        world.drop("orders", "1", by="nope")


# --- query(): the token check ----------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM orders",
        "select * from orders",
        "  -- a leading comment\nSELECT * FROM orders",
        "/* block */ SELECT * FROM orders",
        "WITH recent AS (SELECT * FROM orders) SELECT * FROM recent",
        "VALUES (1), (2)",
        "SELECT * FROM orders;",
        "SELECT * FROM orders WHERE note = 'ends with a semicolon;'",
        "SELECT * FROM orders WHERE note = 'ask for update'",
        "SELECT * FROM orders WHERE note = E'a semicolon inside an escape string: ;'",
        "SELECT * FROM orders WHERE note = E'it\\'s for update'",
    ],
)
def test_query_accepts_one_read_statement(sql) -> None:
    """Not rejected, whatever it finds to read - the token check is a shape question only."""
    world = _world({"orders": [{"id": 1}]})
    assert isinstance(world.query(sql), list)


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE orders SET item = 'x'",
        "DELETE FROM orders",
        "INSERT INTO orders (item) VALUES ('x')",
        "SELECT * FROM orders; DROP TABLE orders",
        "SELECT * FROM orders FOR UPDATE",
        "SELECT * FROM orders FOR SHARE",
        "   ",
    ],
)
def test_query_rejects_anything_that_is_not_one_plain_read(sql) -> None:
    world = _world({"orders": [{"id": 1}]})
    with pytest.raises(WorldQueryRejected):
        world.query(sql)


def test_query_accepts_a_semicolon_inside_a_dollar_quoted_literal() -> None:
    """A dollar-quoted span is a literal like any other quoted one; a `;` inside `$$...$$` must
    not count toward the one-statement rule any more than one inside `'...'` does."""
    world = _world({"orders": [{"id": 1}]})
    assert isinstance(world.query("SELECT $$contains a semicolon: ;$$ AS note"), list)


def test_query_rejects_a_second_statement_hidden_behind_a_dollar_quoted_apostrophe() -> None:
    """Before `_blank` knew dollar-quoting, the `'` inside `$$it's fine$$` opened a bogus quote
    span that swallowed everything after it, including the real statement separator, so a
    second statement rode through unrejected. Recognising the dollar-quoted span as one unit is
    what keeps the semicolon between the two real statements visible."""
    world = _world({"orders": [{"id": 1}]})
    with pytest.raises(WorldQueryRejected):
        world.query("SELECT $$it's fine$$ AS a; SELECT 2")


def test_query_rejects_a_second_statement_hidden_behind_an_escaped_quote() -> None:
    """Before `_blank` knew `E'...'`'s backslash escapes, the `\\'` inside one closed the
    literal early; the real closing quote right after it then read as opening a fresh span, and
    everything up to the next quote — semicolon, second statement and all — vanished into it
    unrejected."""
    world = _world({"orders": [{"id": 1}]})
    with pytest.raises(WorldQueryRejected):
        world.query("SELECT * FROM orders WHERE note = E'x\\'y' ; DELETE FROM orders")


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT * FROM {CONFORMANCE_TABLE}",
        f'SELECT * FROM "{CONFORMANCE_TABLE}"',
    ],
)
def test_query_rejects_naming_the_conformance_table(sql) -> None:
    world = _world({"orders": [{"id": 1}]})
    with pytest.raises(WorldQueryRejected):
        world.query(sql)


# --- read_only(): the view ready() and check() actually get ---------------------------------


def test_read_only_carries_world_index_and_rng() -> None:
    rng = random.Random(5)
    world = HostedWorld(
        FakeStore({"orders": []}), world_index=2, rng=rng, baseline_row_counts={"orders": 0}
    )
    view = world.read_only()
    assert view.world_index == 2
    assert view.rng is rng


def test_read_only_still_reads() -> None:
    world = _world({"orders": [{"id": 1}]})
    view = world.read_only()
    assert view.state() == {"orders": [{"id": 1}]}
    assert view.query('SELECT * FROM "orders"') == [{"id": 1}]


@pytest.mark.parametrize(
    "act",
    [
        lambda view: view.put("orders", {"item": "x"}),
        lambda view: view.change("orders", "1", {"item": "x"}, by="id"),
        lambda view: view.drop("orders", "1", by="id"),
        lambda view: view.call("some_tool"),
    ],
)
def test_read_only_refuses_every_write_verb(act) -> None:
    world = _world({"orders": [{"id": 1}]})
    view = world.read_only()
    with pytest.raises(WorldReadOnly):
        act(view)


def test_read_only_hides_unknown_capability_probes_behind_a_plain_attribute_error() -> None:
    """`prove.py`/`probe.py`/`run/voice.py` all reach for capability attributes on world objects
    through `hasattr`/`getattr(..., default)`; only a plain `AttributeError` makes that pattern
    work, so an unknown name here must read as one instead of as a write refusal."""
    world = _world({"orders": [{"id": 1}]})
    view = world.read_only()
    assert hasattr(view, "forward") is False
    assert getattr(view, "runtime_tools", set()) == set()


def test_read_only_dunder_lookups_stay_plain_attribute_errors() -> None:
    world = _world({"orders": [{"id": 1}]})
    view = world.read_only()
    with pytest.raises(AttributeError):
        view.__wrapped__


# --- call(): not implemented, the http_tool shim's wire format is unpinned ------------------


def test_call_is_unavailable() -> None:
    world = _world({"orders": []})
    with pytest.raises(WorldUnavailable):
        world.call("some_tool")


# --- with docker: the deltas that only Postgres itself can really prove --------------------

pg = pytest.mark.skipif(not docker_available(), reason="docker daemon unavailable")

SCHEMA = """
CREATE TABLE _alk_conformance (
    id     serial PRIMARY KEY,
    marker text NOT NULL
);
CREATE TABLE customers (
    id   serial PRIMARY KEY,
    name text NOT NULL
);
CREATE TABLE orders (
    id          serial PRIMARY KEY,
    customer_id int  NOT NULL REFERENCES customers(id),
    item        text NOT NULL
);
"""

SEED = """
INSERT INTO _alk_conformance (marker) VALUES ('alive');
INSERT INTO customers (name) VALUES ('ana'), ('bo');
INSERT INTO orders (customer_id, item) VALUES (1, 'turkey');
"""


@pytest.fixture(scope="module")
def store():
    running = PostgresStore(version="16")
    running.start()
    try:
        running.apply(SCHEMA)
        yield running
    finally:
        running.stop()


@pytest.fixture()
def seeded(store):
    store.restore(Snapshot())
    store.apply(SEED)
    return store


@pg
def test_postgres_query_reads_real_rows(seeded) -> None:
    rows = seeded.query('SELECT * FROM "customers" ORDER BY "id"')
    assert [row["name"] for row in rows] == ["ana", "bo"]


@pg
def test_postgres_query_refuses_a_write_at_the_database(seeded) -> None:
    """The friendliness check lives in `HostedWorld`; the store's own guard is this."""
    import psycopg

    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        seeded.query("DELETE FROM orders")
    # And the delete really did not happen.
    assert seeded.state()["orders"]


@pg
def test_postgres_query_accepts_a_percent_literal_with_no_bound_params(seeded) -> None:
    """An empty params tuple is still `not None` to psycopg's placeholder scanner, which would
    otherwise read the `%t` in `%turkey%` as an unmatched placeholder and raise before the
    statement ever reaches Postgres."""
    rows = seeded.query("SELECT * FROM orders WHERE item LIKE '%turkey%'")
    assert [row["item"] for row in rows] == ["turkey"]


@pg
def test_postgres_add_returns_the_row_with_its_generated_key(seeded) -> None:
    stored = seeded.add("customers", {"name": "cy"})
    assert stored["name"] == "cy"
    assert isinstance(stored["id"], int)
    assert stored in seeded.state()["customers"]


@pg
def test_hosted_world_query_refuses_a_data_modifying_cte(seeded) -> None:
    """A data-modifying CTE passes the token check (its leading keyword is WITH, a read
    keyword); the read-only transaction underneath is what actually stops the INSERT it
    hides, exactly as the friendliness check was always meant to be backed up."""
    import psycopg

    world = HostedWorld(
        seeded,
        world_index=0,
        rng=random.Random(9),
        baseline_row_counts={"customers": 2, "orders": 1},
    )
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        world.query(
            "WITH w AS (INSERT INTO orders (customer_id, item) VALUES (1, 'ham') "
            "RETURNING *) SELECT * FROM w"
        )
    assert len(world.state()["orders"]) == 1


@pg
def test_hosted_world_query_refuses_a_result_with_duplicate_column_labels(seeded) -> None:
    """`o.*, c.*` on a join gives both tables' `id` the same label; silently keeping only the
    last one under `dict(zip(...))` would let a check read the wrong table's key with no error
    anywhere - refusing beats a check that is quietly wrong."""
    world = HostedWorld(
        seeded,
        world_index=0,
        rng=random.Random(3),
        baseline_row_counts={"customers": 2, "orders": 1},
    )
    with pytest.raises(WorldQueryRejected):
        world.query("SELECT o.*, c.* FROM orders o JOIN customers c ON c.id = o.customer_id")


@pg
def test_hosted_world_end_to_end_against_a_real_database(seeded) -> None:
    world = HostedWorld(
        seeded,
        world_index=0,
        rng=random.Random(42),
        baseline_row_counts={"customers": 2, "orders": 1},
    )

    state = world.state()
    assert CONFORMANCE_TABLE not in state
    assert [row["name"] for row in state["customers"]] == ["ana", "bo"]

    with pytest.raises(WorldReservedName):
        world.state(CONFORMANCE_TABLE)

    stored = world.put("customers", {"name": "cy"})
    assert stored["name"] == "cy"

    changed = world.change("orders", "1", {"item": "ham"}, by="id")
    assert changed == 1
    assert world.state("orders")["orders"][0]["item"] == "ham"

    dropped = world.drop("orders", "1", by="id")
    assert dropped == 1
    assert world.state("orders") == {"orders": []}

    assert world.query("SELECT name FROM customers ORDER BY name") == [
        {"name": "ana"},
        {"name": "bo"},
        {"name": "cy"},
    ]


@pg
def test_hosted_world_cap_uses_the_baseline_not_a_live_count(seeded) -> None:
    world = HostedWorld(
        seeded,
        world_index=0,
        rng=random.Random(1),
        baseline_row_counts={"customers": STATE_ROW_CAP + 1, "orders": 1},
    )
    with pytest.raises(WorldStateTooLarge):
        world.state("customers")
    # A table the baseline did not flag reads normally in the same call.
    assert world.state("orders") == {"orders": [{"id": 1, "customer_id": 1, "item": "turkey"}]}


@pg
def test_hosted_world_on_an_empty_schema_is_unavailable() -> None:
    empty = PostgresStore(version="16")
    empty.start()
    try:
        world = HostedWorld(empty, world_index=0, rng=random.Random(1), baseline_row_counts={})
        with pytest.raises(WorldUnavailable):
            world.state()
    finally:
        empty.stop()
