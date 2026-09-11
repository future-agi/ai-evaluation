"""The gates a store has to clear before anything is measured against it.

The store's engine-specific half is written per agent, by a model, for an engine nobody vetted
in advance. So the interesting tests here are not that a good store passes -- they are that a
plausible-looking broken one fails, because that is the only thing standing between a wrong
reset and a suite of results that mean nothing.

Nearly all of this runs offline against a deliberately broken store. The gate is engine
independent by design, so it does not need a real engine to be exercised; one Docker-gated
test at the end confirms it says the same thing about a real Postgres.
"""

from __future__ import annotations

import copy

import pytest

from fi.alk.bench._docker import docker_available
from fi.alk.harness.world.stores import PostgresStore, Snapshot
from fi.alk.harness.world.stores.prove import prove_checks_bite, prove_store

ADD_ONE = "add one row"
NOTHING = "change nothing"


class FakeStore:
    """A store in a dict, so the gate can be tested without an engine.

    ``restores_counters`` is the whole point: a store that puts its rows back but lets its
    counter climb is the most common way a reset is subtly wrong, and it looks perfect if you
    only compare rows.
    """

    engine = "fake"

    def __init__(self, restores_counters: bool = True, seeded: bool = True) -> None:
        self.tables: dict[str, list[dict]] = {"orders": []}
        self.next_id = 1
        self.restores_counters = restores_counters
        if seeded:
            self.apply(ADD_ONE)

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def dsn(self) -> str:
        return "fake://"

    def apply(self, script: str) -> None:
        if script == NOTHING:
            return
        self.tables["orders"].append({"id": self.next_id, "item": "turkey"})
        self.next_id += 1

    def state(self) -> dict[str, list[dict]]:
        return copy.deepcopy(self.tables)

    def freeze(self) -> Snapshot:
        return Snapshot(rows=copy.deepcopy(self.tables), counters={"orders": self.next_id})

    def restore(self, snapshot: Snapshot) -> None:
        # Truncate then insert, which is what a real one does.
        self.tables = {name: [] for name in self.tables}
        for name, rows in snapshot.rows.items():
            self.tables[name] = copy.deepcopy(rows)
        if self.restores_counters:
            self.next_id = snapshot.counters.get("orders", 1)


def failures(report) -> list[str]:
    return [result.name for result in report.results if not result.passed]


# --- a sound store clears it ------------------------------------------------------------


def test_a_sound_store_passes_every_gate() -> None:
    report = prove_store(FakeStore(), ADD_ONE)
    assert failures(report) == []
    assert report.score == 1.0


# --- and the broken ones do not ---------------------------------------------------------


def test_a_store_that_lets_ids_drift_is_caught() -> None:
    """Rows go back, the counter does not. The gate never asks what a counter is called.

    It runs the same change twice from the same starting point and compares, so drift in
    anything at all shows up without the gate knowing the engine.
    """
    report = prove_store(FakeStore(restores_counters=False), ADD_ONE)
    assert "ids do not drift" in failures(report)
    assert "restore is exact" not in failures(report), "the rows themselves were fine"


def test_a_mutation_that_changes_nothing_is_refused() -> None:
    """Otherwise a broken restore looks perfect: nothing moved, so nothing failed to move back."""
    report = prove_store(FakeStore(), NOTHING)
    assert "the mutation moves it" in failures(report)


def test_a_store_with_no_schema_is_caught_before_anything_else() -> None:
    empty = FakeStore(seeded=False)
    empty.tables = {}
    report = prove_store(empty, ADD_ONE)
    assert "holds a schema" in failures(report)


def test_an_unseeded_store_is_reported_but_not_fatal() -> None:
    """An empty table is worth saying out loud; it is not necessarily wrong."""
    report = prove_store(FakeStore(seeded=False), ADD_ONE)
    assert "holds a seed" in failures(report)
    assert "restore is exact" not in failures(report)


def test_a_restore_that_raises_is_reported_as_ours() -> None:
    class Broken(FakeStore):
        def restore(self, snapshot: Snapshot) -> None:
            raise RuntimeError("no")

    report = prove_store(Broken(), ADD_ONE)
    assert "restore runs" in failures(report)


def test_a_restore_that_drops_rows_is_caught() -> None:
    class Forgetful(FakeStore):
        def restore(self, snapshot: Snapshot) -> None:
            self.tables = {name: [] for name in self.tables}

    report = prove_store(Forgetful(), ADD_ONE)
    assert "restore is exact" in failures(report)


# --- the checks themselves have to bite --------------------------------------------------


def reads_the_store(store) -> str | None:
    return None if store.state()["orders"] else "no orders"


def never_complains(store) -> str | None:
    return None


def test_a_check_that_passes_against_an_empty_store_is_flagged() -> None:
    """Break the world on purpose; whatever stays green was never load-bearing."""
    store = FakeStore()
    report = prove_checks_bite(
        store, {"reads_the_store": reads_the_store, "never_complains": never_complains}
    )
    assert failures(report) == ["never_complains"]


def test_proving_the_checks_leaves_the_store_as_it_found_it() -> None:
    store = FakeStore()
    before = store.state()
    prove_checks_bite(store, {"reads_the_store": reads_the_store})
    assert store.state() == before


def test_a_check_that_raises_counts_as_having_noticed() -> None:
    def explodes(store) -> str | None:
        raise KeyError("orders")

    report = prove_checks_bite(FakeStore(), {"explodes": explodes})
    assert failures(report) == []


# --- and it says the same thing about a real engine ---------------------------------------


@pytest.mark.skipif(not docker_available(), reason="docker daemon unavailable")
def test_the_gate_clears_a_real_postgres() -> None:
    store = PostgresStore(version="16")
    store.start()
    try:
        store.apply(
            "CREATE TABLE orders ("
            " id serial PRIMARY KEY, item text NOT NULL, quantity int NOT NULL);"
        )
        store.apply("INSERT INTO orders (item, quantity) VALUES ('turkey', 2)")

        report = prove_store(
            store, "INSERT INTO orders (item, quantity) VALUES ('ham', 1)"
        )
        assert failures(report) == [], report.summary()

        report = prove_checks_bite(
            store,
            {
                "reads_the_store": reads_the_store,
                "never_complains": never_complains,
            },
        )
        assert failures(report) == ["never_complains"]
        # and the seed is back afterwards, so the suite can carry on
        assert [row["item"] for row in store.state()["orders"]] == ["turkey"]
    finally:
        store.stop()
