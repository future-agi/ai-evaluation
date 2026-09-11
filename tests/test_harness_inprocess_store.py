"""A store that starts no process, for agents that keep their data in memory.

Two things are being checked. That the agent's own loader is what fills it, rather than
anything we reimplemented. And that ``Store`` describes stores rather than describing
containers -- the gate here is the same gate that judges Postgres, and it never learns that
nothing was started.
"""

from __future__ import annotations

import pytest

from fi.alk.harness.world.stores import InProcessStore, Snapshot, StoreError, resolve
from fi.alk.harness.world.stores.prove import prove_checks_bite, prove_store


def loader() -> dict:
    """Stands in for an agent's ``load_data``: groups keyed by id, as agents actually keep them."""
    return {
        "orders": {
            "#W001": {"status": "pending", "user_id": "ana", "total": 42.0},
            "#W002": {"status": "delivered", "user_id": "bo", "total": 17.5},
        },
        "users": {"ana": {"zip": "19122"}, "bo": {"zip": "60614"}},
        "log": [],
    }


@pytest.fixture()
def store() -> InProcessStore:
    one = InProcessStore(loader=loader)
    one.start()
    return one


# --- it is the agent's own data ------------------------------------------------------------


def test_it_is_registered_as_an_engine() -> None:
    # Manifests normalize the legacy spelling to the canonical engine identifier.
    assert resolve("inprocess").engine == "in_process"


def test_the_agents_loader_is_what_fills_it(store: InProcessStore) -> None:
    assert store.data["orders"]["#W001"]["status"] == "pending"


def test_a_loader_that_returns_the_wrong_shape_is_refused() -> None:
    with pytest.raises(StoreError, match="not a dict"):
        InProcessStore(loader=lambda: [1, 2, 3]).start()


def test_a_module_that_will_not_import_says_so_rather_than_inventing_data() -> None:
    with pytest.raises(StoreError, match="cannot import"):
        InProcessStore(module="nothing.like.this").start()


def test_a_missing_loader_function_is_named() -> None:
    with pytest.raises(StoreError, match="not a function"):
        InProcessStore(module="json", function="definitely_not_here").start()


def test_nothing_connects_to_it_and_that_is_reported_not_raised(store) -> None:
    assert store.dsn() == "inprocess://"


# --- what a check reads --------------------------------------------------------------------


def test_keyed_groups_become_rows_that_keep_their_id(store: InProcessStore) -> None:
    """An agent keyed by id would otherwise read as zero rows, with the id thrown away."""
    orders = store.state()["orders"]
    assert len(orders) == 2
    assert {row["_id"] for row in orders} == {"#W001", "#W002"}
    assert [row["status"] for row in orders if row["_id"] == "#W001"] == ["pending"]


def test_list_groups_stay_lists(store: InProcessStore) -> None:
    assert store.state()["log"] == []


# --- freeze and restore --------------------------------------------------------------------


def test_restore_puts_the_structure_back_after_a_real_change(store: InProcessStore) -> None:
    baseline = store.freeze()
    store.data["orders"]["#W001"]["status"] = "cancelled"
    del store.data["orders"]["#W002"]
    assert store.state() != baseline.rows

    store.restore(baseline)
    assert store.state() == baseline.rows
    # and the agent's own shape is back, not just our row view of it
    assert store.data["orders"]["#W001"]["status"] == "pending"
    assert "#W002" in store.data["orders"]


def test_restore_hands_back_the_shape_the_agent_expects(store: InProcessStore) -> None:
    """The tools index by id, so a restore that returned a list would break every one of them."""
    store.restore(store.freeze())
    assert isinstance(store.data["orders"], dict)
    assert isinstance(store.data["log"], list)


def test_the_snapshot_is_a_copy_not_a_view(store: InProcessStore) -> None:
    baseline = store.freeze()
    store.data["orders"]["#W001"]["status"] = "cancelled"
    assert baseline.rows["orders"][0]["status"] == "pending"


def test_apply_runs_a_snippet_against_the_data(store: InProcessStore) -> None:
    store.apply("data['orders']['#W003'] = {'status': 'pending', 'user_id': 'cy', 'total': 1.0}")
    assert len(store.state()["orders"]) == 3


def test_a_snippet_that_raises_is_reported_as_ours(store: InProcessStore) -> None:
    with pytest.raises(StoreError, match="KeyError"):
        store.apply("data['nope']['x'] = 1")


# --- and the gate does not care that nothing was started -------------------------------------


def test_the_same_gate_clears_it() -> None:
    one = InProcessStore(loader=loader)
    one.start()
    report = prove_store(
        one, "data['orders']['#W900'] = {'status': 'pending', 'user_id': 'ana', 'total': 9.0}"
    )
    assert [r.name for r in report.results if not r.passed] == [], report.summary()


def test_the_gate_still_catches_a_restore_that_drops_things() -> None:
    class Forgetful(InProcessStore):
        def restore(self, snapshot: Snapshot) -> None:
            self.data["orders"] = {}

    one = Forgetful(loader=loader)
    one.start()
    report = prove_store(one, "data['orders']['#W900'] = {'status': 'pending'}")
    assert "restore is exact" in [r.name for r in report.results if not r.passed]


def test_a_check_that_ignores_the_store_is_still_flagged() -> None:
    one = InProcessStore(loader=loader)
    one.start()
    report = prove_checks_bite(
        one,
        {
            "reads_orders": lambda s: None if s.state()["orders"] else "no orders",
            "never_complains": lambda s: None,
        },
    )
    assert [r.name for r in report.results if not r.passed] == ["never_complains"]
