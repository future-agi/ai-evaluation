"""What the hosted world handle raises when scenario code asks it for something it will not do.

Every one of these is scenario code at fault, never the world's contents and never the agent
under test: a `KeyError` from a missing table reads as a finding about data, and a bare
`StoreError` from the postgres driver reads as an infrastructure fault. Neither is right for
"you called `change` without saying which column `key` names," so the handle has its own
vocabulary, and folds it under one base class for whoever has to route "scenario code misused
the handle" to one outcome without naming all six.
"""

from __future__ import annotations


class WorldError(RuntimeError):
    """Scenario code asked the world handle for something it will not do."""


class WorldReadOnly(WorldError):
    """`put`, `change`, `drop` or `call` reached the handle `ready()` or `check()` were given.

    Those two only ever observe a run. A check that could write would be able to change the very
    thing it is grading, and nothing downstream could tell the difference between a check that
    found a problem and one that quietly fixed it.
    """


class WorldReservedName(WorldError):
    """Scenario code named the harness's own conformance canary.

    That table exists to prove worlds are really isolated from each other, not to hold scenario
    data, and it never appears in `state()` either.
    """


class WorldQueryRejected(WorldError):
    """`query()` was handed something that is not one plain read.

    The database's own read-only transaction is what actually stops a write; this is the
    friendlier rejection in front of it, so a statement that was never going to be allowed fails
    on a message naming the reason rather than a lock error three layers down.
    """


class WorldStateTooLarge(WorldError):
    """`state()` reached a table whose row count, measured when the baseline was frozen, passed
    the cap.

    Measured once, at freeze time, by the provisioner — never recomputed here, so which tables
    raise is fixed before a scenario ever runs and nothing a call does during one can move it.
    """


class WorldUnavailable(WorldError):
    """The handle cannot do this, given how the world in front of it is built — not what it holds.

    A postgres world whose `public` schema has no tables, and `call()` — which raises
    unconditionally until the `http_tool` shim's wire format is pinned somewhere in the contracts
    — are both this: nothing went wrong, the capability was never there.
    """


class WorldUsageError(WorldError):
    """A `put`, `change` or `drop` cannot be carried out as asked.

    Inserting into something that is not a table, or changing or dropping a record without
    saying which column `key` names — hosted worlds cannot invent tables or guess a column, so
    both are reported here rather than attempted.
    """
